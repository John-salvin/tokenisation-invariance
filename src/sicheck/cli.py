"""SI-Check command line.

Tokenisation Invariance (TI): a metric should give the same score to the same
string however it is segmented into tokens. SI-Check re-segments each
translation into byte-identical, non-canonical segmentations across a ladder
of fragmentation ratios, scores them, and tests whether agreement with human
judgement falls as fragmentation rises.

Script Invariance (SI): a metric should agree with humans equally well when
the same translation is written in another script. SI-Check compares
agreement on two renderings of the same segments with a test for dependent
correlations, and reports whether quantile normalisation (a monotone
post-hoc rescaling) could repair the gap.

Input: CSV, TSV, JSONL or parquet with columns
    mt      the translation (required)
    src     the English source (required by most metrics)
    ref     the reference (required by reference-based metrics)
    human   a segment-level human score, higher is better (required)
    seg_id  optional; one row per (segment, system) is fine
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tokinv import corrtests, sampler, stats, tokenisers
from tokinv.paths import LADDER8

RATIO_BUCKETS = sampler.RATIO_BUCKETS


def read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(path)
    if s == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep="\t" if s in (".tsv", ".txt") else ",")


# ------------------------------------------------------------------ sample -- #
def sample(df: pd.DataFrame, tokeniser: str, seeds: int, max_rows: int | None,
           seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Canonical plus `seeds` draws per ladder bucket for every row. Rows whose
    text cannot round-trip even canonically are excluded and counted, never
    forced through (zero-width joiners, for example, are dropped by XLM-R)."""
    for c in ("mt", "human"):
        if c not in df.columns:
            raise SystemExit(f"input needs a {c!r} column; has {list(df.columns)}")
    df = df.reset_index(drop=True)
    if "seg_id" not in df.columns:
        df["seg_id"] = df.index.astype(str)
    # `row` must index the input file as read, so that `score` finds each
    # sample's source and reference even after subsampling
    df["_row"] = df.index
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    tok = tokenisers.get(tokeniser)
    rows, excluded, unreachable = [], 0, 0
    for i, r in df.iterrows():
        text = sampler.normalize_text_for_reseg(str(r.mt))
        try:
            sampler.build_mdds(text, tokeniser)
        except sampler.NoncanonicalError:
            excluded += 1
            continue
        can = tok(text, add_special_tokens=False)["input_ids"]
        base = {"row": int(r._row), "unit": r.seg_id, "human": float(r.human)}
        rows.append({**base, "bucket": "canonical", "seed": -1, "ratio": 1.0, "mt_ids": can})
        for lo, hi in RATIO_BUCKETS:
            for s in range(seeds):
                out = sampler.sample_bucket(text, tokeniser, lo, hi, random.Random(s))
                if not out["reachable"]:
                    unreachable += 1
                    continue
                sampler.assert_round_trip(text, out["ids"], tokeniser)
                rows.append({**base, "bucket": f"[{lo:.2f},{hi:.2f})", "seed": s,
                             "ratio": out["ratio"], "mt_ids": out["ids"]})
    info = {"rows_used": len(df), "rows_excluded_roundtrip": excluded,
            "draws_unreachable": unreachable, "samples": len(rows), "tokeniser": tokeniser}
    return pd.DataFrame(rows), info


# ------------------------------------------------------------------- score -- #
def score(samples: pd.DataFrame, source: pd.DataFrame, metric: str, batch_size: int,
          tokeniser: str) -> pd.DataFrame:
    from tokinv import comet_ids
    model = comet_ids.load(metric)
    tok = tokenisers.get(tokeniser)
    ids = lambda s: tok(str(s), add_special_tokens=False)["input_ids"]
    src = [ids(source.loc[r, "src"]) for r in samples.row] if "src" in source else None
    ref = ([ids(source.loc[r, "ref"]) for r in samples.row]
           if "ref" in source and not comet_ids.is_referenceless(model) else None)
    out = []
    for a in range(0, len(samples), batch_size):
        b = slice(a, a + batch_size)
        out += comet_ids.score_batch(model, src[b] if src else None,
                                     list(samples.mt_ids.iloc[b]), ref[b] if ref else None)
    return samples.assign(score=out)


# ------------------------------------------------------------------ report -- #
def ti_report(sc: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    curve = stats.dose_curve(sc)
    can = sc[sc.bucket == "canonical"]
    rho = stats.spearman(curve.mean_ratio, curve.rho) if len(curve) >= 3 else float("nan")
    lo, hi = stats.segment_bootstrap(sc, stats.monotonicity, "unit",
                                     n_resamples=n_resamples, seed=seed)
    p = stats.exact_p(rho, len(curve)) if len(curve) <= 9 and len(curve) >= 3 else float("nan")
    if hi < 0:
        verdict = "VIOLATED: agreement falls with fragmentation"
    elif lo > 0:
        verdict = "VIOLATED: agreement rises with fragmentation"
    else:
        verdict = "NOT DETECTED at this sample size (interval includes 0)"
    return {"canonical_agreement": stats.spearman(can.score, can.human),
            "buckets": curve.round(4).to_dict(orient="records"),
            "monotonicity_rho": rho, "exact_p": p, "ci95": [lo, hi],
            "n_segments": int(sc.unit.nunique()), "verdict": verdict}


def si_report(native: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Both frames: seg_id, score, human; same segments, same human scores."""
    m = native.merge(other, on="seg_id", suffixes=("_a", "_b")).dropna(
        subset=["score_a", "score_b", "human_a"])
    r_a, r_b = stats.spearman(m.score_a, m.human_a), stats.spearman(m.score_b, m.human_a)
    r_ab = stats.spearman(m.score_a, m.score_b)
    t = corrtests.meng_rosenthal_rubin(r_a, r_b, r_ab, len(m))
    # QN: map both renderings onto the first rendering's distribution
    ref = np.sort(m.score_a.to_numpy())
    qn = lambda x: np.quantile(ref, pd.Series(x).rank().to_numpy() / (len(x) + 1))
    pooled_before = stats.spearman(np.r_[m.score_a, m.score_b], np.r_[m.human_a, m.human_a])
    pooled_after = stats.spearman(np.r_[qn(m.score_a), qn(m.score_b)], np.r_[m.human_a, m.human_a])
    return {"n": len(m), "agreement_a": r_a, "agreement_b": r_b, "drop": r_a - r_b,
            "mrr_z": t.statistic, "mrr_p": t.p_value,
            "verdict": ("VIOLATED: agreement differs between scripts" if t.p_value < 0.05
                        else "NOT DETECTED at this sample size"),
            "qn_pooled_before": pooled_before, "qn_pooled_after": pooled_after,
            "qn_note": ("QN can only remove the offset between renderings; it cannot change "
                        "agreement within a rendering, so a within-script drop is not repairable "
                        "post hoc.")}


def print_ti(r: dict) -> None:
    print(f"Tokenisation Invariance  ({r['n_segments']} segments)")
    print(f"  canonical agreement      {r['canonical_agreement']:+.3f}")
    for b in r["buckets"]:
        print(f"  {b['bucket']:13s} ratio {b['mean_ratio']:.2f}  agreement {b['rho']:+.3f}  (n={b['n']})")
    print(f"  monotonicity rho         {r['monotonicity_rho']:+.3f}   exact p {r['exact_p']:.2g}")
    print(f"  95% segment bootstrap    [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    print(f"  => {r['verdict']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="si-check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tokeniser", default="xlmr", choices=sorted(tokenisers.PINNED))
    common.add_argument("--seeds", type=int, default=5, help="draws per bucket")
    common.add_argument("--max-rows", type=int, default=None)
    common.add_argument("--resamples", type=int, default=stats.BOOT_RESAMPLES)
    common.add_argument("--seed", type=int, default=stats.BOOT_SEED)

    p = sub.add_parser("sample", parents=[common], help="draw re-segmentations (CPU)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score", parents=[common], help="score samples with a COMET-family metric")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--metric", default="comet22")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report", parents=[common], help="TI verdict from scored samples")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--json", type=Path)
    p = sub.add_parser("script", parents=[common], help="SI verdict from two scored renderings")
    p.add_argument("--a", type=Path, required=True, help="seg_id, score, human (e.g. native)")
    p.add_argument("--b", type=Path, required=True, help="seg_id, score, human (e.g. romanised)")
    p.add_argument("--json", type=Path)
    p = sub.add_parser("run", parents=[common], help="sample + score + report in one go")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--metric", default="comet22")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    a = ap.parse_args(argv)

    if a.cmd == "sample":
        s, info = sample(read_table(a.input), a.tokeniser, a.seeds, a.max_rows)
        s.to_parquet(a.out, index=False)
        print(json.dumps(info, indent=2))
    elif a.cmd == "score":
        score(pd.read_parquet(a.samples), read_table(a.input), a.metric, a.batch_size,
              a.tokeniser).drop(columns="mt_ids").to_parquet(a.out, index=False)
    elif a.cmd == "report":
        r = ti_report(read_table(a.scores), a.resamples, a.seed)
        print_ti(r)
        if a.json:
            a.json.write_text(json.dumps(r, indent=2, default=float))
    elif a.cmd == "script":
        r = si_report(read_table(a.a), read_table(a.b))
        for k, v in r.items():
            print(f"  {k:18s} {v:+.4f}" if isinstance(v, float) else f"  {k:18s} {v}")
        if a.json:
            a.json.write_text(json.dumps(r, indent=2, default=float))
    elif a.cmd == "run":
        a.out.mkdir(parents=True, exist_ok=True)
        src = read_table(a.input).reset_index(drop=True)
        s, info = sample(src, a.tokeniser, a.seeds, a.max_rows)
        print(json.dumps(info, indent=2))
        sc = score(s, src, a.metric, a.batch_size, a.tokeniser)
        sc.drop(columns="mt_ids").to_parquet(a.out / "scores.parquet", index=False)
        r = ti_report(sc, a.resamples, a.seed)
        print_ti(r)
        (a.out / "report.json").write_text(json.dumps({**r, "sampling": info}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
