"""Byte-identical re-segmentation and COMET scoring on WMT24/25 language pairs.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
LIVE = SB_ROOT / "repo" / "panel-pipeline"
sys.path.insert(0, str(LIVE / "src"))
sys.path.insert(0, str(ROB / "scripts" / "beyond"))
import noncanonical as NC
import tokenisers as T
import metrics_panel as MP

DATA_ROOT = Path(os.environ.get(
    "MTME_ROOT", SB_ROOT / "staging" / "wmt25" / "mt-metrics-eval-v2"))
OUT = ROB / "results" / "tables"
TOKENISER = "xlmr"
SEEDS = [0, 1, 2, 3, 4]
MAX_SEQ_IDS = 510
MAX_CANON_K = 160
SEED = 20260929
SCORE_PREFERENCE = ["mqm", "esa-merged", "esa-human1", "esa", "da-sqm", "wmt-z", "wmt"]


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return float("nan")
    ra = np.asarray(pd.Series(a).rank(), dtype=float)
    rb = np.asarray(pd.Series(b).rank(), dtype=float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def exact_perm_p(x, y) -> float:
    if len(y) > 9:
        return float("nan")
    obs = abs(spearman_small(x, y))
    hit = tot = 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        if abs(spearman_small(x, [y[i] for i in perm])) >= obs - 1e-12:
            hit += 1
    return hit / tot


def spearman_small(a, b) -> float:
    ra = np.asarray(pd.Series(list(a)).rank(), dtype=float)
    rb = np.asarray(pd.Series(list(b)).rank(), dtype=float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def read_lines(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in open(p, encoding="utf-8")]


def find_human_scores(d: Path, lp: str) -> tuple[Path, str]:
    hs = d / "human-scores"
    found = {}
    for f in sorted(hs.glob(f"{lp}.*.seg.score")):
        kind = f.name[len(lp) + 1:-len(".seg.score")]
        found[kind] = f
    if not found:
        raise SystemExit(f"no segment-level human scores for {lp} in {hs}")
    for pref in SCORE_PREFERENCE:
        if pref in found:
            return found[pref], pref
    kind, f = sorted(found.items())[0]
    return f, kind


def buckets_for(max_ratio: float, n_buckets: int | None) -> list[tuple[float, float]]:
    if n_buckets is None:
        return [(lo, hi) for lo, hi in NC.RATIO_BUCKETS if hi <= max_ratio + 1e-9]
    edges = np.linspace(1.0, max_ratio, n_buckets + 1)
    return [(round(float(edges[i]), 4), round(float(edges[i + 1]), 4))
            for i in range(n_buckets)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="wmt25")
    ap.add_argument("--pair", required=True, help="e.g. en-ru_RU")
    ap.add_argument("--n-seg", type=int, default=int(os.environ.get("N_SEG", "300")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", "48")))
    ap.add_argument("--max-ratio", type=float, default=3.0,
                    help="top of the bucket ladder; use ~1.75 for CJK")
    ap.add_argument("--n-buckets", type=int, default=None,
                    help="lay this many even buckets over [1, max-ratio] instead "
                         "of truncating the standard eight; needed for CJK, where "
                         "truncation would leave too few buckets to test")
    ap.add_argument("--adapter", default=None,
                    help="path to a saved LoRA checkpoint. Applies it to COMET-22's "
                         "encoder before scoring, so an adapter trained on Indic "
                         "languages can be evaluated zero-shot on a language it "
                         "never saw, in a script it never saw.")
    ap.add_argument("--adapter-lambda", type=float, default=1.0,
                    help="interpolation weight on the adapter delta, W = W_0 + "
                         "lambda*delta. The LoRA update is additive and gated by "
                         "LoRALinear.scaling, so this traces the base-to-adapted "
                         "frontier from one checkpoint. The frontier analysis puts "
                         "the Pareto point near 0.5, not 1.0, so full strength "
                         "overstates the canonical cost of the repair.")
    ap.add_argument("--metric", default="comet22",
                    help="comet22 (reference-based) or cometkiwi22 / cometkiwi23 "
                         "(reference-free QE). CometKiwi-22 is InfoXLM-large, which "
                         "inherits XLM-R's vocabulary, so the same sampler applies.")
    ap.add_argument("--boot", type=int, default=300,
                    help="segment-level bootstrap draws for the CI on the trend. "
                         "The permutation p over bucket means treats the buckets as "
                         "independent, which they are not: they share segments.")
    ap.add_argument("--pre-normalise", action="store_true",
                    help="replace each target by decode(tokenise(target)) before "
                         "the experiment. XLM-R's SentencePiece normaliser rewrites "
                         "fullwidth CJK punctuation, so decode(tokenise(x)) != x for "
                         "essentially every Chinese segment and the byte-identity "
                         "guard excludes them all. This makes the text testable by "
                         "moving to the string the tokeniser can actually represent; "
                         "the count of changed segments is reported and must be "
                         "stated wherever the result is used.")
    ap.add_argument("--keep-ref", action="store_true",
                    help="keep refA as a scored system (degenerate for COMET-22)")
    ap.add_argument("--tag", default=None, help="output tag, defaults to the pair")
    a = ap.parse_args()

    d = DATA_ROOT / a.testset
    lp, tag = a.pair, (a.tag or a.pair)
    rng_buckets = buckets_for(a.max_ratio, a.n_buckets)
    print(f"== {a.testset} {lp}: {len(rng_buckets)} buckets up to ratio {a.max_ratio}",
          flush=True)

    src = read_lines(d / "sources" / f"{lp}.txt")
    ref = read_lines(d / "references" / f"{lp}.refA.txt")
    assert len(ref) == len(src), "source/reference length mismatch"
    score_file, score_kind = find_human_scores(d, lp)
    print(f"   human scores: {score_file.name} (kind={score_kind})", flush=True)

    by_sys: dict[str, list[str]] = {}
    for line in open(score_file, encoding="utf-8"):
        s, v = line.rstrip("\n").split("\t")
        by_sys.setdefault(s, []).append(v)
    for s, v in by_sys.items():
        assert len(v) == len(src), f"{s}: {len(v)} scores, expected {len(src)}"
    scored = sorted(s for s, v in by_sys.items() if any(x != "None" for x in v))
    if not a.keep_ref:
        scored = [s for s in scored if not s.lower().startswith("ref")]
    print(f"   systems with scores: {len(scored)} of {len(by_sys)}", flush=True)

    rows = []
    for s in scored:
        out = read_lines(d / "system-outputs" / lp / f"{s}.txt")
        assert len(out) == len(src), f"{s}: output length mismatch"
        for i, (txt, sc) in enumerate(zip(out, by_sys[s])):
            if sc == "None":
                continue
            rows.append({"system": s, "seg_idx": i, "human": float(sc),
                         "source": src[i], "refA": ref[i], "target": txt})
    df = pd.DataFrame(rows)
    print(f"   scored (system, segment) pairs: {len(df)}", flush=True)

    n_prenorm = 0
    if a.pre_normalise:
        tk = T.load(TOKENISER)
        def _norm(x: str) -> str:
            return tk.decode(T.tokenise(x, TOKENISER, add_special_tokens=False))
        newt = [_norm(str(x)) for x in df["target"]]
        n_prenorm = sum(1 for a_, b_ in zip(df["target"], newt) if a_ != b_)
        df = df.assign(target=newt)
        print(f"   pre-normalised {n_prenorm}/{len(df)} targets to the tokeniser's "
              f"own representation", flush=True)

    model = MP.load_comet(a.metric, device="cuda")
    if a.adapter:
        import torch
        sys.path.insert(0, str(ROB / "src"))
        sys.path.insert(0, str(ROB / "scripts"))
        import lora_torch as L
        from lora_lolo_pipeline import is_attention_linear
        replaced = L.inject_lora(model.encoder,
                                 target_names=("query", "key", "value", "dense"),
                                 r=8, alpha=16, path_filter=is_attention_linear)
        state = torch.load(a.adapter, map_location="cpu", weights_only=False)
        L.load_lora_state_dict(model, state["adapter"], device="cuda")
        if a.adapter_lambda != 1.0:
            for m in model.modules():
                if isinstance(m, L.LoRALinear):
                    if not hasattr(m, "_base_scaling"):
                        m._base_scaling = m.scaling
                    m.scaling = m._base_scaling * a.adapter_lambda
        model.to("cuda").eval()
        print(f"   adapter applied: {a.adapter} "
              f"({len(replaced)} layers, step={state.get('step')}, "
              f"lambda={a.adapter_lambda})", flush=True)
    is_qe = a.metric.startswith("cometkiwi")
    print(f"   {a.metric} loaded (reference-{'free' if is_qe else 'based'})", flush=True)

    base = df.sample(n=min(2000, len(df)), random_state=SEED)

    n_before = len(df)
    ck, sk, rk = [], [], []
    for _, r in df.iterrows():
        ck.append(len(T.tokenise(NC.normalize_text_for_reseg(str(r["target"])), TOKENISER,
                                 add_special_tokens=False)))
        sk.append(len(T.tokenise(str(r["source"]), TOKENISER, add_special_tokens=False)))
        rk.append(len(T.tokenise(str(r["refA"]), TOKENISER, add_special_tokens=False)))
    df = df.assign(_ck=ck, _sk=sk, _rk=rk)
    df = df[(df._ck <= MAX_CANON_K) & (df._sk <= MAX_SEQ_IDS) & (df._rk <= MAX_SEQ_IDS)]
    print(f"   length filter: kept {len(df)}/{n_before}; excluded {n_before - len(df)}",
          flush=True)

    pool = df.sample(n=min(a.n_seg, len(df)), random_state=SEED).reset_index(drop=True)
    samples, n_attempt, n_excl, n_unreach, n_toolong = [], 0, 0, 0, 0
    for i, r in pool.iterrows():
        text = NC.normalize_text_for_reseg(str(r["target"]))
        if not text.strip():
            continue
        try:
            can = T.tokenise(text, TOKENISER, add_special_tokens=False)
            NC.assert_round_trip(text, can, TOKENISER)
            samples.append((i, "canonical", -1, can, 1.0))
        except Exception:
            n_excl += 1
            continue
        for lo, hi in rng_buckets:
            lab = f"[{lo:.2f},{hi:.2f})"
            for sd in SEEDS:
                n_attempt += 1
                try:
                    s = NC.sample_bucket(text, TOKENISER, lo, hi, random.Random(sd))
                    if not s["reachable"]:
                        n_unreach += 1
                        continue
                    NC.assert_round_trip(text, s["ids"], TOKENISER)
                    if len(s["ids"]) > MAX_SEQ_IDS:
                        n_toolong += 1
                        continue
                    samples.append((i, lab, sd, s["ids"], s["ratio"]))
                except NC.NoncanonicalError:
                    n_excl += 1
    print(f"   samples={len(samples)} attempted={n_attempt} unreachable={n_unreach} "
          f"excluded={n_excl} over_position_limit={n_toolong}", flush=True)

    scores = []
    for st in range(0, len(samples), a.batch):
        chunk = samples[st:st + a.batch]
        src_b, mt_b, ref_b = [], [], []
        for (i, _, _, ids, _) in chunk:
            r = pool.loc[i]
            src_b.append(T.tokenise(str(r["source"]), TOKENISER, add_special_tokens=False))
            ref_b.append(T.tokenise(str(r["refA"]), TOKENISER, add_special_tokens=False))
            mt_b.append(ids if ids else T.tokenise(" ", TOKENISER, add_special_tokens=False))
        scores.extend(MP.score_from_ids_batch(model, src_b, mt_b, ref_b))
        if st % (a.batch * 40) == 0:
            print(f"   scored {st + len(chunk)}/{len(samples)}", flush=True)

    out_rows = pd.DataFrame({
        "pair": tag,
        "row_idx": [s[0] for s in samples],
        "bucket": [s[1] for s in samples],
        "seed": [s[2] for s in samples],
        "ratio": [s[4] for s in samples],
        "score": scores,
        "human": [pool.loc[s[0], "human"] for s in samples],
        "system": [pool.loc[s[0], "system"] for s in samples],
        "seg_idx": [pool.loc[s[0], "seg_idx"] for s in samples],
    })
    out_rows.to_csv(OUT / f"wmt_reseg_{tag}_samples.csv", index=False)
    print(f"   wrote wmt_reseg_{tag}_samples.csv ({len(out_rows)} rows)", flush=True)

    order = ["canonical"] + [f"[{lo:.2f},{hi:.2f})" for lo, hi in rng_buckets]
    per = []
    for b in order:
        s = out_rows[out_rows.bucket == b]
        if len(s) < 10:
            continue
        per.append({"pair": tag, "bucket": b, "n": len(s),
                    "mean_ratio": float(s.ratio.mean()),
                    "rho_vs_human": spearman(s.score, s.human),
                    "mean_score": float(s.score.mean())})
    pdf = pd.DataFrame(per)
    pdf.to_csv(OUT / f"wmt_reseg_{tag}_buckets.csv", index=False)
    print(pdf.to_string(index=False), flush=True)

    if pdf.empty or "bucket" not in pdf.columns:
        print("   NO USABLE BUCKETS: every bucket held fewer than 10 samples. "
              "This pair cannot be run on this ladder; see the unreachable and "
              "excluded counts above.", flush=True)
        pd.DataFrame([{"pair": tag, "testset": a.testset, "status": "no_buckets",
                       "human_score_kind": score_kind,
                       "n_samples": len(samples), "unreachable": n_unreach,
                       "excluded_roundtrip": n_excl,
                       "pre_normalised": n_prenorm}]).to_csv(
            OUT / f"wmt_reseg_{tag}_summary.csv", index=False)
        return 1
    nc = pdf[pdf.bucket != "canonical"]
    mono = spearman_small(list(nc.mean_ratio), list(nc.rho_vs_human)) if len(nc) >= 3 else float("nan")

    def _mono(df: pd.DataFrame) -> float:
        rs, ag = [], []
        for _b, s in df[df.bucket != "canonical"].groupby("bucket"):
            if len(s) >= 10:
                v = spearman(s.score, s.human)
                if not np.isnan(v):
                    rs.append(float(s.ratio.mean())); ag.append(v)
        return spearman_small(rs, ag) if len(rs) >= 3 else float("nan")

    by_seg = {k: v for k, v in out_rows.groupby("seg_idx")}
    seg_ids = np.array(list(by_seg.keys()))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(a.boot):
        pick = rng.choice(seg_ids, len(seg_ids), replace=True)
        v = _mono(pd.concat([by_seg[x] for x in pick], ignore_index=True))
        if not np.isnan(v):
            draws.append(v)
    if draws:
        ci_lo, ci_hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    else:
        ci_lo = ci_hi = float("nan")
    print(f"   segment-level bootstrap ({len(draws)} draws): "
          f"rho={mono:.3f} 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
    pval = exact_perm_p(list(nc.mean_ratio), list(nc.rho_vs_human)) if len(nc) >= 3 else float("nan")
    summary = {
        "pair": tag, "testset": a.testset, "human_score_kind": score_kind,
        "n_buckets": len(nc), "max_ratio_requested": a.max_ratio,
        "ladder": "standard8" if a.n_buckets is None else f"even{a.n_buckets}",
        "canonical_rho": float(pdf[pdf.bucket == "canonical"].rho_vs_human.iloc[0])
                         if (pdf.bucket == "canonical").any() else float("nan"),
        "first_bucket_rho": float(nc.rho_vs_human.iloc[0]) if len(nc) else float("nan"),
        "last_bucket_rho": float(nc.rho_vs_human.iloc[-1]) if len(nc) else float("nan"),
        "monotonicity_rho": mono, "exact_p": pval,
        "boot_ci_lo": ci_lo, "boot_ci_hi": ci_hi,
        "boot_excludes_zero": bool(ci_hi < 0 or ci_lo > 0),
        "metric": a.metric,
        "n_systems": int(out_rows.system.nunique()),
        "n_segments": int(out_rows.seg_idx.nunique()),
        "unreachable_frac": n_unreach / max(n_attempt, 1),
        "excluded_roundtrip": n_excl,
        "pre_normalised": n_prenorm,
        "status": "ok",
    }
    pd.DataFrame([summary]).to_csv(OUT / f"wmt_reseg_{tag}_summary.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
