"""Byte-identical re-segmentation on WMT24 English-German and English-Spanish, with the IndicMT Eval pipeline.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import os, random, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
LIVE = SB_ROOT / "repo" / "panel-pipeline"
sys.path.insert(0, str(LIVE / "src"))
import noncanonical as NC
import tokenisers as T
import metrics_panel as MP

XLSX = LIVE / "data" / "latin" / "wmt24_ende_enes_metrics.xlsx"
OUT = ROB / "results" / "tables"
TOKENISER = "xlmr"
SEEDS = [0, 1, 2, 3, 4]
N_SEG = int(os.environ.get("N_SEG", "300"))
BATCH = int(os.environ.get("BATCH", "48"))
SEED = 20260919
MAX_SEQ_IDS = 510
MAX_CANON_K = 160


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else float("nan")


def load_pair(sheet):
    d = pd.read_excel(XLSX, sheet_name=sheet)
    d = d.dropna(subset=["source", "target", "refA", "mqm_score"])
    d = d.drop_duplicates(subset=["system", "seg_id"])
    return d


def main():
    rng = random.Random(SEED)
    model = MP.load_comet("comet22", device="cuda")
    print("comet22 loaded", flush=True)

    all_rows, summary = [], []
    for sheet, tag in [("German", "deu"), ("Spanish", "spa")]:
        d = load_pair(sheet)
        print(f"\n== {tag}: {len(d)} rows, {d.system.nunique()} systems, "
              f"{d.seg_id.nunique()} segments", flush=True)
        print(f"   orientation check: spearman(COMET, mqm_score) = "
              f"{spearman(d['COMET'], d['mqm_score']):.4f}", flush=True)

        d = d.copy()
        d["_canon_k"] = [len(T.tokenise(NC.normalize_text_for_reseg(str(t)), TOKENISER,
                                        add_special_tokens=False)) for t in d["target"]]
        d["_src_k"] = [len(T.tokenise(str(t), TOKENISER, add_special_tokens=False))
                       for t in d["source"]]
        d["_ref_k"] = [len(T.tokenise(str(t), TOKENISER, add_special_tokens=False))
                       for t in d["refA"]]
        n_before = len(d)
        d = d[(d["_canon_k"] <= MAX_CANON_K) & (d["_src_k"] <= MAX_SEQ_IDS)
              & (d["_ref_k"] <= MAX_SEQ_IDS)]
        print(f"   length filter: kept {len(d)}/{n_before} rows with canonical_k <= "
              f"{MAX_CANON_K} (worst-case 3.0x fragmentation must fit XLM-R's 514 "
              f"positions); excluded {n_before - len(d)}", flush=True)

        pool = d.sample(n=min(N_SEG, len(d)), random_state=SEED).reset_index(drop=True)
        samples = []
        n_attempt = n_excl = n_unreach = n_toolong = 0
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
            for lo, hi in NC.RATIO_BUCKETS:
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
        for st in range(0, len(samples), BATCH):
            chunk = samples[st:st+BATCH]
            src_b, mt_b, ref_b = [], [], []
            for (i, _, _, ids, _) in chunk:
                r = pool.loc[i]
                src_b.append(T.tokenise(str(r["source"]), TOKENISER, add_special_tokens=False))
                ref_b.append(T.tokenise(str(r["refA"]), TOKENISER, add_special_tokens=False))
                mt_b.append(ids if ids else T.tokenise(" ", TOKENISER, add_special_tokens=False))
            scores.extend(MP.score_from_ids_batch(model, src_b, mt_b, ref_b))
            if st % (BATCH*40) == 0:
                print(f"   scored {st+len(chunk)}/{len(samples)}", flush=True)

        rows = pd.DataFrame({
            "pair": tag,
            "row_idx": [s[0] for s in samples],
            "bucket": [s[1] for s in samples],
            "seed": [s[2] for s in samples],
            "ratio": [s[4] for s in samples],
            "comet22": scores,
            "mqm_score": [pool.loc[s[0], "mqm_score"] for s in samples],
            "system": [pool.loc[s[0], "system"] for s in samples],
            "seg_id": [pool.loc[s[0], "seg_id"] for s in samples],
        })
        all_rows.append(rows)

        order = ["canonical"] + [f"[{lo:.2f},{hi:.2f})" for lo, hi in NC.RATIO_BUCKETS]
        per = []
        for b in order:
            s = rows[rows.bucket == b]
            if len(s) < 10:
                continue
            per.append({"pair": tag, "bucket": b, "n": len(s),
                        "mean_ratio": float(s.ratio.mean()),
                        "rho_vs_mqm": spearman(s.comet22, s.mqm_score),
                        "mean_comet": float(s.comet22.mean())})
        pdf = pd.DataFrame(per)
        nc = pdf[pdf.bucket != "canonical"]
        mono = spearman(np.arange(len(nc)), nc.rho_vs_mqm.to_numpy()) if len(nc) >= 3 else float("nan")
        can_rho = float(pdf[pdf.bucket == "canonical"]["rho_vs_mqm"].iloc[0]) if (pdf.bucket == "canonical").any() else float("nan")
        last_rho = float(nc.rho_vs_mqm.iloc[-1]) if len(nc) else float("nan")
        summary.append({"pair": tag, "canonical_rho": can_rho,
                        "most_fragmented_rho": last_rho,
                        "absolute_drop": can_rho - last_rho,
                        "relative_drop": (can_rho - last_rho)/can_rho if can_rho else float("nan"),
                        "monotonicity_spearman": mono, "n_buckets": len(nc)})
        print(pdf.round(4).to_string(index=False), flush=True)

    pd.concat(all_rows).to_csv(OUT / "latin_wmt24_reseg_samples.csv", index=False)
    per_all = []
    for rows in all_rows:
        order = ["canonical"] + [f"[{lo:.2f},{hi:.2f})" for lo, hi in NC.RATIO_BUCKETS]
        for b in order:
            s = rows[rows.bucket == b]
            if len(s) < 10:
                continue
            per_all.append({"pair": s.pair.iloc[0], "bucket": b, "n": len(s),
                            "mean_ratio": float(s.ratio.mean()),
                            "rho_vs_mqm": spearman(s.comet22, s.mqm_score),
                            "mean_comet": float(s.comet22.mean())})
    pd.DataFrame(per_all).to_csv(OUT / "latin_wmt24_reseg_buckets.csv", index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "latin_wmt24_reseg_summary.csv", index=False)
    print("\n== SUMMARY ==")
    print(sdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
