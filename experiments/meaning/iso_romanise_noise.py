"""ISO 15919 romanisation, round-trip character error rate and synthetic noise (CPU).

Ran on the GPU cluster; see experiments/README.md.
"""
import sys, os, json
import numpy as np
import pandas as pd
from aksharamukha import transliterate as tr
import jiwer

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
SEG = f"{REPO}/data/interim/segments.parquet"
T = f"{REPO}/results/tables"
INTERIM = f"{REPO}/data/interim"
SEED = 20260919
NOISE_LEVELS = [0.02, 0.05, 0.10, 0.20]

SCRIPT = {"hin": "Devanagari", "mar": "Devanagari", "guj": "Gujarati",
          "tam": "Tamil", "mal": "Malayalam"}


def cer(ref, hyp):
    if not ref:
        return float("nan")
    try:
        return float(jiwer.cer(ref, hyp))
    except Exception:
        return float("nan")


def noise_text(s, target_cer, rng, alphabet):
    idx = [i for i, c in enumerate(s) if not c.isspace()]
    if not idx or target_cer <= 0:
        return s
    k = int(round(target_cer * len(s)))
    k = min(k, len(idx))
    if k == 0:
        return s
    pos = rng.choice(len(idx), size=k, replace=False)
    chars = list(s)
    drop = set()
    for p in pos:
        i = idx[p]
        if rng.random() < 0.5:
            chars[i] = alphabet[int(rng.integers(len(alphabet)))]
        else:
            drop.add(i)
    return "".join(c for j, c in enumerate(chars) if j not in drop)


def main():
    rng = np.random.default_rng(SEED)
    seg = pd.read_parquet(SEG)
    seg["lang"] = seg["lang"].astype(str)
    seg["condition"] = seg["condition"].astype(str)
    nat = seg[seg["condition"] == "native"].copy()
    print(f"native rows: {len(nat)}")

    rows = []
    for lang, sub in nat.groupby("lang"):
        src_script = SCRIPT[lang]
        print(f"-- {lang} ({src_script}) n={len(sub)}", flush=True)
        for _, r in sub.iterrows():
            tgt_iso = tr.process(src_script, "ISO", r["target"])
            ref_iso = tr.process(src_script, "ISO", r["reference"])
            tgt_back = tr.process("ISO", src_script, tgt_iso)
            ref_back = tr.process("ISO", src_script, ref_iso)
            rows.append({
                "segment_id": r["segment_id"], "lang": lang, "system": r["system"],
                "source_en": r["source_en"], "mqm_score": r["mqm_score"],
                "mqm_valid": r["mqm_valid"],
                "target_native": r["target"], "reference_native": r["reference"],
                "target_iso": tgt_iso, "reference_iso": ref_iso,
                "target_roundtrip": tgt_back, "reference_roundtrip": ref_back,
                "cer_target": cer(r["target"], tgt_back),
                "cer_reference": cer(r["reference"], ref_back),
            })
    df = pd.DataFrame(rows)

    cer_rows = []
    for lang, sub in df.groupby("lang"):
        c = sub["cer_target"].dropna()
        cer_rows.append({
            "lang": lang, "n": int(len(c)),
            "mean": c.mean(), "std": c.std(), "min": c.min(),
            "p25": c.quantile(.25), "median": c.median(), "p75": c.quantile(.75),
            "p90": c.quantile(.90), "p95": c.quantile(.95), "p99": c.quantile(.99),
            "max": c.max(),
            "frac_exact_zero": float((c == 0).mean()),
            "frac_below_0.01": float((c < 0.01).mean()),
            "mean_cer_reference": sub["cer_reference"].dropna().mean(),
        })
    cdf = pd.DataFrame(cer_rows).sort_values("mean")
    cdf.to_csv(f"{T}/iso15919_cer.csv", index=False)
    print("\n== ISO 15919 round-trip CER (target) ==")
    print(cdf.round(5).to_string(index=False))

    alpha = sorted({ch for s in df["target_iso"].head(2000) for ch in s if not ch.isspace()})
    print(f"\nsubstitution alphabet size: {len(alpha)}")
    for lv in NOISE_LEVELS:
        tag = f"{int(lv*100):02d}"
        df[f"target_noise{tag}"] = [noise_text(s, lv, rng, alpha) for s in df["target_iso"]]
        df[f"reference_noise{tag}"] = [noise_text(s, lv, rng, alpha) for s in df["reference_iso"]]
        m = np.mean([cer(a, b) for a, b in zip(
            df["target_iso"].head(1500), df[f"target_noise{tag}"].head(1500))])
        print(f"  noise level {lv:.2f} -> measured CER vs ISO = {m:.4f}")

    df.to_parquet(f"{INTERIM}/iso15919_segments.parquet", index=False)
    print(f"\nwrote {INTERIM}/iso15919_segments.parquet  rows={len(df)}")


if __name__ == "__main__":
    main()
