"""Merge the eleven metric scores into one segment-level panel.

Ran on the GPU cluster; see experiments/README.md.
"""
import sys, json
import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
SEG = f"{REPO}/data/interim/segments.parquet"
T = f"{REPO}/results/tables"

PRIOR = {
    "comet_22": +1, "bertscore": +1, "bleu": +1, "chrf": +1, "ter": -1,
    "bleurt20": +1, "metricx24": -1, "cometkiwi22": +1, "cometkiwi23": +1,
    "xcomet_xl": +1, "xcomet_xxl": +1,
}

def main():
    seg = pd.read_parquet(SEG)
    seg["lang"] = seg["lang"].astype(str)
    seg["condition"] = seg["condition"].astype(str)
    base = seg[["segment_id", "lang", "condition", "system", "source_en",
                "mqm_score", "mqm_valid", "comet_22", "bertscore", "bleu",
                "chrf", "ter"]].copy()
    n0 = len(base)
    assert base.duplicated(["segment_id", "condition"]).sum() == 0, "segment_id x condition not unique"

    report = {"n_rows_segments_parquet": int(n0)}

    def merge_csv(df, path, valcol, newname, model_filter=None):
        c = pd.read_csv(path)
        if model_filter is not None:
            c = c[c["model"] == model_filter]
        c = c[["segment_id", "condition", valcol]].rename(columns={valcol: newname})
        assert c.duplicated(["segment_id", "condition"]).sum() == 0, f"dupe keys in {path}"
        out = df.merge(c, on=["segment_id", "condition"], how="left")
        assert len(out) == len(df), f"merge changed row count for {newname}"
        return out

    base = merge_csv(base, f"{T}/metricx24_scores.csv", "metricx24", "metricx24")
    base = merge_csv(base, f"{T}/bleurt20_scores.csv", "bleurt20", "bleurt20")
    for nm, fn in [("cometkiwi22", "unified_metric_cometkiwi22_scores.csv"),
                   ("cometkiwi23", "unified_metric_cometkiwi23_scores.csv"),
                   ("xcomet_xl", "unified_metric_xcomet_xl_scores.csv"),
                   ("xcomet_xxl", "unified_metric_xcomet_xxl_scores.csv")]:
        base = merge_csv(base, f"{T}/{fn}", "score", nm)

    metrics = list(PRIOR.keys())

    cov = {m: int(base[m].notna().sum()) for m in metrics}
    report["coverage_non_null"] = cov

    piv = base.pivot_table(index="segment_id", columns="condition",
                           values="mqm_score", aggfunc="first")
    both = piv.dropna()
    mqm_identical = bool((both["native"] == both["romanised"]).all())
    report["mqm_identical_across_conditions"] = mqm_identical
    report["mqm_n_compared"] = int(len(both))
    if not mqm_identical:
        report["mqm_n_differing"] = int((both["native"] != both["romanised"]).sum())

    struct = {}
    for (lg, cond), sub in base.groupby(["lang", "condition"]):
        g = sub.groupby("source_en")["system"].nunique()
        struct[f"{lg}|{cond}"] = {
            "n_rows": int(len(sub)),
            "n_systems": int(sub["system"].nunique()),
            "n_sources": int(len(g)),
            "n_sources_ge2sys": int((g >= 2).sum()),
            "n_sources_full": int((g == sub["system"].nunique()).sum()),
            "n_pairs_within_source": int(sum(k * (k - 1) // 2 for k in g.values)),
        }
    report["structure"] = struct

    def spearman(a, b):
        ok = (~pd.isna(a)) & (~pd.isna(b))
        a, b = a[ok], b[ok]
        if len(a) < 10:
            return float("nan")
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        ra = ra - ra.mean(); rb = rb - rb.mean()
        d = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra * rb).sum() / d) if d > 0 else float("nan")

    nat = base[(base["condition"] == "native") & (base["mqm_valid"])]
    orient = {}
    for m in metrics:
        rho = spearman(nat[m].to_numpy(), nat["mqm_score"].to_numpy())
        detected = int(np.sign(rho)) if not np.isnan(rho) else 0
        orient[m] = {"spearman_native_pooled": rho, "detected": detected,
                     "prior": PRIOR[m], "agrees": bool(detected == PRIOR[m])}
    report["orientation"] = orient

    disagree = [m for m, v in orient.items() if not v["agrees"]]
    report["orientation_disagreements"] = disagree

    base.to_csv(f"{T}/metaeval_panel.csv", index=False)
    report["panel_rows"] = int(len(base))
    print(json.dumps(report, indent=2))
    with open(f"{T}/metaeval_panel_provenance.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
