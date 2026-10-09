"""WMT25 English-Serbian: COMET-22 on Cyrillic and on Latin (Gaj) renderings.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
LIVE = SB_ROOT / "repo" / "panel-pipeline"
sys.path.insert(0, str(LIVE / "src"))
sys.path.insert(0, str(ROB / "scripts" / "metaeval"))
sys.path.insert(0, str(ROB / "scripts" / "beyond"))
import tokenisers as T
import metrics_panel as MP
from panel_metaeval import acc_star_eq_pooled, make_perm, spa
from srp_translit import cyr_to_lat, lat_to_cyr, cyrillic_fraction

D = SB_ROOT / "staging" / "wmt25" / "mt-metrics-eval-v2" / "wmt25"
LP = "en-sr_Cyrl_RS"
OUT = ROB / "results" / "tables"
BATCH = int(os.environ.get("BATCH", "32"))
TOKENISER = "xlmr"
SEED = 20260919


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


def read_lines(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8")]


def main():
    src = read_lines(D / "sources" / f"{LP}.txt")
    ref = read_lines(D / "references" / f"{LP}.refA.txt")
    n_seg = len(src)
    assert len(ref) == n_seg, "source/reference length mismatch"
    print(f"segments: {n_seg}", flush=True)

    esa_rows = [l.rstrip("\n").split("\t") for l in
                open(D / "human-scores" / f"{LP}.esa-merged.seg.score", encoding="utf-8")]
    by_sys = {}
    for s, v in esa_rows:
        by_sys.setdefault(s, []).append(v)
    for s, v in by_sys.items():
        assert len(v) == n_seg, f"{s} has {len(v)} scores, expected {n_seg}"
    scored_sys = sorted(s for s, v in by_sys.items() if any(x != "None" for x in v))
    print(f"systems with ESA scores: {len(scored_sys)} of {len(by_sys)}", flush=True)

    rows = []
    for s in scored_sys:
        out = read_lines(D / "system-outputs" / LP / f"{s}.txt")
        assert len(out) == n_seg, f"{s} output length {len(out)} != {n_seg}"
        for i, (txt, sc) in enumerate(zip(out, by_sys[s])):
            if sc == "None":
                continue
            rows.append({"system": s, "seg_idx": i, "esa": float(sc),
                         "src": src[i], "ref_cyr": ref[i], "mt_cyr": txt})
    df = pd.DataFrame(rows)
    print(f"scored (system, segment) pairs: {len(df)}", flush=True)

    df["mt_lat"] = [cyr_to_lat(t) for t in df.mt_cyr]
    df["ref_lat"] = [cyr_to_lat(t) for t in df.ref_cyr]
    df["mt_back"] = [lat_to_cyr(t) for t in df.mt_lat]
    df["roundtrip_exact"] = df.mt_back == df.mt_cyr
    df["cyr_frac"] = [cyrillic_fraction(t) for t in df.mt_cyr]

    def charerr(a, b):
        if not a:
            return float("nan")
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
            prev = cur
        return prev[-1] / len(a)

    sub = df.sample(n=min(400, len(df)), random_state=SEED)
    cers = [charerr(a, b) for a, b in zip(sub.mt_cyr, sub.mt_back)]
    print(f"\n== round trip ==", flush=True)
    print(f"exact round trips: {df.roundtrip_exact.mean():.4f} "
          f"({int(df.roundtrip_exact.sum())}/{len(df)})", flush=True)
    print(f"mean round-trip CER (n={len(sub)} sample): {np.nanmean(cers):.6f}", flush=True)
    print(f"mean Cyrillic letter fraction of MT output: {df.cyr_frac.mean():.4f}", flush=True)

    df["ntok_cyr"] = [len(T.tokenise(t, TOKENISER, add_special_tokens=False)) for t in df.mt_cyr]
    df["ntok_lat"] = [len(T.tokenise(t, TOKENISER, add_special_tokens=False)) for t in df.mt_lat]
    df["tok_ratio"] = df.ntok_lat / df.ntok_cyr.replace(0, np.nan)
    print(f"\n== tokenisation ==", flush=True)
    print(f"mean tokens cyr={df.ntok_cyr.mean():.1f} lat={df.ntok_lat.mean():.1f} "
          f"ratio={df.tok_ratio.mean():.4f}", flush=True)
    over = ((df.ntok_cyr > 510) | (df.ntok_lat > 510))
    print(f"segments exceeding 510 XLM-R tokens in either script: {int(over.sum())} "
          f"({over.mean():.3f}) -- COMET truncates these identically in both "
          f"conditions, reported as a sensitivity split below", flush=True)
    df["over_limit"] = over

    model = MP.load_comet("comet22", device="cuda")
    for cond, mtc, rfc in [("cyrillic", "mt_cyr", "ref_cyr"), ("latin", "mt_lat", "ref_lat")]:
        samples = [{"src": s, "mt": m, "ref": r} for s, m, r in
                   zip(df.src, df[mtc], df[rfc])]
        res = model.predict(samples, batch_size=BATCH, gpus=1, progress_bar=False)
        df[f"comet22_{cond}"] = np.asarray(res.scores, dtype=float)
        print(f"scored {cond}: mean={df[f'comet22_{cond}'].mean():.4f}", flush=True)
    df.drop(columns=["src", "ref_cyr", "mt_cyr", "mt_lat", "ref_lat", "mt_back"]).to_csv(
        OUT / "serbian_scores.csv", index=False)

    res_rows = []
    for label, d in [("all", df), ("fits_512", df[~df.over_limit])]:
        for cond in ("cyrillic", "latin"):
            col = f"comet22_{cond}"
            dh, dm = [], []
            for _, g in d.groupby("seg_idx"):
                m = g[col].to_numpy(float); h = g["esa"].to_numpy(float)
                if len(m) < 2:
                    continue
                i, j = np.triu_indices(len(m), 1)
                dh.append(h[i]-h[j]); dm.append(m[i]-m[j])
            acc, eps = acc_star_eq_pooled(np.concatenate(dh), np.concatenate(dm))
            res_rows.append({"subset": label, "condition": cond, "n": len(d),
                             "spearman_vs_esa": spearman(d[col], d["esa"]),
                             "acc_star_eq": acc, "epsilon": eps,
                             "n_pairs": int(sum(len(x) for x in dh))})
    agree = pd.DataFrame(res_rows)
    agree.to_csv(OUT / "serbian_agreement.csv", index=False)
    print("\n== agreement with ESA ==")
    print(agree.round(4).to_string(index=False))

    flips = []
    for label, d in [("all", df), ("fits_512", df[~df.over_limit])]:
        mc = d.groupby("system")["comet22_cyrillic"].mean()
        ml = d.groupby("system")["comet22_latin"].mean()
        me = d.groupby("system")["esa"].mean()
        common = sorted(set(mc.index) & set(ml.index))
        n_flip = tot = 0
        for a, b in itertools.combinations(common, 2):
            tot += 1
            if np.sign(mc[a]-mc[b]) != np.sign(ml[a]-ml[b]):
                n_flip += 1
        flips.append({"subset": label, "n_systems": len(common), "n_pairs": tot,
                      "n_flips": n_flip,
                      "spearman_syslevel_cyr_vs_esa": spearman(mc[common], me[common]),
                      "spearman_syslevel_lat_vs_esa": spearman(ml[common], me[common])})
    fl = pd.DataFrame(flips)
    fl.to_csv(OUT / "serbian_flips.csv", index=False)
    print("\n== system ranking flips (Cyrillic vs Latin) ==")
    print(fl.round(4).to_string(index=False))

    with open(OUT / "serbian_provenance.json", "w") as f:
        json.dump({"n_segments": n_seg, "n_systems_scored": len(scored_sys),
                   "n_pairs_scored": int(len(df)),
                   "roundtrip_exact_rate": float(df.roundtrip_exact.mean()),
                   "mean_roundtrip_cer_sample": float(np.nanmean(cers)),
                   "mean_tok_ratio_lat_over_cyr": float(df.tok_ratio.mean()),
                   "n_over_512": int(over.sum())}, f, indent=2)


if __name__ == "__main__":
    main()
