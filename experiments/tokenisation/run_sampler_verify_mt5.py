#!/usr/bin/env python3
"""Uniformity checks for the mT5 re-segmentation sampler.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import noncanonical as NC
import noncanonical_mt5 as NCM
import provenance as PROV
import tokenisers as T

TOKENISER = "mt5"
N_SAMPLES_PER_STRING = 60_000
MAX_EXHAUSTIVE = 600
N_TEST_STRINGS_PER_CELL = 2
N_SEGMENTS_PER_CELL_BUCKETS = 15
SEEDS = [0, 1, 2, 3, 4]
MIN_SEGMENTATIONS_FOR_TEST = 8


def _pick_test_words(seg: pd.DataFrame, lang: str, condition: str,
                      tries: NC.Tries, n: int) -> list[str]:
    sub = seg[(seg.lang == lang) & (seg.condition == condition)]
    if sub.empty:
        raise RuntimeError(f"no segments for {lang}/{condition} -- refusing to drop a cell")
    words: set[str] = set()
    for text in sub["target"]:
        words.update(NC.words_with_spans(text))

    candidates: list[tuple[int, str]] = []
    for w in words:
        try:
            mdd = NC.WordMDD(w, tries)
        except NC.NoncanonicalError:
            continue
        ways = mdd.ways_total()[0]
        if MIN_SEGMENTATIONS_FOR_TEST <= ways <= MAX_EXHAUSTIVE:
            candidates.append((ways, w))
    candidates.sort(reverse=True)
    chosen = [w for _ways, w in candidates[:n]]
    if len(chosen) < n:
        raise RuntimeError(
            f"{lang}/{condition}: only found {len(chosen)}/{n} words with "
            f"{MIN_SEGMENTATIONS_FOR_TEST}-{MAX_EXHAUSTIVE} segmentations")
    return chosen


def verify_canonical_is_a_path(word: str, tries: NC.Tries) -> dict:
    if not NCM.canonical_roundtrips_mt5(word, TOKENISER):
        return {"word": word, "canonical_is_a_path": None, "n_segmentations": None,
                "reason": "EXCLUDED: fails canonical round-trip under mT5 -- not a trie bug"}
    tok = T.load(TOKENISER)
    canonical_ids = tok(word, add_special_tokens=False)["input_ids"]
    mdd = NC.WordMDD(word, tries)
    ways = mdd.ways_total()[0]
    if ways == 0:
        return {"word": word, "canonical_is_a_path": False, "n_segmentations": 0,
                "reason": "MDD has zero valid segmentations"}
    if ways <= MAX_EXHAUSTIVE:
        all_paths = mdd.enumerate_all(limit=MAX_EXHAUSTIVE)
        found = list(canonical_ids) in all_paths
    else:
        found = True
        pos = None
        remaining_ids = list(canonical_ids)
        first_tid = remaining_ids.pop(0)
        match = [j for j, t in mdd.start_edges if t == first_tid]
        if not match:
            found = False
        else:
            pos = match[0]
            for tid in remaining_ids:
                cands = mdd.edges[pos]
                match = [j for j, t in cands if t == tid]
                if not match:
                    found = False
                    break
                pos = match[0]
        if found and pos != mdd.n:
            found = False
    return {"word": word, "canonical_is_a_path": found, "n_segmentations": ways,
            "canonical_ids": canonical_ids, "reason": "" if found else "CANONICAL PATH MISSING FROM MDD"}


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    tries = NCM.build_tries_mt5(TOKENISER)
    seg = pd.read_parquet(Path.home() / "tokinv_work/repo/panel-pipeline/data/interim/segments.parquet")
    seg = seg[seg.mqm_valid].copy()
    seg["target"] = seg["target"].astype(str).map(NC.normalize_text_for_reseg)

    print("=" * 78); print("1. Canonical-path sanity check (mT5)"); print("=" * 78)
    canon_rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = seg[(seg.lang == lang) & (seg.condition == cond)]
            sample_words = []
            for text in sub["target"].head(30):
                sample_words.extend(NC.words_with_spans(text))
            sample_words = list(dict.fromkeys(sample_words))[:10]
            for w in sample_words:
                r = verify_canonical_is_a_path(w, tries)
                r["lang"] = lang
                r["condition"] = cond
                canon_rows.append(r)
    canon_df = pd.DataFrame(canon_rows)
    n_excluded = int(canon_df.canonical_is_a_path.isna().sum())
    checked = canon_df[canon_df.canonical_is_a_path.notna()]
    n_bad = int((~checked.canonical_is_a_path.astype(bool)).sum())
    print(f"checked {len(canon_df)} words; excluded (canonical round-trip fails): "
          f"{n_excluded}; canonical-path-missing (real trie bugs): {n_bad}")
    canon_df.drop(columns=["canonical_ids"], errors="ignore").to_csv(
        C.OUT_TABLES / "reseg_sampler_canonical_check_mt5.csv", index=False)
    if n_bad:
        bad = checked[~checked.canonical_is_a_path.astype(bool)]
        raise NC.NoncanonicalError(
            f"STOP: canonical tokenisation missing from the MDD for {n_bad} words -- "
            f"trie construction is wrong. Examples:\n"
            f"{bad[['lang','condition','word','reason']].head(10).to_string(index=False)}")

    print(); print("=" * 78); print("2. Uniformity test -- chi-square goodness-of-fit (mT5)"); print("=" * 78)
    rng = random.Random(C.SEED)
    uniformity_rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            test_words = _pick_test_words(seg, lang, cond, tries, N_TEST_STRINGS_PER_CELL)
            for w in test_words:
                mdd = NC.WordMDD(w, tries)
                n_seg = mdd.ways_total()[0]
                all_paths = mdd.enumerate_all(limit=MAX_EXHAUSTIVE)
                assert len(all_paths) == n_seg, "enumeration count mismatch with DP -- bug"
                path_index = {tuple(p): i for i, p in enumerate(all_paths)}
                counts = Counter()
                for _ in range(N_SAMPLES_PER_STRING):
                    ids = mdd.sample_unconditional(rng)
                    NCM.assert_round_trip_mt5(w, ids, TOKENISER)
                    counts[path_index[tuple(ids)]] += 1
                observed = np.array([counts.get(i, 0) for i in range(n_seg)], dtype=float)
                expected = np.full(n_seg, N_SAMPLES_PER_STRING / n_seg)
                if n_seg == 1:
                    chi2_stat, p_value = 0.0, 1.0
                else:
                    chi2_stat, p_value = stats.chisquare(observed, expected)
                uniformity_rows.append({
                    "lang": lang, "condition": cond, "word": w,
                    "n_segmentations": n_seg, "n_samples": N_SAMPLES_PER_STRING,
                    "chi2_stat": float(chi2_stat), "p_value": float(p_value),
                })
                print(f"{lang} {cond:10s} {w!r:20s} n_seg={n_seg:4d} chi2={chi2_stat:8.3f} p={p_value:.4f}")

    unif_df = pd.DataFrame(uniformity_rows)
    unif_df.to_csv(C.OUT_TABLES / "sampler_uniformity_mt5.csv", index=False)
    n_fail_05 = int((unif_df.p_value < 0.05).sum())
    print(f"\np-value distribution: min={unif_df.p_value.min():.4f} "
          f"median={unif_df.p_value.median():.4f} max={unif_df.p_value.max():.4f}")
    print(f"strings with p < 0.05: {n_fail_05} / {len(unif_df)}")

    ks_stat, ks_p = stats.kstest(unif_df.p_value, "uniform")
    print(f"KS test of p-value distribution against Uniform(0,1): stat={ks_stat:.4f} p={ks_p:.4f}")
    print("  (go/no-go signal, same as the XLM-R verification)")

    print(); print("=" * 78); print("3. Length-ratio bucket reachability (5 seeds each, mT5)"); print("=" * 78)
    bucket_rows = []
    n_texts_excluded = 0
    excluded_examples = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = seg[(seg.lang == lang) & (seg.condition == cond)]
            texts = sub["target"].dropna().head(N_SEGMENTS_PER_CELL_BUCKETS).tolist()
            for text in texts:
                if not text or not text.strip():
                    continue
                try:
                    text_bucket_rows = []
                    mdds = NCM.build_mdds_mt5(text, TOKENISER)
                    tok = T.load(TOKENISER)
                    canonical_k = len(tok(text, add_special_tokens=False)["input_ids"])
                    dist = NC.whole_text_length_distribution(mdds)
                    for lo, hi in NC.RATIO_BUCKETS:
                        reachable_totals = NC.reachable_totals_in_bucket(dist, canonical_k, lo, hi)
                        n_reachable_seeds = 0
                        for seed in SEEDS:
                            r_rng = random.Random(seed)
                            if reachable_totals:
                                weights = [dist[k] for k in reachable_totals]
                                total_k = r_rng.choices(reachable_totals, weights=weights, k=1)[0]
                                ids = NC.sample_text_exact_total(mdds, total_k, r_rng)
                                NCM.assert_round_trip_mt5(text, ids, TOKENISER)
                                n_reachable_seeds += 1
                        text_bucket_rows.append({
                            "lang": lang, "condition": cond, "bucket_lo": lo, "bucket_hi": hi,
                            "text_len_chars": len(text),
                            "n_reachable_seeds": n_reachable_seeds, "n_seeds": len(SEEDS),
                        })
                except NC.NoncanonicalError as exc:
                    n_texts_excluded += 1
                    if len(excluded_examples) < 10:
                        excluded_examples.append({"lang": lang, "condition": cond,
                                                   "text": text[:60], "reason": str(exc)[:120]})
                    continue
                bucket_rows.extend(text_bucket_rows)

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(C.OUT_TABLES / "reseg_sampler_bucket_reachability_mt5.csv", index=False)
    print(f"\ntexts excluded (canonical round-trip or trie failure): {n_texts_excluded} "
          f"/ {n_texts_excluded + len(bucket_df.groupby(['lang','condition','text_len_chars']))} approx")
    if excluded_examples:
        print(pd.DataFrame(excluded_examples).to_string(index=False))
    print("\nReachability rate by (lang, condition, bucket):")
    bucket_df["bucket_label"] = bucket_df.apply(lambda r: f"[{r.bucket_lo:.2f},{r.bucket_hi:.2f})", axis=1)
    summary = (bucket_df.assign(any_reach=bucket_df.n_reachable_seeds > 0)
               .groupby(["lang", "condition", "bucket_label"], observed=True)["any_reach"]
               .mean().reset_index())
    print(summary.to_string(index=False))

    try:
        PROV.register(
            "reseg_sampler_verification_mt5", {
                "tokeniser": TOKENISER,
                "canonical_check_n": len(canon_df), "canonical_check_n_bad": n_bad,
                "canonical_check_n_excluded_roundtrip": n_excluded,
                "uniformity_n_strings": len(unif_df),
                "uniformity_p_value_min": float(unif_df.p_value.min()),
                "uniformity_p_value_median": float(unif_df.p_value.median()),
                "uniformity_p_value_max": float(unif_df.p_value.max()),
                "uniformity_n_below_alpha05": n_fail_05,
                "uniformity_ks_stat_vs_uniform01": float(ks_stat),
                "uniformity_ks_p_vs_uniform01": float(ks_p),
                "n_samples_per_string": N_SAMPLES_PER_STRING,
                "n_texts_excluded_bucket_stage": n_texts_excluded,
            },
            PROV.stamp(Path(__file__), modules=["noncanonical", "noncanonical_mt5", "tokenisers", "constants"],
                       seed=C.SEED,
                       note="mT5 re-segmentation sampler verification, mirroring the XLM-R check exactly"))
    except Exception as exc:
        print(f"(non-fatal: PROV.register failed, does not affect the verification result above: {exc})")

    print(f"\nSTOP CONDITION CHECK: KS p-value = {ks_p:.4f}. "
          f"{'PASS -- sampler is demonstrably uniform, proceeding to re-segmentation proper is justified.' if ks_p >= 0.05 else 'FAIL -- KS test indicates non-uniform p-values, DO NOT PROCEED TO SCORING.'}")
    return 0 if ks_p >= 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
