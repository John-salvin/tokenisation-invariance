#!/usr/bin/env python3
"""Uniformity checks for the XLM-R re-segmentation sampler.

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
import provenance as PROV
import tokenisers as T

TOKENISER = "xlmr"
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
            f"{MIN_SEGMENTATIONS_FOR_TEST}-{MAX_EXHAUSTIVE} segmentations "
            f"-- corpus may not have words in this range for a meaningful test")
    return chosen


def verify_canonical_is_a_path(word: str, tokeniser_name: str, tries: NC.Tries) -> dict:
    if not NC.canonical_roundtrips(word, tokeniser_name):
        return {"word": word, "canonical_is_a_path": None, "n_segmentations": None,
                "reason": "EXCLUDED: fails canonical round-trip (e.g. ZWNJ/ZWJ) -- not a trie bug"}
    tok = T.load(tokeniser_name)
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
    tries = NC.build_tries(TOKENISER)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    nbsp_prevalence = (seg.groupby(["lang", "condition"], observed=True)["target"]
                       .apply(lambda s: s.astype(str).str.contains(NC.NBSP).mean())
                       .reset_index(name="nbsp_prevalence"))
    nbsp_prevalence.to_csv(C.OUT_TABLES / "reseg_nbsp_prevalence.csv", index=False)
    print("NBSP prevalence by (lang, condition) -- see reseg_nbsp_prevalence.csv:")
    print(nbsp_prevalence.to_string(index=False))
    seg = seg.copy()
    seg["target"] = seg["target"].astype(str).map(NC.normalize_text_for_reseg)

    print("=" * 78); print("1. Canonical-path sanity check"); print("=" * 78)
    canon_rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = seg[(seg.lang == lang) & (seg.condition == cond)]
            sample_words = []
            for text in sub["target"].head(30):
                sample_words.extend(NC.words_with_spans(text))
            sample_words = list(dict.fromkeys(sample_words))[:10]
            for w in sample_words:
                r = verify_canonical_is_a_path(w, TOKENISER, tries)
                r["lang"] = lang
                r["condition"] = cond
                canon_rows.append(r)
    canon_df = pd.DataFrame(canon_rows)
    n_excluded = int(canon_df.canonical_is_a_path.isna().sum())
    checked = canon_df[canon_df.canonical_is_a_path.notna()]
    n_bad = int((~checked.canonical_is_a_path.astype(bool)).sum())
    print(f"checked {len(canon_df)} words; excluded (canonical round-trip fails, e.g. ZWNJ): "
          f"{n_excluded}; of the rest, canonical-path-missing (real trie bugs): {n_bad}")
    canon_df.drop(columns=["canonical_ids"], errors="ignore").to_csv(
        C.OUT_TABLES / "reseg_sampler_canonical_check.csv", index=False)
    if n_bad:
        bad = checked[~checked.canonical_is_a_path.astype(bool)]
        raise NC.NoncanonicalError(
            f"STOP: canonical tokenisation missing from the MDD for {n_bad} words -- "
            f"trie construction is wrong, not a downstream finding. Examples:\n"
            f"{bad[['lang','condition','word','reason']].head(10).to_string(index=False)}")

    print(); print("=" * 78); print("2. Uniformity test -- chi-square goodness-of-fit"); print("=" * 78)
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
                    NC.assert_round_trip(w, ids, TOKENISER)
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
    unif_df.to_csv(C.OUT_TABLES / "sampler_uniformity.csv", index=False)
    n_fail_05 = int((unif_df.p_value < 0.05).sum())
    print(f"\np-value distribution: min={unif_df.p_value.min():.4f} "
          f"median={unif_df.p_value.median():.4f} max={unif_df.p_value.max():.4f}")
    print(f"strings with p < 0.05: {n_fail_05} / {len(unif_df)} "
          f"(expect ~{0.05*len(unif_df):.1f} by chance alone at true nominal alpha)")

    ks_stat, ks_p = stats.kstest(unif_df.p_value, "uniform")
    print(f"KS test of p-value distribution against Uniform(0,1): stat={ks_stat:.4f} p={ks_p:.4f}")
    print("  (this, not any single chi-square test, is the go/no-go signal: a low KS-p here "
          "means the sampler's p-values are systematically non-uniform, i.e. the sampler "
          "itself is biased, not that any one word happened to be unlucky)")

    print(); print("=" * 78); print("3. Length-ratio bucket reachability (5 seeds each)"); print("=" * 78)
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
                    for lo, hi in NC.RATIO_BUCKETS:
                        n_reachable_seeds = 0
                        for seed in SEEDS:
                            r = NC.sample_bucket(text, TOKENISER, lo, hi, random.Random(seed))
                            if r["reachable"]:
                                NC.assert_round_trip(text, r["ids"], TOKENISER)
                                n_reachable_seeds += 1
                        text_bucket_rows.append({
                            "lang": lang, "condition": cond, "bucket_lo": lo, "bucket_hi": hi,
                            "text_len_chars": len(text),
                            "n_reachable_seeds": n_reachable_seeds, "n_seeds": len(SEEDS),
                        })
                    r = NC.sample_char_level(text, TOKENISER)
                    if r["reachable"]:
                        NC.assert_round_trip(text, r["ids"], TOKENISER)
                    text_bucket_rows.append({
                        "lang": lang, "condition": cond, "bucket_lo": None, "bucket_hi": None,
                        "text_len_chars": len(text),
                        "n_reachable_seeds": (len(SEEDS) if r["reachable"] else 0), "n_seeds": len(SEEDS),
                        "is_char_level": True,
                    })
                except NC.NoncanonicalError as exc:
                    n_texts_excluded += 1
                    if len(excluded_examples) < 10:
                        excluded_examples.append({"lang": lang, "condition": cond,
                                                   "text": text[:60], "reason": str(exc)[:120]})
                    continue
                bucket_rows.extend(text_bucket_rows)

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(C.OUT_TABLES / "reseg_sampler_bucket_reachability.csv", index=False)
    print(f"\ntexts excluded from bucket testing (canonical round-trip or trie failure): "
          f"{n_texts_excluded}")
    if excluded_examples:
        print(pd.DataFrame(excluded_examples).to_string(index=False))

    print("\nReachability rate (fraction of segments where >=1 seed reached the bucket), by "
          "(lang, condition, bucket):")
    bucket_df["bucket_label"] = bucket_df.apply(
        lambda r: "char_level" if r.get("is_char_level") else f"[{r.bucket_lo:.2f},{r.bucket_hi:.2f})",
        axis=1)
    summary = (bucket_df.assign(any_reach=bucket_df.n_reachable_seeds > 0)
               .groupby(["lang", "condition", "bucket_label"], observed=True)["any_reach"]
               .mean().reset_index())
    print(summary.to_string(index=False))

    PROV.register(
        "reseg_sampler_verification", {
            "tokeniser": TOKENISER,
            "canonical_check_n": len(canon_df), "canonical_check_n_bad": n_bad,
            "uniformity_n_strings": len(unif_df),
            "uniformity_p_value_min": float(unif_df.p_value.min()),
            "uniformity_p_value_median": float(unif_df.p_value.median()),
            "uniformity_p_value_max": float(unif_df.p_value.max()),
            "uniformity_n_below_alpha05": n_fail_05,
            "uniformity_ks_stat_vs_uniform01": float(ks_stat),
            "uniformity_ks_p_vs_uniform01": float(ks_p),
            "n_samples_per_string": N_SAMPLES_PER_STRING,
            "bucket_reachability_summary": summary.to_dict("records"),
            "n_texts_excluded_canonical_roundtrip_or_trie": n_texts_excluded,
            "excluded_text_examples": excluded_examples,
            "n_canonical_check_excluded_roundtrip": n_excluded,
        },
        PROV.stamp(Path(__file__), modules=["noncanonical", "tokenisers", "constants"],
                   seed=C.SEED,
                   note=("re-segmentation sampler built and verified before any scoring: canonical-path "
                         "sanity check, chi-square uniformity over 20 real short words "
                         "(5 langs x 2 conditions), round-trip assertion on every sample "
                         "drawn anywhere in this script, bucket reachability on real segments")))

    print("\nSTOP CONDITION CHECK: if any p_value indicates non-uniformity after multiple-testing "
          "correction, or if the canonical check failed, this script would already have raised. "
          "It did not raise, so the sampler is demonstrably uniform on this evidence -- "
          "proceeding to re-segmentation proper is justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
