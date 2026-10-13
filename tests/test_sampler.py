"""The sampler is the paper's intervention: every draw must decode to the
identical string, land in its bucket, and match the committed samples."""
import random

import pandas as pd
import pytest

from tokinv import sampler as S
from tokinv import tokenisers as T
from tokinv.paths import DATA, INDIC

TEXTS = ["The cat sat on the mat.", "ગુજરાતી ભાષા ખૂબ સુંદર છે",
         "தமிழ் ஒரு பழமையான மொழி", "jaheratanum"]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("lo,hi", S.RATIO_BUCKETS[:4])
def test_byte_identical_and_in_bucket(text, lo, hi):
    out = S.sample_bucket(text, "xlmr", lo, hi, random.Random(0))
    if not out["reachable"]:
        pytest.skip("bucket unreachable for this short text")
    S.assert_round_trip(text, out["ids"], "xlmr")
    assert lo <= out["ratio"] < hi
    assert len(out["ids"]) == out["total_k"]


def test_canonical_path_is_in_the_mdd():
    text = "announcement"
    canon = T.get("xlmr")(text, add_special_tokens=False)["input_ids"]
    paths = S.build_mdds(text, "xlmr")[0].enumerate_all()
    assert canon in paths


def test_uniform_over_small_mdd():
    """Chi-square: every segmentation of a short word is equally likely."""
    from scipy.stats import chisquare
    mdd = S.build_mdds("jaheratanum", "xlmr")[0]
    paths = [tuple(p) for p in mdd.enumerate_all()]
    rng = random.Random(1)
    counts = pd.Series([tuple(mdd.sample_unconditional(rng)) for _ in range(20 * len(paths))]).value_counts()
    assert set(counts.index) <= set(paths)
    assert chisquare(counts.reindex(paths, fill_value=0)).pvalue > 1e-3


def test_reproduces_committed_samples():
    """Re-drawing committed rows with the same seed gives identical token ids."""
    s = pd.read_parquet(DATA / "scores" / "resegmentation" / "indic_samples_meta.parquet")
    seg = pd.read_parquet(INDIC / "segments.parquet")
    nat = seg[seg.condition == "native"].set_index("segment_id")
    sc = pd.read_parquet(DATA / "scores" / "resegmentation" / "indic_comet22.parquet")
    rows = s[(s.direction == "forward") & s.reachable & (s.bucket != "char_level")].sample(8, random_state=3)
    for r in rows.itertuples():
        lo, hi = (float(x) for x in r.bucket.strip("[)").split(","))
        text = S.normalize_text_for_reseg(str(nat.loc[r.segment_id, "target"]))
        out = S.sample_bucket(text, "xlmr", lo, hi, random.Random(int(r.seed)))
        assert out["total_k"] == r.total_k
        hit = sc[(sc.segment_id == r.segment_id) & (sc.bucket == r.bucket) & (sc.seed == r.seed)]
        assert abs(hit.ratio.iloc[0] - out["ratio"]) < 1e-12


def test_nbsp_is_normalised():
    assert S.normalize_text_for_reseg("a b") == "a b"
