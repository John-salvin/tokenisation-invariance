"""SI-Check on the committed Gujarati scores reproduces the paper's row."""
import pandas as pd
import pytest

from sicheck.cli import ti_report
from tokinv.paths import DATA, INDIC, LADDER8


def test_gujarati_verdict_matches_table_1():
    e = pd.read_parquet(DATA / "scores" / "resegmentation" / "indic_comet22.parquet")
    e = e[(e.lang == "guj") & (e.direction == "forward") & e.bucket.isin(LADDER8)]
    seg = pd.read_parquet(INDIC / "segments.parquet")
    n = seg[(seg.condition == "native") & seg.segment_id.isin(e.segment_id)]
    can = pd.DataFrame({"unit": n.segment_id, "bucket": "canonical", "ratio": 1.0,
                        "score": n.comet_22 / 100, "human": n.mqm_score})
    sc = pd.concat([can, e.rename(columns={"segment_id": "unit", "comet22": "score",
                                           "mqm_score": "human"})[can.columns]])
    r = ti_report(sc, n_resamples=300, seed=0)
    assert r["monotonicity_rho"] == pytest.approx(-1.0)
    assert r["ci95"][0] == pytest.approx(-1.0) and r["ci95"][1] == pytest.approx(-0.833, abs=5e-4)
    assert r["verdict"].startswith("VIOLATED")
    assert r["buckets"][0]["rho"] == pytest.approx(0.452, abs=5e-4)
    assert r["buckets"][-1]["rho"] == pytest.approx(0.038, abs=5e-4)


def test_sample_rows_index_the_input_after_subsampling():
    """`score` looks up src and ref by the sample's `row`, so `row` must point
    into the input as read, also when --max-rows subsamples it."""
    from sicheck.cli import sample
    from tokinv import sampler, tokenisers
    words = ["translation", "evaluation", "segmentation", "tokeniser", "metric", "agreement"]
    df = pd.DataFrame({"mt": [f"a {w} example" for w in words], "human": range(len(words))})
    s, info = sample(df, "xlmr", seeds=1, max_rows=3, seed=0)
    assert info["rows_used"] == 3
    tok = tokenisers.get("xlmr")
    for r in s.itertuples():
        assert tok.decode(r.mt_ids) == sampler.normalize_text_for_reseg(df.loc[r.row, "mt"])
        assert r.human == df.loc[r.row, "human"]
