import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from tokinv import corrtests as CT
from tokinv import stats
from tokinv.qn import qn_map


def test_spearman_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    a[3] = a[4]  # a tie
    assert stats.spearman(a, b) == pytest.approx(sps.spearmanr(a, b)[0], abs=1e-12)


def test_exact_p_perfect_monotone_n8():
    # one of 8! orderings is perfectly decreasing, one perfectly increasing
    assert stats.exact_p(-1.0, 8) == pytest.approx(2 / 40320)


def test_exact_p_is_two_sided_and_bounded():
    for rho in [-0.9, -0.3, 0.0, 0.5]:
        p = stats.exact_p(rho, 6)
        assert 0 < p <= 1
        assert p == pytest.approx(stats.exact_p(-rho, 6))


def test_mrr_null_when_correlations_equal():
    t = CT.meng_rosenthal_rubin(0.5, 0.5, 0.8, 500)
    assert t.statistic == pytest.approx(0.0) and t.p_value == pytest.approx(1.0)


def test_mrr_type_one_error_is_nominal():
    """Under the null the test rejects about 5 percent of the time."""
    rng = np.random.default_rng(1)
    rej = 0
    for _ in range(400):
        h = rng.normal(size=200)
        j = h + rng.normal(size=200)
        k = h + rng.normal(size=200)
        r = lambda x, y: sps.spearmanr(x, y)[0]
        rej += CT.meng_rosenthal_rubin(r(j, h), r(k, h), r(j, k), 200).p_value < 0.05
    assert 0.02 < rej / 400 < 0.09


def test_qn_preserves_within_cell_ranking():
    rng = np.random.default_rng(2)
    x, ref = rng.normal(size=300), np.sort(rng.gamma(2, size=1000))
    q = qn_map(x, ref)
    assert stats.spearman(x, q) == pytest.approx(1.0)


def test_segment_bootstrap_resamples_whole_units():
    df = pd.DataFrame({"unit": np.repeat(np.arange(30), 4), "v": np.arange(120.0)})
    seen = []
    stats.segment_bootstrap(df, lambda d: seen.append(d.groupby("unit").size().unique()) or 0.0,
                            "unit", n_resamples=5)
    assert all(set(s) <= {4, 8, 12, 16, 20} for s in seen)  # whole units, possibly repeated
