"""Off-target systems: systems that mostly wrote a
language variety other than the target, identified from their output text by
GlotLID v3 (experiments/data_prep/offtarget_lid.py). Their human scores record the
variety, not translation quality, so main-text WMT analyses drop them and the
appendix keeps all systems."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .paths import DATA

SYSTEMS = DATA / "derived" / "offtarget" / "offtarget_systems.csv"


@lru_cache(maxsize=None)
def _table() -> pd.DataFrame:
    return pd.read_csv(SYSTEMS)


def off_target(setting: str) -> list[str]:
    """Off-target systems of a WMT setting, e.g. 'en-ar_EG' (empty if none)."""
    t = _table()
    return sorted(t.system[(t.setting == setting) & t.off_target])
