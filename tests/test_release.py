"""Every number printed in the paper reproduces from the rebuilt tables."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results" / "tables" / "reseg_all_settings.csv").exists(),
                    reason="run the notebooks or scripts/run_all.py first")
def test_every_paper_number_reproduces():
    r = subprocess.run([sys.executable, "scripts/verify_paper_numbers.py"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:]
