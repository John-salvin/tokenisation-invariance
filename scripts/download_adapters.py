#!/usr/bin/env python3
"""Download the trained adapters of Section 8 (74 files, about 770 MB) into
artifacts/adapters/. They are only needed to re-evaluate the adapters; the
tables and numbers of the paper do not need them.

    python scripts/download_adapters.py

Folders inside artifacts/adapters/:
  plain_lolo/          A1, one adapter per held-out language
  anchored/            A2, one adapter per held-out language (the paper's adapters)
  anchored_first_run/  A2, first training run (kept for the record)
  grid/                the rank x mu_sd grid of App. Hyperparameter Frontier
  crossscript/         adapters trained with one script held out (App. Zero-Shot Transfer)
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
# Direct download link of tokenisation-invariance-adapters.tar
ARCHIVE_URL = None
EXPECTED_FILES = 74


def main() -> int:
    if ARCHIVE_URL is None:
        print("The adapter download link is not set yet. The tables and numbers of the "
              "paper do not need the adapters; experiments/README.md shows how to train them.")
        return 2
    OUT.mkdir(exist_ok=True)
    tar = OUT / "tokenisation-invariance-adapters.tar"
    if not tar.exists():
        print(f"downloading {ARCHIVE_URL} (about 770 MB) ...")
        urllib.request.urlretrieve(ARCHIVE_URL, tar)
    with tarfile.open(tar) as t:
        t.extractall(OUT, filter="data")
    n = len(list((OUT / "adapters").rglob("*.pt")))
    print(f"{n} of {EXPECTED_FILES} adapters in {OUT / 'adapters'}")
    return 0 if n == EXPECTED_FILES else 1


if __name__ == "__main__":
    sys.exit(main())
