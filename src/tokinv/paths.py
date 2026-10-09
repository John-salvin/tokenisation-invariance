"""Repository paths and fixed orderings. Nothing here depends on the machine."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
INDIC = DATA / "indic"
WMT = DATA / "wmt"
INTERIM = DATA / "interim"

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"

ARTIFACTS = ROOT / "artifacts"          # adapters fetched by scripts/download_adapters.py

# Display order, fixed everywhere: never sort languages by value.
INDIC_LANGS = ["guj", "tam", "mal", "mar", "hin"]
LANG_NAMES = {
    "guj": "Gujarati", "tam": "Tamil", "mal": "Malayalam",
    "mar": "Marathi", "hin": "Hindi", "deu": "German", "spa": "Spanish",
}

# The fragmentation ladder of Section 4: eight buckets of width 0.25 on [1, 3).
LADDER8 = [f"[{a:.2f},{a + 0.25:.2f})" for a in [1 + 0.25 * i for i in range(8)]]
