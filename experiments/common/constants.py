"""Language order, paths, seeds and workbook column aliases.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

from pathlib import Path

LANG_ORDER: list[str] = ["guj", "tam", "mal", "mar", "hin"]
CONTROL_ORDER: list[str] = ["deu", "spa"]
ALL_LANGS: list[str] = LANG_ORDER + CONTROL_ORDER

CONDITIONS: list[str] = ["native", "romanised"]

SEED: int = 42
E1_SEEDS: list[int] = [0, 1, 2, 3, 4]

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_CACHE = REPO_ROOT / "data" / "cache"
OUT_TABLES = REPO_ROOT / "results" / "tables"
OUT_FIGURES = REPO_ROOT / "results" / "figures"
OUT_LOGS = REPO_ROOT / "results" / "logs"

XLSX_XLMR = DATA_RAW / "indic" / "indic_parity_xlmr.xlsx"
XLSX_MULTI = DATA_RAW / "indic" / "indic_parity_multi_tokenizer.xlsx"
XLSX_LATIN = DATA_RAW / "latin" / "wmt24_ende_enes_metrics.xlsx"

SEGMENTS_PARQUET = DATA_INTERIM / "segments.parquet"

STAGING = Path.home() / "tokinv_work" / "staging"
TOKENISER_DIR = STAGING / "models"
FLORES_DIR = STAGING / "data" / "flores200_dataset"
MORPHSCORE_DIR = STAGING / "data" / "morphscore"

SHEET_PREFIX_TO_LANG: dict[str, str] = {
    "Indic_mt _for_analysis - Gujara": "guj",
    "Indic_mt _for_analysis - Tamil_": "tam",
    "Indic_mt _for_analysis - Malaya": "mal",
    "Indic_mt _for_analysis - Marath": "mar",
    "Indic_mt _for_analysis - Hindi_": "hin",
}

KNOWN_JUNK_COLUMNS: dict[str, str] = {
    "Unnamed: 18": "stray annotation column in the Malayalam sheet: 11 non-null of 1400",
}

NATIVE_ALIASES: dict[str, list[str]] = {
    "target":     ["Translation"],
    "reference":  ["Reference"],
    "source_en":  ["Source"],
    "comet_22":   ["COMET"],
    "bertscore":  ["BertScore"],
    "bleurt":     ["BLEURT"],
    "bleu":       ["BLEU"],
    "chrf":       ["chrF", "Chrf"],
    "ter":        ["TER"],
    "comet_qe":   ["comet_qe"],
    "tp_xlmr":    ["Translation_xlmr_TP"],
    "ip_xlmr_bloom560m": ["Translation_xlmr_IP"],
    "tok_count_xlmr":  ["Translation_xlmr_token_count"],
    "tok_count_mbert": ["Translation_mbert_token_count"],
    "tok_count_byt5":  ["Translation_byt5_token_count"],
    "tok_count_gpt2":  ["Translation_gpt2_token_count"],
}

ROMANISED_ALIASES: dict[str, list[str]] = {
    "target":     ["Translation_Transliteration_romanized"],
    "reference":  ["Reference_Transliteration_romanized"],
    "source_en":  ["Source"],
    "comet_22":   ["COMET_romanized"],
    "bertscore":  ["Bertscore_romanized"],
    "bleurt":     ["Bleurt_romanized"],
    "bleu":       ["BLEU_romanized"],
    "chrf":       ["CHRF_romanized"],
    "ter":        ["TER_romanized"],
    "comet_qe":   ["comet_qe_romanized"],
    "tp_xlmr":    ["Translation_Transliteration_romanized_xlmr_TP"],
    "ip_xlmr_bloom560m": ["Translation_Transliteration_romanized_xlmr_IP"],
    "tok_count_xlmr":  ["Translation_Transliteration_romanized_xlmr_token_count"],
    "tok_count_mbert": ["Translation_Transliteration_romanized_mbert_token_count"],
    "tok_count_byt5":  ["Translation_Transliteration_romanized_byt5_token_count"],
    "tok_count_gpt2":  ["Translation_Transliteration_romanized_gpt2_token_count"],
}

MQM_COLUMN = "Human_scores"
N_SEGMENTS_TOTAL = 7000
N_SEGMENTS_MQM_VALID = 6995
N_PER_LANG = 1400

SYSTEM_COLUMN = "model"
ERROR_TYPE_COLUMNS = [f"Error{i}_Type" for i in range(1, 6)]
ERROR_SEVERITY_COLUMNS = [f"Error{i}_Severity" for i in range(1, 6)]

SEVERITY_LEVELS = ["Very Low", "Low", "Default", "Medium", "High", "Very High"]
