"""Lightweight wrapper around the IndicXlit fairseq checkpoints for offline use.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logging.getLogger("fairseq").setLevel(logging.WARNING)

INDICXLIT_SRC = Path.home() / "tokinv_work/staging/repos/_indicxlit_src/IndicXlit-master/app"

_CI_PATH = INDICXLIT_SRC / "ai4bharat/transliteration/transformer/custom_interactive.py"
_spec = importlib.util.spec_from_file_location("indicxlit_custom_interactive", _CI_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Transliterator = _mod.Transliterator

STAGING = Path.home() / "tokinv_work/staging/indicxlit"
LANG_LIST_FILE = STAGING / "lang_list.txt"
ALL_LANGS = [l.strip() for l in LANG_LIST_FILE.read_text().splitlines() if l.strip()]
INDIC_LANGS = [l for l in ALL_LANGS if l != "en"]


class IndicXlitEngine:
    def __init__(self, direction: str, beam_width: int = 4):
        if direction not in ("en2indic", "indic2en"):
            raise ValueError(direction)
        self.direction = direction
        models_path = STAGING / ("en-indic" if direction == "en2indic" else "indic-en")
        lang_pairs_csv = (",".join(f"en-{l}" for l in INDIC_LANGS) if direction == "en2indic"
                          else ",".join(f"{l}-en" for l in INDIC_LANGS))
        self.transliterator = Transliterator(
            data_bin_dir=str(models_path / "corpus-bin"),
            model_checkpoint_path=str(models_path / "transformer" / "indicxlit.pt"),
            lang_pairs_csv=lang_pairs_csv,
            lang_list_file=str(LANG_LIST_FILE),
            beam=beam_width, batch_size=32,
        )

    @staticmethod
    def _pre_process(word: str, lang_code: str) -> str:
        spaced = " ".join(list(word.lower()))
        return f"__{lang_code}__ {spaced}"

    @staticmethod
    def _post_process(translation_str: str, topk: int, n_inputs: int) -> list[list[str]]:
        lines = translation_str.split("\n")
        list_h = [l for l in lines if l.startswith("H-")]
        by_id: dict[int, list[tuple[str, float]]] = {}
        for h in list_h:
            hid = int(h.split("\t")[0].split("-")[1])
            parts = h.split("\t")
            hyp_str = parts[2] if len(parts) > 2 else ""
            score = float(parts[1])
            by_id.setdefault(hid, []).append((hyp_str, score))
        out = []
        for i in range(n_inputs):
            cands = sorted(by_id.get(i, []), key=lambda x: x[1], reverse=True)[:topk]
            out.append(["".join(c[0].split(" ")) for c in cands])
        return out

    def transliterate_words(self, words: list[str], lang_code: str, topk: int = 4) -> list[list[str]]:
        if lang_code not in INDIC_LANGS:
            raise ValueError(f"{lang_code!r} not in supported Indic langs {INDIC_LANGS}")
        preprocessed = [self._pre_process(w, lang_code) for w in words]
        raw = self.transliterator.translate(preprocessed, nbest=topk)
        return self._post_process(raw, topk, len(words))

    def transliterate_words_top1(self, words: list[str], lang_code: str,
                                 max_batch: int = 512) -> list[str]:
        out: list[str] = []
        for start in range(0, len(words), max_batch):
            chunk = words[start:start + max_batch]
            results = self.transliterate_words(chunk, lang_code, topk=1)
            for w, r in zip(chunk, results):
                out.append(r[0] if r else w.lower())
        return out
