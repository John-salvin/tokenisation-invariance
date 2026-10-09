"""Byte-identical non-canonical re-segmentation sampler (mT5).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noncanonical as NC
import tokenisers as T

WORD_INITIAL_MARKER = NC.WORD_INITIAL_MARKER
MT5_DECODE_KWARGS = {"clean_up_tokenization_spaces": False}

_TRIE_CACHE_MT5: dict[str, NC.Tries] = {}


def build_tries_mt5(tokeniser_name: str = "mt5") -> NC.Tries:
    if tokeniser_name in _TRIE_CACHE_MT5:
        return _TRIE_CACHE_MT5[tokeniser_name]

    tok = T.load(tokeniser_name)
    vocab = tok.get_vocab()
    special_ids = set(tok.all_special_ids)

    byte_fallback_ids = {tid for piece, tid in vocab.items()
                          if piece.startswith("<0x") and piece.endswith(">")}
    if not byte_fallback_ids:
        raise NC.NoncanonicalError(
            f"{tokeniser_name}: expected byte-fallback entries to exclude (mT5 has 256), "
            f"found 0 -- this tokenizer's vocabulary does not match the verified mT5 "
            f"structure this module was built for; do not reuse it for another tokeniser "
            f"without re-verifying byte-fallback usage on the actual corpus first.")

    initial, continuation = NC._Trie(), NC._Trie()
    n_initial = n_cont = n_excluded_bf = 0
    for piece, tid in vocab.items():
        if tid in special_ids or tid in byte_fallback_ids:
            n_excluded_bf += 1 if tid in byte_fallback_ids else 0
            continue
        if not piece:
            continue
        if piece.startswith(WORD_INITIAL_MARKER):
            stripped = piece[1:]
            if WORD_INITIAL_MARKER in stripped:
                raise NC.NoncanonicalError(
                    f"{tokeniser_name}: piece {piece!r} has an internal marker beyond "
                    f"position 0 -- the 'leading-marker-only' assumption is violated")
            if stripped:
                initial.insert(stripped, tid)
                n_initial += 1
        else:
            if WORD_INITIAL_MARKER in piece:
                raise NC.NoncanonicalError(
                    f"{tokeniser_name}: piece {piece!r} contains the word-initial marker "
                    f"without leading it -- unexpected vocabulary structure")
            continuation.insert(piece, tid)
            n_cont += 1

    if n_initial == 0 or n_cont == 0:
        raise NC.NoncanonicalError(
            f"{tokeniser_name}: trie construction produced {n_initial} initial / "
            f"{n_cont} continuation pieces -- vocabulary does not look like SentencePiece's "
            f"leading-marker convention; stopping rather than building an empty/wrong MDD.")

    bare_marker_id = vocab.get(WORD_INITIAL_MARKER)
    if bare_marker_id is not None and (bare_marker_id in special_ids or bare_marker_id in byte_fallback_ids):
        bare_marker_id = None

    result = NC.Tries(initial, continuation, bare_marker_id, tokeniser_name)
    _TRIE_CACHE_MT5[tokeniser_name] = result
    print(f"build_tries_mt5: {n_initial} initial + {n_cont} continuation pieces, "
          f"{n_excluded_bf} byte-fallback pieces excluded (verified unreachable via "
          f"the fast tokenizer this project actually scores with)")
    return result


def canonical_roundtrips_mt5(word: str, tokeniser_name: str = "mt5") -> bool:
    tok = T.load(tokeniser_name)
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return tok.decode(ids, **MT5_DECODE_KWARGS) == word


def assert_round_trip_mt5(text: str, ids: list[int], tokeniser_name: str = "mt5") -> None:
    tok = T.load(tokeniser_name)
    decoded = tok.decode(ids, **MT5_DECODE_KWARGS)
    if decoded != text:
        raise NC.NoncanonicalError(
            f"ROUND-TRIP FAILURE ({tokeniser_name}): decode(ids) != original text.\n"
            f"  original: {text!r}\n  decoded:  {decoded!r}\n  ids: {ids}\n"
            f"This is the hard stop the design requires: no skips.")


def sample_bucket_mt5(text: str, lo: float, hi: float, rng, tokeniser_name: str = "mt5",
                       inclusive_hi: bool = False) -> dict:
    mdds = build_mdds_mt5(text, tokeniser_name)
    tok = T.load(tokeniser_name)
    canonical_k = len(tok(text, add_special_tokens=False)["input_ids"])
    dist = NC.whole_text_length_distribution(mdds)
    reachable = NC.reachable_totals_in_bucket(dist, canonical_k, lo, hi, inclusive_hi)
    if not reachable:
        return {"reachable": False, "canonical_k": canonical_k, "bucket": (lo, hi)}
    weights = [dist[k] for k in reachable]
    total_k = rng.choices(reachable, weights=weights, k=1)[0]
    ids = NC.sample_text_exact_total(mdds, total_k, rng)
    return {"reachable": True, "canonical_k": canonical_k, "total_k": total_k,
            "ratio": total_k / canonical_k, "ids": ids, "bucket": (lo, hi)}


def build_mdds_mt5(text: str, tokeniser_name: str = "mt5") -> list[NC.WordMDD]:
    tries = build_tries_mt5(tokeniser_name)
    mdds = []
    for w in NC.words_with_spans(text):
        if not canonical_roundtrips_mt5(w, tokeniser_name):
            raise NC.CanonicalRoundtripFailure(
                f"{tokeniser_name}: word {w!r} in text {text!r} does not round-trip "
                f"even under its OWN canonical tokenisation -- typically mT5's own "
                f"internal Unicode normalisation (e.g. superscript digits) or an "
                f"embedded control character. Exclude this word/segment from re-segmentation, "
                f"do not force an MDD onto it.")
        mdd = NC.WordMDD(w, tries)
        if not mdd.is_segmentable():
            raise NC.NoncanonicalError(
                f"{tokeniser_name}: word {w!r} in text {text!r} has NO valid segmentation "
                f"under the trie -- fail loudly, do not silently skip.")
        mdds.append(mdd)
    return mdds
