"""Byte-identical non-canonical re-segmentation sampler (XLM-R).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tokenisers as T

WORD_INITIAL_MARKER = "▁"


class NoncanonicalError(RuntimeError):
    pass


class _Trie:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict[str, "_Trie"] = {}
        self.token_id: int | None = None

    def insert(self, piece: str, token_id: int) -> None:
        node = self
        for ch in piece:
            node = node.children.setdefault(ch, _Trie())
        if node.token_id is not None:
            raise NoncanonicalError(
                f"duplicate vocab piece {piece!r} (ids {node.token_id} and {token_id}) "
                f"-- vocabulary assumption violated, refusing to silently pick one")
        node.token_id = token_id

    def matches_at(self, text: str, start: int) -> list[tuple[int, int]]:
        out = []
        node = self
        i = start
        n = len(text)
        while i < n and text[i] in node.children:
            node = node.children[text[i]]
            i += 1
            if node.token_id is not None:
                out.append((i, node.token_id))
        return out


class Tries:
    __slots__ = ("initial", "continuation", "bare_marker_id", "tokeniser_name")

    def __init__(self, initial: _Trie, continuation: _Trie,
                 bare_marker_id: int | None, tokeniser_name: str):
        self.initial = initial
        self.continuation = continuation
        self.bare_marker_id = bare_marker_id
        self.tokeniser_name = tokeniser_name


_TRIE_CACHE: dict[str, Tries] = {}


def build_tries(tokeniser_name: str) -> Tries:
    if tokeniser_name in _TRIE_CACHE:
        return _TRIE_CACHE[tokeniser_name]

    tok = T.load(tokeniser_name)
    vocab = tok.get_vocab()
    special_ids = set(tok.all_special_ids)

    byte_fallback = [p for p in vocab if p.startswith("<0x") and p.endswith(">")]
    if byte_fallback:
        raise NoncanonicalError(
            f"{tokeniser_name}: {len(byte_fallback)} byte-fallback vocab entries found "
            f"(e.g. {byte_fallback[:3]!r}) -- this module does not handle byte-fallback "
            f"segmentation and would silently under-count valid segmentations for any "
            f"word touching an out-of-normal-vocab character. Stopping rather than "
            f"working around it.")

    initial, continuation = _Trie(), _Trie()
    n_initial = n_cont = 0
    for piece, tid in vocab.items():
        if tid in special_ids:
            continue
        if not piece:
            continue
        if piece.startswith(WORD_INITIAL_MARKER):
            stripped = piece[1:]
            if WORD_INITIAL_MARKER in stripped:
                raise NoncanonicalError(
                    f"{tokeniser_name}: piece {piece!r} has an internal marker beyond "
                    f"position 0 -- the 'leading-marker-only' assumption is violated")
            if stripped:
                initial.insert(stripped, tid)
                n_initial += 1
        else:
            if WORD_INITIAL_MARKER in piece:
                raise NoncanonicalError(
                    f"{tokeniser_name}: piece {piece!r} contains the word-initial marker "
                    f"without leading it -- unexpected vocabulary structure")
            continuation.insert(piece, tid)
            n_cont += 1

    if n_initial == 0 or n_cont == 0:
        raise NoncanonicalError(
            f"{tokeniser_name}: trie construction produced {n_initial} initial / "
            f"{n_cont} continuation pieces -- this tokeniser's vocabulary does not "
            f"look like SentencePiece's leading-marker convention; stopping rather "
            f"than silently building an empty/wrong MDD.")

    bare_marker_id = vocab.get(WORD_INITIAL_MARKER)
    if bare_marker_id is not None and bare_marker_id in special_ids:
        bare_marker_id = None

    result = Tries(initial, continuation, bare_marker_id, tokeniser_name)
    _TRIE_CACHE[tokeniser_name] = result
    return result


class WordMDD:

    def __init__(self, word: str, tries: Tries):
        if not word:
            raise NoncanonicalError("empty word passed to WordMDD")
        self.word = word
        self.n = len(word)
        self.tries = tries
        self.edges: list[list[tuple[int, int]]] = [
            tries.continuation.matches_at(word, i) for i in range(self.n)
        ]
        self.start_edges: list[tuple[int, int]] = list(tries.initial.matches_at(word, 0))
        if tries.bare_marker_id is not None:
            self.start_edges.append((0, tries.bare_marker_id))
        self._ways_cont: list[int] | None = None
        self._ways_cont_by_k: list[dict[int, int]] | None = None

    def _compute_ways_cont(self) -> list[int]:
        if self._ways_cont is not None:
            return self._ways_cont
        n = self.n
        ways = [0] * (n + 1)
        ways[n] = 1
        for i in range(n - 1, -1, -1):
            ways[i] = sum(ways[j] for j, _tid in self.edges[i] if ways[j] > 0)
        self._ways_cont = ways
        return ways

    def _compute_ways_cont_by_k(self) -> list[dict[int, int]]:
        if self._ways_cont_by_k is not None:
            return self._ways_cont_by_k
        n = self.n
        tables: list[dict[int, int]] = [dict() for _ in range(n + 1)]
        tables[n][0] = 1
        for i in range(n - 1, -1, -1):
            acc: dict[int, int] = {}
            for j, _tid in self.edges[i]:
                for k, c in tables[j].items():
                    acc[k + 1] = acc.get(k + 1, 0) + c
            tables[i] = acc
        self._ways_cont_by_k = tables
        return tables

    def ways_total(self) -> list[int]:
        ways_cont = self._compute_ways_cont()
        out = list(ways_cont)
        out[0] = sum(ways_cont[j] for j, _tid in self.start_edges if ways_cont[j] > 0)
        return out

    def ways_by_k(self) -> list[dict[int, int]]:
        tables_cont = self._compute_ways_cont_by_k()
        out = list(tables_cont)
        start: dict[int, int] = {}
        for j, _tid in self.start_edges:
            for k, c in tables_cont[j].items():
                start[k + 1] = start.get(k + 1, 0) + c
        out[0] = start
        return out

    def is_segmentable(self) -> bool:
        return self.ways_total()[0] > 0

    def sample_unconditional(self, rng: random.Random) -> list[int]:
        ways_cont = self._compute_ways_cont()
        cands = [(j, tid) for j, tid in self.start_edges if ways_cont[j] > 0]
        if not cands:
            raise NoncanonicalError(f"word {self.word!r} has zero valid segmentations")
        weights = [ways_cont[j] for j, _ in cands]
        j, tid = rng.choices(cands, weights=weights, k=1)[0]
        ids: list[int] = [tid]
        i = j
        while i < self.n:
            cands = [(j2, t2) for j2, t2 in self.edges[i] if ways_cont[j2] > 0]
            weights = [ways_cont[j2] for j2, _ in cands]
            j2, t2 = rng.choices(cands, weights=weights, k=1)[0]
            ids.append(t2)
            i = j2
        return ids

    def sample_exact_k(self, k: int, rng: random.Random) -> list[int]:
        tables_cont = self._compute_ways_cont_by_k()
        cands = [(j, tid) for j, tid in self.start_edges if tables_cont[j].get(k - 1, 0) > 0]
        if not cands:
            raise NoncanonicalError(f"word {self.word!r} has no {k}-token segmentation")
        weights = [tables_cont[j][k - 1] for j, _ in cands]
        j, tid = rng.choices(cands, weights=weights, k=1)[0]
        ids: list[int] = [tid]
        i, remaining = j, k - 1
        while i < self.n:
            cands = [(j2, t2) for j2, t2 in self.edges[i]
                     if tables_cont[j2].get(remaining - 1, 0) > 0]
            weights = [tables_cont[j2][remaining - 1] for j2, _ in cands]
            j2, t2 = rng.choices(cands, weights=weights, k=1)[0]
            ids.append(t2)
            i = j2
            remaining -= 1
        if remaining != 0:
            raise NoncanonicalError(f"internal error: {self.word!r} k={k} left remaining={remaining}")
        return ids

    def enumerate_all(self, limit: int = 200_000) -> list[list[int]]:
        ways = self.ways_total()
        if ways[0] == 0:
            return []
        if ways[0] > limit:
            raise NoncanonicalError(
                f"word {self.word!r} has {ways[0]} segmentations > limit={limit}; "
                f"not meant for exhaustive enumeration, use sampling instead")
        ways_cont = self._compute_ways_cont()
        out: list[list[int]] = []

        def dfs(i: int, path: list[int]) -> None:
            if i == self.n:
                out.append(list(path))
                return
            for j, tid in self.edges[i]:
                if ways_cont[j] > 0:
                    path.append(tid)
                    dfs(j, path)
                    path.pop()

        for j, tid in self.start_edges:
            if ways_cont[j] > 0:
                dfs(j, [tid])
        return out


NBSP = " "


def normalize_text_for_reseg(text: str) -> str:
    return text.replace(NBSP, " ")


def words_with_spans(text: str) -> list[str]:
    return text.split()


def canonical_ids_per_word(text: str, tokeniser_name: str) -> list[list[int]]:
    tok = T.load(tokeniser_name)
    out = []
    for w in words_with_spans(text):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        out.append(ids)
    return out


class CanonicalRoundtripFailure(NoncanonicalError):
    """The tokeniser's OWN canonical segmentation of this word does not survive
    decode() -- a pre-existing property of the (word, tokeniser) pair, not a bug
    in this module. Verified mechanism: XLM-R silently DROPS zero-width
    non-joiners (U+200C) from its output (tok('\\u200c') -> [] , decode -> ''),
    and re-inserts a literal space between pieces that were originally joined by
    one ('a\\u200cb' -> decode 'a b'). Any word containing such a character
    cannot satisfy the design's round-trip requirement by construction, at ANY
    segmentation, canonical or not -- so it must be excluded before attempting
    to build an MDD for it, not forced through one."""


def canonical_roundtrips(word: str, tokeniser_name: str) -> bool:
    tok = T.load(tokeniser_name)
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return tok.decode(ids) == word


def build_mdds(text: str, tokeniser_name: str) -> list[WordMDD]:
    tries = build_tries(tokeniser_name)
    mdds = []
    for w in words_with_spans(text):
        if not canonical_roundtrips(w, tokeniser_name):
            raise CanonicalRoundtripFailure(
                f"{tokeniser_name}: word {w!r} in text {text!r} does not round-trip "
                f"even under its OWN canonical tokenisation (decode(tok(w)) != w) -- "
                f"typically a zero-width joiner/non-joiner the tokeniser drops. "
                f"Exclude this word/segment from re-segmentation, do not force an MDD onto it.")
        mdd = WordMDD(w, tries)
        if not mdd.is_segmentable():
            raise NoncanonicalError(
                f"{tokeniser_name}: word {w!r} in text {text!r} has NO valid segmentation "
                f"under the trie -- likely an <unk>-triggering character with no "
                f"byte-fallback path; this module does not handle that case (fail loudly)")
        mdds.append(mdd)
    return mdds


def sample_text_unconditional(text: str, tokeniser_name: str, rng: random.Random) -> list[int]:
    mdds = build_mdds(text, tokeniser_name)
    ids: list[int] = []
    for mdd in mdds:
        ids.extend(mdd.sample_unconditional(rng))
    return ids


def _convolve(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for k1, c1 in a.items():
        for k2, c2 in b.items():
            out[k1 + k2] = out.get(k1 + k2, 0) + c1 * c2
    return out


def whole_text_length_distribution(mdds: list[WordMDD]) -> dict[int, int]:
    dist: dict[int, int] = {0: 1}
    for mdd in mdds:
        dist = _convolve(dist, mdd.ways_by_k()[0])
    return dist


def sample_text_exact_total(mdds: list[WordMDD], total_k: int, rng: random.Random) -> list[int]:
    m = len(mdds)
    suffix_dist: list[dict[int, int]] = [dict() for _ in range(m + 1)]
    suffix_dist[m] = {0: 1}
    for r in range(m - 1, -1, -1):
        suffix_dist[r] = _convolve(mdds[r].ways_by_k()[0], suffix_dist[r + 1])

    if suffix_dist[0].get(total_k, 0) == 0:
        raise NoncanonicalError(f"total_k={total_k} is not achievable for this text "
                                 f"(achievable totals: {sorted(suffix_dist[0].keys())})")

    ids: list[int] = []
    remaining = total_k
    for r in range(m):
        dist_r = mdds[r].ways_by_k()[0]
        cands, weights = [], []
        for k_r, count_r in dist_r.items():
            rest = remaining - k_r
            w_rest = suffix_dist[r + 1].get(rest, 0)
            if w_rest > 0:
                cands.append(k_r)
                weights.append(count_r * w_rest)
        if not cands:
            raise NoncanonicalError(f"internal error: no achievable split at word {r}, "
                                     f"remaining={remaining}")
        k_r = rng.choices(cands, weights=weights, k=1)[0]
        ids.extend(mdds[r].sample_exact_k(k_r, rng))
        remaining -= k_r
    if remaining != 0:
        raise NoncanonicalError(f"internal error: remaining={remaining} after all words")
    return ids


RATIO_BUCKETS: list[tuple[float, float]] = [
    (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
    (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00),
]
CHAR_LEVEL_BUCKET = "char_level"


def reachable_totals_in_bucket(dist: dict[int, int], canonical_k: int,
                                lo: float, hi: float, inclusive_hi: bool = False) -> list[int]:
    lo_k = canonical_k * lo
    hi_k = canonical_k * hi
    out = []
    for k in sorted(dist):
        if dist[k] <= 0:
            continue
        if inclusive_hi:
            if lo_k <= k <= hi_k:
                out.append(k)
        else:
            if lo_k <= k < hi_k:
                out.append(k)
    return out


def sample_bucket(text: str, tokeniser_name: str, lo: float, hi: float,
                   rng: random.Random, inclusive_hi: bool = False) -> dict:
    mdds = build_mdds(text, tokeniser_name)
    tok = T.load(tokeniser_name)
    canonical_k = len(tok(text, add_special_tokens=False)["input_ids"])
    dist = whole_text_length_distribution(mdds)
    reachable = reachable_totals_in_bucket(dist, canonical_k, lo, hi, inclusive_hi)
    if not reachable:
        return {"reachable": False, "canonical_k": canonical_k, "bucket": (lo, hi)}
    weights = [dist[k] for k in reachable]
    total_k = rng.choices(reachable, weights=weights, k=1)[0]
    ids = sample_text_exact_total(mdds, total_k, rng)
    return {"reachable": True, "canonical_k": canonical_k, "total_k": total_k,
            "ratio": total_k / canonical_k, "ids": ids, "bucket": (lo, hi)}


def sample_char_level(text: str, tokeniser_name: str) -> dict:
    tries = build_tries(tokeniser_name)
    tok = T.load(tokeniser_name)
    canonical_k = len(tok(text, add_special_tokens=False)["input_ids"])
    ids: list[int] = []
    for w in words_with_spans(text):
        for pos, ch in enumerate(w):
            trie = tries.initial if pos == 0 else tries.continuation
            m = trie.matches_at(w, pos)
            hit = [(j, tid) for j, tid in m if j == pos + 1]
            if not hit:
                return {"reachable": False, "canonical_k": canonical_k,
                        "bucket": CHAR_LEVEL_BUCKET,
                        "reason": f"char {ch!r} in word {w!r} has no single-char vocab piece"}
            ids.append(hit[0][1])
    return {"reachable": True, "canonical_k": canonical_k, "total_k": len(ids),
            "ratio": len(ids) / canonical_k, "ids": ids, "bucket": CHAR_LEVEL_BUCKET}


MIN_FLOOR_BUCKET = "min_floor"

REVERSE_RATIO_BUCKETS: list[tuple[float, float]] = [
    (0.75, 1.00), (0.50, 0.75), (0.25, 0.50),
]


def sample_min_floor(text: str, tokeniser_name: str, rng: random.Random) -> dict:
    mdds = build_mdds(text, tokeniser_name)
    tok = T.load(tokeniser_name)
    canonical_k = len(tok(text, add_special_tokens=False)["input_ids"])
    dist = whole_text_length_distribution(mdds)
    if not dist:
        return {"reachable": False, "canonical_k": canonical_k, "bucket": MIN_FLOOR_BUCKET,
                "reason": "empty length distribution"}
    min_k = min(k for k, c in dist.items() if c > 0)
    ids = sample_text_exact_total(mdds, min_k, rng)
    return {"reachable": True, "canonical_k": canonical_k, "total_k": min_k,
            "ratio": min_k / canonical_k, "ids": ids, "bucket": MIN_FLOOR_BUCKET}


def assert_round_trip(text: str, ids: list[int], tokeniser_name: str) -> None:
    tok = T.load(tokeniser_name)
    decoded = tok.decode(ids)
    if decoded != text:
        raise NoncanonicalError(
            f"ROUND-TRIP FAILURE ({tokeniser_name}): decode(ids) != original text.\n"
            f"  original: {text!r}\n  decoded:  {decoded!r}\n  ids: {ids}\n"
            f"This is the hard stop the design requires: no skips.")
