"""Score COMET-family metrics from token ids, bypassing the metric's own
tokenisation. This is what lets a non-canonical segmentation reach the model
at all: given text, COMET would always re-tokenise canonically.

Needs ``unbabel-comet`` and ``torch`` (requirements-gpu.txt). A GPU is
recommended but not required.

Two model families, verified against the comet source:
  * RegressionMetric (COMET-22): source, translation and reference are encoded
    separately; ids are wrapped in <s> ... </s> and right-padded per side.
  * UnifiedMetric (CometKiwi-22/23): sides are CONCATENATED by the library's
    own ``encoder.concat_sequences``; reference-free models take one pass on
    (mt, src).
xCOMET-XL/XXL are not supported: their final score clips and weights three
passes and adds a word-level error-span term, which needs subword offsets and
cannot be computed from token ids alone. (The paper scores xCOMET from text,
on canonical input only.)
The ids must not include special tokens; this module adds them.
"""
from __future__ import annotations

from functools import lru_cache

import torch

COMET_MODELS = {
    "comet22": "Unbabel/wmt22-comet-da",
    "cometkiwi22": "Unbabel/wmt22-cometkiwi-da",
    "cometkiwi23": "Unbabel/wmt23-cometkiwi-da-xl",
}
UNSUPPORTED = {"xcomet_xl", "xcomet_xxl", "Unbabel/XCOMET-XL", "Unbabel/XCOMET-XXL"}


@lru_cache(maxsize=None)
def load(name: str, device: str | None = None):
    """Load a checkpoint by short name or Hugging Face id (downloads once)."""
    if name in UNSUPPORTED:
        raise SystemExit(f"{name}: xCOMET cannot be scored from token ids (its score adds a "
                         "word-level error-span term); see the docstring of tokinv.comet_ids.")
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(COMET_MODELS.get(name, name)))
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def is_referenceless(model) -> bool:
    segs = getattr(model.hparams, "input_segments", None)
    if segs is not None:
        return "ref" not in segs
    import inspect
    return "ref_input_ids" not in inspect.signature(model.forward).parameters


def is_unified(model) -> bool:
    import comet.models.multitask.unified_metric as um
    return isinstance(model, um.UnifiedMetric)


def _sides(model, ids_list, device):
    tok = model.encoder.tokenizer
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    seqs = [[bos] + list(ids) + [eos] for ids in ids_list]
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), n), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :len(s)] = 1
    return ids, mask


def _concat(model, parts, device):
    # concat_sequences rebuilds tensors without a device argument, so move back
    out = model.encoder.concat_sequences(parts, return_label_ids=False)[0]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}


def score_batch(model, src_ids, mt_ids, ref_ids=None) -> list[float]:
    """Scores for parallel lists of id sequences (no special tokens)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        if type(model).__name__ == "XCOMETMetric":
            raise ValueError("xCOMET cannot be scored from token ids; see the module docstring")
        if is_unified(model):
            side = lambda b: dict(zip(("input_ids", "attention_mask"), _sides(model, b, device)))
            mt, src = side(mt_ids), side(src_ids)
            if is_referenceless(model) or ref_ids is None:
                if not is_referenceless(model):
                    raise ValueError("this model needs references")
                return model.forward(**_concat(model, [mt, src], device)).score.cpu().tolist()
            ref = side(ref_ids)
            passes = [_concat(model, p, device) for p in ([mt, src], [mt, ref], [mt, src, ref])]
            return torch.stack([model.forward(**c).score for c in passes]).mean(0).cpu().tolist()
        s, sm = _sides(model, src_ids, device)
        m, mm = _sides(model, mt_ids, device)
        if is_referenceless(model):
            return model.forward(s, sm, m, mm).score.cpu().tolist()
        if ref_ids is None:
            raise ValueError("this model needs references")
        r, rm = _sides(model, ref_ids, device)
        return model.forward(s, sm, m, mm, r, rm).score.cpu().tolist()
