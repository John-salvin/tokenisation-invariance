"""Layerwise retrieval probe helpers.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as C


class LayerwiseError(RuntimeError):
    pass


_MODEL_CACHE = {}


def load_xlmr_encoder(device: str = "cuda"):
    if "xlmr_large_encoder" in _MODEL_CACHE:
        return _MODEL_CACHE["xlmr_large_encoder"]
    from transformers import AutoModel, AutoTokenizer
    path = C.STAGING / "models" / "xlmr_large"
    if not path.exists():
        raise LayerwiseError(f"xlmr_large not staged at {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModel.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    _MODEL_CACHE["xlmr_large_encoder"] = (model, tokenizer)
    return model, tokenizer


@torch.no_grad()
def embed_batch(ids_list: list[list[int]], model, tokenizer, device: str,
                batch_size: int = 16, max_length: int = 256) -> np.ndarray:
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    all_out = []
    for start in range(0, len(ids_list), batch_size):
        batch = ids_list[start:start + batch_size]
        framed = [[bos] + list(ids)[:max_length - 2] + [eos] for ids in batch]
        max_len = max(len(f) for f in framed)
        input_ids = torch.full((len(framed), max_len), pad, dtype=torch.long)
        attn = torch.zeros((len(framed), max_len), dtype=torch.long)
        for i, f in enumerate(framed):
            input_ids[i, :len(f)] = torch.tensor(f, dtype=torch.long)
            attn[i, :len(f)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        mask = attn.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        layer_means = []
        for h in out.hidden_states:
            pooled = (h.float() * mask).sum(dim=1) / denom
            layer_means.append(pooled.cpu().numpy())
        all_out.append(np.stack(layer_means, axis=1))
    return np.concatenate(all_out, axis=0)


def retrieval_acc1(en_embeds: np.ndarray, tgt_embeds: np.ndarray) -> float:
    en_n = en_embeds / (np.linalg.norm(en_embeds, axis=1, keepdims=True) + 1e-9)
    tgt_n = tgt_embeds / (np.linalg.norm(tgt_embeds, axis=1, keepdims=True) + 1e-9)
    sims = en_n @ tgt_n.T
    preds = sims.argmax(axis=1)
    return float(np.mean(preds == np.arange(len(preds))))
