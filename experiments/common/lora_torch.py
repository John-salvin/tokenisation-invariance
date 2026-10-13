"""Minimal LoRA implementation (after Hu et al., 2022).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * delta

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}, in={self.base.in_features}, out={self.base.out_features}"


def inject_lora(module: nn.Module, target_names: tuple[str, ...] = ("query", "key", "value", "dense"),
                 r: int = 8, alpha: int = 16, dropout: float = 0.0, path_filter=None) -> list[str]:
    replaced = []

    def _recurse(mod: nn.Module, prefix: str):
        for name, child in list(mod.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if (isinstance(child, nn.Linear) and name in target_names
                    and (path_filter is None or path_filter(full_name))):
                setattr(mod, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                _recurse(child, full_name)

    _recurse(module, "")
    return replaced


def lora_state_dict(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(module: nn.Module, state: dict, device=None) -> None:
    sd = module.state_dict()
    for k, v in state.items():
        sd[k] = v.to(device) if device is not None else v
    module.load_state_dict(sd, strict=True)


def trainable_parameters(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]
