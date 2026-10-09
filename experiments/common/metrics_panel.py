"""Metric panel: scoring from text and from token ids.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tokenisers as T


class MetricsPanelError(RuntimeError):
    pass


_MODEL_CACHE: dict[str, object] = {}

_HF_BARE_ID_REDIRECT: dict[str, str] | None = None


def _hf_bare_id_redirect() -> dict[str, str]:
    global _HF_BARE_ID_REDIRECT
    if _HF_BARE_ID_REDIRECT is not None:
        return _HF_BARE_ID_REDIRECT
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import constants as C
    _HF_BARE_ID_REDIRECT = {
        "xlm-roberta-large": str(C.STAGING / "models" / "xlmr_large"),
        "microsoft/infoxlm-large": str(C.STAGING / "models" / "infoxlm_large"),
        "facebook/xlm-roberta-xl": str(C.STAGING / "models" / "xlm_roberta_xl"),
        "facebook/xlm-roberta-xxl": str(C.STAGING / "models" / "xlm_roberta_xxl"),
    }
    return _HF_BARE_ID_REDIRECT


_HF_PATCHED = False


def _patch_hf_offline_redirect() -> None:
    global _HF_PATCHED
    if _HF_PATCHED:
        return
    import transformers

    redirect = _hf_bare_id_redirect()

    def _wrap(orig):
        def wrapped(cls, pretrained_model_name_or_path, *args, **kwargs):
            local = redirect.get(pretrained_model_name_or_path)
            target = local if local is not None else pretrained_model_name_or_path
            return orig(cls, target, *args, **kwargs)
        return wrapped

    transformers.PreTrainedTokenizerBase.from_pretrained = classmethod(
        _wrap(transformers.PreTrainedTokenizerBase.from_pretrained.__func__))
    transformers.PreTrainedModel.from_pretrained = classmethod(
        _wrap(transformers.PreTrainedModel.from_pretrained.__func__))
    transformers.PretrainedConfig.from_pretrained = classmethod(
        _wrap(transformers.PretrainedConfig.from_pretrained.__func__))
    _HF_PATCHED = True

CHECKPOINTS: dict[str, str] = {
    "comet22": "Unbabel/wmt22-comet-da",
    "cometkiwi22": "Unbabel/wmt22-cometkiwi-da",
    "bleurt20": "lucadiliello/BLEURT-20",
    "metricx24": "google/metricx-24-hybrid-large-v2p6",
    "cometkiwi23": "Unbabel/wmt23-cometkiwi-da-xl",
    "xcomet_xl": "Unbabel/XCOMET-XL",
    "xcomet_xxl": "Unbabel/XCOMET-XXL",
}


def _resolve_checkpoint_path(name: str) -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import constants as C

    staged = C.STAGING / "models" / name / "checkpoints" / "model.ckpt"
    if staged.exists():
        return staged

    hf_id = CHECKPOINTS.get(name)
    if hf_id is None:
        raise MetricsPanelError(f"unknown COMET checkpoint name {name!r}")
    org, repo = hf_id.split("/", 1)
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}"
    candidates = sorted(hub_dir.glob("snapshots/*/checkpoints/model.ckpt"))
    if candidates:
        return candidates[0]

    raise MetricsPanelError(
        f"no checkpoint found for {name!r} ({hf_id}) at either {staged} or under {hub_dir} "
        f"-- not staged, fail loudly rather than falling back to a network download "
        f"that cannot succeed offline")


def load_comet(name: str, device: str | None = None):
    if name in _MODEL_CACHE:
        model = _MODEL_CACHE[name]
        if device is not None:
            model = model.to(device)
        return model
    _patch_hf_offline_redirect()
    import comet
    path = _resolve_checkpoint_path(name)
    model = comet.load_from_checkpoint(str(path))
    model.eval()
    if device is not None:
        model = model.to(device)
    _MODEL_CACHE[name] = model
    return model


def verify_tokenizer_matches_panel(model, panel_name: str = "xlmr") -> dict:
    comet_tok = model.encoder.tokenizer
    panel_tok = T.load(panel_name)
    test_text = "hello world, राष्ट्रीय ตัวอย่าง"
    comet_ids = comet_tok(test_text, add_special_tokens=False)["input_ids"]
    panel_ids = panel_tok(test_text, add_special_tokens=False)["input_ids"]
    return {
        "comet_bos": comet_tok.bos_token_id, "comet_eos": comet_tok.eos_token_id,
        "panel_bos": panel_tok.bos_token_id, "panel_eos": panel_tok.eos_token_id,
        "bos_match": comet_tok.bos_token_id == panel_tok.bos_token_id,
        "eos_match": comet_tok.eos_token_id == panel_tok.eos_token_id,
        "vocab_size_comet": comet_tok.vocab_size, "vocab_size_panel": panel_tok.vocab_size,
        "ids_match": list(comet_ids) == list(panel_ids),
        "comet_ids": comet_ids, "panel_ids": panel_ids,
    }


def _is_referenceless(model) -> bool:
    input_segments = getattr(model.hparams, "input_segments", None)
    if input_segments is not None:
        return "ref" not in input_segments
    import inspect
    return "ref_input_ids" not in inspect.signature(model.forward).parameters


def _is_unified_metric(model) -> bool:
    import comet.models.multitask.unified_metric as _um
    return isinstance(model, _um.UnifiedMetric)


def score_from_text(model, src: str, mt: str, ref: str | None = None) -> float:
    sample = {"src": src, "mt": mt}
    if not _is_referenceless(model):
        if ref is None:
            raise MetricsPanelError("this model needs a reference but none was given")
        sample["ref"] = ref
    out = model.predict([sample], batch_size=1, gpus=0, progress_bar=False)
    return float(out.scores[0])


def _side_tensors(model, ids: list[int], device) -> tuple[torch.Tensor, torch.Tensor]:
    tok = model.encoder.tokenizer
    bos, eos = tok.bos_token_id, tok.eos_token_id
    if bos is None or eos is None:
        raise MetricsPanelError(f"tokenizer has no bos/eos id (bos={bos}, eos={eos})")
    full = [bos] + list(ids) + [eos]
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _unified_side_batch(model, ids_batch: list[list[int]], device) -> dict[str, torch.Tensor]:
    input_ids, attention_mask = _side_tensors_batch(model, ids_batch, device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _concat_to_device(concat: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in concat.items()}


def _score_unified_batch(model, src_ids_batch: list[list[int]], mt_ids_batch: list[list[int]],
                          ref_ids_batch: list[list[int]] | None, device) -> list[float]:
    mt_in = _unified_side_batch(model, mt_ids_batch, device)
    src_in = _unified_side_batch(model, src_ids_batch, device)
    referenceless = _is_referenceless(model)
    with torch.no_grad():
        if referenceless or ref_ids_batch is None:
            if not referenceless and ref_ids_batch is None:
                raise MetricsPanelError("this model uses references but none were given")
            concat = model.encoder.concat_sequences([mt_in, src_in], return_label_ids=False)[0]
            concat = _concat_to_device(concat, device)
            out = model.forward(**concat)
            return [float(x) for x in out.score.detach().cpu().tolist()]
        ref_in = _unified_side_batch(model, ref_ids_batch, device)
        c_src = _concat_to_device(
            model.encoder.concat_sequences([mt_in, src_in], return_label_ids=False)[0], device)
        c_ref = _concat_to_device(
            model.encoder.concat_sequences([mt_in, ref_in], return_label_ids=False)[0], device)
        c_full = _concat_to_device(
            model.encoder.concat_sequences([mt_in, src_in, ref_in], return_label_ids=False)[0], device)
        scores = torch.stack([model.forward(**c).score for c in (c_src, c_ref, c_full)], dim=0)
        return [float(x) for x in scores.mean(dim=0).detach().cpu().tolist()]


def score_from_ids(model, src_ids: list[int], mt_ids: list[int],
                    ref_ids: list[int] | None = None) -> float:
    device = next(model.parameters()).device
    if _is_unified_metric(model):
        ref_batch = [ref_ids] if ref_ids is not None else None
        return _score_unified_batch(model, [src_ids], [mt_ids], ref_batch, device)[0]
    src_input_ids, src_mask = _side_tensors(model, src_ids, device)
    mt_input_ids, mt_mask = _side_tensors(model, mt_ids, device)
    with torch.no_grad():
        if _is_referenceless(model):
            out = model.forward(src_input_ids, src_mask, mt_input_ids, mt_mask)
        else:
            if ref_ids is None:
                raise MetricsPanelError("this model needs a reference but none was given")
            ref_input_ids, ref_mask = _side_tensors(model, ref_ids, device)
            out = model.forward(src_input_ids, src_mask, mt_input_ids, mt_mask,
                                ref_input_ids, ref_mask)
    return float(out.score.item())


def _side_tensors_batch(model, ids_list: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    tok = model.encoder.tokenizer
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    if bos is None or eos is None or pad is None:
        raise MetricsPanelError(f"tokenizer missing bos/eos/pad id (bos={bos}, eos={eos}, pad={pad})")
    sequences = [[bos] + list(ids) + [eos] for ids in ids_list]
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(sequences):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attention_mask[i, :len(s)] = 1
    return input_ids, attention_mask


def score_from_ids_batch(model, src_ids_batch: list[list[int]], mt_ids_batch: list[list[int]],
                          ref_ids_batch: list[list[int]] | None = None) -> list[float]:
    device = next(model.parameters()).device
    if _is_unified_metric(model):
        return _score_unified_batch(model, src_ids_batch, mt_ids_batch, ref_ids_batch, device)
    src_input_ids, src_mask = _side_tensors_batch(model, src_ids_batch, device)
    mt_input_ids, mt_mask = _side_tensors_batch(model, mt_ids_batch, device)
    with torch.no_grad():
        if _is_referenceless(model):
            out = model.forward(src_input_ids, src_mask, mt_input_ids, mt_mask)
        else:
            if ref_ids_batch is None:
                raise MetricsPanelError("this model needs references but none were given")
            ref_input_ids, ref_mask = _side_tensors_batch(model, ref_ids_batch, device)
            out = model.forward(src_input_ids, src_mask, mt_input_ids, mt_mask,
                                ref_input_ids, ref_mask)
    return [float(x) for x in out.score.detach().cpu().tolist()]


import torch.nn as _nn

METRICX_SCORE_TOKEN_ID = 250089
METRICX_SCORE_MIN, METRICX_SCORE_MAX = 0.0, 25.0


class MT5ForRegression(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        from transformers.models.mt5.modeling_mt5 import MT5Stack
        import copy as _copy
        self.config = config
        self.model_dim = config.d_model
        self.shared = _nn.Embedding(config.vocab_size, config.d_model)

        enc_cfg = _copy.deepcopy(config)
        enc_cfg.is_decoder = False
        enc_cfg.use_cache = False
        enc_cfg.is_encoder_decoder = False
        self.encoder = MT5Stack(enc_cfg, self.shared)

        dec_cfg = _copy.deepcopy(config)
        dec_cfg.is_decoder = True
        dec_cfg.is_encoder_decoder = False
        dec_cfg.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(dec_cfg, self.shared)

        self.lm_head = _nn.Linear(config.d_model, config.vocab_size, bias=False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        device = input_ids.device
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs[0]
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        predictions = lm_logits[:, 0, METRICX_SCORE_TOKEN_ID]
        predictions = torch.clamp(predictions, METRICX_SCORE_MIN, METRICX_SCORE_MAX)
        return predictions


_METRICX_TOKENIZER = None


def load_metricx24():
    global _METRICX_TOKENIZER
    if "metricx24" in _MODEL_CACHE:
        return _MODEL_CACHE["metricx24"], _METRICX_TOKENIZER

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import constants as C
    from transformers import AutoConfig, AutoTokenizer

    ckpt_dir = C.STAGING / "models" / "metricx24"
    if not (ckpt_dir / "pytorch_model.bin").exists():
        raise MetricsPanelError(f"metricx24 checkpoint not staged at {ckpt_dir}")

    config = AutoConfig.from_pretrained(str(ckpt_dir))
    model = MT5ForRegression(config)
    state_dict = torch.load(str(ckpt_dir / "pytorch_model.bin"), map_location="cpu",
                            weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.eval()

    tok_dir = C.STAGING / "models" / "mt5"
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

    _MODEL_CACHE["metricx24"] = model
    _METRICX_TOKENIZER = tokenizer
    return model, tokenizer


def _metricx_input_text(source: str, hypothesis: str, reference: str) -> str:
    return f"source: {source} candidate: {hypothesis} reference: {reference}"


def metricx_score_from_text(source: str, hypothesis: str, reference: str) -> float:
    model, tokenizer = load_metricx24()
    text = _metricx_input_text(source, hypothesis, reference)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"][:, :-1]
    attention_mask = enc["attention_mask"][:, :-1]
    pred = model.forward(input_ids, attention_mask)
    return float(pred.item())


def metricx_score_from_ids(source_ids: list[int], hypothesis_ids: list[int],
                            reference_ids: list[int]) -> float:
    model, tokenizer = load_metricx24()
    marker1 = tokenizer("source: ", add_special_tokens=False)["input_ids"]
    marker2 = tokenizer(" candidate: ", add_special_tokens=False)["input_ids"]
    marker3 = tokenizer(" reference: ", add_special_tokens=False)["input_ids"]
    full_ids = (marker1 + list(source_ids) + marker2 + list(hypothesis_ids)
                + marker3 + list(reference_ids))
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pred = model.forward(input_ids, attention_mask)
    return float(pred.item())


_BLEURT_MODEL = None
_BLEURT_TOKENIZER = None


def load_bleurt20():
    global _BLEURT_MODEL, _BLEURT_TOKENIZER
    if _BLEURT_MODEL is not None:
        return _BLEURT_MODEL, _BLEURT_TOKENIZER

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import constants as C
    repo_dir = C.STAGING / "repos" / "bleurt_pytorch"
    if not repo_dir.exists():
        raise MetricsPanelError(f"bleurt_pytorch package not staged at {repo_dir}")
    sys.path.insert(0, str(repo_dir.parent))
    from bleurt_pytorch.bleurt.configuration_bleurt import BleurtConfig
    from bleurt_pytorch.bleurt.modeling_bleurt import BleurtForSequenceClassification
    from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer

    ckpt_dir = C.STAGING / "models" / "bleurt20"
    config = BleurtConfig.from_pretrained(str(ckpt_dir))
    model = BleurtForSequenceClassification.from_pretrained(str(ckpt_dir), config=config)
    model.eval()
    tokenizer = BleurtSPTokenizer.from_pretrained(str(ckpt_dir))

    _BLEURT_MODEL, _BLEURT_TOKENIZER = model, tokenizer
    return model, tokenizer


def bleurt_score_from_text(reference: str, candidate: str) -> float:
    model, tokenizer = load_bleurt20()
    with torch.no_grad():
        inputs = tokenizer(reference, candidate, return_tensors="pt")
        logit = model(**inputs).logits.flatten()[0]
    return float(logit.item())


def bleurt_score_from_ids(reference_ids: list[int], candidate_ids: list[int]) -> float:
    model, tokenizer = load_bleurt20()
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    ref_side = [cls] + list(reference_ids) + [sep]
    cand_side = list(candidate_ids) + [sep]
    input_ids = torch.tensor([ref_side + cand_side], dtype=torch.long)
    token_type_ids = torch.tensor([[0] * len(ref_side) + [1] * len(cand_side)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logit = model(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits.flatten()[0]
    return float(logit.item())
