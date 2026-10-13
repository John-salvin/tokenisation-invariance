"""Tokenisers at pinned revisions. Only vocabulary files are downloaded (a few
MB each, cached by huggingface_hub); no model weights are needed for the notebooks."""
from __future__ import annotations

from functools import lru_cache

# repo id -> exact commit on the Hugging Face Hub
PINNED = {
    "xlmr":  ("FacebookAI/xlm-roberta-large", "c23d21b0620b635a76227c604d44e43a9f0ee389"),
    "mt5":   ("google/mt5-base", "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f"),
    "mbert": ("google-bert/bert-base-multilingual-cased", "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"),
    "byt5":  ("google/byt5-base", "92d8c008d55cf7c254915bac165171dfe6c20c44"),
    "gpt2":  ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "labse": ("sentence-transformers/LaBSE", "836121a0533e5664b21c7aacc5d22951f2b8b25b"),
    "rembert": ("google/rembert", "65da5133da36e29dfca67d4f0dd9f7f9db21b563"),
}

NBSP = "\u00a0"


@lru_cache(maxsize=None)
def get(name: str):
    """The pinned tokeniser `name` (one of PINNED)."""
    import warnings
    from transformers import AutoTokenizer, logging
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
    repo, rev = PINNED[name]
    return AutoTokenizer.from_pretrained(repo, revision=rev)


def n_tokens(name: str, text: str) -> int:
    return len(get(name)(text, add_special_tokens=False)["input_ids"])
