"""AutoAIS model loader for Trust-Align plugin metrics."""

from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@lru_cache(maxsize=2)
def get_autoais_model_and_tokenizer(model_name: str):
    """Load and cache the AutoAIS model/tokenizer pair."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def run_nli_autoais(passage: str, claim: str, autoais_model: str) -> int:
    """Run TrueNLI entailment with AutoAIS and return a binary prediction."""
    model, tokenizer = get_autoais_model_and_tokenizer(autoais_model)

    input_text = f"premise: {passage} hypothesis: {claim}"
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids.to(model.device)

    with torch.inference_mode():
        outputs = model.generate(input_ids, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return 1 if result == "1" else 0
