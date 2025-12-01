# HW5/models/gpt2_qa.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_NAME = "openai-community/gpt2"
DEFAULT_CKPT_DIR = "HW5/checkpoints/gpt2_qa"

_tokenizer = None
_model = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_tokenizer_and_model(
    ckpt_dir: str = DEFAULT_CKPT_DIR,
    base_model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
):
    """
    Load (and cache) a GPT-2 model and tokenizer.

    If a fine-tuned checkpoint exists in `ckpt_dir`, load from there.
    Otherwise, fall back to the base model `base_model_name`.
    """
    global _tokenizer, _model, _DEVICE

    if device is None:
        device = _DEVICE
    else:
        _DEVICE = device

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model, _DEVICE

    # Check if fine-tuned checkpoint directory exists
    if os.path.isdir(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print(f"[HW5] Loading fine-tuned GPT-2 from {ckpt_dir}")
        _tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        _model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    else:
        print(f"[HW5] Fine-tuned checkpoint not found, loading base model: {base_model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        _model = AutoModelForCausalLM.from_pretrained(base_model_name)

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model.to(device)
    _model.eval()
    return _tokenizer, _model, _DEVICE


@torch.no_grad()
def generate_with_llm(
    prompt: str,
    max_new_tokens: int = 50,
    ckpt_dir: str = DEFAULT_CKPT_DIR,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """
    Generate text from GPT-2 (fine-tuned if available) given a prompt string.
    """
    tokenizer, model, device = _load_tokenizer_and_model(ckpt_dir=ckpt_dir)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text
