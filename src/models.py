import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None  # older transformers

from qwen_utils import process_vision_info


# ─── PIL → base64 helper (needed for Ministral image encoding) ────────────

def pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── MistralTokenizerAdapter ─────────────────────────────────────────────
#
# TAM() internally calls:
#   processor.tokenizer.tokenize(text, skip_special_tokens=False, ...)
#   processor.batch_decode([[id], ...])
#
# MistralCommonBackend already has batch_decode / decode.
# We proxy `.tokenizer` as a small inner object that provides `.tokenize()`.


class _MistralTokenizerProxy:
    """Inner proxy: makes `processor.tokenizer.tokenize(text)` work."""

    def __init__(self, backend: MistralCommonBackend):
        self._backend = backend

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize *text* → list of token strings (no special tokens added)."""
        ids = self._backend.encode(text)
        return self._backend.batch_decode([[i] for i in ids])

    def batch_decode(self, token_ids, **kwargs) -> List[str]:
        return self._backend.batch_decode(token_ids, **kwargs)


class MistralTokenizerAdapter:
    """
    Thin wrapper around `MistralCommonBackend` that adds the `.tokenizer` proxy
    attribute expected by `tam.py`, while passing everything else through.
    """

    def __init__(self, backend: MistralCommonBackend):
        self._backend = backend
        self.tokenizer = _MistralTokenizerProxy(backend)

    # ── Delegation ──────────────────────────────────────────────────────────

    def apply_chat_template(self, messages, **kwargs):
        return self._backend.apply_chat_template(messages, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self._backend.decode(token_ids, **kwargs)

    def batch_decode(self, sequences, **kwargs):
        return self._backend.batch_decode(sequences, **kwargs)

    def encode(self, text, **kwargs):
        return self._backend.encode(text, **kwargs)

    # Allow attribute pass-through for anything else
    def __getattr__(self, name: str):
        return getattr(self._backend, name)


# ─── Model Loading ────────────────────────────────────────────────────────

def load_vl_model(model_id: str) -> Dict[str, Any]:
    """
    Load a vision-language model and return a bundle dict:
        {"model": ..., "processor": ..., "backend": "qwen" | "ministral"}
    """
    if "Ministral" in model_id or "mistral" in model_id.lower():
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
        )
        raw_backend = MistralCommonBackend.from_pretrained(model_id)
        processor = MistralTokenizerAdapter(raw_backend)
        return {"model": model, "processor": processor, "backend": "ministral"}

    elif "Qwen" in model_id:
        # Try Qwen3-VL first, fall back to Qwen2.5-VL
        cls = Qwen3VLForConditionalGeneration or Qwen2_5_VLForConditionalGeneration
        model = cls.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        return {"model": model, "processor": processor, "backend": "qwen"}

    else:
        raise ValueError(f"Unsupported model_id: {model_id!r}")


# ─── Inference (single sample, no batching for TAM correctness) ──────────

@torch.no_grad()
def run_single_sample(
    sample: Dict[str, Any],
    model,
    processor,
    backend: str,
    max_new_tokens: int = 256,
) -> Tuple[Any, Dict, List[Image.Image], Any, List[torch.Tensor]]:
    """
    Run inference on a single sample and return everything TAM needs.

    Args:
        sample:  dict with keys "image" (PIL.Image) and "prompt" (str).
        model:   loaded VL model.
        processor: loaded processor / tokenizer adapter.
        backend: "qwen" or "ministral".
        max_new_tokens: max tokens to generate.

    Returns:
        (outputs, inputs, vision_images, vision_shape, logits)

        - outputs:       GenerateOutput with .sequences and .hidden_states
        - inputs:        tokenised dict on model.device
        - vision_images: list[PIL.Image] passed to the model
        - vision_shape:  (H_tokens, W_tokens) for the single image
        - logits:        list of Tensor (1, seq_so_far, vocab), one per gen step
    """
    image: Image.Image = sample["image"]
    prompt: str = sample["prompt"]

    if backend == "qwen":
        return _run_qwen(image, prompt, model, processor, max_new_tokens)
    elif backend == "ministral":
        return _run_ministral(image, prompt, model, processor, max_new_tokens)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


# ─── Qwen backend ─────────────────────────────────────────────────────────

def _run_qwen(image, prompt, model, processor, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # vision_shape: (H_tokens, W_tokens) after the 2× merge
    grid = inputs["image_grid_thw"][0]  # (T, H, W) merging ratio
    vision_shape = (int(grid[1]) // 2, int(grid[2]) // 2)

    logits = _extract_logits(model, outputs)
    return outputs, dict(inputs), image_inputs, vision_shape, logits


# ─── Ministral backend ────────────────────────────────────────────────────

def _run_ministral(image, prompt, model, processor, max_new_tokens):
    img_b64 = pil_to_base64(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }
    ]

    tokenized = processor.apply_chat_template(messages, return_tensors="pt", return_dict=True)

    inputs: Dict[str, torch.Tensor] = {}
    for k, v in tokenized.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(
                device=model.device,
                dtype=torch.bfloat16 if v.dtype.is_floating_point else None,
            )
        else:
            inputs[k] = v

    # image_sizes: list of (H_pixels, W_pixels) for each image
    pixel_values = inputs.get("pixel_values")
    image_sizes = [pixel_values.shape[-2:]] if pixel_values is not None else []

    outputs = model.generate(
        **inputs,
        image_sizes=image_sizes,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # vision_shape: number of vision tokens ≈ (H_pixels/patch, W_pixels/patch)
    # Ministral-3 / Pixtral uses a 14x14 patch and merges them 2x2.
    PATCH = 14
    if image_sizes:
        import math
        h_px, w_px = image_sizes[0]
        h_patches = math.ceil(h_px / PATCH)
        w_patches = math.ceil(w_px / PATCH)
        vision_shape = (math.ceil(h_patches / 2), math.ceil(w_patches / 2))
    else:
        vision_shape = (16, 16)  # fallback

    logits = _extract_logits(model, outputs)
    return outputs, inputs, [image], vision_shape, logits


# ─── Logits extraction ────────────────────────────────────────────────────

def _extract_logits(model, outputs) -> List[torch.Tensor]:
    """
    Convert hidden_states from GenerateOutput into a list of logit tensors.

    Each element corresponds to one generation step and has shape
    (1, sequence_len_up_to_that_step, vocab_size).
    """
    logits = []
    for step_hidden in outputs.hidden_states:
        # step_hidden is a tuple of per-layer tensors; take the last layer
        last_hidden = step_hidden[-1]          # (1, seq_len, hidden)
        step_logits = model.lm_head(last_hidden)  # (1, seq_len, vocab)
        logits.append(step_logits)
    return logits