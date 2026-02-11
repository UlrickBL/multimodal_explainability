import os
import json
import torch
import numpy as np
from typing import List, Dict, Any

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    FineGrainedFP8Config
)

from datasets import Dataset

def load_vl_model(model_id):
    if "Qwen3-VL" in model_id:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)

        return {
            "model": model,
            "processor": processor,
            "backend": "qwen"
        }

    elif "Ministral-3" in model_id:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )
        tokenizer = MistralCommonBackend.from_pretrained(model_id)

        return {
            "model": model,
            "processor": tokenizer,
            "backend": "ministral"
        }

    else:
        raise ValueError(f"Unsupported model: {model_id}")



@torch.no_grad()
def run_vl_batch(
    batch,
    model,
    processor,
    backend,
    max_new_tokens=256,
):
    if backend == "qwen":
        messages = []
        for s in batch:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": s["image"]},
                        {"type": "text", "text": s["prompt"]},
                    ],
                }
            ])

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        vision_info = inputs["image_grid_thw"]

        return outputs, inputs, vision_info

    elif backend == "ministral":
        messages = []
        for s in batch:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": s["prompt"]},
                        {
                            "type": "image_url",
                            "image_url": {"url": s["image"]},
                        },
                    ],
                }
            )

        tokenized = processor.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )

        tokenized["input_ids"] = tokenized["input_ids"].to(model.device)
        tokenized["pixel_values"] = tokenized["pixel_values"].to(
            dtype=torch.bfloat16, device=model.device
        )

        image_sizes = [
            tokenized["pixel_values"][i].shape[-2:]
            for i in range(len(batch))
        ]

        outputs = model.generate(
            **tokenized,
            image_sizes=image_sizes,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Vision grid ≈ patch grid
        vision_info = [
            list(tokenized["pixel_values"][i].shape[-2:])
            for i in range(len(batch))
        ]

        return outputs, tokenized, vision_info