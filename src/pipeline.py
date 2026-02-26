from collections import Counter
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from density_tam_reward import compute_additive_map, compute_density_reward
from models import load_vl_model, run_single_sample
from tam import TAM
from utils import parse_json, score_fields

SPECIAL_IDS_QWEN: Dict[str, Any] = {
    "img_id":    [151652, 151653],
    "prompt_id": [151653, [151645, 198, 151644, 77091]],
    "answer_id": [[198, 151644, 77091, 198], -1],
}

def _get_ministral_special_ids(processor, tokens: List[int]) -> Dict[str, Any]:
    try:
        encoded_img = processor.encode("[IMG]")
        if len(encoded_img) == 1:
            img_start_id = encoded_img[0]
            img_end_id   = encoded_img[0]
        else:
            img_start_id = img_end_id = None
    except Exception:
        img_start_id = img_end_id = None

    try:
        inst_start_id = processor.encode("[INST]")[-1]
        inst_end_id   = processor.encode("[/INST]")[0]
    except Exception:
        inst_start_id = inst_end_id = None

    if img_start_id is None or img_start_id not in tokens:
        max_run = 0
        img_tok = 10
        curr_run = 0
        curr_tok = None
        for t in tokens:
            if t == curr_tok:
                curr_run += 1
            else:
                if curr_run > max_run:
                    max_run = curr_run
                    img_tok = curr_tok
                curr_tok = t
                curr_run = 1
        if curr_run > max_run:
            img_tok = curr_tok
        img_start_id = img_end_id = img_tok

    if inst_start_id is None:
        inst_start_id = tokens[0]
        inst_end_id   = -1

    return {
        "img_id":    [img_start_id] if img_start_id == img_end_id or img_end_id is None
                     else [img_start_id, img_end_id],
        "prompt_id": [inst_start_id, inst_end_id if inst_end_id != -1 else [tokens[-1]]],
        "answer_id": [inst_end_id if inst_end_id != -1 else [tokens[-1]], -1],
    }


def run_tam_for_sample(
    tokens: List[int],
    vision_shape: Tuple[int, int],
    logits: List[torch.Tensor],
    vision_images: List[Image.Image],
    processor,
    backend: str,
    sample_dir: Path,
    save_overlays: bool = True,
) -> List[np.ndarray]:
    n_steps = len(logits)
    special_ids = (
        SPECIAL_IDS_QWEN
        if backend == "qwen"
        else _get_ministral_special_ids(processor, tokens)
    )

    img_scores_list: List[np.ndarray] = []
    tam_maps: List[np.ndarray] = []

    for step in range(n_steps):
        save_fn = ""
        if save_overlays and sample_dir:
            save_fn = str(sample_dir / f"token_{step:04d}.png")

        try:
            img_map = TAM(
                tokens=tokens,
                vision_shape=vision_shape,
                logit_list=logits,
                special_ids=special_ids,
                vision_input=vision_images,
                processor=processor,
                save_fn=save_fn,
                target_token=step,
                img_scores_list=img_scores_list,
                eval_only=(not save_overlays),
            )
        except Exception as e:
            print(f"  [TAM] step {step} failed: {e}")
            img_map = np.zeros(vision_shape, dtype=np.uint8)

        if img_map is None:
            img_map = np.zeros(vision_shape, dtype=np.uint8)

        tam_maps.append(np.array(img_map, dtype=np.uint8))

    return tam_maps

def _find_value_token_span(
    key: str,
    value: Any,
    generated_tokens: List[int],
    prompt_len: int,
    processor,
) -> Tuple[int, int]:
    if value is None:
        return -1, -1

    value_str = str(value)
    try:
        val_ids: List[int] = processor.encode(value_str)
    except Exception:
        return -1, -1

    n = len(val_ids)
    gen = generated_tokens[prompt_len:]
    for i in range(len(gen) - n + 1):
        if gen[i: i + n] == val_ids:
            return (prompt_len + i, prompt_len + i + n)

    return -1, -1


def compute_key_tam_maps(
    pred_struct: Optional[Dict],
    generated_tokens: List[int],
    prompt_len: int,
    tam_maps: List[np.ndarray],
    processor,
) -> Dict[str, Dict]:
    if not pred_struct:
        return {}

    results: Dict[str, Dict] = {}

    for key, value in pred_struct.items():
        start, end = _find_value_token_span(
            key, value, generated_tokens, prompt_len, processor
        )

        if start == -1 or end <= start:
            # Value tokens not found — use all maps as fallback
            maps_for_key = tam_maps
        else:
            # generated steps are 0-indexed; step i corresponds to token prompt_len+i
            gen_start = start - prompt_len
            gen_end   = end   - prompt_len
            # Clamp to available maps
            gen_start = max(0, min(gen_start, len(tam_maps) - 1))
            gen_end   = max(gen_start + 1, min(gen_end, len(tam_maps)))
            maps_for_key = tam_maps[gen_start:gen_end]

        additive = compute_additive_map(maps_for_key)
        reward   = compute_density_reward(additive.astype(np.float32))

        results[key] = {
            "additive_map":   additive,
            "density_reward": reward,
            "token_span":     (start, end),
        }

    return results

def save_key_tam_images(
    key_maps: Dict[str, Dict],
    raw_image: Image.Image,
    sample_dir: Path,
) -> Dict[str, str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_bgr = cv2.cvtColor(np.array(raw_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    saved: Dict[str, str] = {}
    for key, data in key_maps.items():
        amap = data["additive_map"]
        if amap is None or amap.size == 0:
            continue
        colored = cv2.applyColorMap(amap, cv2.COLORMAP_JET)

        h, w = raw_bgr.shape[:2]
        colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)

        overlay = (colored.astype(np.float32) * 0.5 + raw_bgr.astype(np.float32) * 0.5).astype(np.uint8)

        safe_key = key.replace("/", "_").replace(" ", "_")
        out_path = sample_dir / f"{safe_key}_tam.png"
        cv2.imwrite(str(out_path), overlay)
        saved[key] = str(out_path)

    return saved

def run_pipeline(
    samples: List[Dict[str, Any]],
    model_id: str,
    output_dir: str = "./output",
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
    save_token_overlays: bool = False,
) -> List[Dict[str, Any]]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    bundle    = load_vl_model(model_id)
    model     = bundle["model"]
    processor = bundle["processor"]
    backend   = bundle["backend"]

    if max_samples is not None:
        samples = samples[:max_samples]

    records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(samples, desc=f"[{model_id}]")):
        sample_id  = sample.get("id", str(idx))
        sample_dir = out_root / str(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        raw_image: Image.Image = sample["image"]
        img_path = sample_dir / "input.jpg"
        raw_image.save(str(img_path))

        try:
            outputs, inputs, vision_images, vision_shape, logits = run_single_sample(
                sample, model, processor, backend, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"[{sample_id}] Inference failed: {e}")
            records.append({"id": sample_id, "error": str(e), "model_id": model_id})
            continue

        seq = outputs.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        generated_text = processor.decode(seq[prompt_len:], skip_special_tokens=True)

        pred_struct, parse_ok = parse_json(generated_text)
        gt = sample.get("ground_truth", {})
        field_scores = score_fields(pred_struct, gt, parse_ok=parse_ok)
        is_hallucinated = parse_ok and any(v == "hallucination" for v in field_scores.values())

        tokens: List[int] = seq.cpu().tolist()

        print(f"[{sample_id}] Running TAM for {len(logits)} generation steps …")
        
        special_ids = (
            SPECIAL_IDS_QWEN
            if backend == "qwen"
            else _get_ministral_special_ids(processor, tokens)
        )
        img_tok = special_ids["img_id"][0]
        img_count = tokens.count(img_tok)
        print(f"DEBUG: img_tok={img_tok}, in tokens={img_count}, total_tokens={len(tokens)}, vision_shape={vision_shape}, expected_img_tokens={vision_shape[0]*vision_shape[1]}")

        tam_maps = run_tam_for_sample(
            tokens=tokens,
            vision_shape=vision_shape,
            logits=logits,
            vision_images=vision_images,
            processor=processor,
            backend=backend,
            sample_dir=(sample_dir if save_token_overlays else Path("")),
            save_overlays=save_token_overlays,
        )

        key_maps = compute_key_tam_maps(
            pred_struct=pred_struct,
            generated_tokens=tokens,
            prompt_len=prompt_len,
            tam_maps=tam_maps,
            processor=processor,
        )

        key_image_paths = save_key_tam_images(key_maps, raw_image, sample_dir)

        key_density_rewards = {k: v["density_reward"] for k, v in key_maps.items()}
        key_token_spans     = {k: v["token_span"]     for k, v in key_maps.items()}

        record = {
            "id":                    sample_id,
            "image_path":            str(img_path),
            "prompt":                sample.get("prompt", ""),
            "ground_truth":          gt,

            # Prediction
            "generated_text":        generated_text,
            "prediction_structured": pred_struct,
            "parse_success":         parse_ok,
            "field_scores":          field_scores,
            "is_hallucinated":       is_hallucinated,

            # TAM
            "key_density_rewards":   key_density_rewards,
            "key_token_spans":       key_token_spans,
            "key_tam_image_paths":   key_image_paths,

            # Meta
            "model_id":              model_id,
            "vision_shape":          list(vision_shape),
            "n_generated_tokens":    len(logits),
        }
        records.append(record)

        records_path = out_root / "records.json"
        with open(records_path, "w") as f:
            json.dump(
                [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                 for r in records],
                f, indent=2, default=str,
            )

    print(f"\nPipeline complete. {len(records)} records saved to {out_root}/records.json")
    return records
