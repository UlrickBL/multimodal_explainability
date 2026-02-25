import io
import json
import re
from typing import Any, Dict, Optional, Tuple
from datasets import Dataset
from PIL import Image

def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_prompt(task_prompt: str, schema: Dict[str, Any]) -> str:
    keys_block = json.dumps({k: None for k in schema.keys()}, indent=2)
    return (
        f"{task_prompt}\n\n"
        f"Return a JSON object with EXACTLY these keys:\n{keys_block}\n\n"
        "Rules:\n"
        "- If a value is not present in the image, use null\n"
        "- Do NOT guess or hallucinate values\n"
        "- Output JSON only, no extra text"
    )

def parse_json(text: str) -> Tuple[Optional[Dict], bool]:
    if isinstance(text, list):
        text = text[0]
    text = text.replace("assistant\n", "").replace("user\n", "").strip()

    codeblock = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if codeblock:
        try:
            return json.loads(codeblock.group(1)), True
        except Exception:
            pass

    start = text.find("{")
    if start == -1:
        return None, False

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate), True
                except Exception:
                    pass

    return None, False

def score_fields(
    pred: Optional[Dict],
    gt: Dict,
    parse_ok: bool = True,
) -> Dict[str, str]:
    scores: Dict[str, str] = {}
    for k, gt_val in gt.items():
        if not parse_ok or pred is None:
            scores[k] = "parse_error"
        elif pred.get(k) is None and gt_val is None:
            scores[k] = "correct_null"
        elif pred.get(k) == gt_val:
            scores[k] = "correct"
        elif pred.get(k) is not None and gt_val is None:
            scores[k] = "hallucination"
        elif pred.get(k) is None and gt_val is not None:
            scores[k] = "miss"
        else:
            scores[k] = "wrong"
    return scores


def push_dataset(records, repo_id: str) -> None:
    ds = Dataset.from_list(records)
    ds.push_to_hub(repo_id)
