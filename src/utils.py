import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset
from PIL import Image


def levenshtein_distance(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (no external deps)."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def score_vqa_answer(
    pred: Optional[str],
    references: List[str],
    threshold: float = 0.4,
) -> str:
    """Score a VQA prediction against a list of reference answers.

    Returns:
        'correct'    – normalised Levenshtein ≤ threshold against at least one ref
        'wrong'      – answer found but too different from all refs
        'no_answer'  – pred is None or empty
        'parse_error'– pred is not a string
    """
    if pred is None or (isinstance(pred, str) and not pred.strip()):
        return "no_answer"
    if not isinstance(pred, str):
        return "parse_error"
    pred_s = pred.lower().strip()
    best = min(
        levenshtein_distance(pred_s, ref) / max(len(pred_s), len(ref), 1)
        for ref in references
    )
    return "correct" if best <= threshold else "wrong"


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
