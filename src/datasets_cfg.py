"""
Dataset registry for the TAM explainability pipeline.

Each dataset entry is a dict with:
    'hf_id'    : HuggingFace dataset identifier
    'split'    : default split to load
    'loader'   : function(row, idx, prompt) → sample dict
    'prompt'   : prompt string passed to the model
    'score_fn' : function(pred_struct, row) → field_scores dict
                 where field_scores maps field names → status strings

To add a new dataset:
    1. Define a loader and a score_fn below.
    2. Register them in DATASET_REGISTRY.
"""
import json
from typing import Any, Dict, List, Optional

from utils import (
    build_prompt,
    decode_image_bytes,
    parse_json,
    score_fields,
    score_vqa_answer,
)


# ─── Receipt Extraction (nanonets/key_information_extraction) ─────────────

_RECEIPT_SCHEMA = {
    "date": None,
    "doc_no": None,
    "seller_name": None,
    "seller_address": None,
    "seller_gst_id": None,
    "seller_phone": None,
    "total_amount": None,
    "total_tax": None,
}

_RECEIPT_PROMPT = build_prompt(
    "You are a document understanding assistant. "
    "Extract structured information from this receipt image.",
    _RECEIPT_SCHEMA,
)


def _load_receipt_row(row: Dict, idx: int, prompt: str) -> Dict[str, Any]:
    gt = row.get("annotations") or row.get("ground_truth") or {}
    image = row.get("image")
    if isinstance(image, bytes):
        image = decode_image_bytes(image)
    return {
        "id": str(idx),
        "image": image,
        "prompt": prompt,
        "ground_truth": gt,
    }


def _score_receipt(pred_struct: Optional[Dict], row: Dict) -> Dict[str, str]:
    gt = row.get("annotations") or row.get("ground_truth") or {}
    parse_ok = pred_struct is not None
    return score_fields(pred_struct, gt, parse_ok=parse_ok)


# ─── InfoVQA (LIME-DATA/infovqa) ──────────────────────────────────────────

_INFOVQA_PROMPT = (
    "Look at the image carefully and answer the following question.\n\n"
    "Question: {question}\n\n"
    'Return a JSON object with EXACTLY this key:\n{{"answer": null}}\n\n'
    "Rules:\n"
    "- Put your answer as a concise string in the 'answer' field\n"
    "- Output JSON only, no extra text"
)


def _load_infovqa_row(row: Dict, idx: int, _prompt_template: str) -> Dict[str, Any]:
    question = row.get("question", "")
    prompt = _prompt_template.format(question=question)

    image = row.get("image")
    if isinstance(image, bytes):
        image = decode_image_bytes(image)

    # 'answers' is a list of acceptable strings
    answers = row.get("answers", [])
    if isinstance(answers, str):
        # some rows may serialize as a JSON string
        try:
            answers = json.loads(answers)
        except Exception:
            answers = [answers]

    return {
        "id": row.get("questionId", str(idx)),
        "image": image,
        "prompt": prompt,
        "ground_truth": {"answer": answers},   # list of acceptable answers
        "question": question,
    }


def _score_infovqa(pred_struct: Optional[Dict], row: Dict) -> Dict[str, str]:
    references: List[str] = row.get("answers", [])
    if isinstance(references, str):
        try:
            references = json.loads(references)
        except Exception:
            references = [references]

    if not pred_struct:
        return {"answer": "parse_error"}

    pred_answer = pred_struct.get("answer")
    return {"answer": score_vqa_answer(pred_answer, references)}


# ─── Registry ─────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Dict] = {
    "receipt": {
        "hf_id":    "nanonets/key_information_extraction",
        "split":    "test",
        "prompt":   _RECEIPT_PROMPT,
        "loader":   _load_receipt_row,
        "score_fn": _score_receipt,
    },
    "infovqa": {
        "hf_id":    "LIME-DATA/infovqa",
        "split":    "train",
        "prompt":   _INFOVQA_PROMPT,
        "loader":   _load_infovqa_row,
        "score_fn": _score_infovqa,
    },
}


def load_samples(
    dataset_name: str,
    max_samples: Optional[int] = None,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load and format samples for a registered dataset.

    Args:
        dataset_name: Key in DATASET_REGISTRY (e.g. 'receipt', 'infovqa').
        max_samples:  Optional cap on the number of samples.
        split:        Override the default split (e.g. 'validation').

    Returns:
        List of sample dicts ready to be fed to run_pipeline().
    """
    from datasets import load_dataset  # local import to keep module lightweight

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    cfg = DATASET_REGISTRY[dataset_name]
    hf_split = split or cfg["split"]
    raw_ds = load_dataset(cfg["hf_id"], split=hf_split)

    if max_samples is not None:
        raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

    samples = []
    for idx, row in enumerate(raw_ds):
        sample = cfg["loader"](row, idx, cfg["prompt"])
        # Attach the raw row so score_fn can access it later
        sample["_raw_row"] = dict(row)
        sample["_dataset"] = dataset_name
        samples.append(sample)

    return samples


def get_score_fn(dataset_name: str):
    """Return the scoring function for the given dataset."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    return DATASET_REGISTRY[dataset_name]["score_fn"]
