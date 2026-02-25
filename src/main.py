import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from utils import build_prompt, decode_image_bytes, push_dataset
from pipeline import run_pipeline

DATASET_ID = "nanonets/key_information_extraction"
DATASET_SPLIT = "test"

RECEIPT_SCHEMA = {
    "date":           None,   # DD/MM/YYYY
    "doc_no":         None,   # receipt / invoice number
    "seller_name":    None,
    "seller_address": None,
    "seller_gst_id":  None,
    "seller_phone":   None,
    "total_amount":   None,   # numeric string
    "total_tax":      None,
}

TASK_INSTRUCTION = (
    "You are a document understanding assistant. "
    "Extract structured information from this receipt image."
)
PROMPT = build_prompt(TASK_INSTRUCTION, RECEIPT_SCHEMA)

MODEL_MINISTRAL = "mistralai/Ministral-3-3B-Instruct-2512"
MODEL_QWEN      = "Qwen/Qwen2.5-VL-3B-Instruct"   # or Qwen3-VL-2B-Instruct

def load_samples(max_samples=None):
    raw_ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    if max_samples is not None:
        raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

    samples = []
    for idx, row in enumerate(raw_ds):
        gt = row.get("annotations") or row.get("ground_truth") or {}

        image = row.get("image")
        if isinstance(image, bytes):
            image = decode_image_bytes(image)

        samples.append({
            "id":           str(idx),
            "image":        image,
            "prompt":       PROMPT,
            "ground_truth": gt,
        })

    return samples


def parse_args():
    p = argparse.ArgumentParser(description="TAM hallucination-detection pipeline")
    p.add_argument(
        "--model",
        choices=["ministral", "qwen", "both"],
        default="ministral",
        help="Which model(s) to run.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Cap on samples to process.")
    p.add_argument("--output_dir", default="./output", help="Root output directory.")
    p.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max tokens to generate per sample."
    )
    p.add_argument(
        "--save_token_overlays",
        action="store_true",
        help="Save per-token TAM overlay PNGs (disk-intensive).",
    )
    p.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push records to HuggingFace Hub (requires HF_TOKEN env variable).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    samples = load_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples from '{DATASET_ID}'.")

    models_to_run = []
    if args.model in ("ministral", "both"):
        models_to_run.append(("ministral", MODEL_MINISTRAL))
    if args.model in ("qwen", "both"):
        models_to_run.append(("qwen", MODEL_QWEN))

    for name, model_id in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running model: {model_id}")
        print(f"{'='*60}\n")

        out_dir = Path(args.output_dir) / name
        records = run_pipeline(
            samples=samples,
            model_id=model_id,
            output_dir=str(out_dir),
            max_new_tokens=args.max_new_tokens,
            save_token_overlays=args.save_token_overlays,
        )

        if args.push_to_hub:
            repo_id = f"UlrickBL/tam-{name}-hallucination"
            print(f"Pushing {len(records)} records to {repo_id} …")
            push_dataset(records, repo_id)
            print("Done.")


if __name__ == "__main__":
    main()
