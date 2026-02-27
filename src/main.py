import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_pipeline
from datasets_cfg import load_samples, get_score_fn, DATASET_REGISTRY

MODEL_MINISTRAL = "mistralai/Ministral-3-3B-Instruct-2512"
MODEL_QWEN      = "Qwen/Qwen3-VL-2B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="TAM hallucination-detection pipeline")
    
    p.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="receipt",
        help=f"Dataset to run on. Available: {list(DATASET_REGISTRY.keys())}",
    )
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
    
    # Dynamic loader based on CLI arg
    samples = load_samples(args.dataset, max_samples=args.max_samples)
    score_fn = get_score_fn(args.dataset)
    
    print(f"Loaded {len(samples)} samples for dataset '{args.dataset}'.")

    models_to_run = []
    if args.model in ("ministral", "both"):
        models_to_run.append(("ministral", MODEL_MINISTRAL))
    if args.model in ("qwen", "both"):
        models_to_run.append(("qwen", MODEL_QWEN))

    for name, model_id in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running model: {model_id} on dataset: {args.dataset}")
        print(f"{'='*60}\n")

        # Include dataset name in output path
        out_root = Path(args.output_dir) / args.dataset / name
        
        records = run_pipeline(
            samples=samples,
            model_id=model_id,
            output_dir=str(out_root),
            max_new_tokens=args.max_new_tokens,
            save_token_overlays=args.save_token_overlays,
            score_fn=score_fn,
        )

        if args.push_to_hub:
            from utils import push_dataset
            repo_id = f"UlrickBL/tam-{args.dataset}-{name}-hallucination"
            print(f"Pushing {len(records)} records to {repo_id} …")
            push_dataset(records, repo_id)
            print("Done.")


if __name__ == "__main__":
    main()
