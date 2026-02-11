from pipeline import run_pipeline
from datasets import load_dataset

MODEL_QWEN = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_MINISTRAL = "mistralai/Ministral-3-3B-Instruct-2512"

raw_ds = load_dataset("nanonets/key_information_extraction", split="test")

samples = [
    {
        "id": row["id"],
        "image": row["image"],
        "prompt": row["prompt"],
        "ground_truth": row["ground_truth"],
    }
    for row in raw_ds
]

records_qwen = run_pipeline(
    samples=my_samples,
    model_id=MODEL_QWEN,
    batch_size=2,
)

push_dataset(records_qwen, "UlrickBL/tam-qwen3-vl")

records_ministral = run_pipeline(
    samples=my_samples,
    model_id=MODEL_MINISTRAL,
    batch_size=1,
)

push_dataset(records_ministral, "UlrickBL/tam-ministral-3-vl")
