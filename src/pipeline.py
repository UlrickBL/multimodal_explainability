def run_tam_pipeline(
    samples,
    model_id,
    batch_size=2,
):
    bundle = load_vl_model(model_id)
    model = bundle["model"]
    processor = bundle["processor"]
    backend = bundle["backend"]

    records = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]

        outputs, inputs, vision_info = run_vl_batch(
            batch, model, processor, backend
        )

        for j, sample in enumerate(batch):
            seq = outputs.sequences[j]

            if backend == "qwen":
                text = processor.decode(seq, skip_special_tokens=True)
            else:
                prompt_len = inputs["input_ids"][j].shape[0]
                text = processor.decode(seq[prompt_len:])

            pred_struct, parse_ok = parse_json(text)
            field_scores = score_fields(pred_struct, sample["ground_truth"])

            record = {
                "id": sample["id"],
                "image": sample["image"],
                "prompt": sample["prompt"],
                "ground_truth": sample["ground_truth"],

                "generated_text": text,
                "prediction_structured": pred_struct,
                "parse_success": parse_ok,
                "field_scores": field_scores,
                "is_hallucinated": any(
                    v == "hallucination" for v in field_scores.values()
                ),

                "generated_ids": seq.cpu().tolist(),
                "logits": extract_logits(model, outputs, j),
                "vision_grid": vision_info[j],

                "model_id": model_id,
            }

            records.append(record)

    return records
