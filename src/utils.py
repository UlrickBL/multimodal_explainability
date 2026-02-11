def build_prompt(task_prompt: str, schema: Dict[str, Any]) -> str:
    return f"""
{task_prompt}

Return a JSON object with EXACTLY these keys:
{json.dumps(schema, indent=2)}

Rules:
- If a value is not present, return null
- Do NOT guess
- Output JSON only
"""

def parse_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end]), True
    except Exception:
        return None, False


def score_fields(pred, gt):
    scores = {}
    for k in gt.keys():
        if pred is None:
            scores[k] = "invalid"
        elif pred.get(k) is None and gt[k] is None:
            scores[k] = "correct_null"
        elif pred.get(k) == gt[k]:
            scores[k] = "correct"
        elif pred.get(k) is not None and gt[k] is None:
            scores[k] = "hallucination"
        elif pred.get(k) is None and gt[k] is not None:
            scores[k] = "miss"
        else:
            scores[k] = "wrong"
    return scores

def extract_logits(model, outputs, sample_idx):
    return np.stack([
        model.lm_head(h[sample_idx]).cpu().numpy()
        for h in outputs.hidden_states
    ])

def push_dataset(records, repo_id):
    ds = Dataset.from_list(records)
    ds.push_to_hub(repo_id)
