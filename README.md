# Multimodal Explainability — TAM Hallucination Detection

> **Goal:** Detect hallucinations in structured-extraction outputs of VLMs by analysing where the model is "looking" when it generates each field value, using Token Activation Maps (TAM).

## Intuition

When a model correctly extracts a value from an image (e.g. a date on a receipt), its TAM activation map for those tokens should be **dense and clustered** around the relevant region.  When it hallucinates, the activations tend to be **scattered** — the model isn't really attending to any specific part of the image.

This is quantified by the **spatial density reward**:

| Behaviour | Entropy | Area for 85 % mass | Reward |
|-----------|---------|---------------------|--------|
| Dense / focused | Low | Small | **High** |
| Scattered / hallucinated | High | Large | **Low** |

---

## Quick start

```bash
# Install dependencies (on the remote GPU machine)
pip install git+https://github.com/huggingface/transformers
pip install mistral-common[opencv] accelerate datasets scipy PyMuPDF opencv-python tqdm

# Run Ministral on 5 samples
cd src
python main.py --model ministral --max_samples 5 --output_dir ./output

# Run QwenVL on 10 samples
python main.py --model qwen --max_samples 10 --output_dir ./output_qwen

# Run both, also save per-token overlays (disk-intensive!)
python main.py --model both --max_samples 5 --save_token_overlays --output_dir ./both_output
```

---

### Record schema

Each entry in `records.json` has:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Sample index |
| `image_path` | str | Path to saved input image |
| `generated_text` | str | Raw model output |
| `prediction_structured` | dict\|null | Parsed JSON |
| `parse_success` | bool | Whether JSON parsing succeeded |
| `field_scores` | dict | Per-key: `"correct"` / `"wrong"` / `"hallucination"` / `"miss"` / `"correct_null"` / `"parse_error"` |
| `is_hallucinated` | bool | True if any field is `"hallucination"` |
| `key_density_rewards` | dict | Per-key spatial density reward (float, higher = more focused) |
| `key_token_spans` | dict | Per-key token span `(start, end)` in the full sequence |
| `key_tam_image_paths` | dict | Per-key path to the saved PNG overlay |
| `model_id` | str | Model identifier |
| `vision_shape` | [H, W] | Vision patch grid size |

---

## How the density reward works

For each JSON key (e.g. `"date"`):

1. Find the token span in the generated sequence that encodes the value.
2. **Sum** the raw TAM maps for those tokens → *additive activation map*.
3. Compute:
   - **Spatial entropy** — lower means more concentrated activation.
   - **Area ratio** — what fraction of image patches is needed to cover 85 % of total activation.
4. Combine into a single score in **[0, 10]** (higher = more focused = less likely hallucination).

See `density_tam_reward.py` for the exact formula.

---

## Supported models

| Model | `--model` flag | Notes |
|-------|---------------|-------|
| `mistralai/Ministral-3-3B-Instruct-2512` | `ministral` | Uses `MistralCommonBackend` + FP8 quantisation |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `qwen` | Uses `AutoProcessor`; Qwen3-VL also supported |

---

## TAM algorithm

Token Activation Maps (TAM) explain *which image regions* a multimodal LLM attends to when generating each output token.  The core algorithm is in `src/tam.py` (unchanged from the [original implementation](https://github.com/xmed-lab/TAM)).

Key components used here:
- **Class Activation Map** from logits of the predicted token class.
- **Estimated Causal Inference (ECI)** — removes interference from repeated tokens.
- **Rank Gaussian Filter** — denoises the spatial activation map.