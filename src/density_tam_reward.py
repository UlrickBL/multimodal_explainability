import numpy as np


def compute_additive_map(tam_maps: list[np.ndarray]) -> np.ndarray:
    """
    Additively combine a list of per-token TAM maps into a single activation map.

    Each element of *tam_maps* should be a 2-D float32 or uint8 array of the
    same spatial shape (H, W).  The result is normalised to [0, 255] uint8.

    Args:
        tam_maps: List of H×W arrays (one per token that contributed to a field value).

    Returns:
        Additive map normalised to uint8 [0, 255].
    """
    if not tam_maps:
        return np.zeros((1, 1), dtype=np.uint8)

    stacked = np.stack([m.astype(np.float32) for m in tam_maps], axis=0)
    summed = stacked.sum(axis=0)

    s_min, s_max = summed.min(), summed.max()
    if s_max > s_min:
        summed = (summed - s_min) / (s_max - s_min) * 255.0
    else:
        summed = np.zeros_like(summed)

    return summed.astype(np.uint8)


def compute_density_reward(tam_map: np.ndarray, mass_threshold: float = 0.85) -> float:
    """
    Compute a reward reflecting how focused (dense) the TAM activations are.

    Higher reward → activations form a tight cluster (good, less likely hallucination).
    Lower reward  → activations are scattered all over the image.

    The reward is inspired by the spatial entropy + density-area metric shown in the
    project picture (see README).

    Args:
        tam_map:        2-D numpy array of activations (H, W). Any dtype.
        mass_threshold: Fraction of total activation mass to capture (default 0.85 → 85 %).

    Returns:
        Scalar float in roughly [0, 10]. Returns 0.0 for empty / all-zero maps.
    """
    if tam_map is None:
        return 0.0

    arr = np.array(tam_map, dtype=np.float32)

    total = arr.sum()
    if total == 0.0:
        return 0.0
    p = arr / total

    entropy = float(-np.sum(p * np.log(p + 1e-8)))

    sorted_p = np.sort(p.flatten())[::-1]
    cumulative = np.cumsum(sorted_p)
    n_patches = int(np.argmax(cumulative >= mass_threshold)) + 1
    area_ratio = n_patches / p.size

    entropy_reward = 1.0 / (entropy + 1.0)
    density_reward = 1.0 - area_ratio

    total_reward = 0.5 * entropy_reward + 0.5 * density_reward
    return float(total_reward * 10.0)