#!/usr/bin/env python3
"""
KV Cache Analysis Tool for SmolVLA Hallucination Investigation.

Captures and analyzes the Key-Value cache that stores visual and language
representations. Since cross-attention patterns are similar between
hallucination and normal cases, the root cause may be in the KV cache
CONTENTS - same attention weights reading different representations.

KV Cache Structure:
    past_key_values[layer_idx] = {
        "key_states": [batch, seq_len, num_heads, head_dim],
        "value_states": [batch, seq_len, num_heads, head_dim]
    }

Token positions in KV cache (prefix):
    [0-1]     Image special tokens
    [2-730]   Image patches (729 = 27x27 grid)
    [731]     Image end token
    [732-779] Language tokens (~48)
    [780]     State token

Usage (integrated into inference):
    from kv_cache_analysis import KVCacheCapture

    capture = KVCacheCapture()
    capture.capture(past_key_values, inference_step=200)
    capture.save(output_dir)

Standalone analysis:
    python kv_cache_analysis.py \
        --halluc-cache logs/kv_captures/halluc_step200.pt \
        --normal-cache logs/kv_captures/normal_step200.pt \
        --output-dir logs/kv_analysis
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict
import numpy as np

try:
    import torch
except ImportError:
    torch = None


# Token layout constants
IMAGE_PATCHES_START = 2
IMAGE_PATCHES_END = 731  # 2 + 729
LANGUAGE_START = 732
LANGUAGE_END = 780
STATE_INDEX = 780


@dataclass
class LayerKVStats:
    """Statistics for KV cache at a single layer."""
    layer_idx: int

    # Key states statistics
    key_mean: float
    key_std: float
    key_norm: float  # Frobenius norm

    # Value states statistics
    value_mean: float
    value_std: float
    value_norm: float

    # Per-region statistics (image vs language)
    image_key_norm: float
    image_value_norm: float
    language_key_norm: float
    language_value_norm: float


@dataclass
class KVCacheComparison:
    """Comparison between two KV caches."""
    halluc_path: str
    normal_path: str
    inference_step: int
    timestamp: str

    # Per-layer comparisons
    layer_comparisons: list  # List of dicts with per-layer metrics

    # Overall statistics
    total_key_diff_norm: float
    total_value_diff_norm: float

    # Per-region differences
    image_key_diff_norm: float
    image_value_diff_norm: float
    language_key_diff_norm: float
    language_value_diff_norm: float

    # Cosine similarity
    image_key_cosine_sim: float
    language_key_cosine_sim: float

    # Most different layers
    max_diff_layer: int
    max_diff_value: float


class KVCacheCapture:
    """Captures KV cache during inference for later analysis."""

    def __init__(self):
        self.captures = {}  # {inference_step: past_key_values}

    def capture(self, past_key_values: Dict, inference_step: int):
        """
        Capture KV cache at given inference step.

        Args:
            past_key_values: Dict[layer_idx] -> {"key_states", "value_states"}
            inference_step: Which inference step this capture is from
        """
        # Deep copy to avoid issues with in-place modifications
        capture_data = {}
        for layer_idx, kv in past_key_values.items():
            if torch is not None:
                capture_data[layer_idx] = {
                    "key_states": kv["key_states"].detach().cpu().clone(),
                    "value_states": kv["value_states"].detach().cpu().clone()
                }
            else:
                capture_data[layer_idx] = {
                    "key_states": np.array(kv["key_states"]),
                    "value_states": np.array(kv["value_states"])
                }

        self.captures[inference_step] = capture_data
        print(f"[KVCapture] Captured KV cache at step {inference_step}, "
              f"{len(capture_data)} layers")

    def save(self, output_dir: Path, inference_step: Optional[int] = None):
        """Save captured KV caches to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if inference_step is not None and inference_step in self.captures:
            steps_to_save = [inference_step]
        else:
            steps_to_save = list(self.captures.keys())

        for step in steps_to_save:
            output_file = output_dir / f"kv_cache_step_{step:04d}.pt"
            if torch is not None:
                torch.save(self.captures[step], output_file)
            else:
                np.savez(output_file.with_suffix('.npz'),
                        **{f"layer_{k}_{kk}": v
                           for k, kv in self.captures[step].items()
                           for kk, v in kv.items()})
            print(f"[KVCapture] Saved to {output_file}")

    def compute_stats(self, inference_step: int) -> list[LayerKVStats]:
        """Compute statistics for captured KV cache."""
        if inference_step not in self.captures:
            return []

        kv_cache = self.captures[inference_step]
        stats = []

        for layer_idx, kv in kv_cache.items():
            key_states = kv["key_states"]
            value_states = kv["value_states"]

            if torch is not None:
                key_states = key_states.numpy()
                value_states = value_states.numpy()

            # Overall statistics
            key_mean = float(np.mean(key_states))
            key_std = float(np.std(key_states))
            key_norm = float(np.linalg.norm(key_states))

            value_mean = float(np.mean(value_states))
            value_std = float(np.std(value_states))
            value_norm = float(np.linalg.norm(value_states))

            # Per-region statistics
            # Assuming key_states shape: [batch, seq_len, num_heads, head_dim]
            seq_len = key_states.shape[1]
            img_end = min(IMAGE_PATCHES_END, seq_len)
            lang_end = min(LANGUAGE_END, seq_len)

            image_keys = key_states[:, IMAGE_PATCHES_START:img_end]
            image_values = value_states[:, IMAGE_PATCHES_START:img_end]
            lang_keys = key_states[:, LANGUAGE_START:lang_end] if LANGUAGE_START < seq_len else None
            lang_values = value_states[:, LANGUAGE_START:lang_end] if LANGUAGE_START < seq_len else None

            stats.append(LayerKVStats(
                layer_idx=layer_idx,
                key_mean=key_mean,
                key_std=key_std,
                key_norm=key_norm,
                value_mean=value_mean,
                value_std=value_std,
                value_norm=value_norm,
                image_key_norm=float(np.linalg.norm(image_keys)),
                image_value_norm=float(np.linalg.norm(image_values)),
                language_key_norm=float(np.linalg.norm(lang_keys)) if lang_keys is not None else 0.0,
                language_value_norm=float(np.linalg.norm(lang_values)) if lang_values is not None else 0.0
            ))

        return stats


def load_kv_cache(path: Path) -> Dict:
    """Load KV cache from file."""
    if path.suffix == '.pt':
        if torch is None:
            raise ImportError("torch required to load .pt files")
        return torch.load(path, map_location='cpu')
    elif path.suffix == '.npz':
        data = np.load(path)
        # Reconstruct dict structure
        kv_cache = {}
        for key in data.files:
            parts = key.split('_')
            layer_idx = int(parts[1])
            state_type = '_'.join(parts[2:])
            if layer_idx not in kv_cache:
                kv_cache[layer_idx] = {}
            kv_cache[layer_idx][state_type] = data[key]
        return kv_cache
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def compare_kv_caches(
    halluc_cache: Dict,
    normal_cache: Dict,
    halluc_path: str,
    normal_path: str,
    inference_step: int
) -> KVCacheComparison:
    """Compare two KV caches and compute difference metrics."""

    layer_comparisons = []
    total_key_diff = 0.0
    total_value_diff = 0.0
    image_key_diff = 0.0
    image_value_diff = 0.0
    language_key_diff = 0.0
    language_value_diff = 0.0

    image_key_cosines = []
    language_key_cosines = []

    max_diff_layer = 0
    max_diff_value = 0.0

    for layer_idx in sorted(halluc_cache.keys()):
        if layer_idx not in normal_cache:
            continue

        h_key = halluc_cache[layer_idx]["key_states"]
        n_key = normal_cache[layer_idx]["key_states"]
        h_value = halluc_cache[layer_idx]["value_states"]
        n_value = normal_cache[layer_idx]["value_states"]

        # Convert to numpy
        if torch is not None and torch.is_tensor(h_key):
            h_key = h_key.numpy()
            n_key = n_key.numpy()
            h_value = h_value.numpy()
            n_value = n_value.numpy()

        # Overall difference
        key_diff = np.linalg.norm(h_key - n_key)
        value_diff = np.linalg.norm(h_value - n_value)

        total_key_diff += key_diff
        total_value_diff += value_diff

        # Per-region differences
        seq_len = h_key.shape[1]
        img_end = min(IMAGE_PATCHES_END, seq_len)
        lang_start = min(LANGUAGE_START, seq_len)
        lang_end = min(LANGUAGE_END, seq_len)

        # Image region
        h_img_key = h_key[:, IMAGE_PATCHES_START:img_end].flatten()
        n_img_key = n_key[:, IMAGE_PATCHES_START:img_end].flatten()
        h_img_val = h_value[:, IMAGE_PATCHES_START:img_end].flatten()
        n_img_val = n_value[:, IMAGE_PATCHES_START:img_end].flatten()

        img_key_diff = np.linalg.norm(h_img_key - n_img_key)
        img_val_diff = np.linalg.norm(h_img_val - n_img_val)
        image_key_diff += img_key_diff
        image_value_diff += img_val_diff

        # Cosine similarity for image keys
        if np.linalg.norm(h_img_key) > 0 and np.linalg.norm(n_img_key) > 0:
            cosine = np.dot(h_img_key, n_img_key) / (np.linalg.norm(h_img_key) * np.linalg.norm(n_img_key))
            image_key_cosines.append(cosine)

        # Language region
        if lang_start < seq_len:
            h_lang_key = h_key[:, lang_start:lang_end].flatten()
            n_lang_key = n_key[:, lang_start:lang_end].flatten()
            h_lang_val = h_value[:, lang_start:lang_end].flatten()
            n_lang_val = n_value[:, lang_start:lang_end].flatten()

            lang_key_diff = np.linalg.norm(h_lang_key - n_lang_key)
            lang_val_diff = np.linalg.norm(h_lang_val - n_lang_val)
            language_key_diff += lang_key_diff
            language_value_diff += lang_val_diff

            if np.linalg.norm(h_lang_key) > 0 and np.linalg.norm(n_lang_key) > 0:
                cosine = np.dot(h_lang_key, n_lang_key) / (np.linalg.norm(h_lang_key) * np.linalg.norm(n_lang_key))
                language_key_cosines.append(cosine)

        # Track max difference layer
        layer_total_diff = key_diff + value_diff
        if layer_total_diff > max_diff_value:
            max_diff_value = layer_total_diff
            max_diff_layer = layer_idx

        layer_comparisons.append({
            "layer_idx": layer_idx,
            "key_diff_norm": float(key_diff),
            "value_diff_norm": float(value_diff),
            "image_key_diff": float(img_key_diff),
            "language_key_diff": float(lang_key_diff) if lang_start < seq_len else 0.0
        })

    return KVCacheComparison(
        halluc_path=halluc_path,
        normal_path=normal_path,
        inference_step=inference_step,
        timestamp=datetime.now().isoformat(),
        layer_comparisons=layer_comparisons,
        total_key_diff_norm=float(total_key_diff),
        total_value_diff_norm=float(total_value_diff),
        image_key_diff_norm=float(image_key_diff),
        image_value_diff_norm=float(image_value_diff),
        language_key_diff_norm=float(language_key_diff),
        language_value_diff_norm=float(language_value_diff),
        image_key_cosine_sim=float(np.mean(image_key_cosines)) if image_key_cosines else 0.0,
        language_key_cosine_sim=float(np.mean(language_key_cosines)) if language_key_cosines else 0.0,
        max_diff_layer=max_diff_layer,
        max_diff_value=float(max_diff_value)
    )


def analyze_kv_caches(
    halluc_path: Path,
    normal_path: Path,
    output_dir: Path,
    inference_step: int = 200
):
    """Main analysis function for comparing KV caches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading hallucination KV cache from {halluc_path}")
    halluc_cache = load_kv_cache(halluc_path)

    print(f"Loading normal KV cache from {normal_path}")
    normal_cache = load_kv_cache(normal_path)

    print(f"Comparing {len(halluc_cache)} layers...")
    comparison = compare_kv_caches(
        halluc_cache, normal_cache,
        str(halluc_path), str(normal_path),
        inference_step
    )

    # Save comparison results
    output_file = output_dir / "kv_cache_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(comparison), f, indent=2)
    print(f"Saved comparison to {output_file}")

    # Print summary
    print_summary(comparison)

    # Generate visualizations
    generate_visualizations(comparison, output_dir)

    return comparison


def print_summary(comparison: KVCacheComparison):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("KV CACHE COMPARISON SUMMARY")
    print("="*70)

    print(f"\nInference step: {comparison.inference_step}")
    print(f"Layers compared: {len(comparison.layer_comparisons)}")

    print(f"\n--- Overall Differences ---")
    print(f"Total key diff norm: {comparison.total_key_diff_norm:.4f}")
    print(f"Total value diff norm: {comparison.total_value_diff_norm:.4f}")

    print(f"\n--- Per-Region Differences ---")
    print(f"Image key diff: {comparison.image_key_diff_norm:.4f}")
    print(f"Image value diff: {comparison.image_value_diff_norm:.4f}")
    print(f"Language key diff: {comparison.language_key_diff_norm:.4f}")
    print(f"Language value diff: {comparison.language_value_diff_norm:.4f}")

    print(f"\n--- Cosine Similarity ---")
    print(f"Image key cosine sim: {comparison.image_key_cosine_sim:.4f}")
    print(f"Language key cosine sim: {comparison.language_key_cosine_sim:.4f}")

    print(f"\n--- Max Difference ---")
    print(f"Most different layer: {comparison.max_diff_layer}")
    print(f"Max diff value: {comparison.max_diff_value:.4f}")

    print("\n" + "="*70)


def generate_visualizations(comparison: KVCacheComparison, output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    # Per-layer difference plot
    layers = [lc["layer_idx"] for lc in comparison.layer_comparisons]
    key_diffs = [lc["key_diff_norm"] for lc in comparison.layer_comparisons]
    value_diffs = [lc["value_diff_norm"] for lc in comparison.layer_comparisons]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Key differences per layer
    ax1 = axes[0]
    ax1.bar(layers, key_diffs, alpha=0.7, label='Key diff')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('L2 Norm of Difference')
    ax1.set_title('Key States Difference per Layer')
    ax1.axhline(y=np.mean(key_diffs), color='r', linestyle='--',
               label=f'Mean: {np.mean(key_diffs):.2f}')
    ax1.legend()

    # Value differences per layer
    ax2 = axes[1]
    ax2.bar(layers, value_diffs, alpha=0.7, color='orange', label='Value diff')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('L2 Norm of Difference')
    ax2.set_title('Value States Difference per Layer')
    ax2.axhline(y=np.mean(value_diffs), color='r', linestyle='--',
               label=f'Mean: {np.mean(value_diffs):.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'kv_diff_per_layer.png', dpi=150)
    plt.close()
    print(f"Saved per-layer plot to {output_dir / 'kv_diff_per_layer.png'}")

    # Image vs Language difference
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Image Keys', 'Image Values', 'Language Keys', 'Language Values']
    values = [
        comparison.image_key_diff_norm,
        comparison.image_value_diff_norm,
        comparison.language_key_diff_norm,
        comparison.language_value_diff_norm
    ]

    bars = ax.bar(categories, values, alpha=0.7)
    ax.set_ylabel('L2 Norm of Difference')
    ax.set_title('KV Cache Difference by Region')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'kv_diff_by_region.png', dpi=150)
    plt.close()
    print(f"Saved region plot to {output_dir / 'kv_diff_by_region.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze KV cache differences")
    parser.add_argument("--halluc-cache", type=str, help="Path to hallucination KV cache")
    parser.add_argument("--normal-cache", type=str, help="Path to normal KV cache")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--inference-step", type=int, default=200, help="Inference step")

    args = parser.parse_args()

    if args.halluc_cache and args.normal_cache:
        analyze_kv_caches(
            Path(args.halluc_cache),
            Path(args.normal_cache),
            Path(args.output_dir),
            args.inference_step
        )
    else:
        print("Usage: Provide --halluc-cache and --normal-cache for comparison")
        print("       Or integrate KVCacheCapture into inference script")


if __name__ == "__main__":
    main()
