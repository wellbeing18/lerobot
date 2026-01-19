#!/usr/bin/env python3
"""
Distractor Attention Analyzer for SmolVLA Hallucination Investigation.

This tool quantifies how much attention the action expert pays to distractor
objects (like the banana) versus task-relevant objects (like the bin).

Given bounding boxes for regions of interest, it computes:
1. Distractor attention ratio - % of attention to distractor region
2. Target attention ratio - % of attention to task-relevant region
3. Attention shift analysis across denoising steps
4. Comparison between hallucination and normal cases

Usage:
    python distractor_attention.py \
        --hallucination-dir logs/yogurt_banana_leftarm/cross_attention_analysis/case1_v2 \
        --normal-dir logs/yogurt_banana_leftarm/cross_attention_analysis/case3_normal_v2 \
        --distractor-bbox 380,260,430,310 \
        --target-bbox 50,100,250,250 \
        --output-dir logs/yogurt_banana_leftarm/distractor_analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RegionAttention:
    """Attention statistics for a specific image region."""
    name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    attention_ratio: float  # Fraction of total attention in this region
    attention_sum: float  # Raw attention sum


@dataclass
class StepRegionAnalysis:
    """Region attention analysis for a single denoising step."""
    step: int
    time: float
    distractor_attention: float
    target_attention: float
    other_attention: float
    distractor_to_target_ratio: float  # distractor / target


@dataclass
class CaseComparison:
    """Comparison between hallucination and normal case."""
    hallucination_case: str
    normal_case: str
    timestamp: str

    # Per-step comparison
    hallucination_steps: List[StepRegionAnalysis] = field(default_factory=list)
    normal_steps: List[StepRegionAnalysis] = field(default_factory=list)

    # Aggregated metrics
    hall_avg_distractor_attention: float = 0.0
    norm_avg_distractor_attention: float = 0.0
    hall_avg_target_attention: float = 0.0
    norm_avg_target_attention: float = 0.0

    # Difference metrics
    distractor_attention_diff: float = 0.0  # hallucination - normal


# ============================================================================
# SPATIAL MAPPING UTILITIES
# ============================================================================

def pixel_bbox_to_patch_mask(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    patch_grid_size: int,
) -> np.ndarray:
    """
    Convert a pixel bounding box to a patch-level mask.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        image_size: (width, height) of the input image
        patch_grid_size: Size of the patch grid (e.g., 15 for 15x15)

    Returns:
        mask: Boolean array of shape (patch_grid_size, patch_grid_size)
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size

    # Convert to normalized coordinates [0, 1]
    x1_norm = x1 / img_w
    y1_norm = y1 / img_h
    x2_norm = x2 / img_w
    y2_norm = y2 / img_h

    # Convert to patch indices
    p1_x = int(x1_norm * patch_grid_size)
    p1_y = int(y1_norm * patch_grid_size)
    p2_x = int(np.ceil(x2_norm * patch_grid_size))
    p2_y = int(np.ceil(y2_norm * patch_grid_size))

    # Clamp to grid bounds
    p1_x = max(0, min(p1_x, patch_grid_size - 1))
    p1_y = max(0, min(p1_y, patch_grid_size - 1))
    p2_x = max(0, min(p2_x, patch_grid_size))
    p2_y = max(0, min(p2_y, patch_grid_size))

    # Create mask
    mask = np.zeros((patch_grid_size, patch_grid_size), dtype=bool)
    mask[p1_y:p2_y, p1_x:p2_x] = True

    return mask


def compute_region_attention(
    spatial_attention: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    """
    Compute the fraction of attention within a region.

    Args:
        spatial_attention: Attention map (H, W) normalized to sum to 1
        region_mask: Boolean mask (H, W) indicating the region

    Returns:
        Fraction of total attention within the region
    """
    total = spatial_attention.sum()
    if total == 0:
        return 0.0

    region_attention = spatial_attention[region_mask].sum()
    return float(region_attention / total)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_spatial_attention_from_heatmaps(heatmaps_dir: Path) -> Dict[int, np.ndarray]:
    """
    Load spatial attention maps from saved heatmap .npy files.

    If .npy files don't exist, attempt to reconstruct from JSON.
    """
    attention_maps = {}

    # Check for .npy files first
    npy_files = list(heatmaps_dir.glob("step_*_spatial.npy"))
    if npy_files:
        for npy_file in npy_files:
            step = int(npy_file.stem.split('_')[1])
            attention_maps[step] = np.load(npy_file)
        return attention_maps

    # Otherwise, we need to re-run capture with saving enabled
    return attention_maps


def analyze_single_case(
    case_dir: Path,
    distractor_bbox: Tuple[int, int, int, int],
    target_bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int] = (640, 480),
    patch_grid_size: int = 15,
    target_inference_steps: List[int] = None,
) -> Dict[int, List[StepRegionAnalysis]]:
    """
    Analyze region attention for a single case.

    Returns dict mapping inference_step -> list of StepRegionAnalysis (per denoising step).
    """
    heatmaps_dir = case_dir / "heatmaps"
    results = {}  # {inference_step: [StepRegionAnalysis]}

    # Create patch masks
    distractor_mask = pixel_bbox_to_patch_mask(distractor_bbox, image_size, patch_grid_size)
    target_mask = pixel_bbox_to_patch_mask(target_bbox, image_size, patch_grid_size)

    # Load analysis JSON to get step info
    analysis_path = case_dir / "cross_attention_analysis.json"
    if not analysis_path.exists():
        print(f"No analysis JSON found at {analysis_path}")
        return results

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Get available inference steps
    inference_steps = analysis.get("inference_steps", [0])
    if target_inference_steps:
        inference_steps = [s for s in inference_steps if s in target_inference_steps]

    print(f"  Analyzing inference steps: {inference_steps}")

    # Load spatial attention for each inference step and denoising step
    for inf_step in inference_steps:
        results[inf_step] = []

        # Try loading all 10 denoising steps
        for denoise_step in range(10):
            # New format: inf_{step}_denoise_{step}_spatial.npy
            npy_path = heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_spatial.npy"

            if not npy_path.exists():
                # Try old format for backward compatibility
                npy_path = heatmaps_dir / f"step_{denoise_step:04d}_spatial.npy"
                if not npy_path.exists():
                    continue

            spatial_attention = np.load(npy_path)

            # Normalize
            spatial_attention = spatial_attention / (spatial_attention.sum() + 1e-10)

            # Compute region attention
            distractor_att = compute_region_attention(spatial_attention, distractor_mask)
            target_att = compute_region_attention(spatial_attention, target_mask)
            other_att = 1.0 - distractor_att - target_att

            # Compute ratio (avoid division by zero)
            dist_to_target = distractor_att / (target_att + 1e-10)

            # Estimate time from denoising step
            time = 1.0 - denoise_step * 0.1

            results[inf_step].append(StepRegionAnalysis(
                step=denoise_step,
                time=time,
                distractor_attention=distractor_att,
                target_attention=target_att,
                other_attention=other_att,
                distractor_to_target_ratio=dist_to_target,
            ))

    return results


def compare_cases(
    hallucination_dir: Path,
    normal_dir: Path,
    distractor_bbox: Tuple[int, int, int, int],
    target_bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int] = (640, 480),
    target_inference_steps: List[int] = None,
) -> Tuple[CaseComparison, Dict]:
    """
    Compare attention patterns between hallucination and normal cases.

    Returns:
        comparison: Overall comparison summary
        per_step_data: {inf_step: {"hallucination": [...], "normal": [...]}}
    """
    print(f"Analyzing hallucination case: {hallucination_dir}")
    hall_data = analyze_single_case(
        hallucination_dir, distractor_bbox, target_bbox,
        image_size=image_size, target_inference_steps=target_inference_steps
    )

    print(f"Analyzing normal case: {normal_dir}")
    norm_data = analyze_single_case(
        normal_dir, distractor_bbox, target_bbox,
        image_size=image_size, target_inference_steps=target_inference_steps
    )

    # Flatten for overall comparison
    hall_flat = [step for steps in hall_data.values() for step in steps]
    norm_flat = [step for steps in norm_data.values() for step in steps]

    comparison = CaseComparison(
        hallucination_case=str(hallucination_dir),
        normal_case=str(normal_dir),
        timestamp=datetime.now().isoformat(),
        hallucination_steps=hall_flat,
        normal_steps=norm_flat,
    )

    # Compute aggregated metrics
    if hall_flat:
        comparison.hall_avg_distractor_attention = np.mean([s.distractor_attention for s in hall_flat])
        comparison.hall_avg_target_attention = np.mean([s.target_attention for s in hall_flat])

    if norm_flat:
        comparison.norm_avg_distractor_attention = np.mean([s.distractor_attention for s in norm_flat])
        comparison.norm_avg_target_attention = np.mean([s.target_attention for s in norm_flat])

    comparison.distractor_attention_diff = (
        comparison.hall_avg_distractor_attention - comparison.norm_avg_distractor_attention
    )

    # Build per-inference-step data for detailed analysis
    per_step_data = {}
    all_inf_steps = set(hall_data.keys()) | set(norm_data.keys())
    for inf_step in sorted(all_inf_steps):
        per_step_data[inf_step] = {
            "hallucination": hall_data.get(inf_step, []),
            "normal": norm_data.get(inf_step, []),
        }

    return comparison, per_step_data


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_region_comparison(
    comparison: CaseComparison,
    output_path: Optional[Path] = None,
):
    """
    Plot comparison of distractor attention between cases (overall summary).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    hall_steps = [s.step for s in comparison.hallucination_steps]
    hall_dist = [s.distractor_attention for s in comparison.hallucination_steps]
    hall_targ = [s.target_attention for s in comparison.hallucination_steps]

    norm_steps = [s.step for s in comparison.normal_steps]
    norm_dist = [s.distractor_attention for s in comparison.normal_steps]
    norm_targ = [s.target_attention for s in comparison.normal_steps]

    # Plot 1: Distractor attention over denoising steps
    ax1 = axes[0, 0]
    if hall_steps:
        ax1.plot(hall_steps, hall_dist, 'r-o', label='Hallucination', linewidth=2, markersize=6)
    if norm_steps:
        ax1.plot(norm_steps, norm_dist, 'b-o', label='Normal', linewidth=2, markersize=6)
    ax1.set_xlabel('Denoising Step')
    ax1.set_ylabel('Attention Fraction')
    ax1.set_title('Distractor Attention Over Denoising')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Target attention over denoising steps
    ax2 = axes[0, 1]
    if hall_steps:
        ax2.plot(hall_steps, hall_targ, 'r-o', label='Hallucination', linewidth=2, markersize=6)
    if norm_steps:
        ax2.plot(norm_steps, norm_targ, 'b-o', label='Normal', linewidth=2, markersize=6)
    ax2.set_xlabel('Denoising Step')
    ax2.set_ylabel('Attention Fraction')
    ax2.set_title('Target (Bin) Attention Over Denoising')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distractor/Target ratio
    ax3 = axes[1, 0]
    hall_ratio = [s.distractor_to_target_ratio for s in comparison.hallucination_steps]
    norm_ratio = [s.distractor_to_target_ratio for s in comparison.normal_steps]
    if hall_steps:
        ax3.plot(hall_steps, hall_ratio, 'r-o', label='Hallucination', linewidth=2, markersize=6)
    if norm_steps:
        ax3.plot(norm_steps, norm_ratio, 'b-o', label='Normal', linewidth=2, markersize=6)
    ax3.set_xlabel('Denoising Step')
    ax3.set_ylabel('Distractor / Target Ratio')
    ax3.set_title('Distractor-to-Target Attention Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Bar chart of average attention
    ax4 = axes[1, 1]
    x = np.arange(2)
    width = 0.35

    hall_bars = [comparison.hall_avg_distractor_attention, comparison.hall_avg_target_attention]
    norm_bars = [comparison.norm_avg_distractor_attention, comparison.norm_avg_target_attention]

    bars1 = ax4.bar(x - width/2, hall_bars, width, label='Hallucination', color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, norm_bars, width, label='Normal', color='blue', alpha=0.7)

    ax4.set_ylabel('Average Attention')
    ax4.set_title('Average Region Attention')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Distractor', 'Target'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.suptitle('Distractor vs Target Attention Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {output_path}")

    plt.close()


def plot_per_inference_step_comparison(
    per_step_data: Dict,
    output_dir: Path,
):
    """
    Plot comparison for each inference step separately.
    Critical for understanding WHERE in the task hallucination occurs.
    """
    for inf_step, data in per_step_data.items():
        hall_steps = data["hallucination"]
        norm_steps = data["normal"]

        if not hall_steps and not norm_steps:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Distractor attention over denoising
        ax1 = axes[0]
        if hall_steps:
            ax1.plot([s.step for s in hall_steps], [s.distractor_attention for s in hall_steps],
                    'r-o', label='Hallucination', linewidth=2, markersize=6)
        if norm_steps:
            ax1.plot([s.step for s in norm_steps], [s.distractor_attention for s in norm_steps],
                    'b-o', label='Normal', linewidth=2, markersize=6)
        ax1.set_xlabel('Denoising Step')
        ax1.set_ylabel('Attention Fraction')
        ax1.set_title('Distractor Attention')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Target attention over denoising
        ax2 = axes[1]
        if hall_steps:
            ax2.plot([s.step for s in hall_steps], [s.target_attention for s in hall_steps],
                    'r-o', label='Hallucination', linewidth=2, markersize=6)
        if norm_steps:
            ax2.plot([s.step for s in norm_steps], [s.target_attention for s in norm_steps],
                    'b-o', label='Normal', linewidth=2, markersize=6)
        ax2.set_xlabel('Denoising Step')
        ax2.set_ylabel('Attention Fraction')
        ax2.set_title('Target (Bin) Attention')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Bar comparison
        ax3 = axes[2]
        x = np.arange(2)
        width = 0.35

        hall_avg_dist = np.mean([s.distractor_attention for s in hall_steps]) if hall_steps else 0
        hall_avg_targ = np.mean([s.target_attention for s in hall_steps]) if hall_steps else 0
        norm_avg_dist = np.mean([s.distractor_attention for s in norm_steps]) if norm_steps else 0
        norm_avg_targ = np.mean([s.target_attention for s in norm_steps]) if norm_steps else 0

        bars1 = ax3.bar(x - width/2, [hall_avg_dist, hall_avg_targ], width, label='Hallucination', color='red', alpha=0.7)
        bars2 = ax3.bar(x + width/2, [norm_avg_dist, norm_avg_targ], width, label='Normal', color='blue', alpha=0.7)

        ax3.set_ylabel('Avg Attention')
        ax3.set_title('Avg Region Attention')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Distractor', 'Target'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Inference Step {inf_step}: Distractor vs Target Attention', fontsize=12, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"inf_step_{inf_step:04d}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_region_overlay(
    image_path: Path,
    distractor_bbox: Tuple[int, int, int, int],
    target_bbox: Tuple[int, int, int, int],
    output_path: Optional[Path] = None,
):
    """
    Plot the image with region bounding boxes overlaid.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)

    # Draw distractor bbox (red)
    x1, y1, x2, y2 = distractor_bbox
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=3, edgecolor='red', facecolor='none',
                             label='Distractor (banana)')
    ax.add_patch(rect)

    # Draw target bbox (green)
    x1, y1, x2, y2 = target_bbox
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=3, edgecolor='green', facecolor='none',
                             label='Target (bin)')
    ax.add_patch(rect)

    ax.legend(loc='upper right')
    ax.set_title('Region Definitions for Attention Analysis')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved region overlay: {output_path}")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze distractor attention in SmolVLA")
    parser.add_argument("--hallucination-dir", type=str, required=True,
                       help="Directory with hallucination case cross-attention analysis")
    parser.add_argument("--normal-dir", type=str, required=True,
                       help="Directory with normal case cross-attention analysis")
    parser.add_argument("--distractor-bbox", type=str, required=True,
                       help="Distractor bounding box as x1,y1,x2,y2")
    parser.add_argument("--target-bbox", type=str, required=True,
                       help="Target region bounding box as x1,y1,x2,y2")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--image-path", type=str, default=None,
                       help="Path to reference image for region overlay visualization")

    args = parser.parse_args()

    # Parse bounding boxes
    distractor_bbox = tuple(map(int, args.distractor_bbox.split(',')))
    target_bbox = tuple(map(int, args.target_bbox.split(',')))

    print("=" * 60)
    print("SmolVLA Distractor Attention Analysis")
    print("=" * 60)
    print(f"Distractor bbox: {distractor_bbox}")
    print(f"Target bbox: {target_bbox}")

    hallucination_dir = Path(args.hallucination_dir)
    normal_dir = Path(args.normal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare cases
    comparison, per_step_data = compare_cases(
        hallucination_dir=hallucination_dir,
        normal_dir=normal_dir,
        distractor_bbox=distractor_bbox,
        target_bbox=target_bbox,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"\nHallucination case:")
    print(f"  Avg distractor attention: {comparison.hall_avg_distractor_attention:.4f}")
    print(f"  Avg target attention: {comparison.hall_avg_target_attention:.4f}")

    print(f"\nNormal case:")
    print(f"  Avg distractor attention: {comparison.norm_avg_distractor_attention:.4f}")
    print(f"  Avg target attention: {comparison.norm_avg_target_attention:.4f}")

    print(f"\nDifference (hallucination - normal):")
    print(f"  Distractor attention diff: {comparison.distractor_attention_diff:+.4f}")

    # Print per-inference-step summary
    print("\n" + "=" * 60)
    print("PER-INFERENCE-STEP SUMMARY")
    print("=" * 60)
    for inf_step, data in sorted(per_step_data.items()):
        hall = data["hallucination"]
        norm = data["normal"]
        hall_dist = np.mean([s.distractor_attention for s in hall]) if hall else 0
        norm_dist = np.mean([s.distractor_attention for s in norm]) if norm else 0
        diff = hall_dist - norm_dist
        print(f"  Inf step {inf_step:4d}: Hall dist={hall_dist:.4f}, Norm dist={norm_dist:.4f}, Diff={diff:+.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_region_comparison(
        comparison,
        output_path=output_dir / "distractor_comparison.png"
    )

    # Plot per-inference-step comparisons
    plot_per_inference_step_comparison(
        per_step_data,
        output_dir=output_dir,
    )

    # Plot region overlay if image provided
    if args.image_path:
        plot_region_overlay(
            Path(args.image_path),
            distractor_bbox,
            target_bbox,
            output_path=output_dir / "region_overlay.png"
        )

    # Save JSON results
    results = {
        "hallucination_case": comparison.hallucination_case,
        "normal_case": comparison.normal_case,
        "timestamp": comparison.timestamp,
        "distractor_bbox": distractor_bbox,
        "target_bbox": target_bbox,
        "summary": {
            "hall_avg_distractor_attention": comparison.hall_avg_distractor_attention,
            "norm_avg_distractor_attention": comparison.norm_avg_distractor_attention,
            "hall_avg_target_attention": comparison.hall_avg_target_attention,
            "norm_avg_target_attention": comparison.norm_avg_target_attention,
            "distractor_attention_diff": comparison.distractor_attention_diff,
        },
        "per_inference_step": {
            str(inf_step): {
                "hallucination_avg_distractor": np.mean([s.distractor_attention for s in data["hallucination"]]) if data["hallucination"] else 0,
                "normal_avg_distractor": np.mean([s.distractor_attention for s in data["normal"]]) if data["normal"] else 0,
            }
            for inf_step, data in per_step_data.items()
        },
        "hallucination_steps": [asdict(s) for s in comparison.hallucination_steps],
        "normal_steps": [asdict(s) for s in comparison.normal_steps],
    }

    with open(output_dir / "distractor_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis saved to: {output_dir}")
    print(f"  - distractor_comparison.png")
    print(f"  - inf_step_*_comparison.png")
    print(f"  - distractor_analysis.json")
    if args.image_path:
        print(f"  - region_overlay.png")


if __name__ == "__main__":
    main()
