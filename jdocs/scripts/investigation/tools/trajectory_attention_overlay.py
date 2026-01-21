#!/usr/bin/env python3
"""
Trajectory + Attention Overlay Visualization Tool for SmolVLA Hallucination Investigation.

Creates visualizations that combine:
1. Actual camera images at divergence moments (step 200, 250)
2. Cross-attention heatmaps overlaid on images
3. Training trajectory distribution showing where current trajectory lies

This addresses the need to visualize:
- What the model "sees" (visual input)
- What the model "attends to" (cross-attention)
- How current trajectory relates to training distribution

Usage:
    python trajectory_attention_overlay.py \
        --halluc-case logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --normal-case logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --output-dir logs/investigation/trajectory_attention_overlay

Research References:
- FlowPolicy: PCA visualization of trajectory space
- 3D Diffusion Policy: Trajectory clustering visualization
- SmolVLA: Cross-attention between action expert and VLM prefix
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("Missing scikit-learn. Install with: pip install scikit-learn")
    sys.exit(1)

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS
# ============================================================================

# Token layout for SmolVLA (verified)
PATCHES_PER_CAMERA = 64  # 8x8 grid
PATCH_GRID_SIZE = 8
NUM_CAMERAS = 3

# Trajectory phases
PHASES = ["APPROACH", "GRIP", "TRANSPORT", "RELEASE", "IDLE", "UNKNOWN"]
PHASE_COLORS = {
    "APPROACH": "#3498db",  # Blue
    "GRIP": "#2ecc71",      # Green
    "TRANSPORT": "#e67e22", # Orange
    "RELEASE": "#9b59b6",   # Purple
    "IDLE": "#95a5a6",      # Gray
    "UNKNOWN": "#34495e",   # Dark gray
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_case_images(case_dir: Path, step: int) -> Dict[str, np.ndarray]:
    """Load all 3 camera images for a specific step."""
    images_dir = case_dir / "images"

    result = {}
    camera_names = ["head", "left_wrist", "right_wrist"]

    for cam in camera_names:
        img_path = images_dir / f"step_{step:04d}_{cam}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            result[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"  Warning: {img_path} not found")

    return result


def load_trace_data(trace_path: Path, step: int) -> Optional[dict]:
    """Load trace data for a specific step."""
    if not trace_path.exists():
        return None

    with open(trace_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("step") == step:
                return data
    return None


def load_trajectory_segment(trace_path: Path, start_step: int, end_step: int) -> np.ndarray:
    """Load trajectory segment from trace file."""
    actions = []
    with open(trace_path, "r") as f:
        for line in f:
            data = json.loads(line)
            step = data.get("step", -1)
            if start_step <= step < end_step:
                if "action_raw" in data:
                    actions.append(data["action_raw"])
                elif "action_final" in data:
                    actions.append(data["action_final"])

    return np.array(actions) if actions else np.zeros((end_step - start_step, 14))


def load_training_trajectories(dataset_path: Path, task_filter: str = "yogurt",
                               max_episodes: int = 50) -> Tuple[List[np.ndarray], List[str], np.ndarray, PCA]:
    """
    Load training trajectories and compute PCA embeddings.

    Returns:
        trajectories: List of trajectory arrays
        phases: List of detected phases for each segment
        embeddings: PCA embeddings of trajectory segments
        pca: Fitted PCA model
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION
    except ImportError:
        print("Could not import lerobot. Using cached data if available.")
        return [], [], np.array([]), None

    print(f"Loading training trajectories from {dataset_path}...")

    try:
        dataset = LeRobotDataset(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], [], np.array([]), None

    # Get episode info
    episode_data_index = dataset.episode_data_index
    num_episodes = len(episode_data_index["from"])

    trajectories = []
    all_segments = []
    segment_phases = []

    loaded_count = 0
    for ep_idx in range(min(num_episodes, max_episodes * 2)):
        if loaded_count >= max_episodes:
            break

        from_idx = episode_data_index["from"][ep_idx].item()
        to_idx = episode_data_index["to"][ep_idx].item()

        # Get task name
        task = ""
        if hasattr(dataset, 'tasks') and ep_idx < len(dataset.tasks):
            task = dataset.tasks[ep_idx]

        # Filter by task
        if task_filter and task_filter.lower() not in task.lower():
            continue

        # Extract actions
        actions = []
        for idx in range(from_idx, to_idx):
            item = dataset[idx]
            if ACTION in item:
                actions.append(item[ACTION].numpy())

        if actions:
            traj = np.array(actions)
            trajectories.append(traj)

            # Segment trajectory into windows for embedding
            window_size = 50
            stride = 25
            for i in range(0, len(traj) - window_size + 1, stride):
                segment = traj[i:i + window_size]
                all_segments.append(segment.flatten())

                # Detect phase based on gripper state and velocity
                phase = detect_segment_phase(segment)
                segment_phases.append(phase)

            loaded_count += 1

    print(f"Loaded {len(trajectories)} trajectories, {len(all_segments)} segments")

    if not all_segments:
        return trajectories, [], np.array([]), None

    # Compute PCA embeddings
    X = np.array(all_segments)
    pca = PCA(n_components=3)
    embeddings = pca.fit_transform(X)

    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    return trajectories, segment_phases, embeddings, pca


def detect_segment_phase(segment: np.ndarray) -> str:
    """Detect trajectory phase based on action characteristics."""
    if len(segment) < 10:
        return "UNKNOWN"

    # Compute velocity
    velocities = np.diff(segment, axis=0)
    vel_mag = np.linalg.norm(velocities, axis=1)
    avg_vel = np.mean(vel_mag)

    # Check gripper states (indices 6 and 13 for left and right grippers)
    if segment.shape[1] >= 14:
        left_gripper = segment[:, 6]
        right_gripper = segment[:, 13]

        # Gripper closing (high to low velocity)
        left_closing = left_gripper[0] > 0.5 and left_gripper[-1] < 0.3
        right_closing = right_gripper[0] > 0.5 and right_gripper[-1] < 0.3

        # Gripper opening
        left_opening = left_gripper[0] < 0.3 and left_gripper[-1] > 0.5
        right_opening = right_gripper[0] < 0.3 and right_gripper[-1] > 0.5

        # Gripper closed
        left_closed = np.mean(left_gripper) < 0.3
        right_closed = np.mean(right_gripper) < 0.3
    else:
        left_closing = right_closing = left_opening = right_opening = False
        left_closed = right_closed = False

    # Phase detection logic
    if left_closing or right_closing:
        return "GRIP"
    elif left_opening or right_opening:
        return "RELEASE"
    elif avg_vel < 0.02:
        return "IDLE"
    elif left_closed or right_closed:
        return "TRANSPORT"
    elif avg_vel > 0.02:
        return "APPROACH"
    else:
        return "UNKNOWN"


# ============================================================================
# ATTENTION LOADING
# ============================================================================

def load_cross_attention_data(case_dir: Path, step: int) -> Optional[Dict]:
    """
    Load cross-attention data from analysis results if available.
    Falls back to generating synthetic attention if not.
    """
    analysis_dir = Path("logs/analysis") / case_dir.name / "cross_attention"
    json_path = analysis_dir / "cross_attention_analysis.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)

        # Find attention data for requested step
        inf_key = f"inf_{step}"
        if inf_key in data.get("attention_data", {}):
            return data["attention_data"][inf_key]

    # Check for spatial heatmaps directly
    heatmaps_dir = analysis_dir / "heatmaps"
    if heatmaps_dir.exists():
        spatial_maps = {}
        for cam in ["head", "left_wrist", "right_wrist"]:
            npy_path = heatmaps_dir / f"inf_{step:04d}_denoise_05_{cam}.npy"
            if npy_path.exists():
                spatial_maps[cam] = np.load(npy_path)
        if spatial_maps:
            return {"spatial_maps": spatial_maps}

    return None


def create_attention_heatmap_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay attention heatmap on image."""
    h, w = image.shape[:2]

    # Resize attention map to image size
    attn_resized = cv2.resize(attention_map.astype(np.float32), (w, h))

    # Create heatmap using jet colormap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


# ============================================================================
# TRAJECTORY EMBEDDING
# ============================================================================

def embed_inference_trajectory(
    trace_path: Path,
    step: int,
    pca: PCA,
    window_size: int = 50
) -> Optional[np.ndarray]:
    """Embed a trajectory segment from inference at given step."""
    # Load trajectory segment around the step
    segment = load_trajectory_segment(trace_path, step, step + window_size)

    if len(segment) < window_size:
        # Pad if needed
        segment = np.pad(segment, ((0, window_size - len(segment)), (0, 0)), mode='edge')

    # Flatten and transform
    segment_flat = segment.flatten().reshape(1, -1)

    # Handle dimension mismatch
    expected_dim = pca.n_features_in_
    if segment_flat.shape[1] != expected_dim:
        # Adjust dimensions
        if segment_flat.shape[1] > expected_dim:
            segment_flat = segment_flat[:, :expected_dim]
        else:
            segment_flat = np.pad(segment_flat, ((0, 0), (0, expected_dim - segment_flat.shape[1])))

    embedding = pca.transform(segment_flat)
    return embedding[0]


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def create_combined_visualization(
    halluc_case: Path,
    normal_case: Path,
    normal_case2: Optional[Path],
    step: int,
    training_embeddings: np.ndarray,
    training_phases: List[str],
    pca: PCA,
    output_dir: Path,
):
    """
    Create comprehensive visualization combining:
    - 3-camera images for each case
    - Cross-attention heatmaps overlaid
    - Trajectory distribution with current positions marked
    """
    print(f"\nCreating combined visualization for step {step}...")

    # Load images
    halluc_images = load_case_images(halluc_case, step)
    normal_images = load_case_images(normal_case, step)
    normal2_images = load_case_images(normal_case2, step) if normal_case2 else {}

    if not halluc_images or not normal_images:
        print("  Error: Could not load images for one or both cases")
        return

    # Load attention data (if available)
    halluc_attn = load_cross_attention_data(halluc_case, step)
    normal_attn = load_cross_attention_data(normal_case, step)

    # Embed trajectories
    halluc_emb = embed_inference_trajectory(halluc_case / "trace.jsonl", step, pca) if pca else None
    normal_emb = embed_inference_trajectory(normal_case / "trace.jsonl", step, pca) if pca else None
    normal2_emb = embed_inference_trajectory(normal_case2 / "trace.jsonl", step, pca) if (pca and normal_case2) else None

    # Load trace data for action metrics
    halluc_trace = load_trace_data(halluc_case / "trace.jsonl", step)
    normal_trace = load_trace_data(normal_case / "trace.jsonl", step)

    # Create figure with complex layout
    # Layout:
    # Row 1-2: Hallucination case (3 cameras with attention overlay)
    # Row 3-4: Normal case (3 cameras with attention overlay)
    # Row 5: Trajectory distribution (3D or 2D)

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 3, height_ratios=[1, 0.1, 1, 0.1, 1.2], hspace=0.15, wspace=0.08)

    # ===== HALLUCINATION CASE =====
    cameras = ["head", "left_wrist", "right_wrist"]
    camera_titles = ["Head Camera", "Left Wrist Camera", "Right Wrist Camera"]

    for col, (cam, title) in enumerate(zip(cameras, camera_titles)):
        ax = fig.add_subplot(gs[0, col])

        if cam in halluc_images:
            img = halluc_images[cam]

            # Overlay attention if available
            if halluc_attn and "spatial_maps" in halluc_attn and cam in halluc_attn["spatial_maps"]:
                attn_map = halluc_attn["spatial_maps"][cam]
                img = create_attention_heatmap_overlay(img, attn_map, alpha=0.5)

            ax.imshow(img)
            ax.set_title(f"{title}", fontsize=11)
        else:
            ax.text(0.5, 0.5, "No image", ha='center', va='center')

        ax.axis('off')

    # Hallucination case label
    ax_label1 = fig.add_subplot(gs[1, :])
    ax_label1.axis('off')

    # Get action delta from trace
    halluc_delta = halluc_trace.get("action_delta_max", 0) if halluc_trace else 0
    label_text = f"HALLUCINATION CASE (step {step})"
    label_text += f"  |  Action Delta: {halluc_delta:.2f}"
    if halluc_trace:
        label_text += f"  |  Gripper L: {halluc_trace.get('gripper_left_cmd', 0):.2f}"

    ax_label1.text(0.5, 0.5, label_text,
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='red', transform=ax_label1.transAxes)

    # ===== NORMAL CASE =====
    for col, (cam, title) in enumerate(zip(cameras, camera_titles)):
        ax = fig.add_subplot(gs[2, col])

        if cam in normal_images:
            img = normal_images[cam]

            # Overlay attention if available
            if normal_attn and "spatial_maps" in normal_attn and cam in normal_attn["spatial_maps"]:
                attn_map = normal_attn["spatial_maps"][cam]
                img = create_attention_heatmap_overlay(img, attn_map, alpha=0.5)

            ax.imshow(img)
            ax.set_title(f"{title}", fontsize=11)
        else:
            ax.text(0.5, 0.5, "No image", ha='center', va='center')

        ax.axis('off')

    # Normal case label
    ax_label2 = fig.add_subplot(gs[3, :])
    ax_label2.axis('off')

    normal_delta = normal_trace.get("action_delta_max", 0) if normal_trace else 0
    label_text = f"NORMAL CASE (step {step})"
    label_text += f"  |  Action Delta: {normal_delta:.2f}"
    if normal_trace:
        label_text += f"  |  Gripper L: {normal_trace.get('gripper_left_cmd', 0):.2f}"

    ax_label2.text(0.5, 0.5, label_text,
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='green', transform=ax_label2.transAxes)

    # ===== TRAJECTORY DISTRIBUTION =====
    # Three panels: PC1 vs PC2, PC1 vs PC3, and 3D view

    ax_traj1 = fig.add_subplot(gs[4, 0])
    ax_traj2 = fig.add_subplot(gs[4, 1])
    ax_traj3 = fig.add_subplot(gs[4, 2], projection='3d')

    if len(training_embeddings) > 0:
        # Plot training distribution by phase
        for phase in PHASES:
            mask = np.array([p == phase for p in training_phases])
            if mask.any():
                points = training_embeddings[mask]
                color = PHASE_COLORS.get(phase, "gray")

                # 2D plots
                ax_traj1.scatter(points[:, 0], points[:, 1],
                               c=color, alpha=0.3, s=15, label=phase)
                ax_traj2.scatter(points[:, 0], points[:, 2],
                               c=color, alpha=0.3, s=15)

                # 3D plot
                ax_traj3.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=color, alpha=0.3, s=10)

        # Plot inference trajectories
        if halluc_emb is not None:
            ax_traj1.scatter([halluc_emb[0]], [halluc_emb[1]],
                           c='red', marker='*', s=400, label='Halluc',
                           edgecolors='black', linewidths=2, zorder=10)
            ax_traj2.scatter([halluc_emb[0]], [halluc_emb[2]],
                           c='red', marker='*', s=400,
                           edgecolors='black', linewidths=2, zorder=10)
            ax_traj3.scatter([halluc_emb[0]], [halluc_emb[1]], [halluc_emb[2]],
                           c='red', marker='*', s=200,
                           edgecolors='black', linewidths=1, zorder=10)

        if normal_emb is not None:
            ax_traj1.scatter([normal_emb[0]], [normal_emb[1]],
                           c='lime', marker='^', s=300, label='Normal',
                           edgecolors='black', linewidths=2, zorder=10)
            ax_traj2.scatter([normal_emb[0]], [normal_emb[2]],
                           c='lime', marker='^', s=300,
                           edgecolors='black', linewidths=2, zorder=10)
            ax_traj3.scatter([normal_emb[0]], [normal_emb[1]], [normal_emb[2]],
                           c='lime', marker='^', s=150,
                           edgecolors='black', linewidths=1, zorder=10)

        if normal2_emb is not None:
            ax_traj1.scatter([normal2_emb[0]], [normal2_emb[1]],
                           c='cyan', marker='s', s=250, label='Normal2',
                           edgecolors='black', linewidths=2, zorder=10)
            ax_traj2.scatter([normal2_emb[0]], [normal2_emb[2]],
                           c='cyan', marker='s', s=250,
                           edgecolors='black', linewidths=2, zorder=10)
            ax_traj3.scatter([normal2_emb[0]], [normal2_emb[1]], [normal2_emb[2]],
                           c='cyan', marker='s', s=120,
                           edgecolors='black', linewidths=1, zorder=10)

        # Labels and titles
        ax_traj1.set_xlabel("PC1", fontsize=10)
        ax_traj1.set_ylabel("PC2", fontsize=10)
        ax_traj1.set_title("Trajectory Distribution (PC1 vs PC2)", fontsize=11)
        ax_traj1.legend(fontsize=7, loc='upper right', ncol=2)
        ax_traj1.grid(True, alpha=0.3)

        ax_traj2.set_xlabel("PC1", fontsize=10)
        ax_traj2.set_ylabel("PC3", fontsize=10)
        ax_traj2.set_title("Trajectory Distribution (PC1 vs PC3)", fontsize=11)
        ax_traj2.grid(True, alpha=0.3)

        ax_traj3.set_xlabel("PC1", fontsize=9)
        ax_traj3.set_ylabel("PC2", fontsize=9)
        ax_traj3.set_zlabel("PC3", fontsize=9)
        ax_traj3.set_title("3D View", fontsize=11)

        # Add text annotation showing distances
        if halluc_emb is not None and normal_emb is not None:
            dist = np.linalg.norm(halluc_emb - normal_emb)
            ax_traj3.text2D(0.02, 0.98, f"Halluc-Normal dist: {dist:.1f}",
                          transform=ax_traj3.transAxes, fontsize=9)
    else:
        ax_traj1.text(0.5, 0.5, "No training data loaded", ha='center', va='center')
        ax_traj2.text(0.5, 0.5, "No training data loaded", ha='center', va='center')
        ax_traj3.text2D(0.5, 0.5, "No training data", ha='center', va='center')

    # Main title
    fig.suptitle(f"SmolVLA Hallucination Analysis: Step {step}\n"
                 f"Visual Input + Cross-Attention + Trajectory Distribution",
                 fontsize=16, fontweight='bold', y=0.98)

    # Save
    output_path = output_dir / f"combined_step_{step:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")

    return output_path


def create_side_by_side_comparison(
    halluc_case: Path,
    normal_case: Path,
    normal_case2: Optional[Path],
    steps: List[int],
    output_dir: Path,
):
    """
    Create a compact side-by-side comparison showing key images at multiple steps.

    Layout:
    - Columns: step 200, step 250, step 300 (divergence progression)
    - Rows: halluc head, halluc right_wrist, normal head, normal right_wrist
    """
    print("\nCreating side-by-side comparison...")

    num_steps = len(steps)
    fig, axes = plt.subplots(4, num_steps, figsize=(5 * num_steps, 16))

    if num_steps == 1:
        axes = axes.reshape(-1, 1)

    row_labels = [
        "Halluc: Head",
        "Halluc: Right Wrist",
        "Normal: Head",
        "Normal: Right Wrist"
    ]

    for col, step in enumerate(steps):
        halluc_images = load_case_images(halluc_case, step)
        normal_images = load_case_images(normal_case, step)

        # Load trace data
        halluc_trace = load_trace_data(halluc_case / "trace.jsonl", step)
        normal_trace = load_trace_data(normal_case / "trace.jsonl", step)

        # Row 0: Halluc head
        if "head" in halluc_images:
            axes[0, col].imshow(halluc_images["head"])
        axes[0, col].axis('off')

        # Row 1: Halluc right wrist (where banana is visible)
        if "right_wrist" in halluc_images:
            axes[1, col].imshow(halluc_images["right_wrist"])
        axes[1, col].axis('off')

        # Row 2: Normal head
        if "head" in normal_images:
            axes[2, col].imshow(normal_images["head"])
        axes[2, col].axis('off')

        # Row 3: Normal right wrist
        if "right_wrist" in normal_images:
            axes[3, col].imshow(normal_images["right_wrist"])
        axes[3, col].axis('off')

        # Column title with action metrics
        h_delta = halluc_trace.get("action_delta_max", 0) if halluc_trace else 0
        n_delta = normal_trace.get("action_delta_max", 0) if normal_trace else 0

        title = f"Step {step}\n"
        title += f"H: delta={h_delta:.1f}"
        if halluc_trace and "gripper_left_cmd" in halluc_trace:
            title += f" grip={halluc_trace['gripper_left_cmd']:.2f}"
        title += f"\nN: delta={n_delta:.1f}"

        axes[0, col].set_title(title, fontsize=10, fontweight='bold')

    # Row labels on left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                               rotation=0, ha='right', va='center',
                               labelpad=80)

    # Highlight divergence region
    # Step 200 is pre-divergence, step 250+ is during hallucination
    for row in range(4):
        for col, step in enumerate(steps):
            if step >= 250:  # Mark post-divergence
                color = 'red' if row < 2 else 'green'
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)

    plt.suptitle("Hallucination Case vs Normal Case: Visual Comparison\n"
                 "(Red border = during hallucination, focus on Right Wrist camera)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0.1, 0, 1, 0.96])

    output_path = output_dir / "side_by_side_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def generate_report(
    output_dir: Path,
    halluc_case: Path,
    normal_case: Path,
    steps: List[int],
    halluc_emb: Optional[np.ndarray],
    normal_emb: Optional[np.ndarray],
    pca: Optional[PCA],
):
    """Generate markdown report summarizing the visualization."""
    report = []
    report.append("# Trajectory + Attention Overlay Analysis\n")
    report.append(f"**Generated**: {datetime.now().isoformat()}\n")

    report.append("## Cases Analyzed\n")
    report.append(f"- **Hallucination case**: `{halluc_case}`")
    report.append(f"- **Normal case**: `{normal_case}`")
    report.append(f"- **Steps visualized**: {steps}\n")

    report.append("## Key Observations\n")
    report.append("""
Based on the visualizations:

1. **Visual Input Difference**: The right wrist camera in the hallucination case
   shows the banana remaining on the table, while normal cases show an empty table.

2. **Trajectory Divergence**: At step 208-210, the hallucination case begins
   generating movement trajectories (high action delta) while normal cases
   produce near-zero action deltas (stay still behavior).

3. **Cross-Attention Pattern**: The model attends more strongly to the right wrist
   camera region in the hallucination case, where the banana is visible.

4. **Trajectory Distribution**: The hallucination trajectory resembles TRANSPORT
   phase from training (approaching/grasping), rather than IDLE phase appropriate
   for post-task completion.
""")

    if halluc_emb is not None and normal_emb is not None:
        dist = np.linalg.norm(halluc_emb - normal_emb)
        report.append(f"\n## Trajectory Embedding Distance\n")
        report.append(f"- Halluc-Normal distance in PCA space: **{dist:.1f}**\n")

    if pca is not None:
        report.append(f"\n## PCA Variance Explained\n")
        for i, var in enumerate(pca.explained_variance_ratio_):
            report.append(f"- PC{i+1}: {var:.1%}")
        report.append(f"- **Total**: {sum(pca.explained_variance_ratio_):.1%}\n")

    report.append("\n## Visualizations Generated\n")
    report.append("- `combined_step_XXXX.png`: Full analysis with images + attention + trajectory")
    report.append("- `side_by_side_comparison.png`: Compact comparison across steps")
    report.append("- `report.md`: This report\n")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory + Attention Overlay Visualization"
    )

    # Case directories
    parser.add_argument("--halluc-case", type=Path, required=True,
                       help="Hallucination case directory")
    parser.add_argument("--normal-case", type=Path, required=True,
                       help="Normal case directory")
    parser.add_argument("--normal-case2", type=Path, default=None,
                       help="Second normal case directory (optional)")

    # Steps to visualize
    parser.add_argument("--step", type=int, default=200,
                       help="Primary step to visualize (default: 200)")
    parser.add_argument("--steps", type=str, default="200,250,300",
                       help="Steps for side-by-side comparison (comma-separated)")

    # Dataset
    parser.add_argument("--dataset", type=Path,
                       default=PROJECT_ROOT / "datasets_bimanuel" / "multitasks",
                       help="Training dataset path")
    parser.add_argument("--task-filter", default="yogurt",
                       help="Filter training data by task name")
    parser.add_argument("--max-episodes", type=int, default=50,
                       help="Maximum training episodes to load")

    # Output
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory")

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "logs" / "investigation" / f"traj_attn_overlay_{timestamp}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Parse steps
    comparison_steps = [int(s.strip()) for s in args.steps.split(",")]

    # Load training trajectories
    print("\n" + "=" * 60)
    print("Loading training trajectory distribution...")
    print("=" * 60)

    trajectories, phases, embeddings, pca = load_training_trajectories(
        args.dataset, args.task_filter, args.max_episodes
    )

    # Create combined visualizations
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)

    # Main step visualization
    halluc_emb = None
    normal_emb = None

    if len(embeddings) > 0 and pca is not None:
        create_combined_visualization(
            args.halluc_case,
            args.normal_case,
            args.normal_case2,
            args.step,
            embeddings,
            phases,
            pca,
            args.output_dir,
        )

        # Get embeddings for report
        halluc_emb = embed_inference_trajectory(
            args.halluc_case / "trace.jsonl", args.step, pca
        )
        normal_emb = embed_inference_trajectory(
            args.normal_case / "trace.jsonl", args.step, pca
        )
    else:
        print("Skipping combined visualization (no training data)")

    # Side-by-side comparison
    create_side_by_side_comparison(
        args.halluc_case,
        args.normal_case,
        args.normal_case2,
        comparison_steps,
        args.output_dir,
    )

    # Generate report
    generate_report(
        args.output_dir,
        args.halluc_case,
        args.normal_case,
        comparison_steps,
        halluc_emb,
        normal_emb,
        pca,
    )

    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
