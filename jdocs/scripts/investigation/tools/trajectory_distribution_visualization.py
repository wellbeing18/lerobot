#!/usr/bin/env python3
"""
Trajectory Distribution Visualization Tool for SmolVLA Hallucination Investigation.

Visualizes the space of trajectories from training data and shows where
hallucination and normal case trajectories lie in this distribution.

This tool helps answer:
1. Does halluc trajectory fall in a valid training cluster?
2. Which training phase does the post-completion halluc resemble?
3. Is halluc trajectory out-of-distribution?

Research References:
- 3D Diffusion Policy (DP3): PCA visualization of trajectory space
- Motion2Vec: t-SNE/PCA for action segmentation embeddings
- FlowPolicy: PCA fit across all positions, timesteps, seeds

Usage:
    python trajectory_distribution_visualization.py \
        --dataset datasets_bimanuel/multitasks \
        --task-filter "yogurt" \
        --halluc-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
        --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
        --output-dir outputs/trajectory_distribution
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install scikit-learn")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available, will use matplotlib only")

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS
# ============================================================================

# Trajectory phases for bimanual pick-and-place
PHASES = ["APPROACH", "GRIP", "TRANSPORT", "RELEASE", "IDLE", "UNKNOWN"]

# Key joints for analysis (bimanual)
LEFT_ARM_JOINTS = list(range(0, 6))
LEFT_GRIPPER = 6
RIGHT_ARM_JOINTS = list(range(7, 13))
RIGHT_GRIPPER = 13

# Phase detection thresholds
VELOCITY_THRESHOLD = 0.05
GRIPPER_CLOSED_THRESHOLD = 0.3
GRIPPER_OPEN_THRESHOLD = 0.7


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrajectorySegment:
    """A segment of a trajectory with phase label."""
    episode_id: int
    start_step: int
    end_step: int
    phase: str
    actions: np.ndarray  # [num_steps, action_dim]
    embedding: Optional[np.ndarray] = None  # PCA embedding


@dataclass
class TrajectoryAnalysis:
    """Complete analysis of trajectory distribution."""
    num_episodes: int
    num_segments: int
    total_steps: int

    # Phase distribution
    phase_counts: Dict[str, int]
    phase_avg_duration: Dict[str, float]

    # PCA info
    pca_variance_explained: List[float]

    # Embeddings
    segment_embeddings: np.ndarray  # [num_segments, 3]
    segment_phases: List[str]

    # Inference case embeddings
    halluc_embedding: Optional[np.ndarray] = None
    normal_embedding: Optional[np.ndarray] = None

    # Nearest neighbor analysis
    halluc_nearest_phase: Optional[str] = None
    halluc_nearest_distance: Optional[float] = None
    normal_nearest_phase: Optional[str] = None
    normal_nearest_distance: Optional[float] = None


# ============================================================================
# PHASE DETECTION
# ============================================================================

def compute_velocity(actions: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute smoothed velocity from action sequence."""
    if len(actions) < window:
        return np.zeros(len(actions))

    velocities = np.diff(actions, axis=0)
    velocities = np.vstack([velocities[0], velocities])

    # Smooth with moving average
    smoothed = np.convolve(np.linalg.norm(velocities, axis=1),
                          np.ones(window)/window, mode='same')
    return smoothed


def detect_phase(actions: np.ndarray, step: int, velocities: np.ndarray,
                 left_gripper_history: List[float], right_gripper_history: List[float]) -> str:
    """Detect trajectory phase based on gripper state and velocity."""
    if len(actions) == 0:
        return "UNKNOWN"

    # Current gripper states
    left_gripper = actions[step, LEFT_GRIPPER] if actions.shape[1] > LEFT_GRIPPER else 0.5
    right_gripper = actions[step, RIGHT_GRIPPER] if actions.shape[1] > RIGHT_GRIPPER else 0.5

    # Velocity
    vel = velocities[step] if step < len(velocities) else 0

    # Check gripper transitions
    prev_left = left_gripper_history[-1] if left_gripper_history else left_gripper
    prev_right = right_gripper_history[-1] if right_gripper_history else right_gripper

    # IDLE: Low velocity and stable gripper
    if vel < VELOCITY_THRESHOLD:
        if len(velocities) > step + 10:
            future_vel = np.mean(velocities[step:step+10])
            if future_vel < VELOCITY_THRESHOLD:
                return "IDLE"

    # GRIP: Gripper closing (high to low)
    if prev_left > GRIPPER_OPEN_THRESHOLD and left_gripper < GRIPPER_CLOSED_THRESHOLD:
        return "GRIP"
    if prev_right > GRIPPER_OPEN_THRESHOLD and right_gripper < GRIPPER_CLOSED_THRESHOLD:
        return "GRIP"

    # RELEASE: Gripper opening (low to high)
    if prev_left < GRIPPER_CLOSED_THRESHOLD and left_gripper > GRIPPER_OPEN_THRESHOLD:
        return "RELEASE"
    if prev_right < GRIPPER_CLOSED_THRESHOLD and right_gripper > GRIPPER_OPEN_THRESHOLD:
        return "RELEASE"

    # TRANSPORT: Moving with gripper closed
    if left_gripper < GRIPPER_CLOSED_THRESHOLD or right_gripper < GRIPPER_CLOSED_THRESHOLD:
        if vel > VELOCITY_THRESHOLD:
            return "TRANSPORT"

    # APPROACH: Moving with gripper open
    if vel > VELOCITY_THRESHOLD:
        return "APPROACH"

    return "UNKNOWN"


def segment_trajectory(actions: np.ndarray, episode_id: int,
                      window_size: int = 50) -> List[TrajectorySegment]:
    """Segment a trajectory into phases."""
    if len(actions) < window_size:
        return [TrajectorySegment(
            episode_id=episode_id,
            start_step=0,
            end_step=len(actions),
            phase="UNKNOWN",
            actions=actions
        )]

    velocities = compute_velocity(actions)
    segments = []
    current_phase = None
    segment_start = 0
    left_gripper_history = []
    right_gripper_history = []

    for step in range(len(actions)):
        phase = detect_phase(actions, step, velocities,
                            left_gripper_history, right_gripper_history)

        # Track gripper history
        if actions.shape[1] > LEFT_GRIPPER:
            left_gripper_history.append(actions[step, LEFT_GRIPPER])
            if len(left_gripper_history) > 10:
                left_gripper_history.pop(0)
        if actions.shape[1] > RIGHT_GRIPPER:
            right_gripper_history.append(actions[step, RIGHT_GRIPPER])
            if len(right_gripper_history) > 10:
                right_gripper_history.pop(0)

        if current_phase is None:
            current_phase = phase
            continue

        # Phase transition
        if phase != current_phase and step - segment_start >= 10:
            segments.append(TrajectorySegment(
                episode_id=episode_id,
                start_step=segment_start,
                end_step=step,
                phase=current_phase,
                actions=actions[segment_start:step]
            ))
            segment_start = step
            current_phase = phase

    # Final segment
    if segment_start < len(actions):
        segments.append(TrajectorySegment(
            episode_id=episode_id,
            start_step=segment_start,
            end_step=len(actions),
            phase=current_phase or "UNKNOWN",
            actions=actions[segment_start:]
        ))

    return segments


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset_trajectories(dataset_path: str, task_filter: Optional[str] = None,
                             max_episodes: int = 100) -> Tuple[List[np.ndarray], List[str]]:
    """Load trajectories from LeRobot dataset."""
    dataset_path = Path(dataset_path)
    trajectories = []
    episode_tasks = []

    # Try to load as LeRobot dataset
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION

        print(f"Loading dataset from {dataset_path}...")
        dataset = LeRobotDataset(str(dataset_path))

        # Get episode info
        episode_data_index = dataset.episode_data_index
        num_episodes = len(episode_data_index["from"])

        print(f"Found {num_episodes} episodes")

        loaded_count = 0
        for ep_idx in range(min(num_episodes, max_episodes * 2)):  # Load extra in case of filtering
            if loaded_count >= max_episodes:
                break

            from_idx = episode_data_index["from"][ep_idx].item()
            to_idx = episode_data_index["to"][ep_idx].item()

            # Get episode task (if available)
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
                trajectories.append(np.array(actions))
                episode_tasks.append(task)
                loaded_count += 1

        print(f"Loaded {len(trajectories)} trajectories (filtered by: {task_filter})")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load from parquet files...")

        # Fallback: try to load parquet files directly
        parquet_files = list(dataset_path.glob("**/*.parquet"))
        if parquet_files:
            import pandas as pd
            for pf in parquet_files[:max_episodes]:
                try:
                    df = pd.read_parquet(pf)
                    if 'action' in df.columns:
                        actions = np.array(df['action'].tolist())
                        trajectories.append(actions)
                        episode_tasks.append("")
                except Exception as e2:
                    print(f"  Error loading {pf}: {e2}")

    return trajectories, episode_tasks


def load_trace_trajectory(trace_path: str, step_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load trajectory from inference trace file."""
    trace_path = Path(trace_path)
    actions = []

    with open(trace_path, "r") as f:
        for line in f:
            data = json.loads(line)
            step = data["step"]

            if step_range:
                if step < step_range[0] or step >= step_range[1]:
                    continue

            if "action_raw" in data:
                actions.append(data["action_raw"])
            elif "action_final" in data:
                actions.append(data["action_final"])

    return np.array(actions)


# ============================================================================
# EMBEDDING AND ANALYSIS
# ============================================================================

def embed_segments(segments: List[TrajectorySegment], target_length: int = 50,
                  n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """Embed trajectory segments using PCA."""
    # Normalize segment lengths by resampling
    embedded_segs = []
    for seg in segments:
        if len(seg.actions) < 2:
            continue

        # Resample to target_length
        indices = np.linspace(0, len(seg.actions) - 1, target_length).astype(int)
        resampled = seg.actions[indices]

        # Flatten
        embedded_segs.append(resampled.flatten())

    if not embedded_segs:
        return np.array([]), None

    X = np.array(embedded_segs)

    # Apply PCA
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    embeddings = pca.fit_transform(X)

    return embeddings, pca


def find_nearest_segment(embedding: np.ndarray, segment_embeddings: np.ndarray,
                        segment_phases: List[str]) -> Tuple[str, float, int]:
    """Find nearest training segment to a given embedding."""
    distances = np.linalg.norm(segment_embeddings - embedding, axis=1)
    nearest_idx = np.argmin(distances)
    return segment_phases[nearest_idx], distances[nearest_idx], nearest_idx


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_3d_distribution_matplotlib(analysis: TrajectoryAnalysis, output_dir: Path):
    """Create 3D visualization using matplotlib."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for phases
    phase_colors = {
        "APPROACH": "blue",
        "GRIP": "green",
        "TRANSPORT": "orange",
        "RELEASE": "purple",
        "IDLE": "gray",
        "UNKNOWN": "black"
    }

    # Plot training segments
    for phase in PHASES:
        mask = [p == phase for p in analysis.segment_phases]
        if any(mask):
            points = analysis.segment_embeddings[mask]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=phase_colors.get(phase, "black"),
                      label=f"{phase} ({sum(mask)})",
                      alpha=0.5, s=20)

    # Plot inference cases
    if analysis.halluc_embedding is not None:
        ax.scatter([analysis.halluc_embedding[0]], [analysis.halluc_embedding[1]],
                  [analysis.halluc_embedding[2]],
                  c='red', marker='*', s=300, label='Hallucination', edgecolors='black')

    if analysis.normal_embedding is not None:
        ax.scatter([analysis.normal_embedding[0]], [analysis.normal_embedding[1]],
                  [analysis.normal_embedding[2]],
                  c='lime', marker='^', s=300, label='Normal', edgecolors='black')

    ax.set_xlabel(f"PC1 ({analysis.pca_variance_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({analysis.pca_variance_explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({analysis.pca_variance_explained[2]:.1%})")
    ax.set_title("Trajectory Distribution (Training Data)")
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "3d_trajectory_distribution.png", dpi=150)
    plt.close()


def visualize_3d_distribution_plotly(analysis: TrajectoryAnalysis, output_dir: Path):
    """Create interactive 3D visualization using plotly."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return

    # Prepare data
    data = []

    # Training segments
    for phase in PHASES:
        mask = [p == phase for p in analysis.segment_phases]
        if any(mask):
            points = analysis.segment_embeddings[mask]
            trace = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                name=f"{phase} ({sum(mask)})",
                marker=dict(size=4, opacity=0.6)
            )
            data.append(trace)

    # Hallucination case
    if analysis.halluc_embedding is not None:
        data.append(go.Scatter3d(
            x=[analysis.halluc_embedding[0]],
            y=[analysis.halluc_embedding[1]],
            z=[analysis.halluc_embedding[2]],
            mode='markers',
            name='Hallucination',
            marker=dict(size=15, color='red', symbol='diamond')
        ))

    # Normal case
    if analysis.normal_embedding is not None:
        data.append(go.Scatter3d(
            x=[analysis.normal_embedding[0]],
            y=[analysis.normal_embedding[1]],
            z=[analysis.normal_embedding[2]],
            mode='markers',
            name='Normal',
            marker=dict(size=15, color='lime', symbol='cross')
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title="3D Trajectory Distribution (Interactive)",
        scene=dict(
            xaxis_title=f"PC1 ({analysis.pca_variance_explained[0]:.1%})",
            yaxis_title=f"PC2 ({analysis.pca_variance_explained[1]:.1%})",
            zaxis_title=f"PC3 ({analysis.pca_variance_explained[2]:.1%})"
        )
    )

    fig.write_html(str(output_dir / "3d_trajectory_distribution.html"))


def visualize_2d_projections(analysis: TrajectoryAnalysis, output_dir: Path):
    """Create 2D projections for easier viewing."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    phase_colors = {
        "APPROACH": "blue",
        "GRIP": "green",
        "TRANSPORT": "orange",
        "RELEASE": "purple",
        "IDLE": "gray",
        "UNKNOWN": "black"
    }

    projections = [
        (0, 1, "PC1 vs PC2"),
        (0, 2, "PC1 vs PC3"),
        (1, 2, "PC2 vs PC3")
    ]

    for ax, (dim1, dim2, title) in zip(axes, projections):
        # Plot training segments
        for phase in PHASES:
            mask = [p == phase for p in analysis.segment_phases]
            if any(mask):
                points = analysis.segment_embeddings[mask]
                ax.scatter(points[:, dim1], points[:, dim2],
                          c=phase_colors.get(phase, "black"),
                          label=phase, alpha=0.5, s=20)

        # Plot inference cases
        if analysis.halluc_embedding is not None:
            ax.scatter([analysis.halluc_embedding[dim1]], [analysis.halluc_embedding[dim2]],
                      c='red', marker='*', s=200, label='Halluc', edgecolors='black', zorder=10)

        if analysis.normal_embedding is not None:
            ax.scatter([analysis.normal_embedding[dim1]], [analysis.normal_embedding[dim2]],
                      c='lime', marker='^', s=200, label='Normal', edgecolors='black', zorder=10)

        ax.set_title(title)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "2d_projections.png", dpi=150)
    plt.close()


def visualize_phase_distribution(analysis: TrajectoryAnalysis, output_dir: Path):
    """Visualize phase distribution in training data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Phase counts
    ax1 = axes[0]
    phases = list(analysis.phase_counts.keys())
    counts = [analysis.phase_counts[p] for p in phases]
    colors = ['blue', 'green', 'orange', 'purple', 'gray', 'black']
    ax1.bar(phases, counts, color=colors[:len(phases)])
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Number of Segments")
    ax1.set_title("Phase Distribution")
    ax1.tick_params(axis='x', rotation=45)

    # Average duration
    ax2 = axes[1]
    durations = [analysis.phase_avg_duration.get(p, 0) for p in phases]
    ax2.bar(phases, durations, color=colors[:len(phases)])
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Average Duration (steps)")
    ax2.set_title("Average Phase Duration")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "phase_distribution.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analysis: TrajectoryAnalysis, output_dir: Path,
                   halluc_trace: Optional[str] = None, normal_trace: Optional[str] = None):
    """Generate markdown report."""
    report = []
    report.append("# Trajectory Distribution Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Dataset Summary")
    report.append(f"- Number of episodes analyzed: {analysis.num_episodes}")
    report.append(f"- Total trajectory segments: {analysis.num_segments}")
    report.append(f"- Total steps: {analysis.total_steps}")

    report.append("\n## Phase Distribution")
    report.append("\n| Phase | Count | Avg Duration |")
    report.append("|-------|-------|--------------|")
    for phase in PHASES:
        count = analysis.phase_counts.get(phase, 0)
        duration = analysis.phase_avg_duration.get(phase, 0)
        report.append(f"| {phase} | {count} | {duration:.1f} steps |")

    report.append("\n## PCA Analysis")
    report.append(f"\n**Variance explained by components:**")
    for i, var in enumerate(analysis.pca_variance_explained):
        report.append(f"- PC{i+1}: {var:.1%}")
    report.append(f"- **Total**: {sum(analysis.pca_variance_explained):.1%}")

    if halluc_trace or normal_trace:
        report.append("\n## Inference Case Analysis")

        if analysis.halluc_embedding is not None:
            report.append(f"\n### Hallucination Case")
            report.append(f"- Trace: `{halluc_trace}`")
            report.append(f"- **Nearest training phase**: {analysis.halluc_nearest_phase}")
            report.append(f"- Distance to nearest: {analysis.halluc_nearest_distance:.3f}")
            report.append(f"- Embedding: [{', '.join([f'{x:.3f}' for x in analysis.halluc_embedding])}]")

        if analysis.normal_embedding is not None:
            report.append(f"\n### Normal Case")
            report.append(f"- Trace: `{normal_trace}`")
            report.append(f"- **Nearest training phase**: {analysis.normal_nearest_phase}")
            report.append(f"- Distance to nearest: {analysis.normal_nearest_distance:.3f}")
            report.append(f"- Embedding: [{', '.join([f'{x:.3f}' for x in analysis.normal_embedding])}]")

    report.append("\n## Key Findings")

    if analysis.halluc_embedding is not None and analysis.normal_embedding is not None:
        # Distance between cases
        case_distance = np.linalg.norm(analysis.halluc_embedding - analysis.normal_embedding)
        report.append(f"\n- **Distance between halluc and normal**: {case_distance:.3f}")

        # Compare to training distribution
        avg_training_distance = np.mean(np.linalg.norm(
            analysis.segment_embeddings - analysis.segment_embeddings.mean(axis=0), axis=1
        ))
        report.append(f"- Average training trajectory distance from centroid: {avg_training_distance:.3f}")

        if analysis.halluc_nearest_distance > avg_training_distance * 1.5:
            report.append(f"- **Hallucination trajectory appears OUT-OF-DISTRIBUTION**")
            report.append(f"  (distance {analysis.halluc_nearest_distance:.3f} > 1.5x avg {avg_training_distance:.3f})")
        else:
            report.append(f"- Hallucination trajectory is within training distribution")
            report.append(f"  (nearest phase: {analysis.halluc_nearest_phase})")

        if analysis.halluc_nearest_phase != "IDLE":
            report.append(f"\n- **Critical**: Hallucination trajectory resembles **{analysis.halluc_nearest_phase}** phase")
            report.append(f"  This suggests the model is generating movement trajectories "
                         f"when it should be generating IDLE trajectories")

    report.append("\n## Visualizations")
    report.append("\n- `3d_trajectory_distribution.png`: 3D PCA visualization (matplotlib)")
    if PLOTLY_AVAILABLE:
        report.append("- `3d_trajectory_distribution.html`: Interactive 3D visualization (plotly)")
    report.append("- `2d_projections.png`: 2D projections of trajectory space")
    report.append("- `phase_distribution.png`: Phase distribution in training data")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Import config for defaults
    from investigation_config import (
        get_output_dir, DATASET_PATH, TASK_FILTER, CASE_HALLUC, CASE_NORMAL_CLEAN, STEP_RANGE
    )

    default_halluc_trace = str(CASE_HALLUC / "trace.jsonl")
    default_normal_trace = str(CASE_NORMAL_CLEAN / "trace.jsonl")
    default_step_range = f"{STEP_RANGE[0]},{STEP_RANGE[1]}"

    parser = argparse.ArgumentParser(description="Visualize trajectory distribution")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                       help="Path to LeRobot dataset")
    parser.add_argument("--task-filter", default=TASK_FILTER,
                       help="Filter episodes by task name")
    parser.add_argument("--halluc-trace", default=default_halluc_trace,
                       help="Path to hallucination trace.jsonl")
    parser.add_argument("--normal-trace", default=default_normal_trace,
                       help="Path to normal trace.jsonl")
    parser.add_argument("--step-range", default=default_step_range,
                       help="Step range for inference traces (e.g., '200,300')")
    parser.add_argument("--max-episodes", type=int, default=100,
                       help="Maximum episodes to load")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: auto-generated with timestamp)")

    args = parser.parse_args()

    # Auto-generate output dir if not specified
    if args.output_dir is None:
        output_dir = get_output_dir("trajectory_dist")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Parse step range
    step_range = None
    if args.step_range:
        parts = args.step_range.split(",")
        step_range = (int(parts[0]), int(parts[1]))

    # Load training trajectories
    print(f"\nLoading training trajectories from {args.dataset}...")
    trajectories, tasks = load_dataset_trajectories(
        args.dataset, args.task_filter, args.max_episodes
    )

    if not trajectories:
        print("ERROR: No trajectories loaded. Check dataset path and task filter.")
        return

    # Segment trajectories
    print("\nSegmenting trajectories into phases...")
    all_segments = []
    for ep_idx, traj in enumerate(trajectories):
        segments = segment_trajectory(traj, ep_idx)
        all_segments.extend(segments)
        if (ep_idx + 1) % 10 == 0:
            print(f"  Processed {ep_idx + 1}/{len(trajectories)} episodes")

    print(f"Total segments: {len(all_segments)}")

    # Embed segments
    print("\nEmbedding segments with PCA...")
    embeddings, pca = embed_segments(all_segments, target_length=50, n_components=3)

    if embeddings.size == 0:
        print("ERROR: No valid embeddings generated")
        return

    # Update segments with embeddings
    for seg, emb in zip(all_segments, embeddings):
        seg.embedding = emb

    # Compute phase statistics
    phase_counts = {}
    phase_durations = {}
    for phase in PHASES:
        phase_segs = [s for s in all_segments if s.phase == phase]
        phase_counts[phase] = len(phase_segs)
        if phase_segs:
            phase_durations[phase] = np.mean([len(s.actions) for s in phase_segs])
        else:
            phase_durations[phase] = 0

    # Process inference traces
    halluc_emb = None
    normal_emb = None
    halluc_nearest_phase = None
    halluc_nearest_dist = None
    normal_nearest_phase = None
    normal_nearest_dist = None

    if args.halluc_trace:
        print(f"\nProcessing hallucination trace: {args.halluc_trace}")
        halluc_traj = load_trace_trajectory(args.halluc_trace, step_range)
        if len(halluc_traj) > 0:
            # Resample and embed
            target_length = 50
            indices = np.linspace(0, len(halluc_traj) - 1, target_length).astype(int)
            resampled = halluc_traj[indices].flatten().reshape(1, -1)
            halluc_emb = pca.transform(resampled)[0]
            halluc_nearest_phase, halluc_nearest_dist, _ = find_nearest_segment(
                halluc_emb, embeddings, [s.phase for s in all_segments]
            )
            print(f"  Embedded. Nearest phase: {halluc_nearest_phase} (dist: {halluc_nearest_dist:.3f})")

    if args.normal_trace:
        print(f"\nProcessing normal trace: {args.normal_trace}")
        normal_traj = load_trace_trajectory(args.normal_trace, step_range)
        if len(normal_traj) > 0:
            target_length = 50
            indices = np.linspace(0, len(normal_traj) - 1, target_length).astype(int)
            resampled = normal_traj[indices].flatten().reshape(1, -1)
            normal_emb = pca.transform(resampled)[0]
            normal_nearest_phase, normal_nearest_dist, _ = find_nearest_segment(
                normal_emb, embeddings, [s.phase for s in all_segments]
            )
            print(f"  Embedded. Nearest phase: {normal_nearest_phase} (dist: {normal_nearest_dist:.3f})")

    # Build analysis
    analysis = TrajectoryAnalysis(
        num_episodes=len(trajectories),
        num_segments=len(all_segments),
        total_steps=sum(len(t) for t in trajectories),
        phase_counts=phase_counts,
        phase_avg_duration=phase_durations,
        pca_variance_explained=[float(x) for x in pca.explained_variance_ratio_],
        segment_embeddings=embeddings,
        segment_phases=[s.phase for s in all_segments],
        halluc_embedding=halluc_emb,
        normal_embedding=normal_emb,
        halluc_nearest_phase=halluc_nearest_phase,
        halluc_nearest_distance=halluc_nearest_dist,
        normal_nearest_phase=normal_nearest_phase,
        normal_nearest_distance=normal_nearest_dist
    )

    # Save analysis data
    analysis_dict = asdict(analysis)
    # Convert numpy arrays to lists for JSON serialization
    analysis_dict["segment_embeddings"] = analysis_dict["segment_embeddings"].tolist()
    if analysis_dict["halluc_embedding"] is not None:
        analysis_dict["halluc_embedding"] = analysis_dict["halluc_embedding"].tolist()
    if analysis_dict["normal_embedding"] is not None:
        analysis_dict["normal_embedding"] = analysis_dict["normal_embedding"].tolist()

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis_dict, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_3d_distribution_matplotlib(analysis, output_dir)
    visualize_3d_distribution_plotly(analysis, output_dir)
    visualize_2d_projections(analysis, output_dir)
    visualize_phase_distribution(analysis, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(analysis, output_dir, args.halluc_trace, args.normal_trace)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
