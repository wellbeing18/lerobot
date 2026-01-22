#!/usr/bin/env python3
"""
UMAP Trajectory Analysis for SmolVLA Hallucination Investigation.

Uses UMAP (Uniform Manifold Approximation and Projection) instead of PCA
to better preserve global structure and identify coverage gaps in the
training trajectory distribution.

Why UMAP over PCA/t-SNE:
- UMAP preserves global structure (distances between clusters are meaningful)
- Better for identifying coverage gaps (empty regions in manifold)
- Computationally efficient for large datasets

Key Questions to Answer:
- Where are the coverage gaps in trajectory space?
- Does hallucination trajectory fall in a void (no training coverage)?
- Is there a "bridge" between TRANSPORT and IDLE that's missing?

Usage:
    python umap_trajectory_analysis.py \
        --dataset datasets_bimanuel/multitasks \
        --task-filter yogurt \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --output-dir outputs/umap_analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
    sys.exit(1)

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
except ImportError:
    print("sklearn not installed. Install with: pip install scikit-learn")
    sys.exit(1)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS
# ============================================================================

# Phase classification thresholds
VELOCITY_IDLE_THRESHOLD = 0.5
VELOCITY_HIGH_THRESHOLD = 2.0

# Phase labels
PHASE_APPROACH = "APPROACH"
PHASE_GRIP = "GRIP"
PHASE_TRANSPORT = "TRANSPORT"
PHASE_RELEASE = "RELEASE"
PHASE_IDLE = "IDLE"

PHASE_COLORS = {
    PHASE_APPROACH: "#1f77b4",  # blue
    PHASE_GRIP: "#ff7f0e",       # orange
    PHASE_TRANSPORT: "#2ca02c",  # green
    PHASE_RELEASE: "#d62728",    # red
    PHASE_IDLE: "#9467bd",       # purple
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CoverageGap:
    """Identified coverage gap in trajectory manifold."""
    center: List[float]  # UMAP coordinates of gap center
    radius: float  # Estimated radius of gap
    nearest_training_distance: float  # Distance to nearest training point
    nearest_phase: str  # Phase of nearest training trajectory
    severity: str  # "minor", "moderate", "severe"


@dataclass
class TrajectoryManifoldAnalysis:
    """Analysis of trajectory distribution using UMAP."""
    num_training_trajectories: int
    num_inference_trajectories: int

    # UMAP parameters
    n_neighbors: int
    min_dist: float
    metric: str

    # Embedding results
    embedding_shape: Tuple[int, int]
    variance_explained: Optional[float]  # Not applicable for UMAP, but kept for reference

    # Phase distribution
    phase_counts: Dict[str, int]
    phase_centroids: Dict[str, List[float]]  # UMAP centroids per phase

    # Coverage gaps
    coverage_gaps: List[CoverageGap]
    gap_density: float  # Gaps per unit area

    # Inference trajectory analysis
    inference_positions: Dict[str, List[float]]  # Case name -> UMAP position
    inference_nearest_phase: Dict[str, str]
    inference_nearest_distance: Dict[str, float]
    inference_in_gap: Dict[str, bool]


# ============================================================================
# TRAJECTORY EXTRACTION
# ============================================================================

def extract_training_trajectories(dataset_path: str, task_filter: str = None,
                                  chunk_size: int = 50) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract trajectories from training dataset.

    Args:
        dataset_path: Path to LeRobot dataset
        task_filter: Optional filter for task names
        chunk_size: Number of timesteps per trajectory chunk

    Returns:
        trajectories: Array of shape (N, chunk_size * action_dim)
        phases: List of phase labels
        episode_indices: Episode index for each trajectory
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset from {dataset_path}...")
    # Load with both repo_id and root for local datasets
    dataset = LeRobotDataset(repo_id=dataset_path, root=dataset_path)

    trajectories = []
    phases = []
    episode_indices = []

    # Get episode info from meta.episodes
    episodes = dataset.meta.episodes
    total_episodes = dataset.meta.total_episodes

    # Get episode boundaries
    from_indices = episodes['dataset_from_index']
    to_indices = episodes['dataset_to_index']
    episode_tasks = episodes['tasks']

    print(f"Processing {total_episodes} episodes...")

    for ep_idx in range(total_episodes):
        # Get episode data
        from_idx = from_indices[ep_idx]
        to_idx = to_indices[ep_idx]

        # Get episode task (if available) for filtering
        task = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""

        # Check task filter if specified
        if task_filter and task_filter.lower() not in task.lower():
            continue

        # Extract actions directly from hf_dataset (avoid video decoding)
        episode_data = dataset.hf_dataset.select(range(from_idx, to_idx))
        episode_actions = []
        for action in episode_data['action']:
            if hasattr(action, 'numpy'):
                action = action.numpy()
            else:
                action = np.array(action)
            episode_actions.append(action)

        if len(episode_actions) < chunk_size:
            continue

        episode_actions = np.array(episode_actions)

        # Split into chunks
        num_chunks = len(episode_actions) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            chunk = episode_actions[start:end]

            # Classify phase based on velocity
            velocities = np.linalg.norm(np.diff(chunk, axis=0), axis=1)
            mean_vel = np.mean(velocities)

            if mean_vel < VELOCITY_IDLE_THRESHOLD:
                phase = PHASE_IDLE
            elif chunk_idx == 0:
                phase = PHASE_APPROACH
            elif chunk_idx == num_chunks - 1:
                phase = PHASE_IDLE
            else:
                phase = PHASE_TRANSPORT

            # Flatten for embedding
            trajectories.append(chunk.flatten())
            phases.append(phase)
            episode_indices.append(ep_idx)

    return np.array(trajectories), phases, episode_indices


def extract_inference_trajectories(case_dirs: List[str], step_range: Tuple[int, int],
                                   chunk_size: int = 50) -> Dict[str, np.ndarray]:
    """Extract trajectories from inference cases.

    Returns dict mapping case_name -> trajectory array
    """
    results = {}

    for case_dir in case_dirs:
        case_path = Path(case_dir)
        trace_path = case_path / "trace.jsonl"

        if not trace_path.exists():
            print(f"Warning: No trace file found at {trace_path}")
            continue

        # Load trace data
        actions = []
        with open(trace_path, "r") as f:
            for line in f:
                data = json.loads(line)
                step = data["step"]
                if step_range[0] <= step < step_range[1]:
                    action = data.get("action", data.get("action_normalized", data.get("action_final")))
                    if action:
                        actions.append(action)

        if len(actions) >= chunk_size:
            actions = np.array(actions[:chunk_size])
            results[case_path.name] = actions.flatten()

    return results


# ============================================================================
# UMAP ANALYSIS
# ============================================================================

def compute_umap_embedding(training_traj: np.ndarray,
                          inference_traj: Dict[str, np.ndarray],
                          n_neighbors: int = 15,
                          min_dist: float = 0.1,
                          metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
    """Compute UMAP embedding for training and inference trajectories.

    Returns:
        training_embedding: (N_train, 2) UMAP coordinates
        inference_embedding: (N_inf, 2) UMAP coordinates
    """
    print(f"Computing UMAP embedding with n_neighbors={n_neighbors}, min_dist={min_dist}...")

    # Fit UMAP on training data
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )

    training_embedding = reducer.fit_transform(training_traj)

    # Transform inference trajectories
    if inference_traj:
        inference_data = np.stack(list(inference_traj.values()))
        inference_embedding = reducer.transform(inference_data)
    else:
        inference_embedding = np.array([])

    return training_embedding, inference_embedding, reducer


def identify_coverage_gaps(embedding: np.ndarray, phases: List[str],
                          grid_resolution: int = 50,
                          density_threshold: float = 0.1) -> List[CoverageGap]:
    """Identify coverage gaps (voids) in the UMAP embedding.

    Uses a grid-based approach to find regions with low point density.
    """
    # Compute bounds
    x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
    y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1

    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    cell_width = (x_max - x_min) / grid_resolution
    cell_height = (y_max - y_min) / grid_resolution

    # Count points in each cell
    density_grid = np.zeros((grid_resolution, grid_resolution))
    for x, y in embedding:
        x_idx = min(int((x - x_min) / cell_width), grid_resolution - 1)
        y_idx = min(int((y - y_min) / cell_height), grid_resolution - 1)
        density_grid[x_idx, y_idx] += 1

    # Normalize to density
    total_points = len(embedding)
    density_grid /= total_points

    # Find low-density regions that are surrounded by higher density
    # (i.e., actual gaps, not just border areas)
    gaps = []
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(embedding)

    for i in range(1, grid_resolution - 1):
        for j in range(1, grid_resolution - 1):
            cell_density = density_grid[i, j]
            neighbor_density = np.mean([
                density_grid[i-1, j], density_grid[i+1, j],
                density_grid[i, j-1], density_grid[i, j+1]
            ])

            # Gap: low density but neighbors have points
            if cell_density < density_threshold and neighbor_density > density_threshold:
                center_x = x_min + (i + 0.5) * cell_width
                center_y = y_min + (j + 0.5) * cell_height
                center = np.array([[center_x, center_y]])

                # Find distance to nearest training point
                dist, idx = nn.kneighbors(center)
                nearest_dist = dist[0, 0]
                nearest_phase = phases[idx[0, 0]]

                # Classify severity
                if nearest_dist > 3.0:
                    severity = "severe"
                elif nearest_dist > 1.5:
                    severity = "moderate"
                else:
                    severity = "minor"

                gaps.append(CoverageGap(
                    center=[center_x, center_y],
                    radius=max(cell_width, cell_height),
                    nearest_training_distance=float(nearest_dist),
                    nearest_phase=nearest_phase,
                    severity=severity
                ))

    return gaps


def analyze_inference_positions(training_embedding: np.ndarray,
                               inference_embedding: np.ndarray,
                               inference_names: List[str],
                               phases: List[str],
                               gaps: List[CoverageGap]) -> Dict:
    """Analyze where inference trajectories fall in the UMAP space."""
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(training_embedding)

    results = {
        "positions": {},
        "nearest_phase": {},
        "nearest_distance": {},
        "in_gap": {}
    }

    for i, name in enumerate(inference_names):
        pos = inference_embedding[i]
        results["positions"][name] = pos.tolist()

        # Find nearest training points
        dists, idxs = nn.kneighbors([pos])
        nearest_phases = [phases[idx] for idx in idxs[0]]

        # Most common phase among neighbors
        from collections import Counter
        phase_counts = Counter(nearest_phases)
        results["nearest_phase"][name] = phase_counts.most_common(1)[0][0]
        results["nearest_distance"][name] = float(dists[0, 0])

        # Check if in a gap
        in_gap = False
        for gap in gaps:
            dist_to_gap = np.sqrt((pos[0] - gap.center[0])**2 + (pos[1] - gap.center[1])**2)
            if dist_to_gap < gap.radius * 2:  # Within 2x gap radius
                in_gap = True
                break
        results["in_gap"][name] = in_gap

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_umap_manifold(training_embedding: np.ndarray,
                           inference_embedding: np.ndarray,
                           phases: List[str],
                           inference_names: List[str],
                           gaps: List[CoverageGap],
                           output_dir: Path):
    """Create comprehensive UMAP visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. Training data colored by phase
    ax1 = axes[0, 0]
    for phase in PHASE_COLORS.keys():
        mask = [p == phase for p in phases]
        if any(mask):
            points = training_embedding[mask]
            ax1.scatter(points[:, 0], points[:, 1], c=PHASE_COLORS[phase],
                       label=phase, alpha=0.5, s=20)
    ax1.set_title("Training Trajectories by Phase")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend()

    # 2. Density heatmap
    ax2 = axes[0, 1]
    ax2.hexbin(training_embedding[:, 0], training_embedding[:, 1],
               gridsize=30, cmap='YlOrRd', mincnt=1)
    ax2.set_title("Training Density (hexbin)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    plt.colorbar(ax2.collections[0], ax=ax2, label='Count')

    # 3. Coverage gaps
    ax3 = axes[1, 0]
    ax3.scatter(training_embedding[:, 0], training_embedding[:, 1],
               c='lightgray', alpha=0.3, s=10, label='Training')

    # Mark gaps
    for gap in gaps:
        color = {'minor': 'yellow', 'moderate': 'orange', 'severe': 'red'}[gap.severity]
        circle = plt.Circle(gap.center, gap.radius, fill=False,
                           color=color, linewidth=2, linestyle='--')
        ax3.add_patch(circle)

    # Mark inference points
    if len(inference_embedding) > 0:
        colors = ['blue', 'green', 'purple', 'cyan']
        markers = ['*', '^', 's', 'D']
        for i, name in enumerate(inference_names):
            ax3.scatter(inference_embedding[i, 0], inference_embedding[i, 1],
                       c=colors[i % len(colors)], marker=markers[i % len(markers)],
                       s=300, edgecolors='black', linewidth=2, label=name[:20])

    ax3.set_title("Coverage Gaps and Inference Positions")
    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")
    ax3.legend(loc='upper right', fontsize=8)

    # 4. Phase-specific analysis
    ax4 = axes[1, 1]

    # Compute phase centroids
    phase_centroids = {}
    for phase in PHASE_COLORS.keys():
        mask = [p == phase for p in phases]
        if any(mask):
            points = training_embedding[mask]
            phase_centroids[phase] = points.mean(axis=0)

    # Plot centroids and connections
    for phase, centroid in phase_centroids.items():
        ax4.scatter(centroid[0], centroid[1], c=PHASE_COLORS[phase],
                   s=500, marker='o', edgecolors='black', linewidth=2, label=phase)

    # Draw transitions between phases
    transitions = [
        (PHASE_APPROACH, PHASE_GRIP),
        (PHASE_GRIP, PHASE_TRANSPORT),
        (PHASE_TRANSPORT, PHASE_RELEASE),
        (PHASE_RELEASE, PHASE_IDLE),
    ]

    for src, dst in transitions:
        if src in phase_centroids and dst in phase_centroids:
            src_pos = phase_centroids[src]
            dst_pos = phase_centroids[dst]
            ax4.annotate('', xy=dst_pos, xytext=src_pos,
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax4.set_title("Phase Centroids and Transitions")
    ax4.set_xlabel("UMAP 1")
    ax4.set_ylabel("UMAP 2")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "umap_manifold.png", dpi=150)
    plt.close()


def visualize_inference_detail(training_embedding: np.ndarray,
                              inference_embedding: np.ndarray,
                              phases: List[str],
                              inference_names: List[str],
                              inference_analysis: Dict,
                              output_dir: Path):
    """Detailed visualization of inference trajectory positions."""
    n_cases = len(inference_names)
    if n_cases == 0:
        return

    fig, axes = plt.subplots(1, n_cases, figsize=(6 * n_cases, 6))
    if n_cases == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, inference_names)):
        # Background: training points by phase
        for phase in PHASE_COLORS.keys():
            mask = [p == phase for p in phases]
            if any(mask):
                points = training_embedding[mask]
                ax.scatter(points[:, 0], points[:, 1], c=PHASE_COLORS[phase],
                          alpha=0.3, s=10)

        # Inference point
        ax.scatter(inference_embedding[idx, 0], inference_embedding[idx, 1],
                  c='red', marker='*', s=500, edgecolors='black', linewidth=2,
                  label='Inference')

        # Draw arrow to nearest training cluster
        nearest_phase = inference_analysis["nearest_phase"][name]
        nearest_dist = inference_analysis["nearest_distance"][name]

        # Annotations
        info = (f"Nearest phase: {nearest_phase}\n"
                f"Distance: {nearest_dist:.2f}\n"
                f"In gap: {inference_analysis['in_gap'][name]}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        ax.set_title(f"{name[:30]}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(output_dir / "inference_detail.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analysis: TrajectoryManifoldAnalysis, output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# UMAP Trajectory Manifold Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Purpose")
    report.append("""
UMAP (Uniform Manifold Approximation and Projection) provides better visualization
of trajectory distribution than PCA because:
- It preserves **global structure** (cluster distances are meaningful)
- It reveals **coverage gaps** (voids in the manifold)
- It shows non-linear relationships between trajectory phases
""")

    report.append("\n## UMAP Parameters")
    report.append(f"- n_neighbors: {analysis.n_neighbors}")
    report.append(f"- min_dist: {analysis.min_dist}")
    report.append(f"- metric: {analysis.metric}")
    report.append(f"- Embedding shape: {analysis.embedding_shape}")

    report.append("\n## Training Data Analysis")
    report.append(f"\n**Total training trajectories**: {analysis.num_training_trajectories}")
    report.append(f"\n### Phase Distribution")
    report.append("\n| Phase | Count | Percentage |")
    report.append("|-------|-------|------------|")
    total = sum(analysis.phase_counts.values())
    for phase, count in sorted(analysis.phase_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        report.append(f"| {phase} | {count} | {pct:.1f}% |")

    report.append("\n## Coverage Gap Analysis")
    report.append(f"\n**Total gaps identified**: {len(analysis.coverage_gaps)}")
    report.append(f"\n**Gap density**: {analysis.gap_density:.4f} gaps per unit area")

    if analysis.coverage_gaps:
        report.append("\n### Gap Details")
        report.append("\n| Severity | Center | Radius | Nearest Distance | Nearest Phase |")
        report.append("|----------|--------|--------|------------------|---------------|")
        for gap in sorted(analysis.coverage_gaps, key=lambda g: g.nearest_training_distance, reverse=True)[:10]:
            report.append(f"| {gap.severity} | ({gap.center[0]:.2f}, {gap.center[1]:.2f}) | "
                         f"{gap.radius:.2f} | {gap.nearest_training_distance:.2f} | {gap.nearest_phase} |")

    report.append("\n## Inference Trajectory Analysis")
    report.append(f"\n**Inference trajectories analyzed**: {analysis.num_inference_trajectories}")

    if analysis.inference_positions:
        report.append("\n### Position Analysis")
        report.append("\n| Case | UMAP Position | Nearest Phase | Distance | In Gap? |")
        report.append("|------|---------------|---------------|----------|---------|")
        for name in analysis.inference_positions.keys():
            pos = analysis.inference_positions[name]
            phase = analysis.inference_nearest_phase[name]
            dist = analysis.inference_nearest_distance[name]
            in_gap = "**YES**" if analysis.inference_in_gap[name] else "no"
            report.append(f"| {name[:25]} | ({pos[0]:.2f}, {pos[1]:.2f}) | {phase} | {dist:.2f} | {in_gap} |")

    report.append("\n## Key Findings")

    # Analyze findings
    halluc_cases = [n for n in analysis.inference_positions.keys() if "ha_bana" in n or "halluc" in n.lower()]
    normal_cases = [n for n in analysis.inference_positions.keys() if "no_ha" in n]

    if halluc_cases and normal_cases:
        report.append("\n### Hallucination vs Normal Cases")

        for h_case in halluc_cases:
            h_dist = analysis.inference_nearest_distance[h_case]
            h_phase = analysis.inference_nearest_phase[h_case]
            h_gap = analysis.inference_in_gap[h_case]

            report.append(f"\n**Hallucination case ({h_case[:25]}):**")
            report.append(f"- Nearest phase: {h_phase}")
            report.append(f"- Distance to training: {h_dist:.2f}")
            if h_gap:
                report.append(f"- **Falls in coverage gap** - suggests OOD behavior")

        for n_case in normal_cases:
            n_dist = analysis.inference_nearest_distance[n_case]
            n_phase = analysis.inference_nearest_phase[n_case]

            report.append(f"\n**Normal case ({n_case[:25]}):**")
            report.append(f"- Nearest phase: {n_phase}")
            report.append(f"- Distance to training: {n_dist:.2f}")

    # Check for missing TRANSPORT→IDLE transition
    if PHASE_TRANSPORT in analysis.phase_counts and PHASE_IDLE in analysis.phase_counts:
        transport_count = analysis.phase_counts.get(PHASE_TRANSPORT, 0)
        idle_count = analysis.phase_counts.get(PHASE_IDLE, 0)

        if idle_count < transport_count * 0.1:
            report.append(f"\n### Missing Transition: TRANSPORT → IDLE")
            report.append(f"- TRANSPORT segments: {transport_count}")
            report.append(f"- IDLE segments: {idle_count}")
            report.append(f"- Ratio: {idle_count/transport_count:.2%}")
            report.append("- **This suggests the model lacks training on post-task idle behavior**")

    report.append("\n## Visualizations")
    report.append("\n- `umap_manifold.png`: Full UMAP visualization with phases, density, gaps")
    report.append("- `inference_detail.png`: Detailed view of inference trajectory positions")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        DATASET_PATH, TASK_FILTER, get_all_case_dirs, get_output_dir, STEP_RANGE
    )

    parser = argparse.ArgumentParser(description="UMAP trajectory manifold analysis")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                       help="Path to training dataset")
    parser.add_argument("--task-filter", default=TASK_FILTER,
                       help="Filter for task names")
    parser.add_argument("--case-dirs", nargs="+", default=get_all_case_dirs(),
                       help="Paths to inference case directories")
    parser.add_argument("--step-range", nargs=2, type=int, default=list(STEP_RANGE),
                       help="Step range for inference trajectory extraction")
    parser.add_argument("--chunk-size", type=int, default=50,
                       help="Trajectory chunk size")
    parser.add_argument("--n-neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.1,
                       help="UMAP min_dist parameter")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("umap_trajectory")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Extract training trajectories
    print("\n=== Extracting Training Trajectories ===")
    training_traj, phases, episode_indices = extract_training_trajectories(
        args.dataset, args.task_filter, args.chunk_size
    )
    print(f"Extracted {len(training_traj)} training trajectory segments")

    # Extract inference trajectories
    print("\n=== Extracting Inference Trajectories ===")
    inference_traj = extract_inference_trajectories(
        args.case_dirs, tuple(args.step_range), args.chunk_size
    )
    print(f"Extracted {len(inference_traj)} inference trajectory segments")

    # Compute UMAP embedding
    print("\n=== Computing UMAP Embedding ===")
    training_emb, inference_emb, reducer = compute_umap_embedding(
        training_traj, inference_traj,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    # Identify coverage gaps
    print("\n=== Identifying Coverage Gaps ===")
    gaps = identify_coverage_gaps(training_emb, phases)
    print(f"Identified {len(gaps)} coverage gaps")

    # Analyze inference positions
    inference_names = list(inference_traj.keys())
    inference_analysis = analyze_inference_positions(
        training_emb, inference_emb, inference_names, phases, gaps
    )

    # Compute phase statistics
    from collections import Counter
    phase_counts = dict(Counter(phases))

    # Compute phase centroids
    phase_centroids = {}
    for phase in set(phases):
        mask = [p == phase for p in phases]
        if any(mask):
            phase_centroids[phase] = training_emb[mask].mean(axis=0).tolist()

    # Compute gap density
    x_range = training_emb[:, 0].max() - training_emb[:, 0].min()
    y_range = training_emb[:, 1].max() - training_emb[:, 1].min()
    area = x_range * y_range
    gap_density = len(gaps) / area if area > 0 else 0

    # Build analysis object
    analysis = TrajectoryManifoldAnalysis(
        num_training_trajectories=len(training_traj),
        num_inference_trajectories=len(inference_traj),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric='euclidean',
        embedding_shape=training_emb.shape,
        variance_explained=None,
        phase_counts=phase_counts,
        phase_centroids=phase_centroids,
        coverage_gaps=[asdict(g) for g in gaps],
        gap_density=gap_density,
        inference_positions=inference_analysis["positions"],
        inference_nearest_phase=inference_analysis["nearest_phase"],
        inference_nearest_distance=inference_analysis["nearest_distance"],
        inference_in_gap=inference_analysis["in_gap"],
    )

    # Save raw data
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(asdict(analysis), f, indent=2, cls=NumpyJSONEncoder)

    # Save embeddings
    np.save(output_dir / "training_embedding.npy", training_emb)
    np.save(output_dir / "inference_embedding.npy", inference_emb)
    np.save(output_dir / "phases.npy", np.array(phases))

    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    visualize_umap_manifold(training_emb, inference_emb, phases, inference_names, gaps, output_dir)
    visualize_inference_detail(training_emb, inference_emb, phases, inference_names, inference_analysis, output_dir)

    # Generate report
    print("\n=== Generating Report ===")
    generate_report(analysis, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
