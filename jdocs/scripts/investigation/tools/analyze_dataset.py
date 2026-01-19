#!/usr/bin/env python3
"""
Dataset Analyzer for SmolVLA Hallucination Investigation.

Analyzes the bimanual training dataset to identify distribution issues
that may contribute to hallucination behavior:

1. Episode count per task (balance check)
2. Trajectory phase distribution (approach/grip/transport/release/return/idle)
3. Post-completion behavior patterns
4. Action space coverage
5. Visual scene analysis (if frames available)

Key Questions:
- Are post-completion frames under-represented?
- Are multi-object scenes under-represented?
- What do training trajectories do after task completion?

Usage:
    # Full dataset analysis
    python analyze_dataset.py \
        --dataset-path /path/to/datasets_bimanuel/multitasks \
        --output-dir ../reports/dataset_analysis

    # Focus on specific aspects
    python analyze_dataset.py \
        --dataset-path /path/to/datasets_bimanuel/multitasks \
        --output-dir ../reports/dataset_analysis \
        --analyze-phases \
        --analyze-actions
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskStats:
    """Statistics for a single task."""
    task_name: str
    episode_count: int
    total_frames: int
    avg_episode_length: float
    min_episode_length: int
    max_episode_length: int


@dataclass
class PhaseDistribution:
    """Distribution of trajectory phases."""
    approach: int = 0      # Moving toward object
    grip: int = 0          # Grasping object
    transport: int = 0     # Moving with object
    release: int = 0       # Releasing object
    return_home: int = 0   # Returning to start position
    idle: int = 0          # Not moving (includes post-completion)


@dataclass
class ActionStats:
    """Action space statistics."""
    joint_means: list = field(default_factory=list)
    joint_stds: list = field(default_factory=list)
    joint_mins: list = field(default_factory=list)
    joint_maxs: list = field(default_factory=list)
    gripper_open_ratio: float = 0.0
    gripper_close_ratio: float = 0.0


@dataclass
class DatasetAnalysis:
    """Complete dataset analysis results."""
    dataset_path: str
    analysis_timestamp: str
    total_episodes: int
    total_frames: int
    unique_tasks: int
    task_stats: list[TaskStats] = field(default_factory=list)
    phase_distribution: Optional[PhaseDistribution] = None
    action_stats: Optional[ActionStats] = None
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset_info(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load dataset metadata and episode information.

    Returns:
        tasks_df: Tasks dataframe
        episodes_df: Episodes dataframe
        stats: Dataset statistics dict
    """
    # Load tasks.parquet
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
    else:
        print(f"WARNING: tasks.parquet not found at {tasks_path}")
        tasks_df = pd.DataFrame()

    # Load episodes.parquet
    episodes_path = dataset_path / "meta" / "episodes.parquet"
    if episodes_path.exists():
        episodes_df = pd.read_parquet(episodes_path)
    else:
        print(f"WARNING: episodes.parquet not found at {episodes_path}")
        episodes_df = pd.DataFrame()

    # Load stats.json
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        print(f"WARNING: stats.json not found at {stats_path}")
        stats = {}

    return tasks_df, episodes_df, stats


def load_episode_data(dataset_path: Path, episode_idx: int) -> Optional[pd.DataFrame]:
    """Load data for a specific episode."""
    # Try different possible file patterns
    patterns = [
        f"data/train-{episode_idx:05d}.parquet",
        f"data/episode_{episode_idx:06d}.parquet",
        f"data/chunk-{episode_idx // 100:03d}/episode_{episode_idx:06d}.parquet",
    ]

    for pattern in patterns:
        path = dataset_path / pattern
        if path.exists():
            return pd.read_parquet(path)

    return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_task_distribution(tasks_df: pd.DataFrame, episodes_df: pd.DataFrame) -> list[TaskStats]:
    """Analyze distribution of episodes across tasks."""
    if tasks_df.empty or episodes_df.empty:
        return []

    task_stats = []

    # Group episodes by task
    if 'task_index' in episodes_df.columns:
        for task_idx in tasks_df['task_index'].unique() if 'task_index' in tasks_df.columns else range(len(tasks_df)):
            task_episodes = episodes_df[episodes_df.get('task_index', pd.Series()) == task_idx]

            if task_episodes.empty:
                continue

            # Get task name
            task_row = tasks_df[tasks_df['task_index'] == task_idx] if 'task_index' in tasks_df.columns else None
            task_name = task_row['task'].iloc[0] if task_row is not None and not task_row.empty and 'task' in task_row.columns else f"Task {task_idx}"

            # Calculate statistics
            lengths = task_episodes['length'].values if 'length' in task_episodes.columns else []

            stats = TaskStats(
                task_name=task_name,
                episode_count=len(task_episodes),
                total_frames=int(sum(lengths)) if len(lengths) > 0 else 0,
                avg_episode_length=float(np.mean(lengths)) if len(lengths) > 0 else 0,
                min_episode_length=int(min(lengths)) if len(lengths) > 0 else 0,
                max_episode_length=int(max(lengths)) if len(lengths) > 0 else 0,
            )
            task_stats.append(stats)

    return task_stats


def detect_trajectory_phase(
    state: np.ndarray,
    action: np.ndarray,
    prev_state: np.ndarray = None,
    gripper_threshold: float = 30.0,
    movement_threshold: float = 2.0,
) -> str:
    """
    Detect the phase of a trajectory step.

    Args:
        state: Current robot state [12]
        action: Current action [12]
        prev_state: Previous state (optional)
        gripper_threshold: Threshold for gripper open/closed
        movement_threshold: Threshold for movement detection

    Returns:
        Phase name: 'approach', 'grip', 'transport', 'release', 'return_home', or 'idle'
    """
    # Extract gripper states (indices 5 and 11 for left and right)
    gripper_left = state[5] if len(state) > 5 else 0
    action_gripper_left = action[5] if len(action) > 5 else 0

    # Calculate movement
    if prev_state is not None:
        movement = np.abs(state[:5] - prev_state[:5]).max()  # Arm joints only
    else:
        movement = np.abs(action[:5] - state[:5]).max()

    is_moving = movement > movement_threshold
    gripper_closing = action_gripper_left < gripper_left - 5
    gripper_opening = action_gripper_left > gripper_left + 5
    gripper_closed = gripper_left < gripper_threshold

    # Phase detection logic
    if not is_moving and not gripper_closing and not gripper_opening:
        return 'idle'
    elif gripper_closing:
        return 'grip'
    elif gripper_opening:
        return 'release'
    elif is_moving and gripper_closed:
        return 'transport'
    elif is_moving and not gripper_closed:
        return 'approach'  # or return_home, hard to distinguish without trajectory context

    return 'idle'


def analyze_trajectory_phases(
    dataset_path: Path,
    episodes_df: pd.DataFrame,
    max_episodes: int = 50,
) -> PhaseDistribution:
    """
    Analyze distribution of trajectory phases across episodes.
    """
    phase_counts = PhaseDistribution()

    episode_indices = episodes_df['episode_index'].values[:max_episodes] if 'episode_index' in episodes_df.columns else range(min(max_episodes, len(episodes_df)))

    for ep_idx in episode_indices:
        ep_data = load_episode_data(dataset_path, ep_idx)
        if ep_data is None:
            continue

        # Get state and action columns
        state_col = 'observation.state'
        action_col = 'action'

        if state_col not in ep_data.columns or action_col not in ep_data.columns:
            continue

        states = np.array(ep_data[state_col].tolist())
        actions = np.array(ep_data[action_col].tolist())

        prev_state = None
        for i in range(len(states)):
            phase = detect_trajectory_phase(
                states[i],
                actions[i],
                prev_state,
            )

            if phase == 'approach':
                phase_counts.approach += 1
            elif phase == 'grip':
                phase_counts.grip += 1
            elif phase == 'transport':
                phase_counts.transport += 1
            elif phase == 'release':
                phase_counts.release += 1
            elif phase == 'return_home':
                phase_counts.return_home += 1
            else:
                phase_counts.idle += 1

            prev_state = states[i]

    return phase_counts


def analyze_action_space(stats: dict) -> ActionStats:
    """Analyze action space coverage from dataset stats."""
    action_stats = ActionStats()

    if 'action' in stats:
        action_info = stats['action']
        action_stats.joint_means = action_info.get('mean', [])
        action_stats.joint_stds = action_info.get('std', [])
        action_stats.joint_mins = action_info.get('min', [])
        action_stats.joint_maxs = action_info.get('max', [])

    return action_stats


def analyze_post_completion_behavior(
    dataset_path: Path,
    episodes_df: pd.DataFrame,
    max_episodes: int = 20,
) -> dict:
    """
    Analyze what happens after task completion in training data.

    Looks at the last N frames of each episode to understand
    post-completion behavior patterns.
    """
    post_completion_stats = {
        "episodes_analyzed": 0,
        "avg_final_movement": 0.0,
        "stay_still_ratio": 0.0,
        "continue_moving_ratio": 0.0,
        "patterns": [],
    }

    episode_indices = episodes_df['episode_index'].values[:max_episodes] if 'episode_index' in episodes_df.columns else []
    final_movements = []

    for ep_idx in episode_indices:
        ep_data = load_episode_data(dataset_path, ep_idx)
        if ep_data is None:
            continue

        state_col = 'observation.state'
        action_col = 'action'

        if state_col not in ep_data.columns or action_col not in ep_data.columns:
            continue

        states = np.array(ep_data[state_col].tolist())
        actions = np.array(ep_data[action_col].tolist())

        if len(states) < 30:
            continue

        # Analyze last 30 frames
        final_states = states[-30:]
        final_actions = actions[-30:]

        # Calculate movement in final frames
        state_diffs = np.abs(np.diff(final_states[:, :5], axis=0))  # Arm joints only
        avg_movement = state_diffs.mean()
        final_movements.append(avg_movement)

        post_completion_stats["episodes_analyzed"] += 1

    if final_movements:
        post_completion_stats["avg_final_movement"] = float(np.mean(final_movements))
        stay_still = sum(1 for m in final_movements if m < 1.0)
        post_completion_stats["stay_still_ratio"] = stay_still / len(final_movements)
        post_completion_stats["continue_moving_ratio"] = 1 - post_completion_stats["stay_still_ratio"]

    return post_completion_stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_task_distribution(task_stats: list[TaskStats], output_path: Path):
    """Plot episode count distribution across tasks."""
    if not task_stats:
        print("No task stats to plot")
        return

    # Sort by episode count
    sorted_stats = sorted(task_stats, key=lambda x: x.episode_count, reverse=True)

    names = [s.task_name[:30] + "..." if len(s.task_name) > 30 else s.task_name for s in sorted_stats]
    counts = [s.episode_count for s in sorted_stats]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax.barh(names, counts, color=colors)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontsize=9)

    ax.set_xlabel('Episode Count', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)
    ax.set_title('Episode Distribution Across Tasks', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Largest at top

    # Add target line (e.g., 20 episodes per task)
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='Target (20)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved task distribution plot: {output_path}")


def plot_phase_distribution(phase_dist: PhaseDistribution, output_path: Path):
    """Plot trajectory phase distribution."""
    phases = ['approach', 'grip', 'transport', 'release', 'return_home', 'idle']
    counts = [
        phase_dist.approach,
        phase_dist.grip,
        phase_dist.transport,
        phase_dist.release,
        phase_dist.return_home,
        phase_dist.idle,
    ]

    total = sum(counts)
    if total == 0:
        print("No phase data to plot")
        return

    percentages = [c / total * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
    ax1.pie(counts, labels=phases, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Phase Distribution (Frames)', fontsize=12, fontweight='bold')

    # Bar chart
    ax2.bar(phases, percentages, color=colors)
    ax2.set_xlabel('Phase', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Phase Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(percentages) * 1.2)

    # Add percentage labels
    for i, (phase, pct) in enumerate(zip(phases, percentages)):
        ax2.text(i, pct + 1, f'{pct:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved phase distribution plot: {output_path}")


def plot_action_space(action_stats: ActionStats, output_path: Path):
    """Plot action space coverage."""
    if not action_stats.joint_means:
        print("No action stats to plot")
        return

    joint_names = [
        "L_pan", "L_lift", "L_elbow", "L_wflex", "L_wroll", "L_grip",
        "R_pan", "R_lift", "R_elbow", "R_wflex", "R_wroll", "R_grip",
    ]

    n = min(len(joint_names), len(action_stats.joint_means))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n)
    width = 0.35

    means = action_stats.joint_means[:n]
    stds = action_stats.joint_stds[:n] if action_stats.joint_stds else [0] * n
    mins = action_stats.joint_mins[:n] if action_stats.joint_mins else means
    maxs = action_stats.joint_maxs[:n] if action_stats.joint_maxs else means

    # Plot mean with error bars
    ax.bar(x, means, width, yerr=stds, label='Mean ± Std', alpha=0.7, capsize=3)

    # Plot min/max range
    for i in range(n):
        ax.plot([i, i], [mins[i], maxs[i]], 'k-', linewidth=2, alpha=0.5)
        ax.plot(i, mins[i], 'v', color='blue', markersize=6)
        ax.plot(i, maxs[i], '^', color='red', markersize=6)

    ax.set_xlabel('Joint', fontsize=12)
    ax.set_ylabel('Value (degrees)', fontsize=12)
    ax.set_title('Action Space Coverage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names[:n], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved action space plot: {output_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_warnings_and_recommendations(analysis: DatasetAnalysis) -> tuple[list, list]:
    """Generate warnings and recommendations based on analysis."""
    warnings = []
    recommendations = []

    # Check task balance
    if analysis.task_stats:
        counts = [s.episode_count for s in analysis.task_stats]
        min_count = min(counts)
        max_count = max(counts)

        if max_count > 2 * min_count:
            warnings.append(f"Task imbalance detected: {min_count} to {max_count} episodes per task")
            recommendations.append("Consider collecting more episodes for under-represented tasks")

        if min_count < 15:
            warnings.append(f"Some tasks have very few episodes ({min_count})")
            recommendations.append("Aim for at least 20 episodes per task for robust training")

    # Check phase distribution
    if analysis.phase_distribution:
        total = sum([
            analysis.phase_distribution.approach,
            analysis.phase_distribution.grip,
            analysis.phase_distribution.transport,
            analysis.phase_distribution.release,
            analysis.phase_distribution.return_home,
            analysis.phase_distribution.idle,
        ])

        if total > 0:
            idle_ratio = analysis.phase_distribution.idle / total
            if idle_ratio < 0.05:
                warnings.append(f"Idle/post-completion frames are under-represented ({idle_ratio*100:.1f}%)")
                recommendations.append("Add more frames showing arm staying still after task completion")
                recommendations.append("This may help prevent post-completion hallucination")

    return warnings, recommendations


def generate_report(analysis: DatasetAnalysis, output_path: Path):
    """Generate markdown report."""
    report = f"""# Dataset Analysis Report

**Generated**: {analysis.analysis_timestamp}
**Dataset**: {analysis.dataset_path}

## Summary

| Metric | Value |
|--------|-------|
| Total Episodes | {analysis.total_episodes} |
| Total Frames | {analysis.total_frames} |
| Unique Tasks | {analysis.unique_tasks} |

## Task Distribution

| Task | Episodes | Frames | Avg Length |
|------|----------|--------|------------|
"""
    for stat in sorted(analysis.task_stats, key=lambda x: x.episode_count, reverse=True):
        report += f"| {stat.task_name[:40]} | {stat.episode_count} | {stat.total_frames} | {stat.avg_episode_length:.0f} |\n"

    if analysis.phase_distribution:
        total = sum([
            analysis.phase_distribution.approach,
            analysis.phase_distribution.grip,
            analysis.phase_distribution.transport,
            analysis.phase_distribution.release,
            analysis.phase_distribution.return_home,
            analysis.phase_distribution.idle,
        ])
        report += f"""
## Phase Distribution

| Phase | Frames | Percentage |
|-------|--------|------------|
| Approach | {analysis.phase_distribution.approach} | {analysis.phase_distribution.approach/total*100:.1f}% |
| Grip | {analysis.phase_distribution.grip} | {analysis.phase_distribution.grip/total*100:.1f}% |
| Transport | {analysis.phase_distribution.transport} | {analysis.phase_distribution.transport/total*100:.1f}% |
| Release | {analysis.phase_distribution.release} | {analysis.phase_distribution.release/total*100:.1f}% |
| Return Home | {analysis.phase_distribution.return_home} | {analysis.phase_distribution.return_home/total*100:.1f}% |
| Idle | {analysis.phase_distribution.idle} | {analysis.phase_distribution.idle/total*100:.1f}% |
"""

    if analysis.warnings:
        report += "\n## Warnings\n\n"
        for w in analysis.warnings:
            report += f"- ⚠️ {w}\n"

    if analysis.recommendations:
        report += "\n## Recommendations\n\n"
        for r in analysis.recommendations:
            report += f"- 💡 {r}\n"

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Saved report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Analyzer for SmolVLA Hallucination Investigation"
    )

    parser.add_argument("--dataset-path", "-d", type=Path, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                       help="Output directory for analysis")

    # Analysis options
    parser.add_argument("--analyze-phases", action="store_true",
                       help="Analyze trajectory phase distribution (slower)")
    parser.add_argument("--analyze-post-completion", action="store_true",
                       help="Analyze post-completion behavior")
    parser.add_argument("--max-episodes", type=int, default=50,
                       help="Max episodes to analyze for phase detection")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SmolVLA Dataset Analyzer")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")

    # Load dataset info
    print("\nLoading dataset metadata...")
    tasks_df, episodes_df, stats = load_dataset_info(args.dataset_path)

    print(f"  Tasks: {len(tasks_df)}")
    print(f"  Episodes: {len(episodes_df)}")

    # Initialize analysis
    analysis = DatasetAnalysis(
        dataset_path=str(args.dataset_path),
        analysis_timestamp=datetime.now().isoformat(),
        total_episodes=len(episodes_df),
        total_frames=int(episodes_df['length'].sum()) if 'length' in episodes_df.columns else 0,
        unique_tasks=len(tasks_df),
    )

    # Task distribution analysis
    print("\nAnalyzing task distribution...")
    analysis.task_stats = analyze_task_distribution(tasks_df, episodes_df)

    if analysis.task_stats:
        plot_task_distribution(analysis.task_stats, args.output_dir / "task_distribution.png")

    # Phase distribution analysis
    if args.analyze_phases:
        print(f"\nAnalyzing trajectory phases (up to {args.max_episodes} episodes)...")
        analysis.phase_distribution = analyze_trajectory_phases(
            args.dataset_path, episodes_df, args.max_episodes
        )
        plot_phase_distribution(analysis.phase_distribution, args.output_dir / "phase_distribution.png")

    # Post-completion analysis
    if args.analyze_post_completion:
        print("\nAnalyzing post-completion behavior...")
        post_completion = analyze_post_completion_behavior(
            args.dataset_path, episodes_df, args.max_episodes
        )
        with open(args.output_dir / "post_completion_analysis.json", 'w') as f:
            json.dump(post_completion, f, indent=2)

    # Action space analysis
    print("\nAnalyzing action space...")
    analysis.action_stats = analyze_action_space(stats)
    if analysis.action_stats.joint_means:
        plot_action_space(analysis.action_stats, args.output_dir / "action_space.png")

    # Generate warnings and recommendations
    analysis.warnings, analysis.recommendations = generate_warnings_and_recommendations(analysis)

    # Save analysis data
    with open(args.output_dir / "analysis.json", 'w') as f:
        # Convert dataclasses to dict
        data = asdict(analysis)
        json.dump(data, f, indent=2, default=str)

    # Generate report
    generate_report(analysis, args.output_dir / "report.md")

    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    print(f"Total episodes: {analysis.total_episodes}")
    print(f"Total frames: {analysis.total_frames}")
    print(f"Unique tasks: {analysis.unique_tasks}")

    if analysis.warnings:
        print("\nWarnings:")
        for w in analysis.warnings:
            print(f"  ⚠️  {w}")

    if analysis.recommendations:
        print("\nRecommendations:")
        for r in analysis.recommendations:
            print(f"  💡 {r}")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
