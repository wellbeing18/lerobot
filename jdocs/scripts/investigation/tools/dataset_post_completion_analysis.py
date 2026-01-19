#!/usr/bin/env python3
"""
Dataset Post-Completion Behavior Analysis Tool

Analyzes the training dataset to verify H3 (Training Data Bias):
- What % of frames are in "idle/post-completion" phase?
- What are typical action values after task completion?
- Are there clear "stay still" patterns in training data?
- Does training data contain action patterns similar to hallucination?

Usage:
    python dataset_post_completion_analysis.py --dataset-dir datasets_bimanuel/multitasks --output-dir logs/dataset_analysis
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np


def load_dataset(dataset_dir: Path) -> dict:
    """Load dataset using lerobot API or fallback to pyarrow."""
    import pyarrow.parquet as pq

    data_files = sorted(dataset_dir.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise ValueError(f"No data files found in {dataset_dir}")

    print(f"Loading {len(data_files)} data files using pyarrow...")

    # Use pyarrow directly with legacy mode
    tables = []
    for f in data_files:
        table = pq.read_table(str(f))
        tables.append(table)

    import pyarrow as pa
    combined = pa.concat_tables(tables)

    # Convert to dict format
    data = {
        'action': np.stack(combined['action'].to_pylist()),
        'observation.state': np.stack(combined['observation.state'].to_pylist()),
        'episode_index': np.array(combined['episode_index'].to_pylist()).flatten(),
        'frame_index': np.array(combined['frame_index'].to_pylist()).flatten(),
    }

    n_episodes = len(np.unique(data['episode_index']))
    print(f"Loaded {len(data['episode_index'])} frames from {n_episodes} episodes")
    return data


def compute_action_deltas(data: dict, left_arm_only: bool = True) -> np.ndarray:
    """Compute action deltas (magnitude of action changes) per frame."""
    actions = data['action']

    if left_arm_only:
        # Use only left arm joints (indices 0-5)
        actions = actions[:, :6]

    # Compute delta from current state to action (how much movement commanded)
    states = data['observation.state']
    if left_arm_only:
        states = states[:, :6]

    deltas = np.linalg.norm(actions - states, axis=1)
    return deltas


def analyze_episode_phases(data: dict, episode_idx: int, idle_threshold: float = 3.0) -> dict:
    """Analyze phases within a single episode."""
    mask = data['episode_index'] == episode_idx
    if not np.any(mask):
        return None

    # Sort by frame index
    sort_indices = np.argsort(data['frame_index'][mask])

    # Get left arm actions and states
    actions = data['action'][mask][sort_indices][:, :6]
    states = data['observation.state'][mask][sort_indices][:, :6]

    # Compute per-frame action delta
    deltas = np.linalg.norm(actions - states, axis=1)

    # Detect gripper state changes (index 5 is left gripper)
    gripper_positions = actions[:, 5]
    gripper_open = gripper_positions > 15  # Threshold for "open"

    # Find idle periods (low delta, arm not moving)
    is_idle = deltas < idle_threshold

    # Find transition points
    idle_start = None
    for i, idle in enumerate(is_idle):
        if idle and idle_start is None:
            # Check if we have sustained idle (at least 10 frames)
            if np.all(is_idle[i:min(i+10, len(is_idle))]):
                idle_start = i
                break

    # Calculate phase durations
    total_frames = len(actions)
    active_frames = idle_start if idle_start else total_frames
    idle_frames = total_frames - active_frames if idle_start else 0

    # Analyze idle period actions if exists
    idle_action_stats = None
    if idle_start:
        idle_deltas = deltas[idle_start:]
        idle_action_stats = {
            "mean_delta": float(np.mean(idle_deltas)),
            "std_delta": float(np.std(idle_deltas)),
            "max_delta": float(np.max(idle_deltas)),
            "frames": len(idle_deltas)
        }

    return {
        "episode_idx": episode_idx,
        "total_frames": total_frames,
        "active_frames": active_frames,
        "idle_frames": idle_frames,
        "idle_ratio": idle_frames / total_frames if total_frames > 0 else 0,
        "idle_start_frame": idle_start,
        "mean_active_delta": float(np.mean(deltas[:active_frames])) if active_frames > 0 else 0,
        "mean_overall_delta": float(np.mean(deltas)),
        "idle_action_stats": idle_action_stats,
        "gripper_open_ratio": float(np.mean(gripper_open))
    }


def find_replay_patterns(data: dict, episode_idx: int, search_window: int = 30) -> dict:
    """
    Search for action replay patterns within an episode.
    Look for sequences where the arm returns to a previously visited position.
    """
    mask = data['episode_index'] == episode_idx
    if not np.any(mask) or np.sum(mask) < search_window * 2:
        return None

    # Sort by frame index
    sort_indices = np.argsort(data['frame_index'][mask])

    # Get left arm positions (joint positions, not gripper)
    states = data['observation.state'][mask][sort_indices][:, :5]
    episode_length = len(states)

    # Look for position revisits after initial phase
    initial_positions = states[:search_window]  # First 30 frames
    late_positions = states[search_window * 2:]  # After frame 60

    # Compute distances from late positions to initial positions
    replay_candidates = []
    for late_idx, late_pos in enumerate(late_positions):
        actual_frame_idx = late_idx + search_window * 2
        for init_idx, init_pos in enumerate(initial_positions):
            distance = np.linalg.norm(late_pos - init_pos)
            if distance < 20:  # Close enough to be considered a "return"
                replay_candidates.append({
                    "late_frame": actual_frame_idx,
                    "init_frame": init_idx,
                    "distance": float(distance),
                    "late_position": late_pos.tolist(),
                    "init_position": init_pos.tolist()
                })

    # Group by late frame (multiple initial frames could match)
    unique_replays = {}
    for rc in replay_candidates:
        if rc["late_frame"] not in unique_replays or rc["distance"] < unique_replays[rc["late_frame"]]["distance"]:
            unique_replays[rc["late_frame"]] = rc

    return {
        "episode_idx": episode_idx,
        "total_frames": episode_length,
        "replay_count": len(unique_replays),
        "replay_frames": list(unique_replays.keys()),
        "replays": list(unique_replays.values())[:10]  # Limit to first 10
    }


def analyze_dataset_post_completion(
    dataset_dir: Path,
    output_dir: Path,
    idle_threshold: float = 3.0,
    max_episodes: Optional[int] = None
) -> dict:
    """Comprehensive analysis of post-completion behavior in training data."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_dataset(dataset_dir)

    # Analyze each episode
    episode_indices = sorted(np.unique(data['episode_index']))
    if max_episodes:
        episode_indices = episode_indices[:max_episodes]

    print(f"\nAnalyzing {len(episode_indices)} episodes...")

    episode_analyses = []
    replay_analyses = []

    for ep_idx in episode_indices:
        # Phase analysis
        phase_result = analyze_episode_phases(data, ep_idx, idle_threshold)
        if phase_result:
            episode_analyses.append(phase_result)

        # Replay pattern analysis
        replay_result = find_replay_patterns(data, ep_idx)
        if replay_result:
            replay_analyses.append(replay_result)

    # Aggregate statistics
    idle_ratios = [ea["idle_ratio"] for ea in episode_analyses]
    active_deltas = [ea["mean_active_delta"] for ea in episode_analyses]
    overall_deltas = [ea["mean_overall_delta"] for ea in episode_analyses]

    episodes_with_idle = sum(1 for ea in episode_analyses if ea["idle_frames"] > 0)
    episodes_with_replay = sum(1 for ra in replay_analyses if ra["replay_count"] > 0)

    # Compute idle action statistics
    idle_deltas_all = []
    for ea in episode_analyses:
        if ea["idle_action_stats"]:
            # Weight by number of idle frames
            idle_deltas_all.extend([ea["idle_action_stats"]["mean_delta"]] * ea["idle_action_stats"]["frames"])

    results = {
        "dataset_dir": str(dataset_dir),
        "total_episodes": len(episode_indices),
        "idle_threshold": idle_threshold,

        # Post-completion statistics
        "episodes_with_idle": episodes_with_idle,
        "episodes_with_idle_ratio": episodes_with_idle / len(episode_indices) if episode_indices else 0,
        "mean_idle_ratio": float(np.mean(idle_ratios)) if idle_ratios else 0,
        "std_idle_ratio": float(np.std(idle_ratios)) if idle_ratios else 0,

        # Action delta statistics
        "mean_active_delta": float(np.mean(active_deltas)) if active_deltas else 0,
        "mean_overall_delta": float(np.mean(overall_deltas)) if overall_deltas else 0,
        "mean_idle_delta": float(np.mean(idle_deltas_all)) if idle_deltas_all else 0,
        "std_idle_delta": float(np.std(idle_deltas_all)) if idle_deltas_all else 0,

        # Replay pattern statistics (H2 related)
        "episodes_with_replay": episodes_with_replay,
        "episodes_with_replay_ratio": episodes_with_replay / len(episode_indices) if episode_indices else 0,
        "total_replay_events": sum(ra["replay_count"] for ra in replay_analyses),

        # Detailed episode data
        "episode_analyses": episode_analyses,
        "replay_analyses": replay_analyses
    }

    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    results = convert_to_native(results)

    # Save results
    output_file = output_dir / "post_completion_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("POST-COMPLETION BEHAVIOR ANALYSIS RESULTS")
    print("="*60)
    print(f"\nDataset: {dataset_dir}")
    print(f"Episodes analyzed: {len(episode_indices)}")
    print(f"Idle threshold: {idle_threshold}")

    print(f"\n--- Idle Phase Statistics ---")
    print(f"Episodes with idle period: {episodes_with_idle} ({100*episodes_with_idle/len(episode_indices):.1f}%)")
    print(f"Mean idle ratio per episode: {100*results['mean_idle_ratio']:.1f}%")
    print(f"Mean action delta during idle: {results['mean_idle_delta']:.2f} (std: {results['std_idle_delta']:.2f})")
    print(f"Mean action delta during active: {results['mean_active_delta']:.2f}")

    print(f"\n--- Replay Pattern Statistics (H2) ---")
    print(f"Episodes with position replay: {episodes_with_replay} ({100*episodes_with_replay/len(episode_indices):.1f}%)")
    print(f"Total replay events detected: {results['total_replay_events']}")

    # Check for H3 evidence
    print(f"\n--- H3 (Training Data Bias) Hypothesis Check ---")
    if results['mean_idle_ratio'] < 0.1:
        print(f"WARNING: Low idle ratio ({100*results['mean_idle_ratio']:.1f}%) suggests training data lacks post-completion examples!")
        print("This SUPPORTS H3: Model may not learn 'stay still' behavior after task completion.")
    else:
        print(f"Idle ratio ({100*results['mean_idle_ratio']:.1f}%) suggests reasonable post-completion coverage.")

    if results['mean_idle_delta'] > 2.0:
        print(f"WARNING: High idle delta ({results['mean_idle_delta']:.2f}) suggests training 'idle' periods have movement!")
        print("This SUPPORTS H3: Training data may show movement even during 'idle' period.")
    else:
        print(f"Idle delta ({results['mean_idle_delta']:.2f}) suggests training data has clear 'stay still' examples.")

    return results


def compare_with_inference(
    dataset_analysis: dict,
    hallucination_trace_path: Path,
    output_dir: Path
) -> dict:
    """Compare training data patterns with inference hallucination."""
    import json

    # Load inference trace
    with open(hallucination_trace_path) as f:
        trace_lines = [json.loads(line) for line in f]

    # Find hallucination period (after step 200)
    hallucination_deltas = []
    for entry in trace_lines:
        if entry.get('step', 0) >= 200:
            if 'action_delta' in entry:
                hallucination_deltas.append(entry['action_delta'])

    # Compare with training idle period
    training_idle_delta = dataset_analysis['mean_idle_delta']
    training_idle_std = dataset_analysis['std_idle_delta']

    if hallucination_deltas:
        halluc_mean = np.mean(hallucination_deltas)
        halluc_std = np.std(hallucination_deltas)

        comparison = {
            "training_idle_delta_mean": training_idle_delta,
            "training_idle_delta_std": training_idle_std,
            "hallucination_delta_mean": halluc_mean,
            "hallucination_delta_std": halluc_std,
            "delta_difference": halluc_mean - training_idle_delta,
            "z_score": (halluc_mean - training_idle_delta) / training_idle_std if training_idle_std > 0 else float('inf')
        }

        print(f"\n--- Inference vs Training Comparison ---")
        print(f"Training idle delta: {training_idle_delta:.2f} (±{training_idle_std:.2f})")
        print(f"Hallucination delta: {halluc_mean:.2f} (±{halluc_std:.2f})")
        print(f"Z-score: {comparison['z_score']:.2f}")

        if comparison['z_score'] > 2.0:
            print("STRONG EVIDENCE: Hallucination actions are significantly different from training idle!")

        # Save comparison
        output_file = output_dir / "training_vs_hallucination_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        return comparison

    return None


def generate_visualizations(results: dict, output_dir: Path):
    """Generate visualizations of the analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Idle ratio distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Idle ratio histogram
    idle_ratios = [ea["idle_ratio"] for ea in results["episode_analyses"]]
    axes[0, 0].hist(idle_ratios, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=np.mean(idle_ratios), color='r', linestyle='--', label=f'Mean: {np.mean(idle_ratios):.2f}')
    axes[0, 0].set_xlabel('Idle Ratio')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Episode Idle Ratio Distribution')
    axes[0, 0].legend()

    # Active vs idle delta comparison
    active_deltas = [ea["mean_active_delta"] for ea in results["episode_analyses"]]
    idle_deltas = [ea["idle_action_stats"]["mean_delta"] if ea["idle_action_stats"] else 0
                   for ea in results["episode_analyses"]]

    x_pos = [0, 1]
    bars = axes[0, 1].bar(x_pos, [np.mean(active_deltas), np.mean(idle_deltas)],
                          yerr=[np.std(active_deltas), np.std(idle_deltas)],
                          capsize=5, alpha=0.7)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(['Active Phase', 'Idle Phase'])
    axes[0, 1].set_ylabel('Mean Action Delta')
    axes[0, 1].set_title('Action Delta: Active vs Idle')

    # Episode length distribution
    episode_lengths = [ea["total_frames"] for ea in results["episode_analyses"]]
    axes[1, 0].hist(episode_lengths, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=np.mean(episode_lengths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_lengths):.0f}')
    axes[1, 0].set_xlabel('Episode Length (frames)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Episode Length Distribution')
    axes[1, 0].legend()

    # Replay patterns per episode
    replay_counts = [ra["replay_count"] for ra in results["replay_analyses"]]
    axes[1, 1].hist(replay_counts, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Replay Count')
    axes[1, 1].set_ylabel('Episode Count')
    axes[1, 1].set_title('Position Replay Events per Episode')

    plt.tight_layout()
    plt.savefig(output_dir / "post_completion_analysis.png", dpi=150)
    plt.close()
    print(f"Visualization saved to {output_dir / 'post_completion_analysis.png'}")

    # 2. Detailed idle phase timeline
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort episodes by idle start frame
    sorted_eps = sorted(results["episode_analyses"],
                       key=lambda x: x["idle_start_frame"] if x["idle_start_frame"] else x["total_frames"])

    for i, ea in enumerate(sorted_eps[:50]):  # Show first 50
        total = ea["total_frames"]
        idle_start = ea["idle_start_frame"] if ea["idle_start_frame"] else total

        # Active period
        ax.barh(i, idle_start, color='steelblue', alpha=0.7)
        # Idle period
        if idle_start < total:
            ax.barh(i, total - idle_start, left=idle_start, color='lightgreen', alpha=0.7)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Episode')
    ax.set_title('Episode Phase Timeline (Blue=Active, Green=Idle)')
    ax.legend(['Active', 'Idle'], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / "episode_phase_timeline.png", dpi=150)
    plt.close()
    print(f"Timeline saved to {output_dir / 'episode_phase_timeline.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training dataset post-completion behavior")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--idle-threshold", type=float, default=3.0, help="Action delta threshold for idle detection")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes to analyze")
    parser.add_argument("--compare-trace", type=str, default=None, help="Path to hallucination trace for comparison")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    results = analyze_dataset_post_completion(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        idle_threshold=args.idle_threshold,
        max_episodes=args.max_episodes
    )

    # Generate visualizations
    generate_visualizations(results, output_dir)

    # Compare with hallucination trace if provided
    if args.compare_trace:
        compare_with_inference(results, Path(args.compare_trace), output_dir)

    return results


if __name__ == "__main__":
    main()
