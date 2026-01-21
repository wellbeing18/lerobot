#!/usr/bin/env python3
"""
Generate P(trajectory | context) visualization for training dataset.

Shows the IDLE gap in post-completion region that causes hallucination.

This script creates a visualization showing:
- Training data distribution P(velocity | episode_progress)
- Comparison with inference cases
- The missing IDLE region in post-completion

Usage:
    python generate_p_traj_dataset.py --output logs/investigation/causal_distribution/p_traj_given_context_dataset.png
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from pathlib import Path


def load_yogurt_episodes(dataset_path: str) -> list:
    """Load all yogurt-related episodes from training dataset."""
    dataset_path = Path(dataset_path)

    # Load tasks info
    tasks_df = pd.read_parquet(dataset_path / 'meta' / 'tasks.parquet')
    task_texts = tasks_df.index.tolist()
    task_indices = tasks_df['task_index'].tolist()
    task_map = {idx: txt for idx, txt in zip(task_indices, task_texts)}

    # Find yogurt task indices
    yogurt_task_idxs = [idx for idx, txt in task_map.items() if 'yogurt' in txt.lower()]
    print(f"Yogurt task indices: {yogurt_task_idxs}")

    # Load from parquet files
    data_dir = dataset_path / 'data' / 'chunk-000'
    parquet_files = sorted(data_dir.glob('*.parquet'))

    yogurt_episodes = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for ep_idx in df['episode_index'].unique():
            ep_data = df[df['episode_index'] == ep_idx]
            task_idx = ep_data['task_index'].iloc[0]

            if task_idx not in yogurt_task_idxs:
                continue

            actions = np.array(ep_data['action'].tolist())
            if len(actions) >= 20:
                yogurt_episodes.append({
                    'episode_idx': ep_idx,
                    'task': task_map[task_idx],
                    'actions': actions,
                    'length': len(actions)
                })

    return yogurt_episodes


def load_inference_velocities(trace_path: str, step_threshold: int = 200) -> list:
    """Load post-completion velocities from inference trace."""
    vels = []
    with open(trace_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            step = data.get('step', -1)
            if step > step_threshold:
                vels.append(data.get('action_delta_max', 0))
    return vels


def main():
    parser = argparse.ArgumentParser(description="Generate P(trajectory | context) for training dataset")
    parser.add_argument("--dataset", type=str, default="datasets_bimanuel/multitasks")
    parser.add_argument("--halluc-trace", type=str,
                       default="logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl")
    parser.add_argument("--normal-trace", type=str,
                       default="logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl")
    parser.add_argument("--output", type=str,
                       default="logs/investigation/causal_distribution/p_traj_given_context_dataset.png")
    args = parser.parse_args()

    # Load training data
    print("Loading training data for yogurt bottle task...")
    yogurt_episodes = load_yogurt_episodes(args.dataset)
    print(f"Loaded {len(yogurt_episodes)} yogurt episodes")

    # Compute velocities at each episode progress point
    all_progress = []
    all_velocities = []

    for ep in yogurt_episodes:
        actions = ep['actions']
        velocities = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        progress = np.arange(len(velocities)) / len(velocities)
        all_progress.extend(progress)
        all_velocities.extend(velocities)

    all_progress = np.array(all_progress)
    all_velocities = np.array(all_velocities)

    print(f"Total training data points: {len(all_progress)}")

    # Load inference data
    print("\nLoading inference traces...")
    halluc_vels = load_inference_velocities(args.halluc_trace)
    normal_vels = load_inference_velocities(args.normal_trace)
    print(f"  Halluc: {len(halluc_vels)} data points, mean={np.mean(halluc_vels):.2f}")
    print(f"  Normal: {len(normal_vels)} data points, mean={np.mean(normal_vels):.2f}")

    # Create the visualization
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(20, 14))

    # === Panel 1: P(velocity | episode_progress) for training data ===
    ax1 = fig.add_subplot(2, 2, 1)

    x_range = np.linspace(0, 1, 100)
    y_range = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Build density from training data
    density = np.zeros_like(X)
    for p, v in zip(all_progress, all_velocities):
        if v < 6:  # Clip outliers
            density += np.exp(-((X - p)**2 / 0.005 + (Y - v)**2 / 0.5))
    density = gaussian_filter(density, sigma=3)

    # Plot
    cmap = LinearSegmentedColormap.from_list('custom',
        ['white', '#e6f2ff', '#99ccff', '#4da6ff', '#0066cc', '#003366'])
    levels = np.linspace(0, density.max(), 40)
    im = ax1.contourf(X, Y, density, levels=levels, cmap=cmap, alpha=0.9)
    ax1.contour(X, Y, density, levels=levels[::8], colors='darkblue', alpha=0.3, linewidths=0.5)

    # Mark IDLE region
    ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='IDLE threshold')
    ax1.fill_between(x_range, 0, 0.5, alpha=0.2, color='green')

    # Mark post-completion region
    ax1.axvline(x=0.8, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax1.axvspan(0.8, 1.0, alpha=0.15, color='red')
    ax1.text(0.9, 5.5, 'POST-\nCOMPLETION\nREGION', ha='center', fontsize=10, color='darkred', fontweight='bold')

    # Add phase labels
    ax1.text(0.15, 4.5, 'APPROACH', fontsize=12, ha='center', color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    ax1.text(0.5, 3.0, 'TRANSPORT', fontsize=12, ha='center', color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    ax1.text(0.1, 0.25, 'IDLE\n(MISSING!)', fontsize=10, ha='center', color='darkgreen', fontweight='bold')
    ax1.text(0.9, 0.25, 'IDLE\n(MISSING!)', fontsize=10, ha='center', color='darkred', fontweight='bold')

    ax1.set_xlabel('Episode Progress (0=start, 1=end)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Action Velocity (Movement Score)', fontsize=12, fontweight='bold')
    ax1.set_title('TRAINING DATA: P(velocity | episode_progress)\nYogurt Bottle Task - Where is IDLE?',
                 fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 6)
    plt.colorbar(im, ax=ax1, label='Probability Density')
    ax1.legend(loc='upper right')

    # === Panel 2: Comparison - Training vs Inference ===
    ax2 = fig.add_subplot(2, 2, 2)

    # Training data: post-completion velocities
    post_completion_vels_train = all_velocities[all_progress > 0.8]

    # Plot histograms
    bins = np.linspace(0, 10, 30)
    ax2.hist(post_completion_vels_train, bins=bins, alpha=0.5, color='blue', density=True,
            label=f'Training post-completion (μ={np.mean(post_completion_vels_train):.2f})', edgecolor='darkblue')
    ax2.hist(halluc_vels, bins=bins, alpha=0.6, color='red', density=True,
            label=f'Halluc inference (μ={np.mean(halluc_vels):.2f})', edgecolor='darkred')
    ax2.hist(normal_vels, bins=bins, alpha=0.6, color='green', density=True,
            label=f'Normal inference (μ={np.mean(normal_vels):.2f})', edgecolor='darkgreen')

    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='IDLE threshold')
    ax2.axvspan(0, 0.5, alpha=0.1, color='green')

    ax2.set_xlabel('Velocity (Movement Score)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Post-Completion Velocity Distribution\nTraining vs Inference Cases', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add annotation
    train_idle_pct = 100 * np.mean(post_completion_vels_train < 0.5)
    normal_idle_pct = 100 * np.mean(np.array(normal_vels) < 0.5)
    halluc_idle_pct = 100 * np.mean(np.array(halluc_vels) < 0.5)
    ax2.text(0.95, 0.95,
            f'Training IDLE: {train_idle_pct:.0f}%\n\nNormal inference: {normal_idle_pct:.0f}% IDLE\nHalluc inference: {halluc_idle_pct:.0f}% IDLE',
            transform=ax2.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # === Panel 3: Combined training + inference overlay ===
    ax3 = fig.add_subplot(2, 2, 3)

    # Plot training density
    im3 = ax3.contourf(X, Y, density / density.max(), levels=20, cmap='Blues', alpha=0.6)

    # Overlay inference data
    halluc_x = 0.9 + np.random.randn(len(halluc_vels)) * 0.02
    ax3.scatter(halluc_x, np.clip(halluc_vels, 0, 6), c='red', s=30, alpha=0.5, marker='o', label='Halluc inference')

    normal_x = 0.85 + np.random.randn(len(normal_vels)) * 0.02
    ax3.scatter(normal_x, np.clip(normal_vels, 0, 6), c='lime', s=30, alpha=0.5, marker='s', label='Normal inference')

    # Mark centroids
    ax3.scatter([0.9], [np.mean(halluc_vels)], c='red', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    ax3.scatter([0.85], [np.mean(normal_vels)], c='lime', s=300, marker='^', edgecolors='black', linewidths=2, zorder=10)

    # Mark the GAP
    ax3.annotate('', xy=(0.9, 0.3), xytext=(0.9, min(np.mean(halluc_vels), 5)),
                arrowprops=dict(arrowstyle='<->', color='red', lw=3))
    ax3.text(0.92, np.mean(halluc_vels)/2, 'GAP!\nNo training\nfor IDLE here', fontsize=10, color='red', fontweight='bold')

    ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=2)
    ax3.fill_between(x_range, 0, 0.5, alpha=0.2, color='green')
    ax3.axvspan(0.8, 1.0, alpha=0.1, color='red')

    ax3.set_xlabel('Context (0=task start, 1=post-completion)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Velocity (Movement Score)', fontsize=12, fontweight='bold')
    ax3.set_title('DATASET DEFECT: Missing IDLE in Post-Completion\nTraining distribution (blue) vs Inference (red/green)',
                 fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 6)
    ax3.legend(loc='upper left', fontsize=9)

    # === Panel 4: Summary diagram ===
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary = f"""
DATASET DEFECT ANALYSIS: P(trajectory | context)

TRAINING DATA (Yogurt Bottle Task)
==================================
Episodes: {len(yogurt_episodes)}
Total steps: {len(all_progress)}

Post-Completion Analysis:
  - Mean velocity in last 20%: {np.mean(post_completion_vels_train):.2f}
  - Training IDLE ratio: {train_idle_pct:.0f}%


THE PROBLEM
===========
Training data teaches:
  P(MOVEMENT | any_context) = HIGH
  P(IDLE | post_completion) = LOW (few examples)

When model sees post-completion + distractor:
  - Has weak training signal for IDLE
  - Defaults to nearest learned behavior: MOVEMENT


INFERENCE COMPARISON
====================
                    | Hallucination | Normal
--------------------|---------------|--------
Post-completion vel |     {np.mean(halluc_vels):.2f}      |  {np.mean(normal_vels):.2f}
P(IDLE)             |      {halluc_idle_pct:.0f}%       |  {normal_idle_pct:.0f}%
Behavior            |   MOVEMENT    | ~STILL

Normal case stays still with weak prior,
Halluc case has no competing signal to stay still.


RECOMMENDATION
==============
Add training data with:
1. Post-completion IDLE segments (stay still after task)
2. Distractor-present scenarios (irrelevant objects visible)
3. Explicit "task complete" conditioning
"""

    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Training Dataset Analysis: P(trajectory | context) - Revealing the IDLE Gap',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
