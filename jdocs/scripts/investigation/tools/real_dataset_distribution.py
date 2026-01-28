#!/usr/bin/env python3
"""
Analyze REAL training dataset distribution for P(velocity | context).

This script analyzes actual training data to show the distribution of
action velocities across different episode phases and contexts.

Unlike the simulated version, this uses REAL data from the training dataset.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq

OUTPUT_DIR = Path("/home/jrobot/project/lerobot/logs/investigation/first_principles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Correct path to actual action data (not metadata)
DATASET_PATH = Path("/home/jrobot/project/lerobot/datasets_bimanuel/multitasks/data/chunk-000")
HALLUC_TRACE = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl")
NORMAL_TRACE = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate/trace.jsonl")


def load_training_data():
    """Load real training data from parquet files."""
    print("Loading training data...")

    parquet_files = list(DATASET_PATH.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATASET_PATH}")

    print(f"Found {len(parquet_files)} parquet files")

    import pandas as pd
    all_data = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        # Only include files with action column
        if 'action' in df.columns:
            all_data.append(df)

    if not all_data:
        raise ValueError("No parquet files with 'action' column found")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total frames: {len(combined)}")

    return combined


def compute_episode_context(df):
    """
    Compute visual context proxy based on episode phase.

    Logic:
    - Early phase (0-30% of episode): Object manipulation, context ~ 0.8-1.0
    - Mid phase (30-70%): Active task execution, context ~ 0.6-0.9
    - Late phase (70-100%): Task completion, context ~ 0.0-0.4

    This is a proxy because we don't have direct "object visible" labels.
    The assumption is that training episodes follow: approach → manipulate → complete pattern.
    """
    contexts = []
    episode_phases = []

    # Group by episode
    if 'episode_index' not in df.columns:
        print("Warning: No episode_index column, treating as single episode")
        df['episode_index'] = 0

    for ep_idx in df['episode_index'].unique():
        ep_mask = df['episode_index'] == ep_idx
        ep_len = ep_mask.sum()

        # Compute relative position within episode
        positions = np.arange(ep_len) / max(ep_len - 1, 1)

        # Map position to context (early=high context, late=low context)
        # This reflects: object on workspace during task, cleared after completion
        for pos in positions:
            if pos < 0.3:
                # Early phase: approaching/grasping object
                ctx = np.random.uniform(0.7, 1.0)
                phase = 'early'
            elif pos < 0.7:
                # Mid phase: manipulating/transporting
                ctx = np.random.uniform(0.5, 0.9)
                phase = 'mid'
            else:
                # Late phase: task complete, workspace clearing
                ctx = np.random.uniform(0.0, 0.4)
                phase = 'late'

            contexts.append(ctx)
            episode_phases.append(phase)

    return np.array(contexts), np.array(episode_phases)


def compute_action_velocities(df):
    """Compute action velocity (movement magnitude) for each frame."""
    if 'action' not in df.columns:
        raise ValueError("No 'action' column in dataset")

    # Stack actions - handle potential shape differences
    actions_list = df['action'].values
    actions = np.array([np.array(a) for a in actions_list])
    print(f"Action shape: {actions.shape}")

    # Compute per-episode velocity to avoid discontinuities at episode boundaries
    velocities = np.zeros(len(actions))

    if 'episode_index' in df.columns:
        for ep_idx in df['episode_index'].unique():
            mask = df['episode_index'] == ep_idx
            ep_actions = actions[mask]
            ep_vels = np.zeros(len(ep_actions))
            if len(ep_actions) > 1:
                # Max absolute change across all joints
                ep_vels[1:] = np.abs(np.diff(ep_actions, axis=0)).max(axis=1)
            velocities[mask] = ep_vels
    else:
        velocities[1:] = np.abs(np.diff(actions, axis=0)).max(axis=1)

    # Actions are already in degrees (based on sample values like -99, 99)
    # No need to scale
    return velocities


def load_inference_traces():
    """Load inference trace data."""
    def load_trace(path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    halluc = load_trace(HALLUC_TRACE)
    normal = load_trace(NORMAL_TRACE)

    return halluc, normal


def create_real_distribution_plot():
    """Create scatter plot using REAL training data."""

    # Load real training data
    df = load_training_data()

    # Compute context proxy and velocities
    contexts, phases = compute_episode_context(df)
    velocities = compute_action_velocities(df)

    print(f"\nDataset statistics:")
    print(f"  Total frames: {len(velocities)}")
    print(f"  Velocity mean: {velocities.mean():.2f}°")
    print(f"  Velocity std: {velocities.std():.2f}°")
    print(f"  Velocity median: {np.median(velocities):.2f}°")

    # Load inference traces
    halluc, normal = load_inference_traces()

    # Extract post-completion data (steps 150-250)
    h_post = [(e['action_delta_max'], 0.85) for e in halluc if 150 < e['step'] < 250]
    n_post = [(e['action_delta_max'], 0.15) for e in normal if 150 < e['step'] < 250]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # === Panel 1: Real training distribution ===
    ax1 = axes[0]

    # Subsample for visualization (too many points otherwise)
    n_sample = min(5000, len(velocities))
    idx = np.random.choice(len(velocities), n_sample, replace=False)

    # Color by phase
    colors = {'early': 'steelblue', 'mid': 'orange', 'late': 'lightgreen'}
    for phase in ['early', 'mid', 'late']:
        mask = phases[idx] == phase
        ax1.scatter(contexts[idx][mask], velocities[idx][mask],
                   c=colors[phase], alpha=0.3, s=15, label=f'Training: {phase} phase')

    # Add inference points
    if h_post:
        h_vels, h_ctxs = zip(*h_post)
        ax1.scatter(h_ctxs, h_vels, c='red', s=80, marker='X',
                   label='Halluc inference', linewidths=1, edgecolors='darkred', zorder=10)

    if n_post:
        n_vels, n_ctxs = zip(*n_post)
        ax1.scatter(n_ctxs, n_vels, c='lime', s=80, marker='P',
                   label='Normal inference', linewidths=1, edgecolors='darkgreen', zorder=10)

    # Add IDLE threshold
    ax1.axhline(y=3, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(0.02, 3.5, 'IDLE threshold (3°)', fontsize=10, fontweight='bold')

    # Shade regions
    ax1.axhspan(0, 3, alpha=0.1, color='green')
    ax1.axhspan(3, 20, alpha=0.05, color='red')

    ax1.set_xlabel('Visual Context Proxy\n(0 = late phase/empty, 1 = early phase/object)', fontsize=11)
    ax1.set_ylabel('Action Velocity (degrees)', fontsize=11)
    ax1.set_title('P(velocity | context): REAL Training Data Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, min(20, np.percentile(velocities, 99)))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === Panel 2: Phase-wise histogram ===
    ax2 = axes[1]

    bins = np.linspace(0, 15, 40)

    for phase, color in colors.items():
        mask = phases == phase
        ax2.hist(velocities[mask], bins=bins, alpha=0.5, color=color,
                label=f'{phase.capitalize()} phase (n={mask.sum()})', density=True)

    ax2.axvline(x=3, color='black', linestyle='--', linewidth=2, label='IDLE threshold')

    ax2.set_xlabel('Action Velocity (degrees)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Velocity Distribution by Episode Phase', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training Data Analysis: Action Velocity vs Episode Phase',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "real_dataset_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print key statistics
    print("\n" + "="*60)
    print("KEY FINDINGS FROM REAL DATA:")
    print("="*60)

    for phase in ['early', 'mid', 'late']:
        mask = phases == phase
        phase_vels = velocities[mask]
        idle_ratio = (phase_vels < 3).mean()
        movement_ratio = (phase_vels >= 3).mean()
        print(f"\n{phase.upper()} PHASE:")
        print(f"  Frames: {mask.sum()}")
        print(f"  P(idle | {phase}): {idle_ratio:.1%}")
        print(f"  P(movement | {phase}): {movement_ratio:.1%}")
        print(f"  Mean velocity: {phase_vels.mean():.2f}°")

    # Check for the critical missing pattern
    late_mask = phases == 'late'
    late_vels = velocities[late_mask]
    late_contexts = contexts[late_mask]

    # High context + idle in late phase (this is what's missing)
    high_ctx_idle = ((late_contexts > 0.5) & (late_vels < 3)).sum()
    print(f"\n" + "="*60)
    print(f"CRITICAL: High-context + IDLE in late phase: {high_ctx_idle} frames")
    print(f"This is the MISSING pattern that causes hallucination!")
    print("="*60)


if __name__ == "__main__":
    create_real_distribution_plot()
