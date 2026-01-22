#!/usr/bin/env python3
"""
Analyze trajectory divergence between hallucination and normal cases.
Find exactly when and why the behaviors diverge.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HALLUC_TRACE = "/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl"
NORMAL_TRACE = "/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate/trace.jsonl"
BASELINE_TRACE = "/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl"

OUTPUT_DIR = Path("/home/jrobot/project/lerobot/logs/investigation/trajectory_divergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_trace(path):
    """Load trace data from jsonl file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_metrics(trace_data):
    """Extract key metrics from trace data."""
    steps = []
    action_deltas = []
    states = []
    actions = []

    for entry in trace_data:
        steps.append(entry["step"])
        action_deltas.append(entry["action_delta_max"])
        states.append(entry["state_normalized"])
        actions.append(entry["action_raw"])

    return {
        "steps": np.array(steps),
        "action_deltas": np.array(action_deltas),
        "states": np.array(states),
        "actions": np.array(actions)
    }

def main():
    print("Loading traces...")
    halluc_data = load_trace(HALLUC_TRACE)
    normal_data = load_trace(NORMAL_TRACE)

    # Check if baseline exists
    try:
        baseline_data = load_trace(BASELINE_TRACE)
        has_baseline = True
    except:
        has_baseline = False
        baseline_data = None

    print(f"Halluc trace: {len(halluc_data)} steps")
    print(f"Normal trace: {len(normal_data)} steps")
    if has_baseline:
        print(f"Baseline trace: {len(baseline_data)} steps")

    # Extract metrics
    halluc = extract_metrics(halluc_data)
    normal = extract_metrics(normal_data)
    if has_baseline:
        baseline = extract_metrics(baseline_data)

    # === Figure 1: Action Delta Over Time ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Full trajectory
    ax1 = axes[0]
    ax1.plot(halluc["steps"], halluc["action_deltas"], 'r-', label='Halluc (banana on table)', alpha=0.8)
    ax1.plot(normal["steps"], normal["action_deltas"], 'b-', label='Normal (banana on plate)', alpha=0.8)
    if has_baseline:
        ax1.plot(baseline["steps"], baseline["action_deltas"], 'g-', label='Baseline (no banana)', alpha=0.8)
    ax1.axhline(y=3.0, color='k', linestyle='--', label='IDLE threshold (3°)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Action Delta Max (degrees)')
    ax1.set_title('Full Trajectory: Action Magnitude Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed to post-completion (step 150-300)
    ax2 = axes[1]
    mask_h = (halluc["steps"] >= 150) & (halluc["steps"] <= 350)
    mask_n = (normal["steps"] >= 150) & (normal["steps"] <= 350)
    ax2.plot(halluc["steps"][mask_h], halluc["action_deltas"][mask_h], 'r-', label='Halluc', linewidth=2)
    ax2.plot(normal["steps"][mask_n], normal["action_deltas"][mask_n], 'b-', label='Normal', linewidth=2)
    if has_baseline:
        mask_b = (baseline["steps"] >= 150) & (baseline["steps"] <= 350)
        ax2.plot(baseline["steps"][mask_b], baseline["action_deltas"][mask_b], 'g-', label='Baseline', linewidth=2)
    ax2.axhline(y=3.0, color='k', linestyle='--', label='IDLE threshold')
    ax2.axvline(x=200, color='purple', linestyle=':', label='Divergence point')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Action Delta Max (degrees)')
    ax2.set_title('Post-Completion Phase (Steps 150-350): WHERE DIVERGENCE OCCURS')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative action delta (total movement)
    ax3 = axes[2]
    halluc_cumsum = np.cumsum(halluc["action_deltas"])
    normal_cumsum = np.cumsum(normal["action_deltas"])
    ax3.plot(halluc["steps"], halluc_cumsum, 'r-', label='Halluc', linewidth=2)
    ax3.plot(normal["steps"], normal_cumsum, 'b-', label='Normal', linewidth=2)
    if has_baseline:
        baseline_cumsum = np.cumsum(baseline["action_deltas"])
        ax3.plot(baseline["steps"], baseline_cumsum, 'g-', label='Baseline', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Action Delta (degrees)')
    ax3.set_title('Total Movement Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_divergence_comparison.png", dpi=150)
    print(f"Saved trajectory comparison to {OUTPUT_DIR / 'trajectory_divergence_comparison.png'}")
    plt.close()

    # === Analysis: Find exact divergence point ===
    print("\n=== DIVERGENCE ANALYSIS ===")

    # Find where halluc exceeds IDLE threshold and normal doesn't
    IDLE_THRESHOLD = 3.0

    # Use shorter length for comparison
    min_len = min(len(halluc["action_deltas"]), len(normal["action_deltas"]))

    divergence_step = None
    for i in range(150, min(300, min_len)):
        h_delta = halluc["action_deltas"][i]
        n_delta = normal["action_deltas"][i]

        # Halluc is moving, normal is idle
        if h_delta > IDLE_THRESHOLD and n_delta < IDLE_THRESHOLD:
            if divergence_step is None:
                divergence_step = i
                print(f"\nFirst divergence at step {i}:")
                print(f"  Halluc action delta: {h_delta:.2f}°")
                print(f"  Normal action delta: {n_delta:.2f}°")

    # Compute statistics for post-divergence period
    if divergence_step:
        post_div_h = halluc["action_deltas"][divergence_step:min(divergence_step+100, min_len)]
        post_div_n = normal["action_deltas"][divergence_step:min(divergence_step+100, min_len)]

        print(f"\nPost-divergence statistics (steps {divergence_step}-{divergence_step+100}):")
        print(f"  Halluc - Mean: {post_div_h.mean():.2f}°, Max: {post_div_h.max():.2f}°")
        print(f"  Normal - Mean: {post_div_n.mean():.2f}°, Max: {post_div_n.max():.2f}°")
        print(f"  Halluc/Normal ratio: {post_div_h.mean() / (post_div_n.mean() + 1e-6):.1f}x")

    # === State comparison at divergence point ===
    if divergence_step:
        div_idx = divergence_step
        print(f"\n=== STATE COMPARISON AT DIVERGENCE (step {div_idx}) ===")

        h_state = halluc["states"][div_idx]
        n_state = normal["states"][div_idx]

        state_diff = np.abs(h_state - n_state)
        print(f"Max state difference: {state_diff.max():.4f}")
        print(f"Mean state difference: {state_diff.mean():.4f}")

        # Joint-by-joint comparison
        joint_names = ["j1", "j2", "j3", "j4", "j5", "grip"] * 2  # left + right arm
        print("\nJoint-by-joint state difference:")
        for i, (name, diff) in enumerate(zip(joint_names, state_diff)):
            arm = "Left" if i < 6 else "Right"
            if diff > 0.01:
                print(f"  {arm} {name}: {diff:.4f} (H={h_state[i]:.4f}, N={n_state[i]:.4f})")

    # === Action direction analysis ===
    print("\n=== ACTION DIRECTION ANALYSIS ===")

    if divergence_step:
        # Get action vectors for first 10 steps after divergence
        h_actions = halluc["actions"][divergence_step:divergence_step+10]
        n_actions = normal["actions"][divergence_step:divergence_step+10]

        # Compute mean action direction
        h_mean_action = h_actions.mean(axis=0)
        n_mean_action = n_actions.mean(axis=0)

        # Compute state at step 50 (approximate approach position)
        approach_state = halluc["states"][50]

        # Direction toward approach position
        h_current_state = halluc["states"][divergence_step]
        toward_approach = approach_state - h_current_state
        toward_approach_norm = toward_approach / (np.linalg.norm(toward_approach) + 1e-8)

        # Halluc action direction
        h_action_norm = h_mean_action / (np.linalg.norm(h_mean_action) + 1e-8)

        # Cosine similarity
        cos_sim = np.dot(h_action_norm, toward_approach_norm)
        print(f"Cosine similarity of halluc action vs toward-approach-position: {cos_sim:.3f}")

        if cos_sim > 0.8:
            print(">>> HIGH SIMILARITY: Halluc is moving TOWARD the original approach position!")
        elif cos_sim < -0.5:
            print(">>> NEGATIVE: Halluc is moving AWAY from approach position")
        else:
            print(">>> MODERATE: Direction is not clearly toward/away from approach")

    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Divergence step: {divergence_step if divergence_step else 'Not found'}")
    if divergence_step:
        print(f"Halluc behavior: Continues MOVEMENT after task completion")
        print(f"Normal behavior: Transitions to IDLE after task completion")
        print(f"Movement ratio: Halluc moves {post_div_h.mean() / (post_div_n.mean() + 1e-6):.1f}x more than normal")
    print("="*60)

    # Save analysis report
    with open(OUTPUT_DIR / "divergence_analysis_report.txt", "w") as f:
        f.write("Trajectory Divergence Analysis Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Halluc trace: {len(halluc_data)} steps\n")
        f.write(f"Normal trace: {len(normal_data)} steps\n")
        if divergence_step:
            f.write(f"\nDivergence step: {divergence_step}\n")
            f.write(f"Post-divergence Halluc mean delta: {post_div_h.mean():.2f}°\n")
            f.write(f"Post-divergence Normal mean delta: {post_div_n.mean():.2f}°\n")
            f.write(f"Movement ratio: {post_div_h.mean() / (post_div_n.mean() + 1e-6):.1f}x\n")

    print(f"\nReport saved to {OUTPUT_DIR / 'divergence_analysis_report.txt'}")

if __name__ == "__main__":
    main()
