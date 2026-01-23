#!/usr/bin/env python3
"""
Visualize P(trajectory | velocity_field): How v drives action generation in flow matching.

The flow matching process:
1. Start with noise x_0
2. Compute v(x_t, t, K) via cross-attention to KV cache
3. Update: x_{t+dt} = x_t + dt * v
4. After T steps, x_T is the final trajectory

This visualization shows how different v fields lead to different trajectories.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path

OUTPUT_DIR = Path("/home/jrobot/project/lerobot/logs/investigation/first_principles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load trace data
HALLUC_TRACE = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl")
NORMAL_TRACE = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate/trace.jsonl")


def load_trace(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_flow_matching_mechanism_figure():
    """
    Create comprehensive visualization of P(traj | v) - how velocity field drives trajectory.
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("P(trajectory | velocity_field): How Flow Matching Generates Actions\n" +
                 "The velocity field v(x_t, t, K) determines trajectory direction and magnitude",
                 fontsize=14, fontweight='bold')

    # === Panel 1: Flow Matching Process Diagram ===
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Flow Matching: From Noise to Trajectory', fontsize=11, fontweight='bold')

    # Draw the denoising steps
    steps_x = [1, 3, 5, 7, 9]
    steps_y = [5, 5, 5, 5, 5]
    labels = ['x₀\n(noise)', 'x₁', 'x₂', '...', 'x_T\n(action)']
    colors = ['lightblue', 'lightyellow', 'lightyellow', 'white', 'lightgreen']

    for i, (x, y, label, color) in enumerate(zip(steps_x, steps_y, labels, colors)):
        circle = plt.Circle((x, y), 0.8, color=color, ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        if i < len(steps_x) - 1:
            # Arrow between steps
            ax1.annotate('', xy=(steps_x[i+1] - 0.9, y), xytext=(x + 0.9, y),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
            # v label
            ax1.text((x + steps_x[i+1]) / 2, y + 0.5, 'v', fontsize=10, ha='center',
                    style='italic', color='red')

    # Update equation
    ax1.text(5, 2.5, r'$x_{t+dt} = x_t + dt \cdot v(x_t, t, K)$',
            fontsize=12, ha='center', family='serif',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

    # Explanation
    ax1.text(5, 1, 'v determines direction & magnitude\nof each denoising step',
            fontsize=10, ha='center', style='italic')

    # === Panel 2: Cross-Attention Computes v ===
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('How v is Computed: Cross-Attention', fontsize=11, fontweight='bold')

    # KV Cache box
    ax2.add_patch(plt.Rectangle((0.5, 6), 4, 3, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax2.text(2.5, 8.5, 'KV Cache (K)', fontsize=10, ha='center', fontweight='bold')
    ax2.text(2.5, 7.5, 'K_head (α=HIGH)', fontsize=9, ha='center', color='blue')
    ax2.text(2.5, 7, 'K_wrist (α=LOW)', fontsize=9, ha='center', color='red')
    ax2.text(2.5, 6.5, 'K_lang, K_state', fontsize=9, ha='center', color='gray')

    # Action tokens box
    ax2.add_patch(plt.Rectangle((5.5, 6), 4, 3, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax2.text(7.5, 8.5, 'Action Tokens', fontsize=10, ha='center', fontweight='bold')
    ax2.text(7.5, 7.5, 'x_t (noisy traj)', fontsize=9, ha='center')
    ax2.text(7.5, 7, 't (timestep)', fontsize=9, ha='center')

    # Cross-attention arrow
    ax2.annotate('', xy=(5.5, 7.5), xytext=(4.5, 7.5),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=3))
    ax2.text(5, 8.2, 'Cross\nAttention', fontsize=9, ha='center', color='purple', fontweight='bold')

    # Output v
    ax2.add_patch(plt.Rectangle((3, 2), 4, 2.5, fill=True, facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax2.text(5, 4, 'v(x_t, t, K)', fontsize=11, ha='center', fontweight='bold')
    ax2.text(5, 3.2, 'Velocity Field', fontsize=10, ha='center')
    ax2.text(5, 2.5, '(direction + magnitude)', fontsize=9, ha='center', style='italic')

    # Arrow down to v
    ax2.annotate('', xy=(5, 4.5), xytext=(5, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Formula
    ax2.text(5, 0.8, r'$v = f(\sum_r \alpha_r \cdot \text{Attn}(Q_{action}, K_r, V_r))$',
            fontsize=10, ha='center', family='serif',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # === Panel 3: v Field Comparison ===
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title('Velocity Field Comparison: Halluc vs Normal', fontsize=11, fontweight='bold')

    # Create a 2D action space visualization
    # X-axis: Joint 1 action, Y-axis: Joint 2 action
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)

    # Halluc case: v points toward movement (upper right)
    U_halluc = 0.3 * np.ones_like(X)
    V_halluc = 0.2 * np.ones_like(Y)

    # Normal case: v points toward zero (staying still)
    U_normal = -0.1 * X
    V_normal = -0.1 * Y

    # Plot both
    ax3.quiver(X - 0.05, Y, U_halluc, V_halluc, color='red', alpha=0.7, scale=5, label='Halluc v')
    ax3.quiver(X + 0.05, Y, U_normal, V_normal, color='green', alpha=0.7, scale=5, label='Normal v')

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)

    ax3.set_xlabel('Action Dimension 1 (e.g., Joint 1)', fontsize=10)
    ax3.set_ylabel('Action Dimension 2 (e.g., Joint 2)', fontsize=10)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.legend(loc='upper left', fontsize=9)

    # Annotations
    ax3.annotate('Halluc: v → movement\n(positive direction)',
                xy=(0.5, 0.5), fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
    ax3.annotate('Normal: v → origin\n(stay still)',
                xy=(-0.8, -0.8), fontsize=9, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen'))

    # === Panel 4: Trajectory Integration ===
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title('Trajectory Integration: x_T = x_0 + ∫v dt', fontsize=11, fontweight='bold')

    # Simulate trajectory integration
    np.random.seed(42)
    T = 10  # denoising steps
    dt = 1.0 / T

    # Start from same noise
    x0 = np.array([0.5, 0.3])

    # Halluc trajectory (v pushes toward movement)
    traj_halluc = [x0.copy()]
    x = x0.copy()
    for t in range(T):
        v = np.array([0.08, 0.05]) + np.random.randn(2) * 0.01  # v toward movement
        x = x + dt * v * 10  # scale for visibility
        traj_halluc.append(x.copy())
    traj_halluc = np.array(traj_halluc)

    # Normal trajectory (v pushes toward zero)
    traj_normal = [x0.copy()]
    x = x0.copy()
    for t in range(T):
        v = -0.05 * x + np.random.randn(2) * 0.01  # v toward origin
        x = x + dt * v * 10
        traj_normal.append(x.copy())
    traj_normal = np.array(traj_normal)

    # Plot trajectories
    ax4.plot(traj_halluc[:, 0], traj_halluc[:, 1], 'r-o', linewidth=2, markersize=5, label='Halluc trajectory')
    ax4.plot(traj_normal[:, 0], traj_normal[:, 1], 'g-o', linewidth=2, markersize=5, label='Normal trajectory')

    # Mark start and end
    ax4.scatter([x0[0]], [x0[1]], s=200, c='blue', marker='*', zorder=5, label='Start (noise)')
    ax4.scatter([traj_halluc[-1, 0]], [traj_halluc[-1, 1]], s=150, c='red', marker='s', zorder=5)
    ax4.scatter([traj_normal[-1, 0]], [traj_normal[-1, 1]], s=150, c='green', marker='s', zorder=5)

    # IDLE region
    idle_circle = plt.Circle((0, 0), 0.3, fill=True, facecolor='lightblue', alpha=0.3, edgecolor='blue', linestyle='--')
    ax4.add_patch(idle_circle)
    ax4.text(0, 0, 'IDLE\nregion', ha='center', va='center', fontsize=8, color='blue')

    ax4.set_xlabel('Action Dimension 1', fontsize=10)
    ax4.set_ylabel('Action Dimension 2', fontsize=10)
    ax4.set_xlim(-0.5, 1.5)
    ax4.set_ylim(-0.5, 1.0)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    ax4.annotate('Halluc ends\nOUTSIDE idle',
                xy=(traj_halluc[-1, 0], traj_halluc[-1, 1]),
                xytext=(1.2, 0.7), fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax4.annotate('Normal ends\nIN idle',
                xy=(traj_normal[-1, 0], traj_normal[-1, 1]),
                xytext=(-0.3, 0.6), fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    # === Panel 5: Real Trajectory Data ===
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title('Real Data: Action Magnitude Over Denoising Steps', fontsize=11, fontweight='bold')

    # Load real trace data at a specific inference step
    halluc = load_trace(HALLUC_TRACE)
    normal = load_trace(NORMAL_TRACE)

    # Get post-completion steps
    h_steps = [e['step'] for e in halluc if 200 <= e['step'] <= 300]
    h_deltas = [e['action_delta_max'] for e in halluc if 200 <= e['step'] <= 300]
    n_steps = [e['step'] for e in normal if 200 <= e['step'] <= 300]
    n_deltas = [e['action_delta_max'] for e in normal if 200 <= e['step'] <= 300]

    ax5.plot(h_steps, h_deltas, 'r-', linewidth=2, label='Halluc |v| (high magnitude)')
    ax5.plot(n_steps, n_deltas, 'g-', linewidth=2, label='Normal |v| (low magnitude)')

    ax5.axhline(y=3, color='black', linestyle='--', label='IDLE threshold (3°)')
    ax5.fill_between([200, 300], [0, 0], [3, 3], alpha=0.2, color='lightblue')
    ax5.fill_between([200, 300], [3, 3], [15, 15], alpha=0.1, color='lightsalmon')

    ax5.set_xlabel('Inference Step', fontsize=10)
    ax5.set_ylabel('|v| = Action Delta (degrees)', fontsize=10)
    ax5.set_xlim(200, 300)
    ax5.set_ylim(0, 15)
    ax5.legend(loc='upper right', fontsize=9)

    ax5.text(250, 13, 'Halluc: |v| > threshold\n→ MOVEMENT output',
            ha='center', fontsize=9, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose'))
    ax5.text(250, 1, 'Normal: |v| < threshold\n→ IDLE output',
            ha='center', fontsize=9, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    # === Panel 6: Complete Causal Chain ===
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    ax6.set_title('Complete Chain: K → v → trajectory', fontsize=11, fontweight='bold')

    # Draw the complete chain
    chain = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                   P(trajectory | v | K)                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   K (KV Cache)         v (Velocity)         τ (Trajectory)     │
    │   ┌─────────┐          ┌─────────┐          ┌─────────┐        │
    │   │ Visual  │ ──────▶  │Direction│ ──────▶  │ Final   │        │
    │   │ Context │ Cross    │   +     │ Integrate│ Action  │        │
    │   │         │ Attn     │Magnitude│          │ Chunk   │        │
    │   └─────────┘          └─────────┘          └─────────┘        │
    │                                                                 │
    │   HALLUC CASE:                                                  │
    │   K encodes "obj on workspace" → v = HIGH magnitude             │
    │   → trajectory = MOVEMENT                                       │
    │                                                                 │
    │   NORMAL CASE:                                                  │
    │   K encodes "empty workspace" → v = LOW magnitude               │
    │   → trajectory = IDLE                                           │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   WHY HALLUC HAS HIGH |v|:                                      │
    │   ─────────────────────────                                     │
    │   1. Head camera sees "object on workspace"                     │
    │   2. Cross-attention weights α_head = HIGH                      │
    │   3. Training taught: obj_visible → move                        │
    │   4. v magnitude is HIGH → trajectory moves                     │
    │                                                                 │
    │   WHY NORMAL HAS LOW |v|:                                       │
    │   ─────────────────────────                                     │
    │   1. Head camera sees "empty workspace"                         │
    │   2. Training taught: empty → stay still                        │
    │   3. v magnitude is LOW → trajectory stays near origin          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """
    ax6.text(0.02, 0.98, chain, transform=ax6.transAxes, fontsize=8,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "p_trajectory_given_velocity_field.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'p_trajectory_given_velocity_field.png'}")
    plt.close()


def create_denoising_steps_visualization():
    """
    Create step-by-step visualization of the 10 denoising steps.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Flow Matching Denoising: 10 Steps from Noise to Action\n" +
                 "Each step: x_{t+1} = x_t + dt · v(x_t, t, K)",
                 fontsize=14, fontweight='bold')

    # Simulate 10 denoising steps for both cases
    np.random.seed(42)
    T = 10
    dt = 1.0 / T

    # Same starting noise
    x0 = np.random.randn(2) * 0.5

    # Halluc: v consistently positive (toward movement)
    halluc_traj = [x0.copy()]
    x = x0.copy()
    for t in range(T):
        time = 1.0 - t * dt  # t goes from 1 to 0
        v_halluc = np.array([0.15, 0.1]) * time + np.random.randn(2) * 0.02
        x = x + dt * v_halluc * 15
        halluc_traj.append(x.copy())

    # Normal: v toward zero (staying still)
    normal_traj = [x0.copy()]
    x = x0.copy()
    for t in range(T):
        time = 1.0 - t * dt
        v_normal = -0.1 * x * time + np.random.randn(2) * 0.02
        x = x + dt * v_normal * 15
        normal_traj.append(x.copy())

    halluc_traj = np.array(halluc_traj)
    normal_traj = np.array(normal_traj)

    # Plot each step
    for i in range(10):
        ax = axes[i // 5, i % 5]

        # Plot accumulated trajectory up to this step
        ax.plot(halluc_traj[:i+2, 0], halluc_traj[:i+2, 1], 'r-o', linewidth=2, markersize=4, label='Halluc')
        ax.plot(normal_traj[:i+2, 0], normal_traj[:i+2, 1], 'g-o', linewidth=2, markersize=4, label='Normal')

        # Current position
        ax.scatter([halluc_traj[i+1, 0]], [halluc_traj[i+1, 1]], s=100, c='red', marker='s', zorder=5)
        ax.scatter([normal_traj[i+1, 0]], [normal_traj[i+1, 1]], s=100, c='green', marker='s', zorder=5)

        # IDLE region
        idle_circle = plt.Circle((0, 0), 0.5, fill=True, facecolor='lightblue', alpha=0.3, edgecolor='blue', linestyle='--')
        ax.add_patch(idle_circle)

        # Velocity arrows
        if i < 9:
            v_h = (halluc_traj[i+2] - halluc_traj[i+1]) * 3
            v_n = (normal_traj[i+2] - normal_traj[i+1]) * 3
            ax.arrow(halluc_traj[i+1, 0], halluc_traj[i+1, 1], v_h[0], v_h[1],
                    head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.7)
            ax.arrow(normal_traj[i+1, 0], normal_traj[i+1, 1], v_n[0], v_n[1],
                    head_width=0.1, head_length=0.05, fc='green', ec='green', alpha=0.7)

        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-1.5, 2)
        ax.set_title(f'Step {i+1}/10 (t={1.0 - i/10:.1f})', fontsize=10)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "denoising_steps_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'denoising_steps_visualization.png'}")
    plt.close()


def main():
    print("Creating flow matching visualizations...")
    create_flow_matching_mechanism_figure()
    create_denoising_steps_visualization()
    print("\nAll visualizations created!")


if __name__ == "__main__":
    main()
