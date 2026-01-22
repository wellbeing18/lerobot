#!/usr/bin/env python3
"""
Visualize the training data distribution bias that causes hallucination.

Key insight: P(movement | object_on_workspace) = 1.0 in training data
            P(idle | object_on_workspace) = 0.0 in training data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("/home/jrobot/project/lerobot/logs/investigation/first_principles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_distribution_bias_figure():
    """Create a clear visualization of the training distribution bias."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === Plot 1: Training Data Distribution ===
    ax1 = axes[0]
    ax1.set_title("Training Data Distribution\n(What the model learned)", fontsize=12, fontweight='bold')

    # Create bar chart
    contexts = ['Object on\nWorkspace', 'Empty\nWorkspace']
    movement_pct = [100, 0]  # P(movement | context)
    idle_pct = [0, 100]      # P(idle | context)

    x = np.arange(len(contexts))
    width = 0.35

    bars1 = ax1.bar(x - width/2, movement_pct, width, label='MOVEMENT', color='coral')
    bars2 = ax1.bar(x + width/2, idle_pct, width, label='IDLE', color='lightblue')

    ax1.set_ylabel('P(behavior | context) %', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Annotations
    ax1.annotate('100%', xy=(x[0] - width/2, 100), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.annotate('0%', xy=(x[0] + width/2, 0), ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    ax1.annotate('0%', xy=(x[1] - width/2, 0), ha='center', va='bottom', fontsize=10)
    ax1.annotate('100%', xy=(x[1] + width/2, 100), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight the missing case
    ax1.add_patch(plt.Rectangle((x[0] + width/2 - width/2, -5), width, 10,
                                 fill=False, edgecolor='red', linewidth=3, linestyle='--'))
    ax1.text(x[0] + width/2, -12, 'MISSING!\n(0 examples)', ha='center', fontsize=9, color='red', fontweight='bold')

    # === Plot 2: Inference Scenario ===
    ax2 = axes[1]
    ax2.set_title("Inference Scenario\n(What the model encounters)", fontsize=12, fontweight='bold')

    # Show the test case
    scenarios = ['Halluc Case\n(banana on table)', 'Normal Case\n(banana on plate)']
    colors = ['coral', 'lightblue']

    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        ax2.add_patch(plt.Rectangle((0.1, 0.6 - i*0.5), 0.8, 0.35,
                                    fill=True, facecolor=color, edgecolor='black', linewidth=2))
        ax2.text(0.5, 0.78 - i*0.5, scenario, ha='center', va='center', fontsize=11, fontweight='bold')

        if i == 0:
            ax2.text(0.5, 0.68 - i*0.5, 'Visual: Object on workspace\nExpected: IDLE (task done)\nModel outputs: MOVEMENT',
                    ha='center', va='center', fontsize=9)
            ax2.annotate('MISMATCH!', xy=(0.92, 0.75), fontsize=10, color='red', fontweight='bold')
        else:
            ax2.text(0.5, 0.68 - i*0.5, 'Visual: Empty workspace\nExpected: IDLE\nModel outputs: IDLE',
                    ha='center', va='center', fontsize=9)
            ax2.annotate('CORRECT', xy=(0.92, 0.25), fontsize=10, color='green', fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # === Plot 3: The Gap ===
    ax3 = axes[2]
    ax3.set_title("The Distribution Gap\n(Root cause of hallucination)", fontsize=12, fontweight='bold')

    # Show conditional probabilities
    ax3.text(0.5, 0.9, "Training Data Conditional Distributions:", ha='center', fontsize=11, fontweight='bold')

    # P(behavior | object on workspace)
    ax3.text(0.5, 0.75, "P(movement | object_on_workspace) = 1.0", ha='center', fontsize=11,
             family='monospace', color='coral')
    ax3.text(0.5, 0.65, "P(idle | object_on_workspace) = 0.0", ha='center', fontsize=11,
             family='monospace', color='red', fontweight='bold')

    # The implication
    ax3.add_patch(plt.Rectangle((0.1, 0.35), 0.8, 0.2, fill=True, facecolor='lightyellow', edgecolor='black'))
    ax3.text(0.5, 0.45, "Model learned: 'object on workspace' → MOVE", ha='center', fontsize=11, fontweight='bold')

    # The fix
    ax3.text(0.5, 0.2, "NEEDED: Training examples where", ha='center', fontsize=10)
    ax3.text(0.5, 0.1, "P(idle | irrelevant_object_on_workspace) > 0", ha='center', fontsize=11,
             family='monospace', color='green', fontweight='bold')

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_distribution_bias.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'training_distribution_bias.png'}")
    plt.close()


def create_mechanism_flow_figure():
    """Create a flow diagram showing how bias propagates."""

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(8, 7.5, "HOW TRAINING BIAS CAUSES HALLUCINATION",
            fontsize=14, fontweight='bold', ha='center')

    # Step 1: Training Data
    ax.add_patch(plt.Rectangle((0.5, 5), 3, 1.5, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(2, 6.1, "TRAINING DATA", fontsize=10, ha='center', fontweight='bold')
    ax.text(2, 5.6, "Object on workspace\n→ Always MOVEMENT", fontsize=8, ha='center')
    ax.text(2, 5.2, "Empty workspace\n→ Always IDLE", fontsize=8, ha='center')

    # Step 2: Learned Distribution
    ax.add_patch(plt.Rectangle((5, 5), 3, 1.5, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(6.5, 6.1, "LEARNED P(τ|K)", fontsize=10, ha='center', fontweight='bold')
    ax.text(6.5, 5.6, "P(move|obj) = 1.0", fontsize=9, ha='center', family='monospace')
    ax.text(6.5, 5.2, "P(idle|obj) = 0.0", fontsize=9, ha='center', family='monospace', color='red')

    # Step 3: Cross-Attention Weights
    ax.add_patch(plt.Rectangle((9.5, 5), 3, 1.5, fill=True, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(11, 6.1, "ATTENTION WEIGHTS", fontsize=10, ha='center', fontweight='bold')
    ax.text(11, 5.6, "α_head = HIGH (167%)", fontsize=9, ha='center')
    ax.text(11, 5.2, "α_wrist = LOW (11%)", fontsize=9, ha='center')

    # Step 4: Inference
    ax.add_patch(plt.Rectangle((13, 5), 2.5, 1.5, fill=True, facecolor='lightsalmon', edgecolor='black', linewidth=2))
    ax.text(14.25, 6.1, "INFERENCE", fontsize=10, ha='center', fontweight='bold')
    ax.text(14.25, 5.5, "Head sees object\n→ MOVEMENT", fontsize=9, ha='center')

    # Arrows
    for x_start, x_end in [(3.5, 5), (8, 9.5), (12.5, 13)]:
        ax.annotate('', xy=(x_end, 5.75), xytext=(x_start, 5.75),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # The problem box
    ax.add_patch(plt.Rectangle((0.5, 1.5), 15, 2.5, fill=True, facecolor='mistyrose', edgecolor='red', linewidth=2))
    ax.text(8, 3.7, "THE PROBLEM", fontsize=12, ha='center', fontweight='bold', color='darkred')

    ax.text(8, 3.1, "At inference: Banana (irrelevant object) is on workspace", fontsize=10, ha='center')
    ax.text(8, 2.5, "Head camera encodes: 'Object present on workspace'", fontsize=10, ha='center')
    ax.text(8, 1.9, "Model applies learned rule: P(move|obj_on_workspace) = 1.0 → HALLUCINATION", fontsize=10, ha='center', fontweight='bold', color='red')

    # The solution box
    ax.add_patch(plt.Rectangle((5, 0.2), 6, 1, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(8, 0.7, "FIX: Add training data where P(idle | irrelevant_obj) > 0", fontsize=10, ha='center', fontweight='bold', color='darkgreen')

    plt.savefig(OUTPUT_DIR / "bias_propagation_flow.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'bias_propagation_flow.png'}")
    plt.close()


def main():
    create_distribution_bias_figure()
    create_mechanism_flow_figure()
    print("\nVisualizations created!")


if __name__ == "__main__":
    main()
