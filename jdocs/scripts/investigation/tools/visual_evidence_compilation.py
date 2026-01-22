#!/usr/bin/env python3
"""
Create a comprehensive visual evidence compilation for the hallucination investigation.
Shows the key visual differences between halluc and normal cases.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

HALLUC_DIR = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/images")
NORMAL_DIR = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate/images")
OUTPUT_DIR = Path("/home/jrobot/project/lerobot/logs/investigation/visual_evidence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_comparison_figure():
    """Create side-by-side comparison of key frames."""

    # Key steps to compare
    steps = [0, 150, 200, 250, 300, 350]
    cameras = ["head", "right_wrist"]

    fig, axes = plt.subplots(len(steps), 4, figsize=(20, 30))
    fig.suptitle("Hallucination vs Normal Case: Visual Comparison\n(Banana on TABLE vs Banana on PLATE)",
                 fontsize=16, fontweight='bold')

    for row, step in enumerate(steps):
        for col, (case, case_dir, label) in enumerate([
            ("Halluc", HALLUC_DIR, f"Halluc Step {step}"),
            ("Normal", NORMAL_DIR, f"Normal Step {step}")
        ]):
            # Head camera
            head_path = case_dir / f"step_{step:04d}_head.jpg"
            if head_path.exists():
                img = mpimg.imread(head_path)
                ax = axes[row, col * 2]
                ax.imshow(img)
                ax.set_title(f"{case} - Head (Step {step})", fontsize=10)
                ax.axis('off')

                # Add annotations for step 200 (divergence point)
                if step == 200:
                    ax.set_title(f"{case} - Head (Step {step}) ★ DIVERGENCE", fontsize=10, color='red')

            # Right wrist camera
            wrist_path = case_dir / f"step_{step:04d}_right_wrist.jpg"
            if wrist_path.exists():
                img = mpimg.imread(wrist_path)
                ax = axes[row, col * 2 + 1]
                ax.imshow(img)
                ax.set_title(f"{case} - R.Wrist (Step {step})", fontsize=10)
                ax.axis('off')

                if step == 200:
                    ax.set_title(f"{case} - R.Wrist (Step {step}) ★ DIVERGENCE", fontsize=10, color='red')

    # Add column labels
    col_labels = ["Halluc Head", "Halluc R.Wrist", "Normal Head", "Normal R.Wrist"]
    for col, label in enumerate(col_labels):
        axes[0, col].annotate(label, xy=(0.5, 1.15), xycoords='axes fraction',
                             fontsize=12, ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / "full_comparison_all_steps.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'full_comparison_all_steps.png'}")
    plt.close()


def create_key_difference_figure():
    """Create focused figure showing the key visual difference at divergence point."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("KEY FINDING: Banana Location Determines Behavior\n" +
                 "Step 200 (Divergence Point) - Task Already Completed",
                 fontsize=14, fontweight='bold')

    # Row 0: HALLUC case
    # Head camera
    img = mpimg.imread(HALLUC_DIR / "step_0200_head.jpg")
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("HALLUC: Head Camera\nBanana on TABLE (workspace)", fontsize=11, color='red')
    axes[0, 0].axis('off')

    # Right wrist
    img = mpimg.imread(HALLUC_DIR / "step_0200_right_wrist.jpg")
    axes[0, 1].imshow(img)
    axes[0, 1].set_title("HALLUC: Right Wrist\nBanana VISIBLE on workspace", fontsize=11, color='red')
    axes[0, 1].axis('off')

    # Step 350 showing movement
    img = mpimg.imread(HALLUC_DIR / "step_0350_head.jpg")
    axes[0, 2].imshow(img)
    axes[0, 2].set_title("HALLUC: Step 350\nArm REACHES toward bin (HALLUCINATION)", fontsize=11, color='red')
    axes[0, 2].axis('off')

    # Row 1: NORMAL case
    # Head camera
    img = mpimg.imread(NORMAL_DIR / "step_0200_head.jpg")
    axes[1, 0].imshow(img)
    axes[1, 0].set_title("NORMAL: Head Camera\nBanana on PLATE (destination)", fontsize=11, color='blue')
    axes[1, 0].axis('off')

    # Right wrist
    img = mpimg.imread(NORMAL_DIR / "step_0200_right_wrist.jpg")
    axes[1, 1].imshow(img)
    axes[1, 1].set_title("NORMAL: Right Wrist\nWorkspace is EMPTY", fontsize=11, color='blue')
    axes[1, 1].axis('off')

    # Step 350 showing idle
    img = mpimg.imread(NORMAL_DIR / "step_0350_head.jpg")
    axes[1, 2].imshow(img)
    axes[1, 2].set_title("NORMAL: Step 350\nArm STAYS in rest position (CORRECT)", fontsize=11, color='blue')
    axes[1, 2].axis('off')

    # Add row labels
    axes[0, 0].annotate("HALLUCINATION\nCASE", xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=12, ha='center', va='center', rotation=90, fontweight='bold', color='red')
    axes[1, 0].annotate("NORMAL\nCASE", xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=12, ha='center', va='center', rotation=90, fontweight='bold', color='blue')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])

    # Add explanatory text at bottom
    fig.text(0.5, 0.02,
             "THE KEY DIFFERENCE: When banana is on TABLE (workspace), model sees 'graspable object' → triggers movement.\n" +
             "When banana is on PLATE (destination), workspace is empty → model correctly stays idle.",
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.savefig(OUTPUT_DIR / "key_visual_difference.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'key_visual_difference.png'}")
    plt.close()


def create_timeline_figure():
    """Create timeline showing behavior divergence."""

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Behavior Timeline: Halluc vs Normal\n" +
                 "(Showing how behavior diverges after task completion)",
                 fontsize=14, fontweight='bold')

    steps = [150, 200, 250, 300, 350]

    # Row 0: Halluc head camera
    for col, step in enumerate(steps):
        img = mpimg.imread(HALLUC_DIR / f"step_{step:04d}_head.jpg")
        axes[0, col].imshow(img)
        title = f"Step {step}"
        if step == 150:
            title += "\nTask completing"
        elif step == 200:
            title += "\n★ DIVERGENCE"
        elif step >= 250:
            title += "\nMoving!"
        axes[0, col].set_title(title, fontsize=10, color='red' if step >= 200 else 'black')
        axes[0, col].axis('off')

    # Row 1: Normal head camera
    for col, step in enumerate(steps):
        img = mpimg.imread(NORMAL_DIR / f"step_{step:04d}_head.jpg")
        axes[1, col].imshow(img)
        title = f"Step {step}"
        if step == 150:
            title += "\nTask completing"
        elif step == 200:
            title += "\n★ DIVERGENCE"
        elif step >= 250:
            title += "\nIDLE (correct)"
        axes[1, col].set_title(title, fontsize=10, color='blue' if step >= 200 else 'black')
        axes[1, col].axis('off')

    # Row 2: Right wrist comparison at key steps
    key_steps = [150, 200, 250, 300, 350]
    for col, step in enumerate(key_steps):
        # Create split view: halluc on left, normal on right
        fig_sub = fig.add_axes([0.02 + col * 0.19, 0.02, 0.18, 0.28])

        h_img = mpimg.imread(HALLUC_DIR / f"step_{step:04d}_right_wrist.jpg")
        n_img = mpimg.imread(NORMAL_DIR / f"step_{step:04d}_right_wrist.jpg")

        # Concatenate horizontally
        combined = np.concatenate([h_img, n_img], axis=1)
        axes[2, col].imshow(combined)
        axes[2, col].set_title(f"R.Wrist Step {step}\n(Halluc | Normal)", fontsize=9)
        axes[2, col].axis('off')
        axes[2, col].axvline(x=combined.shape[1]//2, color='white', linewidth=2)

    # Add row labels
    axes[0, 0].annotate("HALLUC\n(banana on table)", xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=11, ha='center', va='center', rotation=90, fontweight='bold', color='red')
    axes[1, 0].annotate("NORMAL\n(banana on plate)", xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=11, ha='center', va='center', rotation=90, fontweight='bold', color='blue')
    axes[2, 0].annotate("RIGHT WRIST\nCOMPARISON", xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=11, ha='center', va='center', rotation=90, fontweight='bold')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "behavior_timeline.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'behavior_timeline.png'}")
    plt.close()


def main():
    print("Creating visual evidence compilation...")

    create_key_difference_figure()
    create_comparison_figure()
    create_timeline_figure()

    print("\n=== VISUAL EVIDENCE SUMMARY ===")
    print("The images clearly show:")
    print("1. HALLUC case: Banana on TABLE (workspace) → arm moves toward bin area")
    print("2. NORMAL case: Banana on PLATE (destination) → arm stays in rest position")
    print("")
    print("The KEY VISUAL DIFFERENCE is the banana's location:")
    print("  - On TABLE: Workspace has graspable object → triggers movement")
    print("  - On PLATE: Workspace is empty → correct idle behavior")
    print("")
    print(f"All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
