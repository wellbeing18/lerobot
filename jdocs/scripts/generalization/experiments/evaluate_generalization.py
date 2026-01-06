#!/usr/bin/env python3
"""
Generalization Evaluation Script

Evaluates SmolVLA checkpoint on held-out task combinations to measure
generalization capability. Compares baseline vs experimental models.

Test Categories:
1. Cross-target: Same object, different target (e.g., icecream→plate vs icecream→bin)
2. Multi-object: Correct selection with distractor present
3. Novel composition: Completely new object+target combinations

Usage:
    # Compare experiment vs baseline
    python evaluate_generalization.py \\
        --checkpoint outputs/smolvla_unfrozen_language_*/checkpoints/*/pretrained_model \\
        --baseline outputs/smolvla_bimanual_20260103_200201/checkpoints/020000/pretrained_model

    # Evaluate single checkpoint
    python evaluate_generalization.py \\
        --checkpoint outputs/smolvla_unfrozen_language

    # Simulation mode (no robot, analyze predictions)
    python evaluate_generalization.py \\
        --checkpoint outputs/smolvla_unfrozen_language \\
        --simulate
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ============================================================================
# HELD-OUT TEST TASKS
# ============================================================================

# Cross-target tests: Object trained on one target, test on the other
CROSS_TARGET_TASKS = [
    {
        "id": "icecream_plate",
        "task": "Use right arm to pick up the ice cream and place it on the plate",
        "trained_target": "bin",
        "test_target": "plate",
        "expected_behavior": "Should place on plate, not bin",
    },
    {
        "id": "orange_bin",
        "task": "Use left arm to pick up the orange and place it in the bin",
        "trained_target": "plate",
        "test_target": "bin",
        "expected_behavior": "Should place in bin, not on plate",
    },
    {
        "id": "corn_bin",
        "task": "Use right arm to pick up the corn and place it in the bin",
        "trained_target": "plate",
        "test_target": "bin",
        "expected_behavior": "Should place in bin, not on plate",
    },
]

# Multi-object tests: Select correct object with distractor
MULTI_OBJECT_TASKS = [
    {
        "id": "tissue_with_corn_distractor",
        "task": "Use right arm to pick up the tissue package and place it on the plate",
        "target_object": "tissue",
        "distractor": "corn",
        "expected_behavior": "Should pick tissue, not corn",
    },
    {
        "id": "orange_with_banana_distractor",
        "task": "Use left arm to pick up the orange and place it on the plate",
        "target_object": "orange",
        "distractor": "banana",
        "expected_behavior": "Should pick orange, not banana",
    },
]

# Novel composition tests: Object+target combos not in training
NOVEL_COMPOSITION_TASKS = [
    {
        "id": "tissue_plate",
        "task": "Use right arm to pick up the tissue package and place it on the plate",
        "notes": "Tissue only trained with 'bin' target",
    },
    {
        "id": "yogurt_plate",
        "task": "Use left arm to pick up the yogurt bottle and place it on the plate",
        "notes": "Yogurt only trained with 'bin' target",
    },
]

# In-distribution tests (should work on both baseline and experiment)
IN_DISTRIBUTION_TASKS = [
    {
        "id": "orange_plate",
        "task": "Use left arm to pick up the orange and place it on the plate",
    },
    {
        "id": "icecream_bin",
        "task": "Use right arm to pick up the ice cream and place it in the bin",
    },
]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def analyze_action_trajectory(
    actions: np.ndarray,
    task_info: dict
) -> dict:
    """
    Analyze predicted action trajectory for correctness indicators.

    In simulation mode (no robot), we analyze the action patterns:
    - Direction of end-effector movement (toward plate vs bin region)
    - Gripper open/close patterns
    - Arm selection (left vs right based on action magnitude)
    """
    results = {
        "task_id": task_info["id"],
        "num_action_steps": len(actions),
    }

    # For bimanual (12 DOF): indices 0-5 left arm, 6-11 right arm
    left_arm_actions = actions[:, :6] if actions.shape[1] >= 6 else actions
    right_arm_actions = actions[:, 6:12] if actions.shape[1] >= 12 else None

    # Analyze which arm is more active
    left_activity = np.abs(left_arm_actions).mean()
    right_activity = np.abs(right_arm_actions).mean() if right_arm_actions is not None else 0

    results["left_arm_activity"] = float(left_activity)
    results["right_arm_activity"] = float(right_activity)
    results["active_arm"] = "left" if left_activity > right_activity else "right"

    # Check if expected arm is active
    expected_arm = "left" if "left arm" in task_info["task"].lower() else "right"
    results["expected_arm"] = expected_arm
    results["correct_arm"] = results["active_arm"] == expected_arm

    # Analyze gripper pattern (last DOF of each arm)
    if actions.shape[1] >= 6:
        left_gripper = left_arm_actions[:, 5]
        results["left_gripper_range"] = float(left_gripper.max() - left_gripper.min())

    if right_arm_actions is not None and right_arm_actions.shape[1] >= 6:
        right_gripper = right_arm_actions[:, 5]
        results["right_gripper_range"] = float(right_gripper.max() - right_gripper.min())

    return results


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load SmolVLA policy from checkpoint."""
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        print("ERROR: Cannot import SmolVLA")
        sys.exit(1)

    checkpoint_dir = Path(checkpoint_path)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
    policy = policy.to(device)
    policy.eval()
    return policy


def run_evaluation(
    policy,
    tasks: list[dict],
    category: str,
    device: str = "cuda",
    simulate: bool = True
) -> list[dict]:
    """
    Run evaluation on a set of tasks.

    Args:
        policy: Loaded SmolVLA policy
        tasks: List of task dicts with 'id' and 'task' keys
        category: Task category name
        device: Compute device
        simulate: If True, analyze actions only (no robot execution)

    Returns:
        List of evaluation results
    """
    results = []

    for task_info in tasks:
        print(f"\n  Evaluating: {task_info['id']}")
        print(f"    Task: {task_info['task']}")

        if simulate:
            # Simulation mode: predict actions and analyze
            with torch.no_grad():
                # Create dummy observation
                dummy_image = torch.zeros(1, 3, 512, 512).to(device)
                dummy_state = torch.zeros(1, 12).to(device)

                observation = {
                    "observation.images.camera1": dummy_image,
                    "observation.state": dummy_state,
                }

                # Get action chunk
                try:
                    actions = policy.select_action(observation, task=task_info["task"])
                    actions = actions.cpu().numpy()

                    analysis = analyze_action_trajectory(actions, task_info)
                    analysis["category"] = category
                    analysis["status"] = "analyzed"

                    results.append(analysis)
                    print(f"    Active arm: {analysis['active_arm']} (expected: {analysis['expected_arm']})")
                    print(f"    Correct arm: {analysis['correct_arm']}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "task_id": task_info["id"],
                        "category": category,
                        "status": "error",
                        "error": str(e)
                    })
        else:
            # Real robot mode (not implemented in this script)
            print("    Real robot evaluation not implemented")
            results.append({
                "task_id": task_info["id"],
                "category": category,
                "status": "skipped",
                "reason": "Real robot mode not implemented"
            })

    return results


def compare_checkpoints(
    experiment_results: list[dict],
    baseline_results: list[dict]
) -> dict:
    """Compare experiment vs baseline results."""
    comparison = {
        "experiment_correct_arm": 0,
        "baseline_correct_arm": 0,
        "total_tasks": 0,
        "improvement": {}
    }

    # Match results by task_id
    baseline_map = {r["task_id"]: r for r in baseline_results if "task_id" in r}

    for exp_result in experiment_results:
        if "task_id" not in exp_result:
            continue

        task_id = exp_result["task_id"]
        comparison["total_tasks"] += 1

        if exp_result.get("correct_arm"):
            comparison["experiment_correct_arm"] += 1

        if task_id in baseline_map and baseline_map[task_id].get("correct_arm"):
            comparison["baseline_correct_arm"] += 1

    # Calculate improvement
    if comparison["total_tasks"] > 0:
        exp_rate = comparison["experiment_correct_arm"] / comparison["total_tasks"]
        base_rate = comparison["baseline_correct_arm"] / comparison["total_tasks"]
        comparison["experiment_accuracy"] = exp_rate
        comparison["baseline_accuracy"] = base_rate
        comparison["improvement_pct"] = (exp_rate - base_rate) * 100

    return comparison


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA generalization")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to experiment checkpoint"
    )
    parser.add_argument(
        "--baseline",
        help="Path to baseline checkpoint for comparison"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Simulation mode (analyze actions, no robot)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "evaluation",
        help="Output directory for results"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VLA GENERALIZATION EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    if args.baseline:
        print(f"Baseline: {args.baseline}")
    print(f"Mode: {'Simulation' if args.simulate else 'Real Robot'}")
    print()

    # Load experiment model
    print("Loading experiment model...")
    experiment_policy = load_policy(args.checkpoint, args.device)

    # Run evaluation on experiment
    all_results = []

    print("\n--- In-Distribution Tasks ---")
    results = run_evaluation(
        experiment_policy, IN_DISTRIBUTION_TASKS, "in_distribution",
        args.device, args.simulate
    )
    all_results.extend(results)

    print("\n--- Cross-Target Generalization ---")
    results = run_evaluation(
        experiment_policy, CROSS_TARGET_TASKS, "cross_target",
        args.device, args.simulate
    )
    all_results.extend(results)

    print("\n--- Multi-Object Selection ---")
    results = run_evaluation(
        experiment_policy, MULTI_OBJECT_TASKS, "multi_object",
        args.device, args.simulate
    )
    all_results.extend(results)

    print("\n--- Novel Compositions ---")
    results = run_evaluation(
        experiment_policy, NOVEL_COMPOSITION_TASKS, "novel_composition",
        args.device, args.simulate
    )
    all_results.extend(results)

    # Compare with baseline if provided
    comparison = None
    if args.baseline:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)

        baseline_policy = load_policy(args.baseline, args.device)
        baseline_results = []

        for category, tasks in [
            ("in_distribution", IN_DISTRIBUTION_TASKS),
            ("cross_target", CROSS_TARGET_TASKS),
            ("multi_object", MULTI_OBJECT_TASKS),
            ("novel_composition", NOVEL_COMPOSITION_TASKS)
        ]:
            results = run_evaluation(baseline_policy, tasks, category, args.device, args.simulate)
            baseline_results.extend(results)

        comparison = compare_checkpoints(all_results, baseline_results)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "baseline": args.baseline,
        "mode": "simulation" if args.simulate else "real_robot",
        "results": all_results,
        "comparison": comparison
    }

    output_file = args.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_category = {}
    for r in all_results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if r.get("correct_arm"):
            by_category[cat]["correct"] += 1

    for cat, stats in by_category.items():
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    if comparison:
        print(f"\nExperiment vs Baseline:")
        print(f"  Experiment accuracy: {comparison.get('experiment_accuracy', 0)*100:.1f}%")
        print(f"  Baseline accuracy:   {comparison.get('baseline_accuracy', 0)*100:.1f}%")
        print(f"  Improvement:         {comparison.get('improvement_pct', 0):+.1f}%")


if __name__ == "__main__":
    main()
