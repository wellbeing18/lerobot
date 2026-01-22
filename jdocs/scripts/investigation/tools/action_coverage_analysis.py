#!/usr/bin/env python3
"""
Action Space Coverage Analysis for SmolVLA Hallucination Investigation.

Analyzes per-dimension action coverage statistics in training data to identify
joints that may have poor coverage, leading to out-of-distribution behavior.

Key Questions to Answer:
- Which joints have narrow action distributions in training?
- Does hallucination involve joints with poor training coverage?
- What is the effective range vs. utilized range for each joint?

Usage:
    python action_coverage_analysis.py \
        --dataset datasets_bimanuel/multitasks \
        --task-filter yogurt \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --output-dir outputs/action_coverage
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

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS
# ============================================================================

# Joint names for bimanual robot (12 joints per arm = 24 total, but action might be 32)
JOINT_NAMES_LEFT = [
    "L_shoulder_pan", "L_shoulder_lift", "L_elbow", "L_wrist_1",
    "L_wrist_2", "L_wrist_3", "L_gripper"
]

JOINT_NAMES_RIGHT = [
    "R_shoulder_pan", "R_shoulder_lift", "R_elbow", "R_wrist_1",
    "R_wrist_2", "R_wrist_3", "R_gripper"
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class JointCoverageStats:
    """Coverage statistics for a single joint/dimension."""
    joint_idx: int
    joint_name: str

    # Basic statistics
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float

    # Coverage metrics
    coverage_ratio: float  # std / range (higher = more spread out)
    effective_range: float  # 95th - 5th percentile
    quartile_range: float  # 75th - 25th percentile (IQR)

    # Distribution properties
    skewness: float
    kurtosis: float
    percentile_5: float
    percentile_95: float

    # Quality flags
    is_poor_coverage: bool  # coverage_ratio < 0.3
    is_concentrated: bool   # IQR < 0.1 * range


@dataclass
class ActionCoverageAnalysis:
    """Full action space coverage analysis."""
    dataset_path: str
    task_filter: str
    num_episodes: int
    num_actions: int
    action_dim: int

    # Per-joint statistics
    joint_stats: List[JointCoverageStats]

    # Summary metrics
    mean_coverage_ratio: float
    poor_coverage_joints: List[int]
    well_covered_joints: List[int]

    # Comparison with inference (if provided)
    inference_comparison: Optional[Dict[str, Dict]] = None


@dataclass
class InferenceJointAnalysis:
    """Analysis of inference trajectory vs training coverage."""
    case_name: str
    joint_idx: int

    # Inference trajectory stats
    inf_mean: float
    inf_range: float
    inf_min: float
    inf_max: float

    # Training coverage comparison
    train_mean: float
    train_range: float
    train_percentile_5: float
    train_percentile_95: float

    # OOD metrics
    percent_within_training_range: float  # What % of inference falls within training range
    distance_from_training_mean: float    # Normalized by training std
    is_out_of_distribution: bool


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_joint_stats(actions: np.ndarray, joint_idx: int, joint_name: str) -> JointCoverageStats:
    """Compute coverage statistics for a single joint."""
    joint_data = actions[:, joint_idx]

    # Basic statistics
    mean = float(np.mean(joint_data))
    std = float(np.std(joint_data))
    min_val = float(np.min(joint_data))
    max_val = float(np.max(joint_data))
    range_val = max_val - min_val

    # Coverage metrics
    coverage_ratio = std / (range_val + 1e-8)

    percentile_5 = float(np.percentile(joint_data, 5))
    percentile_95 = float(np.percentile(joint_data, 95))
    effective_range = percentile_95 - percentile_5

    percentile_25 = float(np.percentile(joint_data, 25))
    percentile_75 = float(np.percentile(joint_data, 75))
    quartile_range = percentile_75 - percentile_25

    # Distribution properties
    from scipy import stats
    skewness = float(stats.skew(joint_data))
    kurtosis = float(stats.kurtosis(joint_data))

    # Quality flags
    is_poor_coverage = coverage_ratio < 0.3
    is_concentrated = quartile_range < 0.1 * range_val

    return JointCoverageStats(
        joint_idx=joint_idx,
        joint_name=joint_name,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        range_val=range_val,
        coverage_ratio=coverage_ratio,
        effective_range=effective_range,
        quartile_range=quartile_range,
        skewness=skewness,
        kurtosis=kurtosis,
        percentile_5=percentile_5,
        percentile_95=percentile_95,
        is_poor_coverage=is_poor_coverage,
        is_concentrated=is_concentrated,
    )


def analyze_training_coverage(dataset_path: str, task_filter: str = None) -> ActionCoverageAnalysis:
    """Analyze action space coverage in training dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset from {dataset_path}...")
    # Load with both repo_id and root for local datasets
    dataset = LeRobotDataset(repo_id=dataset_path, root=dataset_path)

    # Collect all actions
    all_actions = []

    # Get episode info from meta.episodes
    episodes = dataset.meta.episodes
    total_episodes = dataset.meta.total_episodes
    filtered_episodes = 0

    # Get episode boundaries
    from_indices = episodes['dataset_from_index']
    to_indices = episodes['dataset_to_index']
    episode_tasks = episodes['tasks']

    print(f"Processing {total_episodes} episodes...")

    for ep_idx in range(total_episodes):
        from_idx = from_indices[ep_idx]
        to_idx = to_indices[ep_idx]

        # Get episode task (if available) for filtering
        task = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""

        # Check task filter
        if task_filter and task_filter.lower() not in task.lower():
            continue

        filtered_episodes += 1

        # Extract actions directly from hf_dataset (avoid video decoding)
        episode_data = dataset.hf_dataset.select(range(from_idx, to_idx))
        for action in episode_data['action']:
            if hasattr(action, 'numpy'):
                action = action.numpy()
            else:
                action = np.array(action)
            all_actions.append(action)

    all_actions = np.array(all_actions)
    print(f"Collected {len(all_actions)} actions from {filtered_episodes} episodes")

    action_dim = all_actions.shape[1]

    # Generate joint names
    joint_names = []
    for i in range(action_dim):
        if i < len(JOINT_NAMES_LEFT):
            joint_names.append(f"L_{i}_{JOINT_NAMES_LEFT[i] if i < len(JOINT_NAMES_LEFT) else 'joint'}")
        elif i < 2 * len(JOINT_NAMES_LEFT):
            idx = i - len(JOINT_NAMES_LEFT)
            joint_names.append(f"R_{idx}_{JOINT_NAMES_RIGHT[idx] if idx < len(JOINT_NAMES_RIGHT) else 'joint'}")
        else:
            joint_names.append(f"joint_{i}")

    # Compute per-joint statistics
    joint_stats = []
    for i in range(action_dim):
        stats = compute_joint_stats(all_actions, i, joint_names[i] if i < len(joint_names) else f"joint_{i}")
        joint_stats.append(stats)

    # Summary
    coverage_ratios = [s.coverage_ratio for s in joint_stats]
    mean_coverage = float(np.mean(coverage_ratios))
    poor_joints = [s.joint_idx for s in joint_stats if s.is_poor_coverage]
    well_joints = [s.joint_idx for s in joint_stats if not s.is_poor_coverage]

    return ActionCoverageAnalysis(
        dataset_path=dataset_path,
        task_filter=task_filter or "all",
        num_episodes=filtered_episodes,
        num_actions=len(all_actions),
        action_dim=action_dim,
        joint_stats=[asdict(s) for s in joint_stats],
        mean_coverage_ratio=mean_coverage,
        poor_coverage_joints=poor_joints,
        well_covered_joints=well_joints,
    )


def analyze_inference_vs_training(training_analysis: ActionCoverageAnalysis,
                                  case_dirs: List[str],
                                  step_range: Tuple[int, int] = (200, 300)) -> Dict[str, List[InferenceJointAnalysis]]:
    """Compare inference trajectories against training coverage."""
    results = {}

    for case_dir in case_dirs:
        case_path = Path(case_dir)
        trace_path = case_path / "trace.jsonl"

        if not trace_path.exists():
            print(f"Warning: No trace file at {trace_path}")
            continue

        # Load inference actions
        actions = []
        with open(trace_path, "r") as f:
            for line in f:
                data = json.loads(line)
                step = data["step"]
                if step_range[0] <= step < step_range[1]:
                    action = data.get("action", data.get("action_normalized"))
                    if action:
                        actions.append(action)

        if not actions:
            continue

        actions = np.array(actions)
        print(f"Loaded {len(actions)} inference actions from {case_path.name}")

        # Compare each joint
        joint_analyses = []
        for joint_idx, train_stats in enumerate(training_analysis.joint_stats):
            if joint_idx >= actions.shape[1]:
                break

            inf_data = actions[:, joint_idx]

            # Inference stats
            inf_mean = float(np.mean(inf_data))
            inf_min = float(np.min(inf_data))
            inf_max = float(np.max(inf_data))
            inf_range = inf_max - inf_min

            # Training range
            train_min = train_stats["percentile_5"]
            train_max = train_stats["percentile_95"]
            train_mean = train_stats["mean"]
            train_std = train_stats["std"]

            # % within training range
            within = np.sum((inf_data >= train_min) & (inf_data <= train_max))
            percent_within = float(within / len(inf_data) * 100)

            # Distance from training mean (normalized)
            distance = abs(inf_mean - train_mean) / (train_std + 1e-8)

            # OOD if <70% within training range or >2 std from mean
            is_ood = percent_within < 70 or distance > 2

            analysis = InferenceJointAnalysis(
                case_name=case_path.name,
                joint_idx=joint_idx,
                inf_mean=inf_mean,
                inf_range=inf_range,
                inf_min=inf_min,
                inf_max=inf_max,
                train_mean=train_mean,
                train_range=train_stats["range_val"],
                train_percentile_5=train_min,
                train_percentile_95=train_max,
                percent_within_training_range=percent_within,
                distance_from_training_mean=distance,
                is_out_of_distribution=is_ood,
            )

            joint_analyses.append(analysis)

        results[case_path.name] = joint_analyses

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_coverage_distribution(analysis: ActionCoverageAnalysis, output_dir: Path):
    """Visualize action coverage distribution."""
    joint_stats = analysis.joint_stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Coverage ratio per joint
    ax1 = axes[0, 0]
    coverage_ratios = [s["coverage_ratio"] for s in joint_stats]
    colors = ['red' if s["is_poor_coverage"] else 'green' for s in joint_stats]
    ax1.bar(range(len(coverage_ratios)), coverage_ratios, color=colors, alpha=0.7)
    ax1.axhline(y=0.3, color='red', linestyle='--', label='Poor coverage threshold')
    ax1.set_xlabel("Joint Index")
    ax1.set_ylabel("Coverage Ratio (std/range)")
    ax1.set_title("Action Space Coverage by Joint")
    ax1.legend()

    # 2. Box plot of joint ranges
    ax2 = axes[0, 1]
    mins = [s["min_val"] for s in joint_stats]
    maxs = [s["max_val"] for s in joint_stats]
    means = [s["mean"] for s in joint_stats]
    p5s = [s["percentile_5"] for s in joint_stats]
    p95s = [s["percentile_95"] for s in joint_stats]

    joints = range(len(joint_stats))
    ax2.fill_between(joints, mins, maxs, alpha=0.2, label='Full range')
    ax2.fill_between(joints, p5s, p95s, alpha=0.4, label='5-95 percentile')
    ax2.plot(joints, means, 'k-', linewidth=2, label='Mean')
    ax2.set_xlabel("Joint Index")
    ax2.set_ylabel("Action Value")
    ax2.set_title("Action Range by Joint")
    ax2.legend()

    # 3. Histogram of coverage ratios
    ax3 = axes[1, 0]
    ax3.hist(coverage_ratios, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0.3, color='red', linestyle='--', label='Poor coverage threshold')
    ax3.set_xlabel("Coverage Ratio")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Coverage Ratios")
    ax3.legend()

    # 4. Poor coverage joints table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [["Joint", "Coverage", "Range", "IQR", "Poor?"]]
    for s in joint_stats[:12]:  # First 12 joints
        poor = "YES" if s["is_poor_coverage"] else "no"
        table_data.append([
            s["joint_name"][:15],
            f"{s['coverage_ratio']:.3f}",
            f"{s['range_val']:.3f}",
            f"{s['quartile_range']:.3f}",
            poor
        ])

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.3)
    ax4.set_title(f"First 12 Joints Analysis\n({len(analysis.poor_coverage_joints)} poor coverage joints total)")

    plt.suptitle(f"Action Space Coverage: {analysis.task_filter} task\n"
                f"{analysis.num_episodes} episodes, {analysis.num_actions} actions")
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_distribution.png", dpi=150)
    plt.close()


def visualize_inference_comparison(training_analysis: ActionCoverageAnalysis,
                                   inference_results: Dict[str, List[InferenceJointAnalysis]],
                                   output_dir: Path):
    """Visualize comparison between inference and training coverage."""
    if not inference_results:
        return

    n_cases = len(inference_results)
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 5 * n_cases))

    if n_cases == 1:
        axes = axes.reshape(1, -1)

    for row, (case_name, analyses) in enumerate(inference_results.items()):
        # Left: % within training range per joint
        ax1 = axes[row, 0]
        percents = [a.percent_within_training_range for a in analyses]
        colors = ['red' if not a.is_out_of_distribution else 'blue' for a in analyses]
        ax1.bar(range(len(percents)), percents, color=colors, alpha=0.7)
        ax1.axhline(y=70, color='red', linestyle='--', label='OOD threshold (70%)')
        ax1.set_xlabel("Joint Index")
        ax1.set_ylabel("% Within Training Range")
        ax1.set_title(f"{case_name[:30]}\n% Actions Within Training Range")
        ax1.legend()

        # Right: Distance from training mean
        ax2 = axes[row, 1]
        distances = [a.distance_from_training_mean for a in analyses]
        colors = ['red' if d > 2 else 'green' for d in distances]
        ax2.bar(range(len(distances)), distances, color=colors, alpha=0.7)
        ax2.axhline(y=2, color='red', linestyle='--', label='OOD threshold (2σ)')
        ax2.set_xlabel("Joint Index")
        ax2.set_ylabel("Distance from Training Mean (σ)")
        ax2.set_title(f"Distance from Training Mean")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "inference_vs_training.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analysis: ActionCoverageAnalysis,
                   inference_results: Optional[Dict], output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Action Space Coverage Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Purpose")
    report.append("""
Action space coverage analysis identifies joints/dimensions that have narrow
distributions in training data. These joints are more likely to produce
out-of-distribution behavior during inference.
""")

    report.append("\n## Dataset Summary")
    report.append(f"- Dataset: {analysis.dataset_path}")
    report.append(f"- Task filter: {analysis.task_filter}")
    report.append(f"- Episodes analyzed: {analysis.num_episodes}")
    report.append(f"- Total actions: {analysis.num_actions}")
    report.append(f"- Action dimension: {analysis.action_dim}")

    report.append("\n## Coverage Summary")
    report.append(f"- Mean coverage ratio: **{analysis.mean_coverage_ratio:.3f}**")
    report.append(f"- Joints with poor coverage (<0.3): **{len(analysis.poor_coverage_joints)}**")
    report.append(f"- Joints with good coverage: {len(analysis.well_covered_joints)}")

    if analysis.poor_coverage_joints:
        report.append("\n### Poor Coverage Joints (Potential Problem Areas)")
        report.append("\n| Joint | Name | Coverage | Range | Issue |")
        report.append("|-------|------|----------|-------|-------|")
        for idx in analysis.poor_coverage_joints[:10]:
            s = analysis.joint_stats[idx]
            issue = "Low variance" if s["is_concentrated"] else "Narrow range"
            report.append(f"| {idx} | {s['joint_name'][:15]} | {s['coverage_ratio']:.3f} | "
                         f"{s['range_val']:.3f} | {issue} |")

    report.append("\n## Per-Joint Statistics")
    report.append("\n| Joint | Mean | Std | Range | Coverage | 5th% | 95th% | Poor? |")
    report.append("|-------|------|-----|-------|----------|------|-------|-------|")
    for s in analysis.joint_stats[:16]:
        poor = "**YES**" if s["is_poor_coverage"] else "no"
        report.append(f"| {s['joint_idx']} | {s['mean']:.3f} | {s['std']:.3f} | "
                     f"{s['range_val']:.3f} | {s['coverage_ratio']:.3f} | "
                     f"{s['percentile_5']:.3f} | {s['percentile_95']:.3f} | {poor} |")

    if inference_results:
        report.append("\n## Inference vs Training Comparison")

        for case_name, analyses in inference_results.items():
            report.append(f"\n### {case_name}")

            ood_joints = [a for a in analyses if a.is_out_of_distribution]
            report.append(f"- Out-of-distribution joints: **{len(ood_joints)} / {len(analyses)}**")

            if ood_joints:
                report.append("\n#### OOD Joints Detail")
                report.append("\n| Joint | % Within Range | Distance (σ) | Inf Mean | Train Mean |")
                report.append("|-------|----------------|--------------|----------|------------|")
                for a in ood_joints[:10]:
                    report.append(f"| {a.joint_idx} | {a.percent_within_training_range:.1f}% | "
                                 f"{a.distance_from_training_mean:.2f} | {a.inf_mean:.3f} | "
                                 f"{a.train_mean:.3f} |")

    report.append("\n## Key Findings")

    if len(analysis.poor_coverage_joints) > analysis.action_dim * 0.3:
        report.append("""
### Warning: Many joints have poor coverage

More than 30% of joints have low coverage ratios, indicating the training data
may not adequately cover the action space. This increases the risk of
out-of-distribution behavior during inference.
""")

    if inference_results:
        total_ood = sum(len([a for a in analyses if a.is_out_of_distribution])
                       for analyses in inference_results.values())
        total_joints = sum(len(analyses) for analyses in inference_results.values())

        if total_ood > 0:
            report.append(f"""
### Inference OOD Detection

Found **{total_ood} out-of-distribution joint configurations** across inference cases.
These joints are operating outside the training distribution and may contribute
to hallucination behavior.
""")

    report.append("\n## Visualizations")
    report.append("\n- `coverage_distribution.png`: Per-joint coverage statistics")
    report.append("- `inference_vs_training.png`: Comparison of inference vs training ranges")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        DATASET_PATH, TASK_FILTER, get_all_case_dirs, get_output_dir, STEP_RANGE
    )

    parser = argparse.ArgumentParser(description="Analyze action space coverage")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                       help="Path to training dataset")
    parser.add_argument("--task-filter", default=TASK_FILTER,
                       help="Filter for task names")
    parser.add_argument("--case-dirs", nargs="+", default=get_all_case_dirs(),
                       help="Paths to inference case directories")
    parser.add_argument("--step-range", nargs=2, type=int, default=list(STEP_RANGE),
                       help="Step range for inference analysis")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("action_coverage")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Analyze training coverage
    print("\n=== Analyzing Training Coverage ===")
    analysis = analyze_training_coverage(args.dataset, args.task_filter)
    print(f"Mean coverage ratio: {analysis.mean_coverage_ratio:.3f}")
    print(f"Poor coverage joints: {len(analysis.poor_coverage_joints)}")

    # Compare with inference
    print("\n=== Comparing with Inference ===")
    inference_results = analyze_inference_vs_training(
        analysis, args.case_dirs, tuple(args.step_range)
    )

    # Update analysis with inference comparison
    if inference_results:
        analysis.inference_comparison = {
            case: [asdict(a) for a in analyses]
            for case, analyses in inference_results.items()
        }

    # Save raw data
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(asdict(analysis), f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_coverage_distribution(analysis, output_dir)
    visualize_inference_comparison(analysis, inference_results, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(analysis, inference_results, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
