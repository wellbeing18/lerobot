#!/usr/bin/env python3
"""
Divergence Analysis Tool

Systematically compares hallucination vs normal cases to find:
1. WHEN does the divergence start (which step)?
2. HOW LARGE is the divergence at each step?
3. WHICH joints diverge most?
4. Is divergence related to chunk boundaries?

Key insight from investigation: The divergence is in action_raw (normalized model output),
not in denormalization. The model itself outputs different values.

Usage:
    python divergence_analysis.py \
        --hallucination-case logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --normal-case logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --output-dir logs/yogurt_banana_leftarm/divergence_analysis
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


@dataclass
class StepComparison:
    """Comparison metrics at a single step."""
    step: int
    chunk_index: int
    is_chunk_boundary: bool

    # Action raw (normalized model output) comparison
    action_raw_diff_norm: float  # L2 norm of difference
    action_raw_diff_per_joint: list  # Per-joint difference

    # Action final (denormalized) comparison
    action_final_diff_norm: float
    action_final_diff_per_joint: list

    # State comparison
    state_diff_norm: float
    state_diff_per_joint: list

    # Individual values for inspection
    normal_action_raw: list
    halluc_action_raw: list
    normal_action_final: list
    halluc_action_final: list
    normal_state: list
    halluc_state: list


@dataclass
class DivergencePoint:
    """Identified point where significant divergence begins."""
    step: int
    chunk_index: int
    action_raw_diff: float
    action_final_diff: float
    dominant_joint: int  # Which joint has largest difference
    dominant_joint_diff: float


@dataclass
class DivergenceAnalysis:
    """Complete divergence analysis results."""
    hallucination_case: str
    normal_case: str

    # Overall metrics
    total_steps: int
    first_divergence_step: int
    peak_divergence_step: int
    peak_action_raw_diff: float

    # Chunk boundary analysis
    chunk_boundaries: list
    divergence_at_boundaries: list  # action_raw_diff at each boundary

    # Per-step comparisons (sampled)
    step_comparisons: list

    # Divergence timeline
    action_raw_diff_timeline: list
    action_final_diff_timeline: list

    # Joint-level analysis
    most_divergent_joint: int
    joint_divergence_ranking: list  # [(joint_idx, total_divergence), ...]


def load_trace(trace_path: Path) -> list[dict]:
    """Load inference trace from JSONL file."""
    entries = []
    with open(trace_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compare_step(
    normal_entry: dict,
    halluc_entry: dict,
    step: int,
    chunk_size: int = 50
) -> StepComparison:
    """Compare normal vs hallucination at a single step."""
    # Extract arrays
    n_action_raw = np.array(normal_entry.get('action_raw', [0]*12))
    h_action_raw = np.array(halluc_entry.get('action_raw', [0]*12))
    n_action_final = np.array(normal_entry.get('action_final', [0]*12))
    h_action_final = np.array(halluc_entry.get('action_final', [0]*12))
    n_state = np.array(normal_entry.get('state_raw', [0]*12))
    h_state = np.array(halluc_entry.get('state_raw', [0]*12))

    # Compute differences (left arm only: joints 0-5)
    action_raw_diff = h_action_raw[:6] - n_action_raw[:6]
    action_final_diff = h_action_final[:6] - n_action_final[:6]
    state_diff = h_state[:6] - n_state[:6]

    return StepComparison(
        step=step,
        chunk_index=step // chunk_size,
        is_chunk_boundary=(step % chunk_size == 0),

        action_raw_diff_norm=float(np.linalg.norm(action_raw_diff)),
        action_raw_diff_per_joint=[float(x) for x in action_raw_diff],

        action_final_diff_norm=float(np.linalg.norm(action_final_diff)),
        action_final_diff_per_joint=[float(x) for x in action_final_diff],

        state_diff_norm=float(np.linalg.norm(state_diff)),
        state_diff_per_joint=[float(x) for x in state_diff],

        normal_action_raw=[float(x) for x in n_action_raw[:6]],
        halluc_action_raw=[float(x) for x in h_action_raw[:6]],
        normal_action_final=[float(x) for x in n_action_final[:6]],
        halluc_action_final=[float(x) for x in h_action_final[:6]],
        normal_state=[float(x) for x in n_state[:6]],
        halluc_state=[float(x) for x in h_state[:6]]
    )


def find_first_divergence(
    comparisons: list[StepComparison],
    threshold: float = 0.5
) -> Optional[DivergencePoint]:
    """Find the first step where action_raw divergence exceeds threshold."""
    for comp in comparisons:
        if comp.action_raw_diff_norm > threshold:
            # Find dominant joint
            joint_diffs = [abs(x) for x in comp.action_raw_diff_per_joint]
            dominant_joint = int(np.argmax(joint_diffs))

            return DivergencePoint(
                step=comp.step,
                chunk_index=comp.chunk_index,
                action_raw_diff=comp.action_raw_diff_norm,
                action_final_diff=comp.action_final_diff_norm,
                dominant_joint=dominant_joint,
                dominant_joint_diff=joint_diffs[dominant_joint]
            )
    return None


def analyze_joint_divergence(comparisons: list[StepComparison]) -> tuple[int, list]:
    """Analyze which joints diverge most overall."""
    # Sum absolute differences per joint across all steps
    joint_totals = np.zeros(6)
    for comp in comparisons:
        joint_totals += np.abs(comp.action_raw_diff_per_joint)

    ranking = [(i, float(total)) for i, total in enumerate(joint_totals)]
    ranking.sort(key=lambda x: x[1], reverse=True)

    most_divergent = ranking[0][0]
    return most_divergent, ranking


def analyze_divergence(
    halluc_case_dir: Path,
    normal_case_dir: Path,
    output_dir: Path,
    sample_interval: int = 5
) -> DivergenceAnalysis:
    """Main analysis function."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load traces
    halluc_trace = load_trace(halluc_case_dir / 'trace.jsonl')
    normal_trace = load_trace(normal_case_dir / 'trace.jsonl')

    min_steps = min(len(halluc_trace), len(normal_trace))
    print(f"Loaded traces: halluc={len(halluc_trace)} steps, normal={len(normal_trace)} steps")
    print(f"Analyzing {min_steps} steps")

    # Compare all steps
    all_comparisons = []
    for step in range(min_steps):
        comp = compare_step(normal_trace[step], halluc_trace[step], step)
        all_comparisons.append(comp)

    # Extract timelines
    action_raw_timeline = [c.action_raw_diff_norm for c in all_comparisons]
    action_final_timeline = [c.action_final_diff_norm for c in all_comparisons]

    # Find first divergence
    first_div = find_first_divergence(all_comparisons, threshold=0.3)
    first_divergence_step = first_div.step if first_div else -1

    # Find peak divergence
    peak_step = int(np.argmax(action_raw_timeline))
    peak_diff = action_raw_timeline[peak_step]

    # Analyze chunk boundaries
    chunk_size = 50
    boundaries = list(range(0, min_steps, chunk_size))
    divergence_at_boundaries = [action_raw_timeline[b] for b in boundaries if b < len(action_raw_timeline)]

    # Analyze per-joint divergence
    most_divergent_joint, joint_ranking = analyze_joint_divergence(all_comparisons)

    # Sample comparisons for output
    sampled_comparisons = [all_comparisons[i] for i in range(0, min_steps, sample_interval)]

    # Build result
    analysis = DivergenceAnalysis(
        hallucination_case=str(halluc_case_dir),
        normal_case=str(normal_case_dir),
        total_steps=min_steps,
        first_divergence_step=first_divergence_step,
        peak_divergence_step=peak_step,
        peak_action_raw_diff=peak_diff,
        chunk_boundaries=boundaries,
        divergence_at_boundaries=divergence_at_boundaries,
        step_comparisons=[asdict(c) for c in sampled_comparisons],
        action_raw_diff_timeline=action_raw_timeline,
        action_final_diff_timeline=action_final_timeline,
        most_divergent_joint=most_divergent_joint,
        joint_divergence_ranking=joint_ranking
    )

    # Save results
    output_file = output_dir / 'divergence_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(asdict(analysis), f, indent=2)
    print(f"Analysis saved to {output_file}")

    # Generate visualizations
    generate_visualizations(all_comparisons, analysis, output_dir)

    # Print summary
    print_summary(analysis, first_div)

    return analysis


def generate_visualizations(
    comparisons: list[StepComparison],
    analysis: DivergenceAnalysis,
    output_dir: Path
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    steps = np.arange(analysis.total_steps)

    # Figure 1: Timeline comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Action raw divergence
    ax1 = axes[0]
    ax1.plot(steps, analysis.action_raw_diff_timeline, 'b-', linewidth=1, label='action_raw diff')
    ax1.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Divergence threshold (0.3)')

    # Mark chunk boundaries
    for b in analysis.chunk_boundaries:
        ax1.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

    # Mark first divergence
    if analysis.first_divergence_step >= 0:
        ax1.axvline(x=analysis.first_divergence_step, color='red', linewidth=2,
                   label=f'First divergence (step {analysis.first_divergence_step})')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('action_raw Difference (L2 norm)')
    ax1.set_title('Normalized Model Output (action_raw) Divergence Over Time')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, analysis.total_steps)

    # Plot 2: Action final divergence
    ax2 = axes[1]
    ax2.plot(steps, analysis.action_final_diff_timeline, 'g-', linewidth=1, label='action_final diff')

    for b in analysis.chunk_boundaries:
        ax2.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('action_final Difference (L2 norm)')
    ax2.set_title('Denormalized Action (action_final) Divergence Over Time')
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, analysis.total_steps)

    # Plot 3: State divergence
    state_diff_timeline = [c.state_diff_norm for c in comparisons]
    ax3 = axes[2]
    ax3.plot(steps, state_diff_timeline, 'm-', linewidth=1, label='state diff')

    for b in analysis.chunk_boundaries:
        ax3.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Step')
    ax3.set_ylabel('State Difference (L2 norm)')
    ax3.set_title('Robot State Divergence Over Time (Result of Action Divergence)')
    ax3.legend(loc='upper left')
    ax3.set_xlim(0, analysis.total_steps)

    plt.tight_layout()
    plt.savefig(output_dir / 'divergence_timeline.png', dpi=150)
    plt.close()
    print(f"Timeline saved to {output_dir / 'divergence_timeline.png'}")

    # Figure 2: Per-joint analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    for i, (ax, name) in enumerate(zip(axes.flatten(), joint_names)):
        joint_diff = [c.action_raw_diff_per_joint[i] for c in comparisons]
        ax.plot(steps, joint_diff, linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Difference')
        ax.set_title(f'Joint {i}: {name}')
        ax.set_xlim(0, analysis.total_steps)

    plt.suptitle('Per-Joint action_raw Divergence (Hallucination - Normal)', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_joint_divergence.png', dpi=150)
    plt.close()
    print(f"Per-joint analysis saved to {output_dir / 'per_joint_divergence.png'}")

    # Figure 3: Chunk boundary analysis
    fig, ax = plt.subplots(figsize=(10, 5))

    chunk_indices = list(range(len(analysis.divergence_at_boundaries)))
    ax.bar(chunk_indices, analysis.divergence_at_boundaries, alpha=0.7)
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('action_raw Divergence at Chunk Start')
    ax.set_title('Divergence at Chunk Boundaries (Every 50 Steps)')

    # Add text labels
    for i, val in enumerate(analysis.divergence_at_boundaries):
        ax.text(i, val + 0.05, f'{val:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'chunk_boundary_divergence.png', dpi=150)
    plt.close()
    print(f"Chunk analysis saved to {output_dir / 'chunk_boundary_divergence.png'}")

    # Figure 4: Critical steps detail
    critical_steps = [150, 200, 210, 250, 300]
    fig, axes = plt.subplots(1, len(critical_steps), figsize=(20, 4))

    for ax, step in zip(axes, critical_steps):
        if step < len(comparisons):
            comp = comparisons[step]

            x = np.arange(6)
            width = 0.35

            ax.bar(x - width/2, comp.normal_action_raw, width, label='Normal', alpha=0.7)
            ax.bar(x + width/2, comp.halluc_action_raw, width, label='Halluc', alpha=0.7)

            ax.set_xlabel('Joint')
            ax.set_ylabel('action_raw')
            ax.set_title(f'Step {step}\ndiff={comp.action_raw_diff_norm:.2f}')
            ax.set_xticks(x)
            ax.set_xticklabels(['J0', 'J1', 'J2', 'J3', 'J4', 'G'])
            ax.legend(fontsize=8)

    plt.suptitle('action_raw Comparison at Critical Steps', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'critical_steps_comparison.png', dpi=150)
    plt.close()
    print(f"Critical steps saved to {output_dir / 'critical_steps_comparison.png'}")


def print_summary(analysis: DivergenceAnalysis, first_div: Optional[DivergencePoint]):
    """Print analysis summary."""
    print("\n" + "="*70)
    print("DIVERGENCE ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nCases compared:")
    print(f"  Hallucination: {analysis.hallucination_case}")
    print(f"  Normal: {analysis.normal_case}")
    print(f"  Total steps: {analysis.total_steps}")

    print(f"\n--- Divergence Detection ---")
    if first_div:
        print(f"First significant divergence at step {first_div.step} (chunk {first_div.chunk_index})")
        print(f"  action_raw diff: {first_div.action_raw_diff:.3f}")
        print(f"  action_final diff: {first_div.action_final_diff:.2f}")
        print(f"  Dominant joint: {first_div.dominant_joint} (diff={first_div.dominant_joint_diff:.3f})")
    else:
        print("No significant divergence detected!")

    print(f"\nPeak divergence at step {analysis.peak_divergence_step}")
    print(f"  Peak action_raw diff: {analysis.peak_action_raw_diff:.3f}")

    print(f"\n--- Chunk Boundary Analysis ---")
    print(f"Divergence at chunk boundaries:")
    for i, (boundary, div) in enumerate(zip(analysis.chunk_boundaries, analysis.divergence_at_boundaries)):
        marker = " ← SIGNIFICANT" if div > 0.5 else ""
        print(f"  Chunk {i} (step {boundary}): {div:.3f}{marker}")

    print(f"\n--- Per-Joint Analysis ---")
    print(f"Most divergent joint: {analysis.most_divergent_joint}")
    print("Joint divergence ranking:")
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    for joint_idx, total_div in analysis.joint_divergence_ranking:
        print(f"  Joint {joint_idx} ({joint_names[joint_idx]}): {total_div:.2f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze divergence between hallucination and normal cases")
    parser.add_argument("--hallucination-case", type=str, required=True, help="Path to hallucination case")
    parser.add_argument("--normal-case", type=str, required=True, help="Path to normal case")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sample-interval", type=int, default=5, help="Sampling interval for detailed comparisons")

    args = parser.parse_args()

    analysis = analyze_divergence(
        halluc_case_dir=Path(args.hallucination_case),
        normal_case_dir=Path(args.normal_case),
        output_dir=Path(args.output_dir),
        sample_interval=args.sample_interval
    )

    return analysis


if __name__ == "__main__":
    main()
