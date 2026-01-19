#!/usr/bin/env python3
"""
Evidence Aggregator for SmolVLA Hallucination Investigation.

Synthesizes findings from all investigation tools to:
1. Build evidence correlation matrix
2. Trace causal chain from visual input to hallucinated action
3. Identify root cause with supporting evidence
4. Generate final investigation report

Usage:
    # Aggregate evidence from all sources
    python aggregate_evidence.py \
        --investigation-dir ../cases \
        --ablation-results ../reports/ablation_suite \
        --dataset-analysis ../reports/dataset_analysis \
        --output-dir ../reports/final

    # Generate summary report only
    python aggregate_evidence.py \
        --investigation-dir ../cases \
        --output-dir ../reports/final \
        --summary-only
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CaseEvidence:
    """Evidence from a single investigation case."""
    case_id: str
    case_type: str  # 'hallucination' or 'normal'
    task: str

    # Observed behavior
    hallucination_observed: bool
    hallucination_step: Optional[int] = None
    post_completion_movement: float = 0.0

    # Scene context
    distractor_present: bool = False
    distractor_type: Optional[str] = None
    distractor_position: Optional[str] = None

    # Attention analysis
    attention_entropy: Optional[float] = None
    distractor_attention_ratio: Optional[float] = None

    # Denoising analysis
    denoising_anomaly_step: Optional[int] = None
    velocity_anomaly: bool = False

    # Notes
    notes: str = ""


@dataclass
class AblationEvidence:
    """Evidence from ablation experiments."""
    experiment_name: str
    experiment_type: str
    description: str
    success: bool
    hallucination_prevented: bool = False
    post_completion_movement: float = 0.0
    effect_size: float = 0.0  # Compared to baseline
    notes: str = ""


@dataclass
class DatasetEvidence:
    """Evidence from dataset analysis."""
    task_balance: str  # 'balanced', 'imbalanced'
    idle_frame_ratio: float = 0.0
    multi_object_ratio: float = 0.0
    post_completion_pattern: str = ""  # 'stay_still', 'continue_moving', 'mixed'
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class EvidenceCorrelation:
    """Correlation between symptom and potential cause."""
    symptom: str
    potential_cause: str
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    confidence: float  # 0-1
    notes: str = ""


@dataclass
class RootCauseAnalysis:
    """Final root cause analysis."""
    primary_cause: str
    confidence: float
    causal_chain: list[str]
    supporting_evidence: list[str]
    recommended_solutions: list[str]


@dataclass
class InvestigationReport:
    """Complete investigation report."""
    timestamp: str
    summary: str
    case_evidence: list[CaseEvidence]
    ablation_evidence: list[AblationEvidence]
    dataset_evidence: Optional[DatasetEvidence]
    correlations: list[EvidenceCorrelation]
    root_cause: Optional[RootCauseAnalysis]


# ============================================================================
# EVIDENCE COLLECTION
# ============================================================================

def collect_case_evidence(investigation_dir: Path) -> list[CaseEvidence]:
    """Collect evidence from all investigated cases."""
    evidence = []

    for case_type in ['hallucination', 'normal']:
        case_type_dir = investigation_dir / case_type
        if not case_type_dir.exists():
            continue

        for case_dir in case_type_dir.iterdir():
            if not case_dir.is_dir():
                continue

            case_evidence = load_case_evidence(case_dir, case_type)
            if case_evidence:
                evidence.append(case_evidence)

    return evidence


def load_case_evidence(case_dir: Path, case_type: str) -> Optional[CaseEvidence]:
    """Load evidence from a single case directory."""
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    case_id = case_dir.name
    task = metadata.get("task_original", "Unknown")

    # Initialize evidence
    evidence = CaseEvidence(
        case_id=case_id,
        case_type=case_type,
        task=task,
        hallucination_observed=(case_type == "hallucination"),
    )

    # Load trace data if available
    trace_path = case_dir / "trace.jsonl"
    if trace_path.exists():
        trace_metrics = analyze_trace_for_evidence(trace_path)
        evidence.post_completion_movement = trace_metrics.get("post_completion_movement", 0.0)
        evidence.hallucination_step = trace_metrics.get("hallucination_step")

    # Load analysis results if available
    analysis_path = case_dir / "analysis" / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
        evidence.attention_entropy = analysis.get("attention_entropy")
        evidence.distractor_attention_ratio = analysis.get("distractor_attention_ratio")

    # Try to infer distractor presence from notes or metadata
    notes = metadata.get("notes", "")
    if "banana" in notes.lower():
        evidence.distractor_present = True
        evidence.distractor_type = "banana"
    elif "no object" in notes.lower() or "no distractor" in notes.lower():
        evidence.distractor_present = False

    evidence.notes = notes

    return evidence


def analyze_trace_for_evidence(trace_path: Path) -> dict:
    """Analyze trace file for evidence metrics."""
    metrics = {
        "post_completion_movement": 0.0,
        "hallucination_step": None,
    }

    try:
        steps = []
        with open(trace_path) as f:
            for line in f:
                steps.append(json.loads(line))

        if not steps:
            return metrics

        # Find task modification point (if any)
        mod_step = None
        for step in steps:
            if step.get("task_modified", False):
                mod_step = step["step"]
                break

        # Calculate post-completion movement
        if mod_step:
            post_steps = [s for s in steps if s["step"] > mod_step]
            if post_steps:
                deltas = [s.get("action_delta_max", 0) for s in post_steps]
                metrics["post_completion_movement"] = sum(deltas) / len(deltas)

        # Detect potential hallucination step (large unexpected movement after settling)
        # Look for sudden movement increase after initial settling
        deltas = [s.get("action_delta_max", 0) for s in steps]
        if len(deltas) > 30:
            # Rolling average of movement
            window = 10
            for i in range(30, len(deltas) - window):
                prev_avg = np.mean(deltas[i-window:i])
                curr = deltas[i]
                if prev_avg < 2.0 and curr > 5.0:
                    metrics["hallucination_step"] = i
                    break

    except Exception as e:
        print(f"Warning: Could not analyze trace {trace_path}: {e}")

    return metrics


def collect_ablation_evidence(ablation_dir: Path) -> list[AblationEvidence]:
    """Collect evidence from ablation experiment results."""
    evidence = []

    # Look for suite results
    suite_path = ablation_dir / "suite_results.json"
    if suite_path.exists():
        with open(suite_path) as f:
            suite = json.load(f)

        baseline_movement = None

        for exp in suite.get("experiments", []):
            # Find baseline for comparison
            if exp.get("experiment_name") == "baseline_no_modification":
                baseline_movement = exp.get("metrics", {}).get("post_completion_movement", 0)

        for exp in suite.get("experiments", []):
            metrics = exp.get("metrics", {})
            movement = metrics.get("post_completion_movement", 0)

            # Calculate effect size compared to baseline
            effect = 0.0
            if baseline_movement and baseline_movement > 0:
                effect = (baseline_movement - movement) / baseline_movement

            abl_evidence = AblationEvidence(
                experiment_name=exp.get("experiment_name", "unknown"),
                experiment_type=exp.get("experiment_type", "unknown"),
                description=exp.get("description", ""),
                success=exp.get("success", False),
                hallucination_prevented=(movement < 1.0 if baseline_movement and baseline_movement > 2.0 else False),
                post_completion_movement=movement,
                effect_size=effect,
            )
            evidence.append(abl_evidence)

    # Also look for individual experiment results
    for result_file in ablation_dir.glob("*_result.json"):
        with open(result_file) as f:
            exp = json.load(f)

        if exp.get("experiment_name") not in [e.experiment_name for e in evidence]:
            abl_evidence = AblationEvidence(
                experiment_name=exp.get("experiment_name", "unknown"),
                experiment_type=exp.get("experiment_type", "unknown"),
                description=exp.get("description", ""),
                success=exp.get("success", False),
                post_completion_movement=exp.get("metrics", {}).get("post_completion_movement", 0),
            )
            evidence.append(abl_evidence)

    return evidence


def collect_dataset_evidence(dataset_dir: Path) -> Optional[DatasetEvidence]:
    """Collect evidence from dataset analysis."""
    analysis_path = dataset_dir / "analysis.json"
    if not analysis_path.exists():
        return None

    with open(analysis_path) as f:
        analysis = json.load(f)

    evidence = DatasetEvidence(
        task_balance="unknown",
        warnings=analysis.get("warnings", []),
        recommendations=analysis.get("recommendations", []),
    )

    # Analyze task balance
    task_stats = analysis.get("task_stats", [])
    if task_stats:
        counts = [t.get("episode_count", 0) for t in task_stats]
        if counts:
            ratio = max(counts) / (min(counts) + 1)
            evidence.task_balance = "balanced" if ratio < 1.5 else "imbalanced"

    # Check phase distribution
    phase_dist = analysis.get("phase_distribution")
    if phase_dist:
        total = sum([
            phase_dist.get("approach", 0),
            phase_dist.get("grip", 0),
            phase_dist.get("transport", 0),
            phase_dist.get("release", 0),
            phase_dist.get("return_home", 0),
            phase_dist.get("idle", 0),
        ])
        if total > 0:
            evidence.idle_frame_ratio = phase_dist.get("idle", 0) / total

    return evidence


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_correlations(
    case_evidence: list[CaseEvidence],
    ablation_evidence: list[AblationEvidence],
    dataset_evidence: Optional[DatasetEvidence],
) -> list[EvidenceCorrelation]:
    """Compute correlations between symptoms and potential causes."""
    correlations = []

    # Correlation 1: Distractor presence → Hallucination
    distractor_hallucination = EvidenceCorrelation(
        symptom="Hallucination behavior",
        potential_cause="Visual distractor object present",
        supporting_evidence=[],
        contradicting_evidence=[],
        confidence=0.0,
    )

    for case in case_evidence:
        if case.hallucination_observed and case.distractor_present:
            distractor_hallucination.supporting_evidence.append(
                f"{case.case_id}: Hallucination with {case.distractor_type} present"
            )
        elif not case.hallucination_observed and case.distractor_present:
            distractor_hallucination.contradicting_evidence.append(
                f"{case.case_id}: No hallucination despite distractor"
            )
        elif case.hallucination_observed and not case.distractor_present:
            distractor_hallucination.contradicting_evidence.append(
                f"{case.case_id}: Hallucination without distractor"
            )
        elif not case.hallucination_observed and not case.distractor_present:
            distractor_hallucination.supporting_evidence.append(
                f"{case.case_id}: No hallucination, no distractor"
            )

    # Calculate confidence
    total = len(distractor_hallucination.supporting_evidence) + len(distractor_hallucination.contradicting_evidence)
    if total > 0:
        distractor_hallucination.confidence = len(distractor_hallucination.supporting_evidence) / total

    correlations.append(distractor_hallucination)

    # Correlation 2: Completion phrase → Reduced hallucination
    completion_phrase_effect = EvidenceCorrelation(
        symptom="Hallucination behavior",
        potential_cause="Missing task completion signal",
        supporting_evidence=[],
        contradicting_evidence=[],
        confidence=0.0,
    )

    for abl in ablation_evidence:
        if abl.experiment_type == "language" and "completion" in abl.experiment_name:
            if abl.hallucination_prevented or abl.effect_size > 0.3:
                completion_phrase_effect.supporting_evidence.append(
                    f"{abl.experiment_name}: {abl.effect_size:.0%} reduction in movement"
                )
            else:
                completion_phrase_effect.contradicting_evidence.append(
                    f"{abl.experiment_name}: No significant effect"
                )

    total = len(completion_phrase_effect.supporting_evidence) + len(completion_phrase_effect.contradicting_evidence)
    if total > 0:
        completion_phrase_effect.confidence = len(completion_phrase_effect.supporting_evidence) / total

    correlations.append(completion_phrase_effect)

    # Correlation 3: Dataset idle frames → Hallucination
    if dataset_evidence:
        idle_frames_effect = EvidenceCorrelation(
            symptom="Post-completion hallucination",
            potential_cause="Under-represented idle/post-completion frames in training",
            supporting_evidence=[],
            contradicting_evidence=[],
            confidence=0.0,
        )

        if dataset_evidence.idle_frame_ratio < 0.05:
            idle_frames_effect.supporting_evidence.append(
                f"Idle frames are only {dataset_evidence.idle_frame_ratio*100:.1f}% of training data"
            )
            idle_frames_effect.confidence = 0.7
        else:
            idle_frames_effect.contradicting_evidence.append(
                f"Idle frames are {dataset_evidence.idle_frame_ratio*100:.1f}% of training data"
            )
            idle_frames_effect.confidence = 0.3

        correlations.append(idle_frames_effect)

    return correlations


# ============================================================================
# ROOT CAUSE ANALYSIS
# ============================================================================

def identify_root_cause(
    correlations: list[EvidenceCorrelation],
    case_evidence: list[CaseEvidence],
    ablation_evidence: list[AblationEvidence],
) -> Optional[RootCauseAnalysis]:
    """Identify most likely root cause based on evidence."""

    # Sort correlations by confidence
    sorted_correlations = sorted(correlations, key=lambda x: x.confidence, reverse=True)

    if not sorted_correlations or sorted_correlations[0].confidence < 0.3:
        return None

    top_correlation = sorted_correlations[0]

    # Build causal chain
    causal_chain = []
    solutions = []

    if "distractor" in top_correlation.potential_cause.lower():
        causal_chain = [
            "Visual distractor object detected by vision encoder",
            "Cross-attention mechanism attends to distractor features",
            "Distractor visual features influence KV cache",
            "Action expert generates actions based on distractor context",
            "Robot reaches toward distractor location",
        ]
        solutions = [
            "Add multi-object scenes to training data with explicit 'stay still' after completion",
            "Implement attention masking to suppress distractor regions after task completion",
            "Add task completion detection and automatic action suppression",
        ]

    elif "completion" in top_correlation.potential_cause.lower():
        causal_chain = [
            "Task description does not signal completion",
            "Model continues to generate actions as if task is ongoing",
            "Visual features from environment trigger learned action patterns",
            "Robot executes actions despite task being complete",
        ]
        solutions = [
            "Add completion phrase to task description after task is done",
            "Implement automatic task completion detection",
            "Fine-tune with explicit 'stay still' data after task completion",
        ]

    elif "idle" in top_correlation.potential_cause.lower() or "training" in top_correlation.potential_cause.lower():
        causal_chain = [
            "Training data lacks sufficient post-completion idle frames",
            "Model has not learned to remain still after task completion",
            "Any visual stimulus triggers action generation",
            "Robot moves even when it should stay still",
        ]
        solutions = [
            "Augment training data with more post-completion idle frames",
            "Add explicit 'stay still' episodes to training set",
            "Balance training data to include more idle/holding states",
        ]

    root_cause = RootCauseAnalysis(
        primary_cause=top_correlation.potential_cause,
        confidence=top_correlation.confidence,
        causal_chain=causal_chain,
        supporting_evidence=top_correlation.supporting_evidence[:5],
        recommended_solutions=solutions,
    )

    return root_cause


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_evidence_matrix(
    case_evidence: list[CaseEvidence],
    output_path: Path,
):
    """Plot evidence correlation matrix."""
    # Create feature matrix
    features = []
    labels = []

    for case in case_evidence:
        features.append([
            1 if case.hallucination_observed else 0,
            1 if case.distractor_present else 0,
            case.post_completion_movement,
            case.attention_entropy or 0,
        ])
        labels.append(case.case_id)

    if not features:
        print("No case evidence to plot")
        return

    features = np.array(features)
    feature_names = ['Hallucination', 'Distractor', 'Movement', 'Attention Entropy']

    # Compute correlation matrix
    if features.shape[0] > 1:
        corr_matrix = np.corrcoef(features.T)
    else:
        corr_matrix = np.eye(len(feature_names))

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)

    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Evidence Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evidence matrix: {output_path}")


def plot_ablation_comparison(
    ablation_evidence: list[AblationEvidence],
    output_path: Path,
):
    """Plot ablation experiment comparison."""
    if not ablation_evidence:
        print("No ablation evidence to plot")
        return

    # Sort by effect size
    sorted_evidence = sorted(ablation_evidence, key=lambda x: x.effect_size, reverse=True)

    names = [e.experiment_name.replace("language_completion_", "").replace("_", " ")
             for e in sorted_evidence]
    effects = [e.effect_size * 100 for e in sorted_evidence]
    colors = ['green' if e > 0 else 'red' for e in effects]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(names, effects, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Effect Size (% reduction in post-completion movement)', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    ax.set_title('Ablation Experiment Effects', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, effect in zip(bars, effects):
        x = bar.get_width()
        ax.text(x + 1, bar.get_y() + bar.get_height()/2,
                f'{effect:.0f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation comparison: {output_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_final_report(report: InvestigationReport, output_path: Path):
    """Generate final investigation report in markdown."""
    md = f"""# SmolVLA Hallucination Investigation Report

**Generated**: {report.timestamp}

## Executive Summary

{report.summary}

## Case Evidence

| Case ID | Type | Hallucination | Distractor | Movement | Notes |
|---------|------|---------------|------------|----------|-------|
"""

    for case in report.case_evidence:
        halluc = "Yes" if case.hallucination_observed else "No"
        distract = case.distractor_type if case.distractor_present else "None"
        md += f"| {case.case_id} | {case.case_type} | {halluc} | {distract} | {case.post_completion_movement:.2f} | {case.notes[:30]} |\n"

    md += """
## Ablation Experiment Results

| Experiment | Type | Effect Size | Prevents Hallucination |
|------------|------|-------------|------------------------|
"""

    for abl in report.ablation_evidence:
        prevented = "Yes" if abl.hallucination_prevented else "No"
        md += f"| {abl.experiment_name} | {abl.experiment_type} | {abl.effect_size:.0%} | {prevented} |\n"

    if report.dataset_evidence:
        md += f"""
## Dataset Analysis

| Metric | Value |
|--------|-------|
| Task Balance | {report.dataset_evidence.task_balance} |
| Idle Frame Ratio | {report.dataset_evidence.idle_frame_ratio:.1%} |
| Multi-Object Ratio | {report.dataset_evidence.multi_object_ratio:.1%} |

### Warnings
"""
        for w in report.dataset_evidence.warnings:
            md += f"- {w}\n"

        md += "\n### Recommendations\n"
        for r in report.dataset_evidence.recommendations:
            md += f"- {r}\n"

    md += """
## Evidence Correlations

"""

    for corr in report.correlations:
        md += f"""### {corr.symptom} ↔ {corr.potential_cause}

**Confidence**: {corr.confidence:.0%}

**Supporting Evidence**:
"""
        for e in corr.supporting_evidence[:3]:
            md += f"- {e}\n"

        if corr.contradicting_evidence:
            md += "\n**Contradicting Evidence**:\n"
            for e in corr.contradicting_evidence[:3]:
                md += f"- {e}\n"

        md += "\n"

    if report.root_cause:
        md += f"""
## Root Cause Analysis

### Primary Cause
**{report.root_cause.primary_cause}** (Confidence: {report.root_cause.confidence:.0%})

### Causal Chain
"""
        for i, step in enumerate(report.root_cause.causal_chain, 1):
            md += f"{i}. {step}\n"

        md += """
### Recommended Solutions
"""
        for sol in report.root_cause.recommended_solutions:
            md += f"- {sol}\n"

    md += """
## Appendix

### Files Analyzed
- Case evidence from investigation/cases/
- Ablation results from ablation experiments
- Dataset analysis from analyze_dataset.py

### Methodology
1. Collected case evidence from hallucination and normal inference runs
2. Ran ablation experiments (language modifications, temporal variations)
3. Analyzed training dataset distribution
4. Computed correlations between symptoms and potential causes
5. Identified most likely root cause based on evidence weight
"""

    with open(output_path, 'w') as f:
        f.write(md)
    print(f"Saved final report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evidence Aggregator for SmolVLA Hallucination Investigation"
    )

    parser.add_argument("--investigation-dir", type=Path, required=True,
                       help="Directory containing investigated cases")
    parser.add_argument("--ablation-results", type=Path,
                       help="Directory containing ablation results")
    parser.add_argument("--dataset-analysis", type=Path,
                       help="Directory containing dataset analysis")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--summary-only", action="store_true",
                       help="Generate summary report only")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SmolVLA Evidence Aggregator")
    print("=" * 60)

    # Collect evidence
    print("\nCollecting case evidence...")
    case_evidence = collect_case_evidence(args.investigation_dir)
    print(f"  Found {len(case_evidence)} cases")

    ablation_evidence = []
    if args.ablation_results and args.ablation_results.exists():
        print("\nCollecting ablation evidence...")
        ablation_evidence = collect_ablation_evidence(args.ablation_results)
        print(f"  Found {len(ablation_evidence)} experiments")

    dataset_evidence = None
    if args.dataset_analysis and args.dataset_analysis.exists():
        print("\nCollecting dataset evidence...")
        dataset_evidence = collect_dataset_evidence(args.dataset_analysis)
        print(f"  Loaded dataset analysis")

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(case_evidence, ablation_evidence, dataset_evidence)
    print(f"  Found {len(correlations)} correlations")

    # Identify root cause
    print("\nIdentifying root cause...")
    root_cause = identify_root_cause(correlations, case_evidence, ablation_evidence)
    if root_cause:
        print(f"  Primary cause: {root_cause.primary_cause}")
        print(f"  Confidence: {root_cause.confidence:.0%}")

    # Generate summary
    summary = f"""
This investigation analyzed {len(case_evidence)} inference cases ({sum(1 for c in case_evidence if c.hallucination_observed)} hallucination, {sum(1 for c in case_evidence if not c.hallucination_observed)} normal) and {len(ablation_evidence)} ablation experiments.

{"Root cause identified: " + root_cause.primary_cause if root_cause else "Root cause not yet determined."}
"""

    # Create report
    report = InvestigationReport(
        timestamp=datetime.now().isoformat(),
        summary=summary.strip(),
        case_evidence=case_evidence,
        ablation_evidence=ablation_evidence,
        dataset_evidence=dataset_evidence,
        correlations=correlations,
        root_cause=root_cause,
    )

    # Generate outputs
    if not args.summary_only:
        if case_evidence:
            plot_evidence_matrix(case_evidence, args.output_dir / "evidence_matrix.png")
        if ablation_evidence:
            plot_ablation_comparison(ablation_evidence, args.output_dir / "ablation_comparison.png")

    # Save data
    with open(args.output_dir / "evidence_data.json", 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Generate report
    generate_final_report(report, args.output_dir / "investigation_report.md")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
