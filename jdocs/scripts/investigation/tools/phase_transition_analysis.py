#!/usr/bin/env python3
"""
Phase Transition Analysis for SmolVLA Hallucination Investigation.

Based on Motion2Vec and HSSM research, this tool analyzes phase transitions
in training data to understand how the model learns (or fails to learn)
transitions between behavior phases.

Key Questions to Answer:
- Which phase transitions are underrepresented in training?
- Is TRANSPORT → IDLE transition missing (explaining why model stays in TRANSPORT)?
- What training patterns correlate with the hallucination behavior?

Phases:
- APPROACH: Moving toward target object
- GRIP: Grasping the object
- TRANSPORT: Moving with object
- RELEASE: Releasing the object
- IDLE: Staying still (no movement)

Usage:
    python phase_transition_analysis.py \
        --dataset datasets_bimanuel/multitasks \
        --task-filter yogurt \
        --output-dir outputs/phase_transitions
"""

import argparse
import json
import sys
from collections import defaultdict
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

# Phase classification thresholds
VELOCITY_IDLE_THRESHOLD = 0.3       # Below this = IDLE
VELOCITY_SLOW_THRESHOLD = 1.0       # Below this = transitioning
GRIPPER_OPEN_THRESHOLD = 0.5        # Below this = open gripper
GRIPPER_CHANGE_THRESHOLD = 0.1      # Change detection threshold

# Phase names
PHASE_APPROACH = "APPROACH"
PHASE_GRIP = "GRIP"
PHASE_TRANSPORT = "TRANSPORT"
PHASE_RELEASE = "RELEASE"
PHASE_IDLE = "IDLE"
PHASE_UNKNOWN = "UNKNOWN"

ALL_PHASES = [PHASE_APPROACH, PHASE_GRIP, PHASE_TRANSPORT, PHASE_RELEASE, PHASE_IDLE]

PHASE_COLORS = {
    PHASE_APPROACH: "#1f77b4",
    PHASE_GRIP: "#ff7f0e",
    PHASE_TRANSPORT: "#2ca02c",
    PHASE_RELEASE: "#d62728",
    PHASE_IDLE: "#9467bd",
    PHASE_UNKNOWN: "#7f7f7f",
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PhaseSegment:
    """A segment of an episode belonging to one phase."""
    episode_idx: int
    phase: str
    start_step: int
    end_step: int
    duration: int
    mean_velocity: float
    gripper_state_start: float
    gripper_state_end: float


@dataclass
class PhaseTransition:
    """A transition between two phases."""
    episode_idx: int
    from_phase: str
    to_phase: str
    transition_step: int
    velocity_before: float
    velocity_after: float
    gripper_change: float


@dataclass
class TransitionStats:
    """Statistics for a specific transition type."""
    transition_name: str  # e.g., "APPROACH → GRIP"
    count: int
    percentage: float  # Of all transitions
    avg_velocity_before: float
    avg_velocity_after: float
    avg_gripper_change: float
    example_episodes: List[int]


@dataclass
class PhaseTransitionAnalysis:
    """Full phase transition analysis."""
    dataset_path: str
    task_filter: str
    num_episodes: int

    # Phase statistics
    phase_counts: Dict[str, int]
    phase_durations: Dict[str, List[int]]
    mean_phase_durations: Dict[str, float]

    # Transition statistics
    total_transitions: int
    transition_matrix: Dict[str, Dict[str, int]]  # from -> to -> count
    transition_stats: Dict[str, TransitionStats]

    # Missing/rare transitions
    missing_transitions: List[str]
    rare_transitions: List[str]  # <5% of expected frequency

    # Episode endings analysis
    ending_phase_distribution: Dict[str, int]
    episodes_ending_with_idle: int
    pct_episodes_ending_idle: float


# ============================================================================
# PHASE CLASSIFICATION
# ============================================================================

def classify_phase(velocity: float, gripper_state: float, gripper_change: float,
                  prev_phase: str, episode_progress: float) -> str:
    """Classify the current phase based on velocity, gripper, and context."""

    # IDLE: Low velocity
    if velocity < VELOCITY_IDLE_THRESHOLD:
        return PHASE_IDLE

    # GRIP: Gripper closing
    if gripper_change < -GRIPPER_CHANGE_THRESHOLD:
        return PHASE_GRIP

    # RELEASE: Gripper opening
    if gripper_change > GRIPPER_CHANGE_THRESHOLD:
        return PHASE_RELEASE

    # Early in episode with high velocity = APPROACH
    if episode_progress < 0.3:
        return PHASE_APPROACH

    # Late in episode with high velocity = likely TRANSPORT
    if episode_progress > 0.3:
        return PHASE_TRANSPORT

    # Default to continuing previous phase
    if prev_phase in [PHASE_APPROACH, PHASE_TRANSPORT]:
        return prev_phase

    return PHASE_UNKNOWN


def segment_episode_into_phases(actions: np.ndarray, gripper_idx: int = -1) -> List[PhaseSegment]:
    """Segment an episode into phase segments."""
    n_steps = len(actions)
    segments = []

    # Compute velocities
    velocities = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    velocities = np.concatenate([[0], velocities])  # Pad to match length

    # Extract gripper state (last action dimension by default)
    if gripper_idx == -1:
        gripper_idx = actions.shape[1] - 1
    gripper_states = actions[:, gripper_idx]
    gripper_changes = np.diff(gripper_states)
    gripper_changes = np.concatenate([[0], gripper_changes])

    # Classify each step
    phases = []
    prev_phase = PHASE_APPROACH
    for i in range(n_steps):
        progress = i / n_steps
        phase = classify_phase(
            velocities[i],
            gripper_states[i],
            gripper_changes[i],
            prev_phase,
            progress
        )
        phases.append(phase)
        prev_phase = phase

    # Smooth phases (require min 5 steps per phase)
    smoothed_phases = smooth_phase_sequence(phases, min_duration=5)

    # Extract segments
    current_phase = smoothed_phases[0]
    segment_start = 0

    for i in range(1, len(smoothed_phases)):
        if smoothed_phases[i] != current_phase or i == len(smoothed_phases) - 1:
            # End of segment
            end_step = i if smoothed_phases[i] != current_phase else i + 1
            segment = PhaseSegment(
                episode_idx=-1,  # Will be set later
                phase=current_phase,
                start_step=segment_start,
                end_step=end_step,
                duration=end_step - segment_start,
                mean_velocity=float(np.mean(velocities[segment_start:end_step])),
                gripper_state_start=float(gripper_states[segment_start]),
                gripper_state_end=float(gripper_states[end_step - 1]),
            )
            segments.append(segment)

            current_phase = smoothed_phases[i]
            segment_start = i

    return segments


def smooth_phase_sequence(phases: List[str], min_duration: int = 5) -> List[str]:
    """Smooth phase sequence by removing very short segments."""
    smoothed = phases.copy()

    # Multiple passes to handle nested short segments
    for _ in range(3):
        i = 0
        while i < len(smoothed):
            # Find segment
            start = i
            current = smoothed[i]
            while i < len(smoothed) and smoothed[i] == current:
                i += 1
            end = i

            # If segment too short, replace with surrounding phase
            if end - start < min_duration:
                # Get surrounding phases
                before = smoothed[start - 1] if start > 0 else current
                after = smoothed[end] if end < len(smoothed) else current

                # Replace with most common surrounding
                replacement = before if before == after else before
                for j in range(start, end):
                    smoothed[j] = replacement

    return smoothed


def extract_transitions(segments: List[PhaseSegment], episode_idx: int) -> List[PhaseTransition]:
    """Extract transitions from phase segments."""
    transitions = []

    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]

        if seg1.phase != seg2.phase:
            transition = PhaseTransition(
                episode_idx=episode_idx,
                from_phase=seg1.phase,
                to_phase=seg2.phase,
                transition_step=seg1.end_step,
                velocity_before=seg1.mean_velocity,
                velocity_after=seg2.mean_velocity,
                gripper_change=seg2.gripper_state_start - seg1.gripper_state_end,
            )
            transitions.append(transition)

    return transitions


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_phase_transitions(dataset_path: str, task_filter: str = None) -> PhaseTransitionAnalysis:
    """Analyze phase transitions in training dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset from {dataset_path}...")
    # Load with both repo_id and root for local datasets
    dataset = LeRobotDataset(repo_id=dataset_path, root=dataset_path)

    all_segments = []
    all_transitions = []
    ending_phases = []

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
        actions = episode_data['action']

        if len(actions) < 20:  # Skip very short episodes
            continue

        # Convert to numpy array
        actions = np.array([a.numpy() if hasattr(a, 'numpy') else np.array(a) for a in actions])

        # Segment into phases
        segments = segment_episode_into_phases(actions)
        for seg in segments:
            seg.episode_idx = ep_idx
        all_segments.extend(segments)

        # Extract transitions
        transitions = extract_transitions(segments, ep_idx)
        all_transitions.extend(transitions)

        # Record ending phase
        if segments:
            ending_phases.append(segments[-1].phase)

    print(f"Processed {filtered_episodes} episodes")
    print(f"Found {len(all_segments)} segments and {len(all_transitions)} transitions")

    # Compute phase statistics
    phase_counts = defaultdict(int)
    phase_durations = defaultdict(list)

    for seg in all_segments:
        phase_counts[seg.phase] += 1
        phase_durations[seg.phase].append(seg.duration)

    mean_phase_durations = {
        phase: float(np.mean(durations)) if durations else 0
        for phase, durations in phase_durations.items()
    }

    # Compute transition matrix
    transition_matrix = defaultdict(lambda: defaultdict(int))
    for trans in all_transitions:
        transition_matrix[trans.from_phase][trans.to_phase] += 1

    # Compute transition statistics
    transition_stats = {}
    for from_phase, to_dict in transition_matrix.items():
        for to_phase, count in to_dict.items():
            trans_name = f"{from_phase} → {to_phase}"

            # Get all transitions of this type
            matching = [t for t in all_transitions
                       if t.from_phase == from_phase and t.to_phase == to_phase]

            stats = TransitionStats(
                transition_name=trans_name,
                count=count,
                percentage=count / len(all_transitions) * 100 if all_transitions else 0,
                avg_velocity_before=float(np.mean([t.velocity_before for t in matching])),
                avg_velocity_after=float(np.mean([t.velocity_after for t in matching])),
                avg_gripper_change=float(np.mean([t.gripper_change for t in matching])),
                example_episodes=list(set([t.episode_idx for t in matching]))[:5],
            )
            transition_stats[trans_name] = stats

    # Identify missing and rare transitions
    expected_transitions = [
        f"{PHASE_APPROACH} → {PHASE_GRIP}",
        f"{PHASE_GRIP} → {PHASE_TRANSPORT}",
        f"{PHASE_TRANSPORT} → {PHASE_RELEASE}",
        f"{PHASE_RELEASE} → {PHASE_IDLE}",
        f"{PHASE_TRANSPORT} → {PHASE_IDLE}",
        f"{PHASE_APPROACH} → {PHASE_IDLE}",
    ]

    missing = [t for t in expected_transitions if t not in transition_stats]
    rare = [t for t, s in transition_stats.items() if s.percentage < 5 and t in expected_transitions]

    # Episode ending analysis
    ending_dist = defaultdict(int)
    for phase in ending_phases:
        ending_dist[phase] += 1

    episodes_ending_idle = ending_dist.get(PHASE_IDLE, 0)
    pct_ending_idle = episodes_ending_idle / len(ending_phases) * 100 if ending_phases else 0

    return PhaseTransitionAnalysis(
        dataset_path=dataset_path,
        task_filter=task_filter or "all",
        num_episodes=filtered_episodes,
        phase_counts=dict(phase_counts),
        phase_durations={k: v for k, v in phase_durations.items()},
        mean_phase_durations=mean_phase_durations,
        total_transitions=len(all_transitions),
        transition_matrix={k: dict(v) for k, v in transition_matrix.items()},
        transition_stats={k: asdict(v) for k, v in transition_stats.items()},
        missing_transitions=missing,
        rare_transitions=rare,
        ending_phase_distribution=dict(ending_dist),
        episodes_ending_with_idle=episodes_ending_idle,
        pct_episodes_ending_idle=pct_ending_idle,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_transition_matrix(analysis: PhaseTransitionAnalysis, output_dir: Path):
    """Visualize phase transition matrix as heatmap."""
    phases = ALL_PHASES
    n_phases = len(phases)

    # Build matrix
    matrix = np.zeros((n_phases, n_phases))
    for i, from_phase in enumerate(phases):
        for j, to_phase in enumerate(phases):
            if from_phase in analysis.transition_matrix:
                matrix[i, j] = analysis.transition_matrix[from_phase].get(to_phase, 0)

    # Normalize by row (outgoing transitions)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(matrix, cmap='Blues')
    ax1.set_xticks(range(n_phases))
    ax1.set_yticks(range(n_phases))
    ax1.set_xticklabels(phases, rotation=45, ha='right')
    ax1.set_yticklabels(phases)
    ax1.set_xlabel("To Phase")
    ax1.set_ylabel("From Phase")
    ax1.set_title("Transition Counts")

    # Add text annotations
    for i in range(n_phases):
        for j in range(n_phases):
            ax1.text(j, i, f'{int(matrix[i, j])}', ha='center', va='center', fontsize=9)

    plt.colorbar(im1, ax=ax1)

    # Normalized (%)
    ax2 = axes[1]
    im2 = ax2.imshow(matrix_normalized, cmap='YlOrRd', vmin=0, vmax=100)
    ax2.set_xticks(range(n_phases))
    ax2.set_yticks(range(n_phases))
    ax2.set_xticklabels(phases, rotation=45, ha='right')
    ax2.set_yticklabels(phases)
    ax2.set_xlabel("To Phase")
    ax2.set_ylabel("From Phase")
    ax2.set_title("Transition Probability (%)")

    # Add text annotations
    for i in range(n_phases):
        for j in range(n_phases):
            ax2.text(j, i, f'{matrix_normalized[i, j]:.0f}%', ha='center', va='center', fontsize=9)

    plt.colorbar(im2, ax=ax2, label='%')

    plt.suptitle(f"Phase Transition Matrix: {analysis.task_filter}\n{analysis.num_episodes} episodes")
    plt.tight_layout()
    plt.savefig(output_dir / "transition_matrix.png", dpi=150)
    plt.close()


def visualize_phase_distribution(analysis: PhaseTransitionAnalysis, output_dir: Path):
    """Visualize phase distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Phase counts
    ax1 = axes[0, 0]
    phases = list(analysis.phase_counts.keys())
    counts = list(analysis.phase_counts.values())
    colors = [PHASE_COLORS.get(p, '#7f7f7f') for p in phases]
    ax1.bar(phases, counts, color=colors, alpha=0.7)
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Count")
    ax1.set_title("Phase Segment Counts")

    # 2. Mean durations
    ax2 = axes[0, 1]
    durations = [analysis.mean_phase_durations.get(p, 0) for p in phases]
    ax2.bar(phases, durations, color=colors, alpha=0.7)
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Mean Duration (steps)")
    ax2.set_title("Mean Phase Duration")

    # 3. Episode ending distribution
    ax3 = axes[1, 0]
    end_phases = list(analysis.ending_phase_distribution.keys())
    end_counts = list(analysis.ending_phase_distribution.values())
    end_colors = [PHASE_COLORS.get(p, '#7f7f7f') for p in end_phases]
    ax3.pie(end_counts, labels=end_phases, colors=end_colors, autopct='%1.1f%%')
    ax3.set_title(f"Episode Ending Phases\n({analysis.pct_episodes_ending_idle:.1f}% end with IDLE)")

    # 4. Key transitions analysis
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Table of critical transitions
    table_data = [["Transition", "Count", "%", "Status"]]

    critical_transitions = [
        f"{PHASE_TRANSPORT} → {PHASE_IDLE}",
        f"{PHASE_RELEASE} → {PHASE_IDLE}",
        f"{PHASE_APPROACH} → {PHASE_IDLE}",
    ]

    for trans in critical_transitions:
        if trans in analysis.transition_stats:
            s = analysis.transition_stats[trans]
            status = "OK" if s["count"] > 10 else "**LOW**"
            table_data.append([trans, s["count"], f"{s['percentage']:.1f}%", status])
        else:
            table_data.append([trans, 0, "0%", "**MISSING**"])

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Critical Transitions for Post-Task Behavior")

    plt.suptitle(f"Phase Distribution Analysis: {analysis.task_filter}")
    plt.tight_layout()
    plt.savefig(output_dir / "phase_distribution.png", dpi=150)
    plt.close()


def visualize_transition_flow(analysis: PhaseTransitionAnalysis, output_dir: Path):
    """Visualize phase flow as Sankey-like diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))

    phases = ALL_PHASES
    n_phases = len(phases)

    # Position phases vertically
    phase_y = {phase: i for i, phase in enumerate(phases)}

    # Draw phase boxes
    for phase in phases:
        y = phase_y[phase]
        count = analysis.phase_counts.get(phase, 0)
        color = PHASE_COLORS.get(phase, '#7f7f7f')

        rect = plt.Rectangle((0.1, y - 0.3), 0.2, 0.6, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(0.2, y, f"{phase}\n({count})", ha='center', va='center', fontsize=10)

        rect2 = plt.Rectangle((0.7, y - 0.3), 0.2, 0.6, facecolor=color, alpha=0.7)
        ax.add_patch(rect2)
        ax.text(0.8, y, f"{phase}", ha='center', va='center', fontsize=10)

    # Draw transitions
    max_count = max(s["count"] for s in analysis.transition_stats.values()) if analysis.transition_stats else 1

    for trans_name, stats in analysis.transition_stats.items():
        parts = trans_name.split(" → ")
        if len(parts) != 2:
            continue

        from_phase, to_phase = parts
        if from_phase not in phase_y or to_phase not in phase_y:
            continue

        from_y = phase_y[from_phase]
        to_y = phase_y[to_phase]
        count = stats["count"]

        # Line width proportional to count
        linewidth = 1 + (count / max_count) * 8

        # Color based on importance
        if "IDLE" in to_phase:
            color = 'green'
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.3

        ax.annotate('', xy=(0.7, to_y), xytext=(0.3, from_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=linewidth, alpha=alpha))

        # Add count label
        mid_x = 0.5
        mid_y = (from_y + to_y) / 2
        ax.text(mid_x, mid_y, f"{count}", fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_phases - 0.5)
    ax.axis('off')
    ax.set_title(f"Phase Transition Flow: {analysis.task_filter}\nGreen = transitions to IDLE")

    plt.tight_layout()
    plt.savefig(output_dir / "transition_flow.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analysis: PhaseTransitionAnalysis, output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Phase Transition Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Purpose")
    report.append("""
Phase transition analysis examines how training data teaches the model to
transition between behavior phases. Missing or underrepresented transitions
can cause the model to "get stuck" in a phase or fail to recognize when
to transition to a new behavior (like staying still after task completion).
""")

    report.append("\n## Dataset Summary")
    report.append(f"- Dataset: {analysis.dataset_path}")
    report.append(f"- Task filter: {analysis.task_filter}")
    report.append(f"- Episodes analyzed: {analysis.num_episodes}")
    report.append(f"- Total transitions detected: {analysis.total_transitions}")

    report.append("\n## Phase Distribution")
    report.append("\n| Phase | Count | Mean Duration | % of Total |")
    report.append("|-------|-------|---------------|------------|")
    total = sum(analysis.phase_counts.values())
    for phase in ALL_PHASES:
        count = analysis.phase_counts.get(phase, 0)
        duration = analysis.mean_phase_durations.get(phase, 0)
        pct = count / total * 100 if total > 0 else 0
        report.append(f"| {phase} | {count} | {duration:.1f} | {pct:.1f}% |")

    report.append("\n## Transition Statistics")
    report.append("\n### All Transitions")
    report.append("\n| Transition | Count | % | Avg Vel Before | Avg Vel After |")
    report.append("|------------|-------|---|----------------|---------------|")
    for trans_name, stats in sorted(analysis.transition_stats.items(), key=lambda x: -x[1]["count"]):
        report.append(f"| {trans_name} | {stats['count']} | {stats['percentage']:.1f}% | "
                     f"{stats['avg_velocity_before']:.2f} | {stats['avg_velocity_after']:.2f} |")

    report.append("\n### Missing/Rare Transitions")
    if analysis.missing_transitions:
        report.append(f"\n**Missing transitions**: {', '.join(analysis.missing_transitions)}")
    if analysis.rare_transitions:
        report.append(f"\n**Rare transitions (<5%)**: {', '.join(analysis.rare_transitions)}")

    report.append("\n## Episode Ending Analysis")
    report.append(f"\n**Critical Finding**: Only **{analysis.pct_episodes_ending_idle:.1f}%** of episodes end with IDLE phase")

    report.append("\n| Ending Phase | Count | % |")
    report.append("|--------------|-------|---|")
    total_endings = sum(analysis.ending_phase_distribution.values())
    for phase, count in sorted(analysis.ending_phase_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_endings * 100 if total_endings > 0 else 0
        report.append(f"| {phase} | {count} | {pct:.1f}% |")

    report.append("\n## Key Findings")

    # Check for TRANSPORT → IDLE deficiency
    transport_to_idle = analysis.transition_stats.get(f"{PHASE_TRANSPORT} → {PHASE_IDLE}", {})
    transport_to_idle_count = transport_to_idle.get("count", 0) if transport_to_idle else 0

    if transport_to_idle_count < 10:
        report.append(f"""
### Missing TRANSPORT → IDLE Transition

**Critical Finding**: Only {transport_to_idle_count} examples of TRANSPORT → IDLE transition.

This explains why the model continues moving after completing a task:
- The model has learned TRANSPORT behavior (moving with object)
- But has NOT learned to transition from TRANSPORT to IDLE (staying still)
- When the task is complete, the model defaults to continuing TRANSPORT behavior
""")

    # Check IDLE phase presence
    idle_count = analysis.phase_counts.get(PHASE_IDLE, 0)
    if idle_count < analysis.total_transitions * 0.1:
        report.append(f"""
### Insufficient IDLE Training Data

**Finding**: IDLE phase only appears {idle_count} times ({idle_count/sum(analysis.phase_counts.values())*100:.1f}% of segments).

The model lacks sufficient examples of "staying still" behavior, making it
prone to generating movement trajectories even when the task is complete.
""")

    # Episode ending issue
    if analysis.pct_episodes_ending_idle < 50:
        report.append(f"""
### Episodes Don't End With IDLE

**Finding**: Only {analysis.pct_episodes_ending_idle:.1f}% of episodes end with IDLE phase.

Most training episodes end with movement (TRANSPORT, APPROACH, etc.), teaching the model
that "episode completion" involves movement rather than staying still.
""")

    report.append("\n## Recommendations for Data Collection")
    report.append("""
Based on the analysis, the following data collection would help:

1. **Post-task IDLE demonstrations**: Record episodes where the robot stays still
   for 50+ steps after completing the task

2. **TRANSPORT → IDLE transitions**: Add explicit demonstrations of transitioning
   from carrying an object to staying still

3. **Distractor scenarios during IDLE**: Ensure IDLE segments include scenes
   with visible distractors to teach the model to ignore them
""")

    report.append("\n## Visualizations")
    report.append("\n- `transition_matrix.png`: Heatmap of phase transitions")
    report.append("- `phase_distribution.png`: Phase counts and durations")
    report.append("- `transition_flow.png`: Flow diagram of transitions")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        DATASET_PATH, TASK_FILTER, get_output_dir
    )

    parser = argparse.ArgumentParser(description="Analyze phase transitions in training data")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                       help="Path to training dataset")
    parser.add_argument("--task-filter", default=TASK_FILTER,
                       help="Filter for task names")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("phase_transitions")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Run analysis
    print("\n=== Analyzing Phase Transitions ===")
    analysis = analyze_phase_transitions(args.dataset, args.task_filter)

    print(f"\nPhase counts: {dict(analysis.phase_counts)}")
    print(f"Total transitions: {analysis.total_transitions}")
    print(f"Episodes ending with IDLE: {analysis.pct_episodes_ending_idle:.1f}%")
    print(f"Missing transitions: {analysis.missing_transitions}")

    # Save raw data
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(asdict(analysis), f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_transition_matrix(analysis, output_dir)
    visualize_phase_distribution(analysis, output_dir)
    visualize_transition_flow(analysis, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(analysis, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
