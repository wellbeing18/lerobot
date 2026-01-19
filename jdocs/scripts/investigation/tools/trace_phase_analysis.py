#!/usr/bin/env python3
"""
Trace Phase Analysis for SmolVLA Hallucination Investigation.

Analyzes inference traces to:
1. Identify task phases (approach, pick, transport, place, idle, hallucination)
2. Compare trajectories between cases
3. Find the moment when hallucination diverges from normal behavior

Key insight: The hallucinating arm goes to where the bottle USED TO BE,
suggesting trajectory replay or memory of initial pick position.

Usage:
    python trace_phase_analysis.py \
        --hallucination-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
        --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
        --output-dir logs/yogurt_banana_leftarm/phase_analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskPhase:
    """Identified phase in task execution."""
    name: str  # approach, pick, transport, place, idle, hallucination
    start_step: int
    end_step: int
    duration_steps: int
    avg_action_delta: float
    gripper_state: str  # open, closing, closed, opening


@dataclass
class TrajectoryAnalysis:
    """Analysis of a single trace."""
    case_name: str
    total_steps: int
    phases: List[TaskPhase]
    chunk_boundaries: List[int]

    # Key trajectory points
    pick_position: Optional[List[float]]  # Joint positions when gripper closes
    place_position: Optional[List[float]]  # Joint positions when gripper opens (place)
    hallucination_start: Optional[int]  # Step when hallucination begins

    # Statistics
    idle_action_mean: float
    idle_action_std: float


# ============================================================================
# TRACE LOADING
# ============================================================================

def load_trace(trace_path: Path) -> List[dict]:
    """Load trace JSONL file."""
    entries = []
    with open(trace_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


# ============================================================================
# PHASE DETECTION
# ============================================================================

def detect_gripper_events(trace: List[dict], arm: str = "left") -> List[Tuple[int, str, float]]:
    """
    Detect gripper state changes.

    Returns: List of (step, event_type, gripper_value)
    Event types: "close_start", "close_end", "open_start", "open_end"
    """
    gripper_key = f"gripper_{arm}"
    events = []

    # Get gripper values
    gripper_values = [e[gripper_key] for e in trace if gripper_key in e]

    if not gripper_values:
        return events

    # Detect state changes using thresholds
    # Gripper typically: ~7 = open, ~30 = closed (varies by robot)
    CLOSE_THRESHOLD = 15  # Gripper value above this = closing/closed
    OPEN_THRESHOLD = 10   # Gripper value below this = open/opening

    prev_state = "unknown"
    for i, e in enumerate(trace):
        if gripper_key not in e:
            continue

        val = e[gripper_key]

        if prev_state != "closing" and val > CLOSE_THRESHOLD:
            events.append((i, "close_start", val))
            prev_state = "closing"
        elif prev_state == "closing" and val < OPEN_THRESHOLD:
            events.append((i, "open_start", val))
            prev_state = "opening"

    return events


def compute_action_deltas(trace: List[dict]) -> List[float]:
    """Compute action delta (movement magnitude) for each step."""
    deltas = []
    for i, e in enumerate(trace):
        if "action_delta_max" in e:
            deltas.append(e["action_delta_max"])
        else:
            # Compute from action if not available
            if i > 0 and "action_final" in e and "action_final" in trace[i-1]:
                curr = np.array(e["action_final"])
                prev = np.array(trace[i-1]["action_final"])
                deltas.append(float(np.max(np.abs(curr - prev))))
            else:
                deltas.append(0.0)
    return deltas


def detect_phases(
    trace: List[dict],
    arm: str = "left",
    idle_threshold: float = 3.0,
    hallucination_threshold: float = 8.0,
) -> List[TaskPhase]:
    """
    Detect task phases from trace data.

    Phases:
    - approach: Moving toward object, gripper open
    - pick: Gripper closing
    - transport: Moving with object, gripper closed
    - place: Gripper opening
    - idle: Low movement, task complete
    - hallucination: High movement after task should be complete
    """
    phases = []

    action_deltas = compute_action_deltas(trace)
    gripper_events = detect_gripper_events(trace, arm)

    # Simple phase detection based on action deltas and gripper
    current_phase = "approach"
    phase_start = 0

    # Find key transitions
    # 1. First gripper close = end of approach, start of pick
    # 2. After gripper closes, high delta = transport
    # 3. Gripper open = place
    # 4. Low delta after place = idle
    # 5. High delta after idle = hallucination

    # Find gripper close event
    pick_step = None
    place_step = None

    for step, event, val in gripper_events:
        if event == "close_start" and pick_step is None:
            pick_step = step
        elif event == "open_start" and pick_step is not None:
            place_step = step
            break

    # Compute phase boundaries
    if pick_step:
        # Approach phase: 0 to pick
        avg_delta = np.mean(action_deltas[:pick_step]) if pick_step > 0 else 0
        phases.append(TaskPhase(
            name="approach",
            start_step=0,
            end_step=pick_step,
            duration_steps=pick_step,
            avg_action_delta=float(avg_delta),
            gripper_state="open",
        ))

    if pick_step and place_step:
        # Transport phase: pick to place
        avg_delta = np.mean(action_deltas[pick_step:place_step])
        phases.append(TaskPhase(
            name="transport",
            start_step=pick_step,
            end_step=place_step,
            duration_steps=place_step - pick_step,
            avg_action_delta=float(avg_delta),
            gripper_state="closed",
        ))

    # Post-place analysis: detect idle vs hallucination
    if place_step:
        post_place_deltas = action_deltas[place_step:]

        # Find where movement settles (idle starts)
        window_size = 20
        idle_start = None
        hallucination_start = None

        for i in range(len(post_place_deltas) - window_size):
            window_avg = np.mean(post_place_deltas[i:i+window_size])
            if window_avg < idle_threshold and idle_start is None:
                idle_start = place_step + i
            elif idle_start is not None and window_avg > hallucination_threshold:
                hallucination_start = place_step + i
                break

        # Place phase
        place_end = idle_start if idle_start else place_step + 50
        avg_delta = np.mean(action_deltas[place_step:place_end])
        phases.append(TaskPhase(
            name="place",
            start_step=place_step,
            end_step=place_end,
            duration_steps=place_end - place_step,
            avg_action_delta=float(avg_delta),
            gripper_state="opening",
        ))

        # Idle phase
        if idle_start:
            idle_end = hallucination_start if hallucination_start else len(trace)
            avg_delta = np.mean(action_deltas[idle_start:idle_end])
            phases.append(TaskPhase(
                name="idle",
                start_step=idle_start,
                end_step=idle_end,
                duration_steps=idle_end - idle_start,
                avg_action_delta=float(avg_delta),
                gripper_state="open",
            ))

        # Hallucination phase
        if hallucination_start:
            avg_delta = np.mean(action_deltas[hallucination_start:])
            phases.append(TaskPhase(
                name="hallucination",
                start_step=hallucination_start,
                end_step=len(trace),
                duration_steps=len(trace) - hallucination_start,
                avg_action_delta=float(avg_delta),
                gripper_state="varies",
            ))

    return phases


def get_position_at_step(trace: List[dict], step: int) -> Optional[List[float]]:
    """Get joint position at a specific step."""
    if step < len(trace) and "state_raw" in trace[step]:
        return trace[step]["state_raw"]
    return None


def get_chunk_boundaries(trace: List[dict]) -> List[int]:
    """Get steps where new action chunks begin."""
    boundaries = []
    for e in trace:
        if e.get("is_new_chunk", False):
            boundaries.append(e["step"])
    return boundaries


def analyze_trace(trace_path: Path, arm: str = "left") -> TrajectoryAnalysis:
    """Complete analysis of a single trace."""
    trace = load_trace(trace_path)

    phases = detect_phases(trace, arm)
    chunk_boundaries = get_chunk_boundaries(trace)

    # Find key positions
    pick_position = None
    place_position = None
    hallucination_start = None

    for phase in phases:
        if phase.name == "transport":
            pick_position = get_position_at_step(trace, phase.start_step)
        elif phase.name == "place":
            place_position = get_position_at_step(trace, phase.start_step)
        elif phase.name == "hallucination":
            hallucination_start = phase.start_step

    # Compute idle statistics
    action_deltas = compute_action_deltas(trace)
    idle_deltas = []
    for phase in phases:
        if phase.name == "idle":
            idle_deltas.extend(action_deltas[phase.start_step:phase.end_step])

    idle_mean = float(np.mean(idle_deltas)) if idle_deltas else 0.0
    idle_std = float(np.std(idle_deltas)) if idle_deltas else 0.0

    return TrajectoryAnalysis(
        case_name=trace_path.parent.name,
        total_steps=len(trace),
        phases=phases,
        chunk_boundaries=chunk_boundaries,
        pick_position=pick_position,
        place_position=place_position,
        hallucination_start=hallucination_start,
        idle_action_mean=idle_mean,
        idle_action_std=idle_std,
    )


# ============================================================================
# TRAJECTORY COMPARISON
# ============================================================================

def compare_trajectories(
    hall_trace_path: Path,
    normal_trace_path: Path,
    output_dir: Path,
    arm: str = "left",
):
    """Compare hallucination and normal traces."""
    print(f"Analyzing hallucination trace: {hall_trace_path}")
    hall_analysis = analyze_trace(hall_trace_path, arm)

    print(f"Analyzing normal trace: {normal_trace_path}")
    normal_analysis = analyze_trace(normal_trace_path, arm)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw traces for plotting
    hall_trace = load_trace(hall_trace_path)
    normal_trace = load_trace(normal_trace_path)

    hall_deltas = compute_action_deltas(hall_trace)
    normal_deltas = compute_action_deltas(normal_trace)

    # Print phase summary
    print("\n" + "=" * 60)
    print("PHASE ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nHallucination case phases:")
    for phase in hall_analysis.phases:
        print(f"  {phase.name:15s}: steps {phase.start_step:4d}-{phase.end_step:4d} "
              f"(duration: {phase.duration_steps:3d}, avg_delta: {phase.avg_action_delta:.2f})")

    print("\nNormal case phases:")
    for phase in normal_analysis.phases:
        print(f"  {phase.name:15s}: steps {phase.start_step:4d}-{phase.end_step:4d} "
              f"(duration: {phase.duration_steps:3d}, avg_delta: {phase.avg_action_delta:.2f})")

    if hall_analysis.hallucination_start:
        print(f"\n*** HALLUCINATION DETECTED at step {hall_analysis.hallucination_start} ***")

    # Compare pick positions
    if hall_analysis.pick_position and normal_analysis.pick_position:
        pick_diff = np.array(hall_analysis.pick_position) - np.array(normal_analysis.pick_position)
        print(f"\nPick position difference (hall - normal): max = {np.max(np.abs(pick_diff)):.3f}")

    # Plot comparison
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: Action deltas over time
    ax1 = axes[0, 0]
    ax1.plot(hall_deltas, 'r-', alpha=0.7, label='Hallucination')
    ax1.plot(normal_deltas, 'b-', alpha=0.7, label='Normal')
    ax1.axhline(y=3.0, color='g', linestyle='--', alpha=0.5, label='Idle threshold')
    ax1.axhline(y=8.0, color='orange', linestyle='--', alpha=0.5, label='Hallucination threshold')

    # Mark phase boundaries
    for phase in hall_analysis.phases:
        if phase.name == "hallucination":
            ax1.axvline(x=phase.start_step, color='r', linestyle=':', alpha=0.8)
            ax1.text(phase.start_step, ax1.get_ylim()[1]*0.9, 'hall', color='r', fontsize=8)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Action Delta (max)')
    ax1.set_title('Action Movement Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gripper states
    ax2 = axes[0, 1]
    hall_gripper = [e.get(f"gripper_{arm}", 0) for e in hall_trace]
    normal_gripper = [e.get(f"gripper_{arm}", 0) for e in normal_trace]
    ax2.plot(hall_gripper, 'r-', alpha=0.7, label='Hallucination')
    ax2.plot(normal_gripper, 'b-', alpha=0.7, label='Normal')
    ax2.set_xlabel('Step')
    ax2.set_ylabel(f'Gripper ({arm})')
    ax2.set_title('Gripper State Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Joint 0 (first left arm joint) trajectory
    ax3 = axes[1, 0]
    hall_joint0 = [e["state_raw"][0] for e in hall_trace if "state_raw" in e]
    normal_joint0 = [e["state_raw"][0] for e in normal_trace if "state_raw" in e]
    ax3.plot(hall_joint0, 'r-', alpha=0.7, label='Hallucination')
    ax3.plot(normal_joint0, 'b-', alpha=0.7, label='Normal')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Joint 0 Position')
    ax3.set_title('Left Arm Joint 0 (Base)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Joint 1 trajectory
    ax4 = axes[1, 1]
    hall_joint1 = [e["state_raw"][1] for e in hall_trace if "state_raw" in e]
    normal_joint1 = [e["state_raw"][1] for e in normal_trace if "state_raw" in e]
    ax4.plot(hall_joint1, 'r-', alpha=0.7, label='Hallucination')
    ax4.plot(normal_joint1, 'b-', alpha=0.7, label='Normal')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Joint 1 Position')
    ax4.set_title('Left Arm Joint 1 (Shoulder)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Chunk boundaries
    ax5 = axes[2, 0]
    ax5.plot(hall_deltas, 'r-', alpha=0.5, label='Hallucination deltas')
    for cb in hall_analysis.chunk_boundaries:
        ax5.axvline(x=cb, color='purple', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Action Delta')
    ax5.set_title('Hallucination: Chunk Boundaries (purple lines)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Phase diagram
    ax6 = axes[2, 1]
    phase_colors = {
        'approach': 'green',
        'transport': 'blue',
        'place': 'orange',
        'idle': 'gray',
        'hallucination': 'red',
    }

    y_hall = 1
    y_normal = 0

    for phase in hall_analysis.phases:
        ax6.barh(y_hall, phase.duration_steps, left=phase.start_step,
                color=phase_colors.get(phase.name, 'black'), alpha=0.7,
                label=phase.name if phase.start_step == 0 else "")
        ax6.text(phase.start_step + phase.duration_steps/2, y_hall, phase.name,
                ha='center', va='center', fontsize=8, color='white')

    for phase in normal_analysis.phases:
        ax6.barh(y_normal, phase.duration_steps, left=phase.start_step,
                color=phase_colors.get(phase.name, 'black'), alpha=0.7)
        ax6.text(phase.start_step + phase.duration_steps/2, y_normal, phase.name,
                ha='center', va='center', fontsize=8, color='white')

    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['Normal', 'Hallucination'])
    ax6.set_xlabel('Step')
    ax6.set_title('Task Phase Timeline')

    plt.suptitle('Trajectory Comparison: Hallucination vs Normal', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'trajectory_comparison.png'}")
    plt.close()

    # Save analysis JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "hallucination_case": {
            "path": str(hall_trace_path),
            "total_steps": hall_analysis.total_steps,
            "phases": [asdict(p) for p in hall_analysis.phases],
            "chunk_boundaries": hall_analysis.chunk_boundaries,
            "hallucination_start": hall_analysis.hallucination_start,
            "idle_action_mean": hall_analysis.idle_action_mean,
            "idle_action_std": hall_analysis.idle_action_std,
            "pick_position": hall_analysis.pick_position,
        },
        "normal_case": {
            "path": str(normal_trace_path),
            "total_steps": normal_analysis.total_steps,
            "phases": [asdict(p) for p in normal_analysis.phases],
            "chunk_boundaries": normal_analysis.chunk_boundaries,
            "hallucination_start": normal_analysis.hallucination_start,
            "idle_action_mean": normal_analysis.idle_action_mean,
            "idle_action_std": normal_analysis.idle_action_std,
            "pick_position": normal_analysis.pick_position,
        },
    }

    with open(output_dir / "phase_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'phase_analysis.json'}")

    return hall_analysis, normal_analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze trace phases for SmolVLA")
    parser.add_argument("--hallucination-trace", type=str, required=True,
                       help="Path to hallucination case trace.jsonl")
    parser.add_argument("--normal-trace", type=str, required=True,
                       help="Path to normal case trace.jsonl")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--arm", type=str, default="left",
                       choices=["left", "right"],
                       help="Which arm to analyze")

    args = parser.parse_args()

    print("=" * 60)
    print("SmolVLA Trace Phase Analysis")
    print("=" * 60)

    compare_trajectories(
        hall_trace_path=Path(args.hallucination_trace),
        normal_trace_path=Path(args.normal_trace),
        output_dir=Path(args.output_dir),
        arm=args.arm,
    )


if __name__ == "__main__":
    main()
