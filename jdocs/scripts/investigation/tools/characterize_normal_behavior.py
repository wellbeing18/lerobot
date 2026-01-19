#!/usr/bin/env python3
"""
Characterize Normal VLA Behavior Tool

Analyzes normal (non-hallucinating) inference traces to understand:
1. How does the model transition from "active" to "idle" behavior?
2. What are the action patterns during idle phase?
3. At what step does the transition occur?
4. What are the characteristics of "stay still" actions?

This establishes a baseline understanding of correct behavior before
investigating what goes wrong in hallucination cases.

Usage:
    python characterize_normal_behavior.py \
        --case-dir logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --output-dir logs/yogurt_banana_leftarm/normal_mechanism_analysis
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


@dataclass
class PhaseTransition:
    """Represents a transition between task phases."""
    from_phase: str
    to_phase: str
    step: int
    action_delta_before: float
    action_delta_after: float
    gripper_before: float
    gripper_after: float
    joint_positions: list


@dataclass
class IdleCharacteristics:
    """Characteristics of the idle/stay-still phase."""
    start_step: int
    duration_steps: int
    mean_action_delta: float
    std_action_delta: float
    max_action_delta: float
    mean_gripper: float
    mean_joint_positions: list
    action_delta_percentiles: dict  # 10th, 50th, 90th, 99th


@dataclass
class ActiveCharacteristics:
    """Characteristics of the active phase."""
    start_step: int
    end_step: int
    duration_steps: int
    mean_action_delta: float
    peak_action_delta: float
    peak_step: int
    gripper_open_ratio: float
    phases: list  # approach, transport, place


@dataclass
class NormalBehaviorAnalysis:
    """Complete analysis of normal behavior."""
    case_path: str
    total_steps: int

    # Transition analysis
    active_to_idle_transition: PhaseTransition

    # Phase characteristics
    active_characteristics: ActiveCharacteristics
    idle_characteristics: IdleCharacteristics

    # Chunk boundary analysis
    chunk_boundaries: list
    action_delta_at_boundaries: list

    # Key metrics
    idle_ratio: float  # What fraction of episode is idle
    transition_sharpness: float  # How abrupt is the transition


def load_trace(trace_path: Path) -> list[dict]:
    """Load inference trace from JSONL file."""
    entries = []
    with open(trace_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compute_action_deltas(entries: list[dict], left_arm_only: bool = True) -> np.ndarray:
    """Compute action delta (action - state) magnitude per step."""
    deltas = []
    for entry in entries:
        state = np.array(entry.get('state_raw', [0]*12))
        action = np.array(entry.get('action_raw', [0]*12))

        if left_arm_only:
            state = state[:6]
            action = action[:6]

        delta = np.linalg.norm(action - state)
        deltas.append(delta)

    return np.array(deltas)


def compute_position_changes(entries: list[dict], left_arm_only: bool = True) -> np.ndarray:
    """Compute actual position change between consecutive steps."""
    changes = [0.0]  # First step has no previous

    for i in range(1, len(entries)):
        prev_state = np.array(entries[i-1].get('state_raw', [0]*12))
        curr_state = np.array(entries[i].get('state_raw', [0]*12))

        if left_arm_only:
            prev_state = prev_state[:6]
            curr_state = curr_state[:6]

        change = np.linalg.norm(curr_state - prev_state)
        changes.append(change)

    return np.array(changes)


def find_idle_transition(
    entries: list[dict],
    action_deltas: np.ndarray,
    idle_threshold: float = 3.0,
    sustained_frames: int = 20
) -> Optional[int]:
    """
    Find the step where the model transitions to idle behavior.

    Idle is defined as: action_delta < threshold for sustained_frames consecutive frames.
    """
    is_idle = action_deltas < idle_threshold

    for i in range(len(is_idle) - sustained_frames):
        if np.all(is_idle[i:i + sustained_frames]):
            return i

    return None


def detect_task_phases(
    entries: list[dict],
    action_deltas: np.ndarray
) -> list[dict]:
    """Detect task phases based on gripper state and action patterns."""
    phases = []

    # Get gripper states
    gripper_states = []
    for entry in entries:
        state = entry.get('state_raw', [0]*12)
        gripper_states.append(state[5])  # Left gripper

    gripper_states = np.array(gripper_states)

    # Detect gripper events
    gripper_open = gripper_states > 15
    gripper_closed = gripper_states < 5

    # Find gripper close event (pick)
    pick_step = None
    for i in range(1, len(gripper_states)):
        if gripper_closed[i] and not gripper_closed[i-1]:
            pick_step = i
            break

    # Find gripper open event after pick (place)
    place_step = None
    if pick_step:
        for i in range(pick_step + 10, len(gripper_states)):
            if gripper_open[i] and not gripper_open[i-1]:
                place_step = i
                break

    # Build phase list
    if pick_step:
        phases.append({
            'name': 'approach',
            'start': 0,
            'end': pick_step,
            'duration': pick_step,
            'mean_delta': float(np.mean(action_deltas[:pick_step]))
        })

        if place_step:
            phases.append({
                'name': 'transport',
                'start': pick_step,
                'end': place_step,
                'duration': place_step - pick_step,
                'mean_delta': float(np.mean(action_deltas[pick_step:place_step]))
            })

            # Find where arm settles (return/idle transition)
            idle_start = find_idle_transition(
                entries[place_step:],
                action_deltas[place_step:],
                idle_threshold=3.0,
                sustained_frames=20
            )

            if idle_start:
                idle_start += place_step  # Adjust for offset
                phases.append({
                    'name': 'place_and_return',
                    'start': place_step,
                    'end': idle_start,
                    'duration': idle_start - place_step,
                    'mean_delta': float(np.mean(action_deltas[place_step:idle_start]))
                })
                phases.append({
                    'name': 'idle',
                    'start': idle_start,
                    'end': len(entries),
                    'duration': len(entries) - idle_start,
                    'mean_delta': float(np.mean(action_deltas[idle_start:]))
                })
            else:
                phases.append({
                    'name': 'place_and_return',
                    'start': place_step,
                    'end': len(entries),
                    'duration': len(entries) - place_step,
                    'mean_delta': float(np.mean(action_deltas[place_step:]))
                })

    return phases


def analyze_idle_phase(
    entries: list[dict],
    action_deltas: np.ndarray,
    idle_start: int
) -> IdleCharacteristics:
    """Analyze the characteristics of the idle phase."""
    idle_deltas = action_deltas[idle_start:]

    # Get mean joint positions during idle
    idle_positions = []
    idle_grippers = []
    for entry in entries[idle_start:]:
        state = entry.get('state_raw', [0]*12)
        idle_positions.append(state[:6])  # Left arm
        idle_grippers.append(state[5])    # Left gripper

    idle_positions = np.array(idle_positions)

    return IdleCharacteristics(
        start_step=idle_start,
        duration_steps=len(idle_deltas),
        mean_action_delta=float(np.mean(idle_deltas)),
        std_action_delta=float(np.std(idle_deltas)),
        max_action_delta=float(np.max(idle_deltas)),
        mean_gripper=float(np.mean(idle_grippers)),
        mean_joint_positions=[float(x) for x in np.mean(idle_positions, axis=0)],
        action_delta_percentiles={
            'p10': float(np.percentile(idle_deltas, 10)),
            'p50': float(np.percentile(idle_deltas, 50)),
            'p90': float(np.percentile(idle_deltas, 90)),
            'p99': float(np.percentile(idle_deltas, 99))
        }
    )


def analyze_active_phase(
    entries: list[dict],
    action_deltas: np.ndarray,
    phases: list[dict]
) -> ActiveCharacteristics:
    """Analyze the characteristics of the active phase."""
    # Find where active phase ends (idle begins)
    idle_phase = next((p for p in phases if p['name'] == 'idle'), None)
    active_end = idle_phase['start'] if idle_phase else len(entries)

    active_deltas = action_deltas[:active_end]

    # Get gripper open ratio
    gripper_open_count = 0
    for entry in entries[:active_end]:
        state = entry.get('state_raw', [0]*12)
        if state[5] > 15:  # Gripper open threshold
            gripper_open_count += 1

    return ActiveCharacteristics(
        start_step=0,
        end_step=active_end,
        duration_steps=active_end,
        mean_action_delta=float(np.mean(active_deltas)),
        peak_action_delta=float(np.max(active_deltas)),
        peak_step=int(np.argmax(active_deltas)),
        gripper_open_ratio=gripper_open_count / active_end if active_end > 0 else 0,
        phases=[p['name'] for p in phases if p['name'] != 'idle']
    )


def analyze_chunk_boundaries(
    entries: list[dict],
    action_deltas: np.ndarray,
    chunk_size: int = 50
) -> tuple[list[int], list[float]]:
    """Analyze action deltas at chunk boundaries."""
    boundaries = list(range(0, len(entries), chunk_size))
    deltas_at_boundaries = [float(action_deltas[b]) for b in boundaries if b < len(action_deltas)]
    return boundaries, deltas_at_boundaries


def compute_transition_sharpness(
    action_deltas: np.ndarray,
    transition_step: int,
    window: int = 10
) -> float:
    """
    Compute how sharp the transition from active to idle is.
    Higher value = more abrupt transition.
    """
    if transition_step < window or transition_step + window >= len(action_deltas):
        return 0.0

    before = np.mean(action_deltas[transition_step - window:transition_step])
    after = np.mean(action_deltas[transition_step:transition_step + window])

    # Sharpness = ratio of before to after
    return float(before / after) if after > 0 else float('inf')


def characterize_normal_behavior(
    case_dir: Path,
    output_dir: Path
) -> NormalBehaviorAnalysis:
    """Main analysis function."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trace
    trace_path = case_dir / 'trace.jsonl'
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    entries = load_trace(trace_path)
    print(f"Loaded {len(entries)} steps from {trace_path}")

    # Compute action deltas
    action_deltas = compute_action_deltas(entries)
    position_changes = compute_position_changes(entries)

    # Detect phases
    phases = detect_task_phases(entries, action_deltas)
    print(f"Detected phases: {[p['name'] for p in phases]}")

    # Find idle transition
    idle_phase = next((p for p in phases if p['name'] == 'idle'), None)
    if not idle_phase:
        print("WARNING: No idle phase detected!")
        idle_start = len(entries)
    else:
        idle_start = idle_phase['start']
        print(f"Idle transition at step {idle_start}")

    # Analyze phases
    idle_chars = analyze_idle_phase(entries, action_deltas, idle_start) if idle_phase else None
    active_chars = analyze_active_phase(entries, action_deltas, phases)

    # Analyze chunk boundaries
    boundaries, deltas_at_boundaries = analyze_chunk_boundaries(entries, action_deltas)

    # Compute transition characteristics
    if idle_phase:
        transition_step = idle_start

        # Get state before and after transition
        state_before = np.array(entries[transition_step - 1].get('state_raw', [0]*12))
        state_after = np.array(entries[transition_step].get('state_raw', [0]*12))

        transition = PhaseTransition(
            from_phase='place_and_return',
            to_phase='idle',
            step=transition_step,
            action_delta_before=float(action_deltas[transition_step - 1]),
            action_delta_after=float(action_deltas[transition_step]),
            gripper_before=float(state_before[5]),
            gripper_after=float(state_after[5]),
            joint_positions=[float(x) for x in state_after[:6]]
        )

        sharpness = compute_transition_sharpness(action_deltas, transition_step)
    else:
        transition = None
        sharpness = 0.0

    # Build analysis result
    analysis = NormalBehaviorAnalysis(
        case_path=str(case_dir),
        total_steps=len(entries),
        active_to_idle_transition=transition,
        active_characteristics=active_chars,
        idle_characteristics=idle_chars,
        chunk_boundaries=boundaries,
        action_delta_at_boundaries=deltas_at_boundaries,
        idle_ratio=idle_chars.duration_steps / len(entries) if idle_chars else 0.0,
        transition_sharpness=sharpness
    )

    # Save results
    output_file = output_dir / 'normal_behavior_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(asdict(analysis), f, indent=2)
    print(f"Analysis saved to {output_file}")

    # Generate visualizations
    generate_visualizations(entries, action_deltas, position_changes, phases, analysis, output_dir)

    # Print summary
    print_summary(analysis)

    return analysis


def generate_visualizations(
    entries: list[dict],
    action_deltas: np.ndarray,
    position_changes: np.ndarray,
    phases: list[dict],
    analysis: NormalBehaviorAnalysis,
    output_dir: Path
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    steps = np.arange(len(entries))

    # Plot 1: Action delta over time
    ax1 = axes[0]
    ax1.plot(steps, action_deltas, 'b-', linewidth=1, label='Action Delta')
    ax1.axhline(y=3.0, color='r', linestyle='--', label='Idle Threshold (3.0)')

    # Mark phase transitions
    colors = {'approach': 'green', 'transport': 'orange', 'place_and_return': 'purple', 'idle': 'gray'}
    for phase in phases:
        ax1.axvspan(phase['start'], phase['end'], alpha=0.2, color=colors.get(phase['name'], 'gray'), label=phase['name'])

    # Mark chunk boundaries
    for b in analysis.chunk_boundaries:
        ax1.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Action Delta (||action - state||)')
    ax1.set_title('Action Delta Over Time with Phase Annotations')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, len(entries))

    # Plot 2: Position change (actual movement)
    ax2 = axes[1]
    ax2.plot(steps, position_changes, 'g-', linewidth=1, label='Position Change')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Movement Threshold (1.0)')

    if analysis.active_to_idle_transition:
        ax2.axvline(x=analysis.active_to_idle_transition.step, color='red', linewidth=2,
                   label=f'Idle Transition (step {analysis.active_to_idle_transition.step})')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Position Change (||state_t - state_{t-1}||)')
    ax2.set_title('Actual Position Change Over Time')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, len(entries))

    # Plot 3: Gripper state
    ax3 = axes[2]
    gripper_states = [entry.get('state_raw', [0]*12)[5] for entry in entries]
    ax3.plot(steps, gripper_states, 'm-', linewidth=1, label='Left Gripper')
    ax3.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Open Threshold')
    ax3.axhline(y=5, color='b', linestyle='--', alpha=0.5, label='Closed Threshold')

    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gripper Position')
    ax3.set_title('Gripper State Over Time')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, len(entries))

    plt.tight_layout()
    plt.savefig(output_dir / 'normal_behavior_analysis.png', dpi=150)
    plt.close()
    print(f"Visualization saved to {output_dir / 'normal_behavior_analysis.png'}")

    # Additional plot: Idle phase detail
    if analysis.idle_characteristics:
        idle_start = analysis.idle_characteristics.start_step
        fig2, ax = plt.subplots(figsize=(12, 4))

        idle_steps = steps[idle_start:]
        idle_deltas = action_deltas[idle_start:]

        ax.plot(idle_steps, idle_deltas, 'b-', linewidth=1)
        ax.axhline(y=analysis.idle_characteristics.mean_action_delta, color='r',
                  linestyle='-', label=f'Mean: {analysis.idle_characteristics.mean_action_delta:.2f}')
        ax.axhline(y=analysis.idle_characteristics.mean_action_delta + analysis.idle_characteristics.std_action_delta,
                  color='r', linestyle='--', alpha=0.5, label=f'±1 std: {analysis.idle_characteristics.std_action_delta:.2f}')
        ax.axhline(y=analysis.idle_characteristics.mean_action_delta - analysis.idle_characteristics.std_action_delta,
                  color='r', linestyle='--', alpha=0.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Action Delta')
        ax.set_title(f'Idle Phase Detail (steps {idle_start}-{len(entries)})')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'idle_phase_detail.png', dpi=150)
        plt.close()
        print(f"Idle detail saved to {output_dir / 'idle_phase_detail.png'}")


def print_summary(analysis: NormalBehaviorAnalysis):
    """Print analysis summary."""
    print("\n" + "="*70)
    print("NORMAL BEHAVIOR ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nCase: {analysis.case_path}")
    print(f"Total steps: {analysis.total_steps}")
    print(f"Idle ratio: {analysis.idle_ratio*100:.1f}%")

    print(f"\n--- Active Phase ---")
    if analysis.active_characteristics:
        ac = analysis.active_characteristics
        print(f"Duration: {ac.duration_steps} steps (0 to {ac.end_step})")
        print(f"Mean action delta: {ac.mean_action_delta:.2f}")
        print(f"Peak action delta: {ac.peak_action_delta:.2f} at step {ac.peak_step}")
        print(f"Gripper open ratio: {ac.gripper_open_ratio*100:.1f}%")
        print(f"Phases: {' → '.join(ac.phases)}")

    print(f"\n--- Idle Phase ---")
    if analysis.idle_characteristics:
        ic = analysis.idle_characteristics
        print(f"Start step: {ic.start_step}")
        print(f"Duration: {ic.duration_steps} steps")
        print(f"Mean action delta: {ic.mean_action_delta:.2f} (std: {ic.std_action_delta:.2f})")
        print(f"Max action delta: {ic.max_action_delta:.2f}")
        print(f"Mean gripper: {ic.mean_gripper:.2f}")
        print(f"Percentiles: p10={ic.action_delta_percentiles['p10']:.2f}, "
              f"p50={ic.action_delta_percentiles['p50']:.2f}, "
              f"p90={ic.action_delta_percentiles['p90']:.2f}")
    else:
        print("No idle phase detected!")

    print(f"\n--- Transition ---")
    if analysis.active_to_idle_transition:
        t = analysis.active_to_idle_transition
        print(f"Transition step: {t.step}")
        print(f"Action delta: {t.action_delta_before:.2f} → {t.action_delta_after:.2f}")
        print(f"Transition sharpness: {analysis.transition_sharpness:.2f}x")

    print(f"\n--- Chunk Boundaries ---")
    print(f"Boundaries: {analysis.chunk_boundaries}")
    print(f"Action deltas at boundaries: {[f'{d:.1f}' for d in analysis.action_delta_at_boundaries]}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Characterize normal VLA behavior")
    parser.add_argument("--case-dir", type=str, required=True, help="Path to case directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    analysis = characterize_normal_behavior(
        case_dir=Path(args.case_dir),
        output_dir=Path(args.output_dir)
    )

    return analysis


if __name__ == "__main__":
    main()
