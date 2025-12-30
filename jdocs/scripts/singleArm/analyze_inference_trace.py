#!/usr/bin/env python3
"""
Analyze inference traces from Pi0.5 and SmolVLA RTC trace scripts.

This script loads trace data and provides:
1. Timing analysis (loop, inference, capture durations)
2. RTC-specific analysis (queue depth, inference delays, chunk transitions)
3. Action analysis (magnitude, deltas, per-joint stats)
4. Observation freshness analysis
5. Anomaly detection (slow loops, queue underflows, action jumps)
6. Visualization (timeline, trajectories, histograms)

Usage:
    python analyze_inference_trace.py --trace-dir outputs/inference_traces/trace_YYYYMMDD_HHMMSS
    python analyze_inference_trace.py --trace-dir <path> --show-plots
    python analyze_inference_trace.py --trace-dir <path> --save-plots

Reference:
    - jdocs/scripts/infer_pi05_rtc_trace.py
    - jdocs/scripts/infer_smolvla_rtc_trace.py
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TraceAnalysis:
    """Container for trace analysis results."""
    total_steps: int
    total_time_s: float
    effective_rate_hz: float
    num_inferences: int

    # Timing stats
    loop_duration_stats: dict
    inference_duration_stats: dict
    capture_duration_stats: dict
    time_between_inferences_stats: dict

    # RTC stats
    inference_delay_stats: dict
    queue_size_stats: dict
    queue_empty_count: int

    # Observation freshness
    observation_age_stats: dict

    # Action stats
    action_magnitude_stats: dict
    action_delta_stats: dict
    per_joint_stats: list

    # Anomalies
    slow_loops: list
    slow_inferences: list
    action_jumps: list
    queue_underflows: list


def load_trace(trace_dir: Path) -> tuple[list[dict], dict, dict]:
    """Load trace data from directory.

    Returns:
        traces: List of trace entries
        summary: Summary statistics (if exists)
        config: Run configuration (if exists)
    """
    trace_file = trace_dir / "trace.jsonl"
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    traces = []
    with open(trace_file) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    summary = {}
    summary_file = trace_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

    config = {}
    config_file = trace_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return traces, summary, config


def compute_stats(values: list, name: str = "") -> dict:
    """Compute statistics for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0, "count": 0}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "count": len(arr),
    }


def analyze_timing(traces: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Analyze timing characteristics.

    Returns:
        loop_stats, inference_stats, capture_stats, between_inference_stats
    """
    loop_durations = []
    inference_durations = []
    capture_durations = []
    time_between_inferences = []

    last_inference_time = None

    for entry in traces:
        loop_durations.append(entry.get("loop_duration_ms", 0))

        if entry.get("inference_triggered"):
            inf_dur = entry.get("inference_duration_ms")
            if inf_dur is not None and inf_dur > 0:
                inference_durations.append(inf_dur)

            cap_dur = entry.get("capture_duration_ms")
            if cap_dur is not None and cap_dur > 0:
                capture_durations.append(cap_dur)

            inf_time = entry.get("t_inference_end", 0)
            if last_inference_time is not None and inf_time > 0:
                time_between_inferences.append((inf_time - last_inference_time) * 1000)
            last_inference_time = inf_time

    return (
        compute_stats(loop_durations, "loop_duration"),
        compute_stats(inference_durations, "inference_duration"),
        compute_stats(capture_durations, "capture_duration"),
        compute_stats(time_between_inferences, "between_inference"),
    )


def analyze_rtc_metrics(traces: list[dict]) -> tuple[dict, dict, int]:
    """Analyze RTC-specific metrics.

    Returns:
        inference_delay_stats, queue_size_stats, queue_empty_count
    """
    inference_delays = []
    queue_sizes = []
    queue_empty_count = 0

    for entry in traces:
        delay = entry.get("inference_delay")
        if delay is not None:
            inference_delays.append(delay)

        queue_size = entry.get("queue_size_before", 0)
        queue_sizes.append(queue_size)

        # Count queue underflows
        if entry.get("action_idx_in_chunk", 0) < 0:
            queue_empty_count += 1

    return (
        compute_stats(inference_delays, "inference_delay"),
        compute_stats(queue_sizes, "queue_size"),
        queue_empty_count,
    )


def analyze_observation_freshness(traces: list[dict]) -> dict:
    """Analyze how old observations are when actions execute.

    The observation is captured at inference time. Actions from that
    inference execute with increasingly stale observations.
    """
    observation_ages_ms = []

    last_capture_time = None

    for entry in traces:
        if entry.get("inference_triggered"):
            last_capture_time = entry.get("t_capture_end", 0)

        if last_capture_time is not None:
            action_sent_time = entry.get("t_action_sent", 0)
            if action_sent_time > 0:
                age = (action_sent_time - last_capture_time) * 1000
                observation_ages_ms.append(age)

    return compute_stats(observation_ages_ms, "observation_age")


def analyze_actions(traces: list[dict]) -> tuple[dict, dict, list]:
    """Analyze action characteristics.

    Returns:
        magnitude_stats, delta_stats, per_joint_stats
    """
    magnitudes = []
    deltas = []
    per_joint = {i: [] for i in range(6)}
    per_joint_delta = {i: [] for i in range(6)}

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    for entry in traces:
        action = entry.get("action_executed", [])
        delta = entry.get("action_delta", [])

        if action and len(action) == 6:
            # Action magnitude (L2 norm of arm joints, excluding gripper)
            arm_action = np.array(action[:5])
            magnitudes.append(float(np.linalg.norm(arm_action)))

            for i, v in enumerate(action):
                per_joint[i].append(v)

        if delta and len(delta) == 6:
            # Delta magnitude
            arm_delta = np.array(delta[:5])
            deltas.append(float(np.linalg.norm(arm_delta)))

            for i, v in enumerate(delta):
                per_joint_delta[i].append(v)

    # Per-joint stats
    joint_stats = []
    for i in range(6):
        joint_stats.append({
            "name": joint_names[i],
            "action": compute_stats(per_joint[i]),
            "delta": compute_stats(per_joint_delta[i]),
        })

    return compute_stats(magnitudes), compute_stats(deltas), joint_stats


def detect_anomalies(traces: list[dict]) -> tuple[list, list, list, list]:
    """Detect anomalies in the trace.

    Returns:
        slow_loops, slow_inferences, action_jumps, queue_underflows
    """
    # Thresholds
    loop_threshold_ms = 50  # Expected ~33ms at 30Hz
    inference_threshold_ms = 150  # Inference shouldn't take this long
    action_jump_threshold_deg = 30  # Large sudden movements

    slow_loops = []
    slow_inferences = []
    action_jumps = []
    queue_underflows = []

    prev_action = None

    for entry in traces:
        step = entry.get("step", 0)

        # Check loop duration
        loop_dur = entry.get("loop_duration_ms", 0)
        if loop_dur > loop_threshold_ms:
            slow_loops.append({
                "step": step,
                "duration_ms": loop_dur,
                "inference_triggered": entry.get("inference_triggered", False)
            })

        # Check inference duration
        if entry.get("inference_triggered"):
            inf_dur = entry.get("inference_duration_ms")
            if inf_dur is not None and inf_dur > inference_threshold_ms:
                slow_inferences.append({
                    "step": step,
                    "duration_ms": inf_dur
                })

        # Check action jumps
        action = entry.get("action_executed", [])
        if action and prev_action and len(action) == 6 and len(prev_action) == 6:
            delta = np.array(action) - np.array(prev_action)
            max_jump = float(np.max(np.abs(delta[:5])))  # Arm joints only
            if max_jump > action_jump_threshold_deg:
                action_jumps.append({
                    "step": step,
                    "max_delta_deg": max_jump,
                    "joint_deltas": delta.tolist()
                })
        if action:
            prev_action = action

        # Check queue underflow
        if entry.get("action_idx_in_chunk", 0) < 0:
            queue_underflows.append({
                "step": step,
                "queue_size_before": entry.get("queue_size_before", 0)
            })

    return slow_loops, slow_inferences, action_jumps, queue_underflows


def analyze_chunk_transitions(traces: list[dict]) -> list[dict]:
    """Analyze transitions between action chunks."""
    transitions = []

    for entry in traces:
        if entry.get("inference_triggered"):
            transitions.append({
                "step": entry.get("step", 0),
                "t_inference_end": entry.get("t_inference_end", 0),
                "inference_delay": entry.get("inference_delay"),
                "latency_ms": entry.get("latency_ms"),
                "queue_size_before": entry.get("queue_size_before", 0),
                "action_buffer_size": len(entry.get("action_buffer", [])),
            })

    return transitions


def print_analysis(analysis: TraceAnalysis, config: dict):
    """Print analysis results to console."""
    print("=" * 70)
    print("INFERENCE TRACE ANALYSIS")
    print("=" * 70)

    if config:
        print(f"\nConfiguration:")
        print(f"  Model Type:   {config.get('model_type', 'N/A')}")
        print(f"  Checkpoint:   {config.get('checkpoint', 'N/A')}")
        print(f"  Task:         {config.get('task', 'N/A')}")
        print(f"  RTC Enabled:  {config.get('rtc_enabled', 'N/A')}")
        print(f"  Duration:     {config.get('duration', 'N/A')}s")

    print(f"\nOverview:")
    print(f"  Total steps:      {analysis.total_steps}")
    print(f"  Total time:       {analysis.total_time_s:.1f}s")
    print(f"  Effective rate:   {analysis.effective_rate_hz:.1f}Hz")
    print(f"  Num inferences:   {analysis.num_inferences}")

    print(f"\n{'-' * 70}")
    print("TIMING ANALYSIS")
    print(f"{'-' * 70}")

    print(f"\nLoop Duration (ms) - Target: 33.3ms @ 30Hz")
    stats = analysis.loop_duration_stats
    print(f"  Mean:  {stats['mean']:.2f} +/- {stats['std']:.2f}")
    print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    print(f"  P50:   {stats['p50']:.1f}")
    print(f"  P95:   {stats['p95']:.1f}")
    print(f"  P99:   {stats['p99']:.1f}")

    print(f"\nInference Duration (ms)")
    stats = analysis.inference_duration_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.2f} +/- {stats['std']:.2f}")
        print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    else:
        print("  No inference data")

    print(f"\nCapture Duration (ms)")
    stats = analysis.capture_duration_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.2f} +/- {stats['std']:.2f}")
        print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    else:
        print("  No capture data")

    print(f"\nTime Between Inferences (ms)")
    stats = analysis.time_between_inferences_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.1f} +/- {stats['std']:.1f}")
        print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    else:
        print("  No inference timing data")

    print(f"\n{'-' * 70}")
    print("RTC METRICS")
    print(f"{'-' * 70}")

    print(f"\nInference Delay (steps)")
    stats = analysis.inference_delay_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.1f} +/- {stats['std']:.1f}")
        print(f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
    else:
        print("  No delay data")

    print(f"\nQueue Size")
    stats = analysis.queue_size_stats
    print(f"  Mean:  {stats['mean']:.1f} +/- {stats['std']:.1f}")
    print(f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
    print(f"  Queue empty events: {analysis.queue_empty_count}")

    print(f"\n{'-' * 70}")
    print("OBSERVATION FRESHNESS")
    print(f"{'-' * 70}")

    print(f"\nObservation Age at Action Execution (ms)")
    stats = analysis.observation_age_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.1f} +/- {stats['std']:.1f}")
        print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"  P95:   {stats['p95']:.1f}")
        print(f"  Note: Higher values mean actions based on stale observations")
    else:
        print("  No observation age data")

    print(f"\n{'-' * 70}")
    print("ACTION ANALYSIS")
    print(f"{'-' * 70}")

    print(f"\nAction Magnitude (arm joints L2 norm, degrees)")
    stats = analysis.action_magnitude_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.1f} +/- {stats['std']:.1f}")
        print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    else:
        print("  No action data")

    print(f"\nAction Delta Magnitude (degrees)")
    stats = analysis.action_delta_stats
    if stats['count'] > 0:
        print(f"  Mean:  {stats['mean']:.2f} +/- {stats['std']:.2f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    else:
        print("  No delta data")

    print(f"\n{'-' * 70}")
    print("ANOMALY DETECTION")
    print(f"{'-' * 70}")

    print(f"\nSlow Loops (>50ms): {len(analysis.slow_loops)}")
    if analysis.slow_loops[:5]:
        for sl in analysis.slow_loops[:5]:
            inf_str = " [INF]" if sl.get("inference_triggered") else ""
            print(f"  Step {sl['step']}: {sl['duration_ms']:.1f}ms{inf_str}")
        if len(analysis.slow_loops) > 5:
            print(f"  ... and {len(analysis.slow_loops) - 5} more")

    print(f"\nSlow Inferences (>150ms): {len(analysis.slow_inferences)}")
    if analysis.slow_inferences[:5]:
        for si in analysis.slow_inferences[:5]:
            print(f"  Step {si['step']}: {si['duration_ms']:.1f}ms")

    print(f"\nAction Jumps (>30 deg): {len(analysis.action_jumps)}")
    if analysis.action_jumps[:5]:
        for aj in analysis.action_jumps[:5]:
            print(f"  Step {aj['step']}: max delta {aj['max_delta_deg']:.1f} deg")

    print(f"\nQueue Underflows: {len(analysis.queue_underflows)}")
    if analysis.queue_underflows[:5]:
        for qu in analysis.queue_underflows[:5]:
            print(f"  Step {qu['step']}: queue_size_before={qu['queue_size_before']}")

    print(f"\n{'=' * 70}")


def create_visualizations(traces: list[dict], output_dir: Path, show_plots: bool = False):
    """Create visualization plots."""
    try:
        import matplotlib
        if not show_plots:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping visualizations")
        return

    print("\nGenerating visualizations...")

    # Extract data
    steps = [e.get("step", 0) for e in traces]
    loop_durations = [e.get("loop_duration_ms", 0) for e in traces]

    inference_steps = [e.get("step", 0) for e in traces if e.get("inference_triggered")]
    inference_durations = [e.get("inference_duration_ms", 0) for e in traces
                           if e.get("inference_triggered") and e.get("inference_duration_ms")]

    queue_sizes = [e.get("queue_size_before", 0) for e in traces]

    # Joint trajectories
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    joint_trajectories = {j: [] for j in joint_names}
    joint_states = {j: [] for j in joint_names}

    for entry in traces:
        action = entry.get("action_executed", [])
        state = entry.get("joint_states", [])
        if action and len(action) == 6:
            for i, name in enumerate(joint_names):
                joint_trajectories[name].append(action[i])
        if state and len(state) == 6:
            for i, name in enumerate(joint_names):
                joint_states[name].append(state[i])

    # 1. Timing and Queue Analysis
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Loop duration
    ax = axes[0]
    ax.plot(steps, loop_durations, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=33.3, color='g', linestyle='--', label='Target (33.3ms @ 30Hz)')
    ax.axhline(y=50, color='r', linestyle='--', label='Warning threshold (50ms)')
    for inf_step in inference_steps:
        ax.axvline(x=inf_step, color='orange', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loop Duration (ms)')
    ax.set_title('Loop Duration Over Time (orange = inference steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inference duration
    ax = axes[1]
    if inference_steps and inference_durations:
        ax.bar(inference_steps[:len(inference_durations)], inference_durations,
               color='purple', alpha=0.7, width=1)
        ax.axhline(y=np.mean(inference_durations), color='g', linestyle='--',
                   label=f'Mean ({np.mean(inference_durations):.1f}ms)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Inference Duration (ms)')
    ax.set_title('Inference Duration at Each Inference Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Queue size
    ax = axes[2]
    ax.plot(steps, queue_sizes, 'g-', linewidth=0.5, alpha=0.7)
    for inf_step in inference_steps:
        ax.axvline(x=inf_step, color='orange', alpha=0.3, linewidth=0.5)
    ax.axhline(y=30, color='r', linestyle='--', label='Refill threshold (30)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Size')
    ax.set_title('Action Queue Size Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Observation age
    observation_ages = []
    last_capture_time = None
    for entry in traces:
        if entry.get("inference_triggered"):
            last_capture_time = entry.get("t_capture_end", 0)
        if last_capture_time is not None:
            action_sent_time = entry.get("t_action_sent", 0)
            if action_sent_time > 0:
                age = (action_sent_time - last_capture_time) * 1000
                observation_ages.append(age)
            else:
                observation_ages.append(0)
        else:
            observation_ages.append(0)

    ax = axes[3]
    ax.plot(steps[:len(observation_ages)], observation_ages, 'r-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Observation Age (ms)')
    ax.set_title('Age of Observation When Action Executes')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "timing_analysis.png", dpi=150)
    print(f"  Saved: timing_analysis.png")

    # 2. Joint Trajectories
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(joint_names):
        ax = axes[i]
        if joint_states[name]:
            ax.plot(steps[:len(joint_states[name])], joint_states[name],
                   'b-', linewidth=0.5, alpha=0.5, label='State')
        if joint_trajectories[name]:
            ax.plot(steps[:len(joint_trajectories[name])], joint_trajectories[name],
                   'r-', linewidth=0.5, alpha=0.7, label='Action')
        ax.set_xlabel('Step')
        ax.set_ylabel('Position (deg)')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "joint_trajectories.png", dpi=150)
    print(f"  Saved: joint_trajectories.png")

    # 3. Duration Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(loop_durations, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=33.3, color='g', linestyle='--', label='Target 33.3ms')
    ax.axvline(x=np.mean(loop_durations), color='r', linestyle='--',
               label=f'Mean {np.mean(loop_durations):.1f}ms')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Loop Duration Distribution')
    ax.legend()

    ax = axes[1]
    if inference_durations:
        ax.hist(inference_durations, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(inference_durations), color='r', linestyle='--',
                   label=f'Mean {np.mean(inference_durations):.1f}ms')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Inference Duration Distribution')
    ax.legend()

    ax = axes[2]
    ax.hist(queue_sizes, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(queue_sizes), color='r', linestyle='--',
               label=f'Mean {np.mean(queue_sizes):.1f}')
    ax.set_xlabel('Queue Size')
    ax.set_ylabel('Count')
    ax.set_title('Queue Size Distribution')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "duration_histograms.png", dpi=150)
    print(f"  Saved: duration_histograms.png")

    # 4. Action Deltas
    deltas_by_joint = {name: [] for name in joint_names}
    for entry in traces:
        delta = entry.get("action_delta", [])
        if delta and len(delta) == 6:
            for i, name in enumerate(joint_names):
                deltas_by_joint[name].append(delta[i])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(joint_names):
        ax = axes[i]
        if deltas_by_joint[name]:
            ax.plot(steps[:len(deltas_by_joint[name])], deltas_by_joint[name],
                   'g-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Delta (deg)')
        ax.set_title(f'{name} - Action Delta (command - current)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "action_deltas.png", dpi=150)
    print(f"  Saved: action_deltas.png")

    if show_plots:
        plt.show()
    else:
        plt.close('all')


def save_analysis_report(analysis: TraceAnalysis, chunk_transitions: list,
                         output_dir: Path, config: dict):
    """Save detailed analysis report to JSON."""
    report = {
        "overview": {
            "model_type": config.get("model_type", "unknown"),
            "total_steps": analysis.total_steps,
            "total_time_s": analysis.total_time_s,
            "effective_rate_hz": analysis.effective_rate_hz,
            "num_inferences": analysis.num_inferences,
        },
        "timing": {
            "loop_duration_ms": analysis.loop_duration_stats,
            "inference_duration_ms": analysis.inference_duration_stats,
            "capture_duration_ms": analysis.capture_duration_stats,
            "time_between_inferences_ms": analysis.time_between_inferences_stats,
        },
        "rtc": {
            "inference_delay_steps": analysis.inference_delay_stats,
            "queue_size": analysis.queue_size_stats,
            "queue_empty_count": analysis.queue_empty_count,
        },
        "observation_freshness": {
            "observation_age_ms": analysis.observation_age_stats,
        },
        "actions": {
            "magnitude": analysis.action_magnitude_stats,
            "delta": analysis.action_delta_stats,
            "per_joint": analysis.per_joint_stats,
        },
        "anomalies": {
            "slow_loops_count": len(analysis.slow_loops),
            "slow_loops": analysis.slow_loops[:20],
            "slow_inferences_count": len(analysis.slow_inferences),
            "slow_inferences": analysis.slow_inferences[:20],
            "action_jumps_count": len(analysis.action_jumps),
            "action_jumps": analysis.action_jumps[:20],
            "queue_underflows_count": len(analysis.queue_underflows),
            "queue_underflows": analysis.queue_underflows[:20],
        },
        "chunk_transitions": chunk_transitions[:50],
    }

    report_file = output_dir / "analysis_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inference traces from Pi0.5/SmolVLA RTC scripts"
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        required=True,
        help="Path to trace directory containing trace.jsonl"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save plots to trace directory (default: True)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        sys.exit(1)

    # Load traces
    print(f"Loading traces from: {trace_dir}")
    traces, summary, config = load_trace(trace_dir)
    print(f"Loaded {len(traces)} trace entries")

    if not traces:
        print("Error: No trace entries found")
        sys.exit(1)

    # Run analysis
    print("\nAnalyzing...")

    # Timing analysis
    loop_stats, inf_stats, cap_stats, between_inf_stats = analyze_timing(traces)

    # RTC analysis
    delay_stats, queue_stats, queue_empty_count = analyze_rtc_metrics(traces)

    # Observation freshness
    obs_age_stats = analyze_observation_freshness(traces)

    # Action analysis
    mag_stats, delta_stats, joint_stats = analyze_actions(traces)

    # Anomaly detection
    slow_loops, slow_infs, action_jumps, queue_underflows = detect_anomalies(traces)

    # Chunk transitions
    chunk_transitions = analyze_chunk_transitions(traces)

    # Build analysis result
    total_time = traces[-1].get("t_loop_end", 0) if traces else 0
    analysis = TraceAnalysis(
        total_steps=len(traces),
        total_time_s=total_time,
        effective_rate_hz=len(traces) / total_time if total_time > 0 else 0,
        num_inferences=sum(1 for e in traces if e.get("inference_triggered")),
        loop_duration_stats=loop_stats,
        inference_duration_stats=inf_stats,
        capture_duration_stats=cap_stats,
        time_between_inferences_stats=between_inf_stats,
        inference_delay_stats=delay_stats,
        queue_size_stats=queue_stats,
        queue_empty_count=queue_empty_count,
        observation_age_stats=obs_age_stats,
        action_magnitude_stats=mag_stats,
        action_delta_stats=delta_stats,
        per_joint_stats=joint_stats,
        slow_loops=slow_loops,
        slow_inferences=slow_infs,
        action_jumps=action_jumps,
        queue_underflows=queue_underflows,
    )

    # Print results
    print_analysis(analysis, config)

    # Save report
    save_analysis_report(analysis, chunk_transitions, trace_dir, config)

    # Generate visualizations
    if not args.no_plots:
        create_visualizations(traces, trace_dir, args.show_plots)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutput files saved to: {trace_dir}")
    print("  - analysis_report.json (detailed report)")
    if not args.no_plots:
        print("  - timing_analysis.png")
        print("  - joint_trajectories.png")
        print("  - duration_histograms.png")
        print("  - action_deltas.png")


if __name__ == "__main__":
    main()
