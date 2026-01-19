#!/usr/bin/env python3
"""
Ablation Runner for SmolVLA Hallucination Investigation.

Systematically runs controlled experiments to test hypotheses about
hallucination causes:

1. Visual Ablations: Object removal/addition, position variation
2. Language Ablations: Completion phrases, task description changes
3. Temporal Ablations: Early termination, chunk boundary analysis

Usage:
    # Run single experiment
    python run_ablation.py \
        --experiment language_completion \
        --checkpoint outputs/smolvla_bimanual \
        --output-dir ../reports/ablation_results

    # Run experiment suite from config
    python run_ablation.py \
        --config ablation_config.yaml \
        --output-dir ../reports/ablation_suite

    # List available experiments
    python run_ablation.py --list-experiments
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

EXPERIMENTS = {
    # Language ablations
    "language_completion_hold": {
        "type": "language",
        "description": "Add 'hold position' completion phrase at step 180",
        "params": {
            "dynamic_task": True,
            "completion_phrase": "Task complete. Hold position. Do not move.",
            "completion_trigger": "step_count",
            "completion_step": 180,
        },
    },
    "language_completion_stay_still": {
        "type": "language",
        "description": "Add 'stay still' completion phrase at step 180",
        "params": {
            "dynamic_task": True,
            "completion_phrase": "Stay still. Task is finished.",
            "completion_trigger": "step_count",
            "completion_step": 180,
        },
    },
    "language_completion_early": {
        "type": "language",
        "description": "Trigger completion earlier at step 150",
        "params": {
            "dynamic_task": True,
            "completion_phrase": "Task complete. Hold position.",
            "completion_trigger": "step_count",
            "completion_step": 150,
        },
    },
    "language_completion_late": {
        "type": "language",
        "description": "Trigger completion later at step 220",
        "params": {
            "dynamic_task": True,
            "completion_phrase": "Task complete. Hold position.",
            "completion_trigger": "step_count",
            "completion_step": 220,
        },
    },
    "language_completion_gripper": {
        "type": "language",
        "description": "Trigger completion based on gripper state",
        "params": {
            "dynamic_task": True,
            "completion_phrase": "Task complete. Hold position.",
            "completion_trigger": "gripper_close",
            "completion_step": 0,  # Not used for gripper trigger
        },
    },
    "language_completion_variance": {
        "type": "language",
        "description": "Trigger completion based on action variance",
        "params": {
            "dynamic_task": True,
            "completion_phrase": "Task complete. Hold position.",
            "completion_trigger": "action_variance",
            "completion_step": 0,
        },
    },

    # Temporal ablations
    "temporal_short_duration": {
        "type": "temporal",
        "description": "Short inference duration (30s)",
        "params": {
            "duration": 30,
        },
    },
    "temporal_long_duration": {
        "type": "temporal",
        "description": "Long inference duration (90s)",
        "params": {
            "duration": 90,
        },
    },

    # Baseline
    "baseline_no_modification": {
        "type": "baseline",
        "description": "Baseline run without any modifications",
        "params": {
            "dynamic_task": False,
        },
    },
}


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_name: str
    experiment_type: str
    description: str
    timestamp: str
    task_key: str
    checkpoint: str
    params: dict
    output_dir: str
    success: bool
    error_message: Optional[str] = None
    trace_file: Optional[str] = None
    metrics: dict = field(default_factory=dict)


@dataclass
class AblationSuite:
    """Collection of experiment results."""
    suite_name: str
    timestamp: str
    experiments: list[ExperimentResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    experiment_name: str,
    checkpoint: str,
    task_key: str,
    output_dir: Path,
    dry_run: bool = False,
    duration: float = 60.0,
) -> ExperimentResult:
    """
    Run a single ablation experiment.

    This calls the trace_inference.py tool with appropriate parameters.
    """
    if experiment_name not in EXPERIMENTS:
        return ExperimentResult(
            experiment_name=experiment_name,
            experiment_type="unknown",
            description="Unknown experiment",
            timestamp=datetime.now().isoformat(),
            task_key=task_key,
            checkpoint=checkpoint,
            params={},
            output_dir=str(output_dir),
            success=False,
            error_message=f"Unknown experiment: {experiment_name}",
        )

    exp_config = EXPERIMENTS[experiment_name]
    exp_output = output_dir / experiment_name

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Output: {exp_output}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "trace_inference.py"),
        "--checkpoint", checkpoint,
        "--task-key", task_key,
        "--output-dir", str(exp_output),
        "--duration", str(exp_config["params"].get("duration", duration)),
    ]

    # Add experiment-specific params
    params = exp_config["params"]

    if params.get("dynamic_task", False):
        cmd.append("--dynamic-task")
        cmd.extend(["--completion-phrase", params.get("completion_phrase", "Hold position.")])
        cmd.extend(["--completion-trigger", params.get("completion_trigger", "step_count")])
        cmd.extend(["--completion-step", str(params.get("completion_step", 180))])

    if dry_run:
        cmd.append("--dry-run")

    # Add image capture for analysis
    cmd.append("--capture-images")
    cmd.extend(["--capture-interval", "30"])

    print(f"Command: {' '.join(cmd)}")

    result = ExperimentResult(
        experiment_name=experiment_name,
        experiment_type=exp_config["type"],
        description=exp_config["description"],
        timestamp=datetime.now().isoformat(),
        task_key=task_key,
        checkpoint=checkpoint,
        params=params,
        output_dir=str(exp_output),
        success=False,
    )

    try:
        # Run the command
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 60,  # Add buffer for startup
        )

        if process.returncode == 0:
            result.success = True
            result.trace_file = str(exp_output / "trace.jsonl")

            # Try to load metrics from trace
            metrics = load_trace_metrics(exp_output / "trace.jsonl")
            result.metrics = metrics

            print(f"✓ Experiment completed successfully")
        else:
            result.error_message = process.stderr
            print(f"✗ Experiment failed: {process.stderr[:200]}")

    except subprocess.TimeoutExpired:
        result.error_message = "Experiment timed out"
        print(f"✗ Experiment timed out")

    except Exception as e:
        result.error_message = str(e)
        print(f"✗ Experiment error: {e}")

    return result


def load_trace_metrics(trace_file: Path) -> dict:
    """Load and compute metrics from trace file."""
    if not trace_file.exists():
        return {}

    metrics = {
        "total_steps": 0,
        "task_modification_step": None,
        "max_action_delta": 0,
        "avg_inference_time_ms": 0,
        "post_completion_movement": 0,
    }

    try:
        steps = []
        with open(trace_file) as f:
            for line in f:
                step = json.loads(line)
                steps.append(step)

        if not steps:
            return metrics

        metrics["total_steps"] = len(steps)

        # Find task modification step
        for step in steps:
            if step.get("task_modified", False):
                metrics["task_modification_step"] = step["step"]
                break

        # Calculate max action delta
        deltas = [step.get("action_delta_max", 0) for step in steps]
        metrics["max_action_delta"] = max(deltas)

        # Calculate average inference time
        inf_times = [step.get("inference_time_ms", 0) for step in steps]
        metrics["avg_inference_time_ms"] = sum(inf_times) / len(inf_times)

        # Calculate post-completion movement (if task was modified)
        if metrics["task_modification_step"]:
            mod_step = metrics["task_modification_step"]
            post_steps = [s for s in steps if s["step"] > mod_step]
            if post_steps:
                post_deltas = [s.get("action_delta_max", 0) for s in post_steps]
                metrics["post_completion_movement"] = sum(post_deltas) / len(post_deltas)

    except Exception as e:
        print(f"Warning: Could not load trace metrics: {e}")

    return metrics


def run_suite(
    config_path: Path,
    checkpoint: str,
    task_key: str,
    output_dir: Path,
    dry_run: bool = False,
) -> AblationSuite:
    """
    Run a suite of experiments from config file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    suite = AblationSuite(
        suite_name=config.get("name", "ablation_suite"),
        timestamp=datetime.now().isoformat(),
    )

    experiments = config.get("experiments", [])

    for exp_name in experiments:
        result = run_experiment(
            experiment_name=exp_name,
            checkpoint=checkpoint,
            task_key=task_key,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        suite.experiments.append(result)

    # Compute summary
    suite.summary = compute_suite_summary(suite.experiments)

    return suite


def compute_suite_summary(experiments: list[ExperimentResult]) -> dict:
    """Compute summary statistics across experiments."""
    summary = {
        "total_experiments": len(experiments),
        "successful": sum(1 for e in experiments if e.success),
        "failed": sum(1 for e in experiments if not e.success),
        "by_type": {},
    }

    # Group by type
    for exp in experiments:
        exp_type = exp.experiment_type
        if exp_type not in summary["by_type"]:
            summary["by_type"][exp_type] = {
                "count": 0,
                "successful": 0,
                "avg_post_completion_movement": [],
            }
        summary["by_type"][exp_type]["count"] += 1
        if exp.success:
            summary["by_type"][exp_type]["successful"] += 1
            if "post_completion_movement" in exp.metrics:
                summary["by_type"][exp_type]["avg_post_completion_movement"].append(
                    exp.metrics["post_completion_movement"]
                )

    # Compute averages
    for exp_type in summary["by_type"]:
        movements = summary["by_type"][exp_type]["avg_post_completion_movement"]
        if movements:
            summary["by_type"][exp_type]["avg_post_completion_movement"] = sum(movements) / len(movements)
        else:
            summary["by_type"][exp_type]["avg_post_completion_movement"] = None

    return summary


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_suite_report(suite: AblationSuite, output_path: Path):
    """Generate markdown report for ablation suite."""
    report = f"""# Ablation Experiment Results

**Suite**: {suite.suite_name}
**Generated**: {suite.timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Experiments | {suite.summary['total_experiments']} |
| Successful | {suite.summary['successful']} |
| Failed | {suite.summary['failed']} |

## Results by Type

"""

    for exp_type, stats in suite.summary.get("by_type", {}).items():
        report += f"""### {exp_type.title()} Experiments

- Count: {stats['count']}
- Successful: {stats['successful']}
- Avg Post-Completion Movement: {stats['avg_post_completion_movement']:.2f if stats['avg_post_completion_movement'] else 'N/A'}

"""

    report += """## Individual Experiments

| Experiment | Type | Success | Post-Completion Movement | Notes |
|------------|------|---------|--------------------------|-------|
"""

    for exp in suite.experiments:
        movement = exp.metrics.get("post_completion_movement", "N/A")
        if isinstance(movement, float):
            movement = f"{movement:.2f}"
        notes = exp.error_message[:50] if exp.error_message else exp.description[:50]
        success = "✓" if exp.success else "✗"
        report += f"| {exp.experiment_name} | {exp.experiment_type} | {success} | {movement} | {notes} |\n"

    report += """
## Key Findings

Based on the ablation results:

1. **Language Completion Phrases**: [Analyze which phrases work best]
2. **Trigger Timing**: [Compare early vs late completion triggers]
3. **Trigger Methods**: [Compare step_count vs gripper_close vs variance]

## Recommendations

[To be filled based on analysis]
"""

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Saved report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Runner for SmolVLA Hallucination Investigation"
    )

    # Experiment selection
    parser.add_argument("--experiment", "-e", help="Single experiment to run")
    parser.add_argument("--config", type=Path, help="Config file for experiment suite")
    parser.add_argument("--list-experiments", action="store_true", help="List available experiments")

    # Required for running
    parser.add_argument("--checkpoint", "-c", help="Path to SmolVLA checkpoint")
    parser.add_argument("--task-key", "-k", default="left_yogurt_bin", help="Task key")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory")

    # Options
    parser.add_argument("--dry-run", action="store_true", help="Don't execute robot commands")
    parser.add_argument("--duration", type=float, default=60.0, help="Inference duration")

    args = parser.parse_args()

    print("=" * 60)
    print("SmolVLA Ablation Runner")
    print("=" * 60)

    if args.list_experiments:
        print("\nAvailable Experiments:")
        print("-" * 40)
        for name, config in EXPERIMENTS.items():
            print(f"\n{name}")
            print(f"  Type: {config['type']}")
            print(f"  Description: {config['description']}")
        return

    if not args.checkpoint:
        print("ERROR: --checkpoint required")
        sys.exit(1)

    if not args.output_dir:
        print("ERROR: --output-dir required")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        # Run experiment suite
        suite = run_suite(
            config_path=args.config,
            checkpoint=args.checkpoint,
            task_key=args.task_key,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )

        # Save suite results
        with open(args.output_dir / "suite_results.json", 'w') as f:
            json.dump(asdict(suite), f, indent=2, default=str)

        generate_suite_report(suite, args.output_dir / "suite_report.md")

    elif args.experiment:
        # Run single experiment
        result = run_experiment(
            experiment_name=args.experiment,
            checkpoint=args.checkpoint,
            task_key=args.task_key,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            duration=args.duration,
        )

        # Save result
        with open(args.output_dir / f"{args.experiment}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        print(f"\nResult saved to: {args.output_dir}")

    else:
        print("ERROR: Specify --experiment or --config")
        print("Use --list-experiments to see available experiments")
        sys.exit(1)


if __name__ == "__main__":
    main()
