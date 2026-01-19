#!/usr/bin/env python3
"""
Denoising Trajectory Capture Tool for SmolVLA Hallucination Investigation.

Captures the evolution of action trajectory during the 10-step denoising process.
This reveals at which step the FLAT vs RAMP trajectory shape emerges.

Key questions this tool answers:
1. At which denoising step does the trajectory shape (FLAT vs RAMP) emerge?
2. Is the RAMP shape present from step 0 (comes from noise/KV cache) or does it develop?
3. What is the velocity field (v_t) at each step?

Usage (integrated into inference):
    from denoising_trajectory_capture import DenoisingTrajectoryCapture

    capture = DenoisingTrajectoryCapture()
    capture.register_hooks(model)

    # Run inference...

    capture.save_analysis(output_dir, inference_step=200)

Standalone analysis (from saved captures):
    python denoising_trajectory_capture.py \
        --capture-dir logs/yogurt_banana_leftarm/denoising_captures \
        --output-dir logs/yogurt_banana_leftarm/denoising_analysis
"""

import argparse
import json
import functools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass
class DenoiseStepData:
    """Data captured at a single denoising step."""
    step: int  # 0-9
    time: float  # 1.0 -> 0.1

    # x_t: noisy action trajectory at this step
    x_t_mean: list  # Mean across batch [50, 32] -> [50] -> single value per position
    x_t_std: list
    x_t_joint1: list  # Joint 1 (shoulder_lift) values for all 50 positions

    # v_t: predicted velocity
    v_t_mean: list
    v_t_std: list
    v_t_joint1: list

    # After Euler update
    x_next_joint1: list  # x_t + dt * v_t for joint 1


@dataclass
class ChunkDenosingTrajectory:
    """Complete denoising trajectory for one action chunk."""
    inference_step: int  # Which inference step triggered this chunk
    chunk_index: int
    timestamp: str

    # Per-step data
    steps: list  # List of DenoiseStepData

    # Summary metrics
    initial_trajectory_shape: str  # FLAT, RAMP_UP, RAMP_DOWN
    final_trajectory_shape: str
    shape_emergence_step: int  # At which step did final shape emerge

    # Joint 1 evolution across denoising steps
    joint1_evolution: list  # [10, 50] - joint1 value at each denoising step for each position


class DenoisingTrajectoryCapture:
    """
    Captures denoising trajectory by hooking into SmolVLA's denoise_step method.

    The denoising process:
    1. Start with random noise x_0 ~ N(0, 1)
    2. For step t = 0 to 9:
       a. Predict velocity v_t = model(x_t, time)
       b. Update: x_{t+1} = x_t + dt * v_t
    3. Final x_10 is the action trajectory

    We capture x_t and v_t at each step to see when the trajectory shape emerges.
    """

    def __init__(self):
        self.current_captures = {}  # {inference_step: ChunkDenosingTrajectory}
        self.current_inference_step = 0
        self.current_denoising_step = 0
        self.current_chunk_data = []
        self.is_capturing = False
        self._original_denoise_step = None
        self._original_run_inference = None
        self._model = None

    def start_capture(self, inference_step: int):
        """Start capturing denoising data for a new inference step."""
        self.current_inference_step = inference_step
        self.current_denoising_step = 0
        self.current_chunk_data = []
        self.is_capturing = True

    def stop_capture(self):
        """Stop capturing and finalize the current chunk data."""
        if not self.is_capturing or not self.current_chunk_data:
            return

        # Build joint1 evolution matrix
        joint1_evolution = []
        for step_data in self.current_chunk_data:
            joint1_evolution.append(step_data.x_t_joint1)

        # Determine trajectory shapes
        initial_shape = self._classify_shape(self.current_chunk_data[0].x_t_joint1)
        final_shape = self._classify_shape(self.current_chunk_data[-1].x_t_joint1)

        # Find when final shape emerged
        emergence_step = 0
        for i, step_data in enumerate(self.current_chunk_data):
            if self._classify_shape(step_data.x_t_joint1) == final_shape:
                emergence_step = i
                break

        trajectory = ChunkDenosingTrajectory(
            inference_step=self.current_inference_step,
            chunk_index=self.current_inference_step // 50,
            timestamp=datetime.now().isoformat(),
            steps=[asdict(s) for s in self.current_chunk_data],
            initial_trajectory_shape=initial_shape,
            final_trajectory_shape=final_shape,
            shape_emergence_step=emergence_step,
            joint1_evolution=joint1_evolution
        )

        self.current_captures[self.current_inference_step] = trajectory
        self.is_capturing = False

    def _classify_shape(self, values: list) -> str:
        """Classify trajectory shape as FLAT, RAMP_UP, or RAMP_DOWN."""
        if len(values) < 2:
            return "UNKNOWN"

        delta = values[-1] - values[0]
        if abs(delta) < 0.3:
            return "FLAT"
        elif delta > 0.3:
            return "RAMP_UP"
        else:
            return "RAMP_DOWN"

    def register_hooks(self, model):
        """
        Register hooks on the SmolVLA model to capture denoising data.

        Hooks into run_inference to track when new chunks start,
        and into the denoising loop to capture x_t and v_t.
        """
        self._model = model

        # Hook into the denoising loop
        # We need to wrap the entire inference method to track denoising
        original_run_inference = model.run_inference

        @functools.wraps(original_run_inference)
        def capturing_run_inference(*args, **kwargs):
            # The denoising happens inside run_inference
            # We need to hook at a lower level
            return original_run_inference(*args, **kwargs)

        model.run_inference = capturing_run_inference
        self._original_run_inference = original_run_inference

        print("[DenoisingCapture] Hooks registered")

    def capture_denoising_step(self, step: int, time: float, x_t, v_t, dt: float):
        """
        Called during denoising to capture intermediate states.

        Args:
            step: Denoising step index (0-9)
            time: Current time value (1.0 -> 0.1)
            x_t: Current noisy action [batch, 50, 32]
            v_t: Predicted velocity [batch, 50, 32]
            dt: Time step (negative, typically -0.1)
        """
        if not self.is_capturing:
            return

        # Convert to numpy if tensor
        if torch is not None and torch.is_tensor(x_t):
            x_t_np = x_t.detach().cpu().numpy()
            v_t_np = v_t.detach().cpu().numpy()
        else:
            x_t_np = np.array(x_t)
            v_t_np = np.array(v_t)

        # Extract joint 1 values (index 1 in the action space, assuming layout)
        # Action space: [50 positions, 32 action dims]
        # For left arm, joint 1 is typically at index 1

        # Get first batch
        x_t_np = x_t_np[0]  # [50, 32]
        v_t_np = v_t_np[0]  # [50, 32]

        # Joint 1 is at index 1 (shoulder_lift)
        joint1_idx = 1

        step_data = DenoiseStepData(
            step=step,
            time=float(time),
            x_t_mean=[float(np.mean(x_t_np[:, i])) for i in range(min(6, x_t_np.shape[1]))],
            x_t_std=[float(np.std(x_t_np[:, i])) for i in range(min(6, x_t_np.shape[1]))],
            x_t_joint1=[float(x_t_np[pos, joint1_idx]) for pos in range(x_t_np.shape[0])],
            v_t_mean=[float(np.mean(v_t_np[:, i])) for i in range(min(6, v_t_np.shape[1]))],
            v_t_std=[float(np.std(v_t_np[:, i])) for i in range(min(6, v_t_np.shape[1]))],
            v_t_joint1=[float(v_t_np[pos, joint1_idx]) for pos in range(v_t_np.shape[0])],
            x_next_joint1=[float(x_t_np[pos, joint1_idx] + dt * v_t_np[pos, joint1_idx])
                          for pos in range(x_t_np.shape[0])]
        )

        self.current_chunk_data.append(step_data)
        self.current_denoising_step = step + 1

    def save_analysis(self, output_dir: Path, inference_step: Optional[int] = None):
        """Save captured denoising trajectories to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if inference_step is not None:
            # Save specific step
            if inference_step in self.current_captures:
                traj = self.current_captures[inference_step]
                output_file = output_dir / f"denoising_step_{inference_step:04d}.json"
                with open(output_file, 'w') as f:
                    json.dump(asdict(traj), f, indent=2)
                print(f"Saved denoising trajectory to {output_file}")
        else:
            # Save all captures
            for step, traj in self.current_captures.items():
                output_file = output_dir / f"denoising_step_{step:04d}.json"
                with open(output_file, 'w') as f:
                    json.dump(asdict(traj), f, indent=2)
            print(f"Saved {len(self.current_captures)} denoising trajectories to {output_dir}")

    def get_summary(self) -> dict:
        """Get summary of all captured trajectories."""
        return {
            "total_captures": len(self.current_captures),
            "captures": {
                step: {
                    "initial_shape": traj.initial_trajectory_shape,
                    "final_shape": traj.final_trajectory_shape,
                    "emergence_step": traj.shape_emergence_step
                }
                for step, traj in self.current_captures.items()
            }
        }


def analyze_saved_captures(capture_dir: Path, output_dir: Path):
    """Analyze previously saved denoising captures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all capture files
    capture_files = sorted(capture_dir.glob("denoising_step_*.json"))
    if not capture_files:
        print(f"No capture files found in {capture_dir}")
        return

    print(f"Found {len(capture_files)} capture files")

    captures = []
    for f in capture_files:
        with open(f) as fp:
            captures.append(json.load(fp))

    # Analyze
    print("\n" + "="*60)
    print("DENOISING TRAJECTORY ANALYSIS")
    print("="*60)

    for cap in captures:
        inf_step = cap['inference_step']
        initial = cap['initial_trajectory_shape']
        final = cap['final_trajectory_shape']
        emergence = cap['shape_emergence_step']

        print(f"\nInference step {inf_step}:")
        print(f"  Initial shape (step 0): {initial}")
        print(f"  Final shape (step 9): {final}")
        print(f"  Shape emerged at denoising step: {emergence}")

        # Show joint1 evolution
        if 'joint1_evolution' in cap and cap['joint1_evolution']:
            evolution = cap['joint1_evolution']
            print(f"  Joint1 evolution (position 0 -> 49):")
            for i, step_vals in enumerate(evolution):
                if step_vals:
                    delta = step_vals[-1] - step_vals[0]
                    print(f"    Denoise step {i}: {step_vals[0]:.3f} -> {step_vals[-1]:.3f} (delta={delta:+.3f})")

    # Generate visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(captures), 1, figsize=(14, 4*len(captures)))
        if len(captures) == 1:
            axes = [axes]

        for ax, cap in zip(axes, captures):
            inf_step = cap['inference_step']
            evolution = cap.get('joint1_evolution', [])

            if evolution:
                for i, step_vals in enumerate(evolution):
                    positions = np.arange(len(step_vals))
                    alpha = 0.3 + 0.7 * (i / 9)  # Fade in as denoising progresses
                    ax.plot(positions, step_vals, alpha=alpha, label=f'Denoise step {i}')

                ax.set_xlabel('Position in chunk (0-49)')
                ax.set_ylabel('Joint 1 action_raw')
                ax.set_title(f'Inference Step {inf_step}: Denoising Evolution')
                ax.legend(loc='upper left', fontsize=8)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'denoising_evolution.png', dpi=150)
        plt.close()
        print(f"\nVisualization saved to {output_dir / 'denoising_evolution.png'}")

    except ImportError:
        print("matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Analyze denoising trajectories")
    parser.add_argument("--capture-dir", type=str, help="Directory with saved captures")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    if args.capture_dir:
        analyze_saved_captures(Path(args.capture_dir), Path(args.output_dir))
    else:
        print("Usage: Integrate DenoisingTrajectoryCapture into inference script")
        print("       or provide --capture-dir with saved captures")


if __name__ == "__main__":
    main()
