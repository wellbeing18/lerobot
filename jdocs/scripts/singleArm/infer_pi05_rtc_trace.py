#!/usr/bin/env python3
"""
Pi0.5 Real-Time Chunking (RTC) Inference with Trace Recording for SO-101.

This script extends infer_pi05_rtc.py with comprehensive tracing to investigate
inference performance and RTC behavior.

Trace Data Captured:
- High-resolution timestamps for each component (capture, state, inference, execute)
- Full action buffer on each inference
- Joint states at each step
- RTC-specific metrics (queue depth, inference delay, latency)
- Images saved on inference steps

Output:
    outputs/inference_traces/trace_YYYYMMDD_HHMMSS/
    ├── trace.jsonl           # All trace entries (JSON lines)
    ├── summary.json          # Statistics
    ├── config.json           # Run configuration
    ├── inference.log         # Console log output
    └── images/               # Images on inference steps
        ├── step_0000_head.jpg
        ├── step_0000_left_wrist.jpg
        └── ...

Usage:
    python infer_pi05_rtc_trace.py \\
        --checkpoint outputs/pi05_pickplace/checkpoints/003000/pretrained_model \\
        --task "pick up the block and place it on the plate" \\
        --duration 30

References:
    - jdocs/scripts/infer_pi05_rtc.py (base script)
    - Isaac-GR00T/custom/scripts/ver1_6/infer_groot_so101_trace.py (trace reference)
"""

import argparse
import json
import logging
import math
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
DEFAULT_CHECKPOINT = "outputs/pi05_pickplace/checkpoints/003000/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task description (language prompt for Pi0.5)
DEFAULT_TASK = "pick up the block and place it on the plate"

# RTC Parameters
DEFAULT_FPS = 30  # Control loop frequency
DEFAULT_EXECUTION_HORIZON = 10  # Steps to blend with previous chunk
DEFAULT_MAX_GUIDANCE_WEIGHT = 10.0  # How strongly to enforce consistency
DEFAULT_ACTION_QUEUE_THRESHOLD = 30  # Request new chunk when queue <= this

# Hardware Config - Use Pi0.5-specific config with safety clamping and warmup
HARDWARE_CONFIG = "jdocs/scripts/so101_pi05_hardware.yaml"

# Timing
MAX_DURATION = 60.0  # Maximum run duration in seconds

# Trace defaults
DEFAULT_TRACE_OUTPUT_DIR = "outputs/inference_traces"
# ============================================================================

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Dataset path for loading stats
DATASET_PATH = str(PROJECT_ROOT / "datasets" / "pick_and_place")

# Log directory
LOG_DIR = PROJECT_ROOT / "jdocs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TRACE DATA STRUCTURES
# ============================================================================

@dataclass
class TraceEntry:
    """Single trace entry for one control loop iteration."""
    # Step identification
    step: int
    inference_triggered: bool
    action_idx_in_chunk: int

    # High-resolution timestamps (seconds since trace start)
    t_loop_start: float
    t_capture_start: float = 0.0
    t_capture_end: float = 0.0
    t_state_read: float = 0.0
    t_inference_start: float = 0.0
    t_inference_end: float = 0.0
    t_action_sent: float = 0.0
    t_loop_end: float = 0.0

    # Input data
    joint_states: list = field(default_factory=list)
    task_description: str = ""

    # Inference data (only when triggered)
    action_buffer: list = field(default_factory=list)
    state_after_inference: list = field(default_factory=list)
    state_drift_during_inference: list = field(default_factory=list)

    # Execution data
    action_executed: list = field(default_factory=list)
    action_delta: list = field(default_factory=list)

    # RTC-specific metrics
    inference_delay: int = None
    queue_size_before: int = 0
    queue_size_after: int = 0
    latency_ms: float = None

    # Derived metrics (computed at save time)
    loop_duration_ms: float = 0.0
    inference_duration_ms: float = None
    capture_duration_ms: float = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class InferenceData:
    """Data captured during inference, passed to execute thread."""
    action_buffer: list  # Full action chunk
    t_capture_start: float
    t_capture_end: float
    t_state_read: float
    t_inference_start: float
    t_inference_end: float
    inference_delay: int
    latency_ms: float
    state_before: list
    state_after: list
    images: dict  # Camera frames for saving
    chunk_start_idx: int  # Action index when this chunk starts


class TraceWriter:
    """Thread-safe trace writer."""

    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.lock = Lock()
        self.trace_file = trace_dir / "trace.jsonl"
        self.fp = open(self.trace_file, "w")
        self.entries = []  # Keep in memory for summary

    def write(self, entry: TraceEntry):
        """Write a trace entry to file."""
        with self.lock:
            self.fp.write(json.dumps(entry.to_dict()) + "\n")
            self.fp.flush()
            self.entries.append(entry)

    def close(self):
        """Close the trace file."""
        with self.lock:
            self.fp.close()

    def get_entries(self) -> list[TraceEntry]:
        """Get all entries for summary generation."""
        with self.lock:
            return list(self.entries)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: Path = None) -> logging.Logger:
    """Set up logging to both terminal and file."""
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

    logger = logging.getLogger("pi05_rtc_trace")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


# Global logger (reconfigured in main)
logger = logging.getLogger("pi05_rtc_trace")


def load_hardware_config(config_path: str) -> dict:
    """Load hardware configuration from YAML file."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {full_path}")

    with open(full_path) as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# HARDWARE INTERFACES
# ============================================================================

class ThreadSafeRobot:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, hw_config: dict, dry_run: bool = False):
        self.lock = Lock()
        self.dry_run = dry_run
        self.hw_config = hw_config
        self.robot = None
        self.motor_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper"
        ]

        if not dry_run:
            self._init_robot()

    def _init_robot(self):
        """Initialize robot connection."""
        try:
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
            from lerobot.robots.so101_follower.so101_follower import SO101Follower

            robot_config = self.hw_config.get("robot", {})

            # FIX: Add max_relative_target for safety clamping (GPT recommendation)
            # This limits how far the goal can be from current position per step
            max_relative_target = robot_config.get("max_relative_target", None)

            config = SO101FollowerConfig(
                port=robot_config.get("port", "/dev/ttyACM1"),
                id=robot_config.get("id", "xlerobot_left_arm"),
                max_relative_target=max_relative_target,
                use_degrees=robot_config.get("use_degrees", True),
            )

            self.robot = SO101Follower(config)
            self.robot.connect()
            logger.info(f"Robot connected on {config.port}")
            if max_relative_target is not None:
                logger.info(f"  Safety clamping enabled: max_relative_target={max_relative_target}")

        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            logger.warning("Running in mock mode")
            self.robot = None

    def get_state(self) -> np.ndarray:
        """Get current robot state (thread-safe)."""
        with self.lock:
            if self.robot is None:
                return np.zeros(6, dtype=np.float32)

            obs = self.robot.get_observation()
            state = np.array(
                [obs[f"{name}.pos"] for name in self.motor_names],
                dtype=np.float32
            )
            return state

    def send_action(self, action: np.ndarray):
        """Send action to robot (thread-safe)."""
        with self.lock:
            if self.robot is None or self.dry_run:
                return

            action_dict = {
                f"{name}.pos": float(action[i])
                for i, name in enumerate(self.motor_names)
            }
            self.robot.send_action(action_dict)

    def disconnect(self):
        """Disconnect from robot."""
        with self.lock:
            if self.robot is not None:
                self.robot.disconnect()


class ThreadSafeCameras:
    """Thread-safe camera manager."""

    def __init__(self, hw_config: dict):
        self.lock = Lock()
        self.cameras = {}

        cam_config = hw_config.get("cameras", {})

        # Initialize cameras
        for name, cfg in [("head", cam_config.get("head", {})),
                          ("left_wrist", cam_config.get("left_wrist", {}))]:
            device_index = cfg.get("index_or_path", 4 if name == "head" else 6)
            width = cfg.get("width", 640)
            height = cfg.get("height", 480)

            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FPS, 30)
                self.cameras[name] = cap
                logger.info(f"Camera '{name}' initialized at index {device_index}")
            else:
                logger.warning(f"Failed to open camera '{name}' at index {device_index}")

    def capture(self) -> dict:
        """Capture frames from all cameras (thread-safe)."""
        with self.lock:
            frames = {}
            for name, cap in self.cameras.items():
                ret, frame = cap.read()
                if ret:
                    frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # Return black frame on failure
                    frames[name] = np.zeros((480, 640, 3), dtype=np.uint8)
            return frames

    def release(self):
        """Release all cameras."""
        with self.lock:
            for cap in self.cameras.values():
                cap.release()


# ============================================================================
# LATENCY TRACKER AND ACTION QUEUE
# ============================================================================

class LatencyTracker:
    """Track inference latency for adaptive delay calculation."""

    def __init__(self, window_size: int = 10):
        self.latencies = []
        self.window_size = window_size

    def add(self, latency: float):
        """Add a latency measurement."""
        self.latencies.append(latency)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)

    def max(self) -> float:
        """Get maximum latency in window."""
        return max(self.latencies) if self.latencies else 0.5

    def avg(self) -> float:
        """Get average latency in window."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.5


class ActionQueue:
    """Thread-safe action queue with RTC support and trace tracking.

    FIXED BUGS (from Gemini analysis):
    1. Step counter now tracks time/loops, not just successful action retrievals
    2. New chunks REPLACE outdated future actions instead of appending
    """

    def __init__(self, execution_horizon: int = 10):
        self.lock = Lock()
        self.queue = []  # List of postprocessed actions
        self.action_index = 0  # Actions successfully retrieved
        self.step_counter = 0  # Global step counter (increments every loop)
        self.chunk_action_index = 0  # Index within current chunk
        self.execution_horizon = execution_horizon
        self.prev_chunk_original = None  # For RTC blending
        self.current_chunk_start = 0  # Step when current chunk started
        self.inference_in_flight_step = -1  # Step when inference started

    def qsize(self) -> int:
        """Get current queue size."""
        with self.lock:
            return len(self.queue)

    def get(self) -> tuple[Tensor | None, int]:
        """Get next action from queue. Returns (action, chunk_index)."""
        with self.lock:
            if not self.queue:
                return None, 0
            action = self.queue.pop(0)
            chunk_idx = self.chunk_action_index
            self.action_index += 1
            self.chunk_action_index += 1
            return action, chunk_idx

    def increment_step(self):
        """Increment step counter. Called every loop iteration."""
        with self.lock:
            self.step_counter += 1

    def get_step_counter(self) -> int:
        """Get current step counter."""
        with self.lock:
            return self.step_counter

    def get_action_index(self) -> int:
        """Get current action index."""
        with self.lock:
            return self.action_index

    def mark_inference_start(self):
        """Mark the step when inference started."""
        with self.lock:
            self.inference_in_flight_step = self.step_counter

    def get_left_over(self) -> Tensor | None:
        """Get remaining actions for RTC blending."""
        with self.lock:
            if self.prev_chunk_original is None:
                return None
            return self.prev_chunk_original

    def merge(self, original_actions: Tensor, postprocessed_actions: Tensor,
              inference_delay: int, step_when_started: int) -> int:
        """Merge new action chunk with queue. Returns chunk start index.

        FIXED: Now uses step_counter to calculate how many steps passed during
        inference, instead of action_index which only counts successful retrievals.
        This prevents the "jump bug" where early empty queue steps aren't counted.

        FIXED: Now REPLACES queue instead of appending, so the robot always
        executes the most recent prediction, not stale actions.
        """
        with self.lock:
            # FIX: Calculate elapsed steps based on step_counter, not action_index
            # This correctly counts steps even when queue was empty
            steps_elapsed = self.step_counter - step_when_started

            # Skip actions that are already in the past
            # inference_delay accounts for model latency
            skip = max(0, steps_elapsed)

            if skip < len(postprocessed_actions):
                new_actions = postprocessed_actions[skip:]

                # FIX: REPLACE queue instead of extending
                # This ensures we always use the freshest prediction
                self.queue = list(new_actions.unbind(0))
            else:
                # All actions in this chunk are already stale
                # This shouldn't happen in normal operation
                self.queue = []

            # Store original actions for next RTC iteration
            self.prev_chunk_original = original_actions

            # Reset chunk index and record start
            self.chunk_action_index = 0
            chunk_start = self.step_counter
            self.current_chunk_start = chunk_start

            return chunk_start


# ============================================================================
# OBSERVATION FORMATTING
# ============================================================================

def format_observation(
    images: dict,
    state: np.ndarray,
    task: str,
    device: str = "cuda",
) -> dict:
    """Format observation for Pi0.5 policy input."""
    observation = {}

    # State
    observation["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Images - use original camera names (head, left_wrist) as trained
    for name, frame in images.items():
        img_tensor = torch.from_numpy(frame).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        observation[f"observation.images.{name}"] = img_tensor.to(device)

    observation["task"] = [task]  # Must be a list for Pi0.5

    return observation


# ============================================================================
# INFERENCE THREAD
# ============================================================================

def get_actions_thread(
    policy,
    preprocessor,
    postprocessor,
    cameras: ThreadSafeCameras,
    robot: ThreadSafeRobot,
    action_queue: ActionQueue,
    inference_data_queue: Queue,
    shutdown_event: Event,
    task: str,
    fps: float,
    action_queue_threshold: int,
    device: str,
    trace_start: float,
    images_dir: Path,
    save_images: bool,
):
    """Background thread for action prediction with RTC and tracing."""
    try:
        logger.info("[GET_ACTIONS] Starting action prediction thread with tracing")

        latency_tracker = LatencyTracker()
        time_per_step = 1.0 / fps
        inference_count = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= action_queue_threshold:
                start_time = time.perf_counter()
                # FIX: Use step_counter instead of action_index for timing
                step_when_started = action_queue.get_step_counter()
                action_queue.mark_inference_start()
                prev_actions = action_queue.get_left_over()

                # Calculate inference delay from latency
                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_step)

                # Capture images with timestamps
                t_capture_start = time.perf_counter() - trace_start
                images = cameras.capture()
                t_capture_end = time.perf_counter() - trace_start

                # Get robot state
                state_before = robot.get_state()
                t_state_read = time.perf_counter() - trace_start

                # Format observation
                obs = format_observation(images, state_before, task, device)

                # Preprocess
                obs = preprocessor(obs)

                # Generate actions with RTC
                t_inference_start = time.perf_counter() - trace_start
                actions = policy.predict_action_chunk(
                    obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )
                t_inference_end = time.perf_counter() - trace_start

                # Get state after inference
                state_after = robot.get_state()

                # Store original for RTC
                original_actions = actions.squeeze(0).clone()

                # Postprocess
                postprocessed = postprocessor(actions).squeeze(0)

                # Track latency
                new_latency = time.perf_counter() - start_time
                latency_tracker.add(new_latency)
                new_delay = math.ceil(new_latency / time_per_step)

                # Merge into queue and get chunk start index
                # FIX: Pass step_when_started instead of action_index_before
                chunk_start_idx = action_queue.merge(
                    original_actions, postprocessed,
                    new_delay, step_when_started
                )

                # Save images on inference steps
                if save_images:
                    for name, frame in images.items():
                        img_path = images_dir / f"step_{step_when_started:04d}_{name}.jpg"
                        cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Create inference data for execute thread
                inf_data = InferenceData(
                    action_buffer=postprocessed.cpu().numpy().tolist(),
                    t_capture_start=t_capture_start,
                    t_capture_end=t_capture_end,
                    t_state_read=t_state_read,
                    t_inference_start=t_inference_start,
                    t_inference_end=t_inference_end,
                    inference_delay=new_delay,
                    latency_ms=new_latency * 1000,
                    state_before=state_before.tolist(),
                    state_after=state_after.tolist(),
                    images=images if save_images else {},
                    chunk_start_idx=chunk_start_idx,
                )

                # Push to inference data queue
                inference_data_queue.put(inf_data)

                inference_count += 1
                logger.debug(
                    f"[GET_ACTIONS] Inference #{inference_count}: "
                    f"latency={new_latency:.3f}s, delay={new_delay}, "
                    f"queue_size={action_queue.qsize()}"
                )

                # Log progress periodically
                if inference_count % 10 == 0:
                    logger.info(
                        f"[GET_ACTIONS] Inference #{inference_count}: "
                        f"latency={new_latency*1000:.1f}ms, queue={action_queue.qsize()}"
                    )
            else:
                time.sleep(0.05)

        logger.info(f"[GET_ACTIONS] Thread shutting down. Total inferences: {inference_count}")

    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


# ============================================================================
# EXECUTE THREAD
# ============================================================================

def execute_actions_thread(
    robot: ThreadSafeRobot,
    action_queue: ActionQueue,
    inference_data_queue: Queue,
    trace_writer: TraceWriter,
    shutdown_event: Event,
    fps: float,
    stats: dict,
    task: str,
    trace_start: float,
):
    """Background thread for action execution with trace recording."""
    try:
        logger.info("[EXECUTE] Starting action execution thread with tracing")

        action_interval = 1.0 / fps
        action_count = 0
        empty_count = 0
        step_count = 0

        # Current inference data (updated when new chunk arrives)
        current_inf_data: InferenceData | None = None
        chunk_start_action_idx = -1

        while not shutdown_event.is_set():
            t_loop_start = time.perf_counter() - trace_start

            # FIX: Increment step counter EVERY loop iteration
            # This ensures accurate timing even when queue is empty
            action_queue.increment_step()

            # Get queue size before action
            queue_size_before = action_queue.qsize()

            # Get action and chunk index
            action, chunk_idx = action_queue.get()

            # Get queue size after action
            queue_size_after = action_queue.qsize()

            # Check if we have new inference data
            inference_triggered = False
            if not inference_data_queue.empty():
                current_inf_data = inference_data_queue.get_nowait()
                chunk_start_action_idx = current_inf_data.chunk_start_idx
                inference_triggered = True

            if action is not None:
                action_np = action.cpu().numpy()

                # Get current state for trace
                state = robot.get_state()
                t_state_read = time.perf_counter() - trace_start

                # Send action
                robot.send_action(action_np)
                t_action_sent = time.perf_counter() - trace_start

                # Compute delta
                delta = action_np - state

                t_loop_end = time.perf_counter() - trace_start

                # Build trace entry
                trace_entry = TraceEntry(
                    step=step_count,
                    inference_triggered=inference_triggered,
                    action_idx_in_chunk=chunk_idx,
                    t_loop_start=t_loop_start,
                    t_state_read=t_state_read,
                    t_action_sent=t_action_sent,
                    t_loop_end=t_loop_end,
                    joint_states=state.tolist(),
                    task_description=task,
                    action_executed=action_np.tolist(),
                    action_delta=delta.tolist(),
                    queue_size_before=queue_size_before,
                    queue_size_after=queue_size_after,
                    loop_duration_ms=(t_loop_end - t_loop_start) * 1000,
                )

                # Add inference data if this is an inference step
                if inference_triggered and current_inf_data is not None:
                    trace_entry.t_capture_start = current_inf_data.t_capture_start
                    trace_entry.t_capture_end = current_inf_data.t_capture_end
                    trace_entry.t_inference_start = current_inf_data.t_inference_start
                    trace_entry.t_inference_end = current_inf_data.t_inference_end
                    trace_entry.action_buffer = current_inf_data.action_buffer
                    trace_entry.state_after_inference = current_inf_data.state_after
                    trace_entry.state_drift_during_inference = [
                        a - b for a, b in zip(
                            current_inf_data.state_after,
                            current_inf_data.state_before
                        )
                    ]
                    trace_entry.inference_delay = current_inf_data.inference_delay
                    trace_entry.latency_ms = current_inf_data.latency_ms
                    trace_entry.inference_duration_ms = (
                        current_inf_data.t_inference_end - current_inf_data.t_inference_start
                    ) * 1000
                    trace_entry.capture_duration_ms = (
                        current_inf_data.t_capture_end - current_inf_data.t_capture_start
                    ) * 1000

                # Write trace entry
                trace_writer.write(trace_entry)

                action_count += 1
                step_count += 1
                empty_count = 0
            else:
                empty_count += 1
                if empty_count % 30 == 0:
                    logger.warning(f"[EXECUTE] Action queue empty for {empty_count} steps")

                # Still record trace for empty steps
                t_loop_end = time.perf_counter() - trace_start
                state = robot.get_state()

                trace_entry = TraceEntry(
                    step=step_count,
                    inference_triggered=False,
                    action_idx_in_chunk=-1,  # No action
                    t_loop_start=t_loop_start,
                    t_state_read=t_loop_end,
                    t_loop_end=t_loop_end,
                    joint_states=state.tolist(),
                    task_description=task,
                    queue_size_before=queue_size_before,
                    queue_size_after=queue_size_after,
                    loop_duration_ms=(t_loop_end - t_loop_start) * 1000,
                )
                trace_writer.write(trace_entry)
                step_count += 1

            # Maintain timing
            elapsed = time.perf_counter() - (t_loop_start + trace_start)
            sleep_time = action_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        stats["action_count"] = action_count
        stats["step_count"] = step_count
        logger.info(f"[EXECUTE] Thread shutting down. Total actions: {action_count}, steps: {step_count}")

    except Exception as e:
        logger.error(f"[EXECUTE] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_summary(entries: list[TraceEntry], config: dict) -> dict:
    """Generate summary statistics from trace entries."""
    if not entries:
        return {"error": "No trace entries"}

    # Collect metrics
    loop_durations = [e.loop_duration_ms for e in entries]
    inference_durations = [e.inference_duration_ms for e in entries if e.inference_duration_ms is not None]
    capture_durations = [e.capture_duration_ms for e in entries if e.capture_duration_ms is not None]
    latencies = [e.latency_ms for e in entries if e.latency_ms is not None]
    queue_sizes = [e.queue_size_before for e in entries]
    inference_delays = [e.inference_delay for e in entries if e.inference_delay is not None]

    num_inferences = sum(1 for e in entries if e.inference_triggered)
    total_time = entries[-1].t_loop_end if entries else 0

    def stats(values):
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    summary = {
        "total_steps": len(entries),
        "total_time_s": total_time,
        "effective_rate_hz": len(entries) / total_time if total_time > 0 else 0,
        "num_inferences": num_inferences,
        "timing": {
            "loop_ms": stats(loop_durations),
            "inference_ms": stats(inference_durations),
            "capture_ms": stats(capture_durations),
            "latency_ms": stats(latencies),
        },
        "rtc": {
            "inference_delay_steps": stats(inference_delays),
            "queue_size": stats(queue_sizes),
        },
        "warnings": {
            "slow_loops_gt_50ms": sum(1 for d in loop_durations if d > 50),
            "slow_inferences_gt_150ms": sum(1 for d in inference_durations if d > 150),
            "queue_empty_steps": sum(1 for e in entries if e.action_idx_in_chunk < 0),
        },
        "config": config,
    }

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pi0.5 RTC inference with tracing for SO-101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model
    parser.add_argument("--checkpoint", "-c", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint")
    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK,
                        help="Task description for language conditioning")

    # RTC parameters
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON,
                        help="RTC execution horizon (steps to blend)")
    parser.add_argument("--max-guidance-weight", type=float, default=DEFAULT_MAX_GUIDANCE_WEIGHT,
                        help="RTC guidance weight for consistency")
    parser.add_argument("--action-queue-threshold", type=int, default=DEFAULT_ACTION_QUEUE_THRESHOLD,
                        help="Request new chunk when queue size <= this")

    # Timing
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="Control loop frequency (Hz)")
    parser.add_argument("--duration", type=float, default=MAX_DURATION,
                        help="Max run duration (seconds)")

    # Hardware
    parser.add_argument("--hw-config", type=str, default=HARDWARE_CONFIG,
                        help="Hardware config YAML path")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Inference device (cuda/cpu)")

    # Trace options
    parser.add_argument("--trace-output-dir", type=str, default=DEFAULT_TRACE_OUTPUT_DIR,
                        help="Base output directory for traces")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Disable image saving (reduces disk I/O)")

    # Options
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without sending robot commands")
    parser.add_argument("--no-rtc", action="store_true",
                        help="Disable RTC (use standard chunking)")

    args = parser.parse_args()

    # Create trace directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = Path(args.trace_output_dir) / f"trace_pi05_{timestamp}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    images_dir = trace_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = trace_dir / "inference.log"
    global logger
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Pi0.5 RTC Inference with Tracing for SO-101")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:         {args.checkpoint}")
    logger.info(f"Task:               {args.task}")
    logger.info(f"RTC Enabled:        {not args.no_rtc}")
    logger.info(f"Execution Horizon:  {args.execution_horizon}")
    logger.info(f"FPS:                {args.fps}")
    logger.info(f"Duration:           {args.duration}s")
    logger.info(f"Device:             {args.device}")
    logger.info(f"Dry Run:            {args.dry_run}")
    logger.info(f"Trace Dir:          {trace_dir}")
    logger.info(f"Save Images:        {not args.no_save_images}")
    logger.info("=" * 70)

    # Save config
    config = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "rtc_enabled": not args.no_rtc,
        "execution_horizon": args.execution_horizon,
        "max_guidance_weight": args.max_guidance_weight,
        "action_queue_threshold": args.action_queue_threshold,
        "fps": args.fps,
        "duration": args.duration,
        "hw_config": args.hw_config,
        "device": args.device,
        "dry_run": args.dry_run,
        "save_images": not args.no_save_images,
        "timestamp": timestamp,
        "model_type": "pi05",
    }
    with open(trace_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and not args.checkpoint.startswith("lerobot/"):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    shutdown_event = Event()

    def signal_handler(sig, frame):
        logger.info("\nShutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    robot = None
    cameras = None
    trace_writer = None
    stats = {"action_count": 0, "step_count": 0}

    try:
        # Load hardware config
        hw_config = load_hardware_config(args.hw_config)

        # Override defaults from config file if not specified on command line
        inference_config = hw_config.get("inference", {})
        rtc_config_yaml = hw_config.get("rtc", {})

        # Use FPS from config if available (ensures match with training data)
        if args.fps == DEFAULT_FPS and "fps" in inference_config:
            args.fps = inference_config["fps"]
            logger.info(f"Using FPS from config: {args.fps}")

        # Use RTC parameters from config if available
        if args.execution_horizon == DEFAULT_EXECUTION_HORIZON and "execution_horizon" in rtc_config_yaml:
            args.execution_horizon = rtc_config_yaml["execution_horizon"]
        if args.max_guidance_weight == DEFAULT_MAX_GUIDANCE_WEIGHT and "max_guidance_weight" in rtc_config_yaml:
            args.max_guidance_weight = rtc_config_yaml["max_guidance_weight"]
        if args.action_queue_threshold == DEFAULT_ACTION_QUEUE_THRESHOLD and "action_queue_threshold" in rtc_config_yaml:
            args.action_queue_threshold = rtc_config_yaml["action_queue_threshold"]

        # Load policy with RTC
        logger.info("\nLoading Pi0.5 policy with RTC...")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        # Load policy
        policy = PI05Policy.from_pretrained(str(checkpoint_path))

        # Configure RTC
        if not args.no_rtc:
            rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=args.execution_horizon,
                max_guidance_weight=args.max_guidance_weight,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            policy.config.rtc_config = rtc_config
            policy.init_rtc_processor()
            logger.info(f"  RTC enabled: horizon={args.execution_horizon}, weight={args.max_guidance_weight}")
        else:
            logger.info("  RTC disabled (standard chunking)")

        policy.eval()
        policy.to(args.device)
        logger.info(f"  Policy loaded on {args.device}")

        # Load dataset metadata for stats
        dataset_metadata = LeRobotDatasetMetadata(
            repo_id="pick_and_place",
            root=DATASET_PATH,
        )

        # Create preprocessor/postprocessor
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint_path),
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        logger.info("  Preprocessor/postprocessor created")

        # Initialize hardware
        logger.info("\nInitializing hardware...")
        cameras = ThreadSafeCameras(hw_config)
        robot = ThreadSafeRobot(hw_config, dry_run=args.dry_run)

        # FIX: Hardware pre-warming (Gemini recommendation)
        # Run dummy cycles to prime camera buffers and GPU caches
        warmup_config = hw_config.get("warmup", {})
        warmup_enabled = warmup_config.get("enabled", True)
        warmup_cycles = warmup_config.get("cycles", 2)

        if warmup_enabled and warmup_cycles > 0:
            logger.info(f"\nPre-warming hardware ({warmup_cycles} cycles)...")
            for i in range(warmup_cycles):
                warmup_start = time.perf_counter()

                # Capture images (primes camera buffers)
                images = cameras.capture()

                # Get robot state
                state = robot.get_state()

                # Format observation
                obs = format_observation(images, state, args.task, args.device)
                obs = preprocessor(obs)

                # Run inference (primes GPU caches)
                with torch.no_grad():
                    _ = policy.predict_action_chunk(obs, inference_delay=0)

                warmup_time = (time.perf_counter() - warmup_start) * 1000
                logger.info(f"  Warmup cycle {i+1}/{warmup_cycles}: {warmup_time:.1f}ms")

            logger.info("  Hardware pre-warming complete")

        if args.dry_run:
            logger.info("  DRY RUN mode - no robot commands")

        # Create trace writer
        trace_writer = TraceWriter(trace_dir)

        # Create action queue and inference data queue
        action_queue = ActionQueue(execution_horizon=args.execution_horizon)
        inference_data_queue = Queue()

        # Record trace start time
        trace_start = time.perf_counter()

        # Start threads
        logger.info("\nStarting RTC inference threads with tracing...")

        get_thread = Thread(
            target=get_actions_thread,
            args=(policy, preprocessor, postprocessor, cameras, robot,
                  action_queue, inference_data_queue, shutdown_event,
                  args.task, args.fps, args.action_queue_threshold, args.device,
                  trace_start, images_dir, not args.no_save_images),
            daemon=True,
            name="GetActions"
        )
        get_thread.start()

        exec_thread = Thread(
            target=execute_actions_thread,
            args=(robot, action_queue, inference_data_queue, trace_writer,
                  shutdown_event, args.fps, stats, args.task, trace_start),
            daemon=True,
            name="Execute"
        )
        exec_thread.start()

        # Main loop - monitor and log
        logger.info(f"\nRunning for {args.duration} seconds...")
        logger.info("Press Ctrl+C to stop\n")

        start_time = time.time()
        last_log = start_time

        while not shutdown_event.is_set() and (time.time() - start_time) < args.duration:
            time.sleep(1.0)

            # Periodic status log
            now = time.time()
            if now - last_log >= 5.0:
                elapsed = now - start_time
                queue_size = action_queue.qsize()
                logger.info(
                    f"[STATUS] elapsed={elapsed:.1f}s, queue_size={queue_size}, "
                    f"actions_executed={stats.get('action_count', 0)}"
                )
                last_log = now

        # Shutdown
        logger.info("\nShutting down...")
        shutdown_event.set()

        get_thread.join(timeout=3.0)
        exec_thread.join(timeout=3.0)

        # Close trace writer
        if trace_writer:
            trace_writer.close()

        # Generate summary
        entries = trace_writer.get_entries() if trace_writer else []
        summary = generate_summary(entries, config)

        # Save summary
        with open(trace_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("TRACE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total steps:        {stats.get('step_count', 0)}")
        logger.info(f"Total time:         {total_time:.1f}s")
        logger.info(f"Actions executed:   {stats.get('action_count', 0)}")
        if stats.get("action_count", 0) > 0:
            logger.info(f"Effective rate:     {stats['action_count']/total_time:.1f} Hz")
        logger.info(f"Inferences:         {summary.get('num_inferences', 0)}")

        if summary.get("timing", {}).get("loop_ms"):
            loop_ms = summary["timing"]["loop_ms"]
            logger.info(f"\nLoop timing (ms):")
            logger.info(f"  Mean: {loop_ms['mean']:.1f} +/- {loop_ms['std']:.1f}")
            logger.info(f"  P95:  {loop_ms['p95']:.1f}")
            logger.info(f"  Max:  {loop_ms['max']:.1f}")

        if summary.get("timing", {}).get("inference_ms"):
            inf_ms = summary["timing"]["inference_ms"]
            logger.info(f"\nInference timing (ms):")
            logger.info(f"  Mean: {inf_ms['mean']:.1f} +/- {inf_ms['std']:.1f}")
            logger.info(f"  Max:  {inf_ms['max']:.1f}")

        if summary.get("warnings"):
            warnings = summary["warnings"]
            logger.info(f"\nWarnings:")
            logger.info(f"  Slow loops (>50ms):       {warnings.get('slow_loops_gt_50ms', 0)}")
            logger.info(f"  Slow inferences (>150ms): {warnings.get('slow_inferences_gt_150ms', 0)}")
            logger.info(f"  Queue empty steps:        {warnings.get('queue_empty_steps', 0)}")

        logger.info(f"\nTrace saved to: {trace_dir}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\nError: {e}")
        traceback.print_exc()
        shutdown_event.set()
        sys.exit(1)

    finally:
        if trace_writer:
            trace_writer.close()
        if cameras:
            cameras.release()
        if robot:
            robot.disconnect()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
