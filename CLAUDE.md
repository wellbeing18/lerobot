# LeRobot Project

Hugging Face's hardware-agnostic PyTorch framework for robotics ML - data collection, policy training, and robot control.

## Project Structure

```
src/lerobot/
├── policies/      # ML policies (ACT, Diffusion, Pi0.5, GROOT, SmolVLA, XVLA, VQ-BeT, TDMPC)
├── robots/        # Robot interfaces (SO100/SO101, LeKiwi, Reachy2, Unitree G1, etc.)
├── datasets/      # LeRobotDataset format (Parquet + MP4)
├── cameras/       # Camera abstraction (OpenCV, RealSense)
├── motors/        # Motor control (Dynamixel, Feetech)
├── teleoperators/ # Human control interfaces
├── scripts/       # CLI entry points
├── configs/       # Hydra configuration
├── envs/          # Simulation environments (ALOHA, PushT, LIBERO)
└── rl/            # Reinforcement learning (SAC, HIL-SERL)
```

## Common Commands

```bash
lerobot-train              # Train policy on datasets
lerobot-eval               # Evaluate policies
lerobot-record             # Record demonstrations
lerobot-teleoperate        # Real-time teleoperation
lerobot-calibrate          # Robot calibration
lerobot-find-cameras       # Discover cameras
lerobot-find-port          # Detect motor ports
```

See `pyproject.toml` for all CLI scripts, `Makefile` for test commands.

## Key Concepts

- **LeRobotDataset**: Standard format for robot data (see `src/lerobot/datasets/`)
- **Policy**: Neural network that maps observations → actions (see `src/lerobot/policies/`)
- **Robot**: Hardware abstraction with `connect()`, `teleop_step()`, `send_action()` (see `src/lerobot/robots/`)

## Rules to Follow

### Git Commits
- Never include "Claude Code" or AI attribution in commit messages

### Coding Rules

#### Fail Fast - Never Silently Default Required Values
- **Always throw errors** when key logic is not satisfied; never default to arbitrary values
- **Never use `.get(key, None)` or `.get(key, default)` for required config values** - this hides bugs
- For required values, use explicit validation with clear error messages:

```python
# BAD - Silent failure, bug hidden until much later
self.left_arm_id = config.get("left_arm_id", None)

# GOOD - Fail fast with clear error message
self.left_arm_id = config.get("left_arm_id")
if not self.left_arm_id:
    raise ValueError(
        f"Missing required 'left_arm_id' in config. Got: {config}"
    )
```

- When loading config/YAML, validate structure matches expectations immediately
- Include the actual config values in error messages to help debugging
- Prefer early exceptions over irresponsible defaults which could silently cause some mysteriously unexpected behavior

### Investigation & Research Methodology

When debugging or investigating issues:

1. **Facts & Experiments Based** - Validate hypotheses with concrete data
   - Create experiments/scripts that generate logs specific to assumptions
   - Build probes to expose internal mechanisms (black box → white box)

2. **Process Visualization**
   - Generate process diagrams (mermaid) mirroring each inference step
   - Collect and visualize inputs/outputs/logs at each step

3. **Systematic Probing**
   - Create diagnostic tools for specific hypotheses
   - Log intermediate values at key pipeline stages
   - Compare expected vs actual behavior with data
