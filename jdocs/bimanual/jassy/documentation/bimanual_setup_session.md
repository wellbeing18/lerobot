# Bimanual SO-101 Setup Session Documentation

**Date:** 2025-12-30
**Purpose:** Transition from single-arm SO-101 to bimanual (dual-arm) operation

---

## Table of Contents

- [Overview](#overview)
- [Key Accomplishments](#key-accomplishments)
- [Issues Encountered and Solutions](#issues-encountered-and-solutions)
- [Hardware Configuration](#hardware-configuration)
- [Folder Structure](#folder-structure)
- [Files Created](#files-created)
- [Files Modified in LeRobot Source](#files-modified-in-lerobot-source)
- [Calibration Files](#calibration-files)
- [Usage Guide](#usage-guide)

---

## Overview

This session transitioned from single-arm SO-ARM101 teleoperation to bimanual (dual-arm) operation. The work involved creating new robot/teleoperator classes, updating the data collection script, recalibrating all four arms, and fixing camera configuration issues.

---

## Key Accomplishments

### 1. Created Bimanual SO-101 Robot Classes

LeRobot does not yet have native bimanual support for SO-ARM101. We adapted the existing `bi_so100_follower` and `bi_so100_leader` classes to create `bi_so101_follower` and `bi_so101_leader` for SO-101 hardware.

**Why adaptation was needed:** LeRobot has single-arm `so101_follower`/`so101_leader` and bimanual `bi_so100_follower`/`bi_so100_leader`, but no `bi_so101` classes.

**Key difference from SO-100:** SO-101 calibration does NOT treat `wrist_roll` as a full-turn motor (unlike SO-100). The bi_so101 classes wrap two SO101 instances instead of SO100 instances.

### 2. Updated Data Collection Script

Modified `collect_bimanuel_xlerobot_data.py` to support:
- Bimanual task presets (handover, bimanual_pick, bimanual_place)
- 12-dimensional action space (6 left + 6 right)
- 3 cameras (head, left_wrist, right_wrist)
- Task/arm compatibility validation

### 3. Recalibrated All Four Arms

All arms were recalibrated with grippers positioned "halfway open" at middle of range.

### 4. Fixed Camera Configuration

Corrected camera index mapping:
- Head: `/dev/video4`
- Right wrist: `/dev/video6`
- Left wrist: `/dev/video8`

### 5. Added Explicit Arm ID Support

Added `left_arm_id` and `right_arm_id` parameters to bi_so101 configs to allow reusing existing calibration files instead of creating new ones.

---

## Issues Encountered and Solutions

### Issue 1: Calibration ID Mismatch

**Problem:** The bimanual wrapper was creating arm IDs like `xlerobot_bimanual_left` but existing calibration files used `xlerobot_left_arm`.

**Solution:** Added `left_arm_id` and `right_arm_id` config parameters to explicitly specify calibration IDs:
```python
"robot": {
    "type": "bi_so101_follower",
    "left_arm_id": "xlerobot_left_arm",
    "right_arm_id": "xlerobot_right_arm",
    ...
}
```

### Issue 2: Camera Indices Wrong

**Problem:** Initial config had left_wrist=6 and right_wrist=8, but they were swapped.

**Solution:** Corrected to left_wrist=8, right_wrist=6.

### Issue 3: Right Wrist Camera Blurry

**Problem:** Right wrist camera producing extremely blurry images (bitrate was 0.4 Mbps vs 2.4 Mbps for left wrist).

**Diagnosis:** The low bitrate indicated a focus issue (blurry images compress better). This was NOT oil/smudge but a focus ring out of adjustment.

**Solution:** Manually adjusted the focus ring on the camera lens.

### Issue 4: bi_so101 Classes Not Registered

**Problem:** Running the script failed because `bi_so101_follower` and `bi_so101_leader` config types weren't recognized.

**Solution:** Added imports to `lerobot_record.py` to trigger the `@RobotConfig.register_subclass()` decorators.

---

## Hardware Configuration

### Port Mapping

| Device | Port |
|--------|------|
| Left Leader | `/dev/ttyACM0` |
| Right Follower | `/dev/ttyACM1` |
| Left Follower | `/dev/ttyACM2` |
| Right Leader | `/dev/ttyACM3` |

### Camera Mapping

| Camera | Video Device | Index |
|--------|--------------|-------|
| Head | `/dev/video4` | 4 |
| Right Wrist | `/dev/video6` | 6 |
| Left Wrist | `/dev/video8` | 8 |

### Calibration IDs

| Arm | Calibration ID |
|-----|----------------|
| Left Follower | `xlerobot_left_arm` |
| Right Follower | `xlerobot_right_arm` |
| Left Leader | `xlerobot_left_leader` |
| Right Leader | `xlerobot_right_leader` |

---

## Folder Structure

```
/home/jrobot/project/lerobot/jdocs/bimanual/jassy/
├── backup_calibration_files/       # Backup of calibration JSONs
│   ├── xlerobot_left_arm.json
│   ├── xlerobot_left_leader.json
│   ├── xlerobot_right_arm.json
│   └── xlerobot_right_leader.json
├── configs/
│   └── bimanual.yaml              # General bimanual config template
├── documentation/
│   └── bimanual_setup_session.md  # This file
└── scripts/
    └── collect_bimanuel_xlerobot_data.py  # Data collection script

/home/jrobot/project/lerobot/datasets_bimanuel/  # Dataset storage
├── bimanual/                      # Bimanual task datasets
│   └── bimanual_pick/             # Example recorded dataset
├── left/                          # Left-arm only datasets
└── right/                         # Right-arm only datasets
```

---

## Files Created

### 1. BiSO101Follower Robot Class
**Location:** `src/lerobot/robots/bi_so101_follower/`

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `config_bi_so101_follower.py` | Config dataclass with `left_arm_id`, `right_arm_id` |
| `bi_so101_follower.py` | Robot implementation wrapping two SO101Follower instances |

### 2. BiSO101Leader Teleoperator Class
**Location:** `src/lerobot/teleoperators/bi_so101_leader/`

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `config_bi_so101_leader.py` | Config dataclass with `left_arm_id`, `right_arm_id` |
| `bi_so101_leader.py` | Teleoperator implementation wrapping two SO101Leader instances |

### 3. Data Collection Script
**Location:** `jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py`

Features:
- Interactive task selection
- Bimanual task presets (handover, bimanual_pick, bimanual_place)
- Hardware validation for bimanual mode
- Automatic parameter calculation
- Resume support for adding episodes

### 4. Bimanual Config Template
**Location:** `jdocs/bimanual/jassy/configs/bimanual.yaml`

General-purpose config for bimanual tasks with correct port/camera mappings.

---

## Files Modified in LeRobot Source

### Why These Edits Were Needed

When you run `--robot.type=bi_so101_follower`, LeRobot needs to:
1. **Know the type exists** (registration)
2. **Know how to create it** (factory function)

Each robot class uses a decorator to register itself:
```python
@RobotConfig.register_subclass("bi_so101_follower")  # "Register me as bi_so101_follower"
class BiSO101FollowerConfig:
    ...
```

**Key insight:** Decorators only run when Python **imports** the file. If nobody imports the file, LeRobot never learns about the robot type.

**Example:**
- File `bi_so101_follower.py` exists with the decorator
- But if nothing imports it → decorator never runs → `--robot.type=bi_so101_follower` fails with "unknown type"
- Once something imports it → decorator runs → LeRobot knows the type

### 1. `src/lerobot/robots/utils.py`
Added bi_so101_follower to the **factory function** `make_robot_from_config()`. This tells LeRobot how to actually create an instance:
```python
elif config.type == "bi_so101_follower":
    from .bi_so101_follower import BiSO101Follower
    return BiSO101Follower(config)
```

### 2. `src/lerobot/teleoperators/utils.py`
Added bi_so101_leader to the **factory function** `make_teleoperator_from_config()`:
```python
elif config.type == "bi_so101_leader":
    from .bi_so101_leader import BiSO101Leader
    return BiSO101Leader(config)
```

### 3. `src/lerobot/scripts/lerobot_record.py`
Added imports to **trigger the registration decorators**. Without these imports, the decorators never run and LeRobot doesn't know these types exist:
```python
from lerobot.robots import (
    bi_so101_follower,  # Import triggers @RobotConfig.register_subclass()
    ...
)
from lerobot.teleoperators import (
    bi_so101_leader,    # Import triggers @TeleoperatorConfig.register_subclass()
    ...
)
```

---

## Calibration Files

Calibration files are stored at: `~/.cache/huggingface/lerobot/calibration/`

Backups created at: `jdocs/bimanual/jassy/backup_calibration_files/`

### Recalibration Process

To recalibrate an arm:
1. Delete the corresponding calibration file from `~/.cache/huggingface/lerobot/calibration/`
2. Run teleoperate command
3. Position arm at middle of range with gripper halfway open
4. Press ENTER

Or during teleoperation startup, type 'c' when prompted if calibration exists.

---

## Usage Guide

### Running Bimanual Data Collection

```bash
cd /home/jrobot/project/lerobot/jdocs/bimanual/jassy/scripts

# Interactive mode
python collect_bimanuel_xlerobot_data.py

# Direct task selection
python collect_bimanuel_xlerobot_data.py --task handover --num_episodes 50

# Dry run (show command without executing)
python collect_bimanuel_xlerobot_data.py --task handover --num_episodes 1 --dry-run
```

### Recording Controls

| Key | Action |
|-----|--------|
| RIGHT ARROW (→) | End episode early (task complete) |
| LEFT ARROW (←) | Re-record episode (made mistake) |
| ESCAPE | Stop all recording |

### Dataset Output

Datasets are saved to: `/home/jrobot/project/lerobot/datasets_bimanuel/bimanual/<task_type>/`

Structure:
```
bimanual_pick/
├── data/chunk-000/file-000.parquet    # Joint states and actions
├── meta/
│   ├── info.json                       # Dataset metadata
│   ├── stats.json                      # Statistics
│   └── tasks.parquet                   # Task descriptions
└── videos/
    ├── observation.images.head/
    ├── observation.images.left_wrist/
    └── observation.images.right_wrist/
```

---

## Dataset Schema (Bimanual)

### Action Space (12 dimensions)
```
[0-5]:  left arm  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
[6-11]: right arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
```

### Observations
- `observation.state`: 12-dim joint positions (same order as actions)
- `observation.images.head`: Head camera (640x480)
- `observation.images.left_wrist`: Left wrist camera (640x480)
- `observation.images.right_wrist`: Right wrist camera (640x480)

---

## Notes

- SO-100 vs SO-101: Nearly identical hardware, main software difference is wrist_roll calibration (SO-100 treats it as full-turn motor)
- The bi_so101 classes are exact copies of bi_so100 with SO100→SO101 substitutions
- Camera zoom feature was discussed but not implemented (physical repositioning recommended instead)
