# Central Config Camera Crop Feature

**Date**: 2026-01-02
**Author**: Claude Code Assistant
**Purpose**: Enable head camera cropping for both data collection AND inference through a single central configuration file.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Files Modified](#files-modified)
4. [Central Configuration](#central-configuration)
5. [File 1: Data Collection Script](#file-1-data-collection-script)
6. [File 2: Inference Script](#file-2-inference-script)
7. [How It Works](#how-it-works)
8. [How to Revert](#how-to-revert)
9. [How to Test](#how-to-test)
10. [Troubleshooting](#troubleshooting)

---

## Problem Statement

### Background
The head camera is mounted high and captures too much ceiling/background. We need to crop the top portion of the frame to simulate "tilting the camera down" without physical adjustment.

### The Challenge
Previously, crop settings were added to the data collection script only. The inference script used a separate `CameraManager` class with raw `cv2.VideoCapture` that had no crop logic. This meant:
- Data collection saw cropped view
- Inference saw full uncropped view
- Model trained on cropped data but inferred on uncropped data = mismatch

### Why Not Replace the Inference Camera Module?
We investigated replacing the inference script's `CameraManager` with LeRobot's `OpenCVCamera`. However, this carries risks:
- LeRobot's camera raises errors if resolution doesn't match exactly
- LeRobot has a 1-second warmup delay per camera
- Different threading behavior could affect timing
- Unnecessary risk for a simple crop feature

### Solution Chosen
Keep both scripts' camera code as-is, but make minimal edits so both read crop settings from the same central config file.

---

## Solution Overview

```
xlerobot_bimanual.yaml (Central Config)
         |
         | crop settings: capture_width, capture_height, crop_y_offset
         |
    +----+----+
    |         |
    v         v
Data         Inference
Collection   Script
Script
    |         |
    v         v
LeRobot      Custom
OpenCV       CameraManager
Camera       (with crop logic added)
    |         |
    v         v
Same cropped view for both!
```

**Key principle**: One config file controls cropping for both scripts.

---

## Files Modified

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| `jdocs/configs/hardware/xlerobot_bimanual.yaml` | Already has crop settings | None (no change needed) |
| `jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py` | Copy crop settings from central config | ~8 lines |
| `jdocs/scripts/bimanual/infer_smolvla_bimanual.py` | Add crop logic to CameraManager | ~25 lines |

---

## Central Configuration

**File**: `jdocs/configs/hardware/xlerobot_bimanual.yaml`

The central config already contains crop settings for the head camera:

```yaml
cameras:
  head:
    type: opencv
    index_or_path: 4
    width: 640           # OUTPUT width (what model sees)
    height: 480          # OUTPUT height (what model sees)
    fps: 30
    fourcc: MJPG
    capture_width: 800   # CAPTURE width (camera hardware)
    capture_height: 600  # CAPTURE height (camera hardware)
    crop_y_offset: 60    # Vertical offset: positive = remove more from top
    usb_serial: null
    usb_path: null
  left_wrist:
    # ... no crop settings (captures at width x height directly)
  right_wrist:
    # ... no crop settings (captures at width x height directly)
```

### Crop Settings Explained

| Setting | Description | Current Value |
|---------|-------------|---------------|
| `width` / `height` | Output dimensions (what goes to model) | 640 x 480 |
| `capture_width` / `capture_height` | Camera capture dimensions | 800 x 600 |
| `crop_y_offset` | Vertical crop offset from center | 60 |

### How Crop Math Works

1. Camera captures at 800x600
2. Horizontal crop: (800 - 640) / 2 = 80px from each side
3. Vertical crop: (600 - 480) / 2 = 60px from top and bottom (center)
4. With `crop_y_offset=60`: shifts crop up by 60px
   - Top: 60 + 60 = 120px removed
   - Bottom: 60 - 60 = 0px removed
5. Result: Top 120px cropped, simulating camera tilted down

---

## File 1: Data Collection Script

**File**: `jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py`

### Why This Change Is Needed
The script has a function `update_arm_configs_from_central()` that loads settings from the central config. Currently it only copies `index_or_path` for cameras, ignoring crop settings.

### Location
Function: `update_arm_configs_from_central()`
Lines: ~216-220

### BEFORE (Original Code)

```python
        # Camera indices
        for cam_name in ["head", "left_wrist", "right_wrist"]:
            if camera_cfg.get(cam_name, {}).get("index_or_path") is not None:
                if cam_name in bimanual["cameras"]:
                    bimanual["cameras"][cam_name]["index_or_path"] = camera_cfg[cam_name]["index_or_path"]
```

### AFTER (Modified Code)

```python
        # Camera settings (index, crop, etc.)
        for cam_name in ["head", "left_wrist", "right_wrist"]:
            cam_cfg = camera_cfg.get(cam_name, {})
            if cam_name in bimanual["cameras"]:
                # Copy index
                if cam_cfg.get("index_or_path") is not None:
                    bimanual["cameras"][cam_name]["index_or_path"] = cam_cfg["index_or_path"]
                # Copy crop settings (for digital zoom)
                if cam_cfg.get("capture_width") is not None:
                    bimanual["cameras"][cam_name]["capture_width"] = cam_cfg["capture_width"]
                if cam_cfg.get("capture_height") is not None:
                    bimanual["cameras"][cam_name]["capture_height"] = cam_cfg["capture_height"]
                if cam_cfg.get("crop_y_offset") is not None:
                    bimanual["cameras"][cam_name]["crop_y_offset"] = cam_cfg["crop_y_offset"]
```

### What This Does
- Copies `capture_width`, `capture_height`, and `crop_y_offset` from central config to ARM_CONFIGS
- These settings are then passed to `lerobot-record` command
- LeRobot's OpenCVCamera (which already has crop support) applies the crop

---

## File 2: Inference Script

**File**: `jdocs/scripts/bimanual/infer_smolvla_bimanual.py`

### Why This Change Is Needed
The script has a custom `CameraManager` class that uses raw `cv2.VideoCapture`. It has no crop logic - it just captures at width x height directly.

### Location
Class: `CameraManager`
Lines: ~204-272

### BEFORE (Original Code)

```python
class CameraManager:
    """Manage camera capture for bimanual setup."""

    def __init__(self, hw_config: dict):
        self.cameras = {}
        cam_config = hw_config.get("cameras", {})

        # Initialize head camera
        if "head" in cam_config:
            head_cfg = cam_config["head"]
            self.cameras["head"] = self._init_camera(
                head_cfg.get("index_or_path", 4),
                head_cfg.get("width", 640),
                head_cfg.get("height", 480),
                "head"
            )

        # Initialize left wrist camera
        if "left_wrist" in cam_config:
            left_cfg = cam_config["left_wrist"]
            self.cameras["left_wrist"] = self._init_camera(
                left_cfg.get("index_or_path", 6),
                left_cfg.get("width", 640),
                left_cfg.get("height", 480),
                "left_wrist"
            )

        # Initialize right wrist camera (optional)
        if "right_wrist" in cam_config:
            right_cfg = cam_config["right_wrist"]
            self.cameras["right_wrist"] = self._init_camera(
                right_cfg.get("index_or_path", 8),
                right_cfg.get("width", 640),
                right_cfg.get("height", 480),
                "right_wrist"
            )

    def _init_camera(self, device_index: int, width: int, height: int, name: str) -> cv2.VideoCapture:
        """Initialize a single camera."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {name} camera at index {device_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  {name} camera initialized: {actual_w}x{actual_h}")

        return cap

    def capture(self) -> dict:
        """Capture frames from all cameras."""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

    def release(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()
```

### AFTER (Modified Code)

```python
class CameraManager:
    """Manage camera capture for bimanual setup."""

    def __init__(self, hw_config: dict):
        self.cameras = {}
        self.crop_settings = {}  # Store crop settings per camera
        cam_config = hw_config.get("cameras", {})

        # Initialize head camera
        if "head" in cam_config:
            head_cfg = cam_config["head"]
            self.cameras["head"] = self._init_camera(
                head_cfg.get("index_or_path", 4),
                head_cfg.get("capture_width") or head_cfg.get("width", 640),
                head_cfg.get("capture_height") or head_cfg.get("height", 480),
                "head"
            )
            # Store crop settings if capture dimensions differ from output
            if head_cfg.get("capture_width") and head_cfg.get("capture_height"):
                self.crop_settings["head"] = {
                    "output_width": head_cfg.get("width", 640),
                    "output_height": head_cfg.get("height", 480),
                    "crop_y_offset": head_cfg.get("crop_y_offset", 0),
                }

        # Initialize left wrist camera
        if "left_wrist" in cam_config:
            left_cfg = cam_config["left_wrist"]
            self.cameras["left_wrist"] = self._init_camera(
                left_cfg.get("index_or_path", 6),
                left_cfg.get("capture_width") or left_cfg.get("width", 640),
                left_cfg.get("capture_height") or left_cfg.get("height", 480),
                "left_wrist"
            )
            if left_cfg.get("capture_width") and left_cfg.get("capture_height"):
                self.crop_settings["left_wrist"] = {
                    "output_width": left_cfg.get("width", 640),
                    "output_height": left_cfg.get("height", 480),
                    "crop_y_offset": left_cfg.get("crop_y_offset", 0),
                }

        # Initialize right wrist camera (optional)
        if "right_wrist" in cam_config:
            right_cfg = cam_config["right_wrist"]
            self.cameras["right_wrist"] = self._init_camera(
                right_cfg.get("index_or_path", 8),
                right_cfg.get("capture_width") or right_cfg.get("width", 640),
                right_cfg.get("capture_height") or right_cfg.get("height", 480),
                "right_wrist"
            )
            if right_cfg.get("capture_width") and right_cfg.get("capture_height"):
                self.crop_settings["right_wrist"] = {
                    "output_width": right_cfg.get("width", 640),
                    "output_height": right_cfg.get("height", 480),
                    "crop_y_offset": right_cfg.get("crop_y_offset", 0),
                }

    def _init_camera(self, device_index: int, width: int, height: int, name: str) -> cv2.VideoCapture:
        """Initialize a single camera."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {name} camera at index {device_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  {name} camera initialized: {actual_w}x{actual_h}")

        return cap

    def capture(self) -> dict:
        """Capture frames from all cameras, applying crop if configured."""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply crop if configured for this camera
            if name in self.crop_settings:
                crop = self.crop_settings[name]
                h, w = frame.shape[:2]
                out_w, out_h = crop["output_width"], crop["output_height"]
                crop_y_offset = crop["crop_y_offset"]

                # Calculate crop region
                crop_x = (w - out_w) // 2
                crop_y = (h - out_h) // 2 + crop_y_offset
                # Clamp to valid range
                crop_y = max(0, min(crop_y, h - out_h))

                frame = frame[crop_y:crop_y + out_h, crop_x:crop_x + out_w]

            frames[name] = frame
        return frames

    def release(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()
```

### What This Does
1. Reads `capture_width`, `capture_height`, `crop_y_offset` from config
2. Initializes camera at capture resolution (800x600)
3. Stores crop settings in `self.crop_settings` dict
4. In `capture()`, applies crop to get output resolution (640x480)
5. Uses same crop math as LeRobot's OpenCVCamera for consistency

---

## How It Works

### Data Collection Flow
```
Central Config (xlerobot_bimanual.yaml)
    |
    v
update_arm_configs_from_central() copies crop settings
    |
    v
ARM_CONFIGS["bimanual"]["cameras"]["head"] now has crop settings
    |
    v
lerobot-record command receives camera config as JSON
    |
    v
LeRobot's OpenCVCamera applies crop (already implemented)
    |
    v
Dataset recorded with cropped head camera view
```

### Inference Flow
```
Central Config (xlerobot_bimanual.yaml)
    |
    v
load_hardware_config() loads full config
    |
    v
CameraManager.__init__() reads crop settings, stores in self.crop_settings
    |
    v
CameraManager.capture() applies crop when reading frames
    |
    v
Policy receives same cropped view as training data
```

---

## How to Revert

### Option 1: Revert Code Changes Only (Keep Config)

**Data Collection Script:**
Replace the modified camera loop with original:

```python
        # Camera indices
        for cam_name in ["head", "left_wrist", "right_wrist"]:
            if camera_cfg.get(cam_name, {}).get("index_or_path") is not None:
                if cam_name in bimanual["cameras"]:
                    bimanual["cameras"][cam_name]["index_or_path"] = camera_cfg[cam_name]["index_or_path"]
```

**Inference Script:**
Replace the entire `CameraManager` class with the original version shown in the BEFORE section above.

### Option 2: Full Revert (Remove Config Too)

1. Do Option 1 above
2. Edit `xlerobot_bimanual.yaml` and remove these lines from the head camera:
   ```yaml
   capture_width: 800
   capture_height: 600
   crop_y_offset: 60
   ```

### Option 3: Git Revert

If you committed before making changes:
```bash
git checkout HEAD -- jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py
git checkout HEAD -- jdocs/scripts/bimanual/infer_smolvla_bimanual.py
```

---

## How to Test

### Test 1: Data Collection Crop
```bash
# Record a short test episode
python jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py \
    --task bimanual_pick_and_place \
    --num_episodes 1 \
    --dry-run

# Check the camera config in the generated command
# Should show capture_width: 800, capture_height: 600, crop_y_offset: 60
```

### Test 2: Inference Crop
```bash
# Run inference with dry-run
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
    --checkpoint <your_checkpoint> \
    --dry-run \
    --duration 5

# Check logs for camera initialization
# Should show: "head camera initialized: 800x600"
# And crop should be applied in capture()
```

### Test 3: Visual Comparison
1. Run data collection and save a frame
2. Run inference and save a frame
3. Compare both frames - they should show the same cropped view

---

## Troubleshooting

### Issue: Camera Not Capturing at 800x600
**Symptom**: Logs show camera initialized at different resolution
**Cause**: Camera doesn't support 800x600
**Solution**: Check supported resolutions with `v4l2-ctl --list-formats-ext -d /dev/video4`

### Issue: Crop Looks Wrong
**Symptom**: More/less cropped than expected
**Cause**: Wrong crop_y_offset value
**Solution**: Adjust `crop_y_offset` in central config:
- `0` = center crop (60px from top and bottom)
- `60` = max top crop (120px from top, 0px from bottom)
- `-60` = max bottom crop (0px from top, 120px from bottom)

### Issue: Data Collection Works But Inference Doesn't Crop
**Symptom**: Inference shows full frame
**Cause**: Inference script not updated, or not loading central config
**Solution**: Verify inference script has the modified `CameraManager` class

### Issue: Resolution Mismatch Error During Recording
**Symptom**: LeRobot raises "frame dimensions do not match"
**Cause**: Camera returns different resolution than capture_width x capture_height
**Solution**: Use a resolution the camera actually supports

---

## Configuration Reference

### To Adjust Crop Amount

Edit `jdocs/configs/hardware/xlerobot_bimanual.yaml`:

| Want to... | Change |
|------------|--------|
| Remove more from top | Increase `crop_y_offset` (max = 60) |
| Remove less from top | Decrease `crop_y_offset` (min = -60) |
| More zoom (crop sides too) | Increase `capture_width` and `capture_height` |
| Less zoom | Decrease `capture_width`/`capture_height` closer to 640x480 |
| No crop at all | Remove `capture_width`, `capture_height`, `crop_y_offset` |

### Supported Capture Resolutions (typical USB camera)

| Resolution | Aspect | Horizontal Crop | Vertical Crop |
|------------|--------|-----------------|---------------|
| 640x480 | 4:3 | 0px | 0px (no crop) |
| 800x600 | 4:3 | 160px | 120px |
| 1024x768 | 4:3 | 384px | 288px |
| 1280x960 | 4:3 | 640px | 480px (2x zoom) |

---

## Summary

This modification enables head camera cropping through a single central config file by:
1. Making data collection copy crop settings from central config to LeRobot
2. Adding crop logic to inference's CameraManager class

Both scripts now read from `xlerobot_bimanual.yaml` and produce identical cropped views.
