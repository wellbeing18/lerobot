# OpenCV Camera Crop Feature Modification

**Date**: 2024-12-31
**Purpose**: Add digital crop functionality to LeRobot's OpenCV camera for the head camera to remove unwanted top portion of frame (simulates tilting camera down)

## Overview

This modification adds three parameters to the OpenCV camera configuration:
- `capture_width` / `capture_height` - Capture at a different resolution than output
- `crop_y_offset` - Offset the crop region vertically (positive = remove more from top)

**Current use case**: Head camera mounted high captures too much background/ceiling. By capturing at 800x600 and cropping to 640x480 with `crop_y_offset=60`, we remove the top 120 pixels (20% of frame), effectively "tilting" the camera down without physical adjustment.

**Key concepts**:
- `width` / `height` = **OUTPUT** dimensions (what goes to dataset, what model sees)
- `capture_width` / `capture_height` = **CAPTURE** dimensions (camera hardware resolution)
- `crop_y_offset` = Vertical offset from center crop (positive = remove more from top)

## Files Modified

1. `src/lerobot/cameras/opencv/configuration_opencv.py`
2. `src/lerobot/cameras/opencv/camera_opencv.py`
3. `jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py`

---

## File 1: configuration_opencv.py

### Original Code (FULL FILE - for revert)

```python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OpenCVCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations with FOURCC format
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90, fourcc="MJPG")     # With 90° rotation and MJPG format
    OpenCVCameraConfig(0, 30, 1280, 720, fourcc="YUYV")     # With YUYV format
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        fourcc: FOURCC code for video format (e.g., "MJPG", "YUYV", "I420"). Defaults to None (auto-detect).

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
        - FOURCC codes must be 4-character strings (e.g., "MJPG", "YUYV"). Some common FOUCC codes: https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs#fourcc-constants
        - Setting FOURCC can help achieve higher frame rates on some cameras.
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )
```

### Changes Made

1. Added three new fields after `fourcc`:
   ```python
   # For digital zoom: capture at higher resolution, then center crop to width x height
   capture_width: int | None = None
   capture_height: int | None = None
   # Offset crop from center: positive = remove more from top, negative = remove more from bottom
   crop_y_offset: int = 0
   ```

2. Added validation in `__post_init__()`:
   ```python
   # Validate capture dimensions (for digital zoom/crop)
   if (self.capture_width is None) != (self.capture_height is None):
       raise ValueError(
           "Both `capture_width` and `capture_height` must be specified together, or both must be None."
       )

   if self.capture_width is not None and self.capture_height is not None:
       if self.width is not None and self.capture_width < self.width:
           raise ValueError(
               f"`capture_width` ({self.capture_width}) must be >= `width` ({self.width})."
           )
       if self.height is not None and self.capture_height < self.height:
           raise ValueError(
               f"`capture_height` ({self.capture_height}) must be >= `height` ({self.height})."
           )
   ```

---

## File 2: camera_opencv.py

### Original `__init__` method (lines 104-134)

```python
def __init__(self, config: OpenCVCameraConfig):
    """
    Initializes the OpenCVCamera instance.

    Args:
        config: The configuration settings for the camera.
    """
    super().__init__(config)

    self.config = config
    self.index_or_path = config.index_or_path

    self.fps = config.fps
    self.color_mode = config.color_mode
    self.warmup_s = config.warmup_s

    self.videocapture: cv2.VideoCapture | None = None

    self.thread: Thread | None = None
    self.stop_event: Event | None = None
    self.frame_lock: Lock = Lock()
    self.latest_frame: NDArray[Any] | None = None
    self.new_frame_event: Event = Event()

    self.rotation: int | None = get_cv2_rotation(config.rotation)
    self.backend: int = get_cv2_backend()

    if self.height and self.width:
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width
```

### Original `_postprocess_image` method (lines 385-426)

```python
def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
    """
    Applies color conversion, dimension validation, and rotation to a raw frame.

    Args:
        image (np.ndarray): The raw image frame (expected BGR format from OpenCV).
        color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                         uses the instance's default `self.color_mode`.

    Returns:
        np.ndarray: The processed image frame.

    Raises:
        ValueError: If the requested `color_mode` is invalid.
        RuntimeError: If the raw frame dimensions do not match the configured
                      `width` and `height`.
    """
    requested_color_mode = self.color_mode if color_mode is None else color_mode

    if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
        raise ValueError(
            f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
        )

    h, w, c = image.shape

    if h != self.capture_height or w != self.capture_width:
        raise RuntimeError(
            f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
        )

    if c != 3:
        raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

    processed_image = image
    if requested_color_mode == ColorMode.RGB:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
        processed_image = cv2.rotate(processed_image, self.rotation)

    return processed_image
```

### Changes Made

1. **In `__init__`**: Replaced the capture dimension logic with:
   ```python
   # Digital zoom: capture at higher resolution, crop to width x height
   # If capture_width/capture_height are set, use those for camera capture
   # Otherwise capture at width x height (no crop)
   self.do_center_crop = config.capture_width is not None and config.capture_height is not None
   self.crop_y_offset = config.crop_y_offset

   if self.height and self.width:
       if self.do_center_crop:
           # Capture at higher resolution, will crop to width x height
           self.capture_width, self.capture_height = config.capture_width, config.capture_height
           if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
               self.capture_width, self.capture_height = config.capture_height, config.capture_width
       else:
           # No crop, capture at output resolution
           self.capture_width, self.capture_height = self.width, self.height
           if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
               self.capture_width, self.capture_height = self.height, self.width
   ```

2. **In `_postprocess_image`**: Added crop logic after color conversion, before rotation:
   ```python
   # Apply center crop if configured (for digital zoom effect)
   # Crop from capture resolution to output resolution (self.width x self.height)
   # crop_y_offset: positive = remove more from top, negative = remove more from bottom
   if self.do_center_crop:
       h, w = processed_image.shape[:2]
       crop_x = (w - self.width) // 2
       crop_y = (h - self.height) // 2 + self.crop_y_offset
       # Clamp to valid range
       crop_y = max(0, min(crop_y, h - self.height))
       processed_image = processed_image[
           crop_y : crop_y + self.height,
           crop_x : crop_x + self.width
       ]
   ```

---

## File 3: collect_bimanuel_xlerobot_data.py

### Original head camera config

```python
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
```

### Current head camera config (crop top, minimal zoom)

```python
"head": {
    "type": "opencv",
    "index_or_path": 4,
    "width": 640,
    "height": 480,
    "capture_width": 800,
    "capture_height": 600,
    "crop_y_offset": 60,
    "fps": 30,
    "fourcc": "MJPG"
}
```

Updated in 3 locations:
- `ARM_CONFIGS["left"]["cameras"]["head"]`
- `ARM_CONFIGS["right"]["cameras"]["head"]`
- `ARM_CONFIGS["bimanual"]["cameras"]["head"]`

---

## How to Revert

### Option 1: Git revert (recommended)
```bash
cd /home/jrobot/project/lerobot
git checkout HEAD -- src/lerobot/cameras/opencv/configuration_opencv.py
git checkout HEAD -- src/lerobot/cameras/opencv/camera_opencv.py
```

### Option 2: Manual revert

**For configuration_opencv.py**:
1. Remove the `capture_width`, `capture_height`, and `crop_y_offset` field definitions
2. Remove the capture dimension validation block in `__post_init__`

**For camera_opencv.py**:
1. Remove `self.do_center_crop`, `self.crop_y_offset` and the if/else block that uses `config.capture_width`
2. Restore the original capture dimension logic:
   ```python
   if self.height and self.width:
       self.capture_width, self.capture_height = self.width, self.height
       if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
           self.capture_width, self.capture_height = self.height, self.width
   ```
3. Remove the crop block in `_postprocess_image`

**For collect_bimanuel_xlerobot_data.py**:
Remove `capture_width`, `capture_height`, and `crop_y_offset` from all head camera configs.

---

## Usage Guide

### Config Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `width` | Output width (what goes to dataset) | Required |
| `height` | Output height (what goes to dataset) | Required |
| `capture_width` | Camera capture width (must be >= width) | None (same as width) |
| `capture_height` | Camera capture height (must be >= height) | None (same as height) |
| `crop_y_offset` | Vertical crop offset: + = remove more from top | 0 (center crop) |

### Current Configuration Explained

```python
"head": {
    "width": 640, "height": 480,           # Output: 640x480
    "capture_width": 800, "capture_height": 600,  # Capture: 800x600
    "crop_y_offset": 60,                   # Remove 120px from top (all vertical excess)
    ...
}
```

**What happens:**
1. Camera captures at 800x600
2. Horizontal: 800 - 640 = 160px to crop → 80px from each side (center)
3. Vertical: 600 - 480 = 120px to crop
   - Center crop would be: 60px from top, 60px from bottom
   - With `crop_y_offset=60`: 120px from top, 0px from bottom
4. Output: 640x480 with top portion removed (simulates tilting camera down)

### Adjusting the Crop

**To remove more/less from top**, change `crop_y_offset`:

| crop_y_offset | Top removed | Bottom removed | Effect |
|---------------|-------------|----------------|--------|
| 0 | 60px | 60px | Center crop |
| 30 | 90px | 30px | Slight tilt down |
| 60 | 120px | 0px | Maximum tilt down (current) |
| -30 | 30px | 90px | Tilt up |

**To remove MORE than 120px from top**, you need a larger capture resolution (but this adds horizontal zoom):

| Capture | Vertical crop available | Horizontal crop (zoom) |
|---------|------------------------|----------------------|
| 800x600 | 120px | 160px (minimal) |
| 1024x768 | 288px | 384px (noticeable zoom) |
| 1280x960 | 480px | 640px (2x zoom) |

### Example Configurations

```python
# No crop (original behavior)
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}

# Crop top only, minimal zoom (CURRENT SETUP)
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "capture_width": 800, "capture_height": 600, "crop_y_offset": 60, "fps": 30, "fourcc": "MJPG"}

# 2x zoom with center crop
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "capture_width": 1280, "capture_height": 960, "fps": 30, "fourcc": "MJPG"}

# 2x zoom with top crop
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "capture_width": 1280, "capture_height": 960, "crop_y_offset": 240, "fps": 30, "fourcc": "MJPG"}
```

---

## Testing

### Quick test
```bash
cd /home/jrobot/project/lerobot
python -c "
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Test current config (crop top, minimal zoom)
config = OpenCVCameraConfig(
    index_or_path=4,
    fps=30,
    width=640,
    height=480,
    capture_width=800,
    capture_height=600,
    crop_y_offset=60,
    fourcc='MJPG'
)
camera = OpenCVCamera(config)
camera.connect()
frame = camera.read()
print(f'Frame shape: {frame.shape}')  # Should be (480, 640, 3)
camera.disconnect()
print('SUCCESS!')
"
```

### Full recording test
```bash
python jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py --task bimanual_pick_and_place --num_episodes 1
```

Then check videos:
- Head camera: `datasets_bimanuel/bimanual/<task>/videos/observation.images.head/`

The head camera should show less ceiling/background at the top compared to original.
