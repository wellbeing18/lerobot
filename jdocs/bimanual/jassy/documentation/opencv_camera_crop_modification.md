# OpenCV Camera Digital Zoom Feature Modification

**Date**: 2024-12-31
**Purpose**: Add digital zoom (center crop) functionality to LeRobot's OpenCV camera for the head camera

## Overview

This modification adds `capture_width` and `capture_height` parameters to the OpenCV camera configuration, allowing you to capture at a higher resolution and center crop to the output `width` x `height` for an effective "zoom" effect.

**Use case**: Head camera mounted high captures too much background. By capturing at 1280x960 and cropping center 640x480, we get ~2x zoom.

**Key concept**:
- `width` / `height` = **OUTPUT** dimensions (what goes to dataset, what model sees)
- `capture_width` / `capture_height` = **CAPTURE** dimensions (camera hardware resolution)

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

1. Added two new optional fields after `fourcc`:
   ```python
   # For digital zoom: capture at higher resolution, then center crop to width x height
   capture_width: int | None = None
   capture_height: int | None = None
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

2. **In `_postprocess_image`**: Added center crop logic after color conversion, before rotation:
   ```python
   # Apply center crop if configured (for digital zoom effect)
   # Crop from capture resolution to output resolution (self.width x self.height)
   if self.do_center_crop:
       h, w = processed_image.shape[:2]
       crop_x = (w - self.width) // 2
       crop_y = (h - self.height) // 2
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

### New head camera config (with ~2x zoom)

```python
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "capture_width": 1280, "capture_height": 960, "fps": 30, "fourcc": "MJPG"}
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
1. Remove the `capture_width` and `capture_height` field definitions
2. Remove the capture dimension validation block in `__post_init__`

**For camera_opencv.py**:
1. Remove `self.do_center_crop` and the if/else block that uses `config.capture_width`
2. Restore the original capture dimension logic:
   ```python
   if self.height and self.width:
       self.capture_width, self.capture_height = self.width, self.height
       if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
           self.capture_width, self.capture_height = self.height, self.width
   ```
3. Remove the center crop block in `_postprocess_image`

**For collect_bimanuel_xlerobot_data.py**:
Remove `capture_width` and `capture_height` from all head camera configs.

---

## Usage After Modification

### Config format

```python
# No zoom (normal capture)
"camera_name": {
    "type": "opencv",
    "index_or_path": 4,
    "width": 640,      # Output dimensions
    "height": 480,
    "fps": 30,
    "fourcc": "MJPG"
}

# With ~2x digital zoom
"camera_name": {
    "type": "opencv",
    "index_or_path": 4,
    "width": 640,            # Output dimensions (what goes to dataset)
    "height": 480,
    "capture_width": 1280,   # Capture at higher resolution
    "capture_height": 960,   # Must be >= width/height
    "fps": 30,
    "fourcc": "MJPG"
}
```

### How it works

1. Camera captures frames at `capture_width` x `capture_height` (1280x960)
2. Center region of `width` x `height` (640x480) is cropped
3. Output is 640x480 - same as other cameras, but with narrower field of view (zoom effect)

### Zoom levels

| Capture Resolution | Output | Zoom Factor |
|-------------------|--------|-------------|
| 1280x960 | 640x480 | ~2x |
| 1920x1080 | 640x480 | ~2.25x (with letterboxing) |
| 1024x768 | 640x480 | ~1.6x |

---

## Testing

### Quick test
```bash
cd /home/jrobot/project/lerobot
python -c "
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Test digital zoom config
config = OpenCVCameraConfig(
    index_or_path=4,
    fps=30,
    width=640,
    height=480,
    capture_width=1280,
    capture_height=960,
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
python jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py --task right_arm_pick_and_place --num_episodes 1
```

Then compare videos:
- Head camera: `datasets_bimanuel/bimanual/<task>/videos/observation.images.head/`
- Wrist cameras: `datasets_bimanuel/bimanual/<task>/videos/observation.images.left_wrist/`

The head camera should show a more "zoomed in" view (less background, table/arms appear larger).
