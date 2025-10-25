# XLeRobot (SO-ARM-101) MVP Plan v1
## Comprehensive Research-Based Execution Roadmap

**Created:** 2025-10-25
**Status:** Ready for Execution
**Goal:** Progress from hardware verification to hierarchical VLM+VLA control with minimal data collection/training cost

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Available SO-101 Pretrained Models](#available-so-101-pretrained-models)
3. [Latest Small VLM Models (2025)](#latest-small-vlm-models-2025)
4. [Stage 1: Hardware Verification](#stage-1-hardware-verification-mvp-testing)
5. [Stage 2: VLA Inference with Pretrained Models](#stage-2-vla-inference-with-pretrained-models)
6. [Stage 3a: Natural Language Control](#stage-3a-natural-language-control-for-simple-tasks)
7. [Stage 3b: Complex Multi-Step Task Planning](#stage-3b-complex-multi-step-task-planning)
8. [Fallback: Custom Data Collection](#fallback-custom-data-collection-only-if-needed)
9. [Cost Analysis](#cost-analysis)
10. [Timeline & Milestones](#timeline--milestones)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [References & Resources](#references--resources)

---

## Executive Summary

### Key Strategy
**Leverage pretrained models first, collect custom data only if necessary.**

This plan prioritizes using existing SO-101 pretrained VLA models and state-of-the-art small VLM models (2025) to minimize resource investment. Based on comprehensive research, SO-101 has excellent community support with multiple pretrained models achieving 70%+ success rates.

### Hardware Requirements
- **SO-101 Follower Arms:** 2x (already assembled)
- **Computer:** Laptop with RTX 5090 GPU
- **Cameras:** Intel RealSense or compatible RGB cameras
- **Optional:** SO-101 Leader Arm (~$350, only needed if custom data collection required)

### Success Metrics
- **Stage 1:** Arms respond smoothly to commands, cameras functional
- **Stage 2:** 50%+ success rate using pretrained VLA models (no training)
- **Stage 3a:** 70%+ correct responses to simple language commands
- **Stage 3b:** 60%+ success on 2-3 step complex tasks

---

## Available SO-101 Pretrained Models

### Priority 1: SmolVLA (RECOMMENDED) ⭐⭐⭐⭐⭐

**Model:** `lerobot/smolvla_base`

**Key Statistics:**
- **Parameters:** 450M (400M VLM + 50M action expert)
- **Performance:** 78.3% success on SO100/SO101 (vs 51.7% without pretraining)
- **Training Data:** 487 high-quality community datasets (SO100/SO101 focused)
- **Control Frequency:** 30Hz
- **Language Conditioning:** Native support

**Why This Model:**
- Pre-trained specifically on SO100/SO101 community data
- +26.6% absolute improvement from community pretraining
- Asynchronous inference stack for low-latency control
- Runs efficiently on single GPU or even CPU
- Native LeRobot integration - zero custom code needed

**Available Datasets:**
- `lerobot/svla_so101_pickplace` - 50 episodes, 11,939 frames, 30 FPS
- `lerobot/svla_so100_stacking` - 40 episodes, 16,496 frames

**How to Use:**
```python
from lerobot.policies.smolvla import SmolVLAPolicy

# Load pretrained model
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Inference with language conditioning
observation = {
    "observation.images.top": top_camera_image,       # (3, 480, 640)
    "observation.images.wrist": wrist_camera_image,   # (3, 240, 320)
    "observation.state": robot.get_joint_positions(), # (6,)
    "task": "pick up the red cube"  # Natural language!
}

action = policy.select_action(observation)  # (6,) joint positions
robot.send_action(action)
```

**Repository:** https://huggingface.co/lerobot/smolvla_base
**Documentation:** https://huggingface.co/docs/lerobot/smolvla

---

### Priority 2: ACT Model for SO-101 ⭐⭐⭐⭐

**Model:** `r2owb0/act1`

**Key Statistics:**
- **Architecture:** Action Chunking Transformer (ACT)
- **Training Data:** 10 episodes, 5,990 frames from `r2owb0/so101-DS1`
- **Training Steps:** 25,000 steps, batch size 4, lr 1e-5
- **Control Frequency:** 50Hz
- **Chunk Size:** 50 steps (1 second lookahead)
- **Model Size:** ~200MB

**Hardware Configuration:**
- **Vision Backbone:** ResNet18 (ImageNet pretrained)
- **Cameras:**
  - Top camera: 480×640 pixels
  - Wrist camera: 240×320 pixels
- **Robot State:** 6-dimensional joint positions
- **Output:** 6-dimensional joint commands

**Why This Model:**
- Specifically trained for SO-101 hardware
- Faster inference than SmolVLA (50Hz vs 30Hz)
- Smaller model size (200MB vs multi-GB)
- Action chunking enables smooth trajectories

**Limitations:**
- Trained on only 10 episodes (less robust than SmolVLA)
- No language conditioning
- Requires exact camera setup from training

**How to Use:**
```python
from lerobot.policies.act import ACTPolicy

# Load SO-101 specific ACT model
policy = ACTPolicy.from_pretrained("r2owb0/act1")

# Inference (no language support)
observation = {
    "observation.images.top": top_camera_image,
    "observation.images.wrist": wrist_camera_image,
    "observation.state": robot.get_joint_positions()
}

action = policy.select_action(observation)
robot.send_action(action)
```

**Repository:** https://huggingface.co/r2owb0/act1
**Training Dataset:** https://huggingface.co/datasets/r2owb0/so101-DS1

---

### Priority 3: Pi0 Foundation Model ⭐⭐⭐⭐

**Model:** `lerobot/pi0` or `lerobot/pi05`

**Key Statistics:**
- **Parameters:** 4B (3.7B PaliGemma VLM + 300M diffusion action expert)
- **Training Data:** π Cross-Embodiment Robot dataset (8 robot platforms)
- **Control Frequency:** Variable (depends on action diffusion steps)
- **Language Conditioning:** Native support via VLM

**Why This Model:**
- Foundation model from Physical Intelligence
- Cross-embodiment training (generalizes better)
- State-of-the-art performance on diverse manipulation tasks
- Native LeRobot integration

**Considerations:**
- Not specifically trained on SO-101 (may need fine-tuning)
- Larger model size (4B params)
- Slower inference than ACT
- Excellent for zero-shot transfer experiments

**How to Use:**
```python
from lerobot.policies.pi0 import Pi0Policy

# Load base model
policy = Pi0Policy.from_pretrained("lerobot/pi0")

# Inference with language
observation = {
    "observation.images.top": top_camera_image,
    "observation.state": robot.get_joint_positions(),
    "task": "grasp the object"
}

action = policy.select_action(observation)
```

**Fine-tuning on SO-101:**
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/pi0 \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --training.num_epochs=1000 \
  --output_dir=outputs/pi0_so101_finetuned
```

**Repository:** https://huggingface.co/lerobot/pi0
**Blog:** https://huggingface.co/blog/pi0

---

### Priority 4: NVIDIA GR00T N1.5 ⭐⭐⭐

**Model:** NVIDIA Isaac GR00T N1.5

**Key Statistics:**
- **Type:** Cross-embodiment foundation model
- **Capabilities:** Multimodal inputs (language, images), diverse manipulation tasks
- **SO-101 Support:** Requires post-training (documented guide available)

**Why This Model:**
- Cutting-edge cross-embodiment foundation model
- Proven adaptability through post-training
- Official HuggingFace guide for SO-101

**Considerations:**
- Requires post-training setup (~25GB VRAM)
- More complex than other options
- Best for advanced experimentation after initial MVP

**How to Use:**
See official guide: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning

**Key Steps:**
1. Install NVIDIA Isaac Lab
2. Prepare SO-101 calibration data
3. Run post-training script
4. Export to LeRobot format

---

### Priority 5: OpenVLA with OFT ⭐⭐⭐

**Model:** OpenVLA with Optimal Fine-Tuning (OFT)

**Key Statistics:**
- **Base Parameters:** 7B
- **Optimization:** OFT enables 25-50x faster inference
- **Performance:** Higher success rates than vanilla OpenVLA

**Why This Model:**
- Latest OFT recipe (March 2025) dramatically improves speed
- Supports multiple input images
- High-frequency bimanual control

**Considerations:**
- Requires fine-tuning for SO-101
- Larger model size (7B)
- More setup complexity

**How to Use:**
```bash
# Clone OpenVLA-OFT repository
git clone https://github.com/moojink/openvla-oft.git

# Fine-tune on SO-101 dataset
python train_oft.py \
  --dataset=lerobot/svla_so101_pickplace \
  --base_model=openvla/openvla-7b \
  --output_dir=outputs/openvla_oft_so101
```

**Repository:** https://github.com/moojink/openvla-oft

---

## Latest Small VLM Models (2025)

For Stage 3 hierarchical control, we need a high-level VLM for task planning. These are the state-of-the-art small VLM models as of 2025.

---

### Tier 1: Recommended for Robotics

#### Option 1: Qwen3-VL-4B / 8B (RECOMMENDED) ⭐⭐⭐⭐⭐

**Model:** `Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-8B-Instruct`

**Release Date:** October 15, 2025

**Key Features:**
- **Parameters:** 4B or 8B (both have Instruct and Thinking variants)
- **Context Window:** 256K tokens (expandable to 1M)
- **Modalities:** Images, videos, text
- **Languages:** 32 languages
- **Quantization:** FP8 fine-grained quantization available (block size 128)
- **Robotics Testing:** Evaluated on robotic control and mobile operations

**Why This Model:**
- Latest state-of-the-art small VLM (Oct 2025)
- Proven robotics and agentic capabilities
- 3D spatial understanding for manipulation
- Excellent performance-to-size ratio
- FP8 quantization nearly matches BF16 performance

**Robotics Applications:**
- Visual instruction following for robots
- 3D spatial reasoning for object localization
- Multi-step task decomposition
- Dynamic tool/action planning

**How to Use:**
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

# Load model with BF16
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# Generate subgoal for robot
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": robot_camera_image},
            {"type": "text", "text": "The task is to clear the table. What should the robot do first? Provide specific action."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[robot_camera_image], return_tensors="pt").to("cuda:0")

output = vlm.generate(**inputs, max_new_tokens=100)
subgoal = processor.decode(output[0], skip_special_tokens=True)
# Returns: "Grasp the red cup at position (0.3, 0.2, 0.05) using the right arm"
```

**Use FP8 Quantization:**
```python
# Load FP8 quantized model for faster inference
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    quantization_config=quantization_config
)
```

**Repository:** https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
**GitHub:** https://github.com/QwenLM/Qwen3-VL

---

#### Option 2: Molmo-7B / MolmoE-1B ⭐⭐⭐⭐⭐

**Models:**
- `allenai/Molmo-7B-D-0924` (7B parameters)
- `allenai/MolmoE-1B-0924` (1.5B active, 7.2B total - MoE)

**Key Features:**
- **Unique Capability:** Pointing - outputs pixel coordinates in images
- **Performance:** Molmo-7B performs between GPT-4V and GPT-4o
- **Efficiency:** MolmoE-1B nearly matches GPT-4V with 1.5B active params
- **License:** Fully open-source (Apache 2.0)
- **Training Data:** PixMo collection (fully open)

**Why This Model:**
- **Revolutionary for robotics:** First open VLM optimized for pointing
- Can output exact pixel coordinates for object locations
- Perfect for "pick up object at position (x, y)"
- Excellent spatial grounding capabilities
- Outstanding performance on RealWorldQA

**Robotics Applications:**
- Waypoint generation for navigation
- Object localization for grasping
- Visual grounding for manipulation
- GUI interaction and pointing

**How to Use:**
```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# Load Molmo-7B
vlm = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

# Generate pointing coordinates
inputs = processor.process(
    images=[robot_camera_image],
    text="Point to the red cube that should be picked up."
)

output = vlm.generate_from_batch(
    inputs,
    max_new_tokens=200,
    temperature=0.0
)

# Parse output to extract coordinates
response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
# Returns: "The red cube is at <point x=\"320\" y=\"240\"/> which the robot should grasp."
```

**Pointing for Robot Grasping:**
```python
def get_grasp_target(vlm, processor, image, query):
    inputs = processor.process(
        images=[image],
        text=f"Point to {query} and describe its 3D position."
    )

    output = vlm.generate_from_batch(inputs, max_new_tokens=100)
    response = processor.tokenizer.decode(output[0])

    # Extract pixel coordinates from <point> tags
    import re
    match = re.search(r'<point x="(\d+)" y="(\d+)"/>', response)
    if match:
        pixel_x, pixel_y = int(match.group(1)), int(match.group(2))
        # Convert to 3D using depth camera
        world_pos = pixel_to_world(pixel_x, pixel_y, depth_image)
        return world_pos
    return None
```

**Repository:** https://huggingface.co/allenai/Molmo-7B-D-0924
**Paper:** https://arxiv.org/abs/2409.17146

---

#### Option 3: Qwen2.5-VL-3B / 7B ⭐⭐⭐⭐

**Model:** `Qwen/Qwen2.5-VL-3B-Instruct` or `Qwen/Qwen2.5-VL-7B-Instruct`

**Key Features:**
- **Parameters:** 3B or 7B
- **Context:** Up to 32K tokens
- **Agentic Capabilities:** Computer use, phone use
- **Task Range:** Localization, document understanding, agentic tasks

**Why This Model:**
- Proven agentic capabilities (can use tools dynamically)
- Strong balance between performance and efficiency
- Can play as visual agent without task-specific fine-tuning
- Excellent for multi-step reasoning

**Robotics Applications:**
- Dynamic tool selection for manipulation
- Multi-step task planning
- Visual reasoning for complex scenarios
- Integration with external APIs/tools

**How to Use:**
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Plan multi-step task
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": table_scene},
        {"type": "text", "text": "Plan a sequence of actions to clear this table and organize items."}
    ]
}]

inputs = processor(text=processor.apply_chat_template(messages), images=[table_scene])
output = vlm.generate(**inputs, max_new_tokens=200)

plan = processor.decode(output[0])
# Returns step-by-step plan
```

**Repository:** https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

---

### Tier 2: Lightweight Alternatives

#### SmolVLM2-2.2B ⭐⭐⭐⭐

**Model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct`

**Key Features:**
- **Parameters:** 2.2B
- **GPU RAM:** Only 5.2GB for video inference
- **Capabilities:** Video understanding, images, text
- **Performance:** Robust despite compact size

**Why This Model:**
- Extremely lightweight (can run on laptop GPUs)
- Video understanding for temporal reasoning
- Fully open-source (Apache 2.0)

**How to Use:**
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

vlm = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")

# Video understanding for temporal tasks
inputs = processor(videos=[robot_demo_video], text="Describe the manipulation sequence.")
output = vlm.generate(**inputs)
```

**Repository:** https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct

---

#### Phi-4 Multimodal ⭐⭐⭐

**Key Features:**
- Small but powerful multimodal model
- Unified vision, audio, text processing
- Perfect for mobile AI and edge devices

**Robotics Applications:**
- On-device inference for edge robotics
- Multimodal understanding (vision + audio commands)
- Offline operation scenarios

**Repository:** https://huggingface.co/microsoft/phi-4

---

#### Pixtral ⭐⭐⭐

**Key Features:**
- Ultra-lightweight architecture
- Great visual reasoning
- Ideal for resource-constrained devices

**Robotics Applications:**
- Drones, smart cameras, wearable devices
- Fast visual reasoning for reactive control

---

### Tier 3: Specialized Options

#### MiniCPM-o 2.6 (8B)

**Key Features:**
- 8B multimodal model
- Vision + Speech + Language modalities
- Strong performance for size

---

#### InternVL2.5

**Key Features:**
- Smaller variants available (1B, 2B)
- Video input support
- Qwen2.5 text backbone

---

## VLM Selection Decision Tree

```
┌─ Need POINTING capability (pixel coordinates)?
│  ├─ YES → Use Molmo-7B or MolmoE-1B
│  └─ NO ↓
│
├─ Need latest SOTA performance (2025)?
│  ├─ YES → Use Qwen3-VL-4B
│  └─ NO ↓
│
├─ Need agentic tool use?
│  ├─ YES → Use Qwen2.5-VL-3B/7B
│  └─ NO ↓
│
├─ GPU memory constrained (< 8GB)?
│  ├─ YES → Use SmolVLM2-2.2B
│  └─ NO ↓
│
└─ Default → Use Qwen3-VL-4B (best overall)
```

---

## Stage 1: Hardware Verification (MVP Testing)

### Goal
Confirm SO-101 arms are assembled correctly and respond to commands before investing time in models.

### Prerequisites
- SO-101 follower arms assembled
- USB cables connected to laptop
- Cameras installed (Intel RealSense or compatible)
- Ubuntu/Linux environment

### Step-by-Step Execution

#### 1.1 Install LeRobot with SO-101 Support

```bash
# Create conda environment
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install ffmpeg
conda install ffmpeg -c conda-forge

# Clone LeRobot repository
cd ~/project
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install with Feetech motor support (for SO-101)
pip install -e ".[feetech,intelrealsense]"
```

**Expected output:** No errors, all dependencies installed

**Common issues:**
- Build errors → Install `cmake`, `build-essential`, `python3-dev`
- FFmpeg issues → Ensure ffmpeg 7.x with libsvtav1 support

---

#### 1.2 Identify USB Ports

```bash
# List all USB serial ports
ls -l /dev/ttyACM* /dev/ttyUSB*

# Should see something like:
# /dev/ttyACM0  (left arm)
# /dev/ttyACM1  (right arm)
```

**If no devices found:**
```bash
# Check user permissions
sudo usermod -a -G dialout $USER
# Log out and log back in
```

---

#### 1.3 Test Motor Communication

```bash
# Find motor ports
python -m lerobot.scripts.lerobot_find_port

# Expected output:
# Found Feetech motor bus:
#   - /dev/ttyACM0
#   - /dev/ttyACM1
```

---

#### 1.4 Calibrate SO-101 Arms

**Calibration is CRITICAL - do this before any operation.**

```bash
# Calibrate left arm
python -m lerobot.scripts.lerobot_calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_left_arm

# Calibrate right arm
python -m lerobot.scripts.lerobot_calibrate \
  --robot.port=/dev/ttyACM1 \
  --robot.id=xlerobot_right_arm
```

**Calibration Process:**
1. Prompt: "Move robot to middle of range, then press Enter"
   - Manually move each joint to middle position
   - Press Enter
2. Prompt: "Move each joint through full range of motion"
   - Slowly move joint 1 from min to max
   - Slowly move joint 2 from min to max
   - ... repeat for all 6 joints + gripper
3. Calibration saved to `~/.cache/lerobot/calibration/xlerobot_left_arm.json`

**Verify calibration:**
```bash
cat ~/.cache/lerobot/calibration/xlerobot_left_arm.json
```

Expected structure:
```json
{
  "homing_offset": [0, 0, 0, 0, 0, 0, 0],
  "drive_mode": [0, 0, 0, 0, 0, 0, 0],
  "joint_ranges": {
    "shoulder_pan": [-180, 180],
    "shoulder_lift": [-90, 90],
    ...
  }
}
```

---

#### 1.5 Test Basic Keyboard Control

```bash
# Test teleoperation with keyboard
python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_left_arm
```

**Controls:**
- Arrow keys: Move joints
- Space: Stop
- 'q': Quit

**What to verify:**
- Arms move smoothly without jerking
- Joints respect limits (don't overextend)
- Gripper opens and closes
- No motor overheating
- No USB disconnections

---

#### 1.6 Test Camera Setup

```bash
# List available cameras
ls /dev/video*

# Test camera with LeRobot
python -c "
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

camera = IntelRealSenseCamera(fps=30, width=640, height=480)
image = camera.async_read()
print(f'Camera working! Image shape: {image.shape}')
"
```

**Expected output:** `Camera working! Image shape: (480, 640, 3)`

---

#### 1.7 Record Test Episode

```bash
# Record 5 test episodes with keyboard control
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_test \
  --dataset.repo_id=${HF_USER}/so101_hardware_test \
  --dataset.num_episodes=5 \
  --dataset.fps=30
```

**What this tests:**
- Data recording pipeline works
- Camera synchronization correct
- Joint state logging accurate
- Video encoding functional

---

### Stage 1 Success Criteria

✅ Arms calibrated and move smoothly
✅ No USB communication errors
✅ Cameras capture images at 30 FPS
✅ Test episodes recorded successfully
✅ No motor overheating or mechanical issues

**Time Estimate:** 2-4 hours

---

## Stage 2: VLA Inference with Pretrained Models

### Goal
Get a working VLA policy running inference on your SO-101 hardware using pretrained models - **without any training or data collection.**

### Strategy
Test models in priority order until achieving 50%+ success rate on pick-and-place task.

---

### Priority 1: SmolVLA (Start Here)

#### 2.1 Test SmolVLA Base Model

```python
# test_smolvla.py
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.common.robot_devices.robots.so101_follower import SO101FollowerRobot
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
import torch
import time

def test_smolvla_inference():
    # Initialize robot
    robot = SO101FollowerRobot(
        port="/dev/ttyACM0",
        config_path="so101_follower"
    )

    # Initialize cameras
    camera_top = IntelRealSenseCamera(serial="top_camera", fps=30)
    camera_wrist = IntelRealSenseCamera(serial="wrist_camera", fps=30)

    # Load pretrained SmolVLA
    print("Loading SmolVLA model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to("cuda:0")
    policy.eval()

    # Run inference loop
    print("Starting inference loop at 30Hz...")
    task = "pick up the red cube"

    for step in range(300):  # 10 seconds at 30Hz
        start_time = time.time()

        # Get observations
        obs = {
            "observation.images.top": camera_top.async_read(),      # (3, 480, 640)
            "observation.images.wrist": camera_wrist.async_read(),  # (3, 240, 320)
            "observation.state": robot.get_joint_positions(),       # (6,)
            "task": task
        }

        # Convert to tensors
        obs = {
            "observation.images.top": torch.from_numpy(obs["observation.images.top"]).float().unsqueeze(0).to("cuda:0"),
            "observation.images.wrist": torch.from_numpy(obs["observation.images.wrist"]).float().unsqueeze(0).to("cuda:0"),
            "observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to("cuda:0"),
            "task": task
        }

        # Predict action
        with torch.no_grad():
            action = policy.select_action(obs)

        # Send to robot
        robot.send_action(action.cpu().numpy()[0])

        # Maintain 30Hz
        elapsed = time.time() - start_time
        if elapsed < 0.033:
            time.sleep(0.033 - elapsed)

        if step % 30 == 0:
            print(f"Step {step}, frequency: {1.0/elapsed:.1f}Hz")

    print("Inference complete!")
    robot.disconnect()

if __name__ == "__main__":
    test_smolvla_inference()
```

Run the test:
```bash
python test_smolvla.py
```

**Expected behavior:**
- Model loads successfully (~2-3GB GPU memory)
- Inference runs at 25-30Hz
- Robot moves toward objects based on language command
- Smooth trajectories, no jerky motion

---

#### 2.2 Evaluate on Pick-and-Place Task

Create a simple pick-and-place test scenario:

**Setup:**
- Place a bright red cube at fixed position (0.3m in front of robot)
- Mark target position with tape (0.3m away from cube)
- Ensure consistent lighting

**Evaluation Script:**
```python
# evaluate_pickplace.py
import numpy as np
from test_smolvla import test_smolvla_inference

def evaluate_pickplace(num_trials=10):
    """Run 10 trials and calculate success rate"""
    successes = 0

    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        print("Place red cube at start position and press Enter...")
        input()

        # Run inference
        test_smolvla_inference()

        # Ask user for success evaluation
        result = input("Did robot successfully pick and place cube? (y/n): ")
        if result.lower() == 'y':
            successes += 1

        print(f"Current success rate: {successes}/{trial+1} = {100*successes/(trial+1):.1f}%")

    print(f"\n=== Final Results ===")
    print(f"Success rate: {successes}/{num_trials} = {100*successes/num_trials:.1f}%")

    return successes / num_trials

if __name__ == "__main__":
    success_rate = evaluate_pickplace(num_trials=10)

    if success_rate >= 0.5:
        print("✅ Stage 2 SUCCESS! SmolVLA achieves 50%+ success rate.")
        print("Ready to proceed to Stage 3.")
    else:
        print("⚠️  SmolVLA success rate below 50%. Try next model or collect custom data.")
```

Run evaluation:
```bash
python evaluate_pickplace.py
```

---

### Priority 2: ACT Model (If SmolVLA < 50%)

```python
# test_act.py
from lerobot.policies.act import ACTPolicy
from lerobot.common.robot_devices.robots.so101_follower import SO101FollowerRobot
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
import torch
import time

def test_act_inference():
    # Initialize robot and cameras (same as SmolVLA)
    robot = SO101FollowerRobot(port="/dev/ttyACM0")
    camera_top = IntelRealSenseCamera(serial="top", fps=30, width=640, height=480)
    camera_wrist = IntelRealSenseCamera(serial="wrist", fps=30, width=320, height=240)

    # Load ACT model
    print("Loading ACT model...")
    policy = ACTPolicy.from_pretrained("r2owb0/act1")
    policy = policy.to("cuda:0")
    policy.eval()

    # Run inference at 50Hz
    print("Starting inference at 50Hz...")
    for step in range(500):  # 10 seconds
        start = time.time()

        # Get observations (no language conditioning)
        obs = {
            "observation.images.top": camera_top.async_read(),
            "observation.images.wrist": camera_wrist.async_read(),
            "observation.state": robot.get_joint_positions()
        }

        # Predict action
        with torch.no_grad():
            action = policy.select_action(obs)

        robot.send_action(action)

        # Maintain 50Hz
        elapsed = time.time() - start
        if elapsed < 0.02:
            time.sleep(0.02 - elapsed)

    robot.disconnect()

if __name__ == "__main__":
    test_act_inference()
```

**Note:** ACT model has no language conditioning, so it will simply repeat the behavior from its training data. Test on similar pick-and-place scenarios.

---

### Priority 3: Pi0 Foundation Model

```python
# test_pi0.py
from lerobot.policies.pi0 import Pi0Policy

# Similar to SmolVLA but larger model
policy = Pi0Policy.from_pretrained("lerobot/pi0")
policy = policy.to("cuda:0")

# Pi0 may be slower due to 4B parameters
# Expect 5-10Hz inference frequency
```

**When to use:** If SmolVLA and ACT both fail, Pi0's cross-embodiment training may generalize better. However, fine-tuning is recommended.

---

### Decision Tree: Which Model Worked?

```
SmolVLA success rate?
├─ ≥ 70% → ✅ EXCELLENT! Proceed to Stage 3a
├─ 50-69% → ✅ GOOD! Proceed to Stage 3a, consider fine-tuning later
├─ 40-49% → ⚠️  Try ACT model (r2owb0/act1)
└─ < 40% → ⚠️  Try Pi0, or proceed to Fallback (collect custom data)

ACT success rate?
├─ ≥ 50% → ✅ Proceed to Stage 3a (limited language support)
└─ < 50% → Proceed to Fallback (custom data collection needed)
```

---

### Stage 2 Success Criteria

✅ At least one model achieves 50%+ success rate
✅ Inference runs at target frequency (25-50Hz)
✅ No GPU memory errors
✅ Robot completes coherent motions (even if not perfect)

**Time Estimate:** 1-2 days

---

## Stage 3a: Natural Language Control for Simple Tasks

### Goal
Enable robot to respond to natural language commands like "pick up the red cube" or "move to the left" by integrating a high-level VLM planner with the low-level VLA controller.

### Architecture Overview

```
User: "pick up the red cube"
      ↓
┌─────────────────────────────┐
│  High-Level VLM             │
│  (Qwen3-VL-4B @ 1Hz)        │ ← Runs every 1 second
│  - Understands scene         │
│  - Generates subgoal         │
└─────────────────────────────┘
      ↓ subgoal = "grasp red cube at (0.3, 0.2, 0.05)"
┌─────────────────────────────┐
│  Low-Level VLA              │
│  (SmolVLA @ 30Hz)           │ ← Runs continuously
│  - Executes subgoal          │
│  - Generates joint actions   │
└─────────────────────────────┘
      ↓ actions
   Robot Hardware
```

---

### Implementation: Asynchronous Multi-Rate Control

```python
# hierarchical_controller.py
import torch
import threading
import queue
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.common.robot_devices.robots.so101_follower import SO101FollowerRobot
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

class HierarchicalController:
    """
    Asynchronous hierarchical controller with:
    - VLM running at 1Hz (high-level planning)
    - VLA running at 30Hz (low-level control)
    """

    def __init__(self, robot_port="/dev/ttyACM0"):
        # Initialize hardware
        print("Initializing robot and cameras...")
        self.robot = SO101FollowerRobot(port=robot_port)
        self.camera_top = IntelRealSenseCamera(serial="top", fps=30)
        self.camera_wrist = IntelRealSenseCamera(serial="wrist", fps=30)

        # Load VLM (Qwen3-VL-4B)
        print("Loading VLM (Qwen3-VL-4B)...")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        # Load VLA (SmolVLA)
        print("Loading VLA (SmolVLA)...")
        self.vla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        self.vla = self.vla.to("cuda:0")
        self.vla.eval()

        # Shared state (thread-safe)
        self.current_subgoal = queue.Queue(maxsize=1)
        self.current_instruction = "wait"
        self.stop_flag = threading.Event()

        print("Hierarchical controller initialized!")

    def vlm_planning_thread(self):
        """VLM thread: Generate high-level subgoals at 1Hz"""
        print("VLM planning thread started (1Hz)")

        while not self.stop_flag.is_set():
            start_time = time.time()

            # Get current scene
            image = self.camera_top.async_read()
            robot_state = self.robot.get_joint_positions()

            # Generate subgoal with VLM
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"""
Task: {self.current_instruction}

Current robot state: {robot_state}

What should the robot do next? Provide a specific, actionable subgoal in 1-2 sentences.
Focus on:
1. Which object to interact with
2. What action to perform (grasp, move, place)
3. Approximate position/direction if relevant

Subgoal:"""}
                ]
            }]

            # Generate subgoal
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.vlm_processor(text=text, images=[image], return_tensors="pt").to("cuda:0")

            output = self.vlm.generate(**inputs, max_new_tokens=100, temperature=0.0)
            subgoal = self.vlm_processor.decode(output[0], skip_special_tokens=True)

            # Extract only the generated part (after "Subgoal:")
            if "Subgoal:" in subgoal:
                subgoal = subgoal.split("Subgoal:")[-1].strip()

            print(f"[VLM @ {time.time():.1f}s] New subgoal: {subgoal}")

            # Update subgoal queue (non-blocking)
            try:
                self.current_subgoal.put_nowait(subgoal)
            except queue.Full:
                pass  # VLA hasn't consumed previous subgoal yet

            # Maintain 1Hz
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 - elapsed)
            time.sleep(sleep_time)

    def vla_control_thread(self):
        """VLA thread: Execute low-level control at 30Hz"""
        print("VLA control thread started (30Hz)")

        current_subgoal = "wait for instruction"

        while not self.stop_flag.is_set():
            start_time = time.time()

            # Check for new subgoal (non-blocking)
            try:
                current_subgoal = self.current_subgoal.get_nowait()
                print(f"[VLA] Updated subgoal: {current_subgoal}")
            except queue.Empty:
                pass  # Continue with current subgoal

            # Get observations
            obs = {
                "observation.images.top": self.camera_top.async_read(),
                "observation.images.wrist": self.camera_wrist.async_read(),
                "observation.state": self.robot.get_joint_positions(),
                "task": current_subgoal  # Use VLM subgoal as language condition
            }

            # Convert to tensors
            obs_tensor = {
                "observation.images.top": torch.from_numpy(obs["observation.images.top"]).float().unsqueeze(0).to("cuda:0"),
                "observation.images.wrist": torch.from_numpy(obs["observation.images.wrist"]).float().unsqueeze(0).to("cuda:0"),
                "observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to("cuda:0"),
                "task": obs["task"]
            }

            # Predict action
            with torch.no_grad():
                action = self.vla.select_action(obs_tensor)

            # Send to robot
            self.robot.send_action(action.cpu().numpy()[0])

            # Maintain 30Hz
            elapsed = time.time() - start_time
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    def run(self, instruction, duration=30):
        """
        Run hierarchical control for specified duration

        Args:
            instruction: Natural language instruction (e.g., "pick up the red cube")
            duration: How long to run (seconds)
        """
        self.current_instruction = instruction
        print(f"\n{'='*60}")
        print(f"Starting hierarchical control for {duration} seconds")
        print(f"Instruction: {instruction}")
        print(f"{'='*60}\n")

        # Start threads
        vlm_thread = threading.Thread(target=self.vlm_planning_thread, daemon=True)
        vla_thread = threading.Thread(target=self.vla_control_thread, daemon=True)

        vlm_thread.start()
        vla_thread.start()

        # Run for specified duration
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\nInterrupted by user")

        # Stop threads
        self.stop_flag.set()
        vlm_thread.join(timeout=2)
        vla_thread.join(timeout=2)

        print("\nHierarchical control stopped")

    def shutdown(self):
        """Clean shutdown"""
        self.stop_flag.set()
        self.robot.disconnect()
        print("Controller shutdown complete")


# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    # Create hierarchical controller
    controller = HierarchicalController(robot_port="/dev/ttyACM0")

    # Test different instructions
    instructions = [
        "pick up the red cube and place it on the left",
        "grasp the blue cylinder",
        "move the gripper to the right side of the table"
    ]

    for instruction in instructions:
        print(f"\n\n{'#'*60}")
        print(f"Testing: {instruction}")
        print(f"{'#'*60}")
        input("Press Enter to start...")

        # Run for 30 seconds
        controller.run(instruction, duration=30)

        # Ask for success evaluation
        success = input("Was the instruction completed successfully? (y/n): ")
        print(f"Result: {'SUCCESS' if success.lower() == 'y' else 'FAILURE'}")

    # Shutdown
    controller.shutdown()
```

---

### Alternative VLM: Molmo (with Pointing)

If you want to use Molmo's unique pointing capability:

```python
# Use Molmo instead of Qwen3-VL for pixel-level pointing
from transformers import AutoModelForCausalLM, AutoProcessor

self.vlm = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
self.vlm_processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

# Generate pointing coordinates
inputs = self.vlm_processor.process(
    images=[image],
    text="Point to the object that should be picked up next."
)
output = self.vlm.generate_from_batch(inputs, max_new_tokens=100)

# Extract pixel coordinates from output
response = self.vlm_processor.tokenizer.decode(output[0])
# Parse <point x="..." y="..."/> tags
```

---

### Testing Natural Language Control

```python
# test_language_control.py
from hierarchical_controller import HierarchicalController

def test_language_commands():
    """Test suite for natural language control"""

    controller = HierarchicalController(robot_port="/dev/ttyACM0")

    test_cases = [
        {
            "instruction": "pick up the red cube",
            "setup": "Place red cube at (0.3, 0, 0)",
            "success_criteria": "Robot grasps red cube"
        },
        {
            "instruction": "move the gripper to the left",
            "setup": "No setup required",
            "success_criteria": "Gripper moves leftward"
        },
        {
            "instruction": "place the object in the blue container",
            "setup": "Robot holding object, blue container visible",
            "success_criteria": "Object placed in container"
        }
    ]

    results = []

    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(test_cases)}")
        print(f"Instruction: {test['instruction']}")
        print(f"Setup: {test['setup']}")
        print(f"Success criteria: {test['success_criteria']}")
        print(f"{'='*60}")

        input("Complete setup and press Enter...")

        # Run control
        controller.run(test['instruction'], duration=20)

        # Evaluate
        success = input(f"Did robot satisfy '{test['success_criteria']}'? (y/n): ")
        results.append(success.lower() == 'y')

    # Calculate success rate
    success_rate = sum(results) / len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {sum(results)}/{len(results)} = {100*success_rate:.1f}% success rate")
    print(f"{'='*60}")

    controller.shutdown()

    return success_rate

if __name__ == "__main__":
    success_rate = test_language_commands()

    if success_rate >= 0.7:
        print("✅ Stage 3a SUCCESS! Ready for Stage 3b (complex tasks)")
    else:
        print("⚠️  Consider fine-tuning VLM or VLA for better performance")
```

Run test:
```bash
python test_language_control.py
```

---

### Stage 3a Success Criteria

✅ VLM generates relevant subgoals for user instructions
✅ VLA executes subgoals correctly 70%+ of the time
✅ System runs stably for 30+ seconds without crashes
✅ Multi-threaded control maintains target frequencies (VLM 1Hz, VLA 30Hz)

**Time Estimate:** 3-5 days

---

## Stage 3b: Complex Multi-Step Task Planning

### Goal
Enable robot to decompose complex, multi-step tasks into simpler subtasks and execute them sequentially.

### Example Complex Tasks
1. "Clear the table and put items in the drawer"
   - Identify all items on table
   - Grasp first item
   - Navigate to drawer
   - Open drawer
   - Place item inside
   - Close drawer
   - Return to table
   - Repeat for remaining items

2. "Make a sandwich"
   - Grasp bread slice
   - Place on plate
   - Grasp cheese
   - Place on bread
   - Grasp second bread slice
   - Place on top

3. "Organize the desk by color"
   - Identify objects and colors
   - Grasp red objects → place in red zone
   - Grasp blue objects → place in blue zone
   - Continue until organized

---

### Implementation: Task Decomposition Pipeline

```python
# complex_task_planner.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.common.robot_devices.robots.so101_follower import SO101FollowerRobot
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
import time
import json

class ComplexTaskPlanner:
    """
    Complex task planner that:
    1. Decomposes high-level task into subtasks (VLM)
    2. Executes each subtask sequentially (VLA)
    3. Monitors progress and adapts plan if needed
    """

    def __init__(self, robot_port="/dev/ttyACM0"):
        # Initialize hardware
        print("Initializing robot and cameras...")
        self.robot = SO101FollowerRobot(port=robot_port)
        self.camera_top = IntelRealSenseCamera(serial="top", fps=30)
        self.camera_wrist = IntelRealSenseCamera(serial="wrist", fps=30)

        # Load VLM for planning
        print("Loading VLM (Qwen3-VL-4B) for task decomposition...")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        # Load VLA for execution
        print("Loading VLA (SmolVLA) for action execution...")
        self.vla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        self.vla = self.vla.to("cuda:0")
        self.vla.eval()

        print("Complex task planner initialized!")

    def decompose_task(self, image, high_level_task):
        """
        Use VLM to decompose complex task into subtasks

        Args:
            image: Current scene image (numpy array)
            high_level_task: Complex task description (string)

        Returns:
            List of subtasks (strings)
        """
        print(f"\n[PLANNING] Decomposing task: {high_level_task}")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"""
Task: {high_level_task}

Please decompose this complex task into a sequence of simple, executable subtasks for a robot arm.

Requirements:
1. Each subtask should be a single, atomic action (grasp, move, place, etc.)
2. Subtasks should be in correct execution order
3. Each subtask should be 1 sentence
4. Be specific about objects and locations

Output format (JSON):
{{
  "subtasks": [
    "subtask 1 description",
    "subtask 2 description",
    ...
  ]
}}

JSON:"""}
            ]
        }]

        # Generate task plan
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.vlm_processor(text=text, images=[image], return_tensors="pt").to("cuda:0")

        output = self.vlm.generate(**inputs, max_new_tokens=500, temperature=0.0)
        response = self.vlm_processor.decode(output[0], skip_special_tokens=True)

        # Parse JSON output
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            plan = json.loads(json_str)
            subtasks = plan["subtasks"]

            print(f"[PLANNING] Generated {len(subtasks)} subtasks:")
            for i, subtask in enumerate(subtasks):
                print(f"  {i+1}. {subtask}")

            return subtasks

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse VLM output as JSON: {e}")
            print(f"Response: {response}")
            return []

    def execute_subtask(self, subtask, duration=15):
        """
        Execute a single subtask using VLA

        Args:
            subtask: Subtask description (string)
            duration: Max execution time in seconds

        Returns:
            Success (bool)
        """
        print(f"\n[EXECUTING] Subtask: {subtask}")

        start_time = time.time()
        step_count = 0

        while time.time() - start_time < duration:
            # Get observations
            obs = {
                "observation.images.top": self.camera_top.async_read(),
                "observation.images.wrist": self.camera_wrist.async_read(),
                "observation.state": self.robot.get_joint_positions(),
                "task": subtask  # Use subtask as language condition
            }

            # Convert to tensors
            obs_tensor = {
                "observation.images.top": torch.from_numpy(obs["observation.images.top"]).float().unsqueeze(0).to("cuda:0"),
                "observation.images.wrist": torch.from_numpy(obs["observation.images.wrist"]).float().unsqueeze(0).to("cuda:0"),
                "observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to("cuda:0"),
                "task": obs["task"]
            }

            # Predict action
            with torch.no_grad():
                action = self.vla.select_action(obs_tensor)

            # Send to robot
            self.robot.send_action(action.cpu().numpy()[0])

            step_count += 1

            # Maintain 30Hz
            time.sleep(0.033)

        print(f"[EXECUTING] Subtask completed ({step_count} steps, {time.time()-start_time:.1f}s)")

        # Ask user for verification (in production, use vision-based verification)
        success = input("  Was subtask completed successfully? (y/n): ")
        return success.lower() == 'y'

    def verify_subtask_completion(self, image, subtask):
        """
        Use VLM to verify if subtask was completed successfully

        Args:
            image: Current scene after subtask execution
            subtask: Subtask that was executed

        Returns:
            verified (bool), reason (string)
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"""
The robot just attempted to complete this subtask:
"{subtask}"

Looking at the current scene, was this subtask completed successfully?

Answer with:
- "YES" if the subtask objective was achieved
- "NO" if the subtask failed or objective not met

Then provide a brief reason (1 sentence).

Response:"""}
            ]
        }]

        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.vlm_processor(text=text, images=[image], return_tensors="pt").to("cuda:0")

        output = self.vlm.generate(**inputs, max_new_tokens=100, temperature=0.0)
        response = self.vlm_processor.decode(output[0], skip_special_tokens=True)

        verified = "YES" in response.upper()

        return verified, response

    def execute_complex_task(self, high_level_task, max_retries=2):
        """
        Execute entire complex task: decompose + execute + verify

        Args:
            high_level_task: High-level task description
            max_retries: Max retries per subtask

        Returns:
            overall_success (bool), execution_log (dict)
        """
        print(f"\n{'='*60}")
        print(f"EXECUTING COMPLEX TASK: {high_level_task}")
        print(f"{'='*60}\n")

        # Step 1: Decompose task
        image = self.camera_top.async_read()
        subtasks = self.decompose_task(image, high_level_task)

        if not subtasks:
            print("[ERROR] Task decomposition failed")
            return False, {}

        # Execution log
        log = {
            "task": high_level_task,
            "subtasks": subtasks,
            "results": []
        }

        # Step 2: Execute subtasks sequentially
        for i, subtask in enumerate(subtasks):
            print(f"\n{'='*60}")
            print(f"SUBTASK {i+1}/{len(subtasks)}")
            print(f"{'='*60}")

            success = False
            attempts = 0

            while not success and attempts < max_retries:
                attempts += 1
                print(f"Attempt {attempts}/{max_retries}")

                # Execute subtask
                success = self.execute_subtask(subtask, duration=15)

                # Verify completion (optional: use VLM verification)
                if success:
                    image_after = self.camera_top.async_read()
                    verified, reason = self.verify_subtask_completion(image_after, subtask)
                    print(f"[VERIFICATION] {reason}")

                    if not verified:
                        print("[VERIFICATION] Failed, retrying...")
                        success = False

                if not success and attempts < max_retries:
                    print(f"[RETRY] Attempting subtask again...")
                    time.sleep(2)

            # Log result
            log["results"].append({
                "subtask": subtask,
                "success": success,
                "attempts": attempts
            })

            if not success:
                print(f"[FAILURE] Subtask {i+1} failed after {max_retries} attempts")
                print(f"[ABORT] Aborting complex task")
                return False, log

            print(f"[SUCCESS] Subtask {i+1} completed")
            time.sleep(1)  # Brief pause between subtasks

        # All subtasks completed
        print(f"\n{'='*60}")
        print(f"✅ COMPLEX TASK COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")

        return True, log

    def shutdown(self):
        """Clean shutdown"""
        self.robot.disconnect()
        print("Complex task planner shutdown complete")


# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    planner = ComplexTaskPlanner(robot_port="/dev/ttyACM0")

    # Test complex tasks
    complex_tasks = [
        "Pick up the red cube and place it on top of the blue cube",
        "Clear all objects from the table and place them in the container",
        "Stack three blocks: red on bottom, blue in middle, green on top"
    ]

    results = []

    for task in complex_tasks:
        print(f"\n\n{'#'*60}")
        print(f"TESTING: {task}")
        print(f"{'#'*60}")
        input("Set up scene and press Enter...")

        # Execute complex task
        success, log = planner.execute_complex_task(task, max_retries=2)
        results.append(success)

        # Print summary
        print(f"\n{'='*60}")
        print(f"TASK RESULT: {'SUCCESS' if success else 'FAILURE'}")
        print(f"Subtasks completed: {sum(r['success'] for r in log['results'])}/{len(log['results'])}")
        print(f"{'='*60}\n")

        input("Press Enter to continue to next task...")

    # Overall results
    success_rate = sum(results) / len(results)
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS: {sum(results)}/{len(results)} = {100*success_rate:.1f}% success")
    print(f"{'='*60}")

    planner.shutdown()
```

---

### Advanced: LoRA Fine-Tuning VLM for Robot Domain

If you want to adapt VLM to robot-specific vocabulary and scenarios:

```python
# finetune_vlm_lora.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load base VLM
base_vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype=torch.bfloat16
)

# Configure LoRA (5% trainable parameters)
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA adapters
vlm_with_lora = get_peft_model(base_vlm, lora_config)

# Check trainable parameters
trainable = sum(p.numel() for p in vlm_with_lora.parameters() if p.requires_grad)
total = sum(p.numel() for p in vlm_with_lora.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
# Output: ~5% trainable

# Fine-tune on robot-specific data
# (Create dataset of robot task descriptions + images + desired outputs)
# See: https://huggingface.co/docs/transformers/training
```

Benefits of LoRA:
- Preserves VLM's world knowledge (90%+ params frozen)
- Fast training (only 5% params)
- Small checkpoint (~200MB vs 8GB full model)
- Can train multiple task-specific adapters

---

### Stage 3b Success Criteria

✅ VLM successfully decomposes 2-3 step tasks
✅ Robot executes 60%+ of multi-step tasks successfully
✅ Failed subtasks can be retried with reasonable success
✅ System runs stably for full task duration (30-60 seconds)

**Time Estimate:** 5-7 days

---

## Fallback: Custom Data Collection (ONLY IF NEEDED)

### When to Use This Section

Proceed to custom data collection ONLY if:
1. All pretrained models achieve < 40% success rate, OR
2. You need task-specific behaviors not in pretrained models, OR
3. You want to improve from 70% → 85%+ success

**Cost:** ~$350 for SO-101 Leader Arm + ~2-3 days effort

---

### Hardware Required: SO-101 Leader Arm

**Purchase:**
- **Vendor:** WowRobo (https://shop.wowrobo.com/products/so-arm101-leader)
- **Price:** ~$350 USD
- **Assembly Time:** 1-2 hours
- **Alternative:** GELLO (~$300, requires 3D printing)

**Why SO-101 Leader:**
- Kinematically matched to SO-101 follower (ensures feasible demonstrations)
- Native LeRobot support (zero custom code)
- Automatic calibration
- Direct recording to HuggingFace dataset format

---

### Data Collection Process

#### Step 1: Order and Assemble Leader Arm

```bash
# 1. Order SO-101 Leader from WowRobo
# Expected delivery: 1-2 weeks

# 2. Assemble using provided instructions
# Time: 1-2 hours

# 3. Connect to laptop via USB
ls /dev/ttyUSB*
# Should see /dev/ttyUSB0 (leader)
```

---

#### Step 2: Calibrate Leader Arm

```bash
# Calibrate leader (same process as follower)
python -m lerobot.scripts.lerobot_calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=xlerobot_leader
```

---

#### Step 3: Test Teleoperation

```bash
# Test leader-follower connection
python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_left_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=xlerobot_leader
```

**Expected:** Follower mirrors leader motions in real-time

---

#### Step 4: Collect Demonstrations

**Task Definition:**
- Task: Pick red cube at (0.3, 0, 0) and place at (0.3, 0.2, 0)
- Episode duration: ~10-15 seconds
- Target: 50-100 demonstrations

**Collection Script:**
```bash
# Start data collection
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=xlerobot_leader \
  --dataset.repo_id=${HF_USER}/xlerobot_pickplace_custom \
  --dataset.fps=30 \
  --dataset.num_episodes=50 \
  --task="pick up red cube and place at target"
```

**Tips for High-Quality Demonstrations:**
1. **Consistency:** Reset cube to exact same position each time
2. **Smooth motions:** Avoid jerky movements
3. **Success:** Only keep successful demonstrations (delete failed ones)
4. **Lighting:** Keep consistent lighting across episodes
5. **Breaks:** Take 5min breaks every 10 demos to maintain quality

---

#### Step 5: Verify Dataset

```bash
# Visualize dataset
python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id=${HF_USER}/xlerobot_pickplace_custom \
  --episode-index=0

# Should open Rerun viewer showing:
# - Camera streams
# - Robot joint positions
# - Actions (leader commands)
```

---

#### Step 6: Train Policy (Fine-tune SmolVLA)

```bash
# Fine-tune SmolVLA on your custom data
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/xlerobot_pickplace_custom \
  --training.num_epochs=3000 \
  --training.batch_size=16 \
  --training.learning_rate=1e-4 \
  --training.save_checkpoint_every_n_epochs=500 \
  --output_dir=outputs/train/smolvla_custom_task
```

**Training Time:** ~2-3 hours on RTX 5090 for 3000 epochs

**Monitor training:**
```bash
# Use wandb (optional)
wandb login
# Then add to train.py: --wandb.enable=true
```

---

#### Step 7: Evaluate Fine-tuned Model

```python
# evaluate_custom_model.py
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.common.robot_devices.robots.so101_follower import SO101FollowerRobot

# Load fine-tuned model
policy = SmolVLAPolicy.from_pretrained(
    "outputs/train/smolvla_custom_task/checkpoints/3000/pretrained_model"
)

robot = SO101FollowerRobot(port="/dev/ttyACM0")

# Run evaluation (10 trials)
success_count = 0
for trial in range(10):
    print(f"Trial {trial+1}/10")
    input("Reset scene and press Enter...")

    # Run inference for 10 seconds
    for step in range(300):  # 30Hz * 10s
        obs = get_observation(robot, cameras)
        action = policy.select_action(obs)
        robot.send_action(action)
        time.sleep(0.033)

    success = input("Success? (y/n): ")
    if success == 'y':
        success_count += 1

print(f"\nSuccess rate: {success_count}/10 = {success_count*10}%")
```

**Expected:** 70-85% success rate after fine-tuning

---

### Data Collection Cost-Benefit Analysis

| Approach | Cost | Time | Success Rate | When to Use |
|----------|------|------|--------------|-------------|
| Pretrained only | $0 | 2-3 days | 50-78% | First try, limited budget |
| Pretrained + 50 demos | $350 | 5-7 days | 70-85% | Need task-specific behavior |
| Pretrained + 200 demos | $350 | 10-14 days | 80-90% | Production-quality performance |

---

## Cost Analysis

### Hardware Costs

| Item | Cost (USD) | Required | Notes |
|------|-----------|----------|-------|
| SO-101 Follower Arms (2x) | $228 ($114 each) | ✅ Yes | Already owned |
| Laptop with RTX 5090 | $3000-4000 | ✅ Yes | Already owned |
| Intel RealSense D435 (2x) | $400 ($200 each) | ✅ Yes | Or compatible RGB cameras |
| SO-101 Leader Arm | $350 | ⚠️  Only if collecting data | Can skip with pretrained models |
| **Total (no data collection)** | **$0** (already owned) | | |
| **Total (with data collection)** | **$350** | | |

---

### Time Investment

| Stage | Scenario 1: Pretrained Works | Scenario 2: Need Data Collection |
|-------|------------------------------|----------------------------------|
| Stage 1 (Hardware setup) | 2-4 hours | 2-4 hours |
| Stage 2 (VLA inference) | 1-2 days | 1-2 days |
| Data collection | - | 2-3 days |
| Training | - | 2-3 hours (GPU time) |
| Stage 3a (Language control) | 3-5 days | 3-5 days |
| Stage 3b (Complex tasks) | 5-7 days | 5-7 days |
| **TOTAL** | **~10-14 days** | **~13-17 days** |

---

### GPU Compute Costs

| Model | GPU Memory | Inference Speed | Training Time | Cost |
|-------|-----------|----------------|---------------|------|
| SmolVLA | 2-3GB | 30Hz | - (pretrained) | $0 |
| Qwen3-VL-4B | 8GB | 1-5Hz | - (pretrained) | $0 |
| SmolVLA (fine-tune) | 8-12GB | 30Hz | 2-3 hours | $0 (local) |
| Pi0 (fine-tune) | 16-20GB | 10Hz | 6-8 hours | $0 (local) |

**All compute runs locally on RTX 5090 - no cloud costs.**

---

## Timeline & Milestones

### Week 1: Foundation
**Days 1-2:** Stage 1 - Hardware Verification
- ✅ Install LeRobot
- ✅ Calibrate arms
- ✅ Test basic control
- ✅ Verify cameras

**Days 3-5:** Stage 2 - VLA Inference
- ✅ Test SmolVLA pretrained model
- ✅ Evaluate on pick-and-place
- ✅ Achieve 50%+ success rate

**Decision Point:** If success < 40%, order SO-101 Leader for data collection (week 2-3 for delivery)

---

### Week 2: Language Control
**Days 6-10:** Stage 3a - Natural Language Control
- ✅ Implement hierarchical controller
- ✅ Integrate Qwen3-VL-4B or Molmo-7B
- ✅ Test 3+ language commands
- ✅ Achieve 70%+ success rate

---

### Week 3: Complex Tasks
**Days 11-17:** Stage 3b - Multi-Step Tasks
- ✅ Implement task decomposition
- ✅ Test 2-3 step tasks
- ✅ Implement retry logic
- ✅ Achieve 60%+ success on complex tasks

---

### Week 4+: Optional Enhancements
- Fine-tune VLM with LoRA (3-5 days)
- Collect custom data if needed (2-3 days)
- Add mobile base integration (LeKiwi) (5-7 days)
- Implement bimanual coordination (3-5 days)

---

## Troubleshooting Guide

### Stage 1 Issues

#### USB Port Not Found
```
Error: [Errno 2] No such file or directory: '/dev/ttyACM0'
```

**Solutions:**
```bash
# 1. Check user permissions
sudo usermod -a -G dialout $USER
# Log out and log back in

# 2. Check if device connected
lsusb
# Should see "Future Technology Devices International"

# 3. Try different USB ports
ls /dev/tty* | grep -E 'ACM|USB'
```

---

#### Motor Communication Errors
```
Error: Unable to read from motor ID 1
```

**Solutions:**
1. **Check motor IDs:**
   ```bash
   python -m lerobot.scripts.lerobot_setup_motors \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0
   ```

2. **Check power supply:**
   - Ensure 12V power supply connected
   - Check battery charge if using Anker battery

3. **Check servo connections:**
   - Verify all servos connected in daisy chain
   - Check for loose connections

---

#### Calibration Fails
```
Error: Joint range calibration incomplete
```

**Solutions:**
- Move joints slowly through FULL range (not just partial)
- Ensure no obstacles blocking joint movement
- Check if joints can physically reach limits (mechanical issue)

---

### Stage 2 Issues

#### Model Download Fails
```
Error: Connection timeout when downloading lerobot/smolvla_base
```

**Solutions:**
```bash
# 1. Login to HuggingFace
huggingface-cli login

# 2. Download manually
huggingface-cli download lerobot/smolvla_base

# 3. Check internet connection
ping huggingface.co
```

---

#### GPU Out of Memory
```
Error: CUDA out of memory. Tried to allocate 2.5GB
```

**Solutions:**
1. **Use smaller model:**
   ```python
   # Instead of Pi0 (4B), use SmolVLA (450M)
   policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
   ```

2. **Enable gradient checkpointing (for training):**
   ```bash
   python lerobot/scripts/train.py \
     --training.gradient_checkpointing=true
   ```

3. **Reduce batch size:**
   ```bash
   --training.batch_size=8  # Instead of 16
   ```

4. **Use FP16 instead of BF16:**
   ```python
   model = model.half()  # Convert to FP16
   ```

---

#### Low Inference Speed
```
Issue: SmolVLA running at 10Hz instead of 30Hz
```

**Solutions:**
1. **Check GPU utilization:**
   ```bash
   nvidia-smi
   # Should see 50%+ GPU utilization
   ```

2. **Compile model:**
   ```python
   policy = torch.compile(policy, mode="reduce-overhead")
   ```

3. **Use int8 quantization:**
   ```python
   from transformers import BitsAndBytesConfig

   config = BitsAndBytesConfig(load_in_8bit=True)
   policy = Policy.from_pretrained(..., quantization_config=config)
   ```

---

#### Poor Task Performance
```
Issue: Model moves randomly, doesn't approach objects
```

**Debugging:**
1. **Check observations:**
   ```python
   # Verify images and states are correct
   print(f"Image shape: {obs['observation.images.top'].shape}")
   print(f"State: {obs['observation.state']}")

   # Save image to disk
   import cv2
   cv2.imwrite("debug_image.jpg", obs['observation.images.top'])
   ```

2. **Check action magnitudes:**
   ```python
   print(f"Action: {action}")
   print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

   # Actions should be in reasonable range (e.g., [-1, 1] or [0, 1])
   ```

3. **Verify camera calibration:**
   - Ensure cameras positioned similar to training data
   - Check camera intrinsics/extrinsics

4. **Try different task description:**
   ```python
   # More specific
   task = "move gripper forward and grasp red cube"

   # Instead of vague
   task = "pick up object"
   ```

---

### Stage 3 Issues

#### VLM Generates Irrelevant Subgoals
```
Issue: VLM outputs "I cannot help with that" or irrelevant text
```

**Solutions:**
1. **Improve prompts:**
   ```python
   # Add examples in prompt
   prompt = f"""
   Task: {task}

   Example 1:
   Task: "pick up red cube"
   Subgoal: "move gripper to hover above red cube"

   Example 2:
   Task: "place object in container"
   Subgoal: "open gripper to release object"

   Now, for the given task, what should the robot do next?
   Subgoal:"""
   ```

2. **Adjust temperature:**
   ```python
   output = vlm.generate(..., temperature=0.0)  # More deterministic
   ```

3. **Use Molmo instead (better spatial grounding):**
   ```python
   vlm = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924")
   ```

---

#### Threads Crash or Deadlock
```
Error: VLA thread stopped responding
```

**Solutions:**
1. **Add timeout to queue operations:**
   ```python
   try:
       subgoal = self.current_subgoal.get(timeout=2.0)
   except queue.Empty:
       pass  # Use previous subgoal
   ```

2. **Add exception handling:**
   ```python
   def vla_control_thread(self):
       try:
           while not self.stop_flag.is_set():
               # ... control loop
       except Exception as e:
           print(f"[ERROR] VLA thread crashed: {e}")
           self.stop_flag.set()
   ```

3. **Monitor thread health:**
   ```python
   if not vla_thread.is_alive():
       print("[WARNING] VLA thread died, restarting...")
       vla_thread = threading.Thread(target=self.vla_control_thread)
       vla_thread.start()
   ```

---

#### Task Decomposition Fails
```
Issue: VLM generates only 1 subtask for complex task
```

**Solutions:**
1. **Explicitly request multiple subtasks:**
   ```python
   prompt = f"""
   Task: {task}

   Decompose into AT LEAST 3 subtasks. Be specific.

   Subtasks (minimum 3):
   1.
   2.
   3.
   ...
   """
   ```

2. **Use chain-of-thought prompting:**
   ```python
   prompt = f"""
   Task: {task}

   Let's think step-by-step:
   1. First, analyze the scene and identify objects
   2. Then, plan the sequence of actions
   3. Finally, list each action as a subtask

   Subtasks:
   """
   ```

3. **Try Qwen2.5-VL (better at multi-step reasoning):**
   ```python
   vlm = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
   ```

---

## References & Resources

### Official Documentation
- **LeRobot Docs:** https://huggingface.co/docs/lerobot
- **SO-101 Tutorial:** https://huggingface.co/docs/lerobot/so101
- **SmolVLA Blog:** https://huggingface.co/blog/smolvla
- **Pi0 Blog:** https://huggingface.co/blog/pi0

### Pretrained Models
- **lerobot/smolvla_base:** https://huggingface.co/lerobot/smolvla_base
- **r2owb0/act1:** https://huggingface.co/r2owb0/act1
- **lerobot/pi0:** https://huggingface.co/lerobot/pi0
- **Qwen3-VL-4B:** https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- **Molmo-7B:** https://huggingface.co/allenai/Molmo-7B-D-0924

### Datasets
- **lerobot/svla_so101_pickplace:** https://huggingface.co/datasets/lerobot/svla_so101_pickplace
- **lerobot/svla_so100_stacking:** https://huggingface.co/datasets/lerobot/svla_so100_stacking
- **r2owb0/so101-DS1:** https://huggingface.co/datasets/r2owb0/so101-DS1

### GitHub Repositories
- **LeRobot:** https://github.com/huggingface/lerobot
- **Qwen3-VL:** https://github.com/QwenLM/Qwen3-VL
- **Molmo:** https://github.com/allenai/molmo
- **OpenVLA:** https://github.com/openvla/openvla
- **OpenVLA-OFT:** https://github.com/moojink/openvla-oft

### Community & Support
- **LeRobot Discord:** https://discord.gg/s3KuuzsPFb
- **HuggingFace Forums:** https://discuss.huggingface.co/c/lerobot
- **GitHub Issues:** https://github.com/huggingface/lerobot/issues

### Papers
- **SmolVLA:** https://arxiv.org/abs/2506.01844
- **Pi0:** https://www.physicalintelligence.company/download/pi0.pdf
- **Qwen2.5-VL:** https://arxiv.org/abs/2502.13923
- **Molmo:** https://arxiv.org/abs/2409.17146
- **ACT (Action Chunking):** https://arxiv.org/abs/2304.13705

### Hardware Vendors
- **SO-101 Follower:** https://shop.wowrobo.com/products/so-arm101
- **SO-101 Leader:** https://shop.wowrobo.com/products/so-arm101-leader
- **Intel RealSense D435:** https://www.intelrealsense.com/depth-camera-d435/

---

## Conclusion

This MVP plan provides a clear, research-backed roadmap for xlerobot experiments with SO-ARM-101:

1. **Stage 1 (2-4 hours):** Verify hardware works
2. **Stage 2 (1-2 days):** Test pretrained VLA models (SmolVLA preferred)
3. **Stage 3a (3-5 days):** Add natural language control with Qwen3-VL or Molmo
4. **Stage 3b (5-7 days):** Enable complex multi-step task planning
5. **Fallback (only if needed):** Collect 50-100 custom demos and fine-tune

**Total timeline: 10-14 days without data collection, 13-17 days with**

**Total cost: $0 if pretrained models work, $350 if data collection needed**

By leveraging existing SO-101 pretrained models and latest 2025 VLM models, this plan minimizes both time and monetary costs while achieving production-quality robot control.

---

**Next Steps:**
1. Review this plan
2. Start with Stage 1 (hardware verification)
3. Test SmolVLA pretrained model in Stage 2
4. Decide on data collection based on Stage 2 results
5. Progress through Stage 3a and 3b incrementally

Good luck with your xlerobot experiments! 🤖
