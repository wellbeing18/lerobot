# Robot Arm Positioning Accuracy: Comprehensive Research & Solutions

**Document Type**: Investigation Report
**Created**: 2025-12-25
**Author**: Claude (Anthropic)
**Status**: Complete Research with Implementation Plans

---

## Executive Summary

This document investigates the **positioning accuracy problem** observed in robot manipulation models (SmolVLA, GROOT) trained on the SO-101 pick-and-place dataset. The robot arm attempts to grasp objects in the space adjacent to the target rather than at the correct location.

**Key Findings**:
1. **Frozen backbone** prevents learning precise visual-spatial relationships
2. **Small dataset** (50 episodes) lacks position/approach variety
3. **Open-loop action chunking** compounds small errors over horizon
4. **Multiple solutions exist** ranging from quick fixes to comprehensive pipelines

**Recommended Approach**: Phased implementation starting with training config changes, then data augmentation, simulation debugging, and optionally RL refinement.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Root Cause Investigation](#2-root-cause-investigation)
3. [Solution Category A: Simulation & Digital Twin](#3-solution-category-a-simulation--digital-twin)
4. [Solution Category B: Training Strategies](#4-solution-category-b-training-strategies)
5. [Solution Category C: RL Refinement](#5-solution-category-c-rl-refinement)
6. [Solution Category D: Closed-Loop Corrections](#6-solution-category-d-closed-loop-corrections)
7. [Solution Category E: Camera & Calibration](#7-solution-category-e-camera--calibration)
8. [Solution Category F: Out-of-Box Solutions](#8-solution-category-f-out-of-box-solutions)
9. [Data Collection Guidelines](#9-data-collection-guidelines)
10. [Isaac Sim Setup Guide](#10-isaac-sim-setup-guide)
11. [Implementation Plan](#11-implementation-plan)
12. [References](#12-references)

---

## 1. Problem Analysis

### Observed Behavior

The robot arm:
- Approaches the target object correctly in general trajectory
- Fails to grasp at the precise location
- Grasps in the space **adjacent to** the object (systematic offset)
- Issue persists across multiple model architectures (SmolVLA, GROOT)

### Implications

Since the issue occurs across different models trained on the same dataset, the root cause is likely:
1. **Data-related** (limited variety, distribution gaps)
2. **Training strategy** (frozen backbones preventing spatial learning)
3. **Inference limitations** (open-loop execution without correction)

---

## 2. Root Cause Investigation

### 2.1 Frozen Backbone Problem

**Source**: [Hackaday - Language Conditioning Debug](https://hackaday.io/project/204187/log/244117)

**Finding**: When the VLM backbone is frozen during training:
- The model produces **nearly identical embeddings** for different visual scenarios
- Action-only fine-tuning cannot learn precise visual-spatial relationships
- The model learns **position-based heuristics** rather than accurate object localization

**Evidence**:
> "Model's behavior was 100% determined by visual state, with 0% influence from language instruction"

**Implication for Positioning**:
The frozen backbone produces similar embeddings for objects at slightly different positions, making it impossible for the action head to predict precise grasping positions.

### 2.2 Small Dataset Limitations

**Current State**: 50 episodes in `datasets/pick_and_place`

**Problems**:
- Limited coverage of object positions in workspace
- Likely similar approach angles across demonstrations
- Fixed lighting/background reduces visual generalization
- Model overfits to specific visual patterns

**Research on Dataset Size** ([arXiv:2512.11921](https://arxiv.org/abs/2512.11921)):
> "Effective robot adaptation requires sufficient training data diversity"

### 2.3 Open-Loop Action Chunking

**Source**: [A2C2 Paper](https://arxiv.org/html/2509.23224)

**Problem**: Chunked predictions lack closed-loop reactivity
- Small initial positioning errors compound over action horizon
- No visual servoing to correct during approach
- By the time the gripper reaches the target, error has accumulated

**Finding from ACT Research**:
> "Fine manipulation tasks involve precise, closed-loop feedback and require high degrees of hand-eye coordination to adjust and re-plan in response to changes"

### 2.4 Camera/Calibration Issues

Potential contributing factors:
- Camera intrinsics differ between training and inference
- No explicit depth information (RGB-only)
- Possible lens distortion not corrected

---

## 3. Solution Category A: Simulation & Digital Twin

### 3.1 Isaac Sim Digital Twin for SO-101

**Purpose**: Debug positioning issues in simulation before real hardware

**Resources**:
- [Hackaday: Building Digital Twin](https://hackaday.io/project/204187/log/243785)
- [GitHub: isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)
- [Seeed Studio Guide](https://wiki.seeedstudio.com/lerobot_so100m_isaacsim/)

**Benefits**:
| Benefit | Description |
|---------|-------------|
| Visual debugging | See predicted vs actual trajectories in 3D |
| Safe testing | No hardware damage during debugging |
| State comparison | Real-time virtual vs physical comparison |
| Failure analysis | Record and analyze failure cases |

**Architecture**:
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Leader Arm    │────▶│  Joint State    │────▶│   Isaac Sim     │
│   (Physical)    │     │    Bridge       │     │   (Virtual)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ROS2 Topics
                     /isaac_joint_command
```

**Key Technical Details**:
- Isaac Sim 4.5/5.0 + ROS2 Humble + CycloneDDS
- 20Hz synchronization frequency
- Read from `/dev/follower` (actual robot state, not commands)
- Unique ROS_DOMAIN_ID to avoid network conflicts

### 3.2 MimicGen Data Augmentation

**Purpose**: Multiply small datasets 10-20x with synthetic variations

**Source**: [Hackaday: MimicGen Pipeline](https://hackaday.io/project/204187/log/243819)

**Results Achieved**:
| Metric | Value |
|--------|-------|
| Multiplication | 1 demo → 10 augmented demos |
| Success Rate | 71.4% (10/14 attempts) |
| Compression | 732 MB → 21 MB LeRobot format |
| Generation Time | ~2 minutes per demo |

**Pipeline**:
```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  IK Action   │───▶│   Subtask    │───▶│    Data      │───▶│ Joint Action │
│  Conversion  │    │  Annotation  │    │ Augmentation │    │  Conversion  │
│  (6D → 8D)   │    │ (boundaries) │    │ (recombine)  │    │  (8D → 6D)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Critical Configuration**:
```python
# Height threshold must match your object size
height_threshold = 0.05  # 3.3x cube height (for 1.5cm cube)
# NOT default 0.20 which is too strict

# Subtask structure
intermediate_subtask = {
    "subtask_term_signal": "gripper_closed",
    "subtask_term_offset_range": (0, 10)
}
final_subtask = {
    "subtask_term_signal": None,  # REQUIRED for final
    "subtask_term_offset_range": (0, 0)
}
```

### 3.3 Sandwich Assembly Simulation

**Source**: [Hackaday: Sandwich Simulation](https://hackaday.io/project/204187/log/244016)

**Key Lessons**:
1. USD rigid body hierarchy affects physics success
2. Objects must be direct children of root prim
3. Camera FOV must match real hardware (78° for Nexigo N60)

---

## 4. Solution Category B: Training Strategies

### 4.1 Full Fine-Tuning vs Partial/LoRA

**Research Summary** ([arXiv:2512.11921](https://arxiv.org/abs/2512.11921)):

| Approach | VRAM | Accuracy | Recommendation |
|----------|------|----------|----------------|
| **Frozen Backbone** | 8GB | Poor | NOT for precision tasks |
| **LoRA (rank 32-64)** | 12-16GB | Good | Small distribution shift |
| **Full Fine-Tune** | 24-80GB | Best | Large shift, complex tasks |

**OpenVLA Findings**:
> "LoRA fine-tuning strikes an optimal performance-compute trade-off"
> "Full fine-tuning is only recommended if LoRA is insufficient"

**For Positioning Issues** (24GB GPU):

```bash
# Option 1: Enable backbone fine-tuning with gradient checkpointing
--tune_llm=true
--tune_visual=true
--gradient_checkpointing=true
--batch_size=4
--gradient_accumulation_steps=8

# Option 2: LoRA on backbone (lower memory)
--lora_rank=64
--lora_alpha=128
--tune_llm_with_lora=true
--tune_visual_with_lora=true
```

### 4.2 Training Steps Recommendations

**Source**: Multiple Hackaday investigations

| Task Type | Recommended Steps | Notes |
|-----------|-------------------|-------|
| Simple pick-place | 15,000-20,000 | Current 10k may be borderline |
| Precise positioning | 20,000-30,000 | Your case - need more steps |
| Multi-step assembly | 30,000+ | Complex sequences |

**Warning Signs of Undertraining**:
- Robot twitching/oscillating
- Small action magnitudes
- Inconsistent behavior across trials

### 4.3 Data Augmentation During Training

**Already in ver1_6**:
```python
color_jitter_params = {
    "brightness": 0.3,
    "contrast": 0.4,
    "saturation": 0.5,
    "hue": 0.08
}
```

**Additional Augmentations to Consider**:
- Random cropping/resizing (simulate camera position variation)
- Cutout/CoarseDropout (simulate occlusions)
- Gaussian noise on images
- State noise injection (`state_additive_noise_scale: 0.01`)

---

## 5. Solution Category C: RL Refinement

### 5.1 Residual RL (ResiP)

**Paper**: [From Imitation to Refinement](https://residual-assembly.github.io/)

**Concept**:
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Frozen BC      │────▶│    Residual     │────▶│  Final Action   │
│  Policy         │     │    RL Policy    │     │  (BC + Residual)│
│  (chunked)      │     │ (closed-loop)   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Benefits**:
- Preserves BC competence for coarse motion
- Adds closed-loop corrections for precision
- Works with existing trained models
- Addresses action chunking limitations

### 5.2 IN-RIL: Interleaved RL and IL

**Paper**: [arXiv:2505.10442](https://arxiv.org/abs/2505.10442)

**Innovation**:
- Gradient surgery: RL and IL gradients in independent subspaces
- Prevents destructive interference during training
- Network separation: Residual updated by RL only

**Results**: 12% → 88% success rate on Robomimic Transport

### 5.3 RL-100: Real-World RL Framework

**Paper**: [arXiv:2510.14830](https://arxiv.org/html/2510.14830v1)

**Three-Stage Pipeline**:
1. **Imitation Learning**: Leverage human priors
2. **Iterative Offline RL**: Gated PPO updates with Offline Policy Evaluation
3. **Online RL**: Eliminate residual failure modes

---

## 6. Solution Category D: Closed-Loop Corrections

### 6.1 A2C2: Asynchronous Action Chunk Correction

**Paper**: [Leave No Observation Behind](https://arxiv.org/html/2509.23224)

**Architecture**:
```python
# Lightweight correction head
correction = correction_head(
    latest_observation,      # Current visual input
    predicted_action,        # From base VLA
    chunk_position_encoding, # Where in chunk (0-15)
    base_policy_features     # Intermediate features
)
action = predicted_action + correction  # Per-step correction
```

**Benefits**:
- Runs every control step (not just chunk boundaries)
- Preserves base model competence
- Adds closed-loop responsiveness
- Works with any off-the-shelf VLA

### 6.2 Real-Time Chunking (RTC)

**Already implemented** in your SmolVLA/Pi0.5 RTC scripts.

**Features**:
- Asynchronous chunk generation
- Blending new chunks during execution
- 20% faster than synchronous inference
- Smoother motion

**Current Gap**: RTC improves smoothness but doesn't explicitly add position correction.

---

## 7. Solution Category E: Camera & Calibration

### 7.1 Hand-Eye Calibration

**Learning-Based Method** ([arXiv:2311.01335](https://arxiv.org/html/2311.01335v3)):

| Metric | Achieved |
|--------|----------|
| Translation Error | 0.93mm |
| Rotation Error | 0.27° |
| Time | ~1 second |

**Approach**: Uses robot base as reference, eliminates need for calibration objects.

### 7.2 Depth Integration

**RealD²iff** Approach:
- Learn to synthesize noisy depth from simulation
- Clean-to-noisy paradigm for depth generation
- Zero-shot sim2real without fine-tuning
- Bridges visual sim2real gap

**Application**: If RGB-only training is causing depth perception issues, adding depth modality could help.

### 7.3 Calibration Checklist

```
[ ] Camera intrinsics match between training and inference
[ ] Lens distortion correction applied
[ ] Camera mounting position unchanged
[ ] Lighting conditions similar
[ ] No physical camera drift/movement
```

---

## 8. Solution Category F: Out-of-Box Solutions

### 8.1 Visual Servoing Layer

Add a separate visual servoing policy that:
- Activates when gripper is within 10cm of target
- Uses high-frequency visual feedback (60Hz+)
- Provides fine-grained position corrections

**Implementation**:
```python
class HierarchicalController:
    def __init__(self):
        self.approach_policy = load_vla_model()  # Coarse
        self.servoing_policy = load_servoing_model()  # Fine

    def get_action(self, obs):
        gripper_distance = estimate_distance(obs)
        if gripper_distance > 0.10:  # 10cm threshold
            return self.approach_policy(obs)
        else:
            return self.servoing_policy(obs)  # High-freq corrections
```

### 8.2 Multi-View Training

Train with multiple camera viewpoints:
- **Head camera**: Scene context, approach planning
- **Wrist camera**: Close-up precision, grasp verification
- Learn view-consistent representations

**Benefit**: Wrist camera provides higher resolution view of grasp target.

### 8.3 Hierarchical Policy

Split into two specialized policies:

| Policy | Focus | Training Data |
|--------|-------|---------------|
| **Approach** | Coarse positioning | Full demonstrations |
| **Grasp** | Fine positioning | Close-up data only |

### 8.4 Contact-Based Correction

Use force/torque sensing to:
- Detect contact with workspace
- Trigger corrective movements
- Learn from tactile feedback

**Note**: Requires hardware modification (force sensors).

### 8.5 Self-Supervised Data Collection

Use digital twin for automated failure analysis:
1. Run policy in simulation
2. Automatically identify failure cases
3. Generate corrective demonstrations
4. Retrain on augmented dataset

---

## 9. Data Collection Guidelines

### 9.1 Current Dataset Analysis

**Size**: 50 episodes
**Issue**: Likely lacks variety in critical dimensions

### 9.2 Recommended Variations

#### Priority 1: Object Position (CRITICAL)
```
Current: Objects in similar positions
Needed: Cover entire reachable workspace

Grid Pattern (3x3 minimum):
┌─────┬─────┬─────┐
│ L-F │ C-F │ R-F │  F = Far from robot
├─────┼─────┼─────┤
│ L-M │ C-M │ R-M │  M = Middle
├─────┼─────┼─────┤
│ L-N │ C-N │ R-N │  N = Near robot
└─────┴─────┴─────┘
L = Left, C = Center, R = Right

Target: At least 5-10 episodes per position
Total: 45-90 additional episodes
```

#### Priority 2: Approach Angle
```
- Straight-down grasps (current)
- 30° angled approaches
- 45° angled approaches
- Side approaches (where possible)
```

#### Priority 3: Lighting Variation
```
- Morning light (warm, directional)
- Afternoon (bright, overhead)
- Evening (dim, multiple shadows)
- Artificial lighting variations
```

#### Priority 4: Object Appearance
```
- Different colored cubes (red, blue, green)
- Slightly different sizes (1cm, 2cm, 3cm)
- Different textures if available
```

### 9.3 Recommended Dataset Size

| Current | Minimum | Recommended |
|---------|---------|-------------|
| 50 | 100-150 | 200-300 |

With MimicGen augmentation:
| Real Demos | Augmented | Total Training |
|------------|-----------|----------------|
| 100 | 10x | 1,000+ |

---

## 10. Isaac Sim Setup Guide

### 10.1 System Requirements

```
OS: Ubuntu 22.04 LTS
GPU: NVIDIA with 16GB+ VRAM
Driver: 535+
RAM: 32GB+ recommended
Storage: 100GB+ SSD
```

### 10.2 Installation (Omniverse Launcher)

```bash
# 1. Download Omniverse Launcher
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage

# 2. Make executable and run
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage

# 3. Install Isaac Sim from Launcher
# Navigate to: Exchange → Isaac Sim → Install (version 4.5.0 or 5.0)

# 4. Install Isaac Lab
cd ~/.local/share/ov/pkg/isaac-sim-*/
./python.sh -m pip install isaaclab
```

### 10.3 SO-101 Integration

```bash
# Clone Isaac Lab extension for SO-ARM
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101

# Install dependencies
pip install -e .

# Test basic simulation
python scripts/test_env.py --task SO-ARM100-Reach-v0
```

### 10.4 Digital Twin Bridge Code

```python
#!/usr/bin/env python3
"""joint_state_bridge.py - Read robot state and publish to Isaac Sim."""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateBridge(Node):
    def __init__(self):
        super().__init__('joint_state_bridge')

        # Initialize robot connection
        from lerobot.robots.so101_follower import SO101Follower
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

        config = SO101FollowerConfig(port="/dev/follower")
        self.robot = SO101Follower(config)
        self.robot.connect()

        # Publisher
        self.pub = self.create_publisher(JointState, '/isaac_joint_command', 10)

        # Timer (20Hz)
        self.timer = self.create_timer(0.05, self.publish_state)

        # Joint name mapping (LeRobot → Isaac Sim)
        self.joint_mapping = {
            "shoulder_pan": "Rotation",
            "shoulder_lift": "Pitch",
            "elbow_flex": "Elbow",
            "wrist_flex": "Wrist_Pitch",
            "wrist_roll": "Wrist_Roll",
            "gripper": "Jaw"
        }

    def publish_state(self):
        obs = self.robot.get_observation()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_mapping.values())
        msg.position = [
            obs[f"{k}.pos"]
            for k in self.joint_mapping.keys()
        ]
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = JointStateBridge()
    rclpy.spin(node)
    node.robot.disconnect()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 10.5 ROS2 Configuration

```bash
# Set unique domain ID (avoid conflicts)
export ROS_DOMAIN_ID=42

# Use CycloneDDS for reliability
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Add to ~/.bashrc for persistence
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
```

---

## 11. Implementation Plan

### Phase 1: Analysis & Quick Improvements (Week 1)

**Goal**: Identify if issue is calibration or model, apply quick fixes

| Task | Action | Expected Impact |
|------|--------|-----------------|
| Analyze failures | Use trace scripts to plot predicted vs actual positions | Identify systematic vs random errors |
| Enable backbone fine-tuning | Set `tune_llm=true` for GROOT | Better visual-spatial learning |
| Increase steps | Train to 20-25k steps | More training = better precision |
| Camera check | Verify intrinsics match | Rule out calibration issues |

**Training Config Changes**:

For GROOT:
```bash
--tune_llm true \
--tune_visual true \  # if VRAM allows
--gradient_checkpointing true \
--max_steps 25000
```

For SmolVLA:
```bash
--policy.freeze_vision_encoder=false \
--policy.gradient_checkpointing=true \
--training.max_steps=25000
```

### Phase 2: Data Collection (Week 2-3)

**Goal**: Increase dataset variety, especially position coverage

| Task | Episodes | Variation |
|------|----------|-----------|
| Position grid | 45-90 | 9 positions × 5-10 eps each |
| Approach angles | 20-30 | Mix straight/angled |
| Lighting | 10-20 | Different times/sources |

**Total New Data**: 75-140 episodes → Combined 125-190

### Phase 3: Isaac Sim Setup (Week 3-4)

**Goal**: Enable visual debugging and trajectory comparison

| Task | Deliverable |
|------|-------------|
| Install Isaac Sim | Working simulation environment |
| Import SO-101 URDF | Robot model in simulation |
| Joint state bridge | Real-to-sim synchronization |
| Trajectory visualization | Compare predicted vs actual |

### Phase 4: MimicGen Augmentation (Week 4-5)

**Goal**: Multiply dataset 10x with synthetic variations

| Task | Input | Output |
|------|-------|--------|
| Setup MimicGen | Isaac Sim environment | Configured pipeline |
| Generate augmented data | 150 real demos | 1,500 augmented |
| Quality verification | Visual inspection | Filtered dataset |
| Retrain | Augmented dataset | New models |

### Phase 5: Advanced Solutions (Week 6+)

**Only if still needed after Phase 4**:

| Solution | Complexity | Expected Impact |
|----------|------------|-----------------|
| Residual RL | High | Closed-loop corrections |
| A2C2 correction | Medium | Per-step adjustments |
| Visual servoing | Medium | High-precision close-up |

---

## 12. References

### Hackaday Project Logs

1. [Language Conditioning Debug](https://hackaday.io/project/204187/log/244117) - Frozen backbone analysis
2. [Isaac Sim Digital Twin](https://hackaday.io/project/204187/log/243785) - SO-101 simulation setup
3. [MimicGen Pipeline](https://hackaday.io/project/204187/log/243819) - Data augmentation
4. [Sandwich Assembly Sim](https://hackaday.io/project/204187/log/244016) - Complex task simulation

### Research Papers

5. [LoRA for VLA Fine-Tuning](https://arxiv.org/abs/2512.11921) - Efficient adaptation
6. [IN-RIL: Interleaved RL/IL](https://arxiv.org/abs/2505.10442) - Gradient surgery
7. [ResiP: Residual RL](https://residual-assembly.github.io/) - Precision refinement
8. [A2C2: Action Chunk Correction](https://arxiv.org/html/2509.23224) - Closed-loop fix
9. [Hand-Eye Calibration](https://arxiv.org/html/2311.01335v3) - Learning-based calibration
10. [RL-100](https://arxiv.org/html/2510.14830v1) - Real-world RL framework

### GitHub Repositories

11. [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) - Isaac Lab for SO-ARM
12. [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) - Sim2Real framework
13. [OpenVLA](https://github.com/openvla/openvla) - VLA fine-tuning reference

### Additional Resources

14. [LeRobot Simulation Guide](https://huggingface.co/docs/lerobot/il_sim) - Imitation learning in sim
15. [Seeed Studio Isaac Sim Guide](https://wiki.seeedstudio.com/lerobot_so100m_isaacsim/) - SO-100 setup

---

## Appendix: Quick Reference Commands

### Training with Backbone Fine-Tuning (GROOT)

```bash
python -m gr00t.experiment.launch_train \
  --model nvidia/GR00T-N1.6-3B \
  --dataset /home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot \
  --tune_llm true \
  --tune_visual true \
  --tune_projector true \
  --tune_diffusion_model true \
  --global_batch_size 4 \
  --gradient_checkpointing true \
  --max_steps 25000 \
  --output_dir outputs/groot_full_finetune
```

### Training with Backbone Fine-Tuning (SmolVLA)

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.gradient_checkpointing=true \
  --training.batch_size=4 \
  --training.gradient_accumulation_steps=8 \
  --training.max_steps=25000 \
  --dataset.repo_id=datasets/pick_and_place \
  --output_dir=outputs/smolvla_full_finetune
```

### Run Digital Twin Bridge

```bash
# Terminal 1: Start Isaac Sim with ROS2 bridge
cd ~/.local/share/ov/pkg/isaac-sim-*/
./isaac-sim.sh --ros2_bridge_extension

# Terminal 2: Run joint state bridge
export ROS_DOMAIN_ID=42
python joint_state_bridge.py
```

---

## 13. Model-Specific Fine-Tuning Configuration Reference

### Understanding Each Model's Architecture

| Model | VLM Backbone | Action Head | Trainable Components |
|-------|-------------|-------------|---------------------|
| **SmolVLA** | SmolVLM2 (SigLIP + SmolLM2) | Flow Matching Expert | Action Expert (~100M of 450M params) |
| **GROOT N1.5** | Eagle2.5-HG (~2.8B) | DiT 16-layer | Projector + Diffusion (~210M) |
| **GROOT N1.6** | Cosmos-Reason-2B | DiT 32-layer | Projector + Diffusion + optional LLM layers |
| **Pi0 / Pi0.5** | PaliGemma (Gemma 2B) | Flow Matching Gemma 300M | Action Expert |

### SmolVLA Configuration (LeRobot)

```python
# In SmolVLAConfig (configuration_smolvla.py:72-74)
freeze_vision_encoder: bool = True    # ← Freeze SigLIP vision encoder
train_expert_only: bool = True        # ← Only train action expert
train_state_proj: bool = True         # ← Train state projection

# To enable backbone fine-tuning:
--policy.freeze_vision_encoder=false  # Unfreeze vision encoder
--policy.train_expert_only=false      # REQUIRED - see warning below
```

**⚠️ CRITICAL FLAG INTERACTION BUG** (discovered via code review):

In `smolvlm_with_expert.py:139-147`:
```python
def set_requires_grad(self):
    if self.freeze_vision_encoder:        # Block 1: Skipped if false
        # freeze vision encoder
    if self.train_expert_only:            # Block 2: DEFAULT=True
        for params in self.vlm.parameters():
            params.requires_grad = False   # RE-FREEZES ENTIRE VLM!
```

**Problem**: `train_expert_only=true` (default) freezes the ENTIRE VLM, which OVERRIDES `freeze_vision_encoder=false`. The vision encoder ends up frozen anyway!

**Solution**: You MUST set BOTH flags:
```bash
--policy.freeze_vision_encoder=false \
--policy.train_expert_only=false
```

**Consequence**: This also unfreezes the language model (SmolLM2), not just vision. There's no way to unfreeze vision-only without code modification.

**VRAM Impact**:
- Default (expert only): ~12-16GB with batch_size=32
- `train_expert_only=false` (vision + language unfrozen): ~22-24GB with batch_size=8
- Full VLM fine-tuning may require batch_size=4 on 24GB GPU

**Note**: SmolVLA does NOT support `gradient_checkpointing` flag (unlike Pi0). If you OOM, reduce batch_size.

### GROOT N1.5/N1.6 Configuration (Isaac-GR00T & LeRobot)

```python
# In GrootConfig (configuration_groot.py:68-71) for LeRobot
# In finetune_config.py for Isaac-GR00T
tune_llm: bool = False      # ← Freeze/unfreeze LLM (language model)
tune_visual: bool = False   # ← Freeze/unfreeze vision encoder
tune_projector: bool = True # ← Train state/action projectors
tune_diffusion_model: bool = True  # ← Train DiT action head

# To enable backbone fine-tuning:
--tune_llm=true    # Unfreeze LLM (~1.7B additional params for N1.6)
--tune_visual=true # Unfreeze vision encoder (~300M additional params)
```

**VRAM Impact**:
- Default (frozen backbone): ~18-24GB
- `tune_llm=true` only: ~24-32GB
- Both `tune_llm=true` + `tune_visual=true`: ~32-48GB (requires gradient checkpointing + reduced batch)

### Pi0 / Pi0.5 Configuration (LeRobot)

```python
# In PI0Config (configuration_pi0.py)
# Note: Pi0 does NOT expose freeze flags - it trains action expert by default
paligemma_variant: str = "gemma_2b"      # VLM backbone (frozen)
action_expert_variant: str = "gemma_300m" # Action expert (trained)
gradient_checkpointing: bool = False      # Enable for memory savings

# Pi0 currently has NO LoRA support in LeRobot (GitHub Issue #1914)
# Full backbone fine-tuning requires modifying source code
```

**Key Insight**: Pi0 in LeRobot is designed to train only the action expert. The VLM backbone (PaliGemma) is always frozen. There are no user-exposed flags to unfreeze it without code modification.

**VRAM Impact**:
- Default: ~16-20GB
- With gradient checkpointing: ~12-16GB

---

## 14. Effectiveness Analysis & Ranking

### Empirical Success Rates from Research

| Method | Task Type | Before | After | Improvement | Source |
|--------|-----------|--------|-------|-------------|--------|
| **OpenVLA-OFT** (action chunking + continuous) | LIBERO benchmark | 76.5% | 97.1% | +20.6% | [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) |
| **ResiP** (residual RL) | 0.2mm peg-in-hole | 5% | 99% | +94% | [ResiP](https://residual-assembly.github.io/) |
| **ResiP** (residual RL) | One-leg assembly (50 demos) | ~30% | 98% | +68% | [ResiP](https://residual-assembly.github.io/) |
| **IN-RIL** (interleaved RL/IL) | Robomimic Transport | 12% | 88% | +76% | [arXiv:2505.10442](https://arxiv.org/abs/2505.10442) |
| **MimicGen** (data augmentation) | Pick-and-place | Baseline | +10x demos, 71.4% gen rate | Comparable | [MimicGen](https://ar5iv.labs.arxiv.org/html/2310.17596) |
| **CyberDemo** (sim augmentation) | Dexterous manip | Baseline | +35% success | +35% | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CyberDemo_Augmenting_Simulated_Human_Demonstration_for_Real-World_Dexterous_Manipulation_CVPR_2024_paper.pdf) |
| **RialTo** (real-to-sim-to-real) | With disturbances | 5% (BC) | 75% | +70% | [arXiv:2403.03949](https://arxiv.org/html/2403.03949v1) |
| **Sim-to-Real World Models** | Sim-to-real transfer | Baseline | +23.3% | +23.3% | [arXiv:2510.02538](https://arxiv.org/html/2510.02538) |

### Critical Assessment: Simulation vs Real Data

Before ranking solutions, we must address the **sim-to-real gap** honestly:

#### Sim-to-Real Transfer Reality

| Study | Sim Performance | Real Performance | Gap |
|-------|-----------------|------------------|-----|
| [Google PVR Study](https://arxiv.org/html/2310.02219) | Good | "Near zero" | **Catastrophic** |
| CyberDemo (CVPR 2024) | Baseline | +35% | Only after real fine-tuning |
| RialTo | 91% | 75% | -16% degradation |
| BC baseline (RialTo) | - | 5-25% | Severe sim-to-real drop |

**Key Finding from Google's Large-Scale Study**:
> "For most tasks trained using few-shot imitation learning, the performance achieved when running a simulation-trained policy in the real world **cannot be predicted by that in simulation**, with most tasks' success metrics **dropping to near zero values**."

#### When Simulation IS Valuable

| Use Case | Value | Why |
|----------|-------|-----|
| **Residual RL refinement** | High | Train correction policy in sim, transfer to real |
| **Safe policy testing** | High | Test before hardware deployment |
| **Data augmentation + real fine-tune** | Medium | MimicGen works IF you fine-tune on real data after |
| **Primary training data source** | **Low** | Sim-to-real gap often catastrophic |

#### When Simulation is NOT the Answer

- **Diagnosing real-world positioning errors**: The error is in how model interprets REAL images, not synthetic ones
- **Replacing real data collection**: Sim data alone rarely transfers well
- **Quick fixes**: Setup effort is high, payoff uncertain

### Weighted Ranking of Solutions (Revised)

Based on empirical evidence and your current situation (30k steps already trained, frozen backbone, 50 episodes):

| Rank | Solution | Effectiveness | Effort | Risk | Recommendation |
|------|----------|---------------|--------|------|----------------|
| **1** | **Unfreeze Vision Encoder (SmolVLA)** | High | Low | Medium | **DO FIRST** - Directly tests frozen backbone hypothesis |
| **2** | **Diagnose error pattern** | Critical | Low | None | Record inference, analyze predicted vs actual positions |
| **3** | **Collect 50+ REAL varied episodes** | Very High | Medium | Very Low | More reliable than sim augmentation |
| **4** | **tune_llm=true (GROOT)** | Medium-High | Low | Medium | Alternative for GROOT models |
| **5** | **Residual RL (ResiP)** | Very High | Very High | Medium | After IL plateau, can train in sim |
| **6** | **MimicGen + Real Fine-tune** | Medium | High | Medium | Only if combined with real data fine-tuning |
| **7** | **Isaac Sim Digital Twin** | Medium | High | Low | For RL training and safe testing, NOT primary diagnosis |
| **8** | **Full Fine-Tuning** | Medium | Medium | **HIGH** | Last resort - catastrophic forgetting risk |

**Note**: "More training steps" removed from ranking - you already trained 30k steps with frozen backbone, which didn't solve the issue. The problem is the frozen backbone, not training duration.

### Detailed Effectiveness by Problem Type

#### For Systematic Offset (Object always missed by similar amount)
**Most Likely Cause**: Camera calibration or frozen backbone producing identical embeddings

**Best Solutions**:
1. Camera intrinsic/extrinsic calibration check
2. Unfreeze vision encoder (`freeze_vision_encoder=false` for SmolVLA)
3. Data augmentation with position variety

#### For Random Positioning Errors (Different miss directions each time)
**Most Likely Cause**: Insufficient training data variety

**Best Solutions**:
1. More training data with position variety (3x3 grid minimum)
2. Increase training steps to 25-30k
3. MimicGen augmentation for 10x data

#### For Errors That Compound During Approach
**Most Likely Cause**: Open-loop action chunking without correction

**Best Solutions**:
1. Residual RL policy for corrections
2. A2C2 per-step correction head
3. Reduce chunk size (less error accumulation)

---

## 15. Full Fine-Tuning: Risks & Unexpected Results

### What is "Full Fine-Tuning"?

Full fine-tuning means unfreezing ALL model parameters including the pretrained VLM backbone (vision encoder + language model), not just the action head. This is the most aggressive form of adaptation.

### Documented Risks

#### 1. Catastrophic Forgetting (HIGH RISK)

**Evidence**:
> "Fine-tuning vision-language models (VLMs) on robot teleoperation data... suffers from a fundamental tradeoff: learning to produce actions often diminishes the VLM's foundational reasoning and multimodal understanding" - [VLM2VLA](https://arxiv.org/abs/2509.22195)

**What Happens**:
- Model loses general visual understanding
- Language conditioning may stop working
- Semantic generalization degrades
- Model becomes task-specific, not generalizable

**Severity for Your Case**: **MEDIUM-HIGH**
- With only 50 episodes, full fine-tuning will almost certainly overfit
- The model may lose ability to generalize to slightly different object positions
- Language instructions may become ignored

#### 2. Overfitting to Training Distribution (HIGH RISK)

**Evidence**:
> "Full fine-tuning updates all weights — so it learns more, but also forgets more" - [LoRA Study](https://arxiv.org/abs/2410.21228)

**What Happens**:
- Model memorizes exact positions from training data
- Performs worse on unseen object positions
- May actually increase positioning errors on novel configurations

**Severity for Your Case**: **VERY HIGH**
- 50 episodes is far too small for full fine-tuning
- Minimum 500-1000 episodes recommended for full fine-tune

#### 3. Training Instability

**Evidence**:
> "Users may observe performance differences as large as 5-6% between runs... attributed to non-deterministic operations" - [NVIDIA GROOT Docs](https://github.com/NVIDIA/Isaac-GR00T)

**What Happens**:
- High variance between training runs
- Harder to reproduce results
- Requires more hyperparameter tuning

#### 4. Memory & Compute Requirements

**Impact**:
- Full GROOT fine-tuning: 32-48GB VRAM (exceeds your 24GB)
- Full SmolVLA fine-tuning: 24-32GB VRAM (borderline)
- Training time increases 3-5x

### When Full Fine-Tuning IS Appropriate

Based on [OpenVLA-OFT research](https://arxiv.org/abs/2502.19645):

> "Full fine-tuning is only recommended if you have sufficient compute (e.g., a full node of 8 A100 GPUs) and if LoRA fine-tuning is insufficient for your use case (e.g., if the fine-tuning distribution varies drastically from the pretraining distribution)."

**Conditions for Full Fine-Tuning**:
1. ✅ Large dataset (500+ diverse episodes)
2. ✅ Multi-GPU setup (4+ GPUs with 24GB+ each)
3. ✅ Target task is very different from pretraining
4. ✅ LoRA/partial fine-tuning has plateaued
5. ✅ You can validate against held-out test set

**Your Situation**: ❌ 50 episodes, ❌ single 24GB GPU → **Do NOT use full fine-tuning yet**

### Safer Alternatives to Full Fine-Tuning

| Alternative | VRAM | Catastrophic Forgetting Risk | Recommended |
|-------------|------|------------------------------|-------------|
| **Freeze backbone + more data** | 8-16GB | Very Low | ✅ Yes |
| **Unfreeze vision only** | 14-18GB | Low-Medium | ✅ Yes |
| **LoRA on backbone** | 12-16GB | Low | ✅ Yes |
| **tune_llm=true only (GROOT)** | 24-32GB | Medium | ⚠️ With care |
| **Full unfreeze** | 32-48GB | High | ❌ Not now |

### Mitigation Strategies If You Must Full Fine-Tune

1. **Use Lower Learning Rate**: 1e-5 instead of 1e-4
2. **EMA (Exponential Moving Average)**: Preserve previous knowledge
3. **LoRA Adapters**: Even with full access, use LoRA for backbone
4. **Early Stopping**: Monitor validation loss, stop before overfitting
5. **Regularization**: Add L2 weight decay, dropout

---

## 16. Updated Recommendations for Your Setup

### Your Situation (Updated 2025-12-25)
- **GPU**: 24GB laptop (RTX 4090)
- **Dataset**: 50 episodes in `datasets/pick_and_place`
- **Problem**: Systematic positioning offset during grasping
- **Models Tested**: SmolVLA (30k steps, frozen backbone), GROOT (both show same issue)
- **Previous Training**: `outputs/smolvla_pickplace_20251223_001238` with `freeze_vision_encoder=true`, `batch_size=32`

### Priority-Ordered Action Plan (Revised)

#### Phase 1: Unfreeze Vision Encoder (IMMEDIATE)

Since you already trained 30k steps with frozen backbone and still have positioning issues, the frozen backbone is the likely culprit.

**Training Command**:
```bash
nohup lerobot-train \
  --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --dataset.repo_id=pick_and_place \
  --dataset.root=/home/jrobot/project/lerobot/datasets/pick_and_place \
  --training.batch_size=8 \
  --training.steps=30000 \
  --training.save_freq=5000 \
  --training.log_freq=100 \
  --training.num_workers=4 \
  --output_dir=outputs/smolvla_unfrozen_vision_$(date +%Y%m%d_%H%M%S) \
  > outputs/smolvla_unfrozen_training.log 2>&1 &
```

**CRITICAL: Why `train_expert_only=false` is required**:

In `smolvlm_with_expert.py:139-147`, there's a flag interaction issue:
```python
def set_requires_grad(self):
    if self.freeze_vision_encoder:        # Skipped if false
        # freeze vision
    if self.train_expert_only:            # DEFAULT=True, RE-FREEZES entire VLM!
        for params in self.vlm.parameters():
            params.requires_grad = False   # Includes vision encoder!
```

Setting only `freeze_vision_encoder=false` does NOT work - the `train_expert_only=true` default will RE-FREEZE the entire VLM including vision. You MUST set both flags.

**Why batch_size=8 instead of 32**:
- `train_expert_only=false` unfreezes both vision AND language model (~450M params total)
- VRAM usage increases significantly
- batch_size=32 will OOM; batch_size=8 is safe for 24GB (but tight)
- If OOM occurs, reduce to batch_size=4

**Why still 30k steps**:
- Same dataset, so similar convergence expected
- Can extend if loss still decreasing at 30k

#### Phase 2: Diagnose Error Pattern

While training runs, record inference from your existing 30k frozen model:
- Log images + predicted actions
- Plot predicted end-effector positions vs actual
- Identify: systematic offset? random? compounding?

#### Phase 3: If Unfreezing Doesn't Help

| Action | When | Command/Approach |
|--------|------|------------------|
| Collect 50+ REAL varied episodes | If unfreezing helps partially | 3x3 position grid |
| Try GROOT with `tune_llm=true` | If SmolVLA unfreezing doesn't help | Different architecture test |
| Residual RL | If IL plateau reached | Train in sim, deploy on real |

#### Phase 4: Simulation (Later Priority)

Simulation is valuable for:
- **Residual RL training** after IL reaches plateau
- **Safe testing** of new policies
- **MimicGen augmentation** IF combined with real fine-tuning afterward

Simulation is NOT for:
- Primary diagnosis of real-world positioning errors
- Replacing real data collection entirely

### What NOT To Do

1. ❌ **Do NOT expect sim-only training to transfer** - Sim-to-real gap is severe without real fine-tuning
2. ❌ **Do NOT use full fine-tuning (both vision + LLM)** - Catastrophic forgetting risk with 50 episodes
3. ❌ **Do NOT increase training steps without unfreezing** - Already proved 30k frozen steps doesn't solve it
4. ❌ **Do NOT use batch_size=32 with unfrozen vision** - Will OOM on 24GB GPU

---

## 17. Summary: Effectiveness Ranking

### Final Weighted Ranking (Your Specific Case - Revised)

| Rank | Solution | Why |
|------|----------|-----|
| **#1** | **Unfreeze vision encoder** (`freeze_vision_encoder=false`) | Directly tests frozen backbone hypothesis; you already did 30k steps frozen |
| **#2** | Diagnose error pattern | Understand if systematic, random, or compounding before more changes |
| **#3** | Collect REAL varied data | More reliable than sim augmentation; addresses data variety gap |
| **#4** | tune_llm=true (GROOT) | Alternative backbone unfreezing for GROOT models |
| **#5** | Residual RL (after IL plateau) | Can train in sim, deploy on real for precision corrections |
| **#6** | MimicGen + real fine-tune | Sim augmentation only works WITH subsequent real data fine-tuning |
| **#7** | Isaac Sim digital twin | For RL training and safe testing, NOT primary diagnosis tool |
| **#8** | Full fine-tuning | **Last resort** - only with 500+ episodes and multi-GPU |

### Key Takeaways

1. **The positioning issue likely stems from frozen backbone**, not insufficient training time. You already trained 30k steps with frozen backbone - more steps won't help.

2. **Partial unfreezing is safer than full fine-tuning**: Unfreeze just the vision encoder (SmolVLA) or LLM (GROOT), not both.

3. **Sim-to-real gap is real and severe**: Google's study shows sim-trained policies often drop to "near zero" performance in real world. Simulation is NOT a replacement for real data.

4. **Simulation IS valuable for**: Residual RL training, safe testing, and data augmentation IF combined with real fine-tuning afterward.

5. **Simulation is NOT valuable for**: Diagnosing real-world positioning errors (the error is in how model interprets REAL images).

6. **Reduce batch size when unfreezing**: batch_size=32 → batch_size=8 to avoid OOM with additional trainable parameters.

7. **Pi0/Pi0.5 in LeRobot currently lacks unfreezing options**: You'll need to modify source code or use LoRA (when available).

---

*Document updated: 2025-12-25*
*Author: Claude (Anthropic)*
*Revision: Added effectiveness analysis, model-specific configurations, and full fine-tuning risk assessment*
