# Bimanual Training & Inference Scripts Plan

> Generated: December 30, 2025
>
> This document contains research and implementation plans for bimanual VLA training with SmolVLA and xVLA.

## Overview

Create bimanual training and inference scripts for SmolVLA and xVLA models under `jdocs/scripts/bimanual/`.

---

## Bimanual Best Practices & Research (from Web Research)

### Successful Bimanual Implementations (Reference Links)

| Model | Success Rate | Platform | Reference |
|-------|--------------|----------|-----------|
| **RDT-1B** | 56% improvement over baselines | ALOHA (6K+ episodes) | [ICLR 2025](https://rdt-robotics.github.io/rdt-robotics/) |
| **OpenVLA-OFT+** | Up to 15% better than π0, RDT-1B | ALOHA | [OpenVLA-OFT](https://openvla-oft.github.io/) |
| **π0** | Zero-shot towel folding | ALOHA | [Physical Intelligence](https://www.pi.website/blog/openpi) |
| **TwinVLA** | 16.2% better than RDT-1B (real-world) | Multiple | [arXiv](https://arxiv.org/abs/2511.05275) |
| **GR00T N1** | SOTA on bimanual tasks | Multiple humanoids | [NVIDIA Whitepaper](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T%20N1%20Whitepaper.pdf) |

### Key Challenges & How to Avoid Them

1. **Data Scarcity Problem**
   - **Issue**: Most public datasets are single-arm only (e.g., OpenX)
   - **Solution**:
     - Collect 100+ bimanual demonstrations minimum
     - Consider TwinVLA approach: compose two single-arm VLAs
     - Use simulation data augmentation (RoboTwin 2.0 showed 367% improvement)

2. **Single-Arm Pretraining Mismatch**
   - **Issue**: OpenVLA trained on single-arm OpenX data struggles with bimanual
   - **Solution**:
     - Use models with bimanual pretraining (RDT-1B, π0)
     - Or use TwinVLA twin structure for data-efficient transfer

3. **Latency & Coordination Issues**
   - **Issue**: High latency causes pauses/jerky movements at chunk boundaries
   - **Solution**:
     - Use Real-Time Chunking (RTC) - generate next chunk while executing current
     - Action chunking + parallel decoding for 25-50Hz control
     - OpenVLA-OFT achieves 43x faster throughput with 25-timestep chunks

4. **Embodiment Heterogeneity**
   - **Issue**: Different robots have different DOF and action spaces
   - **Solution**:
     - Use action mode abstraction (xVLA's so101_bimanual)
     - Pad actions to fixed dimension (12D real -> 20D model for xVLA)

5. **Catastrophic Forgetting**
   - **Issue**: Full fine-tuning on small bimanual dataset loses pretrained knowledge
   - **Solution**:
     - Use LoRA or parameter-efficient fine-tuning
     - Freeze some components (vision encoder) for smaller datasets

### Recommended Data Collection

Based on successful implementations:
- **Minimum**: 50-100 demonstrations per task
- **Better**: 500+ demonstrations for complex bimanual tasks
- **RDT-1B**: Used 6K+ episodes for ALOHA fine-tuning
- **TwinVLA**: Showed good results with just small bimanual fine-tuning data

---

## xVLA Module Freezing Options (Similar to SmolVLA)

xVLA supports the same module-level freezing as SmolVLA:

```python
# In configuration_xvla.py (lines 97-100):
freeze_vision_encoder: bool = False   # Freeze VLM vision encoder weights
freeze_language_encoder: bool = False # Freeze VLM language encoder weights
train_policy_transformer: bool = True # Allow policy transformer to train
train_soft_prompts: bool = True       # Allow soft prompts to train
```

**Recommended Settings for Bimanual:**
| Dataset Size | Vision | Language | Policy Transformer | Soft Prompts |
|-------------|--------|----------|-------------------|--------------|
| < 50 episodes | Freeze | Freeze | Train | Train |
| 50-200 episodes | Train | Freeze | Train | Train |
| > 200 episodes | Train | Train | Train | Train |

---

## SmolVLA vs xVLA for Bimanual: Key Insight

**SmolVLA**: No documented bimanual success stories found. SmolVLA was primarily trained and tested on single-arm tasks (SO100, SO101 single arm). The model treats actions as flat vectors without arm-specific handling.

**xVLA**: Has native bimanual support via `BimanualSO101ActionSpace` with proper arm separation, gripper handling, and arm-specific loss computation.

**Recommendation**: For bimanual tasks, **xVLA is the preferred model** due to native support. SmolVLA can be used experimentally with 12D flat action vectors but may have coordination challenges.

---

## Key Differences: Single Arm vs Bimanual

| Aspect | Single Arm | Bimanual |
|--------|------------|----------|
| Action dim | 6 DOF | 12 DOF (2 x 6) |
| State dim | 6 DOF | 12 DOF (2 x 6) |
| Robot class | SO101Follower | BiSO101Follower |
| Motor naming | `shoulder_pan.pos` | `left_shoulder_pan.pos`, `right_shoulder_pan.pos` |
| Cameras | head, left_wrist | head, left_wrist, (optional: right_wrist) |

## Files to Create

### 1. Hardware Configuration
**File:** `jdocs/scripts/bimanual/bimanual_so101_hardware.yaml`
- BiSO101Follower config with left/right arm ports
- Camera setup (head, left_wrist, right_wrist)
- Customizable port placeholders

### 2. xVLA Training Script
**File:** `jdocs/scripts/bimanual/train_xvla_bimanual.sh`

Key changes from single arm:
- `ACTION_MODE=so101_bimanual` (native bimanual support via BimanualSO101ActionSpace)
- Dataset path: configurable bimanual dataset
- Camera mapping: support head, left_wrist, right_wrist -> camera1, camera2, camera3
- Domain ID: new ID for bimanual configuration (e.g., 21)

**Module Freezing Options (similar to SmolVLA):**
```bash
# Fine-tuning strategy options:
FREEZE_VISION="${FREEZE_VISION:-false}"           # Freeze VLM vision encoder
FREEZE_LANGUAGE="${FREEZE_LANGUAGE:-false}"       # Freeze VLM language encoder
TRAIN_POLICY_TRANSFORMER="${TRAIN_POLICY_TRANSFORMER:-true}"  # Train policy transformer
TRAIN_SOFT_PROMPTS="${TRAIN_SOFT_PROMPTS:-true}"  # Train soft prompts

# Recommended for bimanual (larger action space needs visual learning):
# FREEZE_VISION=false, FREEZE_LANGUAGE=false for > 100 episodes
# FREEZE_VISION=false, FREEZE_LANGUAGE=true for 50-100 episodes
```

xVLA has native bimanual support:
- `BimanualSO101ActionSpace` handles 12D real -> 20D model padding
- Separate loss computation for left/right arms
- Gripper sigmoid for indices 5 and 11

### 3. xVLA Inference Script
**File:** `jdocs/scripts/bimanual/infer_xvla_bimanual.py`

Key changes:
- Use `BiSO101Follower` robot class
- Handle 12 DOF state/action (6 per arm)
- Motor names: `left_shoulder_pan`, `right_shoulder_pan`, etc.
- Camera setup: head + left_wrist + right_wrist (configurable)
- Action trimming: 20D model output -> 12D robot action

### 4. SmolVLA Training Script
**File:** `jdocs/scripts/bimanual/train_smolvla_bimanual.sh`

**WARNING**: SmolVLA has NO native bimanual support. This is experimental.

Key considerations:
- SmolVLA treats all actions as flat 12D vector (no arm-specific logic)
- No separate loss computation for left/right arms
- No gripper-specific handling per arm
- Camera mapping: support 3 cameras (camera1, camera2, camera3)

**Module Freezing Options (from single-arm script):**
```bash
# Fine-tuning modes:
FREEZE_VISION="${FREEZE_VISION:-false}"      # Recommend FALSE for bimanual
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
TRAIN_STATE_PROJ="${TRAIN_STATE_PROJ:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"  # Needed for 12D actions

# Mode recommendations for bimanual:
# Mode 2 (Vision + Expert): FREEZE_VISION=false, TRAIN_EXPERT_ONLY=true
#   - 185M trainable params, good for positioning
# Mode 3 (Full VLM): FREEZE_VISION=false, TRAIN_EXPERT_ONLY=false
#   - 450M params, may help with coordination but risk of forgetting
```

Recommended settings:
- `FREEZE_VISION=false` (bimanual tasks need visual-spatial learning)
- `GRADIENT_CHECKPOINTING=true` (more parameters, save VRAM)
- `MAX_STEPS=20000+` (more complex task needs more training)
- `CHUNK_SIZE=50` (keep same as single arm)

### 5. SmolVLA Inference Script
**File:** `jdocs/scripts/bimanual/infer_smolvla_bimanual.py`

Key changes:
- Use `BiSO101Follower` robot class
- Handle 12 DOF state/action as flat vector
- Split action into left (0:6) and right (6:12) for robot control
- Format observation with all motor states

## Critical Implementation Details

### xVLA BimanualSO101ActionSpace (from `src/lerobot/policies/xvla/action_hub.py`)
```
Real action dim: 12 (6 per arm)
Model action dim: 20 (padded for pretrained compatibility)

Left arm indices: 0-5 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
Right arm indices: 6-11 (same joints)
Gripper indices: (5, 11) - sigmoid postprocessing
Dummy indices: 12-19 (zeros for padding)
```

### Camera Mapping Options
```
Option A (2 cameras): head->camera1, left_wrist->camera2
Option B (3 cameras): head->camera1, left_wrist->camera2, right_wrist->camera3
```

### Robot Motor Layout (BiSO101Follower)
```
Left arm:  left_shoulder_pan.pos, left_shoulder_lift.pos, left_elbow_flex.pos,
           left_wrist_flex.pos, left_wrist_roll.pos, left_gripper.pos
Right arm: right_shoulder_pan.pos, right_shoulder_lift.pos, right_elbow_flex.pos,
           right_wrist_flex.pos, right_wrist_roll.pos, right_gripper.pos
```

## Implementation Steps

1. **Create directory structure**
   - `mkdir -p jdocs/scripts/bimanual`

2. **Create bimanual_so101_hardware.yaml**
   - Template with customizable ports
   - 3-camera setup

3. **Create train_xvla_bimanual.sh**
   - Copy from train_xvla_pickplace.sh
   - Add `ACTION_MODE=so101_bimanual`
   - Update camera mapping for 3 cameras
   - Update default dataset path

4. **Create infer_xvla_bimanual.py**
   - Copy from infer_xvla_so101.py
   - Replace SO101Follower with BiSO101Follower
   - Update motor names (left_/right_ prefixes)
   - Handle 12D action output

5. **Create train_smolvla_bimanual.sh**
   - Copy from train_smolvla_pickplace.sh
   - Update camera mapping for 3 cameras
   - Enable FREEZE_VISION=false by default
   - Update default dataset path

6. **Create infer_smolvla_bimanual.py**
   - Copy from infer_smolvla_so101.py
   - Replace SO101Follower with BiSO101Follower
   - Update motor names and action handling

## Best Practices from Research

### xVLA Bimanual (Recommended approach)
- Use `action_mode=so101_bimanual` for proper action space handling
- Use `bfloat16` precision to avoid OOM
- Do NOT freeze vision/language encoders for new embodiment
- Train soft prompts for domain adaptation

### SmolVLA Bimanual
- Since no native bimanual support, treat as 12D flat action
- Enable `gradient_checkpointing` for VRAM efficiency
- Consider `FREEZE_VISION=false` for complex bimanual tasks
- May need longer training (bimanual coordination is harder)

### General Recommendations
- Start with xVLA for bimanual tasks (better native support)
- Use 3 cameras for better spatial awareness
- Collect 100+ bimanual demonstrations for good performance
- Use language prompts that specify bimanual actions (e.g., "use both hands to...")

## Potential Issues & Debugging

### Common Bimanual Training Issues

1. **Arms Moving Out of Sync**
   - **Cause**: Flat action vector loses arm coordination
   - **Solution**: Use xVLA with `so101_bimanual` action mode (has arm-specific loss)

2. **Gripper Not Opening/Closing Correctly**
   - **Cause**: Gripper values not properly normalized per arm
   - **Solution**: xVLA applies sigmoid to gripper indices (5, 11) separately

3. **Action Dimension Mismatch Error**
   - **Cause**: Model expects different action dim than dataset
   - **Solution**: For xVLA, ensure `ACTION_MODE=so101_bimanual` (12D -> 20D padding)

4. **OOM During Training**
   - **Cause**: Bimanual requires more memory (2x cameras, 2x action dim)
   - **Solution**: Use `bfloat16` dtype, enable `GRADIENT_CHECKPOINTING`, reduce batch size

5. **Slow Inference / Jerky Motion**
   - **Cause**: High latency at chunk boundaries
   - **Solution**: Use action chunking (chunk_size=32-50), consider Real-Time Chunking

6. **Poor Left/Right Coordination**
   - **Cause**: Model not learning arm interdependence
   - **Solution**: Collect demonstrations with explicit bimanual coordination tasks

### Debugging Commands
```bash
# Check dataset action dimensions
python -c "from lerobot.datasets import LeRobotDatasetMetadata; m = LeRobotDatasetMetadata(repo_id='bimanual_task', root='datasets/bimanual_task'); print(m.action_keys)"

# Verify BiSO101Follower motor names
python -c "from lerobot.robots.bi_so101_follower import BiSO101Follower; print(BiSO101Follower.name)"
```

---

## Source References
- Single arm SmolVLA: `jdocs/scripts/singleArm/train_smolvla_pickplace.sh`
- Single arm xVLA: `jdocs/scripts/singleArm/train_xvla_pickplace.sh`
- Bimanual robot: `src/lerobot/robots/bi_so101_follower/bi_so101_follower.py`
- xVLA action hub: `src/lerobot/policies/xvla/action_hub.py` (BimanualSO101ActionSpace)
- xVLA config: `src/lerobot/policies/xvla/configuration_xvla.py` (freezing options)
- Groot bimanual reference: `jdocs/bimanual/bin/groot_bimanual_setting.md`

## External References
- [RDT-1B](https://rdt-robotics.github.io/rdt-robotics/) - ICLR 2025, 56% improvement on ALOHA
- [OpenVLA-OFT](https://openvla-oft.github.io/) - 25-50x faster inference, bimanual ALOHA success
- [TwinVLA](https://arxiv.org/abs/2511.05275) - Data-efficient bimanual from single-arm VLAs
- [π0](https://www.pi.website/blog/openpi) - Zero-shot bimanual towel folding
- [GR00T N1](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T%20N1%20Whitepaper.pdf) - NVIDIA bimanual humanoid
- [SmolVLA Docs](https://huggingface.co/docs/lerobot/en/smolvla) - Hugging Face documentation
- [X-VLA Docs](https://huggingface.co/docs/lerobot/en/xvla) - Hugging Face documentation
