# xVLA Guide: Cross-Embodiment Vision-Language-Action Model

This guide covers xVLA, a 0.9B parameter VLA model designed for multi-robot, multi-task learning using soft prompts.

## Table of Contents
1. [Overview](#overview)
2. [xVLA vs SmolVLA](#xvla-vs-smolvla)
3. [Core Concepts](#core-concepts)
4. [Training Guide](#training-guide)
5. [Inference Guide](#inference-guide)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What is xVLA?

**X-VLA** (Cross-embodiment VLA) is a soft-prompted, flow-matching VLA framework that treats each hardware setup as a "task" and encodes it using learnable embeddings called **Soft Prompts**.

```
┌─────────────────────────────────────────────────────────────────┐
│                         xVLA Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────────┐    │
│  │  Images  │   │  Language │   │      Robot State         │    │
│  │ (camera1,│   │  (task)   │   │   (proprioception)       │    │
│  │ camera2) │   │           │   │                          │    │
│  └────┬─────┘   └─────┬─────┘   └────────────┬─────────────┘    │
│       │               │                       │                  │
│       ▼               ▼                       │                  │
│  ┌─────────────────────────┐                 │                  │
│  │      Florence2 VLM      │                 │                  │
│  │  (Vision + Language)    │                 │                  │
│  └───────────┬─────────────┘                 │                  │
│              │                               │                  │
│              ▼                               ▼                  │
│  ┌───────────────────────────────────────────────────────┐     │
│  │              Soft-Prompted Transformer                 │     │
│  │  ┌─────────────────────────────────────────────────┐  │     │
│  │  │  Domain ID → Soft Prompts (learnable tokens)    │  │     │
│  │  │  [prompt_1, prompt_2, ..., prompt_32]           │  │     │
│  │  └─────────────────────────────────────────────────┘  │     │
│  │                                                        │     │
│  │  24 Transformer Blocks + Diffusion Denoising          │     │
│  └───────────────────────────────────────────────────────┘     │
│              │                                                  │
│              ▼                                                  │
│  ┌───────────────────────────────────────────────────────┐     │
│  │              Action Mode Handler                       │     │
│  │  (ee6d / joint / auto / so101_bimanual)               │     │
│  └───────────────────────────────────────────────────────┘     │
│              │                                                  │
│              ▼                                                  │
│         [Actions]                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Soft Prompts** | Learnable embeddings (32 tokens) that encode robot/camera configuration |
| **Domain IDs** | Integer identifiers for different robot setups (0-29) |
| **Action Modes** | Registry system for different action spaces |
| **Florence2 VLM** | Vision-language backbone with ImageNet normalization |
| **0.9B Parameters** | Larger than SmolVLA (~450M), requires bfloat16 |

### When to Use xVLA

| Scenario | Use xVLA? |
|----------|-----------|
| Multi-robot deployment (same model) | **Yes** - soft prompts handle embodiment differences |
| Cross-embodiment transfer | **Yes** - pretrained on 7 robot platforms |
| Single robot, fast inference needed | No - use SmolVLA (smaller, faster) |
| Need RTC smooth motion | No - xVLA uses diffusion, not flow matching |
| Limited GPU memory (<16GB) | No - use SmolVLA or ACT |

---

## xVLA vs SmolVLA

### Architecture Comparison

```
SmolVLA (450M)                      xVLA (0.9B)
─────────────────                   ─────────────────
┌─────────────┐                     ┌─────────────┐
│  SmolVLM    │                     │  Florence2  │
│  (compact)  │                     │  (larger)   │
└──────┬──────┘                     └──────┬──────┘
       │                                   │
       ▼                                   ▼
┌─────────────┐                     ┌─────────────────┐
│ Flow Match  │                     │ Soft-Prompted   │
│ Transformer │                     │ Transformer     │
│             │                     │ + Domain IDs    │
└──────┬──────┘                     └────────┬────────┘
       │                                     │
       ▼                                     ▼
   [Actions]                         ┌─────────────┐
                                     │ Action Mode │
                                     │  Handler    │
                                     └──────┬──────┘
                                            │
                                            ▼
                                        [Actions]
```

### Feature Comparison

| Aspect | SmolVLA | xVLA |
|--------|---------|------|
| **Model Size** | ~450M params | 0.9B params |
| **Pretrained** | `lerobot/smolvla_base` | `lerobot/xvla-base` |
| **Action Generation** | Flow Matching | Diffusion |
| **Denoising Steps** | `num_steps=10` | `num_denoising_steps=10` |
| **RTC Support** | **Yes** | No |
| **Multi-embodiment** | Limited | **Yes** (soft prompts) |
| **Precision** | float32 | **bfloat16** (required) |
| **Batch Size** | 32 | 16 (larger model) |
| **Chunk Size** | 50 | 32 |
| **VLM Freezing** | Freeze (recommended) | **Don't freeze** (recommended) |
| **GPU Memory** | ~12GB | ~20GB |

### Training Parameter Comparison

| Parameter | SmolVLA | xVLA |
|-----------|---------|------|
| `freeze_vision_encoder` | `true` | `false` |
| `freeze_language_encoder` | N/A | `false` |
| `train_expert_only` | `true` | N/A |
| `train_policy_transformer` | N/A | `true` |
| `train_soft_prompts` | N/A | `true` |
| `dtype` | `float32` | `bfloat16` |

---

## Core Concepts

### 1. Soft Prompts

Soft prompts are **learnable token embeddings** that encode robot-specific information:

```
Domain ID: 20 (SO-101 robot)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Soft Prompt Embedding Table                             │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐     │
│  │ D=0 │ D=1 │ D=2 │ ... │D=20 │ ... │D=28 │D=29 │     │
│  └─────┴─────┴─────┴─────┴──┬──┴─────┴─────┴─────┘     │
│                             │                           │
│                             ▼                           │
│               [32 learnable tokens]                     │
│      [p₁, p₂, p₃, ..., p₃₁, p₃₂]  (hidden_size=1024)  │
└─────────────────────────────────────────────────────────┘
           │
           ▼
    Prepended to transformer sequence
    [soft_prompts, vlm_features, action_tokens, ...]
```

**Why soft prompts work:**
- Each robot/camera setup has unique characteristics
- Soft prompts learn to encode these differences
- Transformer can adapt behavior based on prompts
- Only 9M parameters (1% of model) need fine-tuning for new robots

### 2. Domain IDs

Domain IDs are integers (0-29) that select which soft prompt to use:

| Domain ID | Dataset/Robot |
|-----------|---------------|
| 0 | Bridge |
| 1 | RT1 |
| 2 | Calvin |
| 3 | LIBERO |
| 4 | WidowX-air |
| 5 | AIR-AGILEX-HQ |
| 6 | RobotWin2 |
| 7 | RoboCasa-human |
| 8 | VLABench |
| 9 | AGIBOT-challenge |
| 10 | AIR-AGILEX |
| 18 | AIRBOT |
| 20+ | Custom (your robots) |

**Choosing a domain ID for your robot:**
1. If similar to existing robot → use that domain ID
2. If new robot → use unused ID (e.g., 20, 21, ...)
3. Fine-tuning learns new soft prompts for your domain

### 3. Action Modes

Action modes handle different robot action spaces:

```
┌─────────────────────────────────────────────────────────────┐
│                    Action Mode Registry                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "auto" (Recommended)                                        │
│  ├── Detects action_dim from dataset                        │
│  ├── Pads to max_action_dim (20) for model                  │
│  └── Trims output back to real_dim                          │
│                                                              │
│  "ee6d" (End-Effector 6D)                                   │
│  ├── dim_action = 20                                        │
│  ├── Layout: XYZ + 6D rotation + gripper (×2 for bimanual) │
│  └── Loss: position(MSE×500) + rotation(MSE×10) + gripper  │
│                                                              │
│  "joint" (Joint Space)                                      │
│  ├── dim_action = 14                                        │
│  ├── Layout: 7 joints + gripper (×2 for bimanual)          │
│  └── Loss: joints(MSE) + gripper(BCE×0.1)                  │
│                                                              │
│  "so101_bimanual" (SO-101 Bimanual)                         │
│  ├── Real dim = 12, Model dim = 20                          │
│  ├── Layout: 2 arms × (5 joints + gripper)                  │
│  └── Pads/trims automatically                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**For SO-101 single arm (6 DOF):** Use `action_mode=auto`
- Dataset has 6 actions (5 joints + gripper)
- Model outputs 20, automatically trimmed to 6

### 4. Preprocessing Pipeline

xVLA requires specific preprocessing:

```
Input Observation
       │
       ▼
┌─────────────────────────────┐
│ 1. RenameObservations       │  head → camera1
│    (camera mapping)         │  left_wrist → camera2
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 2. AddBatchDimension        │  Add batch dim if missing
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. Tokenizer                │  Task text → token IDs
│    (BART tokenizer)         │  max_length=64
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. XVLAImageToFloat         │  [0, 255] → [0, 1]
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 5. XVLAImageNetNormalize    │  mean=[0.485, 0.456, 0.406]
│                             │  std=[0.229, 0.224, 0.225]
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 6. XVLAAddDomainId          │  Inject domain_id tensor
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 7. DeviceProcessor          │  Move to GPU
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 8. Normalizer               │  Apply dataset stats
└──────────────┬──────────────┘
               │
               ▼
      Processed Observation
```

---

## Training Guide

### Quick Start

```bash
# Default training (20k steps, bfloat16, batch_size=16)
bash jdocs/scripts/train_xvla_pickplace.sh

# Custom settings
MAX_STEPS=30000 BATCH_SIZE=8 DOMAIN_ID=20 bash jdocs/scripts/train_xvla_pickplace.sh
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRETRAINED_MODEL` | `lerobot/xvla-base` | Base model to fine-tune |
| `MAX_STEPS` | 20000 | Training iterations |
| `BATCH_SIZE` | 16 | Batch size (lower than SmolVLA) |
| `DTYPE` | `bfloat16` | **Required** to avoid OOM |
| `DOMAIN_ID` | 20 | Robot configuration ID |
| `ACTION_MODE` | `auto` | Action space handling |
| `CHUNK_SIZE` | 32 | Actions per chunk |
| `LEARNING_RATE` | 1e-4 | Base learning rate |
| `WARMUP_STEPS` | 1000 | LR warmup period |

### Fine-tuning Strategy

**Recommended (best performance):**
```bash
FREEZE_VISION=false \
FREEZE_LANGUAGE=false \
TRAIN_POLICY_TRANSFORMER=true \
TRAIN_SOFT_PROMPTS=true \
bash jdocs/scripts/train_xvla_pickplace.sh
```

**Memory-constrained (freeze VLM):**
```bash
FREEZE_VISION=true \
FREEZE_LANGUAGE=true \
TRAIN_POLICY_TRANSFORMER=true \
TRAIN_SOFT_PROMPTS=true \
BATCH_SIZE=8 \
bash jdocs/scripts/train_xvla_pickplace.sh
```

### Learning Rate Schedule

```
LR
│
│  ┌────────────────────────────────────────────────────┐
│  │                                                    │
1e-4 ─────────────────────────────┐                     │
│  │      VLM at 1/10 LR          │                     │
│  │      (1e-5)                  │                     │
│  │                              │    Cosine decay     │
│  │                              └─────────────────────┼─ 2.5e-6
│  │ Warmup                                             │
│  └──────────────────────────────────────────────────────
│  0      1000                                    30000  Steps
│         ↑                                        ↑
│    warmup_steps                            decay_steps
```

**Note:** VLM (vision + language encoders) automatically trained at 1/10 of base LR for stable optimization.

### Camera Mapping

The training script automatically maps camera names:

```bash
# In train_xvla_pickplace.sh
--rename_map={"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2"}
```

| Dataset Camera | xVLA Expected |
|----------------|---------------|
| `head` | `camera1` |
| `left_wrist` | `camera2` |
| `right_wrist` | `camera3` |

---

## Inference Guide

### Quick Start

```bash
# Basic inference
python jdocs/scripts/infer_xvla_so101.py \
    -c outputs/xvla_pickplace_*/checkpoints/020000/pretrained_model

# With custom task and domain
python jdocs/scripts/infer_xvla_so101.py \
    -c outputs/xvla_pickplace \
    --task "pick up the block and place it on the plate" \
    --domain-id 20 \
    --duration 60

# Dry run (no robot)
python jdocs/scripts/infer_xvla_so101.py -c outputs/xvla_pickplace --dry-run
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint, -c` | Required | Path to trained checkpoint |
| `--task, -t` | "pick up the block..." | Language instruction |
| `--domain-id` | 20 | Must match training domain_id |
| `--duration` | 60 | Max inference duration (seconds) |
| `--device` | cuda | Inference device |
| `--dry-run` | False | Test without robot commands |
| `--record` | False | Save images during inference |

### Inference Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Loop (30Hz)                     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌───────────┐         ┌──────────┐
   │ Cameras │         │   Robot   │         │   Task   │
   │ capture │         │ get_state │         │  string  │
   └────┬────┘         └─────┬─────┘         └────┬─────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   format_observation  │
                 │   + domain_id         │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │     preprocessor      │
                 │ (ImageNet norm, etc)  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  policy.select_action │
                 │  (diffusion denoise)  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    postprocessor      │
                 │  (unnormalize, trim)  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   robot.send_action   │
                 │   (6 DOF for SO-101)  │
                 └───────────────────────┘
```

### Action Trimming

xVLA outputs 20-dim actions but SO-101 needs 6:

```python
# In infer_xvla_so101.py
# xVLA with action_mode=auto may return padded actions
# Trim to actual robot action dimension (6 DOF for SO-101)
if len(action) > len(joint_names):
    action = action[:len(joint_names)]
```

---

## Configuration Reference

### XVLAConfig Parameters

```python
@dataclass
class XVLAConfig:
    # Model Architecture
    hidden_size: int = 1024          # Transformer hidden dim
    depth: int = 24                   # Transformer layers
    num_heads: int = 16               # Attention heads
    mlp_ratio: float = 4.0            # MLP expansion ratio

    # Soft Prompts
    num_domains: int = 30             # Max domain IDs
    len_soft_prompts: int = 32        # Prompt token count

    # Action Space
    action_mode: str = "ee6d"         # Action type (use "auto")
    max_action_dim: int = 20          # Max action dim for padding
    chunk_size: int = 32              # Actions per chunk
    n_action_steps: int = 32          # Steps to execute

    # Inference
    num_denoising_steps: int = 10     # Diffusion steps
    dtype: str = "float32"            # Use "bfloat16"!

    # Training Freezing
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True

    # Optimizer
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 10.0

    # Scheduler
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6
```

### Available Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `lerobot/xvla-base` | Base model (290K episodes, 7 platforms) |
| `lerobot/xvla-libero` | LIBERO fine-tuned (93% success) |
| `lerobot/xvla-widowx` | WidowX pick-and-place |
| `lerobot/xvla-folding` | Cloth folding (100% success) |
| `lerobot/xvla-agibot-world` | AgileX manipulation |
| `lerobot/xvla-google-robot` | Google Robot adapted |

---

## Troubleshooting

### Common Issues

**Issue: Out of Memory (OOM)**
```
CUDA out of memory
```
**Solutions:**
1. Use `DTYPE=bfloat16` (required!)
2. Reduce `BATCH_SIZE` to 8 or 4
3. Reduce `CHUNK_SIZE` to 16
4. Freeze VLM: `FREEZE_VISION=true FREEZE_LANGUAGE=true`

---

**Issue: Action dimension mismatch**
```
Expected action dim 20, got 6
```
**Solution:** Use `action_mode=auto`:
```bash
ACTION_MODE=auto bash jdocs/scripts/train_xvla_pickplace.sh
```

---

**Issue: Domain ID not found**
```
KeyError: domain_id
```
**Solution:** Ensure domain_id is in observation:
```python
observation["domain_id"] = 20  # Your domain ID
```

---

**Issue: ImageNet normalization error**
```
Image values outside [0, 1] range
```
**Solution:** Images must be normalized before ImageNet norm:
```python
img_tensor = torch.from_numpy(frame).float() / 255.0  # [0, 1]
# Then preprocessor applies ImageNet normalization
```

---

**Issue: Camera name mismatch**
```
Missing features: camera1, camera2
```
**Solution:** Add rename_map to training:
```bash
--rename_map={"observation.images.head":"observation.images.camera1",...}
```

---

**Issue: Low success rate on new robot**
```
Model not performing well after fine-tuning
```
**Solutions:**
1. Verify `action_mode` is correct for your robot
2. Ensure `domain_id` matches between training and inference
3. Check soft prompts are being trained: `train_soft_prompts=true`
4. Increase training steps
5. Don't freeze VLM for best results

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Precision** | Always use `bfloat16` |
| **Action Mode** | Use `auto` for new robots |
| **Domain ID** | Pick unused ID (20+) for custom robots |
| **VLM Freezing** | Don't freeze for best results |
| **Batch Size** | 16 (or lower if OOM) |
| **Training Steps** | 20k-30k for ~50 episodes |
| **Camera Mapping** | Use `--rename_map` |
| **Smooth Motion** | Use smaller chunks (no RTC) |

### Quick Comparison with SmolVLA

| Want... | Use |
|---------|-----|
| Faster inference | SmolVLA |
| RTC smooth motion | SmolVLA |
| Multi-robot training | **xVLA** |
| Cross-embodiment transfer | **xVLA** |
| Lower GPU memory | SmolVLA |
| Best multi-task performance | **xVLA** |
