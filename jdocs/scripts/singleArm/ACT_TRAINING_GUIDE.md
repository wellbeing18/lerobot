# ACT Training and Inference Guide for SO101

This guide covers finetuning ACT (Action Chunking with Transformers) on custom datasets and deploying on the SO101 robot arm.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Files](#files)
- [Training](#training)
- [Inference](#inference)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

**ACT (Action Chunking with Transformers)** is a behavior cloning policy that predicts a chunk of future actions given current observations. Key features:

- **Action Chunking**: Predicts multiple future actions at once (default: 100 steps)
- **Transformer Architecture**: Encoder-decoder with attention mechanisms
- **VAE Training**: Uses variational autoencoder objective for better generalization
- **Multi-modal Input**: Processes both images and robot state

### Why ACT?

| Feature | Benefit |
|---------|---------|
| Action chunking | Reduces compounding errors, smoother motions |
| VAE latent space | Better generalization to unseen scenarios |
| Temporal consistency | Actions are coherent over time |
| Efficient inference | One forward pass generates many actions |

---

## Architecture

```
                    ┌─────────────────┐
                    │   Observation   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────┐ ┌─────────────┐
    │ Image Encoder │ │   State   │ │ VAE Encoder │
    │  (ResNet18)   │ │  Encoder  │ │ (Training)  │
    └───────┬───────┘ └─────┬─────┘ └──────┬──────┘
            │               │              │
            └───────────────┼──────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Transformer Encoder  │
                │    (4 layers x 8h)    │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Transformer Decoder  │
                │      (1 layer)        │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Action Chunk [100]  │
                │  (chunk_size actions) │
                └───────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `train_act_pickplace.sh` | Training script with configurable hyperparameters |
| `infer_act_so101.py` | Real robot inference script |
| `so101_hardware.yaml` | Hardware configuration (ports, cameras) |
| `ACT_TRAINING_GUIDE.md` | This documentation |

### Directory Structure

```
jdocs/scripts/
├── train_act_pickplace.sh    # Training launcher
├── infer_act_so101.py        # Robot inference
├── so101_hardware.yaml       # Hardware config
├── ACT_TRAINING_GUIDE.md     # Documentation
└── logs/                     # Inference logs (auto-created)

outputs/act_pickplace/        # Training outputs (auto-created)
├── checkpoints/
│   ├── 010000/
│   │   └── pretrained_model/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── preprocessor.safetensors
│   │       └── postprocessor.safetensors
│   └── ...
└── train_YYYYMMDD_HHMMSS.log
```

---

## Training

### Quick Start

```bash
# Navigate to lerobot root
cd /home/jrobot/project/lerobot

# Run with defaults (50k steps)
bash jdocs/scripts/train_act_pickplace.sh

# Monitor with tensorboard
tensorboard --logdir outputs/act_pickplace
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Dataset** | | |
| `DATASET_PATH` | `${PROJECT_ROOT}/datasets/pick_and_place` | Path to LeRobot dataset |
| `DATASET_NAME` | `pick_and_place` | Dataset identifier |
| **Training** | | |
| `MAX_STEPS` | `50000` | Total training iterations |
| `BATCH_SIZE` | `8` | Samples per batch |
| `NUM_WORKERS` | `4` | Dataloader workers |
| `SEED` | `1000` | Random seed |
| **ACT Architecture** | | |
| `CHUNK_SIZE` | `100` | Actions predicted per forward pass |
| `N_ACTION_STEPS` | `100` | Actions executed before re-inference |
| `DIM_MODEL` | `512` | Transformer hidden dimension |
| `N_HEADS` | `8` | Attention heads |
| `DIM_FEEDFORWARD` | `3200` | FFN expansion dimension |
| `N_ENCODER_LAYERS` | `4` | Encoder transformer layers |
| `N_DECODER_LAYERS` | `1` | Decoder transformer layers |
| `LATENT_DIM` | `32` | VAE latent dimension |
| `DROPOUT` | `0.1` | Dropout rate |
| **VAE** | | |
| `USE_VAE` | `true` | Enable VAE training objective |
| `KL_WEIGHT` | `10.0` | KL divergence loss weight |
| **Vision** | | |
| `VISION_BACKBONE` | `resnet18` | Image encoder architecture |
| `PRETRAINED_BACKBONE` | `ResNet18_Weights.IMAGENET1K_V1` | Pretrained weights |
| **Optimizer** | | |
| `LEARNING_RATE` | `1e-5` | Main learning rate |
| `LR_BACKBONE` | `1e-5` | Vision backbone learning rate |
| `WEIGHT_DECAY` | `1e-4` | L2 regularization |
| `GRAD_CLIP_NORM` | `10.0` | Gradient clipping threshold |
| **Checkpointing** | | |
| `SAVE_STEPS` | `10000` | Checkpoint save frequency |
| `LOG_FREQ` | `200` | Logging frequency |
| `OUTPUT_DIR` | `outputs/act_pickplace` | Output directory |

### Training Examples

```bash
# Longer training
MAX_STEPS=100000 bash jdocs/scripts/train_act_pickplace.sh

# Larger batch size (requires more GPU memory)
BATCH_SIZE=16 bash jdocs/scripts/train_act_pickplace.sh

# Higher learning rate for faster convergence
LEARNING_RATE=5e-5 bash jdocs/scripts/train_act_pickplace.sh

# Smaller chunk size (faster inference, may be less smooth)
CHUNK_SIZE=50 N_ACTION_STEPS=50 bash jdocs/scripts/train_act_pickplace.sh

# Disable VAE (simpler training, potentially worse generalization)
USE_VAE=false bash jdocs/scripts/train_act_pickplace.sh

# Resume from checkpoint
RESUME_FROM=outputs/act_pickplace/checkpoints/010000/pretrained_model \
  bash jdocs/scripts/train_act_pickplace.sh
```

### Loss Components

ACT training optimizes two losses:

1. **L1 Loss (Reconstruction)**: Mean absolute error between predicted and ground truth actions
   - Primary supervision signal
   - Should decrease steadily during training

2. **KL Divergence Loss** (if `USE_VAE=true`): Regularizes VAE latent space
   - Weighted by `KL_WEIGHT`
   - Helps generalization but may slow convergence
   - Typical values: 0.001 - 0.1 (scaled by KL_WEIGHT=10)

**Total Loss** = L1_loss + KL_WEIGHT × KL_loss

### Training Progress Indicators

| Metric | Good Sign | Warning Sign |
|--------|-----------|--------------|
| `loss` | Decreasing steadily | Stuck or oscillating |
| `l1_loss` | < 0.1 after 10k steps | > 0.5 after 10k steps |
| `kld_loss` | Stable around 0.01-0.1 | Exploding or zero |
| `grad_norm` | < 10 | > 100 (unstable) |
| `lr` | As configured | Unexpected changes |

---

## Inference

### Quick Start

```bash
# Run inference on robot
python jdocs/scripts/infer_act_so101.py \
  -c outputs/act_pickplace/checkpoints/050000/pretrained_model

# Test without robot (dry run)
python jdocs/scripts/infer_act_so101.py \
  -c outputs/act_pickplace/checkpoints/050000/pretrained_model \
  --dry-run
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint, -c` | `outputs/act_pickplace/checkpoints/100000/pretrained_model` | Model checkpoint path |
| `--duration` | `60` | Maximum run duration (seconds) |
| `--device` | `cuda` | Inference device (cuda/cpu) |
| `--dry-run` | `false` | Don't send actions to robot |
| `--record` | `false` | Save images to disk |
| `--hw-config` | `jdocs/scripts/so101_hardware.yaml` | Hardware config path |
| `--dataset` | `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place` | Dataset for normalization stats |

### Inference Examples

```bash
# Basic inference (60 seconds)
python jdocs/scripts/infer_act_so101.py -c outputs/act_pickplace/checkpoints/050000/pretrained_model

# Longer duration
python jdocs/scripts/infer_act_so101.py -c outputs/act_pickplace/checkpoints/050000/pretrained_model --duration 120

# Record images for debugging
python jdocs/scripts/infer_act_so101.py -c outputs/act_pickplace/checkpoints/050000/pretrained_model --record

# Use CPU (slower but works without GPU)
python jdocs/scripts/infer_act_so101.py -c outputs/act_pickplace/checkpoints/050000/pretrained_model --device cpu
```

### Hardware Configuration

Edit `so101_hardware.yaml` if your hardware differs:

```yaml
robot:
  type: so101_follower
  port: /dev/ttyACM1       # Check with: ls /dev/ttyACM*
  id: xlerobot_left_arm    # Must match calibration file

cameras:
  head:
    type: opencv
    index_or_path: 4       # Check with: lerobot-find-cameras
    width: 640
    height: 480
    fps: 30
  left_wrist:
    type: opencv
    index_or_path: 6
    width: 640
    height: 480
    fps: 30
```

### Inference Flow

```
┌──────────────────────────────────────────────────────────┐
│                    Inference Loop (30Hz)                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Capture images from head + wrist cameras             │
│                        ▼                                 │
│  2. Read robot state (6 joint positions)                 │
│                        ▼                                 │
│  3. Format observation dict                              │
│     - Images: (B=1, C=3, H=480, W=640) normalized [0,1]  │
│     - State: (B=1, D=6) float32                          │
│                        ▼                                 │
│  4. Preprocess (normalize using dataset stats)           │
│                        ▼                                 │
│  5. policy.select_action(observation)                    │
│     - Returns single action from internal chunk queue    │
│     - Automatically re-infers when queue empty           │
│                        ▼                                 │
│  6. Postprocess (unnormalize action)                     │
│                        ▼                                 │
│  7. Send action to robot                                 │
│                        ▼                                 │
│  8. Sleep to maintain 30Hz rate                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Hyperparameter Tuning

### When to Adjust Parameters

| Symptom | Try Adjusting |
|---------|---------------|
| Training loss not decreasing | Increase `LEARNING_RATE` (e.g., 5e-5) |
| Training unstable/oscillating | Decrease `LEARNING_RATE`, increase `GRAD_CLIP_NORM` |
| Robot motions jerky | Increase `CHUNK_SIZE`, lower inference re-rate |
| Robot too slow to react | Decrease `CHUNK_SIZE` and `N_ACTION_STEPS` |
| Poor generalization | Enable `USE_VAE`, increase `KL_WEIGHT` |
| GPU out of memory | Decrease `BATCH_SIZE`, use smaller backbone |
| Training too slow | Decrease `CHUNK_SIZE`, use `resnet18` backbone |

### Recommended Configurations

**Small Dataset (<50 episodes)**:
```bash
MAX_STEPS=30000 BATCH_SIZE=4 LEARNING_RATE=1e-5 bash jdocs/scripts/train_act_pickplace.sh
```

**Medium Dataset (50-200 episodes)**:
```bash
MAX_STEPS=50000 BATCH_SIZE=8 LEARNING_RATE=1e-5 bash jdocs/scripts/train_act_pickplace.sh
```

**Large Dataset (>200 episodes)**:
```bash
MAX_STEPS=100000 BATCH_SIZE=16 LEARNING_RATE=5e-5 bash jdocs/scripts/train_act_pickplace.sh
```

**Fast Inference (Lower latency)**:
```bash
CHUNK_SIZE=50 N_ACTION_STEPS=50 bash jdocs/scripts/train_act_pickplace.sh
```

---

## Troubleshooting

### Training Issues

#### "CUDA out of memory"
```bash
# Reduce batch size
BATCH_SIZE=4 bash jdocs/scripts/train_act_pickplace.sh

# Or use gradient checkpointing (if supported)
# Or use smaller images via dataset transforms
```

#### "Loss not decreasing"
1. Check dataset quality (visualize with `lerobot-dataset-viz`)
2. Verify normalization stats are correct
3. Try higher learning rate: `LEARNING_RATE=5e-5`
4. Check for data loading issues in logs

#### "NaN loss"
```bash
# Lower learning rate and increase gradient clipping
LEARNING_RATE=1e-6 GRAD_CLIP_NORM=1.0 bash jdocs/scripts/train_act_pickplace.sh
```

#### "Training very slow"
```bash
# Increase workers
NUM_WORKERS=8 bash jdocs/scripts/train_act_pickplace.sh

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Inference Issues

#### "Camera not found"
```bash
# Find available cameras
lerobot-find-cameras

# Update so101_hardware.yaml with correct indices
```

#### "Robot not responding"
```bash
# Check USB connection
ls /dev/ttyACM*

# Test robot connection
python -c "
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
robot = SO101Follower(SO101FollowerConfig(port='/dev/ttyACM1', id='xlerobot_left_arm'))
robot.connect()
print(robot.get_observation())
robot.disconnect()
"
```

#### "Actions don't match expected range"
- Check that inference uses the same dataset for normalization stats
- Verify `--dataset` argument points to correct path
- Ensure preprocessor/postprocessor were saved with checkpoint

#### "Robot moves erratically"
1. Check camera images are correct (use `--record` flag)
2. Verify observation format matches training
3. Test with `--dry-run` to check predicted actions
4. Compare action statistics in logs with training data range

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'lerobot'` | PYTHONPATH not set | Run from lerobot root or set PYTHONPATH |
| `FileNotFoundError: Checkpoint not found` | Wrong path | Verify checkpoint path exists |
| `RuntimeError: Failed to open camera` | Camera disconnected | Check USB, run `lerobot-find-cameras` |
| `SerialException: could not open port` | Robot disconnected | Check USB, verify port in config |
| `KeyError: 'observation.images.head'` | Missing camera key | Ensure camera names match dataset |

---

## References

### LeRobot Documentation
- ACT Policy: https://huggingface.co/docs/lerobot/act
- Training Guide: https://huggingface.co/docs/lerobot/train
- Dataset Format: https://huggingface.co/docs/lerobot/dataset

### Source Code
- ACT Config: `src/lerobot/policies/act/configuration_act.py`
- ACT Model: `src/lerobot/policies/act/modeling_act.py`
- Training Script: `src/lerobot/scripts/lerobot_train.py`
- Examples: `examples/tutorial/act/`

### Original Paper
- **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**
  - Authors: Tony Z. Zhao, et al.
  - Paper: https://arxiv.org/abs/2304.13705
  - Project: https://tonyzhaozh.github.io/aloha/

### Dataset Information
- Location: `datasets/pick_and_place` (relative to lerobot root)
- Robot: SO101 Follower
- Task: Pick and place block on plate
- Episodes: 40
- Frames: 10,775 @ 30 FPS
- Cameras: head (640x480), left_wrist (640x480)
- Action space: 6-DOF (5 arm joints + gripper)

---

## Appendix: Full Parameter Reference

### ACTConfig (from configuration_act.py)

```python
@dataclass
class ACTConfig:
    # Observation settings
    n_obs_steps: int = 1                    # Observation history length

    # Action prediction
    chunk_size: int = 100                   # Actions predicted per forward pass
    n_action_steps: int = 100               # Actions used before re-inference

    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Transformer architecture
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # VAE settings
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Regularization
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Temporal ensembling (optional, inference only)
    temporal_ensemble_coeff: float | None = None

    # Optimizer defaults
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
```

---

*Last updated: 2024-12-22*
*Compatible with: LeRobot v3.0, ACT policy*
