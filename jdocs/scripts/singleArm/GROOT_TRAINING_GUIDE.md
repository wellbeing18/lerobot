# GROOT Training Guide for SO-101

This guide covers training NVIDIA GROOT N1.5 foundation model on SO-101 robot data using LeRobot.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Configuration](#training-configuration)
5. [Running Training](#running-training)
6. [Inference](#inference)
7. [Troubleshooting](#troubleshooting)
8. [Comparison: LeRobot vs Isaac-GR00T](#comparison-lerobot-vs-isaac-groot)
9. [Best Practices](#best-practices)

---

## Overview

### What is GROOT?

GROOT (Generalized Robot 00 Transformer) is NVIDIA's foundation model for robot manipulation. Key features:

- **Cross-embodiment**: Single model works across different robots
- **Vision-Language-Action**: Combines visual understanding with language instructions
- **Flow Matching**: Uses diffusion-based action prediction
- **Pre-trained**: Leverages massive robotics datasets for generalization

### Model Architecture

```
Input: Images (640x480) + Language + State (6 DOF)
           ↓
    ┌─────────────────────────────────────┐
    │   Eagle2.5-HG Vision-Language       │  ← Pre-trained, frozen during fine-tuning
    │   Backbone (~2.8B params)           │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │   VLLN (Vision-Language LayerNorm)  │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │   State Encoder                      │  ← Trainable (~50M params)
    │   (CategorySpecificMLP)             │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │   DiT Action Head                    │  ← Trainable (~160M params)
    │   (16 layers, flow matching)        │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │   Action Decoder                     │
    │   (Multi-embodiment)                │
    └─────────────────────────────────────┘
           ↓
Output: Action chunk [50 steps x 6 DOF]
```

### LeRobot vs Isaac-GR00T

| Aspect | LeRobot | Isaac-GR00T |
|--------|---------|-------------|
| Supported Versions | **N1.5 only** | N1.5 and N1.6 |
| Dataset Format | LeRobot v3.0 (native) | v2.1 + modality.json |
| Training Command | `lerobot-train` | Custom trainer |
| Configuration | GrootConfig dataclass | YAML + Python |
| Preprocessing | Automatic | Manual |

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| System RAM | 32GB | 64GB+ |
| Storage | 100GB | 300GB+ |
| GPU | RTX 4070 Ti | RTX 4090 / H100 |

### Software Requirements

```bash
# Python 3.10+ required
python --version  # Should be 3.10 or higher

# CUDA 11.8+ required
nvidia-smi  # Check CUDA version
```

### Installation

```bash
# 1. Create conda environment
conda create -n groot python=3.10
conda activate groot

# 2. Install PyTorch with CUDA
pip install "torch>=2.2.1,<3.0.0" "torchvision>=0.21.0,<0.22.0"

# 3. Install Flash Attention (REQUIRED for GROOT)
pip install --no-build-isolation flash-attn==2.7.1.post4

# 4. Install LeRobot with GROOT support
cd /home/jrobot/project/lerobot
pip install -e ".[groot]"

# 5. Verify installation
python -c "from lerobot.policies.groot import GrootPolicy; print('GROOT ready!')"
```

---

## Dataset Preparation

### Data Collection Tips

1. **Camera naming consistency is CRITICAL**
   - Use consistent names: `head`, `left_wrist` (or `wrist`)
   - Names in collection MUST match names in training config

2. **Recommended settings**:
   - Resolution: 640x480 (GROOT will resize to 224x224)
   - Frame rate: 30 FPS
   - Episode duration: 10-30 seconds
   - Episodes: 50-100 minimum for simple tasks

3. **Task description consistency**:
   - Use the same phrasing across similar episodes
   - Example: "Pick up the red cube and place it in the box"

### Dataset Structure (LeRobot v3.0)

```
datasets/pick_and_place/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── episodes.json      # Episode information
│   ├── tasks.json         # Task descriptions
│   └── stats.json         # Normalization statistics
├── data/
│   ├── episodes/
│   │   ├── episode_000.parquet
│   │   ├── episode_001.parquet
│   │   └── ...
│   └── videos/
│       ├── episode_000/
│       │   ├── head.mp4
│       │   └── left_wrist.mp4
│       └── ...
```

### Verify Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

metadata = LeRobotDatasetMetadata(
    repo_id="pick_and_place",
    root="/home/jrobot/project/lerobot/datasets/pick_and_place",
)

print(f"Episodes: {metadata.total_episodes}")
print(f"Total frames: {metadata.total_frames}")
print(f"Features: {list(metadata.features.keys())}")
```

---

## Training Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 10000 | Training steps (10k-20k for pick-and-place) |
| `batch_size` | 8 | Reduce to 4 for 16GB GPUs |
| `learning_rate` | 1e-4 | NVIDIA recommended |
| `tune_llm` | false | Keep backbone frozen |
| `tune_visual` | false | Keep vision encoder frozen |
| `tune_projector` | true | Train state/action encoders |
| `tune_diffusion_model` | true | Train DiT action head |

### Memory Optimization

For 16GB GPUs, enable LoRA:

```bash
# Add to training command
--policy.lora_rank=32 \
--policy.lora_alpha=64 \
--policy.lora_dropout=0.1
```

This reduces trainable parameters from ~210M to ~10M.

### Training Steps by Task Complexity

| Task Type | Recommended Steps |
|-----------|-------------------|
| Simple pick-and-place | 10,000-15,000 |
| Multi-object manipulation | 15,000-20,000 |
| Complex sequences | 20,000+ |

**Warning**: Undertrained models cause robot twitching (oscillating movements).

---

## Running Training

### Using the Training Script

```bash
cd /home/jrobot/project/lerobot

# Make executable
chmod +x jdocs/scripts/train_groot_so101_n15.sh

# Run training
./jdocs/scripts/train_groot_so101_n15.sh
```

### Manual Training Command

```bash
lerobot-train \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.optimizer_lr=1e-4 \
  --policy.use_bf16=true \
  --policy.chunk_size=50 \
  --dataset.repo_id=/home/jrobot/project/lerobot/datasets/pick_and_place \
  --training.batch_size=8 \
  --training.max_steps=10000 \
  --training.save_steps=1000 \
  --output_dir=outputs/groot_so101_n15
```

### Multi-GPU Training

```bash
accelerate launch --multi_gpu --num_processes=2 $(which lerobot-train) \
  --policy.type=groot \
  # ... other parameters
```

### Monitoring Training

Training logs are saved to:
- `outputs/groot_so101_n15/training.log`
- WandB (if enabled): `wandb.project=groot-so101`

Key metrics to watch:
- `loss`: Should decrease steadily
- `learning_rate`: Should follow warmup + decay schedule

---

## Inference

### Using the Inference Script

```bash
python jdocs/scripts/infer_groot_so101_n15.py \
  --checkpoint outputs/groot_so101_n15/checkpoints/last \
  --task "Pick up the red cube and place it in the box" \
  --duration 60
```

### Dry Run (No Robot)

```bash
python jdocs/scripts/infer_groot_so101_n15.py \
  --checkpoint outputs/groot_so101_n15/checkpoints/last \
  --dry-run
```

### Recording Images

```bash
python jdocs/scripts/infer_groot_so101_n15.py \
  --checkpoint outputs/groot_so101_n15/checkpoints/last \
  --record
```

Images saved to: `jdocs/eval_images/inference_groot_*/`

---

## Troubleshooting

### Flash Attention Errors

**Error**: `undefined symbol: _ZN3c104cuda...`

**Solution**:
```bash
pip uninstall flash-attn
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--training.batch_size=4`
2. Enable LoRA: `--policy.lora_rank=32`
3. Set memory config:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### Robot Twitching

**Symptom**: Robot oscillates around fixed position instead of moving

**Cause**: Undertrained model

**Solution**:
- Extend training to 10,000-20,000 steps
- Check open-loop MSE (target < 0.02)
- Visualize dataset for quality issues

### Camera Not Found

**Error**: `Failed to open camera at index X`

**Solution**:
1. List available cameras:
   ```bash
   python -c "import cv2; [print(i) for i in range(10) if cv2.VideoCapture(i).isOpened()]"
   ```
2. Update camera indices in hardware config

### Model Download Issues

**Error**: `Connection timeout` when downloading model

**Solution**:
```bash
# Pre-download model
huggingface-cli download nvidia/GR00T-N1.5-3B --local-dir ~/.cache/huggingface/groot
```

---

## Comparison: LeRobot vs Isaac-GR00T

### When to Use LeRobot (This Guide)

- You want simplicity
- Your dataset is in LeRobot format
- You only need GROOT N1.5
- You want integration with LeRobot ecosystem

### When to Use Isaac-GR00T

- You need GROOT N1.6 (deeper DiT, better performance)
- You need advanced features (state dropout, noise augmentation)
- You have datasets in Isaac-GR00T format
- You're doing research on the model itself

### Key Differences Summary

| Feature | LeRobot | Isaac-GR00T |
|---------|---------|-------------|
| Model versions | N1.5 | N1.5, N1.6 |
| DiT layers | 16 | 16 (N1.5), 32 (N1.6) |
| State augmentation | No | Yes (N1.6) |
| Dataset format | v3.0 | v2.1 + modality.json |
| Learning curve | Lower | Higher |

---

## Best Practices

### Data Collection

1. **Consistent task descriptions**: Use exact same phrasing
2. **Varied conditions**: Different object positions, lighting
3. **Clean demonstrations**: Smooth, successful trajectories only
4. **Camera stability**: Fixed camera positions

### Training

1. **Start with more steps**: Better to overtrain than undertrain
2. **Monitor loss curve**: Should steadily decrease
3. **Save checkpoints frequently**: Every 1000 steps
4. **Use validation**: Reserve episodes for evaluation

### Inference

1. **Match training task description**: Use similar phrasing
2. **Warm up robot**: Let motors stabilize before inference
3. **Start with short durations**: Test before long runs
4. **Log everything**: Enable image recording for debugging

### Debugging

1. **Open-loop evaluation first**: Check predictions match ground truth
2. **Visualize predictions**: Plot predicted vs actual trajectories
3. **Check inference latency**: Should be < 50ms
4. **Verify camera alignment**: Same as training setup

---

## Quick Reference

### Training Command

```bash
./jdocs/scripts/train_groot_so101_n15.sh
```

### Inference Command

```bash
python jdocs/scripts/infer_groot_so101_n15.py -c outputs/groot_so101_n15/checkpoints/last -t "your task"
```

### Key Files

| File | Purpose |
|------|---------|
| `jdocs/scripts/train_groot_so101_n15.sh` | Training script |
| `jdocs/scripts/infer_groot_so101_n15.py` | Inference script |
| `jdocs/scripts/so101_hardware.yaml` | Hardware config |
| `jdocs/investigations/claude_lerobot_groot_plans.md` | Full research doc |

### Useful Links

- [LeRobot GROOT Docs](https://huggingface.co/docs/lerobot/groot)
- [GROOT Model](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)

---

*Guide created: 2025-12-25*
