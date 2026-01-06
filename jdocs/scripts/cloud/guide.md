# Cloud Training Guide for VLA Models

This guide covers setting up cloud GPU instances for training Pi0.5 and GROOT 1.6 on the bimanual SO-101 dataset.

## Quick Start

```bash
# 1. Upload dataset to HuggingFace Hub (from local machine)
python jdocs/scripts/data/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/multitasks \
    --repo-id your-username/bimanual-multitasks \
    --private

# 2. SSH to cloud instance and clone repo
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 3. Run training
bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh
# or
bash jdocs/scripts/cloud/groot16/train_groot16_bimanual.sh
```

---

## Cloud Provider Comparison

| Provider | GPU Options | Price/hr | Pros | Cons |
|----------|-------------|----------|------|------|
| **Lambda Labs** | A100, H100 | $1.10-$2.50 | Reliable, good support | Limited availability |
| **RunPod** | A100, H100, 4090 | $0.74-$2.39 | Flexible, spot instances | Variable quality |
| **Vast.ai** | Various | $0.30-$2.00 | Cheapest | Less reliable |
| **Google Cloud** | A100, H100 | $2.50-$4.00 | Enterprise grade | Most expensive |

### Recommended Setup

| Model | GPU | VRAM | Batch Size | Est. Time (120k steps) | Est. Cost |
|-------|-----|------|------------|------------------------|-----------|
| Pi0.5 (LoRA) | A100 40GB | 40GB | 32 | ~8-10 hrs | $20-30 |
| Pi0.5 (LoRA) | RTX 4090 | 24GB | 8 | ~20-24 hrs | $15-25 |
| GROOT 1.6 | A100 40GB | 40GB | 16 | ~6-8 hrs | $15-25 |
| GROOT 1.6 | RTX 4090 | 24GB | 8 | ~12-16 hrs | $10-20 |

---

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n vla python=3.10 -y
conda activate vla

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi05]"  # For Pi0.5
# or
pip install -e ".[groot]"  # For GROOT (if available)

# For GROOT 1.6, also need Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
pip install -e .
```

### Option 2: Docker

```bash
# Pull LeRobot image
docker pull huggingface/lerobot:latest

# Run with GPU
docker run --gpus all -it \
    -v /path/to/data:/data \
    -v /path/to/outputs:/outputs \
    huggingface/lerobot:latest bash
```

---

## Dataset Transfer

### Method 1: HuggingFace Hub (Recommended)

```bash
# Upload from local machine
python jdocs/scripts/data/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/multitasks \
    --repo-id your-username/bimanual-multitasks \
    --private

# Download on cloud instance
huggingface-cli login
# Then training scripts will auto-download
```

### Method 2: Direct Transfer (rsync)

```bash
# From local machine to cloud
rsync -avz --progress \
    ./datasets_bimanuel/multitasks \
    user@cloud-ip:/home/user/lerobot/datasets/

# Compress first for faster transfer
tar -czvf multitasks.tar.gz datasets_bimanuel/multitasks
scp multitasks.tar.gz user@cloud-ip:/home/user/
ssh user@cloud-ip "cd /home/user && tar -xzvf multitasks.tar.gz"
```

### Method 3: Cloud Storage

```bash
# Upload to S3/GCS
aws s3 cp --recursive datasets_bimanuel/multitasks s3://your-bucket/multitasks

# Download on cloud instance
aws s3 cp --recursive s3://your-bucket/multitasks ./datasets/multitasks
```

---

## Training Scripts

### Pi0.5 Training

```bash
# Basic training (uses HuggingFace Hub dataset)
bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# With custom dataset path
DATASET_PATH=/path/to/local/dataset \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# Adjust batch size for GPU memory
BATCH_SIZE=64 MAX_STEPS=100000 \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh
```

### GROOT 1.6 Training

```bash
# First, convert dataset to GROOT format
python jdocs/scripts/cloud/groot16/convert_lerobot_to_groot.py \
    --input ./datasets_bimanuel/multitasks \
    --output ./datasets/multitasks_groot

# Then train
DATASET_PATH=./datasets/multitasks_groot \
    bash jdocs/scripts/cloud/groot16/train_groot16_bimanual.sh
```

---

## Monitoring Training

### Terminal Monitoring

```bash
# Watch training log
tail -f outputs/*/training.log

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Weights & Biases (Optional)

```bash
# Enable W&B logging
export WANDB_API_KEY=your-key
WANDB_ENABLE=true bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh
```

### TensorBoard (Optional)

```bash
# Start tensorboard
tensorboard --logdir outputs/ --port 6006

# Forward port from cloud to local
ssh -L 6006:localhost:6006 user@cloud-ip
# Then open http://localhost:6006
```

---

## Checkpoint Management

### During Training

Checkpoints are saved every `SAVE_STEPS` (default: 5000) to:
- Pi0.5: `outputs/pi05_bimanual_TIMESTAMP/checkpoint-STEP/`
- GROOT: `outputs/groot16_bimanual_TIMESTAMP/checkpoint-STEP/`

### Download Checkpoints

```bash
# From cloud to local
rsync -avz user@cloud-ip:/home/user/lerobot/outputs/pi05_bimanual_*/checkpoint-* ./checkpoints/

# Or use SCP for specific checkpoint
scp -r user@cloud-ip:/path/to/checkpoint-50000 ./checkpoints/
```

### Resume Training

```bash
# Pi0.5
RESUME_FROM=/path/to/checkpoint-50000 \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# GROOT 1.6
RESUME_FROM=/path/to/checkpoint-50000 \
    bash jdocs/scripts/cloud/groot16/train_groot16_bimanual.sh
```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
BATCH_SIZE=4 bash train_pi05_bimanual.sh

# Enable gradient checkpointing (usually already enabled)
GRADIENT_CHECKPOINTING=true bash train_pi05_bimanual.sh
```

### Slow Training

```bash
# Increase number of data workers
NUM_WORKERS=8 bash train_pi05_bimanual.sh

# Use mixed precision (usually already enabled)
DTYPE=bfloat16 bash train_pi05_bimanual.sh
```

### Connection Issues

```bash
# Use screen/tmux for long training
screen -S training
bash train_pi05_bimanual.sh
# Ctrl+A, D to detach
# screen -r training to reattach

# Or use nohup
nohup bash train_pi05_bimanual.sh > training.log 2>&1 &
```

---

## Cost Optimization Tips

1. **Use spot instances** - 50-70% cheaper, but can be interrupted
2. **Start with small runs** - Test with 1000 steps first
3. **Monitor GPU utilization** - Increase batch size if GPU not fully utilized
4. **Download checkpoints regularly** - Don't lose work if instance terminates
5. **Use local RTX 5090** - All models fit in 24GB, just slower

---

## Quick Reference

### Pi0.5 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LORA_RANK` | 16 | LoRA rank (higher = more params) |
| `LORA_ALPHA` | 16 | Must equal rank for 1.0 scaling |
| `LEARNING_RATE` | 2.5e-5 | Learning rate |
| `BATCH_SIZE` | 32 | Batch size (adjust for GPU) |
| `MAX_STEPS` | 120000 | Training steps |

### GROOT 1.6 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `BATCH_SIZE` | 8 | Batch size (24GB GPU) |
| `MAX_STEPS` | 10000 | Training steps |
| `WARMUP_RATIO` | 0.05 | Warmup fraction |

---

## Next Steps

After training completes:

1. **Download checkpoint** to local machine
2. **Run inference test** using `infer_pi05_bimanual.py` or `infer_groot16_bimanual.py`
3. **Evaluate on held-out tasks** (icecream→plate, tissue selection)
4. **Compare with baseline** SmolVLA model
