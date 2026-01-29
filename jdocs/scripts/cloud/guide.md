# Cloud Training Guide for VLA Models

This guide covers setting up cloud GPU instances for training SmolVLA, Pi0.5, and GROOT 1.6 on bimanual SO-101 datasets.

---

## Table of Contents

- [Cloud Training Guide for VLA Models](#cloud-training-guide-for-vla-models)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [SmolVLA (Recommended for Bimanual)](#smolvla-recommended-for-bimanual)
    - [Pi0.5](#pi05)
    - [GROOT 1.6](#groot-16)
  - [HuggingFace Authentication](#huggingface-authentication)
    - [Step 1: Create HuggingFace Account](#step-1-create-huggingface-account)
    - [Step 2: Create Access Token](#step-2-create-access-token)
    - [Step 3: Login via CLI](#step-3-login-via-cli)
    - [Step 4: Verify Login](#step-4-verify-login)
    - [Switching Accounts](#switching-accounts)
  - [Uploading Datasets to HuggingFace](#uploading-datasets-to-huggingface)
    - [Prerequisites](#prerequisites)
    - [Dataset Structure (LeRobot v3.0)](#dataset-structure-lerobot-v30)
    - [Upload Command](#upload-command)
    - [Upload Options](#upload-options)
    - [Examples](#examples)
    - [Troubleshooting Upload](#troubleshooting-upload)
  - [Cloud Provider Comparison](#cloud-provider-comparison)
    - [GPU Pricing by Provider (2026)](#gpu-pricing-by-provider-2026)
    - [Vast.ai: Why Cheaper? Concerns?](#vastai-why-cheaper-concerns)
    - [Persistent Storage (Recommended)](#persistent-storage-recommended)
      - [RunPod Network Volumes](#runpod-network-volumes)
    - [Provider Recommendations](#provider-recommendations)
    - [GPU Selection Guide](#gpu-selection-guide)
    - [Model Finetuning Options](#model-finetuning-options)
      - [SmolVLA (~450M params)](#smolvla-450m-params)
      - [SmolVLA Training Steps for 464 Episodes](#smolvla-training-steps-for-464-episodes)
      - [Pi0.5 (~2.5B params)](#pi05-25b-params)
      - [GROOT 1.6 (~3B params)](#groot-16-3b-params)
    - [GPU Memory \& Batch Size (for 464 Episodes)](#gpu-memory--batch-size-for-464-episodes)
    - [Training Throughput (steps/second)](#training-throughput-stepssecond)
    - [Training Time Estimates (for 464 Episodes)](#training-time-estimates-for-464-episodes)
    - [Training Cost Estimates (for 464 Episodes)](#training-cost-estimates-for-464-episodes)
    - [Cost-Effectiveness Analysis (for 464 Episodes)](#cost-effectiveness-analysis-for-464-episodes)
    - [Recommended GPU by Model (for 464 Episodes)](#recommended-gpu-by-model-for-464-episodes)
    - [Quick Start by Budget](#quick-start-by-budget)
  - [Vast.ai Training Guide](#vastai-training-guide)
    - [Vast.ai Quick Start](#vastai-quick-start)
    - [Vast.ai Templates](#vastai-templates)
      - [Recommended Templates for VLA Training](#recommended-templates-for-vla-training)
      - [How to Select a Template](#how-to-select-a-template)
      - [Template Configuration for VLA Training](#template-configuration-for-vla-training)
      - [Disk Space Requirements](#disk-space-requirements)
      - [Optional: On-Start Script](#optional-on-start-script)
      - [What's Included in PyTorch Template](#whats-included-in-pytorch-template)
      - [Why Not Use a Blank Instance?](#why-not-use-a-blank-instance)
    - [GPU Selection on Vast.ai](#gpu-selection-on-vastai)
    - [SmolVLA on Vast.ai](#smolvla-on-vastai)
      - [SmolVLA Training Steps \& Epochs Analysis](#smolvla-training-steps--epochs-analysis)
      - [Recommended Training Configurations](#recommended-training-configurations)
      - [Memory \& Batch Size Configuration](#memory--batch-size-configuration)
      - [Training Time \& Cost (30,000 steps recommended)](#training-time--cost-30000-steps-recommended)
      - [SmolVLA Commands for Vast.ai](#smolvla-commands-for-vastai)
      - [SmolVLA FP8 Training (H100 Only)](#smolvla-fp8-training-h100-only)
      - [Alternative: Use Fork with Training Script](#alternative-use-fork-with-training-script)
    - [Pi0.5 on Vast.ai](#pi05-on-vastai)
      - [Memory \& Batch Size Configuration](#memory--batch-size-configuration-1)
      - [Training Time \& Cost (120,000 steps)](#training-time--cost-120000-steps)
      - [Pi0.5 Commands for Vast.ai](#pi05-commands-for-vastai)
    - [GROOT 1.6 on Vast.ai](#groot-16-on-vastai)
      - [Memory \& Batch Size Configuration](#memory--batch-size-configuration-2)
      - [Training Time \& Cost (10,000 steps)](#training-time--cost-10000-steps)
      - [GROOT 1.6 Commands for Vast.ai](#groot-16-commands-for-vastai)
    - [Vast.ai Cost Summary](#vastai-cost-summary)
      - [Total Training Cost Comparison (464 Episodes)](#total-training-cost-comparison-464-episodes)
      - [Best Value Recommendations](#best-value-recommendations)
      - [Training All Three Models](#training-all-three-models)
    - [Vast.ai Persistent Storage (Local Volumes)](#vastai-persistent-storage-local-volumes)
      - [Key Features](#key-features)
      - [Workflow: Download Once, Train Multiple Models](#workflow-download-once-train-multiple-models)
      - [Cost Comparison: With vs Without Volume](#cost-comparison-with-vs-without-volume)
      - [Important Limitations](#important-limitations)
      - [CLI Commands Reference](#cli-commands-reference)
      - [When to Use Volumes vs Direct Download](#when-to-use-volumes-vs-direct-download)
    - [Vast.ai Best Practices](#vastai-best-practices)
      - [Before Training](#before-training)
      - [During Training](#during-training)
      - [Checkpointing Strategy](#checkpointing-strategy)
      - [If Instance Gets Interrupted](#if-instance-gets-interrupted)
      - [Download Checkpoints Before Instance Ends](#download-checkpoints-before-instance-ends)
  - [Environment Setup](#environment-setup)
    - [Option 1: Conda (Recommended)](#option-1-conda-recommended)
    - [Option 2: Docker](#option-2-docker)
  - [Dataset Transfer](#dataset-transfer)
    - [Understanding HuggingFace Dataset Caching](#understanding-huggingface-dataset-caching)
    - [Method 1: HuggingFace Hub with Pre-download (Recommended)](#method-1-huggingface-hub-with-pre-download-recommended)
    - [Method 2: Persistent Storage for Cache](#method-2-persistent-storage-for-cache)
    - [Method 3: Direct Transfer (rsync)](#method-3-direct-transfer-rsync)
    - [Method 4: Cloud Storage (S3/GCS)](#method-4-cloud-storage-s3gcs)
    - [Method 5: Direct HuggingFace Hub (Simplest)](#method-5-direct-huggingface-hub-simplest)
    - [Transfer Time Comparison](#transfer-time-comparison)
  - [Training Scripts](#training-scripts)
    - [SmolVLA Training](#smolvla-training)
    - [Pi0.5 Training](#pi05-training)
    - [GROOT 1.6 Training](#groot-16-training)
  - [Monitoring Training](#monitoring-training)
    - [Terminal Monitoring](#terminal-monitoring)
    - [Weights \& Biases (Optional)](#weights--biases-optional)
    - [TensorBoard (Optional)](#tensorboard-optional)
  - [Checkpoint Management](#checkpoint-management)
    - [During Training](#during-training-1)
    - [Download Checkpoints](#download-checkpoints)
    - [Resume Training](#resume-training)
  - [Troubleshooting](#troubleshooting)
    - [Out of Memory (OOM)](#out-of-memory-oom)
    - [Slow Training](#slow-training)
    - [Connection Issues](#connection-issues)
    - [Training Diverges](#training-diverges)
  - [GPU Acceleration Methods for VLA Training](#gpu-acceleration-methods-for-vla-training)
    - [Mixed Precision Training](#mixed-precision-training)
    - [FP8 Training (H100 Only)](#fp8-training-h100-only)
      - [FP8 Support Status](#fp8-support-status)
      - [SmolVLA FP8 Training (H100)](#smolvla-fp8-training-h100)
      - [Enabling FP8 with Accelerate (Manual)](#enabling-fp8-with-accelerate-manual)
      - [TorchAO FP8 (Direct API)](#torchao-fp8-direct-api)
    - [DataLoader Optimization](#dataloader-optimization)
    - [Memory Optimization](#memory-optimization)
      - [Gradient Checkpointing](#gradient-checkpointing)
      - [Pre-cache Dataset to RAM](#pre-cache-dataset-to-ram)
    - [torch.compile (A100/H100)](#torchcompile-a100h100)
    - [TensorFloat32 (TF32)](#tensorfloat32-tf32)
    - [Multi-GPU Training](#multi-gpu-training)
    - [H100 vs A100 Performance Comparison](#h100-vs-a100-performance-comparison)
    - [Quick Optimization Checklist](#quick-optimization-checklist)
    - [References](#references)
  - [Cost Optimization Tips](#cost-optimization-tips)
  - [Quick Reference](#quick-reference)
    - [SmolVLA Key Parameters](#smolvla-key-parameters)
    - [Pi0.5 Key Parameters](#pi05-key-parameters)
    - [GROOT 1.6 Key Parameters](#groot-16-key-parameters)
  - [Next Steps](#next-steps)

---

## Quick Start

### SmolVLA (Recommended for Bimanual)

**Option A: Use existing public dataset (fastest)**
```bash
# 1. SSH to cloud instance and setup
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# 2. Run training (uses public dataset jasmine314342/picknplace-bimanual-464)
# No HuggingFace login required - dataset is public!
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

**Option B: Upload your own dataset first**
```bash
# 1. On LOCAL machine: Login to HuggingFace and upload
huggingface-cli login
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id YOUR_HF_USERNAME/bimanual-dataset

# 2. SSH to cloud instance and setup
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# 3. Run training with your dataset
DATASET_REPO_ID=YOUR_HF_USERNAME/bimanual-dataset \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Pi0.5

```bash
# Same steps 1-3, then:
pip install -e ".[pi05]"
bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh
```

### GROOT 1.6

GROOT requires a separate repo (Isaac-GR00T), Python 3.10, and dataset conversion to v2.1 format.

```bash
# 1. Clone forked Isaac-GR00T repo (with bimanual cloud scripts)
git clone --branch lora/reusable-groot-workflow-rtx5090-fixes \
    https://github.com/wellbeing18/Isaac-GR00T.git /workspace/Isaac-GR00T
cd /workspace/Isaac-GR00T

# 2. Setup Python 3.10 environment (GROOT requires Python 3.10.*)
conda create -n groot python=3.10 -y
conda activate groot

# 3. Install dependencies (flash-attn requires specific order)
pip install torch==2.7.0
pip install numpy psutil ninja
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
pip install jsonlines av huggingface_hub

# 4. Download dataset from HuggingFace Hub
python custom/scripts/cloud/download_hf_dataset.py \
    --repo-id jasmine314342/picknplace-bimanual-464 \
    --output ./datasets_lerobot

# 5. Convert LeRobot v3.0 -> GROOT v2.1 format
python custom/scripts/cloud/convert_bimanual_to_groot.py \
    --input ./datasets_lerobot/picknplace-bimanual-464 \
    --output ./datasets/bimanual_groot

# 6. Start training (A100 40GB recommended)
DATASET_PATH=./datasets/bimanual_groot \
TUNE_VISUAL=true GLOBAL_BATCH_SIZE=16 MAX_STEPS=10000 \
    bash custom/scripts/cloud/train_groot_bimanual.sh
```

---

## HuggingFace Authentication

### Step 1: Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Sign up with email or GitHub/Google
3. Verify your email

### Step 2: Create Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "lerobot-upload")
4. Select **"Write"** permission (required for uploading datasets)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### Step 3: Login via CLI

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login (paste your token when prompted)
huggingface-cli login
```

You'll see:
```
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    ...
Token: <paste your token here>
Add token as git credential? (Y/n) Y
Token is valid.
Your token has been saved to /home/user/.cache/huggingface/token
Login successful
```

### Step 4: Verify Login

```bash
# Check your username
huggingface-cli whoami
```

Output should show your username (e.g., `jasmine314342`).

### Switching Accounts

To login with a different account:
```bash
# Logout current account
huggingface-cli logout

# Login with new account
huggingface-cli login
```

---

## Uploading Datasets to HuggingFace

### Prerequisites

1. HuggingFace account with **Write** token
2. Logged in via `huggingface-cli login`
3. Dataset in LeRobot v3.0 format

### Dataset Structure (LeRobot v3.0)

Your dataset should have this structure:
```
your_dataset/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet
│       ├── file-001.parquet
│       └── ...
├── meta/
│   ├── info.json           # Dataset metadata (required)
│   ├── episodes/           # Episode parquet files (required)
│   │   └── chunk-000/
│   │       └── file-000.parquet
│   ├── stats.json          # Normalization statistics
│   ├── tasks.parquet       # Task descriptions
│   ├── tasks.jsonl         # Task descriptions (for HF upload)
│   ├── episodes.jsonl      # Episode info (for HF upload)
│   └── config.yaml         # Recording config
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       ├── file-000.mp4
    │       └── ...
    └── observation.images.*/
        └── ...
```

### Upload Command

```bash
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/your_dataset \
    --repo-id YOUR_HF_USERNAME/dataset-name
```

```bash
python jdocs/scripts/cloud/upload_dataset_to_hub.py --local-path ./datasets_bimanuel/picknplace_300_144 --repo-id jasmine314342/picknplace-bimanual-464

  Dataset uploaded successfully!
  URL: https://huggingface.co/datasets/jasmine314342/picknplace-bimanual-464
  Visibility: public

  To use this dataset in training:
    --dataset.repo_id=jasmine314342/picknplace-bimanual-464
```

**Important:** Replace `YOUR_HF_USERNAME` with your actual HuggingFace username (NOT your email).

### Upload Options

| Flag | Description |
|------|-------------|
| `--local-path` | Path to local dataset directory (required) |
| `--repo-id` | HuggingFace repo ID: `username/dataset-name` (required) |
| `--private` | Make dataset private (default: public) |
| `--no-videos` | Skip video upload (faster, smaller) |
| `--dry-run` | Validate only, no upload |
| `--tags` | Comma-separated tags (e.g., `bimanual,so101`) |
| `--license` | License type (default: `apache-2.0`) |

### Examples

```bash
# Public dataset with videos
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id jasmine314342/picknplace-bimanual-464

# Private dataset
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id jasmine314342/picknplace-bimanual-464 \
    --private

# Fast upload without videos (for testing)
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id jasmine314342/picknplace-bimanual-464 \
    --no-videos

# Validate before uploading
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id jasmine314342/picknplace-bimanual-464 \
    --dry-run

# With tags for discoverability
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id jasmine314342/picknplace-bimanual-464 \
    --tags bimanual,so101,manipulation,lerobot
```

### Troubleshooting Upload

**Error: "Repository Not Found"**
- Check your username: `huggingface-cli whoami`
- Make sure you're using username, not email
- Re-login: `huggingface-cli logout && huggingface-cli login`

**Error: "Missing required file: meta/episodes.jsonl"**
- Your dataset needs `episodes.jsonl` and `tasks.jsonl` for HF upload
- Generate them from parquet files (see merge script for example)

**Error: "Authentication failed"**
- Run `huggingface-cli login` with a **Write** token
- Check token permissions at https://huggingface.co/settings/tokens

**Slow upload**
- Large video files take time; use `--no-videos` for testing
- Consider using cloud storage (S3/GCS) for very large datasets

---

## Cloud Provider Comparison

### GPU Pricing by Provider (2026)

| Provider | RTX 4090 | A100 40GB | A100 80GB | H100 PCIe | H100 SXM | Best For |
|----------|----------|-----------|-----------|-----------|----------|----------|
| **[RunPod](https://www.runpod.io/pricing)** | $0.34-0.59 | $0.79-1.09 | $1.19-1.64 | $1.99-2.44 | $2.69-3.14 | Best balance of price/features |
| **[Lambda Labs](https://lambdalabs.com/service/gpu-cloud)** | - | $1.10 | $1.29 | $2.49 | $3.29 | Pre-configured ML stacks, reliable |
| **[Vast.ai](https://vast.ai/)** | $0.18-0.35 | $0.50-0.80 | $0.75-1.20 | $1.49-1.99 | - | Lowest prices, marketplace |
| **[Thunder Compute](https://www.thundercompute.com/)** | - | $0.78 | - | $1.47 | - | Budget-friendly |
| **[Hyperstack](https://www.hyperstack.cloud/)** | - | $1.00-1.50 | $1.50-2.00 | $2.00-2.50 | $3.00+ | Zero egress, hibernation |
| **[CoreWeave](https://www.coreweave.com/pricing)** | - | $2.21+ | $2.50+ | $4.76+ | $6.16+ | Enterprise HPC, NVLink clusters |
| **[Jarvis Labs](https://jarvislabs.ai/)** | - | $1.04 | $1.79 | - | - | Beginners, easy setup |
| **AWS/GCP/Azure** | - | $3.50-4.10 | $4.00-5.00 | $5.00-8.00 | - | Enterprise, compliance |

*Prices are approximate and may vary. RunPod has Community (cheaper) and Secure (premium) tiers.*

### Vast.ai: Why Cheaper? Concerns?

**Why cheaper:** Vast.ai is a peer-to-peer marketplace where individuals rent out their GPUs.

| Concern | Details | Mitigation |
|---------|---------|------------|
| **Reliability** | "Instance might disappear if owner needs their gaming rig back" | Checkpoint every 1000 steps |
| **Quality variance** | Some hosts excellent, others have connectivity issues | Check host reviews before renting |
| **Interruptible** | Jobs can be terminated for higher-priority requests | Use for short experiments (<4 hrs) |
| **Support** | Limited compared to managed providers | Be prepared to troubleshoot yourself |
| **Security** | Less controlled than enterprise providers | Don't store sensitive data |

**When Vast.ai is OK:**
- Short experiments (<4 hours)
- Frequent checkpointing enabled
- Budget is primary concern
- You can restart if instance dies

**When to use RunPod/Lambda instead:**
- Multi-day training runs
- Production/research workloads
- Need reliable support
- Data security matters

### Persistent Storage (Recommended)

**Use persistent storage to avoid re-downloading datasets for each training run.**

#### RunPod Network Volumes

| Feature | Details |
|---------|---------|
| **Cost** | $0.07/GB/month (first 1TB), $0.05/GB/month after |
| **Your dataset (~3.5GB)** | ~$0.25/month |
| **Speed** | 200-400 MB/s, peak 10 GB/s |
| **Persistence** | Data survives pod termination |

**Setup:**
```bash
# 1. Create network volume in RunPod console (50GB recommended)
# 2. Mount at /workspace/datasets when creating pod
# 3. Download dataset once:
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jasmine314342/picknplace-bimanual-464',
    repo_type='dataset',
    local_dir='/workspace/datasets/picknplace-bimanual-464'
)
"
# 4. All future pods use the same data - no re-download!
```

**Cost-Benefit Analysis:**
| Approach | First Run | Each Additional Run | 10 Runs Total |
|----------|-----------|---------------------|---------------|
| **No storage** | 20 min download | 20 min download | 200 min |
| **With storage** | 20 min download + $3.50/mo | 0 min | 20 min + $3.50 |

**Savings with storage:** ~3 hours of download time over 10 runs = worth the $3.50/month

### Provider Recommendations

| Use Case | Recommended Provider | Why |
|----------|---------------------|-----|
| **Best Overall** | [RunPod](https://www.runpod.io/) | Good balance of price, reliability, features |
| **Lowest Cost** | [Vast.ai](https://vast.ai/) | Marketplace pricing, 40-60% cheaper |
| **Most Reliable** | [Lambda Labs](https://lambdalabs.com/) | Pre-configured, zero egress fees |
| **Enterprise/HPC** | [CoreWeave](https://www.coreweave.com/) | NVLink clusters, latest GPUs (B200) |
| **Budget + Easy** | [Jarvis Labs](https://jarvislabs.ai/) | Simple setup, good for beginners |

### GPU Selection Guide

| GPU | VRAM | Best For | Performance vs A100 |
|-----|------|----------|---------------------|
| **RTX 4090** | 24GB | Small/medium models, LoRA fine-tuning | ~80% at 10x lower cost |
| **A100 40GB** | 40GB | Most VLA training, good balance | Baseline |
| **A100 80GB** | 80GB | Larger batch sizes, full fine-tuning | 1.5-2x (more memory) |
| **H100 PCIe** | 80GB | Fast training, FP8 support | 2-3x faster |
| **H100 SXM** | 80GB | Fastest single-GPU, NVLink ready | 3-4x faster |

**Note:** H100 offers up to 4x performance vs A100 due to Hopper architecture and FP8 support. Often more cost-effective despite higher hourly rate.

### Model Finetuning Options

#### SmolVLA (~450M params)

| Mode | Trainable | VRAM | Recommended For |
|------|-----------|------|-----------------|
| **Vision + Expert** | ~185M | ~20-24GB | **Bimanual (Recommended)** |
| Expert Only | ~100M | ~15-18GB | Quick experiments |
| Full VLM | ~450M | ~35-40GB | Large datasets (500+ eps) |

**Recommended:** `FREEZE_VISION=false TRAIN_EXPERT_ONLY=true`
- Unfrozen vision encoder learns visual-spatial features for bimanual coordination
- Frozen language model preserves task understanding

#### SmolVLA Training Steps for 464 Episodes

Your dataset: **464 episodes, 118,772 frames**

**Research Summary:**
- [Official SmolVLA docs](https://huggingface.co/docs/lerobot/en/smolvla): 20K steps with batch=64 for ~50 episodes = **~107 epochs**
- [SmolVLA Paper](https://arxiv.org/html/2506.01844v1): 100K steps for simulation benchmarks
- [Vision encoder finetuning](https://github.com/huggingface/lerobot/issues/1774): Unfrozen vision needs more steps but achieves lower loss

**Scaling Analysis:**
- Your dataset is **9.3× larger** than reference (464 vs 50 episodes)
- With more data diversity, fewer epochs needed (diminishing returns)
- Recommended scaling: ~0.3-0.5× the reference epochs → **~32-54 epochs**

| GPU | Batch Size | Recommended Steps | Epochs | Training Time |
|-----|------------|-------------------|--------|---------------|
| **RTX 4090/5090** | 48 | 30,000 | ~12 | 8-10 hrs |
| **A100 40GB** | 96 | **30,000** | **~24** | 3-4 hrs |
| **A100 80GB/H100** | 128 | 30,000 | ~32 | 2.5-3 hrs |

**Recommendation:** Use **30,000 steps** (~24-32 epochs depending on batch size). This balances the official high-epoch recommendation with your larger dataset size.

#### Pi0.5 (~2.5B params)

| Mode | Trainable | VRAM | GPU Required | Dataset Size |
|------|-----------|------|--------------|--------------|
| **LoRA (rank=32)** | ~80M | >30GB | A100 40GB | **<1000 episodes (recommended)** |
| LoRA (rank=16) | ~40M | >22.5GB | RTX 4090 | <1000 episodes |
| Full Finetuning | ~2.5B | >70GB | A100 80GB, H100 | **>10,000 episodes only** |

**IMPORTANT: For your 464 episodes, use LoRA (rank=32)**

Research shows that for datasets <1000 episodes:
- "LoRA often outperforms full fine-tuning by preventing overfitting" ([Source](https://gradientflow.com/lora-or-full-fine-tuning/))
- "Full fine-tuning becomes favorable only with million-scale datasets" ([OpenVLA-OFT](https://openvla-oft.github.io/))
- OpenVLA uses LoRA for 500 demos vs 1M pretraining data

```bash
# Recommended for 464 episodes: LoRA rank=32
# paligemma_variant="gemma_2b_lora"
# action_expert_variant="gemma_300m_lora"
# lora_rank=32, lora_alpha=32
LORA_RANK=32 bash train_pi05_bimanual.sh
```

**When to use Full Finetuning:**
- Dataset >10,000 episodes
- Target domain differs drastically from pretraining
- Have 8x A100 cluster available

#### GROOT 1.6 (~3B params)

| Component | Flag | Default | VRAM Added |
|-----------|------|---------|------------|
| Vision Tower | `--tune_visual` | False | +10-15GB |
| LLM Backbone | `--tune_llm` | False | +15-20GB |
| Projector | `--tune_projector` | True | Minimal |
| DiT (Diffusion) | `--tune_diffusion_model` | True | +5-10GB |

| Mode | Components Trained | VRAM | GPU Required | Dataset Size |
|------|-------------------|------|--------------|--------------|
| Default | Projector + DiT | ~25GB | RTX 4090 | Any |
| **Vision + Action** | Vision + Proj + DiT | ~35GB | **A100 40GB** | **100-1000 eps (recommended)** |
| Full (with LoRA) | All (LoRA on LLM) | ~40GB | A100 40GB | >1000 episodes |

**Recommended for 464 episodes:** Vision + Projector + DiT (partial unfreezing)
```bash
# Unfreeze vision for visual-spatial learning
TUNE_VISUAL=true GLOBAL_BATCH_SIZE=16 bash train_groot16_bimanual.sh
```

**Why GROOT needs only 10k steps (vs Pi0.5's 120k):**
- **Larger DiT (32 layers vs ~16)**: More capacity per step = faster convergence
- **Shorter action horizon (H=16 vs H=50)**: Fewer actions to predict per step
- **Better pretraining**: 300K steps with batch 16384 = strong initialization
- **Embodiment-specific encoders**: Designed for quick adaptation

### GPU Memory & Batch Size (for 464 Episodes)

| GPU | VRAM | SmolVLA | Pi0.5 | GROOT 1.6 |
|-----|------|---------|-------|-----------|
| | | Vision+Expert | **LoRA (rank=32)** | Vision+Proj+DiT |
| **RTX 4090** | 24GB | **batch=48** | LoRA r=16, batch=8 | Default only, batch=8 |
| **A100 40GB** | 40GB | **batch=96** | **LoRA r=32, batch=32** | **Vision+DiT, batch=16** |
| **A100 80GB** | 80GB | **batch=128** | LoRA r=32, batch=64 | Vision+DiT, batch=32 |
| **H100** | 80GB | **batch=128** | LoRA r=32, batch=64 | Vision+DiT, batch=64 |

*Note: Full finetuning for Pi0.5/GROOT is NOT recommended for <1000 episodes (overfitting risk)*

### Training Throughput (steps/second)

*Estimated with gradient checkpointing enabled and recommended finetuning modes for 464 episodes:*

| GPU | SmolVLA (Vision+Expert) | Pi0.5 (LoRA r=32) | GROOT 1.6 (Vision+DiT) |
|-----|-------------------------|-------------------|------------------------|
| **RTX 4090** | ~0.6-0.8 | ~0.8-1.2 | Default: ~0.8-1.2 |
| **A100 40GB** | ~1.2-1.8 | ~1.5-2.5 | Vision+DiT: ~1.0-1.5 |
| **A100 80GB** | ~1.5-2.0 | ~2.0-3.0 | Vision+DiT: ~1.5-2.0 |
| **H100** | ~2.5-3.5 | ~3.0-5.0 | Vision+DiT: ~2.5-3.5 |

### Training Time Estimates (for 464 Episodes)

| Model | Steps | Batch | Finetuning Mode | RTX 4090 | A100 40GB | A100 80GB/H100 |
|-------|-------|-------|-----------------|----------|-----------|----------------|
| **SmolVLA** | **30,000** | 48/96/128 | Vision+Expert | **8-10 hrs** | **3-4 hrs** | **2.5-3 hrs** |
| **Pi0.5** | 120,000 | 16/32/64 | **LoRA r=32** | 28-42 hrs | **13-22 hrs** | 11-17 hrs |
| **GROOT 1.6** | 10,000 | 8/16/32 | Default (Proj+DiT) | 2.3-3.5 hrs | - | - |
| **GROOT 1.6** | 10,000 | 16/32/64 | **Vision+DiT** | - | **1.8-2.8 hrs** | 1.1-1.8 hrs |

### Training Cost Estimates (for 464 Episodes)

*Based on RunPod Community pricing: RTX 4090: $0.34/hr, A100 40GB: $0.79/hr, A100 80GB: $1.19/hr, H100: $1.99/hr*

| Model | Steps | Mode | RTX 4090 | A100 40GB | A100 80GB | H100 |
|-------|-------|------|----------|-----------|-----------|------|
| **SmolVLA** | **30k** | Vision+Expert | **$3-4** | **$2-3** | **$3-4** | **$4-5** |
| **Pi0.5** | 120k | **LoRA r=32** | $10-14 | **$10-17** | $13-20 | $22-34 |
| **GROOT** | 10k | Default | $0.8-1.2 | - | - | - |
| **GROOT** | 10k | **Vision+DiT** | - | **$1.4-2.2** | $1.3-2.1 | $2.2-3.6 |

### Cost-Effectiveness Analysis (for 464 Episodes)

| Model | Best Budget Option | Best Value Option |
|-------|-------------------|-------------------|
| **SmolVLA** | RTX 4090 ($3-4, 8-10 hrs) | **A100 40GB ($2-3, 3-4 hrs)** |
| **Pi0.5** | RTX 4090 LoRA r=16 ($10-14) | **A100 40GB LoRA r=32 ($10-17)** |
| **GROOT 1.6** | RTX 4090 Default ($0.8-1.2) | **A100 40GB Vision+DiT ($1.4-2.2)** |

**Key Insights for 464 Episodes:**
- **SmolVLA is very efficient** - only $2-3 on A100 40GB (30K steps, ~24 epochs)
- **LoRA is optimal for Pi0.5** - prevents overfitting on small datasets
- **GROOT**: Cheapest to train (~$2), Vision+DiT recommended for bimanual
- **A100 40GB best value** - lower cost than RTX 4090, 2.5x faster training

### Recommended GPU by Model (for 464 Episodes)

| Model | Budget Option | Recommended | Notes |
|-------|---------------|-------------|-------|
| **SmolVLA** | RTX 4090 ($3-4) | **A100 40GB ($2-3)** | 30K steps, batch=96, Vision+Expert |
| **Pi0.5** | RTX 4090 ($10-14) | **A100 40GB ($10-17)** | LoRA r=32 for quality |
| **GROOT 1.6** | RTX 4090 ($0.8-1.2) | **A100 40GB ($1.4-2.2)** | Vision+DiT for bimanual |

### Quick Start by Budget

**Tight Budget (<$5):**
```bash
# Best for SmolVLA (~$2-3) or GROOT (~$1-2)
# Use Vast.ai with A100 40GB at $0.65/hr
# SmolVLA: 30K steps, batch=96, ~3-4 hrs = ~$2-3
```

**Moderate Budget ($10-20):**
```bash
# Good for Pi0.5 LoRA (~$10-17)
# Use Vast.ai or RunPod with A100 40GB
# A100 40GB at $0.65/hr × 15 hrs = ~$10
```

**Fast Training ($20-40):**
```bash
# Use RunPod or Lambda Labs with H100
# All models complete faster, Pi0.5 LoRA in 7-11 hrs
```

---

## Vast.ai Training Guide

This section provides detailed configurations for training on Vast.ai, the lowest-cost cloud GPU provider.

### Vast.ai Quick Start

```bash
# 1. Create account at https://vast.ai/
# 2. Add credits (minimum $10 recommended)
# 3. Search for GPU instance:
#    - Filter: A100 40GB or A100 80GB
#    - Sort by: $/hr (lowest first)
#    - Check: DLPerf score > 30, reliability > 95%
# 4. Rent instance and SSH in
# 5. Run setup and training (see commands below)
```

**Instance Selection Tips:**
- Look for **DLPerf score > 30** (measures actual deep learning performance)
- Check **reliability > 95%** (uptime history)
- Prefer hosts with **good reviews** (star ratings)
- Avoid instances with recent interruptions

### Vast.ai Templates

Templates are pre-configured Docker images with software already installed. **You should select a template** - it saves significant setup time.

#### Recommended Templates for VLA Training

| Template | Image | Best For | Notes |
|----------|-------|----------|-------|
| **PyTorch (Vast)** | `vastai/pytorch` | **All VLA training (Recommended)** | PyTorch + CUDA + SSH + Jupyter |
| NVIDIA CUDA | `vastai/base-image` | Custom setups | Base CUDA, need to install PyTorch |
| Ubuntu 22.04 VM | `vastai/kvm` | Full VM access | Slower, more control |

#### How to Select a Template

1. Go to [cloud.vast.ai/templates/](https://cloud.vast.ai/templates/)
2. Click **"PyTorch (Vast)"** template (recommended)
3. Click **"Select & Configure"**
4. Adjust settings if needed:
   - **Disk Space**: 50-100GB (for dataset + checkpoints)
   - **Docker Options**: Leave default
5. Click **"Search"** to find matching instances
6. Select instance and **"Rent"**

#### Template Configuration for VLA Training

```
Template: PyTorch (Vast)
Image:    vastai/pytorch

Recommended Settings:
├── Disk Space:     50 GB (see breakdown below)
├── Docker Options: (leave default)
└── On-start Script: (optional, see below)
```

#### Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Dataset (picknplace-bimanual-464) | ~3.5 GB | Downloads in ~1-3 min |
| LeRobot + Python dependencies | ~8-10 GB | PyTorch, transformers, etc. |
| Model checkpoints (5 saves) | ~10-15 GB | ~2-3 GB per checkpoint |
| HF cache + temp files | ~5 GB | Tokenizers, model weights |
| **Total** | **~30 GB** | |
| **Recommended** | **50 GB** | Safe margin |

**Note:** For larger datasets (>10GB), consider 100GB disk space.

#### Optional: On-Start Script

You can add a script that runs automatically when the instance starts:

```bash
#!/bin/bash
# Auto-setup for VLA training

# Clone LeRobot
cd /root
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install SmolVLA dependencies
pip install -e ".[smolvla]"

# Pre-download dataset to volume (if volume attached)
if [ -d "/data" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jasmine314342/picknplace-bimanual-464',
    repo_type='dataset',
    local_dir='/data/picknplace-bimanual-464'
)
"
fi

echo "Setup complete! Ready for training."
```

#### What's Included in PyTorch Template

| Component | Version (Auto-selected) |
|-----------|------------------------|
| **PyTorch** | 2.x (latest compatible) |
| **CUDA** | Auto-matched to host driver (12.x) |
| **Python** | 3.10+ |
| **SSH** | Enabled (for remote access) |
| **Jupyter** | Enabled (optional notebook access) |

**Note:** Vast.ai automatically selects the best CUDA version based on the host's NVIDIA driver. You don't need to manually match versions.

#### Why Not Use a Blank Instance?

| Approach | Setup Time | Risk |
|----------|------------|------|
| **PyTorch Template** | ~5 min (just install lerobot) | Low - tested image |
| Blank/Base Image | ~30+ min (install CUDA, PyTorch, etc.) | Higher - version conflicts |

**Recommendation:** Always use the **PyTorch (Vast)** template for VLA training.

### GPU Selection on Vast.ai

| GPU | VRAM | Vast.ai Price | Availability | Best For |
|-----|------|---------------|--------------|----------|
| **A100 40GB PCIe** | 40GB | $0.50-0.80/hr | High | SmolVLA, Pi0.5 LoRA, GROOT |
| **A100 80GB PCIe** | 80GB | $0.75-1.20/hr | Medium | Larger batches, all models |
| **A100 80GB SXM** | 80GB | $0.90-1.40/hr | Medium | Fastest A100 variant |
| **H100 PCIe** | 80GB | $1.49-1.99/hr | Low | 2-3x faster than A100 |
| **RTX 4090** | 24GB | $0.18-0.35/hr | High | Budget option, smaller batches |

*Prices vary by host. Sort by $/hr and check DLPerf score.*

### SmolVLA on Vast.ai

**Mode: Vision + Expert (~185M trainable params)**
- `FREEZE_VISION=false` - Unfreezes vision encoder for visual-spatial learning
- `TRAIN_EXPERT_ONLY=true` - Keeps language model frozen

#### SmolVLA Training Steps & Epochs Analysis

**Research Summary:**

| Source | Dataset | Steps | Batch | Epochs | Notes |
|--------|---------|-------|-------|--------|-------|
| [Official HuggingFace Docs](https://huggingface.co/docs/lerobot/en/smolvla) | 50 eps (~12K frames) | 20,000 | 64 | **~107** | Recommended baseline |
| [SmolVLA Paper](https://arxiv.org/html/2506.01844v1) | Simulation | 100,000 | 64 | - | LIBERO benchmarks |
| [SmolVLA Paper](https://arxiv.org/html/2506.01844v1) | Pretraining | 200,000 | 256 | - | 23K episodes |
| [Community Example](https://huggingface.co/masato-ka/smolvla_block_instruction) | Custom | 40,000 | 8 | - | Block instruction |

**Key Insights:**
- Official recommendation uses **~107 epochs** for 50 episodes (high epoch count for robotics)
- [Unfrozen vision needs more training steps](https://github.com/huggingface/lerobot/issues/1774) but achieves lower final loss
- [Frozen vision converges faster](https://arxiv.org/html/2506.01844v1) (~5K steps) but may miss visual-spatial patterns
- [25 episodes was not enough](https://huggingface.co/docs/lerobot/en/smolvla) → data quality/quantity matters

**Epochs Calculation for Your 464 Episodes (118,772 frames):**

```
Epochs = (steps × batch_size) / total_frames
```

| Batch Size | 15K Steps | 20K Steps | 25K Steps | 30K Steps | 40K Steps | 50K Steps |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| **48** (RTX 4090) | 6 eps | 8 eps | 10 eps | 12 eps | 16 eps | 20 eps |
| **96** (A100 40GB) | 12 eps | 16 eps | **20 eps** | 24 eps | **32 eps** | 40 eps |
| **128** (A100 80GB) | 16 eps | 22 eps | 27 eps | **32 eps** | 43 eps | 54 eps |

**Scaling from Official Recommendation:**
- Reference: 50 episodes → 107 epochs → good performance
- Your dataset: 464 episodes (9.3× more data)
- With more data diversity, fewer epochs needed (diminishing returns)
- **Scaling factor:** ~0.3-0.5× epochs → **32-54 epochs recommended**

#### Recommended Training Configurations

| Mode | Epochs | Steps (batch=96) | Time (A100 40GB) | Cost | Use Case |
|------|--------|------------------|------------------|------|----------|
| **Quick Test** | ~10 | 10,000 | 1-1.5 hrs | ~$0.75 | Verify setup, initial evaluation |
| **Conservative** | ~20 | 20,000 | 2-3 hrs | ~$1.50 | Fast iteration, risk of underfitting |
| **Recommended** | **~32** | **30,000** | **3-4 hrs** | **~$2.50** | **Good balance for 464 episodes** |
| **Extended** | ~40-50 | 40,000-50,000 | 5-6 hrs | ~$3.50 | Best quality, slight overfitting risk |
| **Full (per paper)** | ~100+ | 100,000 | 11-14 hrs | ~$8 | Maximum quality, long training |

**Recommendation for 464 Episodes:** Use **30,000-40,000 steps with batch=96** (~25-32 epochs). This balances the official high-epoch recommendation with dataset size scaling.

#### Memory & Batch Size Configuration

**Measured:** Batch=48 uses ~20GB on 24GB GPU (Vision+Expert, gradient checkpointing ON)

| GPU | VRAM | Max Batch | Memory Used | Recommended Batch |
|-----|------|-----------|-------------|-------------------|
| **RTX 4090** | 24GB | ~60 | ~22GB | **48** (tested) |
| **A100 40GB** | 40GB | ~120 | ~35GB | **96** |
| **A100 80GB** | 80GB | ~256 | ~70GB | **128-200** |
| **H100 80GB** | 80GB | ~256 | ~65GB | **128-200** |

#### Training Time & Cost (30,000 steps recommended)

| GPU | Vast.ai $/hr | Batch | Throughput | Time | Total Cost |
|-----|--------------|-------|------------|------|------------|
| **RTX 4090** | $0.25 | 48 | ~1.0 steps/s | 8-10 hrs | **$2-3** |
| **A100 40GB** | $0.65 | 96 | ~2.0-2.5 steps/s | 3-4 hrs | **$2-3** |
| **A100 80GB** | $0.95 | 128 | ~3.0-4.0 steps/s | 2.5-3 hrs | **$2-3** |
| **H100 80GB** | $1.75 | 128 | ~4.0-5.0 steps/s | 1.5-2 hrs | **$3-4** |

#### SmolVLA Commands for Vast.ai

See [Alternative: Use Fork with Training Script](#alternative-use-fork-with-training-script) below for complete setup instructions.

#### SmolVLA FP8 Training (H100 Only)

FP8 mixed precision provides **~1.3-1.5x speedup** over BF16 and **~30% memory reduction** on H100 GPUs, allowing larger batch sizes.

> **Note:** The FP8 script uses torchao's direct API injection method, which works with PyTorch 2.7+ even when the accelerate FP8 backend has compatibility issues.

**Step 1: Install torchao (if not installed)**

```bash
# Install from PyTorch wheel index (NOT PyPI) for version compatibility
pip install torchao --index-url https://download.pytorch.org/whl/cu126
```

**Step 2: Run FP8 Training**

```bash
# FP8 training with torchao (H100 80GB)
FP8_BACKEND=torchao \
DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
BATCH_SIZE=160 \
MAX_STEPS=40000 \
NUM_WORKERS=20 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual_fp8.sh
```

**What the FP8 script does:**
1. Verifies H100 GPU and torchao installation
2. Injects FP8 layers via `torchao.float8.convert_to_float8_training()`
3. Converts all 303 Linear layers to FP8 (layers with dims not divisible by 16 are skipped)
4. Runs training with BF16 base precision + FP8 linear layers

**Performance Comparison (H100 80GB):**

| Precision | Batch Size | Memory | Speed | Training Time (40K steps) |
|-----------|------------|--------|-------|---------------------------|
| BF16 | 128 | ~70GB | ~5-6 it/s | ~2-2.5 hrs |
| **FP8** | **160** | **~55GB** | **~7-8 it/s** | **~1.5-2 hrs** |

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `FP8_BACKEND` | `TE` | FP8 backend: `torchao` (recommended) or `TE` (TransformerEngine) |
| `BATCH_SIZE` | `128` | Can use 160+ with FP8 due to memory savings |
| `MAX_STEPS` | `30000` | Total training steps |
| `GRADIENT_CHECKPOINTING` | `true` | Enable for memory efficiency |
| `NUM_WORKERS` | `8` | DataLoader workers |
| `LOG_FREQ` | `100` | Logging frequency (set to 10 for verbose) |
| `SAVE_STEPS` | `5000` | Checkpoint save frequency |
| `WANDB_ENABLE` | `false` | Enable Weights & Biases logging |

**BF16 Alternative (if FP8 issues occur):**

```bash
# H100 80GB with BF16 (fallback option)
DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
BATCH_SIZE=128 \
MAX_STEPS=40000 \
NUM_WORKERS=20 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

#### Alternative: Use Fork with Training Script

If you prefer using a forked LeRobot repo (e.g., with local modifications like gradient checkpointing):

```bash
# === STEP 1: SETUP (run once after SSH) ===

# Clone your forked repo instead of upstream
git clone https://github.com/wellbeing18/lerobot.git
cd lerobot

# Fix PyTorch version conflict and install
pip uninstall -y torch torchvision torchaudio
pip install -e ".[smolvla]"

# For FP8 training (H100 only), install torchao from PyTorch wheel index:
pip install torchao --index-url https://download.pytorch.org/whl/cu126

# === STEP 2: RUN TRAINING ===

# --- A100 40GB: BF16, batch=64 ---
BATCH_SIZE=64 MAX_STEPS=50000 NUM_WORKERS=20 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# --- A100 80GB: BF16, batch=96 ---
BATCH_SIZE=96 MAX_STEPS=50000 NUM_WORKERS=20 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# --- H100 80GB: BF16, batch=128 ---
DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
BATCH_SIZE=128 MAX_STEPS=40000 NUM_WORKERS=20 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# --- H100 80GB: FP8, batch=160 (FASTEST - ~1.3-1.5x speedup over BF16) ---
FP8_BACKEND=torchao \
DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
BATCH_SIZE=160 \
MAX_STEPS=40000 \
NUM_WORKERS=20 \
LOG_FREQ=10 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual_fp8.sh
```

**Why use a fork?**
- FP8 training support with torchao direct API injection
- In-place operation fixes for FP8 compatibility
- Training scripts in `jdocs/scripts/cloud/` are already included
- Gradient checkpointing support for SmolVLA

**Important:** Do NOT set `HF_HOME` - LeRobot uses its own cache location (`HF_LEROBOT_HOME`), which is different from the HuggingFace Hub cache. Just let LeRobot download automatically.

**Tip:** You can also upload scripts to upstream lerobot:
```bash
scp -P <PORT> -r jdocs/scripts/cloud/ root@<HOST>:~/lerobot/jdocs/scripts/
```

### Pi0.5 on Vast.ai

**Mode: LoRA rank=32 (~80M trainable params)**
- `USE_LORA=true` with `LORA_RANK=32`, `LORA_ALPHA=32`
- Prevents overfitting on 464 episodes
- Steps: 120,000

#### Memory & Batch Size Configuration

| GPU | VRAM | Max Batch | Memory Used | Recommended Batch |
|-----|------|-----------|-------------|-------------------|
| **RTX 4090** | 24GB | 8 | ~22GB | 4-8 |
| **A100 40GB** | 40GB | 32 | ~35GB | 16-32 |
| **A100 80GB** | 80GB | 64 | ~55GB | 32-64 |
| **H100 80GB** | 80GB | 64 | ~50GB | 32-64 |

#### Training Time & Cost (120,000 steps)

| GPU | Vast.ai $/hr | Throughput | Time | Total Cost |
|-----|--------------|------------|------|------------|
| **RTX 4090** | $0.25 | 0.8-1.2 steps/s | 28-42 hrs | **$7-11** |
| **A100 40GB** | $0.65 | 1.5-2.5 steps/s | 13-22 hrs | **$8-14** |
| **A100 80GB** | $0.95 | 2.0-3.0 steps/s | 11-17 hrs | **$10-16** |
| **H100 80GB** | $1.75 | 3.0-5.0 steps/s | 7-11 hrs | **$12-19** |

#### Pi0.5 Commands for Vast.ai

```bash
# === SETUP (run once after SSH) ===
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi05]"

# Verify GPU
nvidia-smi

# === TRAINING COMMANDS ===

# RTX 4090 (24GB) - LoRA r=16, batch=4, ~28-42 hrs, ~$7-11
LORA_RANK=16 \
LORA_ALPHA=16 \
BATCH_SIZE=4 \
GRADIENT_CHECKPOINTING=true \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# A100 40GB - LoRA r=32, batch=16, ~13-22 hrs, ~$8-14 (RECOMMENDED)
LORA_RANK=32 \
LORA_ALPHA=32 \
BATCH_SIZE=16 \
GRADIENT_CHECKPOINTING=true \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# A100 80GB - LoRA r=32, batch=32, ~11-17 hrs, ~$10-16
LORA_RANK=32 \
LORA_ALPHA=32 \
BATCH_SIZE=32 \
GRADIENT_CHECKPOINTING=true \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# H100 80GB - LoRA r=32, batch=32, ~7-11 hrs, ~$12-19
LORA_RANK=32 \
LORA_ALPHA=32 \
BATCH_SIZE=32 \
GRADIENT_CHECKPOINTING=true \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh
```

### GROOT 1.6 on Vast.ai

**Mode: Vision + Projector + DiT (~300M trainable params)**
- `TUNE_VISUAL=true` - Unfreezes vision encoder
- `TUNE_PROJECTOR=true` - Trains projector layers
- `TUNE_DIFFUSION=true` - Trains DiT action head
- `TUNE_LLM=false` - Keeps LLM frozen
- Steps: 10,000

#### Memory & Batch Size Configuration

| GPU | VRAM | Max Batch | Memory Used | Recommended Batch |
|-----|------|-----------|-------------|-------------------|
| **RTX 4090** | 24GB | 8 | ~23GB | 4-8 (Default mode only) |
| **A100 40GB** | 40GB | 24 | ~35GB | 16-24 |
| **A100 80GB** | 80GB | 48 | ~55GB | 32-48 |
| **H100 80GB** | 80GB | 64 | ~50GB | 32-64 |

*Note: RTX 4090 can only run Default mode (Projector + DiT). Vision+DiT requires A100 40GB+.*

#### Training Time & Cost (10,000 steps)

| GPU | Vast.ai $/hr | Throughput | Time | Total Cost |
|-----|--------------|------------|------|------------|
| **RTX 4090** | $0.25 | 0.8-1.2 steps/s | 2.3-3.5 hrs | **$0.6-0.9** |
| **A100 40GB** | $0.65 | 1.0-1.5 steps/s | 1.8-2.8 hrs | **$1.2-1.8** |
| **A100 80GB** | $0.95 | 1.5-2.0 steps/s | 1.4-1.8 hrs | **$1.3-1.7** |
| **H100 80GB** | $1.75 | 2.5-3.5 steps/s | 0.8-1.1 hrs | **$1.4-1.9** |

*GROOT is the cheapest to train due to only 10k steps needed.*

#### GROOT 1.6 Commands for Vast.ai

**Important:** GROOT uses a separate repo (Isaac-GR00T) and requires dataset conversion from LeRobot v3.0 to GROOT v2.1 format.

```bash
# === STEP 1: SETUP (run once after SSH) ===

# 1.1 Clone forked Isaac-GR00T repo (with bimanual cloud scripts)
git clone --branch lora/reusable-groot-workflow-rtx5090-fixes \
    https://github.com/wellbeing18/Isaac-GR00T.git /workspace/Isaac-GR00T
cd /workspace/Isaac-GR00T

# 1.2 Create Python 3.10 environment (GROOT requires Python 3.10.*)
# Check current Python version first:
python --version  # If already 3.10.x, skip conda steps

# If Python is NOT 3.10.x, create conda environment:
conda create -n groot python=3.10 -y
conda activate groot

# 1.3 Install dependencies (in correct order for flash-attn)
# flash-attn requires torch + build deps installed FIRST
pip install torch==2.7.0
pip install numpy psutil ninja
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Now install GROOT and other deps
pip install -e .
pip install jsonlines av huggingface_hub

# 1.4 Verify GPU
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}')"

# === STEP 2: DOWNLOAD DATASET (skip if already cached) ===
# If you already trained SmolVLA on this instance, dataset is cached at:
#   /workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464
# Otherwise, download it:

python custom/scripts/cloud/download_hf_dataset.py \
    --repo-id jasmine314342/picknplace-bimanual-464 \
    --output /workspace/datasets_lerobot

# === STEP 3: CONVERT TO GROOT FORMAT ===
# LeRobot v3.0 -> GROOT v2.1 (per-episode parquet + modality.json)

# If using HF cache (from SmolVLA training):
python custom/scripts/cloud/convert_bimanual_to_groot.py \
    --input /workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 \
    --output /workspace/Isaac-GR00T/datasets/bimanual_groot

# OR if you downloaded in Step 2:
# python custom/scripts/cloud/convert_bimanual_to_groot.py \
#     --input /workspace/datasets_lerobot/picknplace-bimanual-464 \
#     --output /workspace/Isaac-GR00T/datasets/bimanual_groot

# Verify conversion
ls /workspace/Isaac-GR00T/datasets/bimanual_groot/meta/

# === STEP 4: START TRAINING ===

# Choose ONE command based on your GPU:

NUM_WORKERS=8 PIN_MEMORY=true DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot TUNE_VISUAL=true GLOBAL_BATCH_SIZE=24 MAX_STEPS=20000  bash custom/scripts/cloud/train_groot_bimanual.sh

# --- A100 80GB: Vision+DiT, batch=32, ~1.4-1.8 hrs, ~$1.3-1.7 ---
DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot \
TUNE_VISUAL=true \
GLOBAL_BATCH_SIZE=32 \
MAX_STEPS=10000 \
    bash custom/scripts/cloud/train_groot_bimanual.sh

# --- H100 80GB: Vision+DiT, batch=32, ~0.8-1.1 hrs, ~$1.4-1.9 ---
DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot \
TUNE_VISUAL=true \
GLOBAL_BATCH_SIZE=32 \
MAX_STEPS=10000 \
    bash custom/scripts/cloud/train_groot_bimanual.sh

# === STEP 5: MONITOR TRAINING ===

# In another terminal (or use tmux):
watch -n 1 nvidia-smi                                    # GPU utilization
tail -f outputs/groot16_bimanual_*/training.log          # Training progress

# === STEP 6: DOWNLOAD CHECKPOINTS ===
# From your LOCAL machine after training:
scp -r root@<instance-ip>:/workspace/Isaac-GR00T/outputs/groot16_bimanual_*/checkpoint-* ./
```

**Alternative: One-Command Setup Script**

If you prefer a single command that does everything (clone, download, convert, train):

```bash
# Download and run setup script
cd /workspace
curl -O https://raw.githubusercontent.com/wellbeing18/Isaac-GR00T/lora/reusable-groot-workflow-rtx5090-fixes/custom/scripts/cloud/setup_groot_bimanual.sh
chmod +x setup_groot_bimanual.sh

# Run with default settings (downloads dataset, converts, trains)
bash setup_groot_bimanual.sh

# Or with custom settings (A100 40GB recommended)
TUNE_VISUAL=true GLOBAL_BATCH_SIZE=16 MAX_STEPS=10000 bash setup_groot_bimanual.sh

# Or setup only (no training) - useful for debugging
SKIP_TRAINING=true bash setup_groot_bimanual.sh
```

### Vast.ai Cost Summary

#### Total Training Cost Comparison (464 Episodes)

| Model | Mode | RTX 4090 | A100 40GB | A100 80GB | H100 80GB |
|-------|------|----------|-----------|-----------|-----------|
| **SmolVLA** | Vision+Expert (30k) | $2-3 | **$2-3** | $2-3 | $3-4 |
| **Pi0.5** | LoRA r=32 (120k) | $7-11 | **$8-14** | $10-16 | $12-19 |
| **GROOT 1.6** | Vision+DiT (10k) | $0.6-0.9* | **$1.2-1.8** | $1.3-1.7 | $1.4-1.9 |

*RTX 4090 can only run GROOT in Default mode (Projector + DiT only).*

#### Best Value Recommendations

| Model | Best Budget | Best Value | Best Speed |
|-------|-------------|------------|------------|
| **SmolVLA** | RTX 4090 ($2-3) | **A100 40GB ($2-3)** | H100 ($3-4) |
| **Pi0.5** | RTX 4090 ($7-11) | **A100 40GB ($8-14)** | H100 ($12-19) |
| **GROOT 1.6** | RTX 4090 ($0.6-0.9) | **A100 40GB ($1.2-1.8)** | H100 ($1.4-1.9) |

**Recommendation:** A100 40GB offers the best balance of cost and training time on Vast.ai.

#### Training All Three Models

| Approach | GPUs | Total Time | Total Cost |
|----------|------|------------|------------|
| **Sequential (1 GPU)** | A100 40GB | ~18-26 hrs | **$12-18** |
| **Parallel (3 GPUs)** | 3× A100 40GB | ~13-22 hrs | **$12-18** |
| **Budget Sequential** | RTX 4090 | ~40-58 hrs | **$11-16** |

*SmolVLA: 30K steps (batch=96, ~24 epochs), Pi0.5: 120K steps, GROOT: 10K steps*

### Vast.ai Persistent Storage (Local Volumes)

Vast.ai offers **Local Volumes** - persistent Docker volumes that survive instance restarts and deletions. This allows you to download the dataset once and reuse it across multiple training runs.

#### Key Features

| Feature | Details |
|---------|---------|
| **Persistence** | Data survives instance deletion |
| **Sharing** | One volume can be shared across multiple containers on same machine |
| **Mount point** | Automatically mounts at `/data` |
| **Pricing** | Marketplace-based (varies by host, typically ~$0.02-0.05/GB/month) |
| **Limitation** | Machine-specific - cannot move between physical machines |

#### Workflow: Download Once, Train Multiple Models

```bash
# === STEP 1: Create Volume + First Instance ===
# Via CLI (recommended):
vastai search volumes  # Find available volumes
vastai create volume <offer_id> -s 50 -n "vla-datasets"  # 50GB volume

# Or via GUI:
# 1. Go to Search page
# 2. Select template, click "Add volume"
# 3. Set size to 50GB, rent instance

# === STEP 2: Download Dataset (one-time) ===
# SSH into instance, dataset will persist in /data
cd /data

# Download from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jasmine314342/picknplace-bimanual-464',
    repo_type='dataset',
    local_dir='/data/picknplace-bimanual-464'
)
print('Dataset downloaded to /data/picknplace-bimanual-464')
"

/workspace/.hf_home/hub/datasets--jasmine314342--picknplace-bimanual-464/snapshots/71e9f233beaca02dfdb7a35baa111eb511925c70

ssh -p 11761 root@ssh6.vast.ai -L 8080:localhost:8080

# Verify download (~3.5GB)
du -sh /data/picknplace-bimanual-464

# === STEP 3: Train SmolVLA ===
cd ~/lerobot
DATASET_PATH=/data/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Save checkpoint to volume before deleting instance
cp -r outputs/smolvla_*/checkpoints /data/smolvla_checkpoints/

# === STEP 4: Delete Instance, Keep Volume ===
# Delete instance from Vast.ai console (volume persists!)

# === STEP 5: Rent New Instance with Same Volume ===
# From Storage page, click "Rent instance using this volume"
# Or via CLI:
vastai create instance <new_offer_id> --env '-v vla-datasets:/data'

# === STEP 6: Train Pi0.5 (no re-download!) ===
# Dataset is already at /data/picknplace-bimanual-464
cd ~/lerobot
pip install -e ".[pi05]"

DATASET_PATH=/data/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
    bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# === STEP 7: Train GROOT 1.6 ===
pip install -e ".[groot]"

# Convert dataset (save to volume)
python jdocs/scripts/cloud/groot16/convert_lerobot_to_groot.py \
    --input /data/picknplace-bimanual-464 \
    --output /data/picknplace_groot

DATASET_PATH=/data/picknplace_groot \
TUNE_VISUAL=true \
    bash jdocs/scripts/cloud/groot16/train_groot16_bimanual.sh
```

#### Cost Comparison: With vs Without Volume

**For small datasets (~3.5GB like picknplace-bimanual-464):**

| Scenario | Downloads | Download Time | Storage Cost | Recommendation |
|----------|-----------|---------------|--------------|----------------|
| **No volume (3 runs)** | 3× 3.5GB | 3× 2 min = 6 min | $0 | **Recommended** |
| **With volume (3 runs)** | 1× 3.5GB | 1× 2 min = 2 min | ~$0.15/month | Overkill |

**For 3.5GB dataset: Just download each time** - setup overhead for volumes isn't worth 4 minutes saved.

**For large datasets (>10GB):**

| Scenario | Downloads | Download Time | Storage Cost | Recommendation |
|----------|-----------|---------------|--------------|----------------|
| **No volume (3 runs)** | 3× 30GB | 3× 15 min = 45 min | $0 | Time-wasting |
| **With volume (3 runs)** | 1× 30GB | 1× 15 min = 15 min | ~$1.50/month | **Recommended** |

**For >10GB datasets: Use volumes** - saves significant download time

#### Important Limitations

| Limitation | Details | Workaround |
|------------|---------|------------|
| **Machine-specific** | Volume tied to one physical machine | Clone volume to new machine if needed |
| **Size is permanent** | Cannot resize after creation | Create new larger volume, copy data |
| **CLI-only (2025)** | UI support coming soon | Use `vastai` CLI commands |
| **Same machine required** | New instance must be on same host | Filter by "Rent instance using this volume" |

#### CLI Commands Reference

```bash
# Search for available volumes
vastai search volumes

# Create a volume (50GB example)
vastai create volume <offer_id> -s 50 -n "my-volume"

# List your volumes
vastai show volumes

# Create instance with existing volume
vastai create instance <offer_id> --env '-v my-volume:/data'

# Clone volume to different machine
vastai clone volume <volume_id> <dest_contract_id>

# Delete volume (must delete instances first!)
vastai delete volume <volume_id>
```

#### When to Use Volumes vs Direct Download

| Dataset Size | Use Case | Recommendation |
|--------------|----------|----------------|
| **<5GB** | Any | **Direct download** (2 min, not worth volume setup) |
| **5-10GB** | Single run | Direct download |
| **5-10GB** | Multiple runs | Either (marginal benefit) |
| **>10GB** | Any | **Use volume** (significant time savings) |

**For your 3.5GB dataset: Just download each time** - simpler and only adds ~2 min per run.

### Vast.ai Best Practices

#### Before Training

```bash
# 1. Check GPU health
nvidia-smi -q | grep -E "Product Name|Memory|Temperature"

# 2. Test GPU with quick training (10 steps)
MAX_STEPS=10 SAVE_STEPS=5 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# 3. Use screen/tmux (ESSENTIAL for Vast.ai)
screen -S training
# ... run training ...
# Ctrl+A, D to detach
# screen -r training to reattach
```

#### During Training

```bash
# Monitor in separate terminal
watch -n 5 nvidia-smi

# Check training progress
tail -f outputs/*/training.log
```

#### Checkpointing Strategy

```bash
# For Vast.ai, checkpoint more frequently due to potential interruptions
# SmolVLA: every 2000 steps (default 5000)
SAVE_STEPS=2000 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Pi0.5: every 5000 steps (default 5000)
SAVE_STEPS=5000 bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# GROOT: every 1000 steps (default 1000)
SAVE_STEPS=1000 bash jdocs/scripts/cloud/groot16/train_groot16_bimanual.sh
```

#### If Instance Gets Interrupted

```bash
# 1. Rent new instance (same GPU type)
# 2. Re-run setup
# 3. Find latest checkpoint
ls -la outputs/*/checkpoint-*

# 4. Resume training
RESUME_FROM=outputs/smolvla_bimanual_xxx/checkpoints/checkpoint-20000 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

#### Download Checkpoints Before Instance Ends

FP8_BACKEND=torchao DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 DATASET_NAME=picknplace-bimanual-464 GRADIENT_CHECKPOINTING=false BATCH_SIZE=128 MAX_STEPS=40000 NUM_WORKERS=20 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual_fp8.sh

FP8_BACKEND=torchao DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 DATASET_NAME=picknplace-bimanual-464 BATCH_SIZE=160 MAX_STEPS=40000 NUM_WORKERS=20 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual_fp8.sh



```bash
# On LOCAL machine - download checkpoints
rsync -avz --progress -e "ssh -p 17686 -i ~/.ssh/id_ed25519" root@ssh9.vast.ai:/workspace/lerobot/outputs/smolvla_bimanual_20260128_231825/checkpoints/036000/pretrained_model ./36000

rsync -avz --progress -e "ssh -p 17686 -i ~/.ssh/id_ed25519" root@ssh9.vast.ai:/workspace/Isaac-GR00T/outputs/groot16_bimanual_20260129_145515/checkpoint-17500 ./checkpoint-17500

rsync -avz --progress -e "ssh -p 17686 -i ~/.ssh/id_ed25519" root@ssh9.vast.ai:/workspace/Isaac-GR00T/outputs/groot16_bimanual_20260129_145515/checkpoint-20000 ./checkpoint-20000

ACCELERATE_MIXED_PRECISION=fp8 NUM_WORKERS=8 PIN_MEMORY=true DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot TUNE_VISUAL=true GLOBAL_BATCH_SIZE=32 MAX_STEPS=20000  bash custom/scripts/cloud/train_groot_bimanual.sh

NUM_WORKERS=8 PIN_MEMORY=true DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot TUNE_VISUAL=true GLOBAL_BATCH_SIZE=24 MAX_STEPS=20000  bash custom/scripts/cloud/train_groot_bimanual.sh

RESUME_FROM=outputs/groot16_bimanual_20260129_122806/checkpoint-4000 NUM_WORKERS=8 SAVE_TOTAL_LIMIT=4 SAVE_STEPS=2500 PIN_MEMORY=true DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot TUNE_VISUAL=true GLOBAL_BATCH_SIZE=24 MAX_STEPS=20000 bash custom/scripts/cloud/train_groot_bimanual.sh 

DATASET_PATH=/workspace/Isaac-GR00T/datasets/bimanual_groot \
TUNE_VISUAL=true \
GLOBAL_BATCH_SIZE=24 \
MAX_STEPS=20000 \
    bash custom/scripts/cloud/train_groot_bimanual.sh

rsync -avz --progress \
    user@vast-instance:/root/lerobot/outputs/*/checkpoints \
    ./checkpoints_backup/

# Or use SCP for specific checkpoint
scp -r user@vast-instance:/root/lerobot/outputs/*/checkpoints/last ./
```

---

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# 1. Verify GPU is available
nvidia-smi  # Should show your GPU (A100, H100, RTX 4090, etc.)

# 2. Create conda environment
conda create -n vla python=3.10 -y
conda activate vla

# 3. Install PyTorch with CUDA
# Check your CUDA version first: nvidia-smi (top right shows CUDA version)
# For CUDA 12.1+ (most cloud instances):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4+ (newer instances):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Clone and install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# For SmolVLA (recommended for bimanual)
pip install -e ".[smolvla]"

# For Pi0.5
# pip install -e ".[pi05]"

# For GROOT
# pip install -e ".[groot]"

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# PyTorch: 2.x.x
# CUDA available: True
# GPU: NVIDIA A100-SXM4-80GB (or similar)

# 6. (Optional) Login to HuggingFace
# Required ONLY if:
#   - Uploading datasets to HuggingFace Hub
#   - Accessing PRIVATE datasets
# NOT required for public datasets like jasmine314342/picknplace-bimanual-464
huggingface-cli login
```

**Note on HuggingFace Login:**
- The default dataset `jasmine314342/picknplace-bimanual-464` is **public** - no login required for training
- Login is only required for uploading datasets or accessing private datasets
- If you get "401 Unauthorized" errors, run `huggingface-cli login`

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

### Understanding HuggingFace Dataset Caching

When using HuggingFace Hub datasets, the data is cached locally after the first download:

| Scenario | Downloads? | Notes |
|----------|-----------|-------|
| First training run | Yes (~10-20 min) | Cached to `~/.cache/huggingface/hub/` |
| Second run (same instance) | No | Uses cached data |
| New instance / terminated | Yes | Cache is lost |
| Spot instance interrupted | Yes | Cache is lost |

**To avoid re-downloading on every new instance, use one of the methods below.**

### Method 1: HuggingFace Hub with Pre-download (Recommended)

Pre-download the dataset to a local directory, then train from local path:

```bash
# Step 1: Download dataset to local directory (one-time)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jasmine314342/picknplace-bimanual-464',
    repo_type='dataset',
    local_dir='./datasets/picknplace-bimanual-464'
)
print('Download complete!')
"

# Step 2: Train with local path (no re-download needed)
DATASET_PATH=./datasets/picknplace-bimanual-464 \
DATASET_NAME=picknplace-bimanual-464 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

**Advantage:** If instance restarts, just re-run Step 1 (fast if using persistent storage).

### Method 2: Persistent Storage for Cache

Mount persistent storage to the HuggingFace cache directory:

```bash
# Create persistent directory (provider-specific location)
# Lambda Labs: /home/ubuntu/persistent/
# RunPod: /workspace/
# Vast.ai: /workspace/

# Example for Lambda Labs
mkdir -p /home/ubuntu/persistent/huggingface
rm -rf ~/.cache/huggingface  # Remove existing cache
ln -s /home/ubuntu/persistent/huggingface ~/.cache/huggingface

# Now run training - cache persists across instance restarts
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Method 3: Direct Transfer (rsync)

Best for multiple training runs - transfer once, use many times:

```bash
# From LOCAL machine: transfer dataset to cloud
rsync -avz --progress \
    ./datasets_bimanuel/picknplace_300_144 \
    user@cloud-ip:/data/

# On CLOUD instance: train with local path
DATASET_PATH=/data/picknplace_300_144 \
DATASET_NAME=picknplace_300_144 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

**For faster transfer, compress first:**
```bash
# On local machine
tar -czvf dataset.tar.gz datasets_bimanuel/picknplace_300_144
scp dataset.tar.gz user@cloud-ip:/data/

# On cloud instance
cd /data && tar -xzvf dataset.tar.gz
```

### Method 4: Cloud Storage (S3/GCS)

For teams or frequent cloud training:

```bash
# One-time: Upload to S3/GCS from local machine
aws s3 cp --recursive ./datasets_bimanuel/picknplace_300_144 \
    s3://your-bucket/picknplace_300_144

# On each cloud instance: download from S3 (faster than rsync)
aws s3 cp --recursive s3://your-bucket/picknplace_300_144 /data/picknplace_300_144

# Train with local path
DATASET_PATH=/data/picknplace_300_144 \
DATASET_NAME=picknplace_300_144 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Method 5: Direct HuggingFace Hub (Simplest)

If you don't mind re-downloading on new instances:

```bash
# Just run training - dataset downloads automatically
# No HF login needed for public datasets like jasmine314342/picknplace-bimanual-464
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

**To use your own HuggingFace dataset:**
```bash
# 1. Upload from local machine (see "Uploading Datasets" section above)
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id YOUR_HF_USERNAME/bimanual-dataset

# 2. On cloud instance: run training with your dataset
DATASET_REPO_ID=YOUR_HF_USERNAME/bimanual-dataset \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Transfer Time Comparison

| Method | First Instance | Subsequent Runs | New Instance |
|--------|---------------|-----------------|--------------|
| HF Hub (no cache) | ~15-20 min | Instant | ~15-20 min |
| HF Hub + persistent storage | ~15-20 min | Instant | Instant |
| Pre-download to local | ~15-20 min | Instant | ~15-20 min |
| rsync from local | ~20-30 min | Instant | ~20-30 min |
| S3/GCS | ~5-10 min | Instant | ~5-10 min |

**Recommendation:** Use **Method 2 (Persistent Storage)** if your cloud provider supports it, otherwise use **Method 1 (Pre-download)** for reliability.

---

## Training Scripts

### SmolVLA Training

SmolVLA supports different fine-tuning strategies:

| Mode | Trainable Params | Command | Use Case |
|------|------------------|---------|----------|
| **Vision + Expert** | ~185M | `FREEZE_VISION=false TRAIN_EXPERT_ONLY=true` | Recommended for bimanual |
| Expert Only | ~100M | `FREEZE_VISION=true TRAIN_EXPERT_ONLY=true` | Fast, less memory |
| Full VLM | ~450M | `FREEZE_VISION=false TRAIN_EXPERT_ONLY=false` | Large datasets only |

**Recommended for bimanual: Vision + Expert mode** (default in script)
- Unfrozen vision encoder allows visual-spatial learning important for bimanual coordination
- Frozen language model preserves task understanding capabilities

```bash
# FIRST: Test with a few steps to verify setup (recommended before long runs)
MAX_STEPS=10 SAVE_STEPS=5 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
# If this works, you'll see loss values decreasing. Then run full training below.

# Basic training with HuggingFace Hub dataset (recommended)
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Custom HuggingFace dataset
DATASET_REPO_ID=your-username/your-dataset \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Local dataset (must exist on cloud instance)
DATASET_PATH=/data/my_dataset DATASET_NAME=my_dataset \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Adjust batch size for GPU memory
BATCH_SIZE=64 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh  # A100 80GB
BATCH_SIZE=32 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh  # A100 40GB
BATCH_SIZE=8 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh   # RTX 4090

# Enable Weights & Biases logging
WANDB_ENABLE=true WANDB_PROJECT=smolvla-bimanual \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Custom output directory (useful for cloud storage mounts)
OUTPUT_DIR=/data/outputs/smolvla_run1 \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

**Important:** SmolVLA has NO native bimanual support. It treats 12 DOF actions as a flat vector without arm-specific handling.

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

**Note:** GROOT uses a separate repo (Isaac-GR00T) with its own scripts. See [GROOT 1.6 on Vast.ai](#groot-16-on-vastai) for full setup.

```bash
# From Isaac-GR00T repo (NOT lerobot repo)
cd /workspace/Isaac-GR00T  # or ~/Isaac-GR00T locally

# One-command setup (downloads dataset, converts, trains)
bash custom/scripts/cloud/setup_groot_bimanual.sh

# Or manual steps:
# 1. Download dataset from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download('jasmine314342/picknplace-bimanual-464',
                  repo_type='dataset', local_dir='./datasets_lerobot/picknplace-bimanual-464')
"

# 2. Convert to GROOT format
python custom/scripts/cloud/convert_bimanual_to_groot.py \
    --input ./datasets_lerobot/picknplace-bimanual-464 \
    --output ./datasets/bimanual_groot

# 3. Train (A100 40GB with Vision+DiT recommended)
DATASET_PATH=./datasets/bimanual_groot \
TUNE_VISUAL=true GLOBAL_BATCH_SIZE=16 \
    bash custom/scripts/cloud/train_groot_bimanual.sh
```

---

## Monitoring Training

### Terminal Monitoring

```bash
# Watch training log
tail -f outputs/*/training.log

# Check GPU utilization (continuous monitoring)
watch -n 1 nvidia-smi

# Query specific GPU info
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### Weights & Biases (Optional)

```bash
# Enable W&B logging for SmolVLA
WANDB_ENABLE=true WANDB_PROJECT=smolvla-bimanual \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Enable W&B logging for Pi0.5
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
- SmolVLA: `outputs/smolvla_bimanual_cloud_TIMESTAMP/checkpoints/`
- Pi0.5: `outputs/pi05_bimanual_TIMESTAMP/checkpoint-STEP/`
- GROOT: `outputs/groot16_bimanual_TIMESTAMP/checkpoint-STEP/`

### Download Checkpoints

**For Vast.ai (replace `<PORT>` and `<HOST>` with your instance details):**

```bash
# === FIND YOUR CHECKPOINT PATH (on cloud instance) ===
ls -la outputs/
# Example output: smolvla_bimanual_20260128_231914

ls -la outputs/smolvla_bimanual_*/checkpoints/
# Example output: 005000, 010000, 015000, ..., last

# === DOWNLOAD TO LOCAL MACHINE ===

# Option 1: Download latest checkpoint only (recommended, ~2-3GB)
scp -r -P <PORT> root@<HOST>:/workspace/lerobot/outputs/smolvla_bimanual_*/checkpoints/last ./checkpoints/

# Option 2: Download all checkpoints (~10-15GB)
scp -r -P <PORT> root@<HOST>:/workspace/lerobot/outputs/smolvla_bimanual_*/checkpoints ./checkpoints/

# Option 3: Download specific checkpoint
scp -r -P <PORT> root@<HOST>:/workspace/lerobot/outputs/smolvla_bimanual_*/checkpoints/050000 ./checkpoints/

# === EXAMPLE WITH ACTUAL VALUES ===
# If your Vast.ai SSH is: ssh -p 11761 root@ssh6.vast.ai
scp -r -P 11761 root@ssh6.vast.ai:/workspace/lerobot/outputs/smolvla_bimanual_*/checkpoints/last ./checkpoints/

# === USING RSYNC (faster for large files, shows progress) ===
rsync -avz --progress -e "ssh -p <PORT>" \
    root@<HOST>:/workspace/lerobot/outputs/smolvla_bimanual_*/checkpoints/last \
    ./checkpoints/
```

**For other cloud providers (RunPod, Lambda, etc.):**

```bash
# Standard SCP
scp -r user@cloud-ip:/home/user/lerobot/outputs/smolvla_bimanual_*/checkpoints/last ./checkpoints/

# Rsync with progress
rsync -avz --progress user@cloud-ip:/home/user/lerobot/outputs/smolvla_bimanual_*/checkpoints ./checkpoints/
```

**Pi0.5 and GROOT checkpoints:**

```bash
# Pi0.5
scp -r -P <PORT> root@<HOST>:/workspace/lerobot/outputs/pi05_bimanual_*/checkpoints/last ./checkpoints/

# GROOT 1.6
scp -r -P <PORT> root@<HOST>:/workspace/lerobot/outputs/groot16_bimanual_*/checkpoint-* ./checkpoints/
```

### Resume Training

```bash
# SmolVLA
RESUME_FROM=/path/to/checkpoint \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

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
# SmolVLA - reduce batch size
BATCH_SIZE=8 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Pi0.5 - reduce batch size
BATCH_SIZE=4 bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh

# Enable gradient checkpointing (usually already enabled by default)
GRADIENT_CHECKPOINTING=true bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Slow Training

```bash
# Increase number of data workers
NUM_WORKERS=8 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Use mixed precision (usually already enabled)
DTYPE=bfloat16 bash jdocs/scripts/cloud/pi05/train_pi05_bimanual.sh
```

### Connection Issues

```bash
# Use screen/tmux for long training
screen -S training
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
# Ctrl+A, D to detach
# screen -r training to reattach

# Or use nohup
nohup bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh > training.log 2>&1 &
```

### Training Diverges

```bash
# Lower learning rate
LEARNING_RATE=5e-5 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# Increase warmup steps
WARMUP_STEPS=2000 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

---

## GPU Acceleration Methods for VLA Training

This section covers advanced optimization techniques to maximize training speed on A100/H100 GPUs.

### Mixed Precision Training

| GPU | Recommended Precision | Speedup vs FP32 | Notes |
|-----|----------------------|-----------------|-------|
| **A100** | BF16 | 2-2.5x | Native support, stable |
| **H100** | BF16 or FP8 | 3-6x | FP8 requires TransformerEngine |
| **RTX 4090** | BF16 | 1.5-2x | Good for smaller batches |

**BF16 is enabled by default** in most VLA training scripts. To verify:

```bash
# Check if bf16 is being used (look for "bfloat16" in logs)
grep -i "bf16\|bfloat" training.log
```

### FP8 Training (H100 Only)

FP8 provides up to **1.5x speedup over BF16** on H100 GPUs with ~30% memory reduction.

> **Important:** FP8 requires specific version combinations. The updated FP8 script now handles this automatically.

#### FP8 Support Status

| Model | FP8 Support | How to Enable | Notes |
|-------|-------------|---------------|-------|
| **GROOT 1.6** | Not native yet | Requires code modification | N/A |
| **SmolVLA** | **Ready-to-use script** | `bash train_smolvla_bimanual_fp8.sh` | Auto-installs deps |
| **Pi0.5** | Via torchao | `torch.compile` + FP8 recipe | Manual setup |

#### SmolVLA FP8 Training (H100)

The FP8 script uses torchao's direct API injection to convert Linear layers to FP8, which works with PyTorch 2.7+ even when accelerate's FP8 backend has issues.

**How it works:**
1. Verifies H100 GPU and torchao installation
2. Monkey-patches `make_policy` to inject FP8 after model creation
3. Converts 303 Linear layers to FP8 via `convert_to_float8_training()`
4. Runs training with BF16 base + FP8 linear layers

**Run FP8 Training:**

**Installation (if torchao not installed):**

```bash
# Install from PyTorch wheel index (NOT PyPI) for version compatibility
pip install torchao --index-url https://download.pytorch.org/whl/cu126
```

```bash
# FP8 training with torchao (H100 80GB)
GRADIENT_CHECKPOINTING=false FP8_BACKEND=torchao DATASET_PATH=/workspace/.hf_home/lerobot/jasmine314342/picknplace-bimanual-464 DATASET_NAME=picknplace-bimanual-464 BATCH_SIZE=128 MAX_STEPS=40000 NUM_WORKERS=20 LOG_FREQ=10 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual_fp8.sh
```

<!-- 
fp16:
batch: 64, gradient_checkpoint: false, updt_s:0.841 data_s:0.019

fp8:
batch: 128, gradient_checkpoint: true, updt_s:6.233 data_s:3.294

 -->

**Performance Comparison (H100 80GB):**

| Precision | Batch Size | Memory | Speed | Training Time (40K steps) |
|-----------|------------|--------|-------|---------------------------|
| BF16 | 128 | ~70GB | ~5-6 it/s | ~2-2.5 hrs |
| **FP8** | **160** | **~55GB** | **~7-8 it/s** | **~1.5-2 hrs** |

#### Enabling FP8 with Accelerate (Manual)

```bash
# Option 1: TransformerEngine backend
pip install transformer-engine[pytorch]

# Option 2: torchao backend (install from PyTorch wheel index!)
pip install torchao --index-url https://download.pytorch.org/whl/cu126

# Run with accelerate
accelerate config  # Select fp8, then TE or torchao backend
accelerate launch --mixed_precision fp8 your_training_script.py
```

Example accelerate config for FP8 with TransformerEngine:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp8
fp8_config:
  backend: TE
  fp8_format: HYBRID
  amax_history_len: 1024
  amax_compute_algo: max
```

#### TorchAO FP8 (Direct API)

If accelerate's FP8 backend doesn't work, use torchao's direct API:

```python
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

# Configure FP8 (tensorwise is fastest)
config = Float8LinearConfig.from_recipe_name("tensorwise")

# Convert model layers to FP8
def module_filter_fn(mod, fqn):
    # Skip layers with dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)

# IMPORTANT: torch.compile is required for FP8 speedup!
model = torch.compile(model)
```

The SmolVLA FP8 script includes a wrapper (`fp8_train_wrapper.py`) that does this automatically.

### DataLoader Optimization

Data loading is often the bottleneck when GPU utilization fluctuates. Key settings:

```python
DataLoader(
    dataset,
    batch_size=24,
    num_workers=8,              # Set to CPU cores (os.cpu_count())
    pin_memory=True,            # Faster host→GPU transfer
    prefetch_factor=4,          # Queue 4 batches per worker
    persistent_workers=True,    # Avoid worker restart between epochs
)
```

**Environment variables for training scripts:**

```bash
NUM_WORKERS=8 \
PIN_MEMORY=true \
PREFETCH_FACTOR=4 \
    bash train_script.sh
```

**Diagnose data loading bottleneck:**
- If GPU utilization fluctuates (39% → 100% → 39%), it's likely data loading
- Look for "Wait for shard" or "Caching shard" messages in logs

### Memory Optimization

#### Gradient Checkpointing

Trades compute for memory - allows larger batch sizes:

```bash
GRADIENT_CHECKPOINTING=true \
    bash train_script.sh
```

#### Pre-cache Dataset to RAM

For faster data loading, copy dataset to RAM disk:

```bash
# Copy to RAM disk (if you have enough RAM)
cp -r /workspace/dataset /dev/shm/dataset
DATASET_PATH=/dev/shm/dataset bash train_script.sh
```

### torch.compile (A100/H100)

Can provide 10-30% speedup, but has compatibility issues with Flash Attention:

```python
model = torch.compile(model)
```

**Note:** Flash Attention 2 does not work with torch.compile. Choose one or the other.

### TensorFloat32 (TF32)

Enable TF32 for faster matrix operations on A100/H100:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Or environment variable:
# NVIDIA_TF32_OVERRIDE=1
```

### Multi-GPU Training

For large models or faster training:

```bash
# FSDP (Fully Sharded Data Parallel)
accelerate launch --multi_gpu --num_processes 2 train.py

# DeepSpeed ZeRO-2/3
accelerate launch --use_deepspeed train.py
```

### H100 vs A100 Performance Comparison

| Optimization | A100 Speed | H100 Speed | H100 Advantage |
|--------------|------------|------------|----------------|
| FP32 baseline | 1.0x | 1.5x | 1.5x |
| BF16 | 2.0x | 3.0x | 1.5x |
| FP8 | N/A | 4-6x | 2x over BF16 |
| FP8 + compile | N/A | 6-9x | Best case |

### Quick Optimization Checklist

```bash
# Apply these for optimal training on H100:
NUM_WORKERS=8 \
PIN_MEMORY=true \
GRADIENT_CHECKPOINTING=true \
    bash train_script.sh
```

| Setting | A100 | H100 |
|---------|------|------|
| Mixed Precision | BF16 | BF16 (or FP8 if supported) |
| Num Workers | 8-16 | 8-16 |
| Pin Memory | true | true |
| TF32 | Enable | Enable |
| Flash Attention | Enable | Enable |

### References

- [HuggingFace Accelerate FP8 Guide](https://huggingface.co/docs/accelerate/en/usage_guides/low_precision_training)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [PyTorch TorchAO](https://github.com/pytorch/ao)
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## Cost Optimization Tips

1. **Use spot instances** - 50-70% cheaper, but can be interrupted
2. **Start with small runs** - Test with 1000 steps first
3. **Monitor GPU utilization** - Increase batch size if GPU not fully utilized
4. **Download checkpoints regularly** - Don't lose work if instance terminates
5. **Use local RTX 4090/5090** - All models fit in 24GB, just slower
6. **Use HuggingFace Hub** - Faster dataset access than rsync for cloud instances

---

## Quick Reference

### SmolVLA Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_REPO_ID` | `jasmine314342/picknplace-bimanual-464` | HuggingFace dataset |
| `PRETRAINED_MODEL` | `lerobot/smolvla_base` | Base model |
| `FREEZE_VISION` | `false` | Freeze vision encoder |
| `TRAIN_EXPERT_ONLY` | `true` | Only train expert (not language) |
| `GRADIENT_CHECKPOINTING` | `true` | Memory optimization |
| `BATCH_SIZE` | 96 | Batch size (RTX 4090: 48, A100 40GB: 96, A100 80GB: 128) |
| `MAX_STEPS` | 30000 | Training steps (~24 epochs for 464 episodes with batch=96) |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `WARMUP_STEPS` | 1000 | LR warmup steps |
| `CHUNK_SIZE` | 50 | Action chunk size |
| `N_ACTION_STEPS` | 50 | Number of action steps |
| `NUM_STEPS` | 10 | Flow matching denoising steps |
| `SAVE_STEPS` | 5000 | Checkpoint frequency |
| `NUM_WORKERS` | 8 | Data loader workers |
| `WANDB_ENABLE` | false | Enable W&B logging |

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
| `GLOBAL_BATCH_SIZE` | 8 | Batch size (8 for 24GB, 16-32 for 40GB+) |
| `MAX_STEPS` | 10000 | Training steps |
| `WARMUP_RATIO` | 0.05 | Warmup fraction |
| `TUNE_VISUAL` | false | Unfreeze vision encoder (needs 40GB+) |
| `TUNE_LLM` | false | Unfreeze LLM backbone (needs 80GB+) |
| `TUNE_PROJECTOR` | true | Train projector layers |
| `TUNE_DIFFUSION` | true | Train DiT action head |

**Repository:** https://github.com/wellbeing18/Isaac-GR00T (branch: `lora/reusable-groot-workflow-rtx5090-fixes`)

---

## Next Steps

After training completes:

1. **Download checkpoint** to local machine
   ```bash
   # SmolVLA/Pi0.5 (from lerobot outputs)
   rsync -avz user@cloud:outputs/smolvla_bimanual_cloud_*/checkpoints ./checkpoints/

   # GROOT (from Isaac-GR00T outputs)
   rsync -avz user@cloud:/workspace/Isaac-GR00T/outputs/groot16_bimanual_*/checkpoint-* ./checkpoints/
   ```

2. **Run inference test**
   ```bash
   # SmolVLA
   python jdocs/scripts/cloud/smolvla/infer_smolvla_bimanual.py \
       --checkpoint ./checkpoints/last/pretrained_model

   # Pi0.5
   python jdocs/scripts/cloud/pi05/infer_pi05_bimanual.py \
       --checkpoint ./checkpoints/checkpoint-50000

   # GROOT (from Isaac-GR00T repo)
   python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
       --checkpoint ./checkpoints/checkpoint-10000
   ```

3. **Push to HuggingFace Hub** (optional)
   ```bash
   huggingface-cli upload your-username/smolvla-bimanual \
       ./checkpoints/last/pretrained_model
   ```

4. **Deploy on robot**
   ```bash
   python -m lerobot.scripts.control_robot \
       --policy.path=your-username/smolvla-bimanual \
       --robot.type=so101_bimanual
   ```
