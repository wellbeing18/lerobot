# SmolVLA Full Fine-Tuning Analysis

## Problem Statement

Current SmolVLA training uses `train_expert_only=true` which freezes the language encoder. This causes **generalization failure**:
- Wrong target location: "icecream → plate" fails (trained on "icecream → bin")
- Wrong object selection: picks corn when asked for tissue

## Root Cause

With frozen language encoder:
1. **Language Embedding Collapse** - Can't learn task-specific distinctions
2. **Visual-Trajectory Dominance** - Vision features override weak language signal
3. **No Object-Language Grounding** - Can't link "tissue" text → tissue pixels
4. **Trajectory-Object Coupling** - Memorized object→trajectory mappings

## Solution: Full VLM Fine-Tuning (Unfreeze Language)

### Memory Feasibility on RTX 5090 24GB

| Mode | Trainable Params | VRAM (est.) | Batch Size | Local Feasible? |
|------|------------------|-------------|------------|-----------------|
| Mode 1: Expert only (frozen vision) | 100M | 2-2.5 GB | 32 | YES |
| Mode 2: Vision + Expert (current) | 185M | 3-4 GB | 8-16 | YES |
| **Mode 3: Full VLM (unfreeze all)** | **450M** | **5-6 GB** | **2-4** | **YES** |

**Key enabler:** `gradient_checkpointing=true` (already in training script)

---

## Industry Standard & Best Practices

### Official HuggingFace SmolVLA Recommendations

From [HuggingFace SmolVLA docs](https://huggingface.co/docs/lerobot/en/smolvla):
- **batch_size=64** (reduce if GPU memory limited)
- **steps=20,000** for ~50 episodes on A100
- **learning_rate=1e-4** (default in config)
- Default: `freeze_vision_encoder=True`, `train_expert_only=True`

From [smolvla_base model card](https://huggingface.co/lerobot/smolvla_base):
- Base model trained on ~10M frames from 487 datasets
- Fine-tuning examples use batch_size=64, steps=20,000

### Default Configuration (configuration_smolvla.py)

```python
# Fine-tuning settings
freeze_vision_encoder: bool = True
train_expert_only: bool = True        # Freezes language encoder
gradient_checkpointing: bool = False

# Optimizer
optimizer_lr: float = 1e-4
optimizer_weight_decay: float = 1e-10
optimizer_grad_clip_norm: float = 10

# Scheduler (cosine decay with warmup)
scheduler_warmup_steps: int = 1_000
scheduler_decay_steps: int = 30_000
scheduler_decay_lr: float = 2.5e-6
```

### Key Insight: No Official Guidance for Full Fine-Tuning

The [GitHub issue #2259](https://github.com/huggingface/lerobot/issues/2259) shows the community lacks clear guidance on full fine-tuning (unfreezing language encoder). The recommendations below are based on VLM fine-tuning best practices.

---

## Dataset: multitasks

```
Dataset: datasets_bimanuel/multitasks
Total episodes: 320
Total frames: 76,597
FPS: 30
```

### Previous Training (Mode 2: Frozen Language)

```
batch_size=48, steps=40,000
Total samples: 1,920,000
Epochs: 25.1
```

### Steps Calculation for Full Fine-Tuning (batch_size=4)

| Steps | Epochs | Total Samples | Recommendation |
|-------|--------|---------------|----------------|
| 50k | 2.6 | 200k | Quick diagnostic only |
| 100k | 5.2 | 400k | Minimum viable |
| 200k | 10.4 | 800k | **Recommended** |
| 300k | 15.7 | 1.2M | Solid training |
| 480k | 25.1 | 1.92M | Matches previous (may overfit) |

### Full Fine-Tuning Requires Different Strategy

**Why fewer epochs might be acceptable:**
1. Training 450M params (vs 185M) - each update is more impactful
2. Risk of catastrophic forgetting with too many epochs
3. Lower learning rate compensates with more stable updates

**Why similar total samples might be needed:**
1. Model still needs to see data enough times to learn
2. Language encoder needs time to adapt to robot domain

**Recommendation:** Start with **200k steps (10 epochs)**, evaluate, then continue to 300k if needed.

---

## Training Commands

### Quick Diagnostic (100k steps, ~12-16 hours)

```bash
nohup env \
    TRAIN_EXPERT_ONLY=false \
    FREEZE_VISION=false \
    BATCH_SIZE=4 \
    MAX_STEPS=100000 \
    LEARNING_RATE=5e-5 \
    bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
    > logs/smolvla_full_finetune_100k.log 2>&1 &
```

### Recommended Training (200k steps, ~24-32 hours)

```bash
nohup env \
    TRAIN_EXPERT_ONLY=false \
    FREEZE_VISION=false \
    BATCH_SIZE=4 \
    MAX_STEPS=200000 \
    LEARNING_RATE=5e-5 \
    bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
    > logs/smolvla_full_finetune_200k.log 2>&1 &
```

### Solid Training (300k steps, ~36-48 hours)

```bash
nohup env \
    TRAIN_EXPERT_ONLY=false \
    FREEZE_VISION=false \
    BATCH_SIZE=4 \
    MAX_STEPS=300000 \
    LEARNING_RATE=5e-5 \
    bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
    > logs/smolvla_full_finetune_300k.log 2>&1 &
```

### Monitor Training

```bash
# Watch log
tail -f logs/smolvla_full_finetune_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check process
ps aux | grep train_smolvla
```

---

## Key Parameter Changes

| Parameter | Current (Mode 2) | Full Fine-tuning (Mode 3) |
|-----------|------------------|---------------------------|
| `TRAIN_EXPERT_ONLY` | true | **false** |
| `FREEZE_VISION` | false | false |
| `BATCH_SIZE` | 48 | **4** (memory constraint) |
| `LEARNING_RATE` | 1e-4 | **5e-5** (lower to prevent forgetting) |
| `MAX_STEPS` | 40k (25 epochs) | **200k-300k** (10-15 epochs) |

### Why Lower Learning Rate?

**Risk:** Catastrophic forgetting of pretrained language understanding

**Mitigation:** Use 5e-5 instead of 1e-4 for the language encoder components. This is half the default rate, allowing the model to adapt without destroying pretrained knowledge.

---

## Evaluation: Generalization Test

After training, test on held-out task combinations:

```bash
# Test 1: Wrong target (trained on icecream→bin, test icecream→plate)
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
    --checkpoint outputs/smolvla_bimanual_full/checkpoint-200000 \
    --task "Use left arm to pick up the ice cream and place it on the plate"

# Test 2: Wrong object (tissue not in training for plate target)
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
    --checkpoint outputs/smolvla_bimanual_full/checkpoint-200000 \
    --task "Use right arm to pick up the tissue and place it on the plate"
```

### Success Criteria

- **>30% improvement** on held-out tasks → SmolVLA full fine-tuning is sufficient
- **<30% improvement** → Proceed to Pi0.5 or GROOT 1.6 (stronger pretrained grounding)

---

## Comparison with Other VLAs

If SmolVLA full fine-tuning doesn't achieve sufficient generalization:

| Model | Language Encoder | Pretrained Grounding | Training Method |
|-------|------------------|---------------------|-----------------|
| SmolVLA (unfrozen) | 260M trainable | Learning | Full fine-tuning |
| Pi0.5 | Gemma 2B | Strong | LoRA (alpha=rank) |
| GROOT 1.6 | Cosmos-Reason 2B | Strong | Selective freeze |

See `jdocs/scripts/cloud/` for Pi0.5 and GROOT 1.6 training scripts.
