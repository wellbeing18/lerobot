#!/bin/bash
# =============================================================================
# SmolVLA Cloud Training Script for Bimanual SO-101
# =============================================================================
#
# This script trains SmolVLA on bimanual datasets for cloud GPU instances.
# Optimized for A100/H100 GPUs with large batch sizes.
#
# IMPORTANT: SmolVLA has NO native bimanual support. It treats 12 DOF as a
# flat vector without arm-specific loss computation.
#
# Recommended Configuration (Vision + Expert mode):
#   - Unfrozen vision encoder for visual-spatial learning
#   - Train expert only (preserves language capabilities)
#   - Gradient checkpointing for memory efficiency
#   - ~185M trainable params (~41% of 450M total)
#
# Memory Requirements:
#   - RTX 4090 (24GB): batch_size=8-16
#   - A100 40GB: batch_size=32-64
#   - A100 80GB / H100: batch_size=64-128
#
# Usage:
#   # Basic training with HuggingFace dataset
#   bash train_smolvla_bimanual.sh
#
#   # Custom dataset from HuggingFace Hub
#   DATASET_REPO_ID=jasmine314342/picknplace-bimanual-464 bash train_smolvla_bimanual.sh
#
#   # Local dataset (must exist on cloud instance)
#   DATASET_PATH=/data/picknplace_300_144 DATASET_NAME=picknplace_300_144 bash train_smolvla_bimanual.sh
#
#   # Adjust for GPU memory
#   BATCH_SIZE=64 MAX_STEPS=50000 bash train_smolvla_bimanual.sh
#
#   # Enable Weights & Biases logging
#   WANDB_ENABLE=true WANDB_PROJECT=smolvla-bimanual bash train_smolvla_bimanual.sh
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# --- Dataset Configuration ---
# Option 1: HuggingFace Hub dataset (recommended for cloud)
DATASET_REPO_ID="${DATASET_REPO_ID:-jasmine314342/picknplace-bimanual-464}"
# Option 2: Local dataset path (set both if using local)
DATASET_PATH="${DATASET_PATH:-}"
DATASET_NAME="${DATASET_NAME:-}"

# --- Model Configuration ---
PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/smolvla_base}"

# --- Training Mode (Vision + Expert recommended for bimanual) ---
FREEZE_VISION="${FREEZE_VISION:-false}"          # Unfreeze for bimanual
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"   # Keep language frozen
TRAIN_STATE_PROJ="${TRAIN_STATE_PROJ:-true}"
# NOTE: gradient_checkpointing is NOT supported in upstream LeRobot SmolVLA
# It was a local modification. Removed for cloud compatibility.

# --- Training Hyperparameters ---
# For 464 episodes (118K frames): epochs = (steps × batch) / 118772
# Measured: batch=48 uses ~20GB on 24GB GPU (Vision+Expert mode)
# Official SmolVLA: 20K steps for 50 episodes = ~107 epochs
# For 464 eps (9.3x larger): ~30K steps recommended (~24 epochs with batch=96)
BATCH_SIZE="${BATCH_SIZE:-96}"                   # RTX 4090: 48, A100 40GB: 96, A100 80GB: 128
MAX_STEPS="${MAX_STEPS:-30000}"                  # 30K for 464 eps (~24 epochs with batch=96)
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-10}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# --- Scheduler ---
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-${MAX_STEPS}}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# --- SmolVLA Architecture ---
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"  # Flow matching denoising steps

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"  # More workers for cloud
SEED="${SEED:-1000}"

# --- Output ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_bimanual_cloud_${TIMESTAMP}}"

# --- Logging ---
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-bimanual}"

# --- Device ---
DEVICE="${DEVICE:-cuda}"

# =============================================================================
# Determine Dataset Source
# =============================================================================

if [ -n "${DATASET_PATH}" ] && [ -n "${DATASET_NAME}" ]; then
    # Local dataset mode
    DATASET_ARG="--dataset.repo_id=${DATASET_NAME} --dataset.root=${DATASET_PATH}"
    DATASET_DISPLAY="${DATASET_PATH} (local)"
else
    # HuggingFace Hub mode
    DATASET_ARG="--dataset.repo_id=${DATASET_REPO_ID}"
    DATASET_DISPLAY="${DATASET_REPO_ID} (HuggingFace Hub)"
fi

# =============================================================================
# Display Configuration
# =============================================================================

echo "============================================================"
echo "SmolVLA Cloud Training for Bimanual SO-101"
echo "============================================================"
echo ""
echo "WARNING: SmolVLA has NO native bimanual support!"
echo "         Actions are treated as flat 12D vector."
echo ""

if [ "${FREEZE_VISION}" = "false" ] && [ "${TRAIN_EXPERT_ONLY}" = "true" ]; then
    echo ">>> MODE: Vision + Expert (RECOMMENDED) <<<"
    echo "    Trainable: Vision (~86M) + Expert (~100M) = ~185M params"
elif [ "${FREEZE_VISION}" = "true" ] && [ "${TRAIN_EXPERT_ONLY}" = "true" ]; then
    echo ">>> MODE: Expert Only <<<"
    echo "    Trainable: Expert only (~100M params)"
else
    echo ">>> MODE: Full VLM <<<"
    echo "    Trainable: All ~450M params (risk of catastrophic forgetting)"
fi

echo ""
echo "Configuration:"
echo "  Dataset:              ${DATASET_DISPLAY}"
echo "  Pretrained:           ${PRETRAINED_MODEL}"
echo "  Output:               ${OUTPUT_DIR}"
echo "  Max Steps:            ${MAX_STEPS}"
echo "  Batch Size:           ${BATCH_SIZE}"
echo "  Learning Rate:        ${LEARNING_RATE}"
echo "  Chunk Size:           ${CHUNK_SIZE}"
echo "  Warmup Steps:         ${WARMUP_STEPS}"
echo "  W&B Logging:          ${WANDB_ENABLE}"
echo "============================================================"
echo ""

# =============================================================================
# Build Training Command
# =============================================================================

CMD="python -W ignore::UserWarning -W ignore::FutureWarning -W ignore::DeprecationWarning \
    -m lerobot.scripts.lerobot_train \
    ${DATASET_ARG} \
    --dataset.video_backend=pyav \
    --policy.path=${PRETRAINED_MODEL} \
    --policy.device=${DEVICE} \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.num_steps=${NUM_STEPS} \
    --policy.freeze_vision_encoder=${FREEZE_VISION} \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
    --policy.train_state_proj=${TRAIN_STATE_PROJ} \
    --policy.optimizer_lr=${LEARNING_RATE} \
    --policy.optimizer_weight_decay=${WEIGHT_DECAY} \
    --policy.optimizer_grad_clip_norm=${GRAD_CLIP_NORM} \
    --policy.scheduler_warmup_steps=${WARMUP_STEPS} \
    --policy.scheduler_decay_steps=${DECAY_STEPS} \
    --policy.scheduler_decay_lr=${DECAY_LR} \
    --policy.push_to_hub=false \
    --batch_size=${BATCH_SIZE} \
    --steps=${MAX_STEPS} \
    --save_freq=${SAVE_STEPS} \
    --log_freq=${LOG_FREQ} \
    --num_workers=${NUM_WORKERS} \
    --seed=${SEED} \
    --output_dir=${OUTPUT_DIR} \
    --job_name=smolvla_bimanual"

# Camera name mapping (required for SmolVLA)
# Maps: head->camera1, left_wrist->camera2, right_wrist->camera3
CMD="${CMD} --rename_map={\"observation.images.head\":\"observation.images.camera1\",\"observation.images.left_wrist\":\"observation.images.camera2\",\"observation.images.right_wrist\":\"observation.images.camera3\"}"

# W&B logging
if [ "${WANDB_ENABLE}" = "true" ]; then
    CMD="${CMD} --wandb.enable=true --wandb.project=${WANDB_PROJECT}"
else
    CMD="${CMD} --wandb.enable=false"
fi

# =============================================================================
# Log Configuration
# =============================================================================

# Use separate log directory (don't create OUTPUT_DIR - let lerobot handle it)
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

# Helper function for logging (matches local script)
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

# Initialize log file with configuration
{
    echo "============================================================"
    echo "SmolVLA Cloud Training"
    echo "Started: $(date)"
    echo "============================================================"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
    echo ""
    echo "Configuration:"
    echo "  Dataset:              ${DATASET_DISPLAY}"
    echo "  Output:               ${OUTPUT_DIR}"
    echo "  Max steps:            ${MAX_STEPS}"
    echo "  Batch size:           ${BATCH_SIZE}"
    echo "  Learning rate:        ${LEARNING_RATE}"
    echo "  Warmup steps:         ${WARMUP_STEPS}"
    echo "  Freeze vision:        ${FREEZE_VISION}"
    echo "  Expert only:          ${TRAIN_EXPERT_ONLY}"
    echo "  Chunk size:           ${CHUNK_SIZE}"
    echo "  N action steps:       ${N_ACTION_STEPS}"
    echo "  Num denoising steps:  ${NUM_STEPS}"
    echo "============================================================"
    echo ""
    echo "Command:"
    echo "${CMD}"
    echo ""
} > "${LOG_FILE}"

# =============================================================================
# Run Training
# =============================================================================

log "Starting training..."
log "Log file: ${LOG_FILE}"
log ""
log "TIP: Monitor GPU usage in another terminal with:"
log "  watch -n 1 nvidia-smi"
log ""

${CMD} 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# Post-Training
# =============================================================================

if [ ${EXIT_CODE} -eq 0 ]; then
    log ""
    log "============================================================"
    log "Training Complete: $(date)"
    log "============================================================"
    log "Checkpoint saved to: ${OUTPUT_DIR}/checkpoints"
    log ""
    log "Next steps:"
    log "  1. Download checkpoint to local machine:"
    log "     rsync -avz user@cloud:${OUTPUT_DIR}/checkpoints ./checkpoints/"
    log ""
    log "  2. Run inference test:"
    log "     python jdocs/scripts/cloud/smolvla/infer_smolvla_bimanual.py \\"
    log "         --checkpoint ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
    log ""
    log "  3. Or push to HuggingFace Hub:"
    log "     huggingface-cli upload your-username/smolvla-bimanual \\"
    log "         ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
    log "============================================================"
else
    log ""
    log "============================================================"
    log "Training FAILED with exit code: ${EXIT_CODE}"
    log "Check log file: ${LOG_FILE}"
    log "============================================================"
    exit ${EXIT_CODE}
fi
