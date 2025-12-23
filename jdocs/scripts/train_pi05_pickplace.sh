#!/bin/bash
# =============================================================================
# Pi0.5 Training Script for SO101 Pick and Place
# =============================================================================
#
# This script fine-tunes Pi0.5 (Physical Intelligence's Vision-Language-Action
# Model) on the pick_and_place dataset using LeRobot's training infrastructure.
#
# Pi0.5 is a powerful VLA model with open-world generalization capabilities,
# using PaliGemma (vision-language) + Gemma Expert (action) architecture.
#
# Key features:
#   - PaliGemma 2B/300M + Gemma Expert for action prediction
#   - Flow matching for smooth action generation
#   - Supports task/language conditioning
#   - Gradient checkpointing for memory efficiency
#
# Usage:
#   # Fresh training with defaults (3k steps, optimized for 24GB VRAM)
#   bash train_pi05_pickplace.sh
#
#   # Custom training steps and batch size
#   MAX_STEPS=6000 BATCH_SIZE=8 bash train_pi05_pickplace.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
#     bash train_pi05_pickplace.sh
#
# Key differences from SmolVLA:
#   - Larger model (PaliGemma 2B + Gemma 300M vs SmolVLA's smaller backbone)
#   - Uses quantiles normalization by default
#   - Requires gradient_checkpointing for 24GB VRAM
#   - Lower default learning rate (2.5e-5 vs 1e-4)
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Environment Setup
# =============================================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Add lerobot src to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Suppress all UserWarning, FutureWarning, DeprecationWarning (propagates to worker processes)
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# Disable tokenizers parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Change to project root (required for relative paths)
cd "${PROJECT_ROOT}"

# =============================================================================
# Configuration (Override via environment variables)
# =============================================================================

# Dataset Configuration
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/pick_and_place}"
DATASET_NAME="${DATASET_NAME:-pick_and_place}"

# Camera name mapping (dataset → Pi0.5 expected names)
# Pi0.5 is flexible with camera names, but we standardize to common convention
# Our dataset has: head, left_wrist
# Map to: base_0_rgb, left_wrist_0_rgb (Pi0.5 common convention)

# Pretrained Model
# Use HuggingFace model ID or local path
PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/pi05_base}"

# Training Hyperparameters
# Pi0.5 recommends 3k steps for fine-tuning (HuggingFace docs)
# Adjust based on dataset size; for ~50 episodes, 3k-6k steps is reasonable
MAX_STEPS="${MAX_STEPS:-3000}"

# Batch size: CRITICAL for 24GB VRAM
# With gradient_checkpointing + bfloat16, batch_size=8 is safe for 24GB
# Try batch_size=16 if you have headroom, but monitor VRAM usage
BATCH_SIZE="${BATCH_SIZE:-8}"

NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# Pi0.5 Architecture (most should stay at defaults)
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"  # Flow matching denoising steps

# Optimizer (Pi0.5 uses different defaults than SmolVLA)
# Default LR is 2.5e-5 (lower than SmolVLA's 1e-4)
LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

# Scheduler (cosine decay with warmup)
# Note: These auto-scale if steps < scheduler_decay_steps
WARMUP_STEPS="${WARMUP_STEPS:-100}"
DECAY_STEPS="${DECAY_STEPS:-3000}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# Memory Optimization (CRITICAL for 24GB VRAM)
# gradient_checkpointing: Reduces VRAM by ~30-50%, slight speed penalty
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
# compile_model: Uses torch.compile for potential speedup
COMPILE_MODEL="${COMPILE_MODEL:-true}"
# dtype: bfloat16 for memory efficiency (requires Ampere or newer GPU)
DTYPE="${DTYPE:-bfloat16}"

# Normalization (Pi0.5 uses QUANTILES by default)
# If your dataset doesn't have quantile stats, use MEAN_STD:
# NORMALIZATION_MODE="MEAN_STD"
NORMALIZATION_MODE="${NORMALIZATION_MODE:-QUANTILES}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_pickplace_${TIMESTAMP}}"

# Device
DEVICE="${DEVICE:-cuda}"

# Resume Configuration
RESUME_FROM="${RESUME_FROM:-}"

# =============================================================================
# Validation and Setup
# =============================================================================

# Validate dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_PATH}"
    exit 1
fi

if [ ! -f "${DATASET_PATH}/meta/info.json" ]; then
    echo "ERROR: Invalid dataset - missing meta/info.json"
    exit 1
fi

# Check for quantile stats if using QUANTILES normalization
if [ "${NORMALIZATION_MODE}" = "QUANTILES" ]; then
    if [ ! -f "${DATASET_PATH}/meta/stats.json" ]; then
        echo "WARNING: No stats.json found. May need to run quantile augmentation or use MEAN_STD."
        echo "To augment with quantiles, run:"
        echo "  python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --repo-id=${DATASET_NAME}"
        echo ""
        echo "Or set NORMALIZATION_MODE=MEAN_STD to skip quantile normalization."
    fi
fi

# =============================================================================
# Resume Logic
# =============================================================================

RESUME_FLAG=""
if [ -n "${RESUME_FROM}" ]; then
    if [ ! -d "${RESUME_FROM}" ]; then
        echo "ERROR: Resume checkpoint not found: ${RESUME_FROM}"
        exit 1
    fi

    RESUME_FLAG="--resume --config_path=${RESUME_FROM}/train_config.json"

    # Use the checkpoint's output directory
    CHECKPOINT_DIR=$(dirname "$(dirname "${RESUME_FROM}")")
    OUTPUT_DIR="${CHECKPOINT_DIR}"
fi

# =============================================================================
# Build Training Command
# =============================================================================

# Create logs directory (separate from output to avoid conflict with lerobot)
LOG_DIR="${PROJECT_ROOT}/jdocs/logs"
mkdir -p "${LOG_DIR}"

# Log file (named after the output dir for easy matching)
OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

# Function to log to both terminal and file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

# Start logging
log "=============================================="
log "Pi0.5 Training for Pick and Place"
log "=============================================="
log ""
log "Configuration:"
log "  Pretrained:            ${PRETRAINED_MODEL}"
log "  Dataset:               ${DATASET_PATH}"
log "  Max Steps:             ${MAX_STEPS}"
log "  Batch Size:            ${BATCH_SIZE}"
log "  Learning Rate:         ${LEARNING_RATE}"
log "  Chunk Size:            ${CHUNK_SIZE}"
log "  Warmup Steps:          ${WARMUP_STEPS}"
log "  Gradient Checkpointing: ${GRADIENT_CHECKPOINTING}"
log "  Compile Model:         ${COMPILE_MODEL}"
log "  Dtype:                 ${DTYPE}"
log "  Normalization:         ${NORMALIZATION_MODE}"
log "  Output Dir:            ${OUTPUT_DIR}"
log ""
log "Training started at $(date)"
log "Log file: ${LOG_FILE}"
log ""

# Build the training command
# Pi0.5 uses --policy.type=pi05 and --policy.pretrained_path for model loading
CMD="python -W ignore::UserWarning -W ignore::FutureWarning -W ignore::DeprecationWarning -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=${DATASET_NAME} \
    --dataset.root=${DATASET_PATH} \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=${PRETRAINED_MODEL} \
    --policy.device=${DEVICE} \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.num_inference_steps=${NUM_INFERENCE_STEPS} \
    --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    --policy.compile_model=${COMPILE_MODEL} \
    --policy.dtype=${DTYPE} \
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
    --job_name=pi05_pickplace \
    --wandb.enable=false \
    ${RESUME_FLAG}"

# Add normalization mapping if not using default QUANTILES
if [ "${NORMALIZATION_MODE}" = "MEAN_STD" ]; then
    CMD="${CMD} --policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}"
fi

# Add rename_map for camera name mapping (no spaces, no extra quotes)
# Map dataset camera names to Pi0.5 expected names
CMD="${CMD} --rename_map={\"observation.images.head\":\"observation.images.base_0_rgb\",\"observation.images.left_wrist\":\"observation.images.left_wrist_0_rgb\"}"

# =============================================================================
# Run Training
# =============================================================================

log "Running command:"
log "${CMD}"
log ""
log "=============================================="

# Run training and log output to both terminal and file
${CMD} 2>&1 | tee -a "${LOG_FILE}"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ ${EXIT_CODE} -eq 0 ]; then
    log ""
    log "=============================================="
    log "Training completed successfully!"
    log "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
    log "=============================================="
else
    log ""
    log "=============================================="
    log "Training failed with exit code: ${EXIT_CODE}"
    log "Check log file: ${LOG_FILE}"
    log "=============================================="
    exit ${EXIT_CODE}
fi
