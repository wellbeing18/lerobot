#!/bin/bash
# ===========================================================================
# Pi0.5 Training Script for Bimanual SO-101
# ===========================================================================
#
# This script trains Pi0.5 with LoRA on the bimanual multitasks dataset.
#
# Based on: /home/jrobot/project/refs/openpi/LEROBOT_VS_OPENPI_INVESTIGATION.md
#
# CRITICAL LoRA SETTINGS (from OpenPI investigation):
#   - lora_alpha MUST equal lora_rank for 1.0 scaling factor
#   - lora_dropout should be 0.0 (not 0.1)
#   - These are CRITICAL for training to work correctly!
#
# Memory Requirements:
#   - Full Pi0.5 (2.5B params): ~10-12 GB
#   - With LoRA (40M trainable): ~5-6 GB
#   - RTX 4090/5090 (24GB): batch_size=8 recommended
#   - A100 (40GB): batch_size=32-64 recommended
#
# Usage:
#   # Basic training
#   bash train_pi05_bimanual.sh
#
#   # Custom batch size and steps
#   BATCH_SIZE=64 MAX_STEPS=100000 bash train_pi05_bimanual.sh
#
#   # Local dataset instead of HuggingFace Hub
#   DATASET_PATH=/path/to/local/dataset bash train_pi05_bimanual.sh
#
# ===========================================================================

set -e

# ===========================================================================
# KEY HYPERPARAMETERS
# ===========================================================================

# --- Dataset Configuration ---
# Use HuggingFace Hub by default, or set DATASET_PATH for local dataset
DATASET_REPO_ID="${DATASET_REPO_ID:-multitasks}"
DATASET_ROOT="${DATASET_ROOT:-/home/jrobot/project/lerobot/datasets_bimanuel/multitasks}"

# --- Model Configuration ---
POLICY_TYPE="pi05"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05-base}"  # HuggingFace checkpoint
DTYPE="${DTYPE:-bfloat16}"

# --- LoRA Configuration (CRITICAL - match OpenPI!) ---
USE_LORA="${USE_LORA:-true}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"      # MUST equal rank for 1.0 scaling!
LORA_DROPOUT="${LORA_DROPOUT:-0.0}" # OpenPI uses 0.0, not 0.1

# --- Training Hyperparameters ---
BATCH_SIZE="${BATCH_SIZE:-8}"           # 8 for 24GB, 32-64 for A100
MAX_STEPS="${MAX_STEPS:-120000}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

# --- Scheduler ---
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-${MAX_STEPS}}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# --- Action Configuration ---
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

# --- Memory Optimization ---
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"

# --- Output ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_bimanual_${TIMESTAMP}}"
JOB_NAME="${JOB_NAME:-pi05_bimanual}"

# --- Logging ---
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# ===========================================================================
# END OF KEY HYPERPARAMETERS
# ===========================================================================

echo "=================================================================="
echo "Pi0.5 Bimanual Training Configuration"
echo "=================================================================="
echo "Dataset:        ${DATASET_REPO_ID} (root: ${DATASET_ROOT})"
echo "Output:         ${OUTPUT_DIR}"
echo "Max steps:      ${MAX_STEPS}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Learning rate:  ${LEARNING_RATE}"
echo "LoRA:           rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Dtype:          ${DTYPE}"
echo "Grad checkpoint: ${GRADIENT_CHECKPOINTING}"
echo "=================================================================="

# Validate LoRA settings
if [ "${USE_LORA}" = "true" ] && [ "${LORA_ALPHA}" != "${LORA_RANK}" ]; then
    echo "WARNING: LORA_ALPHA (${LORA_ALPHA}) != LORA_RANK (${LORA_RANK})"
    echo "This may cause training issues. OpenPI uses alpha = rank for 1.0 scaling."
    echo "Set LORA_ALPHA=${LORA_RANK} for best results."
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command
CMD="python -W ignore::UserWarning -W ignore::FutureWarning -m lerobot.scripts.lerobot_train"

# Dataset configuration
CMD="${CMD} --dataset.repo_id=${DATASET_REPO_ID}"
CMD="${CMD} --dataset.root=${DATASET_ROOT}"
CMD="${CMD} --dataset.video_backend=pyav"

# Policy configuration
CMD="${CMD} --policy.type=${POLICY_TYPE}"
if [ -n "${PRETRAINED_PATH}" ]; then
    CMD="${CMD} --policy.path=${PRETRAINED_PATH}"
fi
CMD="${CMD} --policy.device=cuda"
CMD="${CMD} --policy.dtype=${DTYPE}"

# Action configuration
CMD="${CMD} --policy.chunk_size=${CHUNK_SIZE}"
CMD="${CMD} --policy.n_action_steps=${N_ACTION_STEPS}"
CMD="${CMD} --policy.num_inference_steps=${NUM_INFERENCE_STEPS}"

# LoRA configuration
if [ "${USE_LORA}" = "true" ]; then
    CMD="${CMD} --policy.use_lora=true"
    CMD="${CMD} --policy.lora_rank=${LORA_RANK}"
    CMD="${CMD} --policy.lora_alpha=${LORA_ALPHA}"
    CMD="${CMD} --policy.lora_dropout=${LORA_DROPOUT}"
fi

# Optimizer configuration
CMD="${CMD} --policy.optimizer_lr=${LEARNING_RATE}"
CMD="${CMD} --policy.optimizer_weight_decay=${WEIGHT_DECAY}"
CMD="${CMD} --policy.optimizer_grad_clip_norm=${GRAD_CLIP_NORM}"

# Scheduler configuration
CMD="${CMD} --policy.scheduler_warmup_steps=${WARMUP_STEPS}"
CMD="${CMD} --policy.scheduler_decay_steps=${DECAY_STEPS}"
CMD="${CMD} --policy.scheduler_decay_lr=${DECAY_LR}"

# Memory optimization
if [ "${GRADIENT_CHECKPOINTING}" = "true" ]; then
    CMD="${CMD} --policy.gradient_checkpointing=true"
fi

# Training configuration
CMD="${CMD} --batch_size=${BATCH_SIZE}"
CMD="${CMD} --steps=${MAX_STEPS}"
CMD="${CMD} --save_freq=${SAVE_STEPS}"
CMD="${CMD} --log_freq=${LOG_FREQ}"
CMD="${CMD} --num_workers=${NUM_WORKERS}"
CMD="${CMD} --seed=1000"

# Output configuration
CMD="${CMD} --output_dir=${OUTPUT_DIR}"
CMD="${CMD} --job_name=${JOB_NAME}"

# Logging
if [ "${WANDB_ENABLE}" = "true" ]; then
    CMD="${CMD} --wandb.enable=true"
else
    CMD="${CMD} --wandb.enable=false"
fi

# Camera name mapping for bimanual (3 cameras)
# Pi0.5 expects: image1, image2, image3 (or similar naming)
# Our dataset has: head, left_wrist, right_wrist
# Note: Check if rename_map is needed based on Pi0.5's expected input_features
# CMD="${CMD} --rename_map={\"observation.images.head\":\"observation.images.image1\",\"observation.images.left_wrist\":\"observation.images.image2\",\"observation.images.right_wrist\":\"observation.images.image3\"}"

echo ""
echo "Command:"
echo "${CMD}"
echo ""

# Log configuration
LOG_FILE="${OUTPUT_DIR}/training.log"
{
    echo "=================================================================="
    echo "Pi0.5 Bimanual Training"
    echo "Started: $(date)"
    echo "=================================================================="
    echo "Dataset:        ${DATASET_REPO_ID}"
    echo "Output:         ${OUTPUT_DIR}"
    echo "Max steps:      ${MAX_STEPS}"
    echo "Batch size:     ${BATCH_SIZE}"
    echo "Learning rate:  ${LEARNING_RATE}"
    echo "LoRA:           rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
    echo "=================================================================="
    echo ""
} > "${LOG_FILE}"

# Run training
echo "Starting training..."
${CMD} 2>&1 | tee -a "${LOG_FILE}"

# Log completion
{
    echo ""
    echo "=================================================================="
    echo "Training Complete: $(date)"
    echo "=================================================================="
    echo "Checkpoint saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Run inference test:"
    echo "     python jdocs/scripts/cloud/pi05/infer_pi05_bimanual.py --checkpoint ${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
    echo ""
    echo "  2. Evaluate generalization:"
    echo "     Test on held-out tasks (icecream->plate, tissue selection)"
    echo "=================================================================="
} | tee -a "${LOG_FILE}"
