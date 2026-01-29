#!/bin/bash
# =============================================================================
# SmolVLA Cloud Training Script with FP8 Precision (H100 Only)
# =============================================================================
#
# This script trains SmolVLA with FP8 mixed precision on H100 GPUs.
# FP8 provides up to 2x speedup over BF16 with ~30% memory reduction.
#
# REQUIREMENTS:
#   - H100 GPU (FP8 not supported on A100 or older)
#   - transformer-engine or torchao installed
#   - accelerate configured for FP8
#
# INSTALLATION:
#   pip install transformer-engine[pytorch]
#   # or
#   pip install torchao
#
# Memory Benefits (FP8 vs BF16):
#   - ~30% less memory usage
#   - Can increase batch size by ~1.3x
#   - H100: batch_size=160-200 (vs 128 with BF16)
#
# Usage:
#   # Basic FP8 training
#   bash train_smolvla_bimanual_fp8.sh
#
#   # Custom batch size (larger due to FP8 memory savings)
#   BATCH_SIZE=160 bash train_smolvla_bimanual_fp8.sh
#
#   # Use torchao backend instead of TransformerEngine
#   FP8_BACKEND=torchao bash train_smolvla_bimanual_fp8.sh
#
# =============================================================================

set -e

# =============================================================================
# FP8 Configuration
# =============================================================================

# FP8 backend: TE (TransformerEngine) or torchao
FP8_BACKEND="${FP8_BACKEND:-TE}"

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_REPO_ID="${DATASET_REPO_ID:-jasmine314342/picknplace-bimanual-464}"
DATASET_PATH="${DATASET_PATH:-}"
DATASET_NAME="${DATASET_NAME:-}"

# =============================================================================
# Model Configuration
# =============================================================================

PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/smolvla_base}"

# Training Mode (Vision + Expert recommended for bimanual)
FREEZE_VISION="${FREEZE_VISION:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
TRAIN_STATE_PROJ="${TRAIN_STATE_PROJ:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# =============================================================================
# Training Hyperparameters
# =============================================================================

# FP8 allows ~30% larger batch sizes
BATCH_SIZE="${BATCH_SIZE:-128}"  # H100 with FP8: 128-200
MAX_STEPS="${MAX_STEPS:-30000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-10}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# Scheduler
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-${MAX_STEPS}}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# SmolVLA Architecture
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_bimanual_fp8_${TIMESTAMP}}"

# Logging
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-bimanual-fp8}"

# =============================================================================
# Check H100 GPU
# =============================================================================

echo "============================================================"
echo "SmolVLA FP8 Training for Bimanual SO-101"
echo "============================================================"
echo ""

# Check for H100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [[ ! "${GPU_NAME}" =~ "H100" ]]; then
    echo "WARNING: FP8 is optimized for H100 GPUs."
    echo "         Detected GPU: ${GPU_NAME}"
    echo "         FP8 may not work or provide benefits on this GPU."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Use train_smolvla_bimanual.sh for non-H100 GPUs."
        exit 1
    fi
else
    echo "Detected H100 GPU - FP8 training enabled"
fi

# =============================================================================
# Check FP8 Dependencies
# =============================================================================

echo ""
echo "Checking FP8 dependencies..."

if [ "${FP8_BACKEND}" = "TE" ]; then
    if python -c "import transformer_engine" 2>/dev/null; then
        echo "  TransformerEngine: OK"
    else
        echo "  TransformerEngine: NOT FOUND"
        echo "  Installing: pip install transformer-engine[pytorch]"
        pip install transformer-engine[pytorch]
    fi
elif [ "${FP8_BACKEND}" = "torchao" ]; then
    if python -c "import torchao" 2>/dev/null; then
        echo "  torchao: OK"
    else
        echo "  torchao: NOT FOUND"
        echo "  Installing: pip install torchao"
        pip install torchao
    fi
fi

# =============================================================================
# Create Accelerate Config for FP8
# =============================================================================

ACCELERATE_CONFIG_DIR="${HOME}/.cache/huggingface/accelerate"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_DIR}/fp8_config.yaml"

mkdir -p "${ACCELERATE_CONFIG_DIR}"

echo ""
echo "Creating accelerate FP8 config at: ${ACCELERATE_CONFIG_FILE}"

if [ "${FP8_BACKEND}" = "TE" ]; then
    cat > "${ACCELERATE_CONFIG_FILE}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp8
fp8_config:
  backend: TE
  fp8_format: HYBRID
  amax_history_len: 1024
  amax_compute_algo: max
  override_linear_precision: false
EOF
else
    cat > "${ACCELERATE_CONFIG_FILE}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp8
fp8_config:
  backend: torchao
EOF
fi

echo "  FP8 Backend: ${FP8_BACKEND}"

# =============================================================================
# Determine Dataset Source
# =============================================================================

if [ -n "${DATASET_PATH}" ] && [ -n "${DATASET_NAME}" ]; then
    DATASET_ARG="--dataset.repo_id=${DATASET_NAME} --dataset.root=${DATASET_PATH}"
    DATASET_DISPLAY="${DATASET_PATH} (local)"
else
    DATASET_ARG="--dataset.repo_id=${DATASET_REPO_ID}"
    DATASET_DISPLAY="${DATASET_REPO_ID} (HuggingFace Hub)"
fi

# =============================================================================
# Display Configuration
# =============================================================================

echo ""
echo "Configuration:"
echo "  Dataset:              ${DATASET_DISPLAY}"
echo "  Pretrained:           ${PRETRAINED_MODEL}"
echo "  Output:               ${OUTPUT_DIR}"
echo "  Max Steps:            ${MAX_STEPS}"
echo "  Batch Size:           ${BATCH_SIZE}"
echo "  Learning Rate:        ${LEARNING_RATE}"
echo "  FP8 Backend:          ${FP8_BACKEND}"
echo "  Gradient Checkpoint:  ${GRADIENT_CHECKPOINTING}"
echo "============================================================"
echo ""

# =============================================================================
# Build Training Command with Accelerate
# =============================================================================

# Use accelerate launch with FP8 config
CMD="accelerate launch --config_file ${ACCELERATE_CONFIG_FILE} \
    -m lerobot.scripts.lerobot_train \
    ${DATASET_ARG} \
    --dataset.video_backend=pyav \
    --policy.path=${PRETRAINED_MODEL} \
    --policy.device=cuda \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.num_steps=${NUM_STEPS} \
    --policy.freeze_vision_encoder=${FREEZE_VISION} \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
    --policy.train_state_proj=${TRAIN_STATE_PROJ} \
    --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
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
    --job_name=smolvla_bimanual_fp8"

# Camera name mapping
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

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

{
    echo "============================================================"
    echo "SmolVLA FP8 Training"
    echo "Started: $(date)"
    echo "============================================================"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
    echo ""
    echo "FP8 Configuration:"
    echo "  Backend:              ${FP8_BACKEND}"
    echo "  Config file:          ${ACCELERATE_CONFIG_FILE}"
    echo ""
    echo "Training Configuration:"
    echo "  Dataset:              ${DATASET_DISPLAY}"
    echo "  Output:               ${OUTPUT_DIR}"
    echo "  Max steps:            ${MAX_STEPS}"
    echo "  Batch size:           ${BATCH_SIZE}"
    echo "  Learning rate:        ${LEARNING_RATE}"
    echo "============================================================"
    echo ""
    echo "Command:"
    echo "${CMD}"
    echo ""
} > "${LOG_FILE}"

# =============================================================================
# Run Training
# =============================================================================

log "Starting FP8 training..."
log "Log file: ${LOG_FILE}"
log ""
log "Expected speedup over BF16: ~1.5-2x"
log "Expected memory savings: ~30%"
log ""

${CMD} 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# Post-Training
# =============================================================================

if [ ${EXIT_CODE} -eq 0 ]; then
    log ""
    log "============================================================"
    log "FP8 Training Complete: $(date)"
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
    log "============================================================"
else
    log ""
    log "============================================================"
    log "FP8 Training FAILED with exit code: ${EXIT_CODE}"
    log ""
    log "Common FP8 issues:"
    log "  - GPU not H100: FP8 requires Hopper architecture"
    log "  - Missing transformer-engine: pip install transformer-engine[pytorch]"
    log "  - Incompatible model layers: Some layers may not support FP8"
    log ""
    log "Try BF16 training instead:"
    log "  bash train_smolvla_bimanual.sh"
    log "============================================================"
    exit ${EXIT_CODE}
fi
