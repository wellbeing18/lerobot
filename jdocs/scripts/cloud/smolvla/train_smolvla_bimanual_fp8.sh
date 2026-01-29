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

# Get PyTorch version and CUDA version for compatibility check
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  PyTorch: ${TORCH_VERSION}"
echo "  CUDA: ${CUDA_VERSION}"

# Determine CUDA wheel suffix (cu126, cu128, etc.)
CUDA_SUFFIX="cu126"
if [[ "${CUDA_VERSION}" == 12.8* ]]; then
    CUDA_SUFFIX="cu128"
elif [[ "${CUDA_VERSION}" == 12.9* ]]; then
    CUDA_SUFFIX="cu129"
fi
echo "  Wheel index: ${CUDA_SUFFIX}"

install_torchao() {
    echo ""
    echo "Installing torchao from PyTorch wheel index (not PyPI)..."
    echo "  This ensures compatibility with PyTorch ${TORCH_VERSION}"

    # IMPORTANT: Install from PyTorch wheel index, not PyPI
    # PyPI wheels may not match the exact PyTorch/CUDA version
    pip uninstall -y torchao 2>/dev/null || true
    pip install torchao --index-url "https://download.pytorch.org/whl/${CUDA_SUFFIX}"

    # Verify installation
    if python -c "import torchao; print(f'torchao {torchao.__version__} installed')" 2>/dev/null; then
        echo "  torchao: OK"
        return 0
    else
        echo "  torchao: Installation from wheel index failed, trying nightly..."
        pip install --pre torchao --index-url "https://download.pytorch.org/whl/nightly/${CUDA_SUFFIX}"
        if python -c "import torchao" 2>/dev/null; then
            echo "  torchao (nightly): OK"
            return 0
        fi
        return 1
    fi
}

install_te() {
    echo ""
    echo "Installing TransformerEngine..."

    # Try pip install first
    pip install transformer-engine[pytorch]

    # Verify installation
    if python -c "import transformer_engine; print(f'TransformerEngine installed')" 2>/dev/null; then
        echo "  TransformerEngine: OK"
        return 0
    else
        echo "  TransformerEngine: Installation failed"
        echo "  Note: TransformerEngine may require building from source for PyTorch ${TORCH_VERSION}"
        return 1
    fi
}

# Install based on selected backend
if [ "${FP8_BACKEND}" = "TE" ]; then
    if python -c "import transformer_engine" 2>/dev/null; then
        echo "  TransformerEngine: OK (already installed)"
    else
        if ! install_te; then
            echo ""
            echo "WARNING: TransformerEngine installation failed."
            echo "         Falling back to torchao backend..."
            FP8_BACKEND="torchao"
            install_torchao || {
                echo ""
                echo "ERROR: Both FP8 backends failed to install."
                echo "       Please use BF16 training instead:"
                echo "       bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh"
                exit 1
            }
        fi
    fi
elif [ "${FP8_BACKEND}" = "torchao" ]; then
    # Check if torchao is already installed AND compatible
    TORCHAO_OK=$(python -c "
import warnings
warnings.filterwarnings('ignore')
try:
    import torchao
    from torchao.float8 import convert_to_float8_training
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
" 2>&1)

    if [[ "${TORCHAO_OK}" == "OK" ]]; then
        echo "  torchao: OK (already installed and compatible)"
    else
        echo "  torchao status: ${TORCHAO_OK}"
        install_torchao || {
            echo ""
            echo "ERROR: torchao installation failed."
            echo "       Please use BF16 training instead:"
            echo "       bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh"
            exit 1
        }
    fi
fi

# =============================================================================
# Verify FP8 Backend Works
# =============================================================================

echo ""
echo "Verifying FP8 backend..."

if [ "${FP8_BACKEND}" = "torchao" ]; then
    FP8_TEST=$(python -c "
import warnings
warnings.filterwarnings('ignore')
try:
    import torch
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    # Quick test
    model = torch.nn.Linear(32, 32).cuda()
    config = Float8LinearConfig()
    convert_to_float8_training(model, config=config)
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
" 2>&1)

    if [[ "${FP8_TEST}" == "OK" ]]; then
        echo "  torchao FP8 test: PASSED"
    else
        echo "  torchao FP8 test: FAILED"
        echo "  Error: ${FP8_TEST}"
        echo ""
        echo "WARNING: torchao FP8 not working. Falling back to BF16..."
        echo "         Using standard training script instead."
        echo ""
        # Fall back to BF16 script
        exec bash "$(dirname "$0")/train_smolvla_bimanual.sh"
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
EOF
else
    # Use MSAMP as a workaround if torchao backend isn't recognized by accelerate
    # Check accelerate version and torchao backend support
    ACCELERATE_HAS_TORCHAO=$(python -c "
try:
    from accelerate.utils import FP8BackendType
    print('torchao' if hasattr(FP8BackendType, 'TORCHAO') else 'no')
except:
    print('no')
" 2>/dev/null)

    if [ "${ACCELERATE_HAS_TORCHAO}" = "torchao" ]; then
        cat > "${ACCELERATE_CONFIG_FILE}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp8
fp8_config:
  backend: TORCHAO
EOF
    else
        echo "  WARNING: accelerate doesn't recognize torchao backend"
        echo "  Using direct torchao API instead of accelerate FP8..."

        # Create BF16 config - we'll inject FP8 manually via torchao
        cat > "${ACCELERATE_CONFIG_FILE}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
EOF

        # Set flag to use direct torchao injection
        USE_DIRECT_TORCHAO="true"
    fi
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
# Build Training Command
# =============================================================================

# Common training arguments
TRAIN_ARGS="${DATASET_ARG} \
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
TRAIN_ARGS="${TRAIN_ARGS} --rename_map={\"observation.images.head\":\"observation.images.camera1\",\"observation.images.left_wrist\":\"observation.images.camera2\",\"observation.images.right_wrist\":\"observation.images.camera3\"}"

# W&B logging
if [ "${WANDB_ENABLE}" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --wandb.enable=true --wandb.project=${WANDB_PROJECT}"
else
    TRAIN_ARGS="${TRAIN_ARGS} --wandb.enable=false"
fi

# Build command based on FP8 method
if [ "${USE_DIRECT_TORCHAO}" = "true" ]; then
    echo ""
    echo "Using direct torchao FP8 injection (accelerate backend not available)..."
    echo ""

    # Use the FP8 wrapper script which injects FP8 directly via torchao API
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CMD="python ${SCRIPT_DIR}/fp8_train_wrapper.py ${TRAIN_ARGS}"
else
    # Use accelerate launch with FP8 config
    CMD="accelerate launch --config_file ${ACCELERATE_CONFIG_FILE} \
        -m lerobot.scripts.lerobot_train ${TRAIN_ARGS}"
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
