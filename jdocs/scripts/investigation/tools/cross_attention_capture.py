#!/usr/bin/env python3
"""
Cross-Attention Capture for SmolVLA Hallucination Investigation.

Captures cross-attention weights from the action expert to VLM prefix
during inference denoising steps. This is critical for understanding
what visual/language features drive action generation.

Key insight: Vision encoder self-attention shows internal image processing,
but cross-attention shows what the ACTION EXPERT attends to when generating
actions - this is what matters for diagnosing hallucination.

Hook location: smolvlm_with_expert.py:575 (after softmax in eager_attention_forward)

Token layout in VLM prefix (778 total):
  [0-1]     Image special tokens
  [2-730]   Image patches (729 = 27x27 grid from SigLIP)
  [731]     Image end token
  [732-779] Language tokens (~48)
  [780]     State token

Usage:
    python cross_attention_capture.py \
        --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --output-dir logs/yogurt_banana_leftarm/cross_attention_analysis/case1
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import functools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# TOKEN LAYOUT CONSTANTS
# ============================================================================

# SigLIP image patch configuration (384x384 input with 14x14 patch size)
PATCH_GRID_SIZE = 27  # 384 / 14 ≈ 27
NUM_IMAGE_PATCHES = PATCH_GRID_SIZE * PATCH_GRID_SIZE  # 729

# Token indices in VLM prefix
IMAGE_SPECIAL_START = 0
IMAGE_PATCHES_START = 2
IMAGE_PATCHES_END = 2 + NUM_IMAGE_PATCHES  # 731
IMAGE_SPECIAL_END = IMAGE_PATCHES_END + 1  # 732
LANGUAGE_START = IMAGE_SPECIAL_END  # 732
LANGUAGE_END = 780  # Approximate, varies by task
STATE_INDEX = 780

# Action tokens
NUM_ACTION_TOKENS = 50  # Chunk size


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DenoiseStepAttention:
    """Cross-attention data for a single denoising step."""
    step: int  # 0-9
    time: float  # 1.0 -> 0.1
    layer_idx: int  # Which transformer layer

    # Attention statistics
    image_attention_ratio: float = 0.0  # % attention to image patches
    language_attention_ratio: float = 0.0  # % attention to language tokens
    state_attention_ratio: float = 0.0  # % attention to state token
    attention_entropy: float = 0.0  # Shannon entropy (lower = more focused)

    # Spatial attention map (average over heads and action tokens)
    spatial_attention: Optional[np.ndarray] = None  # [27, 27]


@dataclass
class CrossAttentionAnalysis:
    """Complete cross-attention analysis for one inference run."""
    case_dir: str
    task: str
    timestamp: str

    # Per-denoising-step attention (key = step index)
    denoising_steps: dict = field(default_factory=dict)

    # Aggregated metrics
    avg_image_attention: float = 0.0
    avg_language_attention: float = 0.0
    avg_entropy: float = 0.0

    # Temporal trend
    image_attention_trend: str = "stable"  # "increasing", "decreasing", "stable"


# ============================================================================
# CROSS-ATTENTION CAPTURE HOOK
# ============================================================================

class CrossAttentionCaptureHook:
    """
    Captures cross-attention weights during SmolVLA inference.

    Hooks into eager_attention_forward() to capture attention probabilities
    after softmax. Only captures when query length = 50 (action tokens),
    indicating cross-attention rather than self-attention.

    Also hooks into denoise_step() to track which denoising iteration we're in.
    """

    def __init__(self):
        self.attention_data = {}  # {denoising_step: {layer_idx: attention_tensor}}
        self.current_denoising_step = 0
        self.is_capturing = False
        self._original_forward = None
        self._original_denoise_step = None
        self._model = None
        self._hooks = []

    def register_hooks(self, vlm_with_expert, model=None):
        """
        Register hook by wrapping the eager_attention_forward method.

        Note: We wrap the method rather than using register_forward_hook because
        we need to capture intermediate values (attention probs) not just output.
        """
        # Store original forward method and model reference
        self._original_forward = vlm_with_expert.eager_attention_forward
        self._vlm_with_expert = vlm_with_expert

        # Create capturing wrapper that properly replicates the original function
        @functools.wraps(self._original_forward)
        def capturing_forward(
            attention_mask,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        ):
            # Get attention head configuration from model
            num_att_heads = vlm_with_expert.num_attention_heads
            num_key_value_heads = vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads

            sequence_length = key_states.shape[1]

            # Expand key states for grouped query attention (lines 548-553)
            key_states_expanded = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states_expanded = key_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            # Expand value states (lines 555-560)
            value_states_expanded = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            value_states_expanded = value_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            # Upcast to float32 (lines 562-564)
            query_states_f32 = query_states.to(dtype=torch.float32)
            key_states_f32 = key_states_expanded.to(dtype=torch.float32)

            # Transpose for attention computation (lines 566-567)
            query_states_t = query_states_f32.transpose(1, 2)
            key_states_t = key_states_f32.transpose(1, 2)

            # Compute attention weights (lines 569-570)
            att_weights = torch.matmul(query_states_t, key_states_t.transpose(2, 3))
            att_weights = att_weights * (head_dim ** -0.5)

            # Apply mask (lines 572-574)
            att_weights = att_weights.to(dtype=torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

            # Softmax (line 575)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)

            # CAPTURE: Store probs if in cross-attention mode
            # Query shape after transpose: [batch, heads, query_len, head_dim]
            query_len = query_states_t.shape[2]
            key_len = key_states_t.shape[2]

            if self.is_capturing and query_len == NUM_ACTION_TOKENS and key_len > NUM_ACTION_TOKENS:
                step_key = self.current_denoising_step
                if step_key not in self.attention_data:
                    self.attention_data[step_key] = {}

                layer_idx = len(self.attention_data[step_key])
                # Store detached, float32 CPU tensor
                # probs shape: [batch, heads, query_len, key_len]
                self.attention_data[step_key][layer_idx] = probs.detach().float().cpu()

            # Continue with original computation (lines 576-582)
            probs = probs.to(dtype=value_states_expanded.dtype)
            att_output = torch.matmul(probs, value_states_expanded.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )

            return att_output

        # Replace method
        vlm_with_expert.eager_attention_forward = capturing_forward

        # Also hook into the model's denoise_step to track denoising iterations
        if model is not None:
            self._model = model
            self._original_denoise_step = model.denoise_step

            @functools.wraps(self._original_denoise_step)
            def tracking_denoise_step(*args, **kwargs):
                # Increment step counter BEFORE denoise_step runs
                # (step counter starts at 0, increments after each iteration)
                result = self._original_denoise_step(*args, **kwargs)
                # After denoise_step completes, increment for next iteration
                if self.is_capturing:
                    self.current_denoising_step += 1
                return result

            model.denoise_step = tracking_denoise_step

    def remove_hooks(self):
        """Restore original methods."""
        if self._vlm_with_expert is not None and self._original_forward is not None:
            self._vlm_with_expert.eager_attention_forward = self._original_forward
        if self._model is not None and self._original_denoise_step is not None:
            self._model.denoise_step = self._original_denoise_step
        self.attention_data = {}
        self._original_forward = None
        self._original_denoise_step = None

    def start_capture(self, denoising_step: int):
        """Start capturing for a specific denoising step."""
        self.is_capturing = True
        self.current_denoising_step = denoising_step

    def stop_capture(self):
        """Stop capturing."""
        self.is_capturing = False

    def clear(self):
        """Clear all captured data."""
        self.attention_data = {}
        self.current_denoising_step = 0

    def get_step_attention(self, step: int) -> Optional[dict]:
        """Get captured attention for a denoising step."""
        return self.attention_data.get(step)


# ============================================================================
# SPATIAL ATTENTION MAPPING
# ============================================================================

def map_attention_to_spatial(
    cross_attention: np.ndarray,
    patch_grid_size: int = PATCH_GRID_SIZE,
    debug: bool = False,
) -> np.ndarray:
    """
    Map cross-attention weights to spatial image regions.

    Args:
        cross_attention: Attention weights [num_heads, action_tokens, prefix_tokens]
                        or [batch, num_heads, action_tokens, prefix_tokens]
        debug: Print debug info about token layout

    Returns:
        spatial_attention: [patch_grid_size, patch_grid_size] normalized attention map
    """
    # Handle batch dimension
    if cross_attention.ndim == 4:
        cross_attention = cross_attention[0]  # [heads, actions, prefix]

    num_heads, num_actions, prefix_len = cross_attention.shape

    if debug:
        print(f"    Cross-attention shape: heads={num_heads}, actions={num_actions}, prefix={prefix_len}")

    # SmolVLA uses dynamic image token count based on the multi_modal_projector
    # The actual number of image tokens can vary. Let's infer it from the prefix length.
    #
    # Typical layout: [image_special] + [image_patches] + [language_tokens] + [state]
    # Language tokens are typically ~20-50, state is 1
    #
    # For prefix_len=291 with ~25 language tokens + 1 state = ~265 image tokens
    # This suggests a different patch configuration than 27x27

    # Try to infer the number of image tokens
    # Assume language tokens + state tokens ≈ 50 tokens at the end
    estimated_lang_state = min(50, prefix_len // 5)  # Conservative estimate
    estimated_image_tokens = prefix_len - estimated_lang_state

    # Find closest perfect square for spatial mapping
    sqrt_tokens = int(np.sqrt(estimated_image_tokens))
    actual_grid_size = sqrt_tokens

    if debug:
        print(f"    Estimated image tokens: {estimated_image_tokens}, grid: {actual_grid_size}x{actual_grid_size}")

    # Extract attention to estimated image region (all but last ~50 tokens)
    image_end = prefix_len - estimated_lang_state
    attn_to_image = cross_attention[:, :, :image_end]

    # Average over heads and action tokens
    attn_avg = attn_to_image.mean(axis=(0, 1))

    # Reshape to spatial grid (use the closest perfect square)
    target_size = actual_grid_size * actual_grid_size
    if len(attn_avg) >= target_size:
        attn_to_reshape = attn_avg[:target_size]
    else:
        # Pad if needed
        attn_to_reshape = np.pad(attn_avg, (0, target_size - len(attn_avg)))

    spatial_attn = attn_to_reshape.reshape(actual_grid_size, actual_grid_size)

    # Normalize to [0, 1]
    spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-10)

    return spatial_attn


def compute_attention_metrics(
    cross_attention: np.ndarray,
    prefix_len: Optional[int] = None,
    debug: bool = False,
) -> dict:
    """
    Compute attention distribution metrics.

    Args:
        cross_attention: [heads, actions, prefix] or [batch, heads, actions, prefix]
        debug: Print debug info

    Returns:
        dict with image_ratio, language_ratio, state_ratio, entropy
    """
    if cross_attention.ndim == 4:
        cross_attention = cross_attention[0]

    num_heads, num_actions, actual_prefix_len = cross_attention.shape

    # Estimate token boundaries based on actual prefix length
    # Assume last ~50 tokens are language + state
    estimated_lang_state = min(50, actual_prefix_len // 5)
    img_end = actual_prefix_len - estimated_lang_state
    lang_start = img_end
    lang_end = actual_prefix_len - 1  # Last token is state
    state_idx = actual_prefix_len - 1

    if debug:
        print(f"    Token layout: image=[0:{img_end}], lang=[{lang_start}:{lang_end}], state=[{state_idx}]")

    # Compute total attention per region (sum over heads and actions)
    total_attn = cross_attention.sum()

    # Image attention
    image_attn = cross_attention[:, :, :img_end].sum() / total_attn if img_end > 0 else 0

    # Language attention
    if lang_end > lang_start:
        lang_attn = cross_attention[:, :, lang_start:lang_end].sum() / total_attn
    else:
        lang_attn = 0

    # State attention
    state_attn = cross_attention[:, :, state_idx:].sum() / total_attn

    # Entropy (measure of focus vs diffusion)
    # Average over heads, then compute entropy over prefix tokens
    attn_avg = cross_attention.mean(axis=0)  # [actions, prefix]
    attn_flat = attn_avg.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-10)
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))

    return {
        "image_attention_ratio": float(image_attn),
        "language_attention_ratio": float(lang_attn),
        "state_attention_ratio": float(state_attn),
        "attention_entropy": float(entropy),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_attention_heatmap_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay attention heatmap on image.

    Args:
        image: RGB image [H, W, 3]
        attention_map: Normalized attention [patch_h, patch_w]
        alpha: Blend factor

    Returns:
        Image with heatmap overlay [H, W, 3]
    """
    h, w = image.shape[:2]

    # Resize attention map to image size
    attn_resized = cv2.resize(attention_map.astype(np.float32), (w, h))

    # Create heatmap using jet colormap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def plot_temporal_evolution(
    step_data: dict[int, DenoiseStepAttention],
    images: Optional[list[np.ndarray]] = None,
    output_path: Path = None,
    title: str = "Cross-Attention Evolution Through Denoising",
):
    """
    Plot 10-panel figure showing attention evolution across denoising steps.

    Args:
        step_data: {step_idx: DenoiseStepAttention}
        images: Optional list of images to overlay attention on
        output_path: Where to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for step in range(10):
        row = step // 5
        col = step % 5
        ax = axes[row, col]

        if step in step_data and step_data[step].spatial_attention is not None:
            spatial_attn = step_data[step].spatial_attention
            time = step_data[step].time
            entropy = step_data[step].attention_entropy

            if images and len(images) > 0:
                # Overlay on first image
                overlay = create_attention_heatmap_overlay(images[0], spatial_attn, alpha=0.6)
                ax.imshow(overlay)
            else:
                # Just show attention map
                im = ax.imshow(spatial_attn, cmap='hot', vmin=0, vmax=1)

            ax.set_title(f"Step {step} (t={time:.1f})\nH={entropy:.2f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Step {step}\nNo data", ha='center', va='center')

        ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal evolution: {output_path}")

    plt.close()


def plot_attention_metrics_over_time(
    step_data: dict[int, DenoiseStepAttention],
    output_path: Path = None,
    title: str = "Attention Metrics Across Denoising Steps",
):
    """
    Plot line graphs of attention metrics over denoising steps.
    """
    steps = sorted(step_data.keys())
    if not steps:
        return

    image_ratios = [step_data[s].image_attention_ratio for s in steps]
    lang_ratios = [step_data[s].language_attention_ratio for s in steps]
    entropies = [step_data[s].attention_entropy for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Image attention ratio
    axes[0].plot(steps, image_ratios, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Denoising Step')
    axes[0].set_ylabel('Image Attention Ratio')
    axes[0].set_title('Attention to Image Patches')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Language attention ratio
    axes[1].plot(steps, lang_ratios, 'g-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Denoising Step')
    axes[1].set_ylabel('Language Attention Ratio')
    axes[1].set_title('Attention to Language Tokens')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Entropy
    axes[2].plot(steps, entropies, 'r-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('Denoising Step')
    axes[2].set_ylabel('Attention Entropy')
    axes[2].set_title('Attention Focus (lower = more focused)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention metrics: {output_path}")

    plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_case_cross_attention(
    case_dir: Path,
    output_dir: Path,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> CrossAttentionAnalysis:
    """
    Analyze cross-attention for a captured inference case.

    This loads the saved images from the case, runs inference with
    cross-attention capture hooks, and generates analysis.
    """
    print(f"Analyzing cross-attention for case: {case_dir}")

    # Load metadata
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {case_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    task = metadata.get("task_original", "Unknown task")
    checkpoint = checkpoint_path or metadata.get("checkpoint")
    print(f"Task: {task}")
    print(f"Checkpoint: {checkpoint}")

    # Load images
    images_dir = case_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images directory in {case_dir}")

    image_files = sorted(images_dir.glob("step_*_head.jpg"))
    if not image_files:
        raise FileNotFoundError("No head camera images found")

    # We'll analyze at specific inference steps (50-step intervals)
    # Pick representative steps: 200, 250, 300 (critical for hallucination)
    priority_steps = [0, 100, 200, 250, 300, 350]
    available_steps = sorted(set(int(p.stem.split('_')[1]) for p in image_files))
    steps_to_analyze = [s for s in priority_steps if s in available_steps]

    if not steps_to_analyze:
        steps_to_analyze = available_steps[:3]

    print(f"Will analyze steps: {steps_to_analyze}")

    # Load model
    print("Loading SmolVLA policy...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.eval()
    policy.to(device)

    # Load preprocessor
    dataset_path = PROJECT_ROOT / "datasets_bimanuel" / "multitasks"
    dataset_metadata = LeRobotDatasetMetadata(repo_id="multitasks", root=str(dataset_path))

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Setup cross-attention capture hook
    capture = CrossAttentionCaptureHook()
    capture.register_hooks(policy.model.vlm_with_expert, model=policy.model)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis results
    analysis = CrossAttentionAnalysis(
        case_dir=str(case_dir),
        task=task,
        timestamp=datetime.now().isoformat(),
    )

    all_step_data = {}  # {inference_step: {denoising_step: DenoiseStepAttention}}
    images_for_overlay = {}  # {inference_step: image}

    # Process each inference step
    for step_num in steps_to_analyze:
        print(f"\n  Processing inference step {step_num}...")

        # Load images for this step
        head_path = images_dir / f"step_{step_num:04d}_head.jpg"
        left_wrist_path = images_dir / f"step_{step_num:04d}_left_wrist.jpg"
        right_wrist_path = images_dir / f"step_{step_num:04d}_right_wrist.jpg"

        if not all(p.exists() for p in [head_path, left_wrist_path, right_wrist_path]):
            print(f"    Missing camera images for step {step_num}")
            continue

        # Load images
        head_img = cv2.cvtColor(cv2.imread(str(head_path)), cv2.COLOR_BGR2RGB)
        left_wrist_img = cv2.cvtColor(cv2.imread(str(left_wrist_path)), cv2.COLOR_BGR2RGB)
        right_wrist_img = cv2.cvtColor(cv2.imread(str(right_wrist_path)), cv2.COLOR_BGR2RGB)

        # Store image for this inference step
        images_for_overlay[step_num] = head_img

        # Create observation
        state = np.zeros(12, dtype=np.float32)

        def img_to_tensor(img):
            return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        observation = {
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device),
            "observation.images.camera1": img_to_tensor(head_img).to(device),
            "observation.images.camera2": img_to_tensor(left_wrist_img).to(device),
            "observation.images.camera3": img_to_tensor(right_wrist_img).to(device),
            "task": task,
        }

        preprocessed_obs = preprocessor(observation)

        # Reset policy to force new chunk generation
        policy.reset()

        # Clear previous capture
        capture.clear()

        # Run inference with capture enabled for all denoising steps
        # Note: We need to modify the policy's sample_actions to capture per-step
        # For now, we capture whatever cross-attention happens during inference

        capture.is_capturing = True

        try:
            with torch.no_grad():
                action = policy.select_action(preprocessed_obs)
            print(f"    Inference OK")
        except Exception as e:
            print(f"    Inference error: {e}")
            continue

        capture.is_capturing = False

        # Process captured attention
        # The capture contains attention from multiple denoising steps
        # (each step calls eager_attention_forward multiple times)

        # Initialize nested dict for this inference step
        if step_num not in all_step_data:
            all_step_data[step_num] = {}

        for denoise_step, layer_data in capture.attention_data.items():
            # Use attention from middle layer (layer 8 typically has semantic info)
            target_layer = min(8, len(layer_data) - 1)

            if target_layer in layer_data:
                attn = layer_data[target_layer].numpy()

                # Compute spatial map and metrics
                spatial_attn = map_attention_to_spatial(attn, debug=(denoise_step == 0 and step_num == steps_to_analyze[0]))
                metrics = compute_attention_metrics(attn, debug=(denoise_step == 0 and step_num == steps_to_analyze[0]))

                # Estimate time from step (t = 1.0 - step * 0.1)
                time = 1.0 - denoise_step * 0.1

                step_attention = DenoiseStepAttention(
                    step=denoise_step,
                    time=time,
                    layer_idx=target_layer,
                    image_attention_ratio=metrics["image_attention_ratio"],
                    language_attention_ratio=metrics["language_attention_ratio"],
                    state_attention_ratio=metrics["state_attention_ratio"],
                    attention_entropy=metrics["attention_entropy"],
                    spatial_attention=spatial_attn,
                )

                all_step_data[step_num][denoise_step] = step_attention

        print(f"    Captured {len(capture.attention_data)} denoising steps")

    # Flatten all data for aggregation: collect all DenoiseStepAttention objects
    all_attention_data = []
    for inf_step, denoise_dict in all_step_data.items():
        for denoise_step, data in denoise_dict.items():
            all_attention_data.append(data)

    # Compute aggregated metrics
    if all_attention_data:
        analysis.avg_image_attention = np.mean([d.image_attention_ratio for d in all_attention_data])
        analysis.avg_language_attention = np.mean([d.language_attention_ratio for d in all_attention_data])
        analysis.avg_entropy = np.mean([d.attention_entropy for d in all_attention_data])

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Temporal evolution - use last inference step's denoising progression
    last_inf_step = max(all_step_data.keys())
    plot_temporal_evolution(
        all_step_data[last_inf_step],
        images=[images_for_overlay.get(last_inf_step)],
        output_path=output_dir / "temporal_evolution.png",
        title=f"Cross-Attention Evolution (inf step {last_inf_step})\nTask: {task[:50]}...",
    )

    # Metrics over time for last inference step
    plot_attention_metrics_over_time(
        all_step_data[last_inf_step],
        output_path=output_dir / "attention_metrics.png",
    )

    # Save individual spatial attention maps for ALL inference steps
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)

    for inf_step, denoise_dict in all_step_data.items():
        for denoise_step, data in denoise_dict.items():
            if data.spatial_attention is not None:
                # Save raw spatial attention as .npy for quantitative analysis
                # Format: inf{inference_step}_denoise{denoising_step}
                np.save(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_spatial.npy", data.spatial_attention)

                if inf_step in images_for_overlay:
                    overlay = create_attention_heatmap_overlay(images_for_overlay[inf_step], data.spatial_attention)

                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(overlay)
                    ax.set_title(f"Inf {inf_step}, Denoise {denoise_step} (t={data.time:.1f})\n"
                                f"Image: {data.image_attention_ratio:.2f}, "
                                f"Lang: {data.language_attention_ratio:.2f}, "
                                f"Entropy: {data.attention_entropy:.2f}")
                    ax.axis('off')
                    plt.savefig(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}.png", dpi=150, bbox_inches='tight')
                    plt.close()

    # Store in analysis with nested structure
    analysis_dict = {}
    for inf_step, denoise_dict in all_step_data.items():
        analysis_dict[f"inf_{inf_step}"] = {
            f"denoise_{denoise_step}": {
                "step": data.step,
                "time": data.time,
                "layer_idx": data.layer_idx,
                "image_attention_ratio": data.image_attention_ratio,
                "language_attention_ratio": data.language_attention_ratio,
                "state_attention_ratio": data.state_attention_ratio,
                "attention_entropy": data.attention_entropy,
            }
            for denoise_step, data in denoise_dict.items()
        }

    # Save analysis data
    analysis_data = {
        "case_dir": str(case_dir),
        "task": task,
        "timestamp": analysis.timestamp,
        "avg_image_attention": analysis.avg_image_attention,
        "avg_language_attention": analysis.avg_language_attention,
        "avg_entropy": analysis.avg_entropy,
        "inference_steps": list(all_step_data.keys()),
        "attention_data": analysis_dict,
    }

    with open(output_dir / "cross_attention_analysis.json", 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\nAnalysis saved to: {output_dir}")
    print(f"  - temporal_evolution.png")
    print(f"  - attention_metrics.png")
    print(f"  - heatmaps/")
    print(f"  - cross_attention_analysis.json")

    return analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Attention Capture for SmolVLA Hallucination Investigation"
    )

    parser.add_argument("--case-dir", type=Path, required=True,
                       help="Case directory with images and metadata")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Model checkpoint (uses case metadata if not specified)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                       help="Compare two cases side-by-side")
    parser.add_argument("--case1", type=Path, help="First case for comparison")
    parser.add_argument("--case2", type=Path, help="Second case for comparison")

    args = parser.parse_args()

    print("=" * 60)
    print("SmolVLA Cross-Attention Analysis")
    print("=" * 60)

    if args.compare:
        if not args.case1 or not args.case2:
            print("ERROR: --case1 and --case2 required for comparison mode")
            sys.exit(1)

        print(f"Comparing:")
        print(f"  Case 1: {args.case1}")
        print(f"  Case 2: {args.case2}")

        out1 = args.output_dir / "case1"
        out2 = args.output_dir / "case2"

        analyze_case_cross_attention(args.case1, out1, args.checkpoint, args.device)
        analyze_case_cross_attention(args.case2, out2, args.checkpoint, args.device)

        # TODO: Generate comparison visualization

    else:
        analyze_case_cross_attention(
            args.case_dir,
            args.output_dir,
            args.checkpoint,
            args.device,
        )


if __name__ == "__main__":
    main()
