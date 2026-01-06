#!/usr/bin/env python3
"""
Experiment 3: Attention Pattern Visualization

Visualizes cross-attention patterns during VLA inference to understand:
1. Which image regions attend to which language tokens
2. Whether target words ("plate", "bin") receive appropriate attention
3. Whether object words ("tissue", "corn") ground to correct visual regions

Hypothesis:
If attention to target/object words is diffuse (not focused), the model
cannot ground language to relevant visual features, explaining why it
ignores task descriptions and follows visual priors.

Usage:
    # With trained checkpoint
    python visualize_attention.py \\
        --checkpoint outputs/smolvla_bimanual_*/checkpoints/*/pretrained_model \\
        --image datasets_bimanuel/multitasks/videos/episode_000001/frame_0050.jpg

    # With sample image
    python visualize_attention.py \\
        --checkpoint outputs/smolvla_bimanual \\
        --image sample_scene.jpg \\
        --task "Use right arm to pick up the tissue and place it on the plate"

Output:
    - Attention heatmaps overlaid on image (per layer, per token)
    - Summary statistics of attention distribution
    - Token-to-region correlation analysis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# ATTENTION HOOKS
# ============================================================================

class AttentionCapture:
    """Context manager to capture attention weights during forward pass."""

    def __init__(self):
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on attention layers."""

        def make_hook(layer_name):
            def hook(module, input, output):
                # Different models store attention differently
                if hasattr(output, 'attentions') and output.attentions is not None:
                    self.attention_weights[layer_name] = output.attentions
                elif isinstance(output, tuple) and len(output) > 1:
                    # Some models return (hidden_states, attention_weights)
                    if output[1] is not None:
                        self.attention_weights[layer_name] = output[1]
            return hook

        # Try to find attention layers in VLM and expert
        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'forward'):
                    hook = module.register_forward_hook(make_hook(name))
                    self.hooks.append(hook)
                    print(f"  Registered hook on: {name}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights = {}


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_attention_heatmap(
    attention_weights: np.ndarray,
    image: np.ndarray,
    token_idx: int,
    num_patches_h: int = 16,
    num_patches_w: int = 16
) -> np.ndarray:
    """
    Create attention heatmap overlay on image.

    Args:
        attention_weights: Attention weights [seq_len, seq_len]
        image: Original image [H, W, 3]
        token_idx: Which token's attention to visualize
        num_patches_h: Number of image patches vertically
        num_patches_w: Number of image patches horizontally

    Returns:
        Image with attention heatmap overlay
    """
    h, w = image.shape[:2]

    # Get attention from token to image patches
    # Assuming image patches are at the beginning of the sequence
    num_patches = num_patches_h * num_patches_w
    attn_to_image = attention_weights[token_idx, :num_patches]

    # Reshape to spatial grid
    attn_map = attn_to_image.reshape(num_patches_h, num_patches_w)

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Resize to image dimensions
    attn_map_resized = cv2.resize(attn_map, (w, h))

    # Create heatmap overlay
    heatmap = plt.cm.jet(attn_map_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay


def plot_token_attention_summary(
    attention_weights: np.ndarray,
    tokens: list[str],
    output_path: Path,
    title: str = "Token Attention Distribution"
):
    """
    Plot summary of which tokens receive most attention.

    Args:
        attention_weights: Attention weights [num_heads, seq_len, seq_len]
        tokens: List of token strings
        output_path: Where to save the plot
        title: Plot title
    """
    # Average over heads
    if len(attention_weights.shape) == 3:
        attn = attention_weights.mean(axis=0)  # [seq_len, seq_len]
    else:
        attn = attention_weights

    # Sum attention TO each token (column-wise sum)
    attention_received = attn.sum(axis=0)

    # Normalize
    attention_received = attention_received / attention_received.sum()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(tokens))
    bars = ax.bar(x, attention_received[:len(tokens)])

    # Highlight target words
    target_words = ['plate', 'bin', 'place', 'pick']
    for i, token in enumerate(tokens):
        token_lower = token.lower().strip()
        if any(t in token_lower for t in target_words):
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)

    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Attention Received (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention summary to: {output_path}")


def analyze_target_word_attention(
    attention_weights: np.ndarray,
    tokens: list[str],
    target_words: list[str] = ['plate', 'bin']
) -> dict:
    """
    Analyze attention specifically on target location words.

    Returns statistics about whether target words receive focused attention.
    """
    # Find target word indices
    target_indices = []
    for i, token in enumerate(tokens):
        token_lower = token.lower().strip()
        if any(t in token_lower for t in target_words):
            target_indices.append((i, token))

    if not target_indices:
        return {"error": "No target words found in tokens"}

    # Average attention over heads if needed
    if len(attention_weights.shape) == 3:
        attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights

    # Attention TO target words (from all tokens)
    results = {}
    total_attention = attn.sum()

    for idx, token in target_indices:
        attention_to_token = attn[:, idx].sum() / total_attention
        attention_from_token = attn[idx, :].sum() / total_attention

        results[token] = {
            "token_index": idx,
            "attention_received": float(attention_to_token),
            "attention_given": float(attention_from_token)
        }

    # Calculate entropy of attention to targets (lower = more focused)
    target_attention = np.array([results[t]["attention_received"] for _, t in target_indices])
    if len(target_attention) > 1:
        target_attention = target_attention / (target_attention.sum() + 1e-8)
        entropy = -np.sum(target_attention * np.log(target_attention + 1e-8))
        results["target_attention_entropy"] = float(entropy)

    return results


# ============================================================================
# INFERENCE WITH ATTENTION CAPTURE
# ============================================================================

def run_inference_with_attention(
    checkpoint_path: str,
    image_path: str,
    task_description: str,
    device: str = "cuda"
) -> tuple[dict, list[str]]:
    """
    Run SmolVLA inference and capture attention weights.

    Returns:
        attention_data: Dict of layer_name -> attention weights
        tokens: List of token strings
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Import here to avoid circular imports
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    except ImportError:
        print("ERROR: Cannot import SmolVLA. Make sure lerobot is installed.")
        sys.exit(1)

    # Load checkpoint
    checkpoint_dir = Path(checkpoint_path)
    config_path = checkpoint_dir / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = SmolVLAConfig(**config_dict)
    else:
        config = SmolVLAConfig()

    # Enable attention output
    config.output_attentions = True

    # Load model
    policy = SmolVLAPolicy(config)
    policy.load_state_dict(torch.load(checkpoint_dir / "model.safetensors", map_location=device))
    policy = policy.to(device)
    policy.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    image_resized = cv2.resize(image, (512, 512))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Tokenize task description
    tokenizer = policy.processor.tokenizer
    tokens = tokenizer.tokenize(task_description)
    token_ids = tokenizer.encode(task_description, return_tensors="pt").to(device)

    # Setup attention capture
    attention_capture = AttentionCapture()
    attention_capture.register_hooks(policy.model)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Create dummy robot state
        state = torch.zeros(1, 12).to(device)  # 12 DOF for bimanual

        # Forward pass
        observation = {
            "observation.images.camera1": image_tensor,
            "observation.state": state,
            "task": task_description
        }

        try:
            action = policy.select_action(observation)
        except Exception as e:
            print(f"Inference error (may still have attention): {e}")

    # Cleanup hooks
    attention_capture.remove_hooks()

    return attention_capture.attention_weights, tokens


# ============================================================================
# SIMPLIFIED STANDALONE ANALYSIS (without full model loading)
# ============================================================================

def analyze_tokenization_only(
    task_description: str,
    model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
) -> list[str]:
    """
    Analyze how a task description is tokenized.
    Useful for understanding which tokens correspond to target words.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Get tokens
    tokens = tokenizer.tokenize(task_description)
    token_ids = tokenizer.encode(task_description)

    print("\nTokenization Analysis:")
    print("-" * 50)
    print(f"Task: {task_description}")
    print(f"Tokens ({len(tokens)}):")

    for i, (tok, tid) in enumerate(zip(tokens, token_ids[1:])):  # Skip [CLS]
        print(f"  {i:3d}: {tok:20s} (id={tid})")

    # Find target words
    target_words = ['plate', 'bin', 'pick', 'place']
    print("\nTarget word positions:")
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().replace('▁', '').strip()
        for target in target_words:
            if target in tok_lower:
                print(f"  '{target}' found at position {i}: '{tok}'")

    return tokens


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns")
    parser.add_argument(
        "--checkpoint",
        help="Path to SmolVLA checkpoint"
    )
    parser.add_argument(
        "--image",
        help="Path to input image"
    )
    parser.add_argument(
        "--task",
        default="Use right arm to pick up the tissue and place it on the plate",
        help="Task description"
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Only analyze tokenization (no model loading)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "attention_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENT 3: Attention Pattern Visualization")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenization-only mode (quick check without model)
    if args.tokenize_only:
        print("\nMode: Tokenization analysis only")
        tokens = analyze_tokenization_only(args.task)

        # Save tokenization analysis
        output_file = args.output_dir / "tokenization_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                "task": args.task,
                "tokens": tokens,
                "num_tokens": len(tokens)
            }, f, indent=2)
        print(f"\nSaved to: {output_file}")
        return

    # Full analysis requires checkpoint and image
    if not args.checkpoint or not args.image:
        print("ERROR: --checkpoint and --image required for full analysis")
        print("Use --tokenize-only for quick tokenization check")
        sys.exit(1)

    # Run inference with attention capture
    attention_data, tokens = run_inference_with_attention(
        args.checkpoint,
        args.image,
        args.task,
        args.device
    )

    if not attention_data:
        print("WARNING: No attention weights captured.")
        print("The model may not output attention weights by default.")
        print("Try running tokenization analysis with --tokenize-only")
        return

    print(f"\nCaptured attention from {len(attention_data)} layers")

    # Analyze each layer
    for layer_name, attn in attention_data.items():
        print(f"\nAnalyzing layer: {layer_name}")

        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()

        # Target word analysis
        target_analysis = analyze_target_word_attention(attn, tokens)
        print(f"  Target word attention: {json.dumps(target_analysis, indent=2)}")

        # Plot attention summary
        layer_safe_name = layer_name.replace("/", "_").replace(".", "_")
        plot_token_attention_summary(
            attn,
            tokens,
            args.output_dir / f"attention_summary_{layer_safe_name}.png",
            f"Attention Distribution - {layer_name}"
        )

    # Save full analysis
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "tokens": tokens,
        "layers_analyzed": list(attention_data.keys()),
        "target_words_found": ["plate", "bin"]  # TODO: dynamic detection
    }

    with open(args.output_dir / "attention_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print("\nInterpretation:")
    print("  - If target words receive < 5% attention: Language is ignored")
    print("  - If attention is diffuse (high entropy): No focused grounding")
    print("  - Red bars in plots indicate target-related tokens")


if __name__ == "__main__":
    main()
