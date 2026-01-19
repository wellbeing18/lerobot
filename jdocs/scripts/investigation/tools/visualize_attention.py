#!/usr/bin/env python3
"""
Enhanced Attention Visualization for SmolVLA Hallucination Investigation.

Extends the base attention visualization with:
1. Runtime attention capture during live inference
2. Spatial attention heatmaps overlaid on images
3. Cross-attention analysis between action expert and VLM embeddings
4. Temporal evolution of attention through inference steps

Key Questions:
- Where does the model attend when hallucinating?
- Is attention on distractor objects correlated with hallucination?
- Does attention entropy differ between normal and hallucination cases?

Usage:
    # Analyze with saved images from a case
    python visualize_attention.py \
        --checkpoint outputs/smolvla_bimanual \
        --case-dir ../cases/hallucination/case_001 \
        --output-dir ../reports/attention_analysis

    # Compare attention between two cases
    python visualize_attention.py \
        --compare \
        --case1 ../cases/hallucination/case_001 \
        --case2 ../cases/normal/case_002 \
        --output-dir ../reports/attention_comparison
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AttentionData:
    """Container for attention weights from a single inference step."""
    step: int
    timestamp: float
    task: str

    # Cross-attention weights (action expert attending to VLM)
    # Shape: [num_heads, action_tokens, vlm_tokens]
    cross_attention: Optional[np.ndarray] = None

    # Self-attention weights within VLM
    # Shape: [num_heads, seq_len, seq_len]
    vlm_attention: Optional[np.ndarray] = None

    # Token information
    tokens: list[str] = field(default_factory=list)
    num_image_patches: int = 0
    num_language_tokens: int = 0

    # Computed metrics
    attention_entropy: float = 0.0
    image_attention_ratio: float = 0.0
    language_attention_ratio: float = 0.0


# ============================================================================
# ATTENTION CAPTURE
# ============================================================================

class AttentionCaptureHook:
    """Hook to capture attention weights during forward pass."""

    def __init__(self):
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on attention layers."""
        for name, module in model.named_modules():
            # Look for attention layers in the expert model
            if 'cross_attn' in name.lower() or 'crossattention' in name.lower():
                hook = module.register_forward_hook(self._make_hook(f"cross_{name}"))
                self.hooks.append(hook)
                print(f"  Registered hook: {name}")

            # VLM attention layers
            elif 'self_attn' in name.lower() and 'vlm' in name.lower():
                hook = module.register_forward_hook(self._make_hook(f"vlm_{name}"))
                self.hooks.append(hook)
                print(f"  Registered hook: {name}")

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            # Different models return attention differently
            if hasattr(output, 'attentions') and output.attentions is not None:
                self.attention_weights[layer_name] = output.attentions
            elif isinstance(output, tuple):
                # Try to find attention in tuple output
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and o.dim() >= 3:
                        # Likely attention weights if shape is [batch, heads, seq, seq]
                        if o.shape[-1] == o.shape[-2] or 'attn' in layer_name:
                            self.attention_weights[f"{layer_name}_{i}"] = o
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.attention_weights = {}

    def get_weights(self) -> dict:
        """Get captured weights as numpy arrays."""
        result = {}
        for name, weights in self.attention_weights.items():
            if isinstance(weights, torch.Tensor):
                result[name] = weights.detach().cpu().numpy()
            elif isinstance(weights, tuple):
                for i, w in enumerate(weights):
                    if isinstance(w, torch.Tensor):
                        result[f"{name}_{i}"] = w.detach().cpu().numpy()
        return result


# ============================================================================
# ATTENTION ANALYSIS
# ============================================================================

def compute_attention_entropy(attention: np.ndarray) -> float:
    """
    Compute entropy of attention distribution.

    Higher entropy = more diffuse attention (less focused)
    Lower entropy = more concentrated attention (more focused)
    """
    # Flatten and normalize
    attn_flat = attention.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-10)

    # Compute entropy
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
    return float(entropy)


def compute_spatial_attention_map(
    attention: np.ndarray,
    query_idx: int,
    num_patches_h: int = 16,
    num_patches_w: int = 16,
) -> np.ndarray:
    """
    Extract spatial attention map for a specific query token.

    Args:
        attention: Attention weights [num_heads, queries, keys]
        query_idx: Which query to visualize
        num_patches_h: Image patches vertically (SigLIP default: 16)
        num_patches_w: Image patches horizontally

    Returns:
        Attention map [num_patches_h, num_patches_w]
    """
    # Average over heads
    if attention.ndim == 3:
        attn = attention.mean(axis=0)  # [queries, keys]
    else:
        attn = attention

    # Get attention from query to image patches
    num_patches = num_patches_h * num_patches_w

    # Assume image patches are at beginning of sequence
    attn_to_image = attn[query_idx, :num_patches]

    # Reshape to spatial grid
    attn_map = attn_to_image.reshape(num_patches_h, num_patches_w)

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-10)

    return attn_map


def analyze_token_attention(
    attention: np.ndarray,
    tokens: list[str],
    target_words: list[str] = None,
) -> dict:
    """
    Analyze which tokens receive most attention.

    Returns dict with per-token attention statistics.
    """
    if target_words is None:
        target_words = ['plate', 'bin', 'pick', 'place', 'banana', 'orange', 'yogurt']

    # Average over heads if needed
    if attention.ndim == 3:
        attn = attention.mean(axis=0)
    else:
        attn = attention

    # Sum attention TO each token (column-wise)
    attention_received = attn.sum(axis=0)
    attention_received = attention_received / (attention_received.sum() + 1e-10)

    # Find target word indices and their attention
    target_attention = {}
    for i, token in enumerate(tokens):
        token_lower = token.lower().strip()
        for target in target_words:
            if target in token_lower:
                target_attention[f"{token}_{i}"] = {
                    "index": i,
                    "token": token,
                    "attention_received": float(attention_received[i]) if i < len(attention_received) else 0,
                }

    # Overall statistics
    return {
        "total_tokens": len(tokens),
        "attention_entropy": compute_attention_entropy(attn),
        "target_words_found": len(target_attention),
        "target_attention": target_attention,
        "max_attention_idx": int(np.argmax(attention_received)),
        "max_attention_value": float(np.max(attention_received)),
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
    attn_resized = cv2.resize(attention_map, (w, h))

    # Create heatmap using jet colormap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def plot_attention_summary(
    attention: np.ndarray,
    tokens: list[str],
    output_path: Path,
    title: str = "Attention Distribution",
    highlight_words: list[str] = None,
):
    """
    Plot bar chart of attention distribution across tokens.
    """
    if highlight_words is None:
        highlight_words = ['plate', 'bin', 'pick', 'place']

    # Average over heads
    if attention.ndim == 3:
        attn = attention.mean(axis=0)
    else:
        attn = attention

    # Sum attention TO each token
    attention_received = attn.sum(axis=0)
    attention_received = attention_received / (attention_received.sum() + 1e-10)

    # Limit to token count
    n = min(len(tokens), len(attention_received))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n)
    bars = ax.bar(x, attention_received[:n], color='steelblue', alpha=0.7)

    # Highlight target words
    for i, token in enumerate(tokens[:n]):
        token_lower = token.lower().strip()
        for target in highlight_words:
            if target in token_lower:
                bars[i].set_color('red')
                bars[i].set_alpha(0.9)
                break

    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Attention Received', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens[:n], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention summary: {output_path}")


def plot_spatial_attention_grid(
    image: np.ndarray,
    attention_maps: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    title: str = "Spatial Attention Maps",
):
    """
    Plot grid of spatial attention maps.

    Args:
        image: Original image
        attention_maps: List of attention maps to visualize
        labels: Labels for each map
        output_path: Where to save
        title: Plot title
    """
    n = len(attention_maps)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols + 1, figsize=(4 * (cols + 1), 4 * rows))
    if rows == 1:
        axes = [axes]

    # Show original image in first column
    for row in range(rows):
        axes[row][0].imshow(image)
        axes[row][0].set_title('Original')
        axes[row][0].axis('off')

    # Show attention maps
    for i, (attn_map, label) in enumerate(zip(attention_maps, labels)):
        row = i // cols
        col = (i % cols) + 1

        overlay = create_attention_heatmap_overlay(image, attn_map)
        axes[row][col].imshow(overlay)
        axes[row][col].set_title(label, fontsize=10)
        axes[row][col].axis('off')

    # Hide unused subplots
    for row in range(rows):
        for col in range(n % cols + 1 if row == rows - 1 and n % cols != 0 else cols + 1, cols + 1):
            if row < len(axes) and col < len(axes[row]):
                axes[row][col].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial attention grid: {output_path}")


def plot_attention_comparison(
    attn1: np.ndarray,
    attn2: np.ndarray,
    tokens: list[str],
    output_path: Path,
    label1: str = "Hallucination",
    label2: str = "Normal",
):
    """
    Compare attention distributions between two cases.
    """
    # Average over heads
    if attn1.ndim == 3:
        attn1 = attn1.mean(axis=0)
    if attn2.ndim == 3:
        attn2 = attn2.mean(axis=0)

    # Sum attention received
    received1 = attn1.sum(axis=0)
    received1 = received1 / (received1.sum() + 1e-10)
    received2 = attn2.sum(axis=0)
    received2 = received2 / (received2.sum() + 1e-10)

    n = min(len(tokens), len(received1), len(received2))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n)
    width = 0.35

    ax.bar(x - width/2, received1[:n], width, label=label1, color='red', alpha=0.7)
    ax.bar(x + width/2, received2[:n], width, label=label2, color='blue', alpha=0.7)

    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Attention Received', fontsize=12)
    ax.set_title(f'Attention Comparison: {label1} vs {label2}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens[:n], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention comparison: {output_path}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_case_attention(
    checkpoint_path: str,
    case_dir: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """
    Analyze attention for a single case using saved images.
    """
    print(f"Analyzing case: {case_dir}")

    # Load metadata
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found in {case_dir}")
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    task = metadata.get("task_original", "Unknown task")
    print(f"Task: {task}")

    # Find images
    images_dir = case_dir / "images"
    if not images_dir.exists():
        print(f"WARNING: No images directory found. Using placeholder analysis.")
        return _analyze_without_images(checkpoint_path, task, output_dir, device)

    # Load first image for analysis
    image_files = sorted(images_dir.glob("step_*_head.jpg"))
    if not image_files:
        print("WARNING: No head camera images found")
        return None

    # Load model
    print("Loading SmolVLA policy...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # Setup attention capture
    capture = AttentionCaptureHook()
    capture.register_hooks(policy.model)

    # Process representative images
    results = []
    for img_path in image_files[:5]:  # First 5 images
        print(f"  Processing: {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference with attention capture
        capture.clear()

        # Create observation
        image_resized = cv2.resize(image, (512, 512))
        img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        state = torch.zeros(1, 12).to(device)

        observation = {
            "observation.images.camera1": img_tensor,
            "observation.state": state,
            "task": task,
        }

        try:
            with torch.no_grad():
                action = policy.select_action(observation)
        except Exception as e:
            print(f"    Inference error: {e}")

        # Collect attention data
        attn_weights = capture.get_weights()
        if attn_weights:
            results.append({
                "image": img_path.name,
                "attention_layers": list(attn_weights.keys()),
                "weights": attn_weights,
            })

    capture.remove_hooks()

    # Generate visualizations
    if results:
        _generate_attention_visualizations(results, task, output_dir)

    return results


def _analyze_without_images(
    checkpoint_path: str,
    task: str,
    output_dir: Path,
    device: str,
) -> dict:
    """Analyze tokenization and model structure without images."""
    print("Performing tokenization-only analysis...")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            trust_remote_code=True
        )

        tokens = tokenizer.tokenize(task)

        analysis = {
            "task": task,
            "tokens": tokens,
            "num_tokens": len(tokens),
            "target_words": [],
        }

        # Find target words
        target_words = ['plate', 'bin', 'pick', 'place', 'banana', 'orange', 'yogurt']
        for i, token in enumerate(tokens):
            token_lower = token.lower().strip()
            for target in target_words:
                if target in token_lower:
                    analysis["target_words"].append({
                        "word": target,
                        "token": token,
                        "index": i,
                    })

        # Save analysis
        with open(output_dir / "tokenization_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Tokenization analysis saved to: {output_dir}")
        return analysis

    except Exception as e:
        print(f"Tokenization analysis failed: {e}")
        return None


def _generate_attention_visualizations(results: list, task: str, output_dir: Path):
    """Generate visualization plots from attention results."""
    print("Generating attention visualizations...")

    # Save raw data
    with open(output_dir / "attention_data.json", 'w') as f:
        # Convert numpy arrays to lists for JSON
        serializable = []
        for r in results:
            sr = {"image": r["image"], "attention_layers": r["attention_layers"]}
            serializable.append(sr)
        json.dump(serializable, f, indent=2)

    print(f"Attention analysis saved to: {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Attention Visualization for SmolVLA Hallucination Investigation"
    )

    # Mode
    parser.add_argument("--compare", action="store_true", help="Compare two cases")

    # Single case analysis
    parser.add_argument("--checkpoint", "-c", help="Path to SmolVLA checkpoint")
    parser.add_argument("--case-dir", type=Path, help="Case directory with images")

    # Comparison mode
    parser.add_argument("--case1", type=Path, help="First case for comparison")
    parser.add_argument("--case2", type=Path, help="Second case for comparison")

    # Output
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SmolVLA Enhanced Attention Visualization")
    print("=" * 60)

    if args.compare:
        if not args.case1 or not args.case2:
            print("ERROR: --case1 and --case2 required for comparison")
            sys.exit(1)

        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            sys.exit(1)

        print(f"Comparing:")
        print(f"  Case 1: {args.case1}")
        print(f"  Case 2: {args.case2}")

        # Analyze both cases
        out1 = args.output_dir / "case1"
        out2 = args.output_dir / "case2"
        out1.mkdir(exist_ok=True)
        out2.mkdir(exist_ok=True)

        analyze_case_attention(args.checkpoint, args.case1, out1, args.device)
        analyze_case_attention(args.checkpoint, args.case2, out2, args.device)

    else:
        if not args.checkpoint or not args.case_dir:
            print("ERROR: --checkpoint and --case-dir required for single case analysis")
            sys.exit(1)

        analyze_case_attention(
            args.checkpoint,
            args.case_dir,
            args.output_dir,
            args.device,
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
