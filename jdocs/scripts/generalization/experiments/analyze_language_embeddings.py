#!/usr/bin/env python3
"""
Experiment 1: Language Embedding Separability Analysis

Analyzes whether SmolVLA's language embeddings can distinguish between:
1. Same object, different targets (e.g., "icecream → plate" vs "icecream → bin")
2. Different objects, same structure (e.g., "tissue" vs "corn")

Hypothesis: If embeddings are too similar (cosine similarity > 0.95),
language is under-conditioned and cannot override visual-trajectory priors.

Usage:
    python analyze_language_embeddings.py
    python analyze_language_embeddings.py --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    python analyze_language_embeddings.py --output-dir ../outputs/embedding_analysis

Output:
    - Cosine similarity heatmap (PNG)
    - Detailed similarity report (JSON)
    - Analysis summary (Markdown)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

# ============================================================================
# TASK DESCRIPTIONS TO ANALYZE
# ============================================================================

# Training tasks (from bimanual training)
TRAINING_TASKS = {
    # Plate tasks
    "left_orange_plate": "Use left arm to pick up the orange and place it on the plate",
    "right_orange_plate": "Use right arm to pick up the orange and place it on the plate",
    "left_corn_plate": "Use left arm to pick up the corn and place it on the plate",
    "right_corn_plate": "Use right arm to pick up the corn and place it on the plate",
    # Bin tasks
    "left_icecream_bin": "Use left arm to pick up the ice cream and place it in the bin",
    "right_icecream_bin": "Use right arm to pick up the ice cream and place it in the bin",
    "left_tissue_bin": "Use left arm to pick up the used tissue and place it in the bin",
    "right_tissue_bin": "Use right arm to pick up the used tissue and place it in the bin",
}

# Novel task combinations (not in training - test generalization)
NOVEL_TASKS = {
    # Cross-target (trained object, different target)
    "right_icecream_plate": "Use right arm to pick up the ice cream and place it on the plate",
    "left_orange_bin": "Use left arm to pick up the orange and place it in the bin",
    # Novel object (not in training)
    "right_tissue_plate": "Use right arm to pick up the tissue package and place it on the plate",
    "left_tissue_plate": "Use left arm to pick up the tissue package and place it on the plate",
}

# Critical comparison pairs for analysis
CRITICAL_PAIRS = [
    # Same object, different target (should be DIFFERENT embeddings)
    ("right_icecream_bin", "right_icecream_plate", "Same object, different target"),
    ("left_orange_plate", "left_orange_bin", "Same object, different target"),

    # Different objects, same structure (should be DIFFERENT embeddings)
    ("right_corn_plate", "right_tissue_plate", "Different objects, same target"),
    ("left_icecream_bin", "left_tissue_bin", "Different objects, same target"),

    # Same task, different arm (should be SIMILAR but distinguishable)
    ("left_orange_plate", "right_orange_plate", "Same task, different arm"),
    ("left_icecream_bin", "right_icecream_bin", "Same task, different arm"),
]


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def load_vlm_model(model_name: str, device: str = "cuda"):
    """Load SmolVLM model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_embeddings(
    model,
    tokenizer,
    texts: dict[str, str],
    device: str = "cuda",
    pooling: str = "mean"
) -> dict[str, np.ndarray]:
    """
    Extract embeddings for task descriptions.

    Args:
        model: VLM model
        tokenizer: VLM tokenizer
        texts: Dict of {task_id: task_description}
        device: Compute device
        pooling: Pooling strategy ("mean", "cls", "last")

    Returns:
        Dict of {task_id: embedding_vector}
    """
    embeddings = {}

    with torch.no_grad():
        for task_id, text in texts.items():
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            # Get token embeddings from embedding layer
            token_embeddings = model.get_input_embeddings()(inputs["input_ids"])

            # Pool to single vector
            if pooling == "mean":
                # Mean over sequence length (excluding padding)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked_emb = token_embeddings * attention_mask
                emb = masked_emb.sum(dim=1) / attention_mask.sum(dim=1)
            elif pooling == "cls":
                emb = token_embeddings[:, 0, :]
            elif pooling == "last":
                # Last non-padding token
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                emb = token_embeddings[0, seq_lens[0], :].unsqueeze(0)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embeddings[task_id] = emb.cpu().numpy().flatten()

    return embeddings


def compute_similarity_matrix(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity matrix."""
    task_ids = list(embeddings.keys())
    n = len(task_ids)

    # Stack embeddings
    emb_matrix = np.stack([embeddings[tid] for tid in task_ids])

    # Normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix_norm = emb_matrix / (norms + 1e-8)

    # Cosine similarity
    sim_matrix = emb_matrix_norm @ emb_matrix_norm.T

    return sim_matrix, task_ids


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_critical_pairs(
    sim_matrix: np.ndarray,
    task_ids: list[str],
    pairs: list[tuple[str, str, str]]
) -> list[dict]:
    """Analyze similarity for critical task pairs."""
    results = []

    for task1, task2, description in pairs:
        if task1 not in task_ids or task2 not in task_ids:
            print(f"Warning: Skipping pair ({task1}, {task2}) - not in embeddings")
            continue

        idx1 = task_ids.index(task1)
        idx2 = task_ids.index(task2)
        similarity = sim_matrix[idx1, idx2]

        # Determine if similarity is problematic
        is_problematic = False
        if "different" in description.lower() and similarity > 0.95:
            is_problematic = True  # Should be different but too similar

        results.append({
            "task1": task1,
            "task2": task2,
            "description": description,
            "similarity": float(similarity),
            "is_problematic": is_problematic
        })

    return results


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    task_ids: list[str],
    output_path: Path,
    title: str = "Task Description Embedding Similarity"
):
    """Create and save similarity heatmap."""
    plt.figure(figsize=(14, 12))

    # Create shorter labels for display
    short_labels = []
    for tid in task_ids:
        parts = tid.split("_")
        if len(parts) >= 3:
            short_labels.append(f"{parts[0][:1]}_{parts[1][:4]}_{parts[2][:3]}")
        else:
            short_labels.append(tid[:12])

    # Plot heatmap
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Red = high similarity (potentially bad for different tasks)
        xticklabels=short_labels,
        yticklabels=short_labels,
        vmin=0.8,
        vmax=1.0,
        center=0.95,
        square=True,
        cbar_kws={"label": "Cosine Similarity"}
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Task Description", fontsize=12)
    plt.ylabel("Task Description", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def generate_report(
    pair_analysis: list[dict],
    sim_matrix: np.ndarray,
    task_ids: list[str],
    output_dir: Path,
    model_name: str
):
    """Generate analysis report in Markdown and JSON."""

    # Calculate summary stats
    n_problematic = sum(1 for p in pair_analysis if p["is_problematic"])
    avg_similarity = float(np.mean(sim_matrix[np.triu_indices(len(task_ids), k=1)]))

    # JSON report
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "summary": {
            "total_tasks": len(task_ids),
            "critical_pairs_analyzed": len(pair_analysis),
            "problematic_pairs": n_problematic,
            "average_similarity": avg_similarity
        },
        "pair_analysis": pair_analysis,
        "task_ids": task_ids
    }

    json_path = output_dir / "embedding_analysis.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"Saved JSON report to: {json_path}")

    # Markdown report
    md_lines = [
        "# Language Embedding Separability Analysis",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: `{model_name}`",
        "",
        "## Summary",
        "",
        f"- **Total tasks analyzed**: {len(task_ids)}",
        f"- **Average pairwise similarity**: {avg_similarity:.4f}",
        f"- **Problematic pairs**: {n_problematic}/{len(pair_analysis)}",
        "",
        "## Critical Pair Analysis",
        "",
        "| Task 1 | Task 2 | Type | Similarity | Status |",
        "|--------|--------|------|------------|--------|",
    ]

    for p in pair_analysis:
        status = "WARNING" if p["is_problematic"] else "OK"
        md_lines.append(
            f"| {p['task1']} | {p['task2']} | {p['description']} | "
            f"{p['similarity']:.4f} | {status} |"
        )

    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "**Problematic pairs** (marked WARNING) indicate:",
        "- Tasks that SHOULD produce different actions",
        "- But have embeddings with similarity > 0.95",
        "- This suggests language cannot distinguish these task variants",
        "",
        "## Recommendations",
        "",
    ])

    if n_problematic > 0:
        md_lines.extend([
            "**Root cause confirmed**: Language embeddings are under-conditioned.",
            "",
            "Recommended actions:",
            "1. **Unfreeze language encoder** during training (`train_expert_only=false`)",
            "2. **Try primitive decomposition** to make targets explicit",
            "3. Consider **task-type soft prompts** to boost language signal",
        ])
    else:
        md_lines.extend([
            "**Language embeddings appear distinguishable.**",
            "",
            "If generalization still fails, investigate:",
            "1. Attention patterns (do they focus on target words?)",
            "2. Visual-trajectory coupling in the action expert",
        ])

    md_path = output_dir / "embedding_analysis.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown report to: {md_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze language embedding separability")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="VLM model to analyze"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device"
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "cls", "last"],
        help="Pooling strategy for embeddings"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "embedding_analysis",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 1: Language Embedding Separability Analysis")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()

    # Load model
    model, tokenizer = load_vlm_model(args.model, args.device)

    # Combine all tasks
    all_tasks = {**TRAINING_TASKS, **NOVEL_TASKS}
    print(f"Analyzing {len(all_tasks)} task descriptions...")

    # Extract embeddings
    embeddings = extract_embeddings(model, tokenizer, all_tasks, args.device, args.pooling)

    # Compute similarity matrix
    sim_matrix, task_ids = compute_similarity_matrix(embeddings)

    # Analyze critical pairs
    print("\n--- Critical Pair Analysis ---")
    pair_analysis = analyze_critical_pairs(sim_matrix, task_ids, CRITICAL_PAIRS)

    for p in pair_analysis:
        status = "WARNING" if p["is_problematic"] else "OK"
        print(f"  {p['description']}")
        print(f"    {p['task1']} vs {p['task2']}")
        print(f"    Similarity: {p['similarity']:.4f} [{status}]")
        print()

    # Generate visualizations and reports
    plot_similarity_heatmap(
        sim_matrix,
        task_ids,
        args.output_dir / "embedding_similarity_heatmap.png",
        f"Task Embedding Similarity ({args.model.split('/')[-1]})"
    )

    generate_report(pair_analysis, sim_matrix, task_ids, args.output_dir, args.model)

    # Summary
    n_problematic = sum(1 for p in pair_analysis if p["is_problematic"])
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Problematic pairs: {n_problematic}/{len(pair_analysis)}")

    if n_problematic > 0:
        print("\nCONCLUSION: Language embeddings are under-conditioned.")
        print("Root cause CONFIRMED: Different targets produce similar embeddings.")
        print("\nRecommendation: Proceed with unfreezing language encoder (Experiment 2)")
    else:
        print("\nCONCLUSION: Language embeddings appear distinguishable.")
        print("Investigate attention patterns (Experiment 3) for further diagnosis.")

    print(f"\nFull results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
