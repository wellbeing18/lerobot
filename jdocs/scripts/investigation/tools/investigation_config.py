#!/usr/bin/env python3
"""
Configuration file for SmolVLA Hallucination Investigation Tools.

This file centralizes all paths and settings used across investigation tools.
Import this config in any tool to use consistent paths.

Usage:
    from investigation_config import CONFIG, get_case_dirs, get_all_case_dirs
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root (4 levels up from this file: tools -> investigation -> scripts -> jdocs -> project)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

CHECKPOINT_PATH = PROJECT_ROOT / "outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model"

# ============================================================================
# CASE DIRECTORIES
# ============================================================================

CASE_BASE_DIR = PROJECT_ROOT / "logs/yogurt_banana_leftarm"

# Individual case directories
CASE_HALLUC = CASE_BASE_DIR / "case_20260119_131914_ha_bana_table"        # Hallucination: banana on table
CASE_NORMAL_PLATE = CASE_BASE_DIR / "case_20260119_132946_no_ha_plate"    # Normal: banana on plate (far)
CASE_NORMAL_CLEAN = CASE_BASE_DIR / "case_20260119_133142_no_ha_no_other_obj"  # Normal: no distractors

# Case descriptions for reference
CASE_DESCRIPTIONS = {
    "halluc": {
        "path": CASE_HALLUC,
        "short_name": "ha_bana_table",
        "description": "Hallucination case - banana on table near workspace",
        "has_hallucination": True,
        "has_distractor": True,
    },
    "normal_plate": {
        "path": CASE_NORMAL_PLATE,
        "short_name": "no_ha_plate",
        "description": "Normal case - banana on plate (far from workspace)",
        "has_hallucination": False,
        "has_distractor": True,
    },
    "normal_clean": {
        "path": CASE_NORMAL_CLEAN,
        "short_name": "no_ha_no_other_obj",
        "description": "Normal case - clean workspace, no distractors",
        "has_hallucination": False,
        "has_distractor": False,
    },
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_BASE_DIR = PROJECT_ROOT / "logs/investigation"

OUTPUT_DIRS = {
    "vision_features": OUTPUT_BASE_DIR / "vision_feature_comparison",
    "prefix_embedding": OUTPUT_BASE_DIR / "prefix_embedding_analysis",
    "kv_cache": OUTPUT_BASE_DIR / "kv_cache_analysis",
    "denoising": OUTPUT_BASE_DIR / "denoising_analysis",
    "trajectory_dist": OUTPUT_BASE_DIR / "trajectory_distribution",
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_PATH = PROJECT_ROOT / "datasets_bimanuel/multitasks"
TASK_FILTER = "yogurt"  # Filter for yogurt bottle tasks

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

DEFAULT_STEP = 200  # Post-task completion step to analyze
STEP_RANGE = (200, 300)  # Step range for trajectory analysis
DEVICE = "cuda"

# ============================================================================
# CONFIG DICTIONARY (for easy access)
# ============================================================================

CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "checkpoint": str(CHECKPOINT_PATH),
    "case_base_dir": str(CASE_BASE_DIR),
    "cases": {
        "halluc": str(CASE_HALLUC),
        "normal_plate": str(CASE_NORMAL_PLATE),
        "normal_clean": str(CASE_NORMAL_CLEAN),
    },
    "output_base": str(OUTPUT_BASE_DIR),
    "outputs": {k: str(v) for k, v in OUTPUT_DIRS.items()},
    "dataset": str(DATASET_PATH),
    "task_filter": TASK_FILTER,
    "default_step": DEFAULT_STEP,
    "step_range": STEP_RANGE,
    "device": DEVICE,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_case_dirs(include_halluc=True, include_normal_plate=False, include_normal_clean=True):
    """Get list of case directories based on selection.

    Default: halluc + normal_clean (most common comparison)

    Args:
        include_halluc: Include hallucination case
        include_normal_plate: Include normal case with banana on plate
        include_normal_clean: Include normal case with clean workspace

    Returns:
        List of case directory paths as strings
    """
    dirs = []
    if include_halluc:
        dirs.append(str(CASE_HALLUC))
    if include_normal_plate:
        dirs.append(str(CASE_NORMAL_PLATE))
    if include_normal_clean:
        dirs.append(str(CASE_NORMAL_CLEAN))
    return dirs


def get_all_case_dirs():
    """Get all 3 case directories."""
    return [str(CASE_HALLUC), str(CASE_NORMAL_PLATE), str(CASE_NORMAL_CLEAN)]


def get_halluc_vs_clean():
    """Get halluc and clean workspace cases (most direct comparison)."""
    return [str(CASE_HALLUC), str(CASE_NORMAL_CLEAN)]


def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("SmolVLA Hallucination Investigation Configuration")
    print("=" * 60)
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"\nCase directories:")
    for name, info in CASE_DESCRIPTIONS.items():
        status = "✓" if info["path"].exists() else "✗"
        print(f"  [{status}] {name}: {info['path'].name}")
        print(f"      {info['description']}")
    print(f"\nOutput base: {OUTPUT_BASE_DIR}")
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Task filter: {TASK_FILTER}")
    print(f"Default step: {DEFAULT_STEP}")
    print("=" * 60)


# ============================================================================
# MAIN (for testing config)
# ============================================================================

if __name__ == "__main__":
    print_config()

    print("\n\nExample usage in tools:")
    print("-" * 60)
    print("""
from investigation_config import (
    CONFIG,
    CHECKPOINT_PATH,
    get_halluc_vs_clean,
    OUTPUT_DIRS,
    DEFAULT_STEP
)

# Get paths
checkpoint = str(CHECKPOINT_PATH)
case_dirs = get_halluc_vs_clean()
output_dir = str(OUTPUT_DIRS["vision_features"])

# Use in argparse defaults
parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
parser.add_argument("--case-dirs", nargs="+", default=get_halluc_vs_clean())
""")
