# SmolVLA Hallucination Investigation Tools

## Quick Start

```bash
cd jdocs/scripts/investigation/tools

# Check configuration
python investigation_config.py
```

## Configuration

All paths are defined in `investigation_config.py`:

| Setting | Value |
|---------|-------|
| Checkpoint | `outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model` |
| Case Base | `logs/yogurt_banana_leftarm/` |
| Output Base | `logs/investigation/` |
| Dataset | `datasets_bimanuel/multitasks` |
| Default Step | 200 (post-task completion) |

### Available Cases

| Case | Directory | Description |
|------|-----------|-------------|
| `halluc` | `case_20260119_131914_ha_bana_table` | **Hallucination** - banana on table near workspace |
| `normal_plate` | `case_20260119_132946_no_ha_plate` | Normal - banana on plate (far) |
| `normal_clean` | `case_20260119_133142_no_ha_no_other_obj` | Normal - clean workspace |

---

## Running the Tools

### Phase A: Offline Analysis (No live robot needed)

#### 1. Vision Feature Comparison

Compares SigLIP visual features between cases.

```bash
# Compare halluc vs clean (recommended for clearest comparison)
python vision_feature_comparison.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/vision_feature_comparison

# Compare all 3 cases
python vision_feature_comparison.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/vision_feature_comparison
```

**Outputs:**
- `comparisons.json` - Raw comparison data
- `pca_visualization.png` - PCA of patch embeddings
- `comparison_matrix.png` - Similarity matrix
- `spatial_diff_*.png` - Per-camera difference heatmaps
- `report.md` - Summary report

#### 2. Prefix Embedding Analysis

Analyzes the combined prefix embedding (vision + language + state).

```bash
# Compare halluc vs clean
python prefix_embedding_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/prefix_embedding_analysis
```

**Outputs:**
- `comparisons.json` - Per-token L2 distances
- `token_distances_*.png` - Token distance heatmaps
- `region_comparison.png` - Region-level comparison
- `pca_prefix_embeddings.png` - PCA visualization
- `report.md` - Summary report

---

### Phase B: Model Hook Integration (Requires GPU)

#### 3. KV Cache Content Analysis

Analyzes KV cache differences by layer and token region.

```bash
python kv_cache_content_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/kv_cache_analysis
```

**Outputs:**
- `comparisons.json` - Per-layer KV statistics
- `layer_comparison_*.png` - Per-layer breakdown
- `region_by_layer_heatmap_*.png` - [Layer x Region] heatmap
- `region_totals.png` - Total by region
- `report.md` - Summary report

#### 4. Denoising Step Analysis

Traces trajectory emergence through 10 denoising steps.

```bash
python denoising_step_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/denoising_analysis
```

**Outputs:**
- `analyses.json` - Per-step metrics
- `comparisons.json` - Case comparison data
- `trajectory_emergence_*.png` - How shapes emerge
- `denoising_comparison_*.png` - Side-by-side comparison
- `joint_evolution_grid.png` - Joint values heatmap
- `report.md` - Summary report

---

### Phase C: Dataset Analysis

#### 5. Trajectory Distribution Visualization

Visualizes training trajectory space and where inference cases lie.

```bash
python trajectory_distribution_visualization.py \
    --dataset datasets_bimanuel/multitasks \
    --task-filter "yogurt" \
    --halluc-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
    --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
    --step-range "200,300" \
    --max-episodes 100 \
    --output-dir logs/investigation/trajectory_distribution
```

**Outputs:**
- `analysis.json` - Full analysis data
- `3d_trajectory_distribution.png` - 3D PCA (matplotlib)
- `3d_trajectory_distribution.html` - Interactive 3D (plotly)
- `2d_projections.png` - 2D views
- `phase_distribution.png` - Phase histogram
- `report.md` - Summary report

---

## Run All Experiments

```bash
cd jdocs/scripts/investigation/tools

# Create output directories
mkdir -p logs/investigation/{vision_feature_comparison,prefix_embedding_analysis,kv_cache_analysis,denoising_analysis,trajectory_distribution}

# Phase A
python vision_feature_comparison.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/vision_feature_comparison

python prefix_embedding_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/prefix_embedding_analysis

# Phase B
python kv_cache_content_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/kv_cache_analysis

python denoising_step_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir logs/investigation/denoising_analysis

# Phase C
python trajectory_distribution_visualization.py \
    --dataset datasets_bimanuel/multitasks \
    --task-filter "yogurt" \
    --halluc-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
    --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
    --step-range "200,300" \
    --output-dir logs/investigation/trajectory_distribution
```

---

## Which Cases to Compare?

### For clearest hallucination comparison (recommended):

Compare **halluc** vs **normal_clean** (2 folders):
- Hallucination case: banana on table (causes hallucination)
- Normal case: clean workspace (no distractors, stays still)

This gives the clearest contrast between hallucination and normal behavior.

### For understanding distractor effect:

Compare all 3 cases:
- Hallucination: banana on table (near workspace)
- Normal plate: banana on plate (far from workspace)
- Normal clean: no distractors

This helps understand if it's the banana proximity that matters.

---

## Token Layout Reference

```
[0-63]     Head camera patches (8x8 grid)
[64-127]   Left wrist camera patches (8x8 grid)
[128-191]  Right wrist camera patches (8x8 grid)
[192-239]  Language tokens (~48)
[240]      State token
─────────────────────────────────────────
TOTAL:     241 prefix tokens
```
