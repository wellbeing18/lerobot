# SmolVLA Hallucination Investigation Tools

## Quick Start

All tools have sensible defaults - just run them without arguments:

```bash
cd jdocs/scripts/investigation/tools

# Run any tool with defaults (checkpoint, case-dirs, output-dir all pre-configured)
python vision_feature_comparison.py
python prefix_embedding_analysis.py
python kv_cache_content_analysis.py
python denoising_step_analysis.py
python trajectory_distribution_visualization.py

# Advanced Investigation Methods (Tier 1 - High Impact)
python attention_knockout.py           # Causal importance of token regions
python umap_trajectory_analysis.py     # Coverage gap identification
python distractor_sensitivity_probe.py # Test if distractor causes hallucination

# Advanced Investigation Methods (Tier 2 - High Value)
python causal_mediation_analysis.py    # Isolate visual vs state effects
python uncertainty_metrics.py          # Quantify model confidence
python action_coverage_analysis.py     # Per-joint coverage statistics
python phase_transition_analysis.py    # Missing transitions in training
```

Output directories are auto-generated with timestamps, e.g.:
- `logs/investigation/vision_features_20260121_143052/`
- `logs/investigation/kv_cache_20260121_143105/`

## Configuration

All defaults are defined in `investigation_config.py`:

| Setting | Default Value |
|---------|---------------|
| Checkpoint | `outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model` |
| Case Dirs | halluc + normal_clean (2 cases) |
| Output Base | `logs/investigation/` |
| Dataset | `datasets_bimanuel/multitasks` |
| Task Filter | `yogurt` |
| Default Step | 200 (post-task completion) |

```bash
# Check configuration
python investigation_config.py
```

### Available Cases

| Case | Directory | Description |
|------|-----------|-------------|
| `halluc` | `case_20260119_131914_ha_bana_table` | **Hallucination** - banana on table |
| `normal_plate` | `case_20260119_132946_no_ha_plate` | Normal - banana on plate (far) |
| `normal_clean` | `case_20260119_133142_no_ha_no_other_obj` | Normal - clean workspace |

---

## Running the Tools

### Phase A: Offline Analysis

#### 1. Vision Feature Comparison

```bash
# With all defaults (recommended)
python vision_feature_comparison.py

# Override specific options if needed
python vision_feature_comparison.py --step 250
python vision_feature_comparison.py --output-dir /custom/path
```

**Outputs:** `comparisons.json`, `pca_visualization.png`, `comparison_matrix.png`, `spatial_diff_*.png`, `report.md`

#### 2. Prefix Embedding Analysis

```bash
python prefix_embedding_analysis.py
```

**Outputs:** `comparisons.json`, `token_distances_*.png`, `region_comparison.png`, `pca_prefix_embeddings.png`, `report.md`

---

### Phase B: Model Hook Integration (Requires GPU)

#### 3. KV Cache Content Analysis

```bash
python kv_cache_content_analysis.py
```

**Outputs:** `comparisons.json`, `layer_comparison_*.png`, `region_by_layer_heatmap_*.png`, `region_totals.png`, `report.md`

#### 4. Denoising Step Analysis

```bash
python denoising_step_analysis.py
```

**Outputs:** `analyses.json`, `comparisons.json`, `trajectory_emergence_*.png`, `denoising_comparison_*.png`, `joint_evolution_grid.png`, `report.md`

---

### Phase C: Dataset Analysis

#### 5. Trajectory Distribution Visualization

```bash
python trajectory_distribution_visualization.py
```

**Outputs:** `analysis.json`, `3d_trajectory_distribution.png`, `3d_trajectory_distribution.html`, `2d_projections.png`, `phase_distribution.png`, `report.md`

---

## Run All Experiments

```bash
cd jdocs/scripts/investigation/tools

# Just run all tools - everything is pre-configured!
python vision_feature_comparison.py
python prefix_embedding_analysis.py
python kv_cache_content_analysis.py
python denoising_step_analysis.py
python trajectory_distribution_visualization.py
```

---

## Customizing Runs

### Compare all 3 cases (not just halluc vs clean)

```bash
python vision_feature_comparison.py \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj
```

### Analyze different step

```bash
python vision_feature_comparison.py --step 150
```

### Custom output directory

```bash
python vision_feature_comparison.py --output-dir logs/my_custom_experiment
```

---

---

## Advanced Investigation Methods

### Tier 1: High Impact (Causal Analysis)

#### 6. Attention Knockout

Tests causal importance of attention to specific token regions by zeroing out KV cache.

```bash
python attention_knockout.py
```

**Outputs:** `analyses.json`, `knockout_effects_*.png`, `trajectory_*.png`, `knockout_comparison_across_cases.png`, `report.md`

**Key Question:** If we knock out right wrist camera attention, does hallucination disappear?

#### 7. UMAP Trajectory Analysis

Uses UMAP to visualize trajectory manifold and identify coverage gaps.

```bash
python umap_trajectory_analysis.py
```

**Outputs:** `analysis.json`, `umap_manifold.png`, `inference_detail.png`, `training_embedding.npy`, `report.md`

**Key Question:** Does hallucination trajectory fall in a coverage gap (void in training distribution)?

#### 8. Distractor Sensitivity Probe

Tests if distractor (banana) causally affects behavior by masking it.

```bash
python distractor_sensitivity_probe.py
```

**Outputs:** `analysis.json`, `masking_visualization.png`, `sensitivity_results_*.png`, `report.md`

**Key Question:** If we mask/inpaint the banana, does hallucination disappear?

---

### Tier 2: High Value (Dataset & Mechanism Analysis)

#### 9. Causal Mediation Analysis

Isolates effects of visual input vs state vs language using counterfactuals.

```bash
python causal_mediation_analysis.py
```

**Outputs:** `analysis.json`, `causal_mediation_results.png`, `trajectory_comparison.png`, `report.md`

**Key Question:** Is visual input alone sufficient to cause hallucination?

#### 10. Uncertainty Metrics

Quantifies model confidence via multiple inference runs.

```bash
python uncertainty_metrics.py
```

**Outputs:** `metrics.json`, `comparison.json`, `uncertainty_comparison.png`, `api_per_joint_*.png`, `report.md`

**Key Question:** Is the model more uncertain during hallucination?

#### 11. Action Coverage Analysis

Analyzes per-joint action distributions in training data.

```bash
python action_coverage_analysis.py
```

**Outputs:** `analysis.json`, `coverage_distribution.png`, `inference_vs_training.png`, `report.md`

**Key Question:** Which joints have poor training coverage?

#### 12. Phase Transition Analysis

Identifies missing/rare phase transitions in training data.

```bash
python phase_transition_analysis.py
```

**Outputs:** `analysis.json`, `transition_matrix.png`, `phase_distribution.png`, `transition_flow.png`, `report.md`

**Key Question:** Is TRANSPORT → IDLE transition missing (explaining why model continues moving)?

---

## Run All Advanced Experiments

```bash
cd jdocs/scripts/investigation/tools

# Tier 1: Causal Analysis
python attention_knockout.py
python distractor_sensitivity_probe.py
python causal_mediation_analysis.py

# Tier 2: Dataset & Mechanism
python umap_trajectory_analysis.py
python uncertainty_metrics.py
python action_coverage_analysis.py
python phase_transition_analysis.py
```

---

## Tool Dependencies

The advanced tools require additional packages:

```bash
pip install umap-learn scipy
```

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
