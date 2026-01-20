# SmolVLA Hallucination Mechanism Findings

**Status**: Template - Awaiting Experiment Results
**Last Updated**: 2026-01-20

## Executive Summary

*To be populated after running Phase A and Phase B experiments*

This document will contain the findings from tracing the complete pipeline to identify WHERE and WHY divergence occurs between hallucination and normal cases.

---

## Part 1: Visual Feature Analysis (Exp 1.1)

### Experiment Setup
- Tool: `vision_feature_comparison.py`
- Checkpoint: `outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model`
- Step analyzed: 200 (post-task completion)
- Cases compared:
  - Hallucination: `case_20260119_131914_ha_bana_table`
  - Normal (banana on plate): `case_20260119_132946_no_ha_plate`
  - Normal (no distractors): `case_20260119_133142_no_ha_no_other_obj`

### Results

*To be populated after running experiment*

| Camera | Cosine Similarity (H vs N) | L2 Distance | Key Difference |
|--------|---------------------------|-------------|----------------|
| Head | | | |
| Left Wrist | | | |
| Right Wrist | | | |

### Key Questions Answered
- [ ] Does the head camera clearly show "no target object" in both cases?
- [ ] Does banana presence change right wrist camera features significantly?
- [ ] Which camera has the largest feature difference between cases?

---

## Part 2: Prefix Embedding Analysis (Exp 1.2)

### Experiment Setup
- Tool: `prefix_embedding_analysis.py`
- Same checkpoint and cases as Exp 1.1

### Results

*To be populated after running experiment*

**Per-Token Region Analysis:**

| Region | Token Range | Mean L2 Distance | % of Total Difference |
|--------|-------------|------------------|----------------------|
| Head Camera | [0-63] | | |
| Left Wrist | [64-127] | | |
| Right Wrist | [128-191] | | |
| Language | [192-239] | | |
| State | [240] | | |

### Key Questions Answered
- [ ] Do vision tokens dominate the difference vs language/state?
- [ ] Within vision, which camera contributes most?
- [ ] Is the difference localized or distributed?

---

## Part 3: KV Cache Content Analysis (Exp 1.3)

### Experiment Setup
- Tool: `kv_cache_content_analysis.py`
- Same checkpoint and cases

### Results

*To be populated after running experiment*

**Per-Layer KV Difference:**

| Layer | Key L2 | Value L2 | Most Different Region |
|-------|--------|----------|----------------------|
| 0 | | | |
| ... | | | |
| 15 | | | |

### Key Questions Answered
- [ ] Is KV difference concentrated in specific layers?
- [ ] Which token region has largest KV difference?
- [ ] Does early vs late layers show more divergence?

---

## Part 4: Denoising Step Analysis (Exp 1.4)

### Experiment Setup
- Tool: `denoising_step_analysis.py`
- Same checkpoint and cases

### Results

*To be populated after running experiment*

**Trajectory Shape Emergence:**

| Case | Initial Shape (Step 0) | Final Shape (Step 9) | Emergence Step |
|------|----------------------|---------------------|----------------|
| Hallucination | | | |
| Normal | | | |

**Divergence Analysis:**

- Shape diverges at step: *TBD*
- Velocity diverges at step: *TBD*
- Divergence driven by: *TBD (initial_noise / early_velocity / late_denoising)*

### Key Questions Answered
- [ ] Does RAMP emerge at step 0 (KV cache driven) or gradually?
- [ ] At which denoising step does halluc case "lock in" to wrong trajectory?
- [ ] Does the velocity field point different directions from step 0?

---

## Causal Chain

Based on the experiment results, the causal chain from visual input to action output:

```
VISUAL INPUT DIFFERENCE
(which camera? what features?)
          │
          ▼
PREFIX EMBEDDING DIFFERENCE
(which region? magnitude?)
          │
          ▼
KV CACHE DIFFERENCE
(which layers? which tokens?)
          │
          ▼
DENOISING VELOCITY FIELD
(when does divergence start?)
          │
          ▼
ACTION OUTPUT DIFFERENCE
(RAMP vs FLAT trajectory)
```

**Root Cause Identification:**

*To be determined after experiments*

---

## Visualizations

After running experiments, the following visualizations will be generated:

- [ ] `pca_visualization.png` - PCA of visual features
- [ ] `spatial_difference_heatmaps/` - Per-camera spatial diffs
- [ ] `token_distances_*.png` - Per-token L2 heatmaps
- [ ] `region_by_layer_heatmap_*.png` - KV cache [layer x region] heatmap
- [ ] `trajectory_emergence_*.png` - How trajectory shapes emerge
- [ ] `denoising_comparison_*.png` - Side-by-side denoising comparison

---

## Next Steps

After populating this document with results:
1. Synthesize findings in `hallucination_root_cause_synthesis.md`
2. Link to dataset distribution analysis
3. Generate recommendations for mitigation
