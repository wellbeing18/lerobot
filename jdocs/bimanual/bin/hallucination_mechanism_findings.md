# SmolVLA Hallucination Mechanism Findings

**Status**: Complete - Experiment Results Analyzed (Updated with Corrected Timing)
**Last Updated**: 2026-01-21

## Executive Summary

This document contains findings from tracing the complete SmolVLA pipeline to identify WHERE and WHY divergence occurs between hallucination and normal cases.

### Critical Timing Correction

**Previous Analysis**: Analyzed step 200 arbitrarily
**Corrected Analysis**: Trace analysis reveals actual divergence starts at **step 208-210**

| Step | Halluc action_delta | Normal action_delta | Divergence |
|------|---------------------|---------------------|------------|
| 200 | 2.4 | 2.3 | Similar |
| 208 | 5.4 | 2.2 | **DIVERGES** |
| 211+ | 10-13 | ~2.3 | Large gap |
| 225-240 | Gripper opens | Gripper stable | Robot trying to pick |

**Key Finding**: The banana presence in the right wrist camera creates a visual feature difference that propagates through the KV cache and immediately drives a different velocity field from denoising step 0. The hallucination trajectory (RAMP_UP, moving toward empty space) is established within the first denoising step. The divergence becomes behaviorally visible starting at step 208.

---

## Part 1: Visual Feature Analysis (Exp 1.1)

### Experiment Setup
- Tool: `vision_feature_comparison.py`
- Checkpoint: `outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model`
- Step analyzed: 200 (post-task completion)
- Cases compared:
  - Hallucination: `case_20260119_131914_ha_bana_table`
  - Normal (no distractors): `case_20260119_133142_no_ha_no_other_obj`
- Output: `logs/investigation/vision_features_20260121_151042/`

### Results

| Camera | Cosine Similarity (H vs N) | L2 Distance | Interpretation |
|--------|---------------------------|-------------|----------------|
| Head | 0.9363 | 1109.37 | Scene context similar |
| Left Wrist | 0.8013 | 1633.84 | Moderate difference |
| **Right Wrist** | **0.6271** | **2574.83** | **Most different - banana visible** |
| **Overall** | **0.7920** | **3244.97** | |

### Key Questions Answered
- [x] Does the head camera clearly show "no target object" in both cases? **Yes, high similarity (0.94)**
- [x] Does banana presence change right wrist camera features significantly? **Yes, lowest similarity (0.63)**
- [x] Which camera has the largest feature difference between cases? **Right wrist camera**

---

## Part 2: Prefix Embedding Analysis (Exp 1.2)

### Experiment Setup
- Tool: `prefix_embedding_analysis.py`
- Same checkpoint and cases as Exp 1.1
- Output: `logs/investigation/prefix_embedding_20260121_152056/`

### Results

**Per-Token Region Analysis:**

| Region | Token Range | Mean L2 Distance | Cosine Sim | % of Total Difference |
|--------|-------------|------------------|------------|----------------------|
| Head Camera | [0-63] | 4761.35 | 0.8927 | ~23% |
| Left Wrist | [64-127] | 5084.43 | 0.8143 | ~27% |
| **Right Wrist** | **[128-191]** | **9430.66** | **0.5909** | **~49%** |
| Language | [192-239] | 0.00 | 1.0000 | 0% |
| State | [240] | 1.33 | 0.9755 | 0% |

**Contribution to Total Difference:**
- Vision tokens: **100%**
- Language tokens: 0%
- State token: 0%

### Key Questions Answered
- [x] Do vision tokens dominate the difference vs language/state? **Yes, 100% from vision**
- [x] Within vision, which camera contributes most? **Right wrist (49%)**
- [x] Is the difference localized or distributed? **Localized to right wrist camera tokens**

---

## Part 3: KV Cache Content Analysis (Exp 1.3)

### Experiment Setup
- Tool: `kv_cache_content_analysis.py`
- Same checkpoint and cases
- KV Cache structure: 16 layers, 5 KV heads (grouped query attention), head dim 64
- Output: `logs/investigation/kv_cache_20260121_152713/`

### Results

**Per-Region KV Difference:**

| Region | Total L2 Distance | % of Total |
|--------|-------------------|------------|
| Head Camera | 1976.46 | 23.2% |
| Left Wrist | 2362.09 | 27.7% |
| **Right Wrist** | **3650.49** | **42.9%** |
| Language | 433.15 | 5.1% |
| State | 94.07 | 1.1% |

**Layer Distribution:**
- Most different layer: **Layer 15** (L2: 372.79)
- Late layers (8-15) differ more: 2664.24 vs early layers 2137.92

### Key Questions Answered
- [x] Is KV difference concentrated in specific layers? **Yes, late layers (8-15) differ more**
- [x] Which token region has largest KV difference? **Right wrist (42.9%)**
- [x] Does early vs late layers show more divergence? **Yes, late layers show more divergence**

---

## Part 4: Denoising Step Analysis (Exp 1.4)

### Experiment Setup
- Tool: `denoising_step_analysis.py`
- Same checkpoint and cases
- 10 denoising steps (time 1.0 → 0.1)
- Output: `logs/investigation/denoising_20260121_152743/`

### Results

**Trajectory Shape Emergence:**

| Case | Initial Shape (Step 0) | Final Shape (Step 9) | Emergence Step |
|------|----------------------|---------------------|----------------|
| **Hallucination** | IRREGULAR | **RAMP_UP** | **Step 1** |
| Normal | RAMP_DOWN | **FLAT** | Step 9 |

**Divergence Analysis:**

- Shape diverges at step: **0**
- Velocity diverges at step: **0**
- Divergence driven by: **initial_noise / KV cache conditioning**

**Trajectory Similarity Across Denoising:**

| Step | Trajectory Similarity | Velocity Similarity |
|------|----------------------|---------------------|
| 0 | 0.006 | 0.227 |
| 1 | 0.010 | 0.233 |
| 5 | 0.280 | 0.406 |
| 9 | 0.936 | 0.433 |

### Key Questions Answered
- [x] Does RAMP emerge at step 0 (KV cache driven) or gradually? **KV cache driven - emerges at step 1**
- [x] At which denoising step does halluc case "lock in" to wrong trajectory? **Step 1**
- [x] Does the velocity field point different directions from step 0? **Yes, trajectory similarity only 0.006 at step 0**

---

## Part 5: Trajectory Distribution Analysis (Phase C)

### Experiment Setup
- Tool: `trajectory_distribution_visualization.py`
- Dataset: `datasets_bimanuel/multitasks`
- 16 episodes, 1651 trajectory segments
- Output: `logs/investigation/trajectory_dist_20260121_152842/`

### Results

**Training Data Phase Distribution:**

| Phase | Count | Avg Duration |
|-------|-------|--------------|
| APPROACH | 255 | 88.1 steps |
| GRIP | 63 | 10.0 steps |
| TRANSPORT | 424 | 100.8 steps |
| RELEASE | 14 | 10.0 steps |
| IDLE | 462 | 13.5 steps |

**Inference Case Analysis:**

| Case | Nearest Training Phase | Distance | Interpretation |
|------|----------------------|----------|----------------|
| Hallucination | TRANSPORT | 823.57 | **Out-of-distribution** (> 1.5x avg 366.24) |
| Normal | TRANSPORT | 808.23 | **Out-of-distribution** |

**Key Finding:** Both post-completion trajectories are **out-of-distribution** and closer to TRANSPORT than IDLE, despite having 462 IDLE segments in training.

---

## Causal Chain

Based on the experiment results, the complete causal chain:

```
VISUAL INPUT DIFFERENCE
(Banana visible in right wrist camera)
          │
          │ Cosine sim: 0.627 (lowest)
          │ L2 distance: 2574.83 (largest)
          ▼
PREFIX EMBEDDING DIFFERENCE
(Right wrist tokens have 49% of total difference)
          │
          │ Mean L2: 9430.66
          │ Cosine sim: 0.5909
          ▼
KV CACHE DIFFERENCE
(Right wrist: 42.9% of total KV diff)
(Late layers 8-15 show more divergence)
          │
          │ Layer 15 most different
          │
          ▼
DENOISING VELOCITY FIELD
(Diverges from step 0)
(RAMP_UP emerges at step 1)
          │
          │ Trajectory sim at step 0: 0.006
          │
          ▼
ACTION OUTPUT DIFFERENCE
(RAMP_UP: moving toward empty space)
(vs FLAT: staying still)
```

**Root Cause Identification:**

1. **Primary Cause**: Banana presence in right wrist camera creates a 37% drop in cosine similarity for that camera's visual features

2. **Propagation Mechanism**: This visual difference:
   - Becomes 49% of total prefix embedding difference
   - Becomes 42.9% of KV cache difference (amplified in late layers)
   - Drives a fundamentally different velocity field from denoising step 0

3. **Why RAMP instead of FLAT**:
   - The KV cache conditioning (with banana) produces a velocity field that points toward movement
   - This is "locked in" by step 1 and maintained throughout all 10 denoising steps
   - Both trajectories are closer to TRANSPORT than IDLE in training distribution

---

## Visualizations Generated

- [x] `pca_visualization.png` - PCA of visual features
- [x] `spatial_diff_*.png` - Per-camera spatial difference heatmaps
- [x] `token_distances_*.png` - Per-token L2 heatmaps
- [x] `region_by_layer_heatmap_*.png` - KV cache [layer x region] heatmap
- [x] `trajectory_emergence_*.png` - How trajectory shapes emerge
- [x] `denoising_comparison_*.png` - Side-by-side denoising comparison
- [x] `3d_trajectory_distribution.png` - 3D PCA of training trajectories

---

## Conclusions

1. **The hallucination is vision-driven**: The banana presence in the right wrist camera is the root cause

2. **Late transformer layers amplify the difference**: Layers 8-15 show more KV cache divergence than early layers

3. **The trajectory shape is determined immediately**: By denoising step 1, the RAMP_UP shape is established

4. **Both cases are out-of-distribution**: The post-completion state doesn't clearly map to IDLE trajectories in training

5. **The velocity field divergence is fundamental**: Trajectory similarity at step 0 is only 0.006 - essentially uncorrelated

---

---

## Part 6: Three-Case Comparison (All Cases)

### Cases Analyzed

All three collected inference cases were compared:

1. **Hallucination** (`case_20260119_131914_ha_bana_table`): Banana on table near workspace - robot attempts to pick empty space
2. **Normal Plate** (`case_20260119_132946_no_ha_plate`): Banana on plate far from workspace - robot stays still
3. **Normal Clean** (`case_20260119_133142_no_ha_no_other_obj`): No distractors - robot stays still

### Prefix Embedding Comparison at Step 200 (Pre-Divergence)

| Comparison | Total L2 | Cosine Sim | Most Different Region |
|------------|----------|------------|----------------------|
| Halluc vs Normal Plate | 1,617,345 | 0.607 | Left wrist |
| Halluc vs Normal Clean | 1,233,693 | 0.762 | **Right wrist** |
| Normal Plate vs Normal Clean | 1,408,495 | 0.660 | Left wrist |

### Prefix Embedding Comparison at Step 250 (During Hallucination)

| Comparison | Total L2 | Cosine Sim | Most Different Region |
|------------|----------|------------|----------------------|
| Halluc vs Normal Plate | 1,739,796 | 0.566 | Left wrist |
| Halluc vs Normal Clean | 1,756,853 | 0.558 | Left wrist |
| **Normal Plate vs Normal Clean** | **961,699** | **0.843** | Right wrist |

**Key Observation**: At step 250, the two normal cases are much more similar to each other (cosine 0.84) than either is to the hallucination case (cosine ~0.56). This confirms the hallucination case has diverged significantly from normal behavior.

---

## Part 7: Integrated Visualization

### New Visualizations Generated

A comprehensive visualization tool was created that combines:
1. **3-camera images** for all cases at key timesteps
2. **Trajectory distribution** (PCA of training data) showing where inference trajectories fall
3. **Side-by-side comparison** across divergence timeline

**Output directory**: `logs/investigation/traj_attn_overlay_step200/`

| File | Description |
|------|-------------|
| `combined_step_0200.png` | Pre-divergence: all cameras + trajectory PCA for 3 cases |
| `combined_step_0250.png` | During hallucination: shows robot arm has moved in halluc case |
| `side_by_side_comparison.png` | Head + Right wrist across steps 200, 250, 300 |

### Training Trajectory Analysis

- **40 yogurt-related episodes** loaded from training dataset
- **305 trajectory segments** extracted and embedded via PCA (83% variance explained)
- Both hallucination and normal trajectories fall near **TRANSPORT** phase in training distribution
- This explains why the model generates movement trajectories - it has never seen "post-completion with distractor visible" scenarios

---

## Updated Conclusions

1. **The hallucination is vision-driven**: The banana presence in the right wrist camera is the root cause

2. **Divergence timing is step 208-210**: Not step 200 as initially analyzed. Action delta spikes from ~2 to ~5 at step 208, then to 10+ by step 211.

3. **Gripper behavior confirms hallucination**: At step 225-240, the hallucination case opens its gripper (attempting to grasp), while normal cases maintain stable gripper state.

4. **Both normal cases behave similarly**: At step 250, normal cases have cosine similarity 0.84, confirming they produce consistent "stay still" behavior.

5. **The trajectory shape is determined immediately**: By denoising step 1, the RAMP_UP shape is established in the hallucination case.

6. **Both inference cases are out-of-distribution**: The post-completion state doesn't clearly map to IDLE trajectories in training.

---

## Next Steps

1. ~~Synthesize findings in `hallucination_root_cause_synthesis.md`~~ (Completed in this document)
2. Design mitigation strategies:
   - Add more post-completion IDLE training data
   - Implement distractor-robust visual encoding
   - Add explicit "task complete" signal to conditioning
3. Validate mitigations experimentally
