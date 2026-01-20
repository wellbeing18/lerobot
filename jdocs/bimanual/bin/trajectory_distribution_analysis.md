# Trajectory Distribution Analysis

**Status**: Template - Awaiting Experiment Results
**Last Updated**: 2026-01-20

## Executive Summary

*To be populated after running Phase C experiments*

This document analyzes the training trajectory distribution and shows where hallucination and normal case trajectories lie in this space.

---

## Dataset Overview

### Dataset Information
- Dataset: `datasets_bimanuel/multitasks`
- Task filter: `yogurt` (yogurt bottle pick and place tasks)
- Episodes analyzed: *TBD*
- Total trajectory segments: *TBD*

### Phase Distribution in Training Data

| Phase | Count | Avg Duration (steps) | % of Total |
|-------|-------|---------------------|------------|
| APPROACH | | | |
| GRIP | | | |
| TRANSPORT | | | |
| RELEASE | | | |
| IDLE | | | |

---

## PCA Analysis

### Variance Explained

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | | |
| PC2 | | |
| PC3 | | |
| **Total** | | |

### Trajectory Clusters

*Description of clusters identified in the PCA space*

---

## Inference Case Analysis

### Hallucination Case
- Trace: `logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl`
- Steps analyzed: 200+ (post-task completion)

**Results:**
- Nearest training phase: *TBD*
- Distance to nearest: *TBD*
- In-distribution or out-of-distribution: *TBD*

### Normal Case
- Trace: `logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl`
- Steps analyzed: 200+ (post-task completion)

**Results:**
- Nearest training phase: *TBD*
- Distance to nearest: *TBD*
- In-distribution or out-of-distribution: *TBD*

---

## Key Questions Answered

### Q1: Does halluc trajectory fall in a valid training cluster?

*Answer with evidence*

### Q2: Which training phase does the post-completion halluc resemble?

*Answer with evidence*

### Q3: Is halluc trajectory out-of-distribution?

*Answer with evidence*

---

## Critical Findings

### Finding 1: IDLE Phase Representation

*Analysis of how well IDLE phase is represented in training data*

- Number of IDLE segments: *TBD*
- Average IDLE duration: *TBD*
- IDLE % of total training: *TBD*

### Finding 2: Hallucination Trajectory Resemblance

*What training phase does the hallucination trajectory most resemble?*

If hallucination resembles APPROACH phase:
- Suggests model is generating "approach" behavior when it should be generating "idle"
- Points to insufficient post-completion training data

### Finding 3: Data Collection Recommendations

Based on the analysis:

1. **More IDLE data needed?** *TBD*
2. **More distractor scenarios needed?** *TBD*
3. **More post-completion transitions needed?** *TBD*

---

## Visualizations

After running experiments:

- [ ] `3d_trajectory_distribution.png` - 3D PCA visualization
- [ ] `3d_trajectory_distribution.html` - Interactive plotly visualization
- [ ] `2d_projections.png` - 2D projection views
- [ ] `phase_distribution.png` - Phase histogram

---

## Relationship to Mechanism Analysis

This dataset analysis should be linked to the mechanism findings:

1. **If KV cache difference is in right_wrist camera:**
   - Check if training has examples of "post-completion with distractor visible"

2. **If denoising diverges at step 0:**
   - Suggests KV cache conditions different sampling regions
   - Dataset may lack examples of this visual context → IDLE behavior

3. **If halluc trajectory resembles APPROACH:**
   - Model defaults to learned movement patterns
   - Need more IDLE training data to override this

---

## Next Steps

1. Cross-reference with `hallucination_mechanism_findings.md`
2. Synthesize in `hallucination_root_cause_synthesis.md`
3. Generate data collection recommendations
