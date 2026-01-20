# SmolVLA Hallucination Root Cause Synthesis

**Status**: Template - Awaiting Experiment Results
**Last Updated**: 2026-01-20

## Executive Summary

*To be populated after completing all experiments*

This document synthesizes findings from the mechanism analysis and dataset distribution analysis to provide a complete causal explanation for the SmolVLA hallucination phenomenon.

---

## The Fundamental Question

**Why does the VLM produce "stay still" in normal cases but "move toward empty space" in the hallucination case, when BOTH have completed the same task?**

The model CAN produce "stay still" behavior (proven by normal cases). The question is: **what's different in the VLM conditioning** that causes the hallucination case to produce movement instead?

---

## Complete Causal Chain

```
                    HALLUCINATION CASE               NORMAL CASE
                    ================               ===========
                         │                              │
                         ▼                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    VISUAL INPUT                              │
    │    Banana on table                    No distractors         │
    │    (right wrist camera)              (clean workspace)       │
    └────────────────────┬───────────────────────────┬─────────────┘
                         │                           │
                         ▼                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              SIGLIP ENCODER FEATURES                         │
    │    Different patch activations         Baseline features    │
    │    in right wrist region                                    │
    │    L2 difference: [TBD]               (Reference)           │
    └────────────────────┬───────────────────────────┬─────────────┘
                         │                           │
                         ▼                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              PREFIX EMBEDDING                                │
    │    Token region [128-191]             Token region [128-191]│
    │    (right wrist) differs              (baseline)            │
    │    by [TBD]% of total                                       │
    └────────────────────┬───────────────────────────┬─────────────┘
                         │                           │
                         ▼                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   KV CACHE                                   │
    │    Layers [TBD] show most             Baseline KV           │
    │    difference in [TBD] region                               │
    └────────────────────┬───────────────────────────┬─────────────┘
                         │                           │
                         ▼                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               DENOISING PROCESS                              │
    │    Divergence at step [TBD]           Stable FLAT shape     │
    │    Velocity points toward movement    Velocity → IDLE       │
    │    RAMP shape emerges                                       │
    └────────────────────┬───────────────────────────┬─────────────┘
                         │                           │
                         ▼                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               ACTION OUTPUT                                  │
    │    RAMP trajectory                    FLAT trajectory       │
    │    (move toward empty space)          (stay still)          │
    └─────────────────────────────────────────────────────────────┘
```

---

## Mechanism Analysis Summary

### Finding 1: Visual Feature Difference

*From `hallucination_mechanism_findings.md`*

- Primary camera with difference: *TBD*
- Feature difference magnitude: *TBD*
- Spatial localization: *TBD*

### Finding 2: Prefix Embedding Difference

- Dominant region: *TBD* ([vision/language/state])
- Within vision, dominant camera: *TBD*
- % contribution of each region: *TBD*

### Finding 3: KV Cache Difference

- Most affected layers: *TBD*
- Most affected token region: *TBD*
- Early vs late layer pattern: *TBD*

### Finding 4: Denoising Divergence

- Divergence step: *TBD*
- Driven by: *TBD* (KV cache / velocity field / late dynamics)
- Shape emergence pattern: *TBD*

---

## Dataset Analysis Summary

### Finding 1: Training Data Composition

*From `trajectory_distribution_analysis.md`*

- IDLE phase representation: *TBD*% of training data
- Post-completion scenarios: *TBD*
- Distractor scenarios: *TBD*

### Finding 2: Hallucination Trajectory Position

- Nearest training phase: *TBD*
- In-distribution: *Yes/No*
- Resembles: *TBD* phase behavior

---

## Root Cause Hypothesis

Based on the evidence:

### Primary Root Cause

*To be determined - one of:*

1. **KV Cache Conditioning**: Visual difference in right wrist camera creates different KV cache that conditions the action expert to sample from "approach" distribution rather than "idle" distribution.

2. **Training Data Gap**: Model lacks examples of "post-completion + distractor visible = stay still", so it defaults to learned movement patterns.

3. **Attention Mechanism**: Cross-attention to distractor region triggers retrieval of movement-related trajectory patterns.

4. **Flow Matching Instability**: Different visual context creates unstable denoising that converges to movement rather than stillness.

### Supporting Evidence

*Evidence for chosen hypothesis*

### Disconfirming Evidence

*Evidence against alternative hypotheses*

---

## Quantitative Summary

| Metric | Halluc vs Normal | Interpretation |
|--------|------------------|----------------|
| Visual feature cosine sim | | |
| Right wrist L2 distance | | |
| KV cache total L2 | | |
| Denoising divergence step | | |
| Action output correlation | | |

---

## Recommendations

### Short-term Mitigations

1. **Data augmentation**: *TBD*
2. **Inference-time intervention**: *TBD*
3. **Prompt modification**: *TBD*

### Long-term Solutions

1. **Dataset collection**: *TBD*
2. **Architecture modification**: *TBD*
3. **Training procedure**: *TBD*

---

## Research Contributions

### Novel Findings

1. *TBD*
2. *TBD*

### Methodology Contributions

1. First-principles pipeline tracing for VLA hallucination diagnosis
2. Denoising step-by-step analysis for understanding trajectory emergence
3. Training distribution visualization for identifying data gaps

### Future Work

1. *TBD*
2. *TBD*

---

## Appendix: Experiment Commands

```bash
# Phase A: Visual and Embedding Analysis
cd jdocs/scripts/investigation/tools

python vision_feature_comparison.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir outputs/vision_feature_comparison

python prefix_embedding_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir outputs/prefix_embedding_analysis

# Phase B: Model Hook Integration
python kv_cache_content_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir outputs/kv_cache_analysis

python denoising_step_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir outputs/denoising_analysis

# Phase C: Dataset Distribution Analysis
python trajectory_distribution_visualization.py \
    --dataset datasets_bimanuel/multitasks \
    --task-filter "yogurt" \
    --halluc-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
    --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
    --step-range "200,300" \
    --output-dir outputs/trajectory_distribution
```

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-20 | 0.1 | Initial template created |
| | | |
