# SmolVLA Hallucination Investigation: Final Summary

**Date**: 2026-01-22
**Status**: ROOT CAUSE IDENTIFIED ✓

---

## The Problem

When the SmolVLA model completes a pick-and-place task but a distractor (banana) is visible, it **hallucinates** movement toward the distractor instead of staying idle.

## The Root Cause

> **Dataset Context Gap**: The training data contains IDLE behavior only in "clean" visual contexts. It has ZERO examples of IDLE behavior when a graspable distractor is visible.

---

## Evidence Summary

### 1. Behavioral Evidence (Inference Traces)

| Metric (Post-Completion, Step 200+) | Hallucination | Normal |
|-------------------------------------|---------------|--------|
| Mean action delta | 8.25° | 2.42° |
| Movement ratio | 3.4x more | baseline |
| % above IDLE threshold | 95.8% | 0.0% |
| Behavior | MOVEMENT | IDLE |

### 2. Dataset Evidence (Training Data Analysis)

| Visual Context | Action | Training Coverage |
|----------------|--------|-------------------|
| Object visible | MOVEMENT | 75% ✓ |
| No object | IDLE | 25% ✓ |
| **Distractor visible** | **IDLE** | **0% ✗** |

Total training frames: 76,597
- IDLE frames: 19,141 (25%)
- IDLE with distractor: 0 (0%) ← **THE GAP**

### 3. Architectural Evidence (Attention Knockout)

| Region | Causal Effect | Interpretation |
|--------|---------------|----------------|
| Head camera | 167% | Model relies heavily on this |
| Right wrist | 11% | Model ignores banana info |

**Why?** The model stores banana info (49% of KV cache difference) but doesn't use it (11% causal effect) because it never learned what to DO when seeing a distractor during post-completion.

---

## The Mechanism

```
TRAINING DATA TAUGHT:
┌─────────────────────────────────────┐
│ "object visible" → MOVEMENT         │
│ "clean scene" → IDLE                │
└─────────────────────────────────────┘

INFERENCE SCENARIO:
┌─────────────────────────────────────┐
│ Task complete + distractor visible  │
│ → Model has NO learned behavior     │
│ → Defaults to "object → MOVEMENT"   │
│ → HALLUCINATION                     │
└─────────────────────────────────────┘
```

---

## Key Insights

### 1. Quantity vs Context
The issue is NOT a lack of IDLE data (25% exists). The issue is a lack of IDLE data **in the specific visual context** encountered at inference.

### 2. Pattern Matching, Not Understanding
The model learned visual patterns, not task semantics. It matches "object visible" to movement regardless of task completion status.

### 3. Head Camera Dominance
The model relies on head camera (167% causal effect) for global scene understanding, not right wrist (11%). The banana creates KV cache differences but doesn't influence decisions.

### 4. Temporal Context Matters
Single-frame experiments don't reproduce hallucination. The behavior is emergent from sequential inference, suggesting temporal patterns matter.

---

## Recommended Fix

### Data Collection: 20-30 Episodes

**Scenario**: Post-completion IDLE with visible distractor

```
1. Execute normal pick-and-place task
2. Complete task (object placed)
3. Introduce distractor on table
4. Demonstrate IDLE for 30-50 frames
5. Keep distractor visible throughout
```

### Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| IDLE + distractor examples | 0 | ~750-1500 frames |
| Coverage of "ignore distractor" | 0% | ~5-10% |
| Hallucination rate | 95.8% | Significantly reduced |

---

## Files Generated

| File | Description |
|------|-------------|
| `dataset_defect_findings.md` | Detailed training data analysis |
| `head_camera_deep_dive.md` | Head camera investigation |
| `bimanual_training_data_analysis.png` | Training data distribution |
| `hallucination_emergence_visualization.png` | Inference trace comparison |

## Investigation Tools Created

| Tool | Purpose |
|------|---------|
| `head_camera_swap_experiment.py` | Test camera causality via swapping |
| `head_camera_spatial_knockout.py` | Test spatial regions of head camera |
| `attention_knockout.py` | Measure causal importance of token regions |
| (+ 6 other tools) | Various analysis methods |

---

## Conclusion

The SmolVLA hallucination is **explainable and fixable**. It's caused by a specific gap in training data coverage where the model has never learned to stay idle when seeing a graspable distractor. Targeted data collection of 20-30 episodes demonstrating this behavior should resolve the issue.
