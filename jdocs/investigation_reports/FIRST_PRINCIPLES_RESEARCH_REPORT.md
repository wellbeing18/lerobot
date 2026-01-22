# SmolVLA Hallucination: First Principles Research Report

**Date**: 2026-01-22
**Status**: MECHANISM IDENTIFIED

---

## Abstract

We present a first-principles analysis of the hallucination behavior in SmolVLA, a Vision-Language-Action (VLA) model using flow matching for trajectory generation. Through systematic decomposition of the model's information flow, we identify a fundamental **information-action gap**: the model encodes relevant information (distractor object in right wrist camera) but does not utilize it for action decisions. Instead, the model relies on head camera features for "should I move?" decisions, where training data has created a perfect correlation: `P(movement | object_on_workspace) = 1.0`. This dataset distribution bias is the root cause of the hallucination.

---

## 1. Problem Statement

### 1.1 Observed Behavior

When SmolVLA completes a pick-and-place task with a distractor object (banana) on the workspace, it exhibits "hallucination" - continued movement toward the workspace area despite task completion.

### 1.2 Research Questions

1. **P(trajectory | KV_cache)**: How does the KV cache conditioning affect trajectory generation?
2. **Information Flow**: Where in the model does the hallucination decision occur?
3. **Training Distribution**: What dataset bias caused the model to learn this behavior?

---

## 2. Mathematical Framework

### 2.1 SmolVLA Architecture

SmolVLA is a conditional flow matching model:

```
trajectory ~ FlowMatching(noise, v(·, ·, K))
```

Where:
- `K = KVCache(images, language, state)` is the conditioning from the VLM
- `v(x_t, t, K)` is the learned velocity field
- The trajectory emerges from integrating the velocity field over denoising steps

### 2.2 KV Cache Structure

The KV cache K is composed of token regions:

```
K = [K_head, K_left_wrist, K_right_wrist, K_language, K_state]
     └────────────────────────────────────────────────────────┘
                        Prefix tokens (~241 total)
```

Each region encodes different information:
- `K_head`: Global scene layout from overhead camera
- `K_left_wrist`, `K_right_wrist`: Local gripper views
- `K_language`: Task instruction
- `K_state`: Current joint positions

### 2.3 Action Generation via Cross-Attention

The velocity field is computed through cross-attention:

```
v(x_t, t, K) = ActionExpert(CrossAttention(action_tokens, K))
```

The cross-attention weights `α_r` determine how much each KV region contributes:

```
v ≈ f(Σ_r α_r · g(x_t, t, K_r))
```

---

## 3. Experimental Results

### 3.1 KV Cache Difference Analysis

Comparing hallucination case vs. baseline (no distractor) at step 200:

| Region | L2 Difference | % of Total |
|--------|---------------|------------|
| right_wrist | 3650.49 | **42.9%** |
| left_wrist | 2362.09 | 27.7% |
| head_camera | 1976.46 | 23.2% |
| language | 433.15 | 5.1% |
| state | 94.07 | 1.1% |

**Finding**: The banana causes the largest KV cache difference in the **right_wrist** region (42.9%).

### 3.2 Causal Effect Analysis (Attention Knockout)

Measuring the effect of zeroing each KV region on action output:

| Region | Causal Effect (% change) |
|--------|-------------------------|
| head_camera | **167.6%** |
| language | 131.1% |
| state | 66.1% |
| left_wrist | 22.2% |
| right_wrist | **11.0%** |

**Finding**: The head_camera has the highest causal effect (167.6%) despite having a smaller KV difference than right_wrist.

### 3.3 Information-Action Ratio

Ratio = Causal Effect / KV Difference (measures information utilization):

| Region | KV Diff % | Causal Effect % | Ratio |
|--------|-----------|-----------------|-------|
| state | 1.1% | 66.1% | 59.84 |
| language | 5.1% | 131.1% | 25.78 |
| head_camera | 23.2% | 167.6% | 7.22 |
| left_wrist | 27.7% | 22.2% | 0.80 |
| **right_wrist** | **42.9%** | **11.0%** | **0.26** |

**Key Finding**: Right wrist has the **lowest** information-action ratio (0.26), meaning the model encodes banana information but **ignores** it for action decisions.

---

## 4. The Information-Action Gap

### 4.1 Definition

The **information-action gap** is the phenomenon where:
- Information is **encoded** in the model's representations
- But this information is **not utilized** for decision-making

### 4.2 Quantification

For the right_wrist region:
- **Encoded**: 42.9% of total KV cache difference
- **Used**: 11.0% causal effect on action
- **Gap**: 42.9 / 11.0 = 3.9x more encoded than used

For the head_camera region:
- **Encoded**: 23.2% of total KV cache difference
- **Used**: 167.6% causal effect on action
- **Utilization**: Head camera is 7.2x more utilized than its encoding proportion

### 4.3 Interpretation

The model has learned asymmetric attention weights:

```
α_head >> α_right_wrist
```

This means:
- Head camera dominates "should I move?" decisions
- Right wrist is used for "how to grasp?" decisions (irrelevant post-completion)
- The banana information in right wrist is encoded but ignored

---

## 5. Training Distribution Analysis

### 5.1 Learned Conditional Distribution

The model approximates the training distribution:

```
P_θ(trajectory | K) ≈ P_train(trajectory | K)
```

### 5.2 Training Data Bias

From training data analysis:

| Visual Pattern | Behavior | Training Coverage |
|----------------|----------|-------------------|
| Object on workspace | MOVEMENT | **100%** |
| Clean workspace | IDLE | ~25% |
| Object on workspace | IDLE | **0%** |

The training data has a **perfect correlation**:

```
P_train(movement | object_visible_on_workspace) = 1.0
P_train(idle | object_visible_on_workspace) = 0.0
```

### 5.3 Why the Bias Propagates

1. Training episodes always have: `object_on_workspace → approach and grasp`
2. IDLE occurs only after: `task_complete AND workspace_empty`
3. Training never shows: `task_complete AND object_on_workspace → IDLE`

The model learns:
- "Object detected in workspace" → "There's work to do" → MOVEMENT
- Without learning the exception: "Irrelevant object" → IGNORE

---

## 6. Causal Mechanism

### 6.1 Complete Causal Chain

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAUSAL MECHANISM                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  VISUAL INPUT: Banana on workspace                                       │
│       │                                                                  │
│       ▼                                                                  │
│  KV CACHE ENCODING:                                                      │
│       - right_wrist: Encodes "banana present" (42.9% of diff)           │
│       - head_camera: Encodes "object in workspace area" (23.2% of diff) │
│       │                                                                  │
│       ▼                                                                  │
│  CROSS-ATTENTION (learned weights):                                      │
│       - α_head = HIGH (167% causal effect)                              │
│       - α_right = LOW (11% causal effect)                               │
│       │                                                                  │
│       ▼                                                                  │
│  DECISION LOGIC (from head_camera):                                      │
│       "Object detected in workspace" → "Work to do" → MOVEMENT          │
│       │                                                                  │
│       ▼                                                                  │
│  VELOCITY FIELD:                                                         │
│       v(x_t, t, K_halluc) → positive magnitude → movement trajectory    │
│       │                                                                  │
│       ▼                                                                  │
│  OUTPUT: Robot moves toward workspace area (HALLUCINATION)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Why Normal Case Works

When banana is on PLATE (not TABLE):
1. Head camera encodes: "Workspace area is empty"
2. Model queries: "Is there work to do?"
3. Head camera answers: "No object in workspace"
4. Velocity field: v → zero magnitude → stay still
5. Result: Correct IDLE behavior

---

## 7. Formal Statement

**Theorem (Hallucination Mechanism)**:

Let `K = [K_head, K_wrist, ...]` be the KV cache, and let `v(x, t, K)` be the learned velocity field with attention weights `{α_r}`.

If training data satisfies:
1. `∀ frame: (object_on_workspace → action = movement)`
2. `∀ frame: (workspace_empty → action = idle)`

Then the model learns:
- `α_head >> α_wrist` (head camera dominates scene-level decisions)
- `P_θ(movement | head_sees_object) = 1.0`

At inference, if:
- `head_camera` shows object on workspace (even if irrelevant)
- Right wrist shows the specific object (banana)

The model produces `movement` because:
- It queries `head_camera` for "should I move?" (high α_head)
- It ignores `right_wrist` for object relevance (low α_wrist)
- The answer from `head_camera` is "object present" → MOVE

This is a **distribution shift** problem: the training distribution lacks examples of `(object_on_workspace, idle)`, so the model cannot generalize to this case.

---

## 8. Implications and Fix

### 8.1 Root Cause Summary

The hallucination is caused by:
1. **Dataset distribution bias**: `P(idle | object_on_workspace) = 0` in training
2. **Learned attention asymmetry**: `α_head >> α_wrist` for scene-level decisions
3. **Information-action gap**: Banana encoded but not utilized for decisions

### 8.2 Recommended Fix

Collect training data that breaks the correlation:

```
NEW TRAINING DATA:
  - Visual: Object on workspace (distractor)
  - Action: IDLE (ignore the object)
  - This teaches: P(idle | irrelevant_object_on_workspace) > 0
```

Estimated requirement: 20-30 episodes (~750-1500 frames)

### 8.3 Alternative Approaches

1. **Attention steering**: Force higher attention to right_wrist for object identification
2. **Task-completion signal**: Add explicit "task done" token to language
3. **Contrastive training**: Train with (relevant vs. irrelevant object) pairs

---

## 9. Conclusion

This investigation reveals that the SmolVLA hallucination is not a model architecture bug but a **dataset distribution problem**. The model correctly learns from training data, but training data lacks the specific examples needed to handle "irrelevant object on workspace" scenarios.

The key insight is the **information-action gap**: the model encodes object information in the right wrist KV cache (42.9% of difference) but ignores it for action decisions (11% causal effect). Instead, it relies on head camera features (23.2% encoded, 167% effect), where training data has created a perfect correlation between "object visible" and "movement required."

This is a fundamental limitation of imitation learning from biased distributions, not a failure of the VLA architecture.

---

## References

1. SmolVLA: [Architecture details]
2. Flow Matching: Conditional flow matching for trajectory generation
3. Attention Knockout: Causal intervention methodology
4. KV Cache Analysis: Token-level representation comparison

---

## Appendix: Key Evidence Files

| File | Description |
|------|-------------|
| `logs/investigation/first_principles/information_action_gap.png` | Visualization of the gap |
| `logs/investigation/first_principles/mechanism_explanation.txt` | Detailed mechanism |
| `logs/investigation/first_principles/causal_mechanism_diagram.png` | Causal diagram |
| `logs/investigation/advanced_analysis_20260121/attention_knockout/` | Knockout experiment data |
| `logs/investigation/kv_cache_20260121_153318/` | KV cache comparison data |
