# Head Camera Deep Dive: Understanding the Primary Causal Driver

**Date**: 2026-01-21
**Status**: Investigation Plan + Initial Findings

---

## Executive Summary

The attention knockout experiment revealed a surprising finding: the **head camera** has the highest causal effect (167%) on hallucination behavior, while the **right wrist camera** (where the banana is visible) has the lowest effect (11%). This contradicts the intuitive hypothesis that the banana directly drives the hallucination.

This report documents what we know, what we don't know, and proposes experiments to identify exactly what in the head camera drives the hallucination behavior.

---

## Part 1: Current Evidence

### 1.1 Attention Knockout Results (Hallucination Case, Step 200)

| Region Knocked Out | % Change | L2 Distance | Cosine Sim | Rank |
|-------------------|----------|-------------|------------|------|
| **head_camera** | **167.6%** | 14.54 | 0.235 | 1st (MOST causal) |
| language | 131.1% | 16.09 | 0.275 | 2nd |
| state | 66.1% | 5.54 | 0.831 | 3rd |
| left_wrist | 22.2% | 1.75 | 0.984 | 4th |
| **right_wrist** | **11.0%** | 3.09 | 0.954 | 5th (LEAST causal) |

**Key Observation**: When head_camera KV cache is zeroed out:
- Action delta changes by 167.6%
- L2 distance = 14.54 (large trajectory change)
- Cosine similarity = 0.235 (trajectory direction completely changes)

### 1.2 Cross-Case Comparison

| Case | Head Camera Effect | Right Wrist Effect |
|------|-------------------|-------------------|
| Hallucination | 167.6% | 11.0% |
| Normal (plate) | 41.8% | 29.1% |
| Normal (no obj) | 436.8% | 167.4% |

**Pattern**: Head camera is consistently the most or second-most causal region across all cases.

### 1.3 Visual Feature Comparison (from vision_features report)

| Camera | Cosine Similarity (Halluc vs Normal) | L2 Distance |
|--------|-------------------------------------|-------------|
| head | 0.9363 | 1109.37 |
| left_wrist | 0.8013 | 1633.84 |
| right_wrist | 0.6271 | 2574.83 |

**Paradox**: Head camera has the **highest similarity** between cases (0.94), yet has the **highest causal effect**. This means:
- Small differences in head camera features → large differences in action
- Large differences in right wrist features → small differences in action

### 1.4 KV Cache Content Analysis

From KV cache analysis at step 200:

| Region | % of Total KV Difference |
|--------|-------------------------|
| right_wrist | 42.9% |
| left_wrist | 27.7% |
| head_camera | 23.2% |
| language | 5.1% |
| state | 1.1% |

**The Disconnect**: Right wrist has 42.9% of KV difference but only 11% causal effect. Head camera has only 23.2% of KV difference but 167% causal effect.

---

## Part 2: What We Don't Know

### 2.1 Unknown: What Specific Features in Head Camera?

The head camera shows the workspace from a global view. Potential features that could drive behavior:

| Possible Feature | Hypothesis |
|-----------------|------------|
| Table surface appearance | Model learned "certain table configs → movement" |
| Object positions on table | Spatial layout triggers approach behavior |
| Background elements | Scene context signals "task active" |
| Lighting/shadows | Visual cues correlated with task phases in training |
| Gripper/arm visibility | Seeing the arm in certain positions → movement |
| Empty space on table | "Empty spot = place to put something" |

### 2.2 Unknown: Why High Causal Effect Despite High Similarity?

The head camera features are 94% similar between halluc and normal cases, yet small differences have huge impact. This suggests:
- The model is **highly sensitive** to specific head camera features
- A small "trigger" in head camera activates movement behavior
- The decision boundary in feature space is near the head camera representations

### 2.3 Unknown: Link to Training Data

Questions about training data:
- What head camera scenes correspond to MOVEMENT vs IDLE in training?
- Is there a specific visual pattern that always precedes movement?
- Did the model learn a "scene → action" shortcut from head camera?

---

## Part 3: Proposed Deep Dive Experiments

### Experiment 3.1: Head Camera Spatial Knockout

**Purpose**: Identify which spatial region of head camera matters most.

**Method**:
1. Divide head camera's 64 tokens into spatial regions (8×8 grid → quadrants)
2. Knock out each quadrant separately:
   - Top-left (tokens 0-15): Usually shows background/ceiling
   - Top-right (tokens 16-31): Background/wall
   - Bottom-left (tokens 32-47): Left side of table
   - Bottom-right (tokens 48-63): Right side of table (workspace)
3. Measure action change for each quadrant knockout

**Expected Insight**: Identify if the causal information is in a specific spatial region (e.g., the table surface vs background).

```python
# Pseudo-code
head_cam_tokens = range(0, 64)  # Head camera token indices in prefix

quadrants = {
    'top_left': range(0, 16),
    'top_right': range(16, 32),
    'bottom_left': range(32, 48),
    'bottom_right': range(48, 64),
}

for name, tokens in quadrants.items():
    # Zero out KV cache for these tokens only
    modified_kv = zero_out_tokens(kv_cache, tokens)
    action = run_inference(modified_kv)
    measure_change(action, baseline_action)
```

---

### Experiment 3.2: Head Camera Swap Between Cases

**Purpose**: Directly test if head camera alone can flip behavior.

**Method**:
1. **Swap A**: Use halluc case but replace head camera with normal case's head camera
   - Keep: halluc left_wrist, halluc right_wrist (banana visible), halluc state
   - Replace: head_camera → normal's head_camera
   - Question: Does hallucination disappear?

2. **Swap B**: Use normal case but replace head camera with halluc case's head camera
   - Keep: normal left_wrist, normal right_wrist (no banana), normal state
   - Replace: head_camera → halluc's head_camera
   - Question: Does hallucination appear?

**Expected Insight**: If Swap A produces IDLE behavior and Swap B produces MOVEMENT, this proves head camera is the causal driver.

```python
# Swap experiment pseudo-code
def swap_head_camera(source_case, head_cam_from_case):
    # Load images
    source_images = load_images(source_case)
    donor_head = load_image(head_cam_from_case, 'head')

    # Create hybrid input
    hybrid_images = {
        'head': donor_head,  # Swapped
        'left_wrist': source_images['left_wrist'],
        'right_wrist': source_images['right_wrist'],
    }

    # Run inference
    action = model.infer(hybrid_images, state, language)
    return action
```

---

### Experiment 3.3: Head Camera Feature Attribution

**Purpose**: Identify which pixels/patches in head camera image contribute most to action.

**Method**:
1. Use gradient-based attribution (Integrated Gradients or GradCAM)
2. Compute gradients of action output w.r.t. head camera input pixels
3. Visualize attribution heatmap overlaid on head camera image
4. Compare attribution maps between halluc vs normal cases

**Expected Insight**: Visual heatmap showing which parts of the head camera image the model "looks at" when generating actions.

```python
# Attribution pseudo-code
def compute_attribution(model, images, target_action_dim):
    images['head'].requires_grad = True

    # Forward pass
    action = model.infer(images, state, language)

    # Backward pass for specific action dimension
    action[0, 0, target_action_dim].backward()

    # Get gradient attribution
    attribution = images['head'].grad.abs().sum(dim=1)  # Sum over channels
    return attribution

# Compare attributions
halluc_attr = compute_attribution(model, halluc_images, action_dim=2)  # e.g., elbow joint
normal_attr = compute_attribution(model, normal_images, action_dim=2)
```

---

### Experiment 3.4: Head Camera → Training Data Correlation

**Purpose**: Find what head camera features correlate with MOVEMENT vs IDLE in training data.

**Method**:
1. Extract head camera images from training episodes
2. Label each frame by action type: MOVEMENT (velocity > threshold) or IDLE (velocity < threshold)
3. Train a simple classifier (or use clustering) on head camera features to predict MOVEMENT/IDLE
4. Analyze what visual features the classifier uses

**Expected Insight**: Identify if specific head camera patterns (table configuration, object layout) are correlated with movement in training data.

```python
# Training data analysis pseudo-code
def analyze_training_head_cameras():
    movement_frames = []
    idle_frames = []

    for episode in training_episodes:
        for t in range(len(episode)):
            velocity = compute_velocity(episode.actions[t])
            head_cam = episode.images['head'][t]

            if velocity > MOVEMENT_THRESHOLD:
                movement_frames.append(head_cam)
            else:
                idle_frames.append(head_cam)

    # Extract features using SigLIP
    movement_features = siglip.encode(movement_frames)
    idle_features = siglip.encode(idle_frames)

    # Find discriminative features
    diff = movement_features.mean(0) - idle_features.mean(0)
    top_dims = diff.abs().topk(10)
    return top_dims
```

---

### Experiment 3.5: Cross-Attention Pattern to Head Camera

**Purpose**: Understand how action tokens query head camera during denoising.

**Method**:
1. Capture cross-attention weights during denoising
2. Focus on attention from action tokens to head camera tokens (0-63)
3. Compare attention patterns between halluc vs normal cases
4. Visualize which head camera patches receive most attention

**Challenge**: SDPA attention doesn't support weight extraction. Need to either:
- Temporarily switch to standard attention
- Use attention approximation methods

**Expected Insight**: See if halluc case attends to different head camera patches than normal case.

---

### Experiment 3.6: Minimal Head Camera Perturbation

**Purpose**: Find the smallest change to head camera that flips behavior.

**Method**:
1. Start with halluc case (produces movement)
2. Gradually interpolate head camera features toward normal case:
   ```
   head_features = α * halluc_head + (1-α) * normal_head
   ```
3. Find the α threshold where behavior flips from MOVEMENT to IDLE
4. Analyze what changes at that threshold

**Expected Insight**: Identify the "decision boundary" in head camera feature space.

---

## Part 4: Dataset Defect Analysis

### 4.1 Hypothesis: Scene-Action Shortcut

The model may have learned a **shortcut** from training data:

```
Training Pattern Observed:
- Certain head camera scenes (object on table, arm visible) → MOVEMENT phase
- These scenes ALWAYS have movement in training
- Model learns: "this scene configuration → MOVE"

Inference Problem:
- Post-completion, head camera shows similar "active scene"
- Model triggers movement even though task is complete
- Banana is irrelevant - it's the overall scene layout
```

### 4.2 Questions to Answer from Dataset

| Question | How to Answer |
|----------|---------------|
| What head camera scenes precede MOVEMENT? | Cluster head cam features, label by action phase |
| What head camera scenes precede IDLE? | Same clustering, look for IDLE clusters |
| Is there overlap between post-completion scene and movement scenes? | Compare inference head cam to training clusters |
| What's missing in IDLE training data? | Look for head cam configurations that NEVER appear with IDLE |

### 4.3 Proposed Dataset Analysis

```python
def analyze_head_camera_action_correlation():
    """
    Build a map: head_camera_features → typical_action
    """
    # 1. Extract all (head_cam, action_velocity) pairs from training
    pairs = []
    for episode in training_data:
        for t in range(len(episode)):
            head_feat = siglip.encode(episode.head_cam[t])
            velocity = compute_velocity(episode.action[t])
            pairs.append((head_feat, velocity))

    # 2. Cluster head camera features
    features = np.array([p[0] for p in pairs])
    velocities = np.array([p[1] for p in pairs])

    clusters = KMeans(n_clusters=10).fit(features)

    # 3. For each cluster, compute mean velocity
    cluster_velocity = {}
    for c in range(10):
        mask = clusters.labels_ == c
        cluster_velocity[c] = velocities[mask].mean()

    # 4. Identify "movement clusters" vs "idle clusters"
    movement_clusters = [c for c, v in cluster_velocity.items() if v > THRESHOLD]
    idle_clusters = [c for c, v in cluster_velocity.items() if v < THRESHOLD]

    # 5. Check where inference cases fall
    halluc_head_feat = siglip.encode(halluc_head_cam)
    halluc_cluster = clusters.predict([halluc_head_feat])[0]

    return {
        'halluc_cluster': halluc_cluster,
        'cluster_type': 'movement' if halluc_cluster in movement_clusters else 'idle',
        'cluster_velocity': cluster_velocity,
    }
```

### 4.4 Expected Dataset Defects

Based on current findings, likely dataset defects:

| Defect | Impact | Evidence |
|--------|--------|----------|
| **No IDLE with "active scene" head camera** | Model never learns to stay still when scene looks active | Head camera high causal effect |
| **Scene-action correlation** | Certain head camera views always paired with movement | Need to verify with clustering |
| **Missing post-completion diversity** | IDLE phases always have same head camera config | Current IDLE data may be too uniform |

---

## Part 5: Experiment Priority and Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT DEPENDENCY GRAPH                   │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: Direct Causal Tests (Highest Priority)
├── Exp 3.2: Head Camera Swap ←── CRITICAL: Proves head cam is causal driver
│                                  If swap flips behavior, case closed
│
└── Exp 3.1: Spatial Knockout ←── Identifies WHICH part of head cam matters
                                   Narrowing down from 64 tokens to ~16

PHASE 2: Mechanism Understanding
├── Exp 3.3: Feature Attribution ←── Requires Phase 1 results
│                                     Visualize exact pixels that matter
│
└── Exp 3.6: Minimal Perturbation ←── Find decision boundary
                                       Requires understanding from 3.1

PHASE 3: Dataset Linkage
├── Exp 3.4: Training Data Correlation ←── Link model behavior to data
│                                           Can run in parallel with Phase 1
│
└── Exp 3.5: Cross-Attention (Optional) ←── Nice to have, technically challenging
```

### Recommended Execution Order

1. **Exp 3.2 (Head Camera Swap)** - Highest priority, directly proves hypothesis
2. **Exp 3.1 (Spatial Knockout)** - Narrow down which region matters
3. **Exp 3.4 (Dataset Correlation)** - Understand training data patterns
4. **Exp 3.3 (Feature Attribution)** - Visualize the mechanism
5. **Exp 3.6 (Minimal Perturbation)** - Find exact decision boundary

---

## Part 6: Expected Outcomes

### If Head Camera Swap (Exp 3.2) Shows Behavior Flip:

```
Evidence Chain:
1. Halluc + Normal_head_cam → IDLE (behavior flips)
2. Normal + Halluc_head_cam → MOVEMENT (behavior flips)

Conclusion: Head camera ALONE determines behavior
Next: Find what's different in head camera images

Implications for Data Collection:
- Collect IDLE data with diverse head camera scenes
- Ensure head camera "active scene" configs paired with IDLE
- Break the scene-action shortcut
```

### If Spatial Knockout (Exp 3.1) Shows Localized Effect:

```
Evidence Chain:
1. Bottom-right quadrant knockout → largest action change
2. This quadrant shows: [table workspace / objects / arm]

Conclusion: Model uses specific spatial region for decisions
Next: Feature attribution on that region

Implications for Data Collection:
- Focus on that spatial region's content during IDLE
- Collect diverse configurations in that region
```

### If Dataset Correlation (Exp 3.4) Shows Scene-Action Mapping:

```
Evidence Chain:
1. Cluster X of head camera features → always MOVEMENT in training
2. Halluc inference head camera falls in Cluster X
3. No IDLE examples exist in Cluster X

Conclusion: Dataset defect - missing IDLE for specific scene configs
Next: Targeted data collection for Cluster X + IDLE

Implications for Data Collection:
- Identify underrepresented (scene, action) pairs
- Fill gaps with targeted demonstrations
```

---

## Part 7: Implementation Status

| Experiment | Status | Script Location |
|------------|--------|-----------------|
| 3.1 Spatial Knockout | TO DO | `jdocs/scripts/investigation/tools/head_camera_spatial_knockout.py` |
| 3.2 Head Camera Swap | TO DO | `jdocs/scripts/investigation/tools/head_camera_swap_experiment.py` |
| 3.3 Feature Attribution | TO DO | `jdocs/scripts/investigation/tools/head_camera_attribution.py` |
| 3.4 Dataset Correlation | TO DO | `jdocs/scripts/investigation/tools/head_camera_dataset_analysis.py` |
| 3.5 Cross-Attention | BLOCKED | SDPA doesn't support attention weights |
| 3.6 Minimal Perturbation | TO DO | `jdocs/scripts/investigation/tools/head_camera_perturbation.py` |

---

## Appendix: Head Camera Token Layout

The head camera is encoded into 64 tokens from an 8×8 patch grid:

```
Head Camera Image (512×512) → SigLIP → 64 tokens

Token Index Layout (assuming row-major):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│  0 │  1 │  2 │  3 │  4 │  5 │  6 │  7 │  ← Top row (background/ceiling)
├────┼────┼────┼────┼────┼────┼────┼────┤
│  8 │  9 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 32 │ 33 │ 34 │ 35 │ 36 │ 37 │ 38 │ 39 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 40 │ 41 │ 42 │ 43 │ 44 │ 45 │ 46 │ 47 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 48 │ 49 │ 50 │ 51 │ 52 │ 53 │ 54 │ 55 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 56 │ 57 │ 58 │ 59 │ 60 │ 61 │ 62 │ 63 │  ← Bottom row (table surface)
└────┴────┴────┴────┴────┴────┴────┴────┘

Quadrants for Spatial Knockout:
- Q1 (Top-Left): tokens 0-3, 8-11, 16-19, 24-27
- Q2 (Top-Right): tokens 4-7, 12-15, 20-23, 28-31
- Q3 (Bottom-Left): tokens 32-35, 40-43, 48-51, 56-59
- Q4 (Bottom-Right): tokens 36-39, 44-47, 52-55, 60-63  ← Likely workspace area
```

---

---

## Part 7: Experimental Results and Revised Understanding

### Exp 3.2 Results: Head Camera Swap

**Experiments Run**:
- Step 200 and Step 250 tested
- All 4 swap combinations tested (head, right_wrist in both directions)

**Unexpected Finding**:

| Experiment | Halluc Baseline | Normal Baseline | Expected Difference |
|------------|-----------------|-----------------|---------------------|
| Step 200 | 0.060 | 0.288 | Halluc >> Normal |
| Step 250 | 0.415 | 3.143 | Halluc >> Normal |
| **Actual** | **Lower** | **Higher** | **REVERSED** |

**Key Observation**: Single-frame inference **does not reproduce** the hallucination behavior seen in sequential inference.

### Critical Insight: Temporal Context Dependency

The hallucination phenomenon appears to be **emergent from temporal/sequential inference**, not from single-frame visual differences:

1. **Original Trace Data** (sequential inference):
   - Halluc at step 250: action_delta_max = 12.16 (MOVEMENT)
   - Normal at step 250: action_delta_max = 2.32 (STAY STILL)

2. **Our Single-Frame Inference** (isolated inference):
   - Halluc at step 250: delta_max = 0.415 (similar low values)
   - Normal at step 250: delta_max = 3.143 (reversed!)

3. **Implication**: The hallucination requires **accumulated temporal context** from previous steps. Running inference on a single frame without history doesn't reproduce the phenomenon.

### Reinterpreting Attention Knockout Results

Given this finding, we must reinterpret the attention knockout results:

| Original Interpretation | Revised Interpretation |
|------------------------|----------------------|
| "Head camera drives hallucination" | "Model RELIES ON head camera for ANY decision" |
| "Right wrist is not causal" | "Right wrist info is STORED but not heavily USED" |
| "Head camera is root cause" | "Head camera is heavily used, but hallucination may be emergent" |

**Attention Knockout Measures**:
- Which token regions the model **relies on** for generating actions
- NOT which regions **cause the specific hallucination**

**The Distinction**:
```
Causal Reliance (what knockout measures):
  "If I remove head_camera, output changes a lot"
  → Model needs head_camera for ANY decision

Hallucination Causation (what we want to know):
  "What makes halluc different from normal?"
  → Requires comparing behaviors, not just knockouts
```

### New Hypothesis: Emergent Hallucination

The hallucination may be caused by:

```
┌─────────────────────────────────────────────────────────────┐
│             TEMPORAL CONTEXT ACCUMULATION                   │
│                                                              │
│  Step 1-150: Task execution                                  │
│  Step 150-200: Task completion, transition                   │
│  Step 200+: POST-COMPLETION                                  │
│                                                              │
│  HALLUC CASE:                    NORMAL CASE:                │
│  ┌──────────────────────┐       ┌──────────────────────┐    │
│  │ Visual: banana seen  │       │ Visual: no distractor│    │
│  │ + Action history of  │       │ + Action history of  │    │
│  │   approach movements │       │   completion + idle  │    │
│  └──────────┬───────────┘       └──────────┬───────────┘    │
│             │                              │                 │
│             ▼                              ▼                 │
│  Model continues movement         Model stays still          │
│  (learned pattern from history)   (learned idle pattern)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implications for Investigation

1. **Single-frame experiments are limited**
   - Cannot directly test hallucination causation
   - Useful for understanding general model reliance

2. **Need sequential experiments**
   - Run full sequential inference with camera interventions
   - Intervene at specific steps during sequence
   - Compare resulting trajectories

3. **Attention knockout is still valuable**
   - Shows model architecture/reliance patterns
   - Confirms head_camera is important for decisions
   - But doesn't prove head_camera CAUSES hallucination specifically

### Revised Experiment Design

**Exp 3.2b: Sequential Camera Swap**

Instead of single-frame inference, run full sequential inference:

```python
def sequential_swap_experiment():
    # Run halluc case normally until step 150
    for step in range(150):
        action = model.infer(halluc_images[step], halluc_state[step])
        update_history(action)

    # At step 150, swap head camera and continue
    for step in range(150, 300):
        images = {
            'head': normal_images[step]['head'],  # SWAPPED
            'left_wrist': halluc_images[step]['left_wrist'],
            'right_wrist': halluc_images[step]['right_wrist'],
        }
        action = model.infer(images, halluc_state[step])
        # Does hallucination disappear after swap?
```

This would test if swapping head camera MID-SEQUENCE changes the behavior.

---

## Summary of Current Understanding

### What We Know With Confidence:

1. **Model relies heavily on head_camera** (attention knockout shows 167% effect)
2. **Model barely uses right_wrist info** (only 11% knockout effect)
3. **Hallucination is emergent** (single-frame inference doesn't reproduce it)
4. **Model is confidently wrong** (low uncertainty during hallucination)

### What Remains Uncertain:

1. **Does head_camera CAUSE hallucination, or just enable it?**
   - High reliance ≠ causation of specific behavior

2. **Is the hallucination caused by visual input or temporal context?**
   - Sequential experiments needed

3. **What specific head_camera features matter?**
   - Spatial knockout still valuable to run

### Recommended Next Steps

1. **Run spatial knockout** (Exp 3.1) - Still valuable for understanding head_camera structure
2. **Design sequential swap experiment** (Exp 3.2b) - True causal test
3. **Analyze training data head_camera patterns** (Exp 3.4) - Dataset linkage
4. **Consider temporal context analysis** - How does action history affect current output?

---

## Files Generated

| File | Location |
|------|----------|
| Swap Results (Step 200) | `logs/investigation/advanced_analysis_20260121/head_camera_swap/` |
| Swap Results (Step 250) | `logs/investigation/advanced_analysis_20260121/head_camera_swap_step250/` |
| This Report | `logs/investigation/advanced_analysis_20260121/head_camera_deep_dive.md` |
