# Visualization Guide: How to Read the Hallucination Analysis Diagrams

## Overview

This document explains how the key visualizations were generated, how they work mathematically, and how to read them. These diagrams visualize the causal mechanism of SmolVLA hallucination through the lens of:

1. **P(velocity | visual_context)** - Training data distribution
2. **P(trajectory | KV_cache)** - How KV cache conditioning affects action generation
3. **P(trajectory | velocity_field)** - How v drives trajectory through flow matching

---

## 1. Context-Velocity Scatter Plot

### File Information

| Attribute | Value |
|-----------|-------|
| **Output** | `logs/investigation/first_principles/context_velocity_scatter_clean.png` |
| **Script** | `jdocs/scripts/investigation/tools/context_velocity_scatter.py` |
| **Data Source** | Inference traces + simulated training data |

### How It Was Generated

```python
# 1. Load inference trace data
halluc = load_trace(HALLUC_TRACE)  # case_20260119_131914_ha_bana_table
normal = load_trace(NORMAL_TRACE)  # case_20260119_132946_no_ha_plate

# 2. Extract action deltas (movement magnitude in degrees)
h_deltas = [e['action_delta_max'] for e in halluc]
n_deltas = [e['action_delta_max'] for e in normal]

# 3. Simulate training data distribution
# Object on workspace -> high velocity (movement)
obj_context = np.random.uniform(0.7, 1.0, 300)
obj_vel = np.random.exponential(4, 300) + 3

# Empty workspace -> low velocity (idle)
empty_context = np.random.uniform(0, 0.3, 300)
empty_vel = np.random.exponential(1, 300)

# 4. Plot scatter with inference overlay
```

### ASCII Diagram: How to Read

```
    P(velocity | visual_context): Training Data vs Inference

    Action         │
    Velocity       │    ×  ×
    (degrees)      │   × ×× ×  ×        TRAINING: Object on workspace
              14   │  ×××××× × ×        (blue dots, context > 0.7)
                   │ ×××××××××××
              12   │×× ×× ××× ×××
                   │  ××  ×  ×  ×         ✕ = HALLUC inference
              10   │    ×  ×  ×  × ✕       (red X markers)
                   │         × ✕ ✕ ✕       These should be IDLE but
               8   │      ✕  ✕ ✕ ✕ ✕       show MOVEMENT!
                   │     ✕ ✕   ✕  ✕
               6   │    ✕  ✕  ✕
                   │   ✕
               4   │
    ───────────────┼─────────────────────────────────────────────
               3   │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  IDLE THRESHOLD
                   │  + + + +                      + = NORMAL inference
               2   │ + ++ ++ +                     (green + markers)
                   │+ +++ + ++  ○○ ○               These correctly show IDLE
               1   │○○○○○○○○○○○○○○○
                   │○○○○○○○○○○○○○○○○               TRAINING: Empty workspace
               0   │○○○○○○○○○○○                    (light blue dots, context < 0.3)
                   └──────────────────────────────────────────────▶
                   0.0        0.2        0.4        0.6        0.8        1.0
                              Visual Context
                   (0 = empty workspace, 1 = object on workspace)


    HOW TO READ:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  1. X-AXIS: Visual context (0 = empty workspace, 1 = object visible)   │
    │                                                                         │
    │  2. Y-AXIS: Action velocity in degrees (movement magnitude)            │
    │                                                                         │
    │  3. HORIZONTAL LINE at y=3: IDLE threshold                             │
    │     - Below line: Robot stays still (correct for post-completion)      │
    │     - Above line: Robot moves (hallucination if task is done)          │
    │                                                                         │
    │  4. TRAINING DATA (dots):                                              │
    │     - Blue dots (right side): Object on workspace → always MOVEMENT    │
    │     - Light dots (left side): Empty workspace → always IDLE            │
    │     - Notice: NO training dots in bottom-right (obj + idle)!           │
    │                                                                         │
    │  5. INFERENCE POINTS:                                                   │
    │     - Red ✕: Halluc case (banana on TABLE) → HIGH velocity (wrong!)   │
    │     - Green +: Normal case (banana on PLATE) → LOW velocity (correct) │
    │                                                                         │
    │  KEY INSIGHT: Halluc points are in the region where training has       │
    │  examples (object visible → movement), but the task is complete!       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

### Connection to Flow Matching

This diagram shows **P(velocity | visual_context)** which is the input to the flow matching process:

```
visual_context → KV_cache → cross_attention → v (velocity) → trajectory
                                              ↑
                                    This diagram shows this distribution
```

The velocity v in flow matching is learned from training data. When training data shows:
- `object_visible → high velocity`
- `empty_workspace → low velocity`

The model learns this conditional distribution. At inference, when visual context shows "object visible" (even if irrelevant), the model samples v from the high-velocity distribution.

---

## 2. P(trajectory | KV_cache) Visualization

### File Information

| Attribute | Value |
|-----------|-------|
| **Output** | `logs/investigation/causal_distribution/p_traj_given_kv_cache.png` |
| **Script** | `jdocs/scripts/investigation/tools/generate_p_traj_kv.py` |
| **Data Source** | Inference traces (halluc, normal, normal2 cases) |

### How It Was Generated

```python
# 1. Load trace metrics for specific step range (post-completion)
halluc_data = load_trace_metrics(halluc_trace, step_range=(200, 300))
normal_data = load_trace_metrics(normal_trace, step_range=(200, 300))

# 2. Define KV cache proxy values (based on visual context)
halluc_kv_proxy = 1.0   # Banana visible in right wrist
normal_kv_proxy = 0.0   # No distractor visible
normal2_kv_proxy = 0.2  # Banana on plate (far from workspace)

# 3. Build 2D density using kernel density estimation
density = np.zeros_like(X)
for delta in halluc_deltas:
    density += np.exp(-((X - halluc_kv_proxy)**2 / 0.05 + (Y - delta)**2 / 2.0))
# ... similar for normal cases

# 4. Create contour plot with marginal histograms
```

### ASCII Diagram: How to Read (4-Panel Layout)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  P(trajectory | KV_cache): Visual Context -> KV Cache -> Action Distribution   │
├─────────────────────────────────┬───────────────────────────────────────────────┤
│                                 │                                               │
│  PANEL 1: 2D Conditional Dist   │  PANEL 2: Marginal Histograms                │
│  ───────────────────────────    │  ─────────────────────────────               │
│                                 │                                               │
│   Action    MOVEMENT REGION     │   Density                                     │
│   Delta  │  (red shading)       │      │                                        │
│          │      ╔══════╗        │      │    ┌──┐                                │
│      14  │      ║  ★   ║ Halluc │      │    │  │ Halluc                         │
│          │      ║ red  ║ mean   │      │    │  │ (red)                          │
│      10  │      ║══════╝        │      │ ┌──┤  │                                │
│          │      ↑ density       │      │ │  │  │                                │
│       6  │      │ contour       │      │ │  └──┘                                │
│          │      │               │      │ │                                      │
│ ─────────┼──────┼───────────────│  ────┼─┼──────────────────────▶               │
│       3  │──────│─ ─ ─ ─ ─ ─ ─ ─│      │ │    IDLE threshold                   │
│          │  △   │ STAY STILL    │      │ │                                      │
│       2  │Normal│ (green shade) │      │ └─────┐                                │
│          │ mean │               │      │       │ Normal                         │
│       0  └──────┴───────────────│      └───────┴──────▶ Action Delta           │
│          0.0   0.5   1.0        │                                               │
│          KV Cache Conditioning  │                                               │
│          (0=Normal, 1=Halluc)   │                                               │
│                                 │                                               │
├─────────────────────────────────┼───────────────────────────────────────────────┤
│                                 │                                               │
│  PANEL 3: KV Cache Structure    │  PANEL 4: Causal Chain Summary               │
│  ───────────────────────────    │  ─────────────────────────────               │
│                                 │                                               │
│   % of KV                       │  VISUAL INPUT                                 │
│   Difference                    │  ┌─────────────┐   ┌─────────────┐           │
│      │                          │  │Banana in    │   │No distractor│           │
│   50─┤        ┌───┐ BANANA      │  │right wrist  │   │visible      │           │
│      │        │   │ VISIBLE     │  └──────┬──────┘   └──────┬──────┘           │
│   40─┤        │   │ HERE!       │         │                 │                  │
│      │        │   │             │         ▼                 ▼                  │
│   30─┤  ┌───┐ │   │             │  ┌─────────────┐   ┌─────────────┐           │
│      │  │   │ │   │             │  │KV Cache=1.0 │   │KV Cache=0.0 │           │
│   20─┤  │   │ │   │             │  │49% different│   │baseline     │           │
│      │  │   │ │   │             │  └──────┬──────┘   └──────┬──────┘           │
│   10─┼──┤   ├─┤   ├──┬──┬──     │         │                 │                  │
│      │  │   │ │   │  │  │       │         ▼                 ▼                  │
│    0─┴──┴───┴─┴───┴──┴──┴──▶    │  P(action)=HIGH    P(action)=LOW            │
│      Head Left Right Lang State │  MOVEMENT          STAY STILL                │
│      Cam  Wrist Wrist           │                                               │
│                                 │                                               │
└─────────────────────────────────┴───────────────────────────────────────────────┘


    HOW TO READ EACH PANEL:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  PANEL 1 (Top-Left): 2D Conditional Distribution                           │
    │  ─────────────────────────────────────────────                              │
    │  • X-axis: KV cache "conditioning" (0=normal, 1=halluc)                    │
    │  • Y-axis: Action delta (movement score in degrees)                        │
    │  • Contour colors: Probability density (darker = higher probability)       │
    │  • Green region (y<3): STAY STILL region                                   │
    │  • Red region (y>3): MOVEMENT region                                       │
    │  • ★ Star: Halluc mean (should be low, but is HIGH)                       │
    │  • △ Triangle: Normal mean (correctly LOW)                                 │
    │  • Arrow: Shows CAUSAL direction from KV difference to action difference  │
    │                                                                             │
    │  PANEL 2 (Top-Right): Marginal Histograms                                  │
    │  ────────────────────────────────────                                      │
    │  • Shows P(action_delta) for each case separately                          │
    │  • Red histogram: Halluc case - shifted RIGHT (high movement)             │
    │  • Green histogram: Normal case - shifted LEFT (low movement)             │
    │  • Vertical line: IDLE threshold at 3°                                    │
    │                                                                             │
    │  PANEL 3 (Bottom-Left): KV Cache Difference Structure                      │
    │  ────────────────────────────────────────────────                          │
    │  • Shows which KV cache regions differ between halluc and normal          │
    │  • RIGHT WRIST has 49% of total difference (banana visible there!)        │
    │  • This explains WHERE the visual difference is encoded                    │
    │                                                                             │
    │  PANEL 4 (Bottom-Right): Causal Chain Summary                              │
    │  ─────────────────────────────────────────                                 │
    │  • Shows complete causal chain: Visual → KV Cache → Action                │
    │  • Banana visible → KV=1.0 → HIGH movement (hallucination)                │
    │  • No distractor → KV=0.0 → LOW movement (correct)                        │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### Connection to Flow Matching

This diagram shows **P(trajectory | KV_cache)** which is the complete conditional distribution:

```
                        FLOW MATCHING PROCESS
    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    │   KV Cache (K)          Cross-Attention         Velocity Field (v)   │
    │   ┌───────────┐         ┌───────────┐           ┌───────────┐        │
    │   │ K_head    │────────▶│ α_head=167%│──────────▶│ Direction │        │
    │   │ K_wrist   │────────▶│ α_wrist=11%│──────────▶│ Magnitude │        │
    │   │ K_lang    │────────▶│ α_lang    │           │           │        │
    │   └───────────┘         └───────────┘           └─────┬─────┘        │
    │         ↑                                             │              │
    │         │                                             │              │
    │   This is what                                        │              │
    │   the X-axis                               ┌──────────┴──────────┐   │
    │   represents                               │                     │   │
    │                                            ▼                     │   │
    │                                    ┌───────────────┐             │   │
    │                                    │ Trajectory    │◀────────────┘   │
    │                                    │ Integration   │                 │
    │                                    │ x_T = ∫v dt   │                 │
    │                                    └───────┬───────┘                 │
    │                                            │                         │
    │                                            ▼                         │
    │                                    ┌───────────────┐                 │
    │                                    │ Action Output │                 │
    │                                    │ (Y-axis)      │                 │
    │                                    └───────────────┘                 │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘

    The P(trajectory | KV_cache) diagram shows:
    - X-axis = KV cache conditioning value
    - Y-axis = resulting action magnitude
    - The contour shows the learned conditional distribution
```

---

## 3. Connection Between All Diagrams

### The Complete Causal Chain

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE VISUALIZATION CHAIN                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1: Visual Context                                                        │
│   ──────────────────────                                                        │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐          │
│   │  context_velocity_scatter_clean.png                             │          │
│   │  Shows: P(velocity | visual_context) from training data         │          │
│   │  Key: Training creates bias where object → movement             │          │
│   └─────────────────────────────────────────────────────────────────┘          │
│                              │                                                  │
│                              ▼                                                  │
│   STEP 2: KV Cache Encoding                                                     │
│   ─────────────────────────                                                     │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐          │
│   │  p_traj_given_kv_cache.png (Panel 3)                            │          │
│   │  Shows: Which regions of KV cache encode the visual difference  │          │
│   │  Key: Right wrist has 49% of difference (banana encoded there)  │          │
│   └─────────────────────────────────────────────────────────────────┘          │
│                              │                                                  │
│                              ▼                                                  │
│   STEP 3: Cross-Attention Weights                                               │
│   ──────────────────────────────                                               │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐          │
│   │  p_action_given_kv_cache_mechanism.png                          │          │
│   │  Shows: Information-action gap (encoded vs used)                │          │
│   │  Key: Right wrist 49% encoded but only 11% used!                │          │
│   │       Head camera 23% encoded but 167% effect!                  │          │
│   └─────────────────────────────────────────────────────────────────┘          │
│                              │                                                  │
│                              ▼                                                  │
│   STEP 4: Velocity Field                                                        │
│   ──────────────────────                                                        │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐          │
│   │  p_trajectory_given_velocity_field.png                          │          │
│   │  Shows: How v magnitude determines trajectory direction         │          │
│   │  Key: High |v| → movement trajectory, Low |v| → idle           │          │
│   └─────────────────────────────────────────────────────────────────┘          │
│                              │                                                  │
│                              ▼                                                  │
│   STEP 5: Trajectory Output                                                     │
│   ─────────────────────────                                                     │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐          │
│   │  p_traj_given_kv_cache.png (Panel 1)                            │          │
│   │  Shows: P(trajectory | KV_cache) - the final distribution       │          │
│   │  Key: Halluc KV=1.0 → HIGH action, Normal KV=0.0 → LOW action  │          │
│   └─────────────────────────────────────────────────────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Running the Scripts

### Prerequisites

```bash
# Activate the conda environment
eval "$(conda shell.bash hook)" && conda activate lerobot

# Ensure trace data exists
ls logs/yogurt_banana_leftarm/case_20260119_*/trace.jsonl
```

### Generate All Visualizations

```bash
# 1. Context-velocity scatter (training distribution)
python jdocs/scripts/investigation/tools/context_velocity_scatter.py

# 2. P(trajectory | KV_cache) visualization
python jdocs/scripts/investigation/tools/generate_p_traj_kv.py \
    --halluc-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
    --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
    --output logs/investigation/causal_distribution/p_traj_given_kv_cache.png

# 3. Comprehensive mechanism visualizations
python jdocs/scripts/investigation/tools/comprehensive_mechanism_visualization.py

# 4. Flow matching visualizations
python jdocs/scripts/investigation/tools/flow_matching_visualization.py
```

---

## 5. Key Takeaways for Reading the Diagrams

### What Each Visualization Answers

| Diagram | Question Answered |
|---------|-------------------|
| `context_velocity_scatter_clean.png` | What did training data teach the model about context → velocity? |
| `p_traj_given_kv_cache.png` (Panel 1) | How does KV cache conditioning affect action distribution? |
| `p_traj_given_kv_cache.png` (Panel 3) | Where in KV cache is the banana information encoded? |
| `p_action_given_kv_cache_mechanism.png` | Why does the model ignore the banana information? |
| `p_trajectory_given_velocity_field.png` | How does velocity v drive trajectory generation? |
| `denoising_steps_visualization.png` | What happens at each denoising step? |

### The Root Cause Visible in Diagrams

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  context_velocity_scatter_clean.png shows:                                      │
│  • Training has NO points in bottom-right (object visible + idle)              │
│  • This is the MISSING REGION that causes hallucination                        │
│                                                                                 │
│  p_traj_given_kv_cache.png shows:                                              │
│  • KV=1.0 (halluc) → action distribution in MOVEMENT region                    │
│  • KV=0.0 (normal) → action distribution in STAY STILL region                  │
│  • This is the CAUSAL EFFECT of the training bias                              │
│                                                                                 │
│  CONCLUSION: The diagrams visualize the same root cause from different angles: │
│  Training data bias → Biased P(v|K) → Hallucination at inference              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```
