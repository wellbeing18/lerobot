# Mechanism Explanation: The Information-Action Gap

## Overview

This document explains the **information-action gap** - the phenomenon where the model **encodes** information about the banana in the KV cache but **ignores** it when making action decisions.

---

## The Core Finding: Information-Action Gap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     THE INFORMATION-ACTION GAP                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   RIGHT WRIST CAMERA:                                                   │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │  Information ENCODED:  42.9% of KV cache difference     │          │
│   │  Information USED:     11.0% causal effect on action    │          │
│   │  Ratio:                0.26 (IGNORED!)                  │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│   HEAD CAMERA:                                                          │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │  Information ENCODED:  23.2% of KV cache difference     │          │
│   │  Information USED:     167.6% causal effect on action   │          │
│   │  Ratio:                7.22 (DOMINANT!)                 │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│   INTERPRETATION:                                                       │
│   The model SEES the banana (encoded in right wrist) but DOESN'T       │
│   USE this information. Instead, it relies on head camera for          │
│   "should I move?" decisions.                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Mechanism

### STEP 1: Visual Encoding → KV Cache

```
                        VISUAL ENCODING PIPELINE

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │    HEAD     │     │    LEFT     │     │   RIGHT     │
    │   CAMERA    │     │   WRIST     │     │   WRIST     │
    │             │     │             │     │  (banana!)  │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │                   SigLIP ENCODER                    │
    │              (512×512 → 729 patches)               │
    └─────────────────────────────────────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │                    KV CACHE (K)                     │
    │  ┌────────┬────────┬────────┬────────┬────────┐   │
    │  │K_head  │K_left  │K_right │K_lang  │K_state │   │
    │  │(64 tok)│(64 tok)│(64 tok)│(48 tok)│(1 tok) │   │
    │  │        │        │ 42.9%  │        │        │   │
    │  │ 23.2%  │ 27.7%  │ DIFF!  │  5.1%  │  1.1%  │   │
    │  └────────┴────────┴────────┴────────┴────────┘   │
    └─────────────────────────────────────────────────────┘

    HALLUC CASE: K_right encodes "banana on workspace" (42.9% of diff)
    NORMAL CASE: K_right encodes "empty workspace"
```

### STEP 2: Cross-Attention → Action Decision

```
                    CROSS-ATTENTION MECHANISM

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ACTION TOKENS                    KV CACHE                 │
    │   (noisy trajectory)               (conditioning)           │
    │                                                             │
    │   ┌─────────────┐                  ┌─────────────────┐     │
    │   │ x_t (noise) │                  │ K = [K_head,    │     │
    │   │             │                  │      K_left,    │     │
    │   │ [a1,a2,...] │                  │      K_right,   │     │
    │   └──────┬──────┘                  │      K_lang,    │     │
    │          │                         │      K_state]   │     │
    │          │                         └────────┬────────┘     │
    │          │                                  │               │
    │          │         CROSS-ATTENTION          │               │
    │          └──────────────┬───────────────────┘               │
    │                         │                                   │
    │                         ▼                                   │
    │          ┌──────────────────────────────────┐               │
    │          │  v = f(∑_r α_r · g(x_t, t, K_r)) │               │
    │          └──────────────────────────────────┘               │
    │                         │                                   │
    │   LEARNED ATTENTION WEIGHTS (α_r):                         │
    │   ┌─────────────────────────────────────────────────┐      │
    │   │  α_head  = HIGH (167% causal effect)            │      │
    │   │  α_right = LOW  (11% causal effect)             │      │
    │   │                                                  │      │
    │   │  Despite: |ΔK_right| >> |ΔK_head|               │      │
    │   └─────────────────────────────────────────────────┘      │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    WHY THIS ASYMMETRY?

    Training data taught the model:
    ┌────────────────────────────────────────────────────────┐
    │  HEAD CAMERA   → "Is there work to do?" (scene layout) │
    │  RIGHT WRIST   → "How to grasp?" (local manipulation)  │
    └────────────────────────────────────────────────────────┘

    For post-completion decisions, "is there work to do?" is
    the relevant question, so head camera dominates.
```

### STEP 3: The Failure Mode

```
                        FAILURE MODE ANALYSIS

    ┌─────────────────────────────────────────────────────────────┐
    │                    HALLUCINATION CASE                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  1. RIGHT WRIST encodes: "There's a banana on workspace"    │
    │     │                                                       │
    │     └──▶ Model assigns LOW weight (α_right = 11%)           │
    │          Result: IGNORED                                    │
    │                                                             │
    │  2. HEAD CAMERA encodes: "Object detected in workspace"     │
    │     │                                                       │
    │     └──▶ Model assigns HIGH weight (α_head = 167%)          │
    │          Result: USED FOR DECISION                          │
    │                                                             │
    │  3. Model asks: "Should I move?"                            │
    │     │                                                       │
    │     └──▶ Head camera says: "Object in workspace" → YES      │
    │                                                             │
    │  4. Model does NOT ask: "Is this object relevant?"          │
    │     │                                                       │
    │     └──▶ Never learned this distinction (no training data)  │
    │                                                             │
    │  RESULT: Model outputs MOVEMENT despite task completion     │
    │          This is the HALLUCINATION                          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### STEP 4: Why Normal Case Works

```
                        NORMAL CASE ANALYSIS

    ┌─────────────────────────────────────────────────────────────┐
    │                      NORMAL CASE                            │
    │            (Banana on PLATE, not TABLE)                     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  1. HEAD CAMERA encodes: "Workspace area is EMPTY"          │
    │     │                                                       │
    │     └──▶ No object detected in workspace region             │
    │                                                             │
    │  2. Model asks: "Should I move?"                            │
    │     │                                                       │
    │     └──▶ Head camera says: "No object in workspace" → NO    │
    │                                                             │
    │  RESULT: Model outputs IDLE (correct behavior)              │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    KEY DIFFERENCE:
    ┌────────────────────────────────────────────────────────────┐
    │  HALLUC: Banana on TABLE (workspace) → Head sees object   │
    │  NORMAL: Banana on PLATE (destination) → Head sees empty  │
    └────────────────────────────────────────────────────────────┘
```

---

## Formal Mathematical Statement

```
Let K = [K_head, K_left, K_right, K_lang, K_state] be the KV cache regions.

The velocity field is:
    v(x_t, t, K) = f(∑_r α_r · g(x_t, t, K_r))

Where α_r are the learned attention weights for region r.

EXPERIMENTAL FINDING:
    α_head >> α_right  (despite |ΔK_right| >> |ΔK_head|)

    Specifically:
    ┌─────────────────────────────────────────────────┐
    │  Region      │  KV Diff  │  Causal Effect      │
    ├─────────────────────────────────────────────────┤
    │  right_wrist │   42.9%   │     11.0%           │
    │  head_camera │   23.2%   │    167.6%           │
    └─────────────────────────────────────────────────┘

The model learned this asymmetry because training data established:
    - Head camera → "is there work to do?" (global scene understanding)
    - Right wrist → "what to grasp?" (local manipulation)

For post-completion decisions, "is there work to do?" is the key question,
so head camera dominates. But head camera only encodes PRESENCE, not RELEVANCE.

DATASET BIAS:
    P_train(move | head_sees_object_on_workspace) = 1.0
    P_train(idle | head_sees_object_on_workspace) = 0.0

This bias propagates through the learned attention weights to cause hallucination.
```

---

## How to Read the Visualizations

### Figure: p_action_given_kv_cache_mechanism.png

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TOP-LEFT PANEL: Information Encoding vs Utilization Scatter            │
│  ─────────────────────────────────────────────────────────────          │
│                                                                         │
│    Y-axis: Information USED (% causal effect on action)                 │
│    X-axis: Information ENCODED (% of KV cache difference)               │
│                                                                         │
│    Each point = one KV cache region                                     │
│    Diagonal line = proportional utilization (y = x scaled)              │
│                                                                         │
│    HOW TO READ:                                                         │
│    ┌───────────────────────────────────────────────────────────┐       │
│    │  Points ABOVE diagonal: Information highly utilized       │       │
│    │  Points BELOW diagonal: Information ignored               │       │
│    │                                                            │       │
│    │  RED region (right, low): ENCODED but IGNORED             │       │
│    │  BLUE region (left, high): UTILIZED for decisions         │       │
│    └───────────────────────────────────────────────────────────┘       │
│                                                                         │
│    LOOK FOR:                                                            │
│    • right_wrist point: Far right (high x), low (low y) = IGNORED      │
│    • head_camera point: Left (moderate x), very high (high y) = USED   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  TOP-RIGHT PANEL: Bar Comparison (Encoded vs Used)                      │
│  ────────────────────────────────────────────────                       │
│                                                                         │
│    Red bars: Information encoded (KV cache difference)                  │
│    Blue bars: Information used (causal effect)                          │
│                                                                         │
│    LOOK FOR: The GAP between red and blue bars for right_wrist         │
│              Red bar is tall, blue bar is short = information ignored   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  BOTTOM-LEFT PANEL: Causal Chain Diagram                                │
│  ──────────────────────────────────────                                 │
│                                                                         │
│    Shows the flow: Visual Input → KV Cache → Cross-Attention → Output   │
│                                                                         │
│    KEY INSIGHT box at bottom explains the paradox:                      │
│    "Right wrist ENCODES banana (43%) but attention IGNORES it (11%)"   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  BOTTOM-RIGHT PANEL: Quantitative Statistics Table                      │
│  ──────────────────────────────────────────────                         │
│                                                                         │
│    Shows exact numbers for each region:                                 │
│    • KV Diff %: How much this region differs between cases              │
│    • Causal Effect %: How much this region affects the action           │
│    • Ratio: Effect / Diff (low = ignored, high = utilized)              │
│                                                                         │
│    LOOK FOR: right_wrist has ratio 0.26 (lowest = most ignored)        │
│              head_camera has ratio 7.22 (high = most utilized)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Figure: complete_hallucination_mechanism.png

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ROW 1: Information Flow Through Pipeline                               │
│  ────────────────────────────────────────                               │
│                                                                         │
│    Panel 1: Visual Input Difference (pixel level)                       │
│             → Shows right_wrist has highest visual difference           │
│                                                                         │
│    Panel 2: KV Cache Encoding (% of total difference)                   │
│             → Shows right_wrist encodes 43% of difference               │
│                                                                         │
│    Panel 3: Causal Effect (% change when zeroed)                        │
│             → Shows head_camera has 167% effect (dominant)              │
│             → Shows right_wrist has only 11% effect (ignored)           │
│                                                                         │
│    Panel 4: Information-Action Ratio                                    │
│             → Shows right_wrist ratio = 0.26 (ignored)                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ROW 2: Training Distribution and Inference                             │
│  ───────────────────────────────────────────                            │
│                                                                         │
│    Panel 5: Training P(velocity | context)                              │
│             → Shows MISSING region (yellow box) where training lacks    │
│               examples of IDLE with object visible                      │
│                                                                         │
│    Panel 6: Learned P(behavior | context)                               │
│             → Shows P(IDLE|obj) = 0 (the problem!)                      │
│                                                                         │
│    Panel 7: Inference Expected vs Actual                                │
│             → Shows MISMATCH for halluc case                            │
│                                                                         │
│    Panel 8: Post-Completion Trajectories                                │
│             → Shows halluc (red) has high velocity, normal (green) low  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ROW 3: Summary                                                         │
│  ────────────                                                           │
│                                                                         │
│    Left: Causal chain diagram (text-based)                              │
│    Right: Key quantitative findings summary                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Metric | Right Wrist | Head Camera | Interpretation |
|--------|-------------|-------------|----------------|
| KV Cache Diff | 42.9% | 23.2% | Right wrist has larger encoding difference |
| Causal Effect | 11.0% | 167.6% | Head camera dominates action decisions |
| Ratio | 0.26 | 7.22 | Right wrist info is IGNORED |

**Conclusion**: The model sees the banana (encoded in right wrist) but doesn't use this information. Instead, it relies on head camera which only encodes "object present" without distinguishing relevance. This is because training data created the bias: `P(move | object_visible) = 1.0`.
