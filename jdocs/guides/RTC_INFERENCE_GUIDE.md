# Real-Time Chunking (RTC) Inference Guide

This guide explains Real-Time Chunking (RTC), a technique for smooth robot control with action-chunking policies like SmolVLA and ACT.

## Table of Contents
1. [The Problem: Jerky Motion](#the-problem-jerky-motion)
2. [How RTC Solves It](#how-rtc-solves-it)
3. [Visual Comparison](#visual-comparison)
4. [Walking Example](#walking-example)
5. [RTC vs Async Inference](#rtc-vs-async-inference)
6. [Implementation Guide](#implementation-guide)
7. [When to Use What](#when-to-use-what)

---

## The Problem: Jerky Motion

Action-chunking policies (ACT, SmolVLA, Diffusion Policy) predict multiple future actions at once. This is efficient but creates a problem:

**Standard Inference Flow:**
```
Time ─────────────────────────────────────────────────────────────►

┌─────────────────────┐     ┌─────────────────────┐
│   Chunk A (50 actions)    │   Chunk B (50 actions)    │
│   Execute actions 1-50    │   Execute actions 1-50    │
└─────────────────────┘     └─────────────────────┘
                        ▲
                        │
                    PROBLEM!
              Sudden jump between
              chunk A → chunk B
```

**Why it happens:**
- Chunk A was predicted at time T₀ with observation O₀
- Chunk B is predicted at time T₁ with observation O₁
- The robot moved during chunk A, so O₁ ≠ what chunk A expected
- Chunk B starts from a "surprised" state → sudden correction → **jerk**

---

## How RTC Solves It

RTC (Real-Time Chunking) uses **guided denoising** to blend new chunks with already-executed actions:

```
Time ─────────────────────────────────────────────────────────────►

┌─────────────────────┐
│   Chunk A (50 actions)    │
│   actions 1-50            │
└───────────┬───────────────┘
            │
    While executing A,    ┌─────────────────────┐
    predict chunk B       │   Chunk B (50 actions)    │
    with "guidance"       │   Blends with A's tail    │
            │             └─────────────────────┘
            │                      │
            └──────────────────────┘
                  Smooth transition!
                  B is "guided" to match
                  what A already did
```

**Key Insight:** When generating chunk B, RTC tells the model:
> "The first N actions should match what chunk A already executed"

This constraint is enforced during the denoising process, resulting in smooth transitions.

---

## Visual Comparison

### Standard Chunking (No RTC)
```
Observation     ○─────────────────────────────────○─────────────────────────────────○
                │                                 │                                 │
                ▼                                 ▼                                 ▼
             Predict                           Predict                           Predict
             Chunk 1                           Chunk 2                           Chunk 3
                │                                 │                                 │
                ▼                                 ▼                                 ▼
Actions     ═══════════════════════════╪═══════════════════════════╪═══════════════════
                                       ▲                           ▲
                                       │                           │
                                    JERK!                       JERK!
                              (discontinuity)              (discontinuity)

Timeline:   |-------- 1.6s --------|-------- 1.6s --------|-------- 1.6s --------|
            (50 actions @ 30Hz)
```

### Async Inference (Parallel Prediction)
```
Observation     ○───────────○───────────○───────────○───────────○───────────○
                │           │           │           │           │           │
                ▼           ▼           ▼           ▼           ▼           ▼
             Predict     Predict     Predict     Predict     Predict     Predict
             (async)     (async)     (async)     (async)     (async)     (async)
                │           │           │           │           │           │
                ▼           ▼           ▼           ▼           ▼           ▼
Actions     ═══════════════════════════════════════════════════════════════════════
                     ▲           ▲           ▲           ▲
                     │           │           │           │
                  smaller     smaller     smaller     smaller
                  jerks       jerks       jerks       jerks

Timeline:   |--- 0.5s ---|--- 0.5s ---|--- 0.5s ---|--- 0.5s ---|
            (predict more frequently, still has discontinuities)
```

### RTC (Guided Denoising)
```
Observation     ○───────────────────────○───────────────────────○
                │                       │                       │
                ▼                       ▼                       ▼
             Predict               Predict with              Predict with
             Chunk 1               guidance from 1           guidance from 2
                │                       │                       │
                │    ┌──────────────────┘                       │
                │    │ "Match first N actions                   │
                │    │  to chunk 1's tail"                      │
                ▼    ▼                                          ▼
Actions     ═══════════════════════════════════════════════════════════════════════
                         │                       │
                         │                       │
                      SMOOTH!                 SMOOTH!
                  (guided blending)       (guided blending)

Timeline:   |-------- 1.6s --------|-------- 1.6s --------|
```

---

## Walking Example

Let's trace through a concrete example with numbers:

### Setup
- **Policy:** SmolVLA with chunk_size=50
- **Control rate:** 30 Hz (33ms per action)
- **Inference time:** ~500ms (15 actions worth of time)

### Step-by-Step Execution

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ T=0ms: Start                                                                  │
│ ─────────────────────────────────────────────────────────────────────────────│
│ • Capture observation O₀                                                      │
│ • Predict chunk A: [a₁, a₂, a₃, ..., a₅₀]                                    │
│ • Start executing actions                                                     │
│ • inference_delay = 0 (first chunk, nothing to skip)                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ T=500ms: Queue getting low, request new chunk                                 │
│ ─────────────────────────────────────────────────────────────────────────────│
│ • Actions executed so far: a₁ through a₁₅ (15 actions @ 33ms each)           │
│ • Queue remaining: [a₁₆, a₁₇, ..., a₅₀] = 35 actions                         │
│ • Capture observation O₁                                                      │
│ • Calculate inference_delay = ceil(500ms / 33ms) = 15                        │
│ • Call: predict_action_chunk(O₁, inference_delay=15, prev_chunk=A)           │
│                                                                               │
│   Inside RTC:                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │ • Generate chunk B: [b₁, b₂, ..., b₅₀]                              │    │
│   │ • But GUIDE b₁..b₁₅ to match a₁..a₁₅ (already executed)            │    │
│   │ • Result: b₁≈a₁, b₂≈a₂, ..., b₁₅≈a₁₅ (smooth continuation)         │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ T=1000ms: Chunk B prediction complete                                         │
│ ─────────────────────────────────────────────────────────────────────────────│
│ • Inference took 500ms, during which 15 more actions executed                │
│ • Actions executed: a₁ through a₃₀                                           │
│ • Total consumed during inference: 15 + 15 = 30 actions                      │
│ • Merge chunk B, skipping first 30 actions (already outdated)                │
│ • New queue: [a₃₁..a₅₀] + [b₃₁..b₅₀]                                         │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │ Queue visualization:                                                │    │
│   │                                                                     │    │
│   │ Before merge: [a₃₁, a₃₂, ..., a₅₀]  (20 remaining from A)          │    │
│   │ After merge:  [a₃₁..a₅₀, b₃₁..b₅₀] (20 + 20 = 40 actions)          │    │
│   │                         ↑                                           │    │
│   │               Smooth transition here because                        │    │
│   │               b₃₁ was guided to match context from A                │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ T=1500ms: Request chunk C                                                     │
│ ─────────────────────────────────────────────────────────────────────────────│
│ • Queue getting low again                                                     │
│ • predict_action_chunk(O₂, inference_delay=15, prev_chunk=B)                 │
│ • C is guided to blend with B's already-executed portion                     │
│ • Cycle continues...                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `inference_delay` | 15 | Skip first 15 actions of new chunk (already outdated) |
| `prev_chunk_left_over` | Chunk A | Previous chunk's actions for guidance |
| `execution_horizon` | 10-15 | How many actions to blend/guide |
| `max_guidance_weight` | 10.0 | Strength of consistency enforcement |

---

## RTC vs Async Inference

### What is Async Inference?

Async inference decouples action prediction from execution using a client-server architecture:

```
┌─────────────────┐         ┌─────────────────┐
│     Client      │         │     Server      │
│  (Robot Loop)   │◄───────►│  (GPU/Policy)   │
│                 │  gRPC   │                 │
│ • Read sensors  │         │ • Run model     │
│ • Execute acts  │         │ • Return chunks │
│ • 30Hz loop     │         │ • Async compute │
└─────────────────┘         └─────────────────┘
```

### Comparison Table

| Aspect | Standard | Async | RTC | Async + RTC |
|--------|----------|-------|-----|-------------|
| **Jerkiness** | High | Medium | Low | Lowest |
| **Latency handling** | Poor | Good | Good | Best |
| **Implementation** | Simple | Complex | Medium | Complex |
| **Hardware** | Single machine | Can distribute | Single machine | Can distribute |
| **Idle frames** | Many | Few | Few | Minimal |
| **Code changes** | None | Major refactor | Add 2 params | Both |

### Do We Still Need Async?

**Yes, async and RTC solve different problems:**

| Problem | Async Solves? | RTC Solves? |
|---------|---------------|-------------|
| Jerky transitions between chunks | Partially | **Yes** |
| GPU on separate machine | **Yes** | No |
| Inference slower than execution | **Yes** | Partially |
| Network latency | **Yes** | No |
| Smooth action blending | No | **Yes** |

**Best Practice:**
- **Single machine, fast GPU:** RTC alone is sufficient
- **Distributed setup:** Use Async + RTC together
- **Slow inference:** Async helps more than RTC
- **Fast inference but jerky:** RTC alone

---

## Implementation Guide

### Standard Inference (No RTC)
```python
# Simple but jerky
while running:
    obs = get_observation()
    actions = policy.select_action(obs)  # Returns chunk
    for action in actions:
        robot.send_action(action)
        time.sleep(1/30)
```

### RTC Inference
```python
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.configs.types import RTCAttentionSchedule

# Configure RTC
rtc_config = RTCConfig(
    enabled=True,
    execution_horizon=10,        # Actions to blend
    max_guidance_weight=10.0,    # Guidance strength
    prefix_attention_schedule=RTCAttentionSchedule.EXP,
)
policy.config.rtc_config = rtc_config
policy.init_rtc_processor()

# Inference with RTC
prev_chunk = None
while running:
    obs = get_observation()

    # Calculate how many steps will pass during inference
    inference_delay = estimate_inference_steps()

    # Predict with guidance from previous chunk
    actions = policy.predict_action_chunk(
        obs,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_chunk,
    )

    prev_chunk = actions  # Save for next iteration

    # Execute (in parallel thread ideally)
    for action in actions[inference_delay:]:
        robot.send_action(action)
```

### Async + RTC (Production Setup)
```python
# Server side (GPU machine)
from lerobot.async_inference.server import AsyncModelServer

server = AsyncModelServer(
    policy=policy,
    rtc_config=rtc_config,
)
server.start()

# Client side (Robot machine)
from lerobot.async_inference.client import AsyncModelClient

client = AsyncModelClient(server_address="gpu-server:50051")

while running:
    obs = get_observation()
    # Non-blocking: returns immediately, result arrives later
    future = client.predict_async(obs)

    # Execute from queue while waiting
    action = action_queue.get()
    robot.send_action(action)
```

---

## When to Use What

### Decision Flowchart

```
                    ┌─────────────────────┐
                    │ Is inference fast   │
                    │ enough for real-time│
                    │ (< 100ms)?          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
             YES                               NO
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ Is motion jerky │              │ Use Async       │
    │ between chunks? │              │ (+ RTC optional)│
    └────────┬────────┘              └─────────────────┘
             │
    ┌────────┴────────┐
    │                 │
   YES               NO
    │                 │
    ▼                 ▼
┌─────────┐    ┌─────────────┐
│ Use RTC │    │ Standard is │
│         │    │ fine        │
└─────────┘    └─────────────┘
```

### Quick Reference

| Scenario | Recommendation |
|----------|----------------|
| SmolVLA on RTX 3090/4090/5090 | RTC alone |
| ACT on mid-range GPU | RTC alone |
| Diffusion Policy (slow) | Async + RTC |
| GPU on cloud/server | Async + RTC |
| Raspberry Pi client | Async required |
| Demo/testing | Standard is fine |

---

## Summary

| Method | Pros | Cons |
|--------|------|------|
| **Standard** | Simple, no overhead | Jerky, idle frames |
| **Async** | Handles latency, distributed | Complex setup, still jerky |
| **RTC** | Smooth motion, simple API | Requires flow-matching policy |
| **Async + RTC** | Best of both worlds | Most complex |

**Bottom Line:**
- For SmolVLA on a decent GPU → **Use RTC** (`infer_smolvla_rtc.py`)
- For distributed setups → **Use Async + RTC**
- For quick tests → **Standard is fine** (`infer_smolvla_so101.py`)
