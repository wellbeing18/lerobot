# Pi0.5 “arm swings in the air” investigation (SO-101 follower)

Date: 2025-12-25  
Repo: `lerobot`  
Subject: Pi0.5 fine-tune on `datasets/pick_and_place` produces unstable inference on real robot: arm swings up/down and does not perform task.

---

## 1) Executive summary

Two independent inference traces show the same failure mode:

- **MEAN_STD run**: `outputs/inference_traces/trace_pi05_20251225_105101`
- **QUANTILES run**: `outputs/inference_traces/trace_pi05_20251224_124451`

In both traces, the policy produces **highly discontinuous absolute joint targets** (large step-to-step target changes) which, when sent at 20–30 Hz, manifests as the arm “swinging in the air”.

**Conclusion:** this does **not** appear to be primarily caused by “MEAN_STD vs QUANTILES”; QUANTILES also swings. The traces point to **execution/config mismatch and/or missing safety/clamping** (and possibly a units mismatch: degrees vs normalized range) as the most likely cause of the violent motion.

---

## 2) How the inference loop works (what is input → output → executed → feedback)

The trace is produced by `jdocs/scripts/infer_pi05_rtc_trace.py`, which uses a background inference thread and a background execute thread.

### 2.1 Inputs to the policy per inference

On each inference trigger (when the action queue is low), the inference thread collects:

- **Robot state**: 6D joint vector from `SO101Follower.get_observation()` fields:
  - `shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`
- **Two camera images** (RGB frames):
  - `observation.images.head`
  - `observation.images.left_wrist`
- **Task text**: `"pick up the block and place it on the plate"` (stored under `task`)

These are formatted by `format_observation(...)` into:

- `observation.state`: tensor `[1,6]`
- `observation.images.head`: tensor `[1,3,480,640]` in `[0,1]`
- `observation.images.left_wrist`: tensor `[1,3,480,640]` in `[0,1]`
- `task`: list[str] with length 1

### 2.2 What the policy outputs

The policy outputs a **chunk** of actions (configured as `chunk_size=50`), then postprocessing produces a 50×6 action buffer.

In `trace.jsonl`:

- `action_buffer`: the full 50×6 chunk (postprocessed)
- `action_executed`: the single 6D action popped and sent at that step

### 2.3 What is executed on the robot

The execute thread sends:

- `action_dict = {f"{name}.pos": float(action[i])}` (absolute targets)  
  then calls `SO101Follower.send_action(action_dict)`.

The trace records:

- `joint_states`: current measured state before sending
- `action_executed`: commanded absolute target
- `action_delta = action_executed - joint_states`

This `action_delta` is a direct indicator of how aggressive the command is.

### 2.4 Feedback used for next step

The next inference uses:

- fresh images + current `robot.get_state()`
- plus RTC “leftover chunk” blending (if enabled)

The trace also records `state_after_inference` and `state_drift_during_inference` to quantify how much state changed during inference latency.

---

## 3) Evidence from traces

### 3.1 MEAN_STD trace: `trace_pi05_20251225_105101`

Key observations:

- Control loop was **20 Hz** (`config.json` shows `fps: 20.0`).
- Inference latency is ~300ms typical; first inference incurred ~1.5s latency (capture spike).
- The first executed actions produced **very large deltas** because the initial robot pose was extreme and the chunk commanded a very different pose.

This explains “big initial lurch”, but does not explain persistent swinging by itself.

### 3.2 QUANTILES trace: `trace_pi05_20251224_124451`

Key observations (from `analysis_report.json`):

- Control loop is **30 Hz**.
- 14 inferences over 15 seconds.
- **34 `action_jumps`** detected where max per-joint delta is ~30–45° (sometimes more).

This matches the user-visible symptom: **rapid, repeated up/down motion**.

Example segment (around steps ~160–168 in `trace.jsonl`) shows elbow and shoulder targets jumping tens of degrees step-to-step. Regardless of whether the policy “knows the task”, sending these targets at 30 Hz will look like violent oscillation.

**Crucially:** QUANTILES also exhibits the same instability. Therefore changing normalization alone is unlikely to fix the core issue.

---

## 4) Most likely root causes (ranked)

### Hypothesis A (high): Missing safety clamping / config parity between recording and inference

`SO101Follower.send_action` supports safety clamping with `max_relative_target`:

- If `max_relative_target` is set, it caps how far the commanded goal can be from the present position per step.
- This prevents violent oscillation even if targets are noisy.

However, the inference script’s robot wrapper constructs `SO101FollowerConfig` with only `port` and `id` (and does not propagate `max_relative_target` from YAML).

**Why this can cause “swinging”:**

- If the policy outputs absolute targets that are sometimes inconsistent step-to-step (common early in finetuning), lacking a per-step clamp turns those inconsistencies into large physical motion.

**Validation experiment:**

- Ensure inference actually uses a non-null `max_relative_target` and re-run trace.
- If the swing becomes bounded/less violent, this is a major contributor.

### Hypothesis B (high): Units mismatch (degrees vs RANGE_M100_100)

Robot config supports `use_degrees`:

- If `use_degrees=true`, motors interpret inputs in **degrees**
- If `use_degrees=false`, motors interpret inputs in **RANGE_M100_100**

The inference hardware YAML (`jdocs/scripts/so101_hardware.yaml`) sets:

- `robot.use_degrees: true`

But the dataset stats (`datasets/pick_and_place/meta/stats.json`) show hard bounds at ±100 for joints, which is consistent with a **normalized range** representation, not physical degrees for all joints.

**If training data is in RANGE but inference sends degrees (or vice versa), actions will be systematically wrong** and can easily create oscillations/jumps.

**Validation experiments:**

- Confirm what mode was used during dataset collection for the follower’s recorded `.pos` values.
- Compare typical recorded values vs expected mechanical degree ranges.
- Run inference with `use_degrees` aligned to the dataset’s semantics.

### Hypothesis C (medium): RTC chunking + latency skip amplifies discontinuities

Both traces show sizable latency and corresponding inference delay steps. RTC logic skips some portion of the freshly predicted chunk:

- `skip = consumed + inference_delay`

If the policy is unstable, repeatedly skipping into different parts of chunks could make behavior look more “jerky”.

**Validation experiments:**

- Run with `--no-rtc`.
- Lower FPS and/or change `action_queue_threshold` to trigger inference more/less frequently.

### Hypothesis D (medium/low): Policy collapsed / insufficient training signal

It’s possible the policy has not learned a stable controller and outputs noisy absolute targets. However:

- The joint targets are not random noise; they often stay in plausible ranges.
- The “swing” can be explained by discontinuities + lack of safety clamp + units mismatch.

So treat this as a secondary hypothesis unless A/B are ruled out.

---

## 5) Concrete “next actions” checklist (no code changes required to start)

### 5.1 Verify what the dataset action/state represent

From `datasets/pick_and_place/meta/info.json`:

- `action` is 6D joint `.pos` with names matching the follower motors.
- `observation.state` has same motor order and names.

The key question: are those `.pos` values **degrees** or **RANGE_M100_100**?

**Recommendation:** sample a few recorded frames and inspect typical values for each joint. If values cluster around ±100 and not realistic degree ranges, assume RANGE.

### 5.2 Verify the inference robot config matches dataset semantics

- Ensure `use_degrees` in inference matches the dataset’s semantics.
- Ensure `max_relative_target` is applied during inference (for safety and for matching “sent action saved in dataset” behavior).

### 5.3 Compare “action delta distribution” in dataset vs inference

Compute:

- Dataset: per-step deltas of recorded `action` (or of `observation.state`).
- Inference: per-step deltas already recorded as `action_delta`.

If dataset deltas are small but inference deltas are large, the issue is execution mismatch or model instability rather than perception.

---

## 6) Notes / limitations

- The trace currently stores images, but this report does not include visual inspection of frames; it focuses on the action/state dynamics visible directly in `trace.jsonl` + `analysis_report.json`.
- This report assumes the trace script’s logging is accurate and the robot returns real states (i.e., not “mock mode”).

---

## 7) Artifacts referenced

- QUANTILES trace: `outputs/inference_traces/trace_pi05_20251224_124451/`
  - `trace.jsonl`, `analysis_report.json`, `summary.json`, `joint_trajectories.png`, `action_deltas.png`
- MEAN_STD trace: `outputs/inference_traces/trace_pi05_20251225_105101/`
  - `trace.jsonl`, `analysis_report.json`, `summary.json`, `joint_trajectories.png`, `action_deltas.png`
- Dataset metadata/stats:
  - `datasets/pick_and_place/meta/info.json`
  - `datasets/pick_and_place/meta/stats.json`
  - `datasets/pick_and_place/meta/config.yaml`






