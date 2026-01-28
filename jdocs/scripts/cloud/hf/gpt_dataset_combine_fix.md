# Dataset Combine Double-Check Report (GPT): `picknplace_300_144`

**Date:** January 28, 2026  
**Dataset:** `datasets_bimanuel/picknplace_300_144`  
**Claimed Sources:**
1. `datasets_bimanuel/multitasks` (expected episodes 0–319)
2. `datasets_bimanuel/pairs20260126` (expected episodes 320–463)

## Objective
Double-check that the combined dataset is internally consistent and **ready for (a) Hugging Face upload** and **(b) cloud training**.

## What I checked (no code execution, file inspection only)
- Dataset folder structure: `data/`, `videos/`, `meta/`
- `meta/info.json`, `meta/stats.json`, `meta/tasks.jsonl`
- Spot-checked `meta/episodes.jsonl` (early, middle, and boundary episodes)
- Compared source dataset totals from `multitasks/meta/info.json` and `pairs20260126/meta/info.json`

## Findings

### 1) Source totals add up cleanly (PASS)
The headline episode/frame totals in the combined dataset match the sources exactly:
- `multitasks`: **320 episodes**, **76,597 frames**
- `pairs20260126`: **144 episodes**, **42,175 frames**
- Combined `picknplace_300_144`: **464 episodes**, **118,772 frames**

This is a strong sign the intended merge size is correct.

### 2) Combined dataset has the expected on-disk chunk layout (PASS)
The combined dataset only contains:
- `data/chunk-000/file-000.parquet` … `file-023.parquet` (**24 files total**)
- `videos/.../chunk-000/file-000.mp4` … `file-023.mp4` (**24 files per camera view**)

This matches the typical “chunked videos + chunked parquet” layout used for LeRobot v3 datasets.

### 3) **Critical: `episodes.jsonl` points to non-existent data/video files for the appended `pairs` portion (FAIL)**
In `datasets_bimanuel/picknplace_300_144/meta/episodes.jsonl`, episodes in the appended portion still reference **original `pairs20260126` file indices** (0–143), but the combined dataset only has files 0–23.

Concrete examples:
- Episode **320** (first appended episode) still points to `data/file_index: 0` and `videos/.../file_index: 0` (i.e., `pairs`-style per-episode files), while the combined dataset expects those episodes to live in the chunked range `file-016..file-023` if it was repacked.
- Episode **463** (last episode) points to `data/file_index: 143` and `videos/.../file_index: 143`, but `picknplace_300_144` has **no** `file-143.parquet` or `file-143.mp4`.

**Impact (Hugging Face upload):**
- HF upload in this repo’s guide explicitly expects `meta/episodes.jsonl` to reflect the dataset’s true on-disk file layout. With stale file pointers, downstream users (and some tooling) can fail or load incorrect slices.

**Impact (cloud training):**
- Cloud training will rely on consistent `meta/*` pointers. If `meta/episodes` (parquet) mirrors the same pointers (likely, since `episodes.jsonl` is usually derived from it), training will fail to locate data/videos for episodes 320–463 or will slice the wrong file.

### 4) **Critical: Task index remapping appears incomplete/inconsistent for appended episodes (LIKELY FAIL)**
At the boundary episode **320**, the line shows:
- `tasks`: `"Use left arm to pick up the banana and place it on the plate"`
- but per-episode stats show `stats/task_index/min = 0` and `max = 0`

In the combined dataset’s `tasks.jsonl`, **task_index 0** corresponds to:
- `"Use left arm to pick up the orange and place it on the plate"`

So episode 320 appears to carry a `pairs`-local `task_index` (banana as 0) while the combined dataset defines task 0 as orange.

**Impact:**
- Even if data loading “works”, multitask training would be learning with **incorrect task labels** for at least some appended episodes.

### 5) `meta/config.yaml` is misleading for the combined dataset (NOT BLOCKING, but confusing)
`picknplace_300_144/meta/config.yaml` currently identifies:
- `experiment.name: xlerobot_bimanual_multitasks`
- `dataset_name: multitasks`
- and a single `task_string` (not representative of a combined multitask dataset)

This doesn’t block training, but it’s confusing when publishing to HF and when you later audit provenance.

### 6) Split definition is train-only (OK, but consider adding val/test later)
`meta/info.json` defines:
- `splits.train: "0:464"`

This is acceptable (train-only datasets are common), but for cloud training you may want an explicit eval split for consistent reporting.

### 7) Video codec is AV1 (potential environment pitfall)
`meta/info.json` reports `"video.codec": "av1"`.

This is fine if your cloud image has FFmpeg with AV1 decode enabled, but it can be a surprise on minimal CUDA images. Worth verifying early.

## Readiness Verdict
- **Cloud training readiness:** **NOT READY** (high risk of missing/incorrect file pointers + task indices for episodes 320–463)
- **Hugging Face upload readiness:** **NOT READY** (stale `episodes.jsonl` pointers will publish a broken manifest)

## Suggested fix path (what to do next)
These are suggestions only (I did not change anything).

### A) Make the metadata consistent with the actual chunked files
You likely need to regenerate or patch **both**:
- `meta/episodes/chunk-000/file-000.parquet` (the authoritative episode table)
- `meta/episodes.jsonl` (HF-facing manifest)

For appended episodes (320–463), the goal is:
- `data/file_index` must refer to **0–23** (the files that exist in `picknplace_300_144/data/chunk-000/`)
- `videos/.../file_index` must refer to **0–23** (the files that exist in `picknplace_300_144/videos/.../chunk-000/`)
- `dataset_from_index` / `dataset_to_index` must be consistent with how frames are stored inside each parquet file

### B) Ensure task indices are globally correct (critical for multitask training)
For the `pairs` portion, remap per-frame `task_index` values so they align with the combined dataset’s `meta/tasks.jsonl` indices.

At minimum, verify that for boundary episodes (e.g., 320, 463):
- `task_index` values in the data match the intended task string

### C) Sanity-check with a minimal load test (recommended before upload/training)
Before uploading to HF or starting a long cloud run, do a quick local sanity check:
- load the dataset
- fetch episode 0, 319, 320, 463
- verify the referenced parquet + mp4 exists and decodes
- verify the `task_index` matches the task string

### D) HF upload practical checklist
- Track `.parquet` and `.mp4` via **Git LFS** (or use the provided upload script in this repo)
- Add/verify dataset card (README) with:
  - source provenance (`multitasks` + `pairs20260126`)
  - robot/camera setup
  - action/state definitions
  - license/consent notes (if applicable)

### E) Cloud training practical checklist
- Ensure FFmpeg AV1 decode is present (or plan to re-encode to H.264 later if needed)
- Consider adding an eval split (even a small one) for stable metrics over time

