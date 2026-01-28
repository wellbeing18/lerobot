# Dataset Fix Report: `picknplace_300_144`

**Date:** January 28, 2026
**Dataset:** `datasets_bimanuel/picknplace_300_144`
**Source Datasets:**
1.  `datasets_bimanuel/multitasks` (Episodes 0–319)
2.  `datasets_bimanuel/pairs20260126` (Episodes 320–463)

## Objective
Double-check the integrity of the combined dataset `picknplace_300_144` and ensure it is ready for Hugging Face upload and cloud training.

## Findings

### 1. Task Consistency (Pass)
*   **Verification:** Checked `meta/tasks.parquet` and task indices in the data files.
*   **Result:** The tasks from the source datasets were correctly merged. Specifically, tasks from the `pairs` dataset were correctly remapped to match the global task indices of the combined dataset.
    *   *Example:* "Pick up banana" (Task 0 in `pairs`) is correctly mapped to Task 6 in the combined dataset.

### 2. Data File Structure (Pass)
*   **Verification:** Inspected `data/chunk-000/`.
*   **Result:** The parquet data files are correctly chunked.
    *   `file-000` to `file-015`: Contain `multitasks` episodes (0–319).
    *   `file-016` to `file-023`: Contain `pairs` episodes (320–463).

### 3. Metadata Issues (Critical Fail -> Fixed)
*   **Issue:** The `meta/episodes` metadata file (stored as parquet) had incorrect file pointers for the appended episodes (indices 320+).
    *   They pointed to file indices `0` through `143` (likely preserving their original indices from the `pairs` dataset) instead of the new chunk locations `16` through `23`.
    *   **Impact:** This would have caused data loading errors or loaded incorrect data segments during training.
*   **Fix:** Updated the `data/file_index` column in `meta/episodes/chunk-000/file-000.parquet` for episodes >= 320.
    *   Logic: `new_file_index = 16 + (episode_index - 320) // 20`

### 4. Video Files (Critical Fail -> Fixed)
*   **Issue:** The video files for the appended episodes were not merged into the chunk structure.
    *   The `multitasks` part had chunked videos (`file-000.mp4` to `file-015.mp4`).
    *   The `pairs` part existed as individual episode files (`file-320.mp4` to `file-463.mp4`) in the video directories.
    *   **Impact:** Data loaders expecting the chunked structure defined in `meta/info.json` would fail to find the video files.
*   **Fix:** Concatenated the individual video files into chunks (`file-016.mp4` to `file-023.mp4`) to match the data file structure.
    *   Applied to all camera views: `observation.images.head`, `observation.images.left_wrist`, `observation.images.right_wrist`.
    *   Removed the old individual video files after successful concatenation.

## Conclusion
The dataset `datasets_bimanuel/picknplace_300_144` has been patched and verified.

*   **Total Episodes:** 464
*   **Total Data Chunks:** 24 (Files 000–023)
*   **Status:** **READY** for Hugging Face upload and cloud training.
