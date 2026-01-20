# SmolVLA Hallucination Investigation: First-Principles Research Implementation Plan

**Created**: 2026-01-20
**Status**: Implementation Phase

## Executive Summary

This document provides the detailed implementation plan for investigating SmolVLA hallucination through first-principles analysis. The investigation aims to answer: **Why does the VLM produce "stay still" in normal cases but "move toward empty space" in the hallucination case, when BOTH have completed the same task?**

---

## Existing Infrastructure

### Collected Data (3 Cases)
```
logs/yogurt_banana_leftarm/
├── case_20260119_131914_ha_bana_table/     # HALLUCINATION: banana on table
├── case_20260119_132946_no_ha_plate/       # NORMAL: banana on plate (far)
└── case_20260119_133142_no_ha_no_other_obj/ # NORMAL: no distractors
```

Each case contains:
- `trace.jsonl`: Per-step state, action, timing data (465 steps)
- `metadata.json`: Run configuration
- `images/`: Every 50 steps, 3 cameras (head, left_wrist, right_wrist)

### Model Architecture Reference
```
TOKEN LAYOUT (VERIFIED):
[0-63]     Head camera patches (8x8 grid)
[64-127]   Left wrist camera patches (8x8 grid)
[128-191]  Right wrist camera patches (8x8 grid)
[192-239]  Language tokens (~48)
[240]      State token
─────────────────────────────────────────
TOTAL:     241 prefix tokens
```

### Key Code Locations
- **Denoising loop**: `modeling_smolvla.py:809-841`
- **KV cache generation**: `modeling_smolvla.py:797-804`
- **Cross-attention**: `smolvlm_with_expert.py:575`
- **Prefix embedding**: `modeling_smolvla.py:791-793`

---

## Implementation Plan

### Phase A: Visual and Embedding Analysis (Offline - No Inference Required)

#### Tool 1: `vision_feature_comparison.py`

**Purpose**: Compare SigLIP encoder visual features between halluc and normal cases at step 200.

**Implementation**:
```python
# Pseudocode structure
class VisionFeatureComparison:
    def __init__(self, checkpoint_path: str):
        # Load SmolVLA model (only need SigLIP encoder)
        self.model = load_smolvla(checkpoint_path)

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract SigLIP features from image."""
        # Load image, preprocess to 512x512
        # Forward through vision encoder
        # Return feature tensor [64, hidden_dim] per camera

    def compare_cases(self, case_dirs: List[str], step: int = 200):
        """Compare visual features between cases."""
        # For each case, load 3 camera images at step 200
        # Extract features
        # Compute:
        #   - Cosine similarity between cases per camera
        #   - L2 distance per camera
        #   - PCA visualization of patch embeddings
        #   - Spatial heatmaps showing where features differ most
```

**Output**:
```
outputs/vision_feature_comparison/
├── feature_similarity_matrix.json       # Cosine sim between all case pairs
├── per_camera_comparison.json           # Per-camera breakdown
├── pca_visualization.png                # 2D PCA of patch embeddings
├── spatial_difference_heatmaps/         # Where features differ per camera
│   ├── head_camera_diff.png
│   ├── left_wrist_diff.png
│   └── right_wrist_diff.png
└── report.md
```

**Key Questions Answered**:
- Does right wrist camera (where banana is visible) have largest feature difference?
- Are features similar enough that model should recognize "same scene type"?

---

#### Tool 2: `prefix_embedding_analysis.py`

**Purpose**: Analyze the combined prefix embedding (vision + language + state) differences.

**Implementation**:
```python
class PrefixEmbeddingAnalysis:
    def __init__(self, checkpoint_path: str):
        self.model = load_smolvla(checkpoint_path)

    def extract_prefix_embedding(self, images: List[Tensor],
                                  lang_tokens: Tensor,
                                  state: Tensor) -> Tensor:
        """Extract full prefix embedding after embed_prefix()."""
        # Call model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        # Return prefix_embs [batch, 241, hidden_dim]

    def analyze_token_differences(self, halluc_emb: Tensor, normal_emb: Tensor):
        """Per-token L2 distance analysis."""
        # Compute L2 distance for each of 241 tokens
        # Group by region:
        #   - Head camera [0:64]
        #   - Left wrist [64:128]
        #   - Right wrist [128:192]
        #   - Language [192:240]
        #   - State [240]
        # Identify which region has largest difference
```

**Output**:
```
outputs/prefix_embedding_analysis/
├── per_token_l2_distance.json           # 241 values
├── region_summary.json                  # Aggregated by region
├── token_distance_heatmap.png           # Visual heatmap
├── pca_embedding_comparison.png         # PCA of full embeddings
└── report.md
```

**Key Questions Answered**:
- Do vision tokens dominate the difference vs language/state?
- Within vision, which camera contributes most?
- Is the difference localized or distributed?

---

### Phase B: Model Hook Integration (Requires Inference)

#### Tool 3: `kv_cache_content_analysis.py`

**Purpose**: Quantify how KV cache content differs between cases.

**Implementation**:
```python
class KVCacheContentAnalysis:
    def __init__(self, checkpoint_path: str):
        self.model = load_smolvla(checkpoint_path)
        self.kv_cache_captures = {}

    def register_kv_capture_hook(self):
        """Hook to capture past_key_values after fill_kv_cache=True."""
        # Wrap vlm_with_expert.forward to capture past_key_values
        # past_key_values structure: List[Tuple[key_states, value_states]]
        # Each: [batch, num_heads, seq_len, head_dim]

    def analyze_kv_cache(self, case_name: str, past_key_values):
        """Analyze KV cache by layer and token region."""
        # For each of 16 layers:
        #   - key_states shape: [batch, 16, 241, 72] (16 heads, 72 head_dim)
        #   - value_states shape: same
        # Compute per-region norms and statistics
        # Identify which layers/regions differ most
```

**Output**:
```
outputs/kv_cache_analysis/
├── layer_comparison.json                # Per-layer KV statistics
├── region_by_layer_heatmap.png          # [16 layers x 5 regions] heatmap
├── key_value_norms.json                 # K,V norms per layer
├── most_different_layers.json           # Top 3 layers with largest diff
└── report.md
```

**Key Questions Answered**:
- Is KV difference concentrated in specific layers?
- Which token region has largest KV difference?
- Does early vs late layers show more divergence?

---

#### Tool 4: `denoising_step_analysis.py`

**Purpose**: Identify exactly when RAMP vs FLAT trajectory emerges during denoising.

**Implementation**:
```python
class DenoisingStepAnalysis:
    def __init__(self, checkpoint_path: str):
        self.model = load_smolvla(checkpoint_path)
        self.step_captures = []  # List of {step, x_t, v_t, metrics}

    def register_denoising_hooks(self):
        """Hook into denoising loop to capture at each of 10 steps."""
        # Wrap the denoising loop (modeling_smolvla.py:809-841)
        # At each step capture:
        #   - x_t: current noisy action [batch, 50, 32]
        #   - v_t: velocity prediction [batch, 50, 32]
        #   - Computed metrics:
        #     * action_delta: |x_t[end] - x_t[start]| per joint
        #     * trajectory_shape: FLAT, RAMP_UP, RAMP_DOWN
        #     * smoothness: L2 norm of consecutive diffs
        #     * velocity_magnitude: ||v_t||

    def compare_trajectory_emergence(self, halluc_captures, normal_captures):
        """Compare when trajectory shape emerges."""
        # For each denoising step 0-9:
        #   - Compare trajectory shape between cases
        #   - Identify step where shapes diverge
        # Plot action delta vs denoising step
```

**Output**:
```
outputs/denoising_analysis/
├── step_by_step_comparison.json         # Metrics at each denoising step
├── trajectory_emergence_plot.png        # Action delta vs denoising step
├── velocity_field_comparison.png        # v_t visualization per step
├── shape_divergence_step.json           # When does RAMP emerge?
├── joint_evolution_grid.png             # [10 steps x 6 joints] grid
└── report.md
```

**Key Questions Answered**:
- Does RAMP emerge at step 0 (KV cache driven) or later?
- At which step does halluc case "lock in" to wrong trajectory?
- Does velocity field point different directions from step 0?

---

#### Tool 5: `cross_attention_per_denoising_step.py`

**Purpose**: Track how cross-attention patterns change across denoising steps.

**Implementation**:
```python
class CrossAttentionPerStep:
    def __init__(self, checkpoint_path: str):
        self.model = load_smolvla(checkpoint_path)
        self.attention_by_step = {}  # {step: attention_weights}

    def register_attention_hook(self):
        """Hook into eager_attention_forward to capture at each step."""
        # Use existing pattern from cross_attention_capture.py
        # Track which denoising step we're in
        # Capture attention weights [batch, heads, 50, 241]

    def analyze_per_step_attention(self):
        """Break down attention by denoising step."""
        # For each step 0-9:
        #   - Compute per-region attention (head, left, right, lang, state)
        #   - Track attention entropy
        #   - Identify if attention shifts during denoising
```

**Output**:
```
outputs/cross_attention_per_step/
├── attention_evolution.json             # Per-step attention breakdown
├── attention_heatmap_evolution.gif      # Animated heatmap across steps
├── region_attention_vs_step.png         # Line plot of region attention
├── entropy_vs_step.png                  # Attention entropy per step
└── report.md
```

**Key Questions Answered**:
- Does attention to right wrist camera increase during denoising in halluc case?
- Is attention pattern different from step 0, or does it develop?
- Which layer shows most divergent attention patterns?

---

### Phase C: Dataset Distribution Analysis

#### Tool 6: `trajectory_distribution_visualization.py`

**Purpose**: Visualize training trajectory space and where hallucination trajectory lies.

**Implementation**:
```python
class TrajectoryDistributionVisualization:
    def __init__(self, dataset_path: str):
        self.dataset = load_lerobot_dataset(dataset_path)

    def extract_trajectories(self, task_filter: str = "yogurt"):
        """Extract all trajectories for task from training data."""
        # Filter episodes by task description
        # Extract action sequences [N_episodes, T_steps, D_action_dim]
        # Return as numpy array

    def embed_trajectories(self, trajectories: np.ndarray) -> np.ndarray:
        """Embed trajectories for visualization."""
        # Flatten: [N, T, D] -> [N, T*D]
        # Apply PCA to reduce to 3D
        # Return [N, 3] coordinates

    def segment_by_phase(self, trajectories: np.ndarray) -> List[str]:
        """Segment trajectories into phases."""
        # Phases: APPROACH, GRIP, TRANSPORT, RELEASE, IDLE
        # Use heuristics based on gripper state and velocity

    def visualize_3d(self, embeddings: np.ndarray, phases: List[str],
                     halluc_embedding: np.ndarray, normal_embedding: np.ndarray):
        """Create 3D visualization."""
        # Plot training trajectories as point cloud
        # Color by phase
        # Overlay halluc and normal cases with distinct markers
        # Identify which cluster halluc resembles
```

**Output**:
```
outputs/trajectory_distribution/
├── trajectory_embeddings.npy            # PCA embeddings
├── phase_labels.json                    # Phase for each trajectory
├── 3d_visualization.html                # Interactive plotly 3D plot
├── pca_variance_explained.json          # How much variance each PC captures
├── nearest_training_trajectories.json   # Which training traj is halluc closest to
└── report.md
```

**Key Questions Answered**:
- Does halluc trajectory fall in a valid training cluster?
- Which training phase does the post-completion halluc resemble?
- Is halluc trajectory out-of-distribution?

---

## Execution Order

### Step 1: Phase A - Offline Analysis
```bash
# Tool 1: Vision feature comparison
python vision_feature_comparison.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --step 200 \
    --output-dir outputs/vision_feature_comparison

# Tool 2: Prefix embedding analysis
python prefix_embedding_analysis.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
    --case-dirs <same as above> \
    --step 200 \
    --output-dir outputs/prefix_embedding_analysis
```

### Step 2: Phase B - Model Hooks (Live Inference)
```bash
# Tool 3: KV cache analysis
python kv_cache_content_analysis.py \
    --checkpoint <path> \
    --task "Use left arm to pick up the yogurt bottle and place it in the bin" \
    --capture-step 200 \
    --output-dir outputs/kv_cache_analysis

# Tool 4: Denoising step analysis
python denoising_step_analysis.py \
    --checkpoint <path> \
    --task <same> \
    --capture-step 200 \
    --output-dir outputs/denoising_analysis

# Tool 5: Cross-attention per step
python cross_attention_per_denoising_step.py \
    --checkpoint <path> \
    --task <same> \
    --capture-step 200 \
    --output-dir outputs/cross_attention_per_step
```

### Step 3: Phase C - Dataset Analysis
```bash
# Tool 6: Trajectory distribution
python trajectory_distribution_visualization.py \
    --dataset datasets_bimanuel/multitasks \
    --task-filter "yogurt" \
    --halluc-trace logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl \
    --normal-trace logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj/trace.jsonl \
    --output-dir outputs/trajectory_distribution
```

### Step 4: Synthesis
Compile all findings into:
- `hallucination_mechanism_findings.md`
- `trajectory_distribution_analysis.md`
- `hallucination_root_cause_synthesis.md`

---

## Success Criteria

### Part 1 (Mechanism):
- [ ] Identify EXACT pipeline stage where divergence occurs
- [ ] Quantify magnitude of difference at that stage
- [ ] Explain WHY that difference leads to different action output

### Part 2 (Dataset):
- [ ] Visualize training trajectory distribution
- [ ] Show where halluc trajectory lies in that distribution
- [ ] Identify what training patterns correlate with hallucination

### Part 3 (Synthesis):
- [ ] Complete causal chain from visual input → action output
- [ ] Link dataset distribution to model behavior
- [ ] Generate publishable research findings

---

## Commit Strategy

1. **Commit 1**: Implementation plan document (this file)
2. **Commit 2**: Phase A tools (vision_feature_comparison.py, prefix_embedding_analysis.py)
3. **Commit 3**: Phase B tools (kv_cache_content_analysis.py, denoising_step_analysis.py, cross_attention_per_denoising_step.py)
4. **Commit 4**: Phase C tools (trajectory_distribution_visualization.py)
5. **Commit 5**: Phase A results and findings
6. **Commit 6**: Phase B results and findings
7. **Commit 7**: Phase C results and findings
8. **Commit 8**: Final synthesis documents

---

## Appendix: Research References

### Trajectory Visualization
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/) - PCA visualization of trajectory space
- [Motion2Vec](https://sites.google.com/view/motion2vec) - t-SNE/PCA for action segmentation
- [FlowPolicy](https://arxiv.org/html/2412.04987v2) - Structure emergence during denoising

### Flow Matching
- [Action Coherence Guidance](https://arxiv.org/pdf/2510.22201) - Action coherence in flow-based VLAs
- [π0 and π0-FAST](https://huggingface.co/blog/pi0) - Flow matching for smooth trajectories

### Interpretability
- [Mechanistic Interpretability for VLAs](https://vla-mech-interp.github.io/) - Activation steering
- [AttnLRP](https://arxiv.org/html/2402.05602v2) - Faithful attention attribution
