# VLA Generalization Experiments

This folder contains diagnostic experiments to understand and improve VLA generalization capability.

**Related Documentation**: `/home/jrobot/project/robotAgent/docs/design/vla_generalization_proposal.md`

## Problem Statement

Our bimanual SmolVLA model exhibits poor generalization:
1. **Wrong target location**: "icecream → plate" still goes to bin (trained on "icecream → bin")
2. **Wrong object selection**: "pick tissue" picks corn when both visible (corn in training tasks)

## Root Causes (Hypothesized)

1. **Language embedding frozen** during training (`train_expert_only=true`)
2. **Visual-trajectory dominance** - images override language conditioning
3. **No object-language grounding** - can't link text to visual regions
4. **Trajectory-object coupling** - objects memorized with fixed target trajectories

---

## Phase 1 Experiments

### Experiment 1: Language Embedding Analysis

**Goal**: Validate that different task targets produce indistinguishable embeddings.

**Hypothesis**: If cosine similarity between "icecream → plate" and "icecream → bin" embeddings is > 0.95, language is under-conditioned.

```bash
cd /home/jrobot/project/lerobot

# Run embedding analysis
python jdocs/scripts/generalization/experiments/analyze_language_embeddings.py

# View results
ls jdocs/scripts/generalization/outputs/embedding_analysis/
# - embedding_similarity_heatmap.png
# - embedding_analysis.json
# - embedding_analysis.md
```

**Expected Outcome**:
- If similarity > 0.95 for different targets → **Root cause confirmed**
- If similarity < 0.90 → Embeddings are distinguishable, investigate attention

---

### Experiment 2: Unfreeze Language Encoder

**Goal**: Test if unfreezing language layers improves task discrimination.

**Hypothesis**: With trainable language embeddings, model can learn task-specific representations.

```bash
cd /home/jrobot/project/lerobot

# Train with unfrozen language (full run ~20k steps)
./jdocs/scripts/generalization/configs/train_unfrozen_language.sh

# Or quick test run (10k steps)
./jdocs/scripts/generalization/configs/train_unfrozen_language.sh 10000
```

**Key Changes from Baseline**:
| Parameter | Baseline | Experiment |
|-----------|----------|------------|
| `train_expert_only` | true | **false** |
| `learning_rate` | 1e-4 | **5e-5** |
| `batch_size` | 32 | **16** |
| `warmup_steps` | 1000 | **2000** |

**After Training**:
```bash
# Evaluate generalization
python jdocs/scripts/generalization/experiments/evaluate_generalization.py \
    --checkpoint outputs/smolvla_unfrozen_language_*/checkpoints/*/pretrained_model \
    --baseline outputs/smolvla_bimanual_20260103_200201/checkpoints/020000/pretrained_model
```

**Expected Outcome**:
- >30% improvement in cross-target tasks → Proceed to Phase 2
- <30% improvement → Consider alternative approaches (primitive decomposition)

---

### Experiment 3: Attention Visualization

**Goal**: Understand if model attends to target words ("plate", "bin") in task descriptions.

**Hypothesis**: Attention to target words is diffuse, so model ignores language.

```bash
cd /home/jrobot/project/lerobot

# Quick tokenization check (no model loading)
python jdocs/scripts/generalization/experiments/visualize_attention.py \
    --tokenize-only \
    --task "Use right arm to pick up the tissue and place it on the plate"

# Full attention analysis (requires checkpoint)
python jdocs/scripts/generalization/experiments/visualize_attention.py \
    --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/020000/pretrained_model \
    --image datasets_bimanuel/multitasks/videos/episode_000001/frame_0050.jpg \
    --task "Use right arm to pick up the tissue and place it on the plate"
```

**Output**:
- Attention summary plots per layer
- Token-wise attention statistics
- Target word attention analysis

---

## Evaluation Script

Evaluates any checkpoint on held-out task combinations:

```bash
python jdocs/scripts/generalization/experiments/evaluate_generalization.py \
    --checkpoint outputs/smolvla_unfrozen_language \
    --simulate
```

**Test Categories**:
1. **In-Distribution**: Tasks from training (sanity check)
2. **Cross-Target**: Same object, different target
3. **Multi-Object**: Correct selection with distractor
4. **Novel Composition**: New object+target combinations

---

## Directory Structure

```
jdocs/scripts/generalization/
├── README.md                           # This guide
├── experiments/
│   ├── analyze_language_embeddings.py  # Experiment 1
│   ├── visualize_attention.py          # Experiment 3
│   └── evaluate_generalization.py      # Evaluation script
├── configs/
│   └── train_unfrozen_language.sh      # Experiment 2
└── outputs/                            # Results stored here
    ├── embedding_analysis/
    ├── attention_analysis/
    └── evaluation/
```

---

## Decision Tree

```
                    Experiment 1
                    (Embedding Analysis)
                          |
                    Similarity > 0.95?
                     /          \
                   YES           NO
                    |             |
              Root cause       Investigate
              confirmed        attention
                    |             |
              Experiment 2    Experiment 3
              (Unfreeze LM)   (Attention Viz)
                    |
              Improvement > 30%?
               /          \
             YES           NO
              |             |
        Phase 2:       Consider:
        Primitives     - xVLA
                       - Soft prompts
                       - Object grounding
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Cross-target (icecream→plate) | ~10% | >70% |
| Multi-object selection | ~30% | >80% |
| Novel compositions | ~0% | >50% |

---

## Next Steps (After Phase 1)

1. **If Experiment 2 shows >30% improvement**:
   - Proceed to Phase 2: Primitive Action Decomposition
   - Implement episode segmentation with Qwen3-VL
   - Generate primitive-labeled dataset

2. **If Experiment 2 shows <30% improvement**:
   - Evaluate xVLA model (better language grounding)
   - Consider task-type soft prompts
   - Implement object detection preprocessing

---

## References

- [VLA Generalization Proposal](../../../robotAgent/docs/design/vla_generalization_proposal.md)
- [Bimanual SmolVLA Training](../bimanual/train_smolvla_bimanual.sh)
- [SmolVLA Architecture](../../src/lerobot/policies/smolvla/modeling_smolvla.py)
