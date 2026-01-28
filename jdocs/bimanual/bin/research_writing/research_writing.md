# Research Writing Resources

## Research Reports

### SmolVLA Hallucination Analysis
**Main Report**: `smolvla_hallucination_analysis.md`
- SmolVLA model architecture with ASCII diagrams
- Model introspection for hallucination behavior
- P(trajectory | KV_cache) visualization explanation
- Complete causal chain analysis

### Dataset Analysis
**Report**: `dataset_analysis_report.md`
- How training data was analyzed
- How `context_velocity_scatter_clean.png` was generated
- Scripts reference for diagram generation
- Connection to flow matching mechanism

## Diagrams

| Directory | Contents |
|-----------|----------|
| `logs/investigation/causal_distribution/` | P(traj \| KV_cache) visualizations |
| `logs/investigation/first_principles/` | Training distribution and flow matching diagrams |
| `logs/investigation/advanced_analysis_20260121/` | Advanced analysis visualizations |
| `logs/investigation/visual_evidence/` | Visual evidence comparison |
| `logs/investigation/trajectory_divergence/` | Trajectory comparison diagrams |

## Documentation

| Document | Description |
|----------|-------------|
| `jdocs/investigation_reports/first_principles_explanation.md` | Mathematical explanation of training bias |
| `jdocs/investigation_reports/flow_matching_explanation.md` | P(trajectory \| velocity_field) explanation |
| `jdocs/investigation_reports/visualization_guide.md` | How to read analysis diagrams |

## Data Collection

| Document | Description |
|----------|-------------|
| `jdocs/bimanual/bin/data_collection_strategy.md` | Strategy to fix hallucination via data collection |
| `jdocs/bimanual/bin/data_collection_checklist.md` | Checklist for data collection sessions |