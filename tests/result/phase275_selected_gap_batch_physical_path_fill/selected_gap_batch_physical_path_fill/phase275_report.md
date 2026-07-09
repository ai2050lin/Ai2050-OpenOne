# Phase275 Selected Gap Batch Physical Path Fill

- status: complete
- component_summary_rows: 9
- causal_fill_rows: 18
- missing_rows: 0
- model_counts: {"qwen3": 3, "glm4": 3, "deepseek7b": 3}
- family_counts: {"closure": 2, "cross_lingual": 3, "state_drift": 2, "content_knowledge": 1, "reasoning_constraint": 1}
- dominant_positive_component_counts: {"mlp": 8, "attention": 1}
- causal_effect_supported_counts: {"True": 8, "False": 10}
- side_effect_risk_counts: {"False": 9, "True": 9}

This phase consumes Phase274 selected gap rows. It is a physical-path fill batch, not closure.
