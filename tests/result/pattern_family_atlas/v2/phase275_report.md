# Phase275 Selected Gap Batch Physical Path Fill

- status: complete
- component_summary_rows: 45
- causal_fill_rows: 90
- missing_rows: 0
- model_counts: {"qwen3": 17, "glm4": 13, "deepseek7b": 15}
- family_counts: {"closure": 5, "cross_lingual": 5, "state_drift": 6, "reasoning_constraint": 5, "output_protocol": 4, "language_action": 6, "readout_competition": 5, "content_knowledge": 4, "syntax_structure": 5}
- dominant_positive_component_counts: {"mlp": 42, "attention": 3}
- causal_effect_supported_counts: {"True": 59, "False": 31}
- side_effect_risk_counts: {"False": 50, "True": 40}

This phase consumes Phase274 selected gap rows. It is a physical-path fill batch, not closure.
