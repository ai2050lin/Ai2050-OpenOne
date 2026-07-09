# Phase279 Phase278 Next Gap Batch Physical Path Fill

- status: complete
- component_summary_rows: 48
- causal_fill_rows: 96
- missing_rows: 0
- model_counts: {"qwen3": 17, "glm4": 15, "deepseek7b": 16}
- family_counts: {"closure": 5, "state_drift": 6, "cross_lingual": 6, "output_protocol": 4, "reasoning_constraint": 6, "language_action": 6, "readout_competition": 6, "content_knowledge": 4, "syntax_structure": 5}
- dominant_positive_component_counts: {"mlp": 46, "attention": 2}
- causal_effect_supported_counts: {"True": 71, "False": 25}
- side_effect_risk_counts: {"True": 55, "False": 41}

This phase consumes Phase278 next_batch_rows. It is physical-path fill, not closure.
