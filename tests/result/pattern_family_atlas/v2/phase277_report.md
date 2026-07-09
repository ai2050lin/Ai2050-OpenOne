# Phase277 Phase276 Next Gap Batch Physical Path Fill

- status: complete
- component_summary_rows: 49
- causal_fill_rows: 98
- missing_rows: 0
- model_counts: {"qwen3": 17, "glm4": 15, "deepseek7b": 17}
- family_counts: {"closure": 5, "state_drift": 6, "cross_lingual": 6, "reasoning_constraint": 6, "output_protocol": 5, "language_action": 6, "readout_competition": 6, "content_knowledge": 4, "syntax_structure": 5}
- dominant_positive_component_counts: {"mlp": 45, "attention": 4}
- causal_effect_supported_counts: {"True": 55, "False": 43}
- side_effect_risk_counts: {"True": 45, "False": 53}

This phase consumes Phase276 next_batch_rows. It is physical-path fill, not closure.
