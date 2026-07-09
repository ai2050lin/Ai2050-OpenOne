# Phase269 MLP Continuation Writer Necessity Audit

- status: complete
- mlp_necessity_rows: 12
- causal_effect_rows: 12
- rollout_effect_rows: 12
- patch_counts: {"mlp_zero_last_token": 6, "mlp_half_last_token": 6}
- necessity_supported_counts: {"True": 8, "False": 4}
- winner_changed_counts: {"False": 10, "True": 2}
- rollout_changed_counts: {"True": 4, "False": 8}
- mean_delta_continue_stop_margin: -2.873698
- mean_delta_target_logit: -0.858337

Note: This is a small-scale causal necessity audit, not closure.
