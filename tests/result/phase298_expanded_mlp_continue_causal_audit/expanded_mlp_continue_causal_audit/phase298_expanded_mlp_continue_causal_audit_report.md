# Phase298 Expanded MLP Continue Causal Audit

- status: complete
- selected_mlp_dominant_cases: 24
- audit_rows: 72
- causal_effect_rows: 72
- rollout_rows: 72
- missing_rows: 0
- patch_counts: {"mlp_zero_last_token": 24, "mlp_quarter_last_token": 24, "mlp_half_last_token": 24}
- necessity_supported_counts: {"True": 48, "False": 24}
- winner_changed_counts: {"False": 72}
- causal_support_level_counts: {"weak": 48, "not_supported": 24}
- rollout_changed_counts: {"True": 46, "False": 26}
- mean_delta_continue_stop_margin: -2.700304
- mean_delta_target_logit: -1.579861

This is a low-side-effect causal audit on expanded Phase296 MLP-dominant samples, not closure.
