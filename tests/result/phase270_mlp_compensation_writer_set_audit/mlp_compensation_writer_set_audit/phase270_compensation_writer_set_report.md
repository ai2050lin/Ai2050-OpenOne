# Phase270 MLP Compensation Writer Set Audit

- status: complete
- compensation_rows: 30
- writer_set_rows: 6
- control_rows: 6
- rollout_effect_rows: 30
- patch_counts: {"single_mlp_zero": 6, "window_mlp_zero": 6, "window_mlp_half": 6, "attn_mlp_window_zero": 6, "random_same_norm_control": 6}
- effect_supported_counts: {"True": 24, "False": 6}
- reverse_effect_counts: {"False": 26, "True": 4}
- writer_set_supported_counts: {"True": 5, "False": 1}
- compensation_suspected_counts: {"False": 4, "True": 2}
- mean_delta_continue_stop_margin: -5.739063
- mean_control_delta_continue_stop_margin: -5.578125

Note: This tests compensation and cross-layer writer-set candidates. It is not closure.
