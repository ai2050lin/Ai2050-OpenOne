# Phase263 Continuation Suppression Candidate Causal Audit

- status: complete
- suppression_rows: 2304
- channel_causal_effect_rows: 2304
- rollout_probe_rows: 45
- mean_stop_margin_delta_by_policy: {"suppress_explanation": 1.121358, "suppress_structured": -1.291441, "suppress_natural": -0.035718, "suppress_boundary_aftereffect": 1.704183, "suppress_top": 2.304891, "stop_plus_top": 4.339233}
- mean_top_channel_logit_delta_by_policy: {"suppress_explanation": -1.217122, "suppress_structured": -0.780436, "suppress_natural": -1.27002, "suppress_boundary_aftereffect": -1.342122, "suppress_top": -2.98877, "stop_plus_top": -2.733236}
- winner_flip_rate_by_policy: {"suppress_explanation": 0.0, "suppress_structured": 0.0, "suppress_natural": 0.0, "suppress_boundary_aftereffect": 0.0, "suppress_top": 0.0, "stop_plus_top": 0.0}
- target_preserved_rate_by_policy: {"suppress_explanation": 1.0, "suppress_structured": 1.0, "suppress_natural": 1.0, "suppress_boundary_aftereffect": 0.973958, "suppress_top": 0.989583, "stop_plus_top": 0.908854}
- rollout_mean_tokens_by_policy: {"no_patch": 24.0, "suppress_top": 24.0, "stop_plus_top": 21.866667}
- rollout_stop_rate_by_policy: {"no_patch": 0.0, "suppress_top": 0.0, "stop_plus_top": 0.2}
