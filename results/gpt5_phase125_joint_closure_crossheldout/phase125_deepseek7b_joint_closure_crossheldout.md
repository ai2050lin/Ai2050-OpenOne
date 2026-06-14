# Phase 125 Joint Closure Cross-heldout: deepseek7b

Generated: 2026-06-14 19:13:23
Monitor layer: L27; patch layers: [24, 25, 26, 27]

| category | residual ref | best head only | best combo | best control | pre-MLP only |
|---|---|---|---|---|---|
| number | residual_pre_reference residual_pre_reference k0 T-2.55 R+0.70 A+0.00 ratio+1.00 | head_set_only projection_discovered k16 T-0.85 R+0.16 A-196.86 ratio+0.33 | head_set_plus_pre_mlp projection_discovered k16 T-1.10 R+0.00 A-212.39 ratio+0.43 | head_set_only object_control k16 T-0.47 R+0.20 A-61.25 ratio+0.18 | pre_mlp_subspace_only pre_mlp_subspace k0 T-0.43 R+0.00 A-19.06 ratio+0.17 |
| container | residual_pre_reference residual_pre_reference k0 T-2.71 R+0.80 A+0.00 ratio+1.00 | head_set_only target_discovered k16 T-0.11 R+0.05 A+53.60 ratio+0.04 | head_set_plus_pre_mlp target_discovered k16 T-0.37 R+0.00 A+66.77 ratio+0.14 | head_set_only object_control k16 T-0.17 R+0.00 A+24.79 ratio+0.06 | pre_mlp_subspace_only pre_mlp_subspace k0 T-0.32 R+0.00 A+15.82 ratio+0.12 |
| plant | residual_pre_reference residual_pre_reference k0 T-2.42 R+1.62 A+0.00 ratio+1.00 | head_set_only target_discovered k16 T-1.33 R+0.00 A-113.78 ratio+0.55 | head_set_plus_pre_mlp target_discovered k16 T-1.80 R+0.00 A-123.57 ratio+0.74 | head_set_only low_pre_value_control k4 T-0.19 R+0.04 A-14.88 ratio+0.08 | pre_mlp_subspace_only pre_mlp_subspace k0 T-0.69 R+0.00 A-16.65 ratio+0.28 |
