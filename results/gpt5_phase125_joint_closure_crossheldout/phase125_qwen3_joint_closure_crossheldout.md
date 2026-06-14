# Phase 125 Joint Closure Cross-heldout: qwen3

Generated: 2026-06-14 19:11:37
Monitor layer: L35; patch layers: [32, 33, 34, 35]

| category | residual ref | best head only | best combo | best control | pre-MLP only |
|---|---|---|---|---|---|
| number | residual_pre_reference residual_pre_reference k0 T-0.12 R+0.44 A+11.07 ratio+1.00 | head_set_only target_discovered k16 T-0.66 R+0.00 A+6.79 ratio+5.30 | head_set_plus_pre_mlp target_discovered k16 T-0.66 R+0.00 A+6.79 ratio+5.28 | head_set_only random_control k16 T-0.31 R+0.00 A+10.64 ratio+2.49 | pre_mlp_subspace_only pre_mlp_subspace k0 T+0.01 R+0.10 A+0.00 ratio-0.07 |
| container | residual_pre_reference residual_pre_reference k0 T-0.02 R+0.30 A-6.59 ratio+1.00 | head_set_only target_discovered k16 T-0.29 R+0.00 A-24.84 ratio+12.86 | head_set_plus_pre_mlp target_discovered k16 T-0.33 R+0.00 A-26.08 ratio+14.49 | head_set_only low_pre_value_control k16 T-0.05 R+0.22 A-25.47 ratio+2.13 | pre_mlp_subspace_only pre_mlp_subspace k0 T-0.05 R+0.06 A-1.80 ratio+2.16 |
| plant | residual_pre_reference residual_pre_reference k0 T-0.19 R+0.35 A+11.45 ratio+1.00 | head_set_only target_discovered k16 T-0.89 R+0.00 A+7.92 ratio+4.78 | head_set_plus_pre_mlp target_discovered k16 T-0.95 R+0.00 A+9.42 ratio+5.10 | head_set_only object_control k16 T-0.11 R+0.21 A+2.66 ratio+0.61 | pre_mlp_subspace_only pre_mlp_subspace k0 T-0.02 R+0.07 A+2.63 ratio+0.12 |
