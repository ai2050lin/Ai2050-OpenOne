# Phase 125 Joint Closure Cross-heldout: qwen3

Generated: 2026-06-14 19:11:19
Monitor layer: L35; patch layers: [34, 35]

| category | residual ref | best head only | best combo | best control | pre-MLP only |
|---|---|---|---|---|---|
| number | residual_pre_reference residual_pre_reference k0 T+0.10 R+0.15 A+0.00 ratio+1.00 | head_set_only target_discovered k2 T-0.06 R+0.15 A+3.20 ratio-0.59 | head_set_plus_pre_mlp target_discovered k2 T-0.05 R+0.15 A+3.42 ratio-0.53 | head_set_only object_control k1 T-0.02 R+0.06 A+0.18 ratio-0.21 | pre_mlp_subspace_only pre_mlp_subspace k0 T+0.01 R+0.02 A+0.32 ratio+0.10 |
