# Phase 809 Late-Layer Head/Channel Closer Decomposition (confirm)

- Status: `complete`
- Boundary: unit-level candidates for closure-solver input, not final token closure.

## By Unit

| model | unit | rows | cases | single net | single resolved | single emerged | emergence rate | single bias | fmt supp | loo net loss | loo bias loss | closure | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head:attn:L34:u20` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | 0.000 | 0.076 | 17.000 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L31:u10` | 1 | 1 | 11.000 | 0.000 | 11.000 | 0.023 | 0.188 | 0.012 | 15.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u2964` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.125 | 0.024 | 12.000 | -0.094 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u9552` | 1 | 1 | -7.000 | 7.000 | 0.000 | 0.000 | -0.125 | 0.023 | 11.000 | -0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2658` | 1 | 1 | -15.000 | 15.000 | 0.000 | 0.000 | 0.094 | -0.004 | 10.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1840` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.001 | 0.000 | 0.009 | 10.000 | -0.094 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u5054` | 1 | 1 | 8.000 | 1.000 | 9.000 | 0.010 | 0.125 | -0.032 | 9.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u228` | 1 | 1 | -27.000 | 27.000 | 0.000 | 0.000 | -0.281 | 0.072 | 7.000 | 0.125 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u8061` | 1 | 1 | -24.000 | 24.000 | 0.000 | 0.000 | -0.031 | 0.039 | 6.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L35:u26` | 4 | 4 | 3.500 | 1.500 | 5.000 | 0.006 | -0.086 | 0.095 | 5.500 | -0.023 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u299` | 1 | 1 | -6.000 | 6.000 | 0.000 | 0.000 | -0.250 | 0.047 | 5.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L31:u28` | 4 | 4 | -3.000 | 3.500 | 0.500 | 0.001 | 0.070 | 0.012 | 4.750 | 0.047 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 2, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u935` | 4 | 2 | -1.750 | 2.000 | 0.250 | 0.006 | 0.016 | 0.355 | 1.750 | -0.031 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u387` | 2 | 2 | -1.500 | 3.000 | 1.500 | 0.002 | -0.031 | 0.029 | 1.500 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u25` | 3 | 3 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.028 | 1.333 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u265` | 5 | 3 | 1.400 | 0.200 | 1.600 | 0.002 | -0.025 | -0.075 | 1.000 | -0.069 | 0.000 | `{"unit_neutral_or_mixed": 4, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u7503` | 1 | 1 | 6.000 | 0.000 | 6.000 | 0.010 | 0.031 | -0.002 | 1.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1804` | 2 | 1 | 6.500 | 0.000 | 6.500 | 0.009 | -0.047 | 0.001 | 1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u124` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.026 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u738` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.040 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `attention_head:attn:L35:u1` | 2 | 2 | -1.500 | 1.500 | 0.000 | 0.000 | -0.062 | 0.017 | 0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1483` | 4 | 2 | 0.500 | 0.250 | 0.750 | 0.001 | 0.062 | -0.018 | -0.250 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L34:u5` | 4 | 4 | 4.750 | 0.750 | 5.500 | 0.006 | 0.008 | 0.035 | -0.250 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u548` | 4 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.016 | -0.250 | -0.016 | 0.000 | `{"unit_neutral_or_mixed": 4}` |
| qwen3 | `mlp_channel:mlp:L35:u186` | 6 | 4 | -8.667 | 8.833 | 0.167 | 0.000 | -0.073 | 0.041 | -0.333 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 2, "unit_weak_reducer": 3}` |
| qwen3 | `mlp_channel:mlp:L34:u1028` | 3 | 2 | -0.333 | 0.667 | 0.333 | 0.006 | -0.042 | -0.021 | -0.333 | 0.021 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1147` | 6 | 3 | 0.000 | 0.167 | 0.167 | 0.003 | 0.000 | -0.014 | -0.333 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 4, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u29` | 2 | 2 | -1.000 | 2.500 | 1.500 | 0.002 | 0.031 | 0.052 | -0.500 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L35:u0` | 2 | 2 | -1.000 | 1.000 | 0.000 | 0.000 | -0.031 | 0.110 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u3372` | 2 | 2 | -1.000 | 1.000 | 0.000 | 0.000 | -0.031 | 0.036 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u213` | 2 | 2 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | 0.058 | -0.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u9` | 2 | 2 | -1.000 | 1.000 | 0.000 | 0.000 | -0.031 | 0.027 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u631` | 2 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.019 | -0.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u991` | 4 | 2 | -0.250 | 0.500 | 0.250 | 0.005 | -0.016 | -0.013 | -0.500 | -0.016 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u716` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.012 | 0.031 | -0.074 | -0.500 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L35:u2` | 4 | 4 | 2.750 | 0.500 | 3.250 | 0.003 | 0.000 | -0.032 | -0.500 | -0.094 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u30` | 2 | 2 | 7.500 | 0.000 | 7.500 | 0.008 | 0.016 | -0.008 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u352` | 2 | 1 | 11.000 | 0.000 | 11.000 | 0.012 | 0.016 | -0.007 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1439` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.088 | -0.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L35:u457` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.004 | -0.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u198` | 7 | 4 | 0.286 | 0.571 | 0.857 | 0.004 | -0.013 | 0.001 | -0.571 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u51` | 3 | 3 | 0.333 | 0.000 | 0.333 | 0.008 | 0.104 | -0.017 | -0.667 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L35:u24` | 4 | 4 | -7.250 | 7.250 | 0.000 | 0.000 | -0.055 | -0.017 | -0.750 | -0.031 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 2, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1730` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.021 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1433` | 2 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | 0.062 | 0.005 | -1.000 | -0.062 | 0.000 | `{"unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u2482` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.006 | -1.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u2839` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -0.004 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u4676` | 2 | 1 | 0.500 | 1.000 | 1.500 | 0.002 | 0.000 | 0.033 | -1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u17` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.000 | -0.016 | -1.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u206` | 2 | 2 | 1.000 | 0.000 | 1.000 | 0.001 | 0.000 | -0.018 | -1.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u717` | 1 | 1 | 4.000 | 0.000 | 4.000 | 0.006 | -0.094 | 0.027 | -1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L34:u8` | 2 | 2 | 5.500 | 0.000 | 5.500 | 0.012 | 0.125 | 0.036 | -1.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u110` | 1 | 1 | 10.000 | 0.000 | 10.000 | 0.011 | 0.125 | -0.043 | -1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u22` | 1 | 1 | 11.000 | 0.000 | 11.000 | 0.011 | 0.000 | -0.048 | -1.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u11` | 3 | 3 | 0.000 | 0.333 | 0.333 | 0.001 | 0.042 | -0.001 | -1.000 | 0.042 | 0.000 | `{"unit_neutral_or_mixed": 3}` |
| qwen3 | `mlp_channel:mlp:L34:u587` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.045 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1275` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | -0.039 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u2001` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | -0.066 | -1.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u156` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.056 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2131` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | -1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2150` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | -1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1643` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | -1.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u3061` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | -0.010 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u30` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | -1.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u112` | 3 | 3 | 3.333 | 0.000 | 3.333 | 0.003 | 0.042 | -0.043 | -1.333 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u2565` | 2 | 1 | -12.000 | 12.000 | 0.000 | 0.000 | -0.016 | 0.015 | -1.500 | -0.062 | 0.000 | `{"unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u9724` | 2 | 1 | 1.000 | 0.000 | 1.000 | 0.001 | 0.125 | -0.028 | -1.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u3317` | 2 | 1 | 2.500 | 2.000 | 4.500 | 0.005 | 0.062 | -0.017 | -1.500 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1806` | 2 | 1 | 0.000 | 0.500 | 0.500 | 0.001 | 0.125 | -0.018 | -1.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u8825` | 1 | 1 | -6.000 | 6.000 | 0.000 | 0.000 | -0.125 | 0.022 | -2.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u490` | 2 | 2 | -4.500 | 4.500 | 0.000 | 0.000 | -0.062 | 0.012 | -2.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2369` | 2 | 1 | -3.000 | 4.000 | 1.000 | 0.001 | 0.000 | 0.002 | -2.000 | -0.078 | 0.000 | `{"unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u5458` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.125 | 0.001 | -2.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u8109` | 1 | 1 | 6.000 | 0.000 | 6.000 | 0.010 | 0.031 | 0.000 | -2.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2142` | 2 | 1 | 9.000 | 0.000 | 9.000 | 0.010 | 0.078 | -0.017 | -2.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u65` | 1 | 1 | 10.000 | 0.000 | 10.000 | 0.010 | 0.000 | -0.032 | -2.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u186` | 2 | 2 | 21.500 | 0.000 | 21.500 | 0.022 | 0.031 | -0.070 | -2.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L35:u27` | 4 | 4 | -1.500 | 1.500 | 0.000 | 0.000 | -0.016 | -0.023 | -2.250 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_weak_reducer": 2}` |
| qwen3 | `attention_head:attn:L31:u19` | 4 | 4 | -1.000 | 2.250 | 1.250 | 0.006 | 0.055 | -0.007 | -2.250 | -0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u218` | 1 | 1 | -4.000 | 5.000 | 1.000 | 0.001 | -0.125 | 0.033 | -3.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u15` | 1 | 1 | 19.000 | 0.000 | 19.000 | 0.020 | 0.031 | -0.047 | -4.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L34:u28` | 2 | 2 | 5.000 | 0.000 | 5.000 | 0.006 | 0.125 | -0.029 | -4.500 | -0.016 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| qwen3 | `mlp_channel:mlp:L35:u2572` | 4 | 2 | 2.000 | 2.750 | 4.750 | 0.006 | 0.094 | -0.052 | -5.000 | -0.141 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u2603` | 1 | 1 | 7.000 | 1.000 | 8.000 | 0.009 | 0.125 | -0.021 | -5.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u6` | 1 | 1 | 21.000 | 0.000 | 21.000 | 0.022 | 0.156 | -0.026 | -5.000 | -0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u3` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.004 | 0.250 | -0.063 | -6.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u30` | 1 | 1 | 33.000 | 0.000 | 33.000 | 0.035 | 0.188 | -0.004 | -8.000 | -0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u2539` | 2 | 1 | -17.000 | 17.000 | 0.000 | 0.000 | -0.031 | -0.008 | 0.000 | -0.062 | 0.000 | `{"unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u136` | 1 | 1 | -11.000 | 11.000 | 0.000 | 0.000 | -0.250 | 0.032 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1166` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.000 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L31:u31` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1061` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -0.021 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1048` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u446` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.012 | 0.031 | 0.006 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u9` | 2 | 2 | 0.500 | 2.000 | 2.500 | 0.003 | 0.062 | -0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u3` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.062 | -0.014 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L34:u12` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.023 | 0.062 | -0.010 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1853` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.023 | 0.062 | -0.014 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u519` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.003 | -0.219 | 0.050 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u7401` | 1 | 1 | 39.000 | 0.000 | 39.000 | 0.041 | 0.062 | -0.066 | 0.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u3135` | 3 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.004 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 3}` |
| qwen3 | `mlp_channel:mlp:L33:u3304` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u219` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.038 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `attention_head:attn:L31:u18` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L35:u130` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L35:u3197` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.008 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L34:u166` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.057 | 0.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u2602` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u143` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u330` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u794` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `attention_head:attn:L34:u0` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.060 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u122` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.020 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u473` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.042 | 0.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u128` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.087 | 0.000 | 0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u9787` | 1 | 1 | -3.000 | 4.000 | 1.000 | 0.011 | -0.062 | 0.154 | 3.000 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| glm4 | `mlp_channel:mlp:L38:u10999` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 2.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u9742` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.031 | 0.040 | 1.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u8500` | 1 | 1 | -1.000 | 2.000 | 1.000 | 0.011 | -0.062 | 0.128 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u3652` | 1 | 1 | -1.000 | 2.000 | 1.000 | 0.011 | -0.062 | 0.129 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u722` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.049 | 1.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L33:u31` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.020 | 1.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u1041` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.008 | 0.000 | -0.005 | 1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u10429` | 1 | 1 | 4.000 | 0.000 | 4.000 | 0.031 | -0.031 | -0.005 | 1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u1401` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7202` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u2807` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u4876` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u629` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u9485` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u7605` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u9480` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L29:u4` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L29:u28` | 2 | 2 | -0.500 | 0.500 | 0.000 | 0.000 | 0.062 | -0.020 | 0.500 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u9366` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.005 | 0.000 | 0.014 | 0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L35:u7` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.013 | 0.500 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L35:u10` | 2 | 2 | 0.000 | 0.500 | 0.500 | 0.026 | 0.016 | 0.005 | 0.500 | 0.016 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u7453` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.500 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| glm4 | `mlp_channel:mlp:L38:u5489` | 3 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.032 | -0.667 | -0.010 | 0.000 | `{"unit_neutral_or_mixed": 3}` |
| glm4 | `mlp_channel:mlp:L27:u7692` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.031 | 0.025 | -1.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12832` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.008 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u10339` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u1012` | 2 | 2 | -0.500 | 0.500 | 0.000 | 0.000 | -0.031 | 0.014 | -2.000 | -0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u2684` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.101 | -3.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u8046` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.062 | 0.024 | -4.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u3299` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.062 | 0.025 | -4.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7968` | 1 | 1 | 7.000 | 0.000 | 7.000 | 0.074 | 0.062 | -0.219 | -7.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u4744` | 1 | 1 | 7.000 | 0.000 | 7.000 | 0.074 | 0.062 | -0.231 | -10.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u10685` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L33:u25` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | -0.021 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L29:u14` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u1` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u0` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | -0.062 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u13` | 2 | 2 | -0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12913` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.005 | -0.031 | 0.044 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L34:u7327` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.005 | 0.000 | 0.014 | 0.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u9` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.011 | 0.000 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4671` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.007 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u11709` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.004 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u5253` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u12358` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | -0.005 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u11370` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | -0.001 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u10600` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12909` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.031 | -0.031 | 0.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u7` | 2 | 2 | 1.000 | 0.000 | 1.000 | 0.036 | 0.016 | 0.001 | 0.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| glm4 | `attention_head:attn:L33:u20` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u17` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L29:u9` | 2 | 2 | 1.000 | 0.000 | 1.000 | 0.036 | 0.047 | -0.021 | 0.000 | 0.016 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| glm4 | `attention_head:attn:L29:u31` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.059 | 0.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L29:u18` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.031 | 0.012 | 0.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L35:u3` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.031 | 0.012 | 0.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u8126` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.062 | -0.010 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u16` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.000 | -0.016 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u14` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.062 | -0.003 | 0.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u12419` | 2 | 2 | 2.000 | 0.000 | 2.000 | 0.016 | -0.047 | -0.001 | 0.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4526` | 3 | 2 | 0.000 | 0.333 | 0.333 | 0.006 | 0.000 | -0.004 | 0.000 | 0.021 | 0.000 | `{"unit_neutral_or_mixed": 3}` |
| glm4 | `mlp_channel:mlp:L38:u4805` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.036 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| glm4 | `mlp_channel:mlp:L38:u4458` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u5084` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.019 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u7532` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u5302` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.019 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7043` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u8761` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.024 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u5668` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u3953` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u6282` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u1917` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u10364` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u4776` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u11792` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | 0.015 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12695` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | 0.000 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u2049` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.039 | 0.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u150` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.047 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L29:u30` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L35:u23` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.004 | 0.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u11329` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u533` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7526` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u11276` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | -0.010 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u10277` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u1803` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u3498` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.006 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u7272` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | -0.004 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L29:u25` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12657` | 1 | 1 | -19.000 | 21.000 | 2.000 | 0.001 | 0.023 | -0.075 | 189.000 | -0.047 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u13015` | 1 | 1 | -28.000 | 39.000 | 11.000 | 0.004 | 0.086 | -0.132 | 184.000 | -0.109 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u11505` | 1 | 1 | -7.000 | 14.000 | 7.000 | 0.003 | 0.055 | -0.056 | 150.000 | -0.047 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u3776` | 2 | 1 | 0.000 | 8.500 | 8.500 | 0.015 | 0.223 | -0.173 | 108.500 | -0.285 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L19:u0` | 1 | 1 | 12.000 | 9.000 | 21.000 | 0.008 | -0.062 | 0.006 | 96.000 | 0.023 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u9169` | 2 | 1 | -4.500 | 9.000 | 4.500 | 0.004 | 0.082 | -0.076 | 83.000 | -0.141 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L19:u4` | 2 | 2 | 16.500 | 1.500 | 18.000 | 0.012 | -0.090 | 0.011 | 73.500 | 0.035 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| deepseek7b | `attention_head:attn:L19:u3` | 1 | 1 | 19.000 | 5.000 | 24.000 | 0.010 | 0.008 | 0.012 | 54.000 | 0.023 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u10423` | 1 | 1 | 10.000 | 2.000 | 12.000 | 0.005 | 0.008 | -0.003 | 48.000 | -0.047 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u77` | 1 | 1 | -10.000 | 14.000 | 4.000 | 0.002 | -0.039 | -0.010 | 47.000 | -0.023 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u12837` | 1 | 1 | -22.000 | 22.000 | 0.000 | 0.000 | -0.016 | 0.000 | 46.000 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u17889` | 1 | 1 | 39.000 | 1.000 | 40.000 | 0.016 | 0.000 | 0.019 | 46.000 | 0.047 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u5622` | 2 | 1 | 1.500 | 2.500 | 4.000 | 0.004 | 0.004 | -0.040 | 43.000 | -0.141 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u8324` | 1 | 1 | -1.000 | 7.000 | 6.000 | 0.002 | 0.000 | 0.015 | 41.000 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u4203` | 1 | 1 | -14.000 | 14.000 | 0.000 | 0.000 | -0.008 | -0.000 | 40.000 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L19:u6` | 1 | 1 | 50.000 | 0.000 | 50.000 | 0.020 | 0.016 | -0.017 | 36.000 | 0.023 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u3596` | 1 | 1 | -3.000 | 13.000 | 10.000 | 0.004 | -0.031 | -0.018 | 27.000 | -0.031 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u17843` | 1 | 1 | -3.000 | 13.000 | 10.000 | 0.004 | -0.023 | -0.017 | 21.000 | -0.023 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u3050` | 1 | 1 | 15.000 | 3.000 | 18.000 | 0.007 | 0.000 | 0.010 | 16.000 | 0.008 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L19:u22` | 2 | 2 | 12.500 | 5.500 | 18.000 | 0.007 | -0.125 | 0.005 | 12.000 | 0.035 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u23` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.125 | 0.082 | 10.000 | 0.156 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L26:u14` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.005 | 0.031 | 0.021 | 7.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u13421` | 1 | 1 | -11.000 | 17.000 | 6.000 | 0.002 | -0.023 | -0.005 | 3.000 | 0.023 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L19:u14` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.031 | -0.005 | 3.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L27:u21` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.125 | 0.056 | 3.000 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L27:u20` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.125 | 0.044 | 3.000 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u5510` | 1 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.031 | -0.001 | 3.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L27:u19` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.016 | 3.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L19:u13` | 1 | 1 | 3.000 | 0.000 | 3.000 | 0.009 | -0.125 | 0.001 | 3.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u15158` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.006 | -0.125 | 0.005 | 2.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L26:u0` | 1 | 1 | -6.000 | 6.000 | 0.000 | 0.000 | -0.266 | 0.093 | 1.000 | 0.109 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u25` | 2 | 2 | -3.000 | 3.000 | 0.000 | 0.000 | -0.016 | 0.018 | 1.000 | -0.055 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `attention_head:attn:L26:u9` | 1 | 1 | 3.000 | 0.000 | 3.000 | 0.008 | 0.031 | 0.022 | 1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u1106` | 1 | 1 | 3.000 | 0.000 | 3.000 | 0.008 | 0.125 | -0.045 | 1.000 | -0.094 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L25:u5` | 1 | 1 | 3.000 | 0.000 | 3.000 | 0.008 | 0.031 | 0.009 | 1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u15731` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.003 | 0.000 | -0.002 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u16013` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12909` | 2 | 1 | -3.000 | 3.500 | 0.500 | 0.001 | -0.141 | 0.032 | 0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u11` | 2 | 2 | 0.500 | 0.500 | 1.000 | 0.004 | 0.062 | -0.031 | -0.500 | -0.133 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u15812` | 2 | 2 | 0.500 | 0.500 | 1.000 | 0.004 | 0.055 | -0.060 | -0.500 | -0.117 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L26:u15` | 2 | 2 | 1.500 | 0.000 | 1.500 | 0.004 | 0.016 | -0.010 | -0.500 | -0.055 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u569` | 2 | 1 | 0.000 | 0.500 | 0.500 | 0.001 | -0.062 | 0.008 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| deepseek7b | `attention_head:attn:L26:u4` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.125 | 0.048 | -1.000 | -0.016 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u4` | 1 | 1 | -4.000 | 4.000 | 0.000 | 0.000 | -0.016 | -0.022 | -1.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L26:u23` | 1 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.125 | 0.011 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u8` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.016 | -0.012 | -1.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u13` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.004 | 0.000 | -0.040 | -1.000 | -0.141 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u16230` | 2 | 1 | 0.000 | 0.500 | 0.500 | 0.001 | -0.062 | 0.009 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| deepseek7b | `mlp_channel:mlp:L27:u12593` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.004 | 0.109 | -0.065 | -1.000 | -0.250 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12818` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.004 | 0.109 | -0.066 | -1.000 | -0.250 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L25:u1` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L27:u12` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.027 | -1.000 | -0.141 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u15791` | 3 | 2 | -15.667 | 17.000 | 1.333 | 0.004 | -0.698 | 0.298 | -1.667 | 0.328 | 0.000 | `{"unit_closer_candidate_no_closure": 2, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L26:u3` | 1 | 1 | 5.000 | 0.000 | 5.000 | 0.013 | 0.156 | -0.035 | -3.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u11715` | 1 | 1 | 0.000 | 4.000 | 4.000 | 0.002 | -0.062 | 0.019 | -3.000 | 0.008 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u2295` | 2 | 1 | -8.500 | 10.000 | 1.500 | 0.004 | -0.219 | 0.086 | -4.000 | 0.078 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u8694` | 1 | 1 | -13.000 | 20.000 | 7.000 | 0.003 | -0.031 | 0.005 | -8.000 | 0.023 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u15099` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.003 | -0.125 | 0.005 | -8.000 | -0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u1787` | 1 | 1 | 2.000 | 1.000 | 3.000 | 0.009 | -0.125 | -0.002 | -8.000 | -0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u11929` | 1 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.125 | 0.029 | -9.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u4816` | 2 | 2 | -4.500 | 6.000 | 1.500 | 0.001 | -0.082 | 0.004 | -10.000 | 0.012 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `mlp_channel:mlp:L26:u9784` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.003 | -0.125 | 0.021 | -10.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u5378` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -0.001 | -11.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u4514` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.125 | 0.006 | -11.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u9394` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.003 | 0.000 | 0.000 | -11.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u15679` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.006 | -11.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u165` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.125 | 0.003 | -12.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u2644` | 1 | 1 | -8.000 | 12.000 | 4.000 | 0.012 | -0.406 | 0.103 | -13.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u10806` | 1 | 1 | 25.000 | 2.000 | 27.000 | 0.011 | 0.008 | 0.006 | -13.000 | 0.023 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L19:u1` | 1 | 1 | 0.000 | 2.000 | 2.000 | 0.006 | -0.125 | 0.012 | -15.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u7492` | 1 | 1 | 80.000 | 0.000 | 80.000 | 0.032 | -0.016 | 0.044 | -24.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u9289` | 1 | 1 | 56.000 | 0.000 | 56.000 | 0.022 | -0.031 | 0.053 | -61.000 | -0.008 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u16446` | 1 | 1 | 44.000 | 8.000 | 52.000 | 0.021 | -0.125 | 0.111 | -147.000 | 0.117 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u1175` | 1 | 1 | 113.000 | 5.000 | 118.000 | 0.047 | -0.219 | 0.215 | -303.000 | 0.242 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L26:u17` | 2 | 2 | -2.000 | 2.500 | 0.500 | 0.001 | -0.062 | 0.032 | 0.000 | -0.008 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u24` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.125 | 0.012 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u27` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u14` | 1 | 1 | 3.000 | 0.000 | 3.000 | 0.008 | 0.031 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u826` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| deepseek7b | `mlp_channel:mlp:L27:u12614` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u11782` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u1219` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L27:u25` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L25:u10` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u5030` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.011 | 0.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L25:u11` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.094 | 0.043 | 0.000 | 0.094 | 0.000 | `{"unit_neutral_or_mixed": 1}` |

## By Component Unit Kind

| model | group | rows | cases | single net | single resolved | single emerged | loo net loss | labels |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `mlp_channel:mlp:L33` | 32 | 4 | 2.719 | 1.156 | 3.875 | 0.531 | `{"unit_closer_candidate_no_closure": 3, "unit_neutral_or_mixed": 15, "unit_new_blocker_or_deformer": 8, "unit_weak_reducer": 6}` |
| qwen3 | `attention_head:attn:L31` | 20 | 4 | 2.450 | 1.500 | 3.950 | 0.450 | `{"unit_closer_candidate_no_closure": 2, "unit_neutral_or_mixed": 9, "unit_new_blocker_or_deformer": 7, "unit_weak_reducer": 2}` |
| qwen3 | `attention_head:attn:L35` | 20 | 4 | -0.750 | 2.400 | 1.650 | 0.400 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 9, "unit_new_blocker_or_deformer": 3, "unit_weak_reducer": 7}` |
| qwen3 | `attention_head:attn:L34` | 20 | 4 | 3.300 | 0.800 | 4.100 | 0.100 | `{"unit_closer_candidate_no_closure": 2, "unit_neutral_or_mixed": 8, "unit_new_blocker_or_deformer": 8, "unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L35` | 64 | 4 | -1.562 | 2.703 | 1.141 | -0.266 | `{"unit_closer_candidate_no_closure": 5, "unit_neutral_or_mixed": 34, "unit_new_blocker_or_deformer": 11, "unit_weak_reducer": 14}` |
| qwen3 | `mlp_channel:mlp:L34` | 64 | 4 | -0.062 | 1.391 | 1.328 | -0.453 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 31, "unit_new_blocker_or_deformer": 15, "unit_weak_reducer": 17}` |
| glm4 | `attention_head:attn:L35` | 10 | 2 | -0.100 | 0.400 | 0.300 | 0.200 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 3, "unit_weak_reducer": 4}` |
| glm4 | `attention_head:attn:L29` | 10 | 2 | 0.200 | 0.200 | 0.400 | 0.200 | `{"unit_neutral_or_mixed": 4, "unit_new_blocker_or_deformer": 4, "unit_weak_reducer": 2}` |
| glm4 | `mlp_channel:mlp:L34` | 16 | 2 | 0.062 | 0.000 | 0.062 | 0.125 | `{"unit_neutral_or_mixed": 15, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33` | 10 | 2 | 0.700 | 0.100 | 0.800 | 0.100 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 8, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38` | 32 | 2 | 0.094 | 0.156 | 0.250 | 0.094 | `{"unit_neutral_or_mixed": 22, "unit_new_blocker_or_deformer": 7, "unit_weak_reducer": 3}` |
| glm4 | `mlp_channel:mlp:L27` | 16 | 2 | 0.500 | 0.250 | 0.750 | -0.562 | `{"unit_neutral_or_mixed": 6, "unit_new_blocker_or_deformer": 6, "unit_weak_reducer": 4}` |
| glm4 | `mlp_channel:mlp:L39` | 16 | 2 | 0.562 | 0.500 | 1.062 | -0.750 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 11, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 2}` |
| deepseek7b | `attention_head:attn:L19` | 10 | 2 | 13.700 | 3.500 | 17.200 | 34.800 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 7, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27` | 32 | 2 | 0.719 | 6.938 | 7.656 | 16.000 | `{"unit_closer_candidate_no_closure": 3, "unit_neutral_or_mixed": 14, "unit_new_blocker_or_deformer": 6, "unit_weak_reducer": 9}` |
| deepseek7b | `mlp_channel:mlp:L26` | 16 | 2 | 6.375 | 4.312 | 10.688 | 4.688 | `{"unit_closer_candidate_no_closure": 4, "unit_neutral_or_mixed": 7, "unit_new_blocker_or_deformer": 3, "unit_weak_reducer": 2}` |
| deepseek7b | `mlp_channel:mlp:L24` | 16 | 2 | 2.125 | 5.438 | 7.562 | 2.938 | `{"unit_closer_candidate_no_closure": 4, "unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 6, "unit_weak_reducer": 4}` |
| deepseek7b | `attention_head:attn:L27` | 10 | 2 | -1.500 | 1.800 | 0.300 | 1.600 | `{"unit_closer_candidate_no_closure": 4, "unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 2}` |
| deepseek7b | `attention_head:attn:L26` | 10 | 2 | -0.500 | 1.900 | 1.400 | 0.300 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 4, "unit_weak_reducer": 4}` |
| deepseek7b | `attention_head:attn:L25` | 10 | 2 | -0.800 | 1.400 | 0.600 | 0.000 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 5}` |
