# Phase 809 Late-Layer Head/Channel Closer Decomposition (main)

- Status: `complete`
- Boundary: unit-level candidates for closure-solver input, not final token closure.

## By Unit

| model | unit | rows | cases | single net | single resolved | single emerged | emergence rate | single bias | fmt supp | loo net loss | loo bias loss | closure | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head:attn:L31:u30` | 1 | 1 | 0.000 | 3.000 | 3.000 | 0.003 | 0.000 | -0.002 | 19.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u7503` | 1 | 1 | 1.000 | 1.000 | 2.000 | 0.003 | 0.125 | -0.002 | 17.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u228` | 1 | 1 | -17.000 | 17.000 | 0.000 | 0.000 | -0.156 | 0.076 | 16.000 | 0.188 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L31:u9` | 1 | 1 | 1.000 | 2.000 | 3.000 | 0.003 | 0.000 | 0.002 | 12.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2369` | 2 | 1 | -10.500 | 10.500 | 0.000 | 0.000 | 0.031 | 0.009 | 7.500 | -0.031 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u29` | 1 | 1 | -17.000 | 17.000 | 0.000 | 0.000 | -0.031 | 0.042 | 6.000 | 0.125 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L31:u19` | 2 | 2 | -4.000 | 5.500 | 1.500 | 0.019 | 0.031 | 0.009 | 5.500 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u186` | 3 | 2 | -15.000 | 15.000 | 0.000 | 0.000 | -0.062 | 0.056 | 5.333 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u935` | 1 | 1 | -4.000 | 4.000 | 0.000 | 0.000 | 0.000 | 0.719 | 3.000 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L34:u5` | 2 | 2 | -1.500 | 3.000 | 1.500 | 0.010 | 0.031 | 0.034 | 3.000 | 0.016 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u28` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.000 | 0.014 | 3.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u18` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.038 | 0.062 | 0.000 | 3.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u352` | 2 | 1 | 7.500 | 0.500 | 8.000 | 0.008 | 0.016 | 0.001 | 3.000 | 0.016 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u8` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | 0.127 | 3.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `attention_head:attn:L34:u28` | 1 | 1 | -5.000 | 7.000 | 2.000 | 0.002 | 0.000 | -0.016 | 2.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u15` | 1 | 1 | -1.000 | 3.000 | 2.000 | 0.002 | 0.000 | -0.022 | 2.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34:u25` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.062 | 0.062 | 2.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u446` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.010 | 2.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2131` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.040 | 2.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2142` | 2 | 1 | -2.500 | 3.000 | 0.500 | 0.001 | 0.062 | -0.003 | 1.500 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u198` | 4 | 2 | -2.750 | 3.250 | 0.500 | 0.009 | -0.023 | 0.008 | 1.250 | -0.023 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u2565` | 2 | 1 | -20.500 | 20.500 | 0.000 | 0.000 | -0.031 | 0.031 | 0.500 | -0.062 | 0.000 | `{"unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L35:u991` | 2 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.031 | -0.011 | -0.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1028` | 2 | 1 | -0.500 | 1.000 | 0.500 | 0.009 | -0.031 | -0.002 | -0.500 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1483` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.008 | -0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L35:u5054` | 1 | 1 | -3.000 | 6.000 | 3.000 | 0.003 | 0.000 | -0.021 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L35:u0` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.226 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u3372` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.021 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u213` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | 0.062 | 0.056 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1730` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.021 | -1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L35:u26` | 2 | 2 | 0.500 | 0.500 | 1.000 | 0.002 | 0.000 | 0.046 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u8109` | 1 | 1 | 1.000 | 1.000 | 2.000 | 0.003 | 0.125 | -0.002 | -1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u2658` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.001 | 0.000 | -0.007 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1275` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.056 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u112` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | -0.076 | -1.000 | -0.125 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u156` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.056 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `attention_head:attn:L35:u27` | 2 | 2 | -4.000 | 4.000 | 0.000 | 0.000 | 0.016 | -0.041 | -1.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u265` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.083 | -1.500 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u8825` | 1 | 1 | -14.000 | 14.000 | 0.000 | 0.000 | 0.094 | 0.014 | -2.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u490` | 1 | 1 | -12.000 | 12.000 | 0.000 | 0.000 | -0.156 | 0.038 | -2.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u9552` | 1 | 1 | -9.000 | 10.000 | 1.000 | 0.002 | 0.094 | -0.004 | -2.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L35:u24` | 2 | 2 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.004 | -2.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L35:u2` | 1 | 1 | 1.000 | 1.000 | 2.000 | 0.003 | 0.125 | -0.019 | -2.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L31:u11` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.038 | 0.062 | 0.005 | -2.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u717` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.002 | 0.000 | 0.015 | -2.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1804` | 2 | 1 | -0.500 | 1.500 | 1.000 | 0.001 | 0.000 | 0.005 | -2.500 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u3317` | 2 | 1 | 4.000 | 4.000 | 8.000 | 0.008 | 0.062 | -0.013 | -2.500 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u2572` | 2 | 1 | -0.500 | 4.000 | 3.500 | 0.005 | 0.125 | -0.090 | -3.000 | -0.188 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L31:u6` | 1 | 1 | -4.000 | 5.000 | 1.000 | 0.001 | 0.000 | -0.014 | -4.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u2539` | 2 | 1 | -18.000 | 18.500 | 0.500 | 0.001 | 0.031 | 0.005 | 0.000 | -0.062 | 0.000 | `{"unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u2839` | 1 | 1 | -9.000 | 10.000 | 1.000 | 0.002 | 0.094 | -0.002 | 0.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1166` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.000 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1147` | 2 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.031 | -0.030 | 0.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u51` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.000 | -0.018 | 0.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u738` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.000 | -0.040 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `attention_head:attn:L34:u17` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.038 | 0.062 | -0.008 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u548` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.013 | 0.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| qwen3 | `mlp_channel:mlp:L33:u3304` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u219` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.038 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u130` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u9787` | 1 | 1 | -3.000 | 4.000 | 1.000 | 0.011 | -0.062 | 0.154 | 3.000 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| glm4 | `mlp_channel:mlp:L38:u10999` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 2.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u9742` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.031 | 0.040 | 1.000 | 0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u8500` | 1 | 1 | -1.000 | 2.000 | 1.000 | 0.011 | -0.062 | 0.128 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u10` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -0.003 | 1.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L33:u31` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.020 | 1.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u10429` | 1 | 1 | 4.000 | 0.000 | 4.000 | 0.031 | -0.031 | -0.005 | 1.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u1401` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7202` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u2807` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u4876` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u9485` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u7605` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u7453` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.002 | 1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u9480` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L29:u4` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L29:u28` | 2 | 2 | -0.500 | 0.500 | 0.000 | 0.000 | 0.062 | -0.020 | 0.500 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u7` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.013 | 0.500 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u7692` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.031 | 0.025 | -1.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u5489` | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.025 | -1.000 | -0.016 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| glm4 | `mlp_channel:mlp:L38:u12832` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.008 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u10339` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.009 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u8046` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.062 | 0.024 | -4.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u3299` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.062 | 0.025 | -4.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7968` | 1 | 1 | 7.000 | 0.000 | 7.000 | 0.074 | 0.062 | -0.219 | -7.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u4744` | 1 | 1 | 7.000 | 0.000 | 7.000 | 0.074 | 0.062 | -0.231 | -10.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u10685` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L33:u25` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | -0.021 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L29:u14` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u1` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.031 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `attention_head:attn:L35:u0` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | -0.062 | 0.000 | `{"unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12913` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.005 | -0.031 | 0.044 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u9` | 2 | 2 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.011 | 0.000 | 0.016 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4671` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.007 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u5253` | 2 | 1 | 0.500 | 0.000 | 0.500 | 0.009 | 0.031 | -0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u9366` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.024 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L34:u7327` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.044 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u12358` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | -0.005 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u10600` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u7` | 2 | 2 | 1.000 | 0.000 | 1.000 | 0.036 | 0.016 | 0.001 | 0.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| glm4 | `attention_head:attn:L33:u20` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L29:u9` | 2 | 2 | 1.000 | 0.000 | 1.000 | 0.036 | 0.047 | -0.021 | 0.000 | 0.016 | 0.000 | `{"unit_new_blocker_or_deformer": 2}` |
| glm4 | `attention_head:attn:L29:u31` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.000 | 0.059 | 0.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L29:u18` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.031 | 0.012 | 0.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L35:u3` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.053 | 0.031 | 0.012 | 0.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u8126` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.062 | -0.010 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u11709` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.062 | -0.007 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33:u16` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.019 | 0.000 | -0.016 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u12419` | 2 | 2 | 2.000 | 0.000 | 2.000 | 0.016 | -0.047 | -0.001 | 0.000 | -0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4526` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.006 | 0.000 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| glm4 | `mlp_channel:mlp:L38:u4805` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.048 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4458` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u5084` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.019 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u5302` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.019 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7043` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u8761` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.024 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u5668` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u3953` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u1917` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u10364` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u1012` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u11792` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | 0.015 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12695` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | 0.000 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u2049` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.039 | 0.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u150` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.047 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L35:u13` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `attention_head:attn:L35:u23` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.004 | 0.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u11329` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u533` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u11276` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | -0.010 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u10277` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u1803` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u3498` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.006 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u13015` | 1 | 1 | -50.000 | 65.000 | 15.000 | 0.006 | 0.117 | -0.131 | 187.000 | -0.188 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12657` | 1 | 1 | -16.000 | 31.000 | 15.000 | 0.006 | 0.070 | -0.080 | 97.000 | -0.141 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u3776` | 2 | 1 | -18.500 | 29.000 | 10.500 | 0.017 | 0.238 | -0.166 | 78.000 | -0.305 | 0.000 | `{"unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u9169` | 2 | 1 | -6.000 | 13.000 | 7.000 | 0.003 | 0.090 | -0.083 | 23.000 | -0.098 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `mlp_channel:mlp:L27:u15791` | 1 | 1 | -9.000 | 9.000 | 0.000 | 0.000 | -0.500 | 0.237 | 11.000 | 0.609 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L27:u23` | 1 | 1 | -6.000 | 7.000 | 1.000 | 0.003 | -0.281 | 0.110 | 6.000 | 0.125 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u4514` | 1 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.031 | -0.009 | 6.000 | 0.031 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u165` | 1 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.031 | -0.006 | 6.000 | 0.031 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u2295` | 2 | 1 | -4.000 | 4.000 | 0.000 | 0.000 | -0.188 | 0.071 | 5.500 | 0.125 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u10806` | 1 | 1 | 1.000 | 10.000 | 11.000 | 0.004 | -0.070 | 0.008 | 5.000 | -0.070 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u15731` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | 5.000 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L26:u9` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.125 | 0.016 | 4.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u12837` | 1 | 1 | -10.000 | 12.000 | 2.000 | 0.001 | -0.078 | -0.004 | 3.000 | -0.070 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u9784` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.017 | 3.000 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u1787` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.003 | 3.000 | 0.062 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `attention_head:attn:L19:u13` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.003 | 0.000 | -0.003 | 3.000 | 0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L27:u21` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.125 | 0.083 | 2.000 | 0.250 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u20` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.125 | 0.065 | 2.000 | 0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u9394` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.004 | 2.000 | 0.062 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L26:u3` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.005 | 0.000 | -0.036 | 2.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u15158` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.006 | 2.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L25:u5` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.125 | 0.012 | 2.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u16230` | 2 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.016 | 0.007 | 1.500 | 0.031 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `mlp_channel:mlp:L27:u12909` | 2 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.078 | 0.016 | 1.500 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L26:u14` | 1 | 1 | -5.000 | 5.000 | 0.000 | 0.000 | -0.156 | 0.034 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u14` | 1 | 1 | -4.000 | 4.000 | 0.000 | 0.000 | -0.156 | 0.027 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L26:u4` | 1 | 1 | -4.000 | 4.000 | 0.000 | 0.000 | -0.125 | 0.068 | 1.000 | 0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L26:u0` | 1 | 1 | -3.000 | 3.000 | 0.000 | 0.000 | -0.109 | 0.085 | 1.000 | 0.109 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u16013` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.001 | 1.000 | 0.062 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u27` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.005 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u19` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.125 | 0.022 | 1.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u826` | 2 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.002 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2}` |
| deepseek7b | `mlp_channel:mlp:L26:u5378` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 1.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u15099` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.005 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L27:u25` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.005 | 1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L26:u17` | 2 | 2 | -3.500 | 3.500 | 0.000 | 0.000 | -0.133 | 0.031 | 0.500 | 0.055 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `mlp_channel:mlp:L27:u569` | 2 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.016 | 0.009 | 0.500 | 0.031 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u25` | 2 | 2 | -1.000 | 1.000 | 0.000 | 0.000 | -0.062 | 0.023 | 0.500 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12593` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.109 | -0.074 | -1.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u15812` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.004 | 0.234 | -0.110 | -1.000 | -0.234 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L19:u1` | 1 | 1 | -8.000 | 8.000 | 0.000 | 0.000 | -0.031 | -0.004 | -2.000 | 0.094 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u11` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.234 | 0.089 | -4.000 | 0.094 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u4816` | 2 | 2 | -5.500 | 8.000 | 2.500 | 0.001 | -0.016 | 0.000 | -10.000 | 0.004 | 0.000 | `{"unit_closer_candidate_no_closure": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u17843` | 1 | 1 | -34.000 | 36.000 | 2.000 | 0.001 | -0.031 | -0.024 | -14.000 | -0.070 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u77` | 1 | 1 | -58.000 | 58.000 | 0.000 | 0.000 | -0.047 | -0.028 | -24.000 | -0.070 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L19:u22` | 2 | 2 | -15.000 | 17.500 | 2.500 | 0.001 | -0.062 | 0.006 | -34.000 | -0.031 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `mlp_channel:mlp:L26:u3596` | 1 | 1 | -33.000 | 33.000 | 0.000 | 0.000 | -0.016 | -0.030 | -37.000 | -0.070 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u4203` | 1 | 1 | -8.000 | 10.000 | 2.000 | 0.001 | 0.000 | 0.007 | -37.000 | -0.070 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u13421` | 1 | 1 | -11.000 | 16.000 | 5.000 | 0.002 | -0.016 | -0.008 | -40.000 | -0.078 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u11715` | 1 | 1 | 33.000 | 0.000 | 33.000 | 0.013 | -0.031 | 0.023 | -65.000 | -0.016 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u17889` | 1 | 1 | 8.000 | 5.000 | 13.000 | 0.005 | -0.039 | 0.021 | -68.000 | -0.023 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L19:u4` | 2 | 2 | -25.000 | 26.000 | 1.000 | 0.000 | -0.031 | -0.004 | -72.000 | -0.047 | 0.000 | `{"unit_weak_reducer": 2}` |
| deepseek7b | `attention_head:attn:L19:u3` | 1 | 1 | 0.000 | 18.000 | 18.000 | 0.007 | -0.008 | -0.005 | -84.000 | -0.094 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u7492` | 1 | 1 | 26.000 | 1.000 | 27.000 | 0.011 | -0.039 | 0.049 | -128.000 | -0.039 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u9289` | 1 | 1 | 41.000 | 1.000 | 42.000 | 0.016 | -0.031 | 0.067 | -138.000 | -0.055 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L19:u0` | 1 | 1 | -46.000 | 50.000 | 4.000 | 0.002 | -0.031 | 0.002 | -149.000 | -0.102 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u16446` | 1 | 1 | 58.000 | 4.000 | 62.000 | 0.024 | -0.102 | 0.105 | -229.000 | 0.117 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u1175` | 1 | 1 | 48.000 | 32.000 | 80.000 | 0.031 | -0.320 | 0.241 | -477.000 | 0.211 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L25:u8` | 1 | 1 | -4.000 | 4.000 | 0.000 | 0.000 | -0.016 | 0.015 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u11` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | 0.109 | -0.053 | 0.000 | -0.125 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27:u13` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.016 | -0.032 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L25:u24` | 1 | 1 | -1.000 | 1.000 | 0.000 | 0.000 | -0.125 | 0.037 | 0.000 | 0.000 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u1106` | 1 | 1 | 5.000 | 0.000 | 5.000 | 0.013 | 0.000 | -0.049 | 0.000 | -0.125 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12614` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u1219` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L25:u10` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L26:u15` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.021 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u5030` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.014 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |

## By Component Unit Kind

| model | group | rows | cases | single net | single resolved | single emerged | loo net loss | labels |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head:attn:L31` | 8 | 2 | -0.750 | 2.625 | 1.875 | 5.250 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 5, "unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L34` | 8 | 2 | -2.875 | 4.125 | 1.250 | 2.625 | `{"unit_closer_candidate_no_closure": 2, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 3, "unit_weak_reducer": 2}` |
| qwen3 | `mlp_channel:mlp:L35` | 24 | 2 | -5.458 | 6.708 | 1.250 | 1.375 | `{"unit_closer_candidate_no_closure": 3, "unit_neutral_or_mixed": 8, "unit_new_blocker_or_deformer": 4, "unit_weak_reducer": 9}` |
| qwen3 | `mlp_channel:mlp:L33` | 12 | 2 | -2.667 | 3.250 | 0.583 | 0.583 | `{"unit_neutral_or_mixed": 6, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 4}` |
| qwen3 | `mlp_channel:mlp:L34` | 24 | 2 | -1.958 | 3.000 | 1.042 | 0.292 | `{"unit_closer_candidate_no_closure": 2, "unit_neutral_or_mixed": 6, "unit_new_blocker_or_deformer": 6, "unit_weak_reducer": 10}` |
| qwen3 | `attention_head:attn:L35` | 8 | 2 | -1.500 | 2.000 | 0.500 | -1.500 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 3}` |
| glm4 | `attention_head:attn:L35` | 8 | 2 | -0.125 | 0.375 | 0.250 | 0.250 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 3}` |
| glm4 | `attention_head:attn:L29` | 8 | 2 | 0.250 | 0.250 | 0.500 | 0.250 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 4, "unit_weak_reducer": 2}` |
| glm4 | `mlp_channel:mlp:L34` | 12 | 2 | 0.083 | 0.000 | 0.083 | 0.167 | `{"unit_neutral_or_mixed": 11, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `attention_head:attn:L33` | 8 | 2 | 0.625 | 0.125 | 0.750 | 0.125 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 6, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L38` | 24 | 2 | 0.125 | 0.125 | 0.250 | 0.042 | `{"unit_neutral_or_mixed": 16, "unit_new_blocker_or_deformer": 6, "unit_weak_reducer": 2}` |
| glm4 | `mlp_channel:mlp:L27` | 12 | 2 | 0.583 | 0.250 | 0.833 | -0.500 | `{"unit_neutral_or_mixed": 5, "unit_new_blocker_or_deformer": 4, "unit_weak_reducer": 3}` |
| glm4 | `mlp_channel:mlp:L39` | 12 | 2 | 0.833 | 0.500 | 1.333 | -0.917 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 8, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 1}` |
| deepseek7b | `attention_head:attn:L27` | 8 | 2 | -2.500 | 2.625 | 0.125 | 1.625 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_weak_reducer": 6}` |
| deepseek7b | `attention_head:attn:L26` | 8 | 2 | -2.125 | 2.375 | 0.250 | 1.250 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 5}` |
| deepseek7b | `mlp_channel:mlp:L27` | 24 | 2 | -1.500 | 10.375 | 8.875 | -7.958 | `{"unit_closer_candidate_no_closure": 2, "unit_neutral_or_mixed": 7, "unit_new_blocker_or_deformer": 4, "unit_weak_reducer": 11}` |
| deepseek7b | `mlp_channel:mlp:L24` | 12 | 2 | -9.333 | 12.333 | 3.000 | -11.917 | `{"unit_closer_candidate_no_closure": 4, "unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 2, "unit_weak_reducer": 4}` |
| deepseek7b | `mlp_channel:mlp:L26` | 12 | 2 | 3.750 | 5.083 | 8.833 | -32.500 | `{"unit_closer_candidate_no_closure": 2, "unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 3, "unit_weak_reducer": 4}` |
| deepseek7b | `attention_head:attn:L19` | 8 | 2 | -16.625 | 20.375 | 3.750 | -55.500 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 1, "unit_weak_reducer": 6}` |
| deepseek7b | `attention_head:attn:L25` | 8 | 2 | -1.625 | 1.625 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 3, "unit_weak_reducer": 5}` |
