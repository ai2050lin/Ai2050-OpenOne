# Phase 809 Late-Layer Head/Channel Closer Decomposition (smoke)

- Status: `complete`
- Boundary: unit-level candidates for closure-solver input, not final token closure.

## By Unit

| model | unit | rows | cases | single net | single resolved | single emerged | emergence rate | single bias | fmt supp | loo net loss | loo bias loss | closure | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `mlp_channel:mlp:L35:u935` | 1 | 1 | -4.000 | 5.000 | 1.000 | 0.037 | -0.062 | 0.726 | 4.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| qwen3 | `attention_head:attn:L35:u0` | 1 | 1 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.198 | 1.000 | -0.062 | 0.000 | `{"unit_weak_reducer": 1}` |
| qwen3 | `attention_head:attn:L35:u27` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.094 | -1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u1147` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | -0.035 | -1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u991` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.014 | -1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L35:u548` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | -0.024 | -1.000 | -0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u265` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | -0.132 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u3304` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1166` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u219` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.010 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L33:u1275` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.017 | -1.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `attention_head:attn:L35:u26` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | 0.038 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u1028` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | 0.007 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u3372` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.062 | 0.007 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| qwen3 | `mlp_channel:mlp:L34:u198` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.010 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7968` | 1 | 1 | 7.000 | 0.000 | 7.000 | 0.074 | 0.062 | -0.219 | -7.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L39:u4744` | 1 | 1 | 7.000 | 0.000 | 7.000 | 0.074 | 0.062 | -0.231 | -10.000 | -0.062 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u12913` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.041 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L34:u7327` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.044 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u12358` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | -0.005 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27:u10600` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4526` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.008 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u4805` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.048 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L38:u5084` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.019 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u7043` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L39:u5302` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.019 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u8761` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.024 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u5668` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L34:u1917` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u1012` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| glm4 | `mlp_channel:mlp:L27:u11792` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | -0.031 | 0.015 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u2295` | 1 | 1 | -7.000 | 9.000 | 2.000 | 0.006 | -0.031 | 0.059 | 7.000 | 0.125 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u826` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.003 | 0.000 | -0.001 | 7.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u12909` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.003 | 0.000 | 0.008 | 7.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u5378` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.003 | 0.000 | 0.010 | 7.000 | 0.031 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u1787` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.006 | 0.062 | -0.003 | 7.000 | 0.094 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u4816` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.006 | 0.062 | -0.006 | 6.000 | 0.094 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L19:u13` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.003 | 0.000 | -0.001 | 6.000 | 0.094 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `attention_head:attn:L19:u0` | 1 | 1 | -5.000 | 6.000 | 1.000 | 0.003 | -0.031 | -0.009 | 4.000 | 0.000 | 0.000 | `{"unit_closer_candidate_no_closure": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u16013` | 1 | 1 | 1.000 | 0.000 | 1.000 | 0.003 | 0.000 | -0.000 | 4.000 | 0.094 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u4514` | 1 | 1 | 2.000 | 0.000 | 2.000 | 0.006 | 0.000 | -0.009 | 4.000 | 0.094 | 0.000 | `{"unit_new_blocker_or_deformer": 1}` |
| deepseek7b | `attention_head:attn:L19:u1` | 1 | 1 | -3.000 | 4.000 | 1.000 | 0.003 | -0.031 | 0.007 | 2.000 | 0.094 | 0.000 | `{"unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L24:u15099` | 1 | 1 | 0.000 | 2.000 | 2.000 | 0.006 | 0.000 | -0.003 | 2.000 | 0.062 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u9394` | 1 | 1 | 0.000 | 1.000 | 1.000 | 0.003 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L27:u16230` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |
| deepseek7b | `mlp_channel:mlp:L26:u1219` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 1}` |

## By Component Unit Kind

| model | group | rows | cases | single net | single resolved | single emerged | loo net loss | labels |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `mlp_channel:mlp:L35` | 4 | 1 | -1.000 | 1.250 | 0.250 | 0.250 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 3}` |
| qwen3 | `mlp_channel:mlp:L34` | 4 | 1 | 0.000 | 0.000 | 0.000 | -0.250 | `{"unit_neutral_or_mixed": 4}` |
| qwen3 | `mlp_channel:mlp:L33` | 4 | 1 | 0.000 | 0.000 | 0.000 | -1.000 | `{"unit_neutral_or_mixed": 4}` |
| qwen3 | `attention_head:attn:L35` | 3 | 1 | -0.667 | 0.667 | 0.000 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_weak_reducer": 1}` |
| glm4 | `mlp_channel:mlp:L39` | 4 | 1 | 3.500 | 0.000 | 3.500 | -4.250 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 2}` |
| glm4 | `mlp_channel:mlp:L38` | 4 | 1 | 0.250 | 0.000 | 0.250 | 0.000 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L34` | 4 | 1 | 0.250 | 0.000 | 0.250 | 0.000 | `{"unit_neutral_or_mixed": 3, "unit_new_blocker_or_deformer": 1}` |
| glm4 | `mlp_channel:mlp:L27` | 4 | 1 | 0.500 | 0.000 | 0.500 | 0.000 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 2}` |
| deepseek7b | `mlp_channel:mlp:L27` | 4 | 1 | -1.250 | 2.250 | 1.000 | 5.250 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 2}` |
| deepseek7b | `mlp_channel:mlp:L24` | 4 | 1 | 1.500 | 0.500 | 2.000 | 4.750 | `{"unit_neutral_or_mixed": 1, "unit_new_blocker_or_deformer": 3}` |
| deepseek7b | `attention_head:attn:L19` | 3 | 1 | -2.667 | 3.667 | 1.000 | 4.000 | `{"unit_closer_candidate_no_closure": 1, "unit_neutral_or_mixed": 1, "unit_weak_reducer": 1}` |
| deepseek7b | `mlp_channel:mlp:L26` | 4 | 1 | 0.500 | 0.250 | 0.750 | 2.750 | `{"unit_neutral_or_mixed": 2, "unit_new_blocker_or_deformer": 2}` |
