# Phase 756 Cross-Domain Writer Control and Downstream Carrier Test (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: fixed source removal vs same-layer controls, then downstream component restoration under the same removal.

## Candidate vs Control Baseline

| model | candidate kind | groups | mean support | mean drop | mean guard | mean release | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | 4 | 0.146 | 0.058 | 0.281 | 0.131 | `{'control_or_weak': 4}` |
| qwen3 | `same_layer_control_head` | 4 | 0.083 | 0.034 | 0.188 | 0.090 | `{'control_or_weak': 4}` |
| glm4 | `phase755_top_candidate` | 4 | 0.042 | 0.002 | 0.062 | 0.073 | `{'control_or_weak': 4}` |
| glm4 | `same_layer_control_head` | 4 | 0.000 | 0.007 | 0.010 | 0.029 | `{'control_or_weak': 4}` |
| deepseek7b | `phase755_top_candidate` | 4 | 0.771 | 0.449 | 0.339 | 0.207 | `{'cross_domain_writer_guard_candidate': 4}` |
| deepseek7b | `same_layer_control_head` | 4 | 0.073 | 0.004 | 0.281 | 0.190 | `{'control_or_weak': 4}` |

## Top Controlled Writer / Guard Candidates

| model | kind | site | head | source | n | domains | support | drop | guard | release | top1 loss | guess |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out | 23 | records_all | 48 | 6 | 0.250 | 0.125 | 0.208 | 0.094 | 0.021 | `control_or_weak` |
| qwen3 | `phase755_top_candidate` | L33:attn_out | 23 | target_record_line | 48 | 6 | 0.188 | 0.107 | 0.167 | 0.078 | 0.042 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 4 | records_all | 48 | 6 | 0.188 | 0.047 | 0.104 | 0.078 | 0.021 | `control_or_weak` |
| qwen3 | `phase755_top_candidate` | L33:attn_out | 15 | target_record_line | 48 | 6 | 0.083 | 0.005 | 0.333 | 0.161 | 0.021 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 28 | records_all | 48 | 6 | 0.062 | 0.049 | 0.188 | 0.073 | 0.021 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 28 | target_record_line | 48 | 6 | 0.062 | 0.029 | 0.229 | 0.102 | 0.021 | `control_or_weak` |
| qwen3 | `phase755_top_candidate` | L33:attn_out | 15 | records_all | 48 | 6 | 0.062 | -0.005 | 0.417 | 0.190 | 0.000 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 4 | target_record_line | 48 | 6 | 0.021 | 0.010 | 0.229 | 0.107 | 0.021 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L35:attn_out | 29 | records_all | 48 | 6 | 0.042 | 0.025 | 0.042 | 0.048 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L35:attn_out | 29 | target_record_line | 48 | 6 | 0.042 | 0.016 | 0.042 | 0.041 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L34:attn_out | 4 | records_all | 48 | 6 | 0.042 | -0.016 | 0.083 | 0.109 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L34:attn_out | 4 | target_record_line | 48 | 6 | 0.042 | -0.017 | 0.083 | 0.094 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L34:attn_out | 17 | records_all | 48 | 6 | 0.000 | 0.013 | 0.000 | 0.032 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L34:attn_out | 17 | target_record_line | 48 | 6 | 0.000 | 0.010 | 0.000 | 0.031 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L35:attn_out | 10 | target_record_line | 48 | 6 | 0.000 | 0.003 | 0.021 | 0.026 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L35:attn_out | 10 | records_all | 48 | 6 | 0.000 | 0.001 | 0.021 | 0.027 | 0.000 | `control_or_weak` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 24 | records_all | 48 | 6 | 0.854 | 0.500 | 0.333 | 0.212 | 0.146 | `cross_domain_writer_guard_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 24 | target_record_line | 48 | 6 | 0.812 | 0.428 | 0.333 | 0.224 | 0.146 | `cross_domain_writer_guard_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 1 | records_all | 48 | 6 | 0.771 | 0.480 | 0.333 | 0.188 | 0.104 | `cross_domain_writer_guard_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 1 | target_record_line | 48 | 6 | 0.646 | 0.385 | 0.354 | 0.206 | 0.125 | `cross_domain_writer_guard_candidate` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 9 | target_record_line | 48 | 6 | 0.104 | 0.005 | 0.312 | 0.211 | 0.000 | `control_or_weak` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 14 | target_record_line | 48 | 6 | 0.083 | 0.013 | 0.333 | 0.202 | 0.000 | `control_or_weak` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 9 | records_all | 48 | 6 | 0.062 | -0.021 | 0.271 | 0.184 | 0.000 | `control_or_weak` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 14 | records_all | 48 | 6 | 0.042 | 0.017 | 0.208 | 0.163 | 0.000 | `control_or_weak` |

## Top Downstream Carrier Restores

| model | kind | writer | source | downstream | n | restore rate | erase drop | restored drop | recovered | recovery frac | release reduced | guess |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L35:attn_out | 48 | 0.188 | 0.047 | -0.026 | 0.073 | 0.833 | -0.086 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L35:mlp_out | 48 | 0.167 | 0.047 | 0.021 | 0.026 | 0.630 | 0.044 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L35:attn_out | 48 | 0.167 | 0.125 | 0.086 | 0.039 | 0.371 | -0.018 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L34:mlp_out | 48 | 0.125 | 0.047 | 0.036 | 0.010 | 0.519 | -0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L34:attn_out | 48 | 0.125 | 0.047 | 0.042 | 0.005 | 0.426 | 0.008 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | target_record_line | L35:attn_out | 48 | 0.125 | 0.107 | 0.068 | 0.039 | 0.301 | -0.016 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L34:attn_out | 48 | 0.083 | 0.125 | 0.117 | 0.008 | 0.183 | -0.034 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L35:mlp_out | 48 | 0.083 | 0.125 | 0.336 | -0.211 | -0.261 | 0.029 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L34:mlp_out | 48 | 0.083 | 0.125 | 0.331 | -0.206 | -2.188 | -0.042 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | L35:mlp_out | 48 | 0.062 | 0.005 | -0.018 | 0.023 | 0.875 | 0.052 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | L34:attn_out | 48 | 0.062 | 0.029 | 0.026 | 0.003 | 0.719 | 0.026 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | L35:mlp_out | 48 | 0.062 | 0.029 | 0.005 | 0.023 | 0.625 | 0.013 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L35:mlp_out | 48 | 0.062 | 0.049 | 0.013 | 0.036 | 0.542 | -0.023 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | L34:mlp_out | 48 | 0.062 | 0.029 | 0.023 | 0.005 | 0.500 | 0.000 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L35:attn_out | 48 | 0.062 | 0.049 | 0.049 | 0.000 | 0.333 | 0.003 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | L35:mlp_out | 48 | 0.042 | -0.005 | -0.036 | 0.031 | 1.131 | 0.057 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | L36:attn_out | 48 | 0.021 | -0.016 | -0.010 | -0.005 | 0.190 | 0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L36:mlp_out | 48 | 0.000 | 0.001 | 0.001 | 0.000 | 0.667 | 0.004 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | L36:mlp_out | 48 | 0.000 | 0.010 | 0.001 | 0.009 | 0.625 | -0.007 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | L37:mlp_out | 48 | 0.000 | 0.010 | 0.004 | 0.007 | 0.625 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L36:mlp_out | 48 | 0.000 | 0.013 | 0.004 | 0.009 | 0.556 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L37:attn_out | 48 | 0.000 | 0.001 | 0.005 | -0.004 | 0.500 | -0.006 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L36:attn_out | 48 | 0.000 | 0.001 | 0.007 | -0.005 | 0.500 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L36:attn_out | 48 | 0.000 | 0.013 | 0.007 | 0.007 | 0.444 | -0.002 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | L37:attn_out | 48 | 0.000 | 0.003 | 0.007 | -0.004 | 0.400 | -0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | L36:mlp_out | 48 | 0.000 | 0.003 | 0.007 | -0.004 | 0.400 | -0.008 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | L37:mlp_out | 48 | 0.000 | -0.016 | -0.009 | -0.007 | 0.400 | 0.012 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | L36:attn_out | 48 | 0.000 | 0.010 | 0.008 | 0.003 | 0.375 | -0.007 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | L37:attn_out | 48 | 0.000 | 0.025 | 0.026 | -0.001 | 0.367 | 0.016 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | target_record_line | L36:attn_out | 48 | 0.000 | 0.016 | 0.020 | -0.004 | 0.345 | 0.007 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L37:attn_out | 48 | 0.000 | 0.013 | -0.005 | 0.018 | 0.333 | -0.015 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L37:mlp_out | 48 | 0.000 | 0.013 | 0.012 | 0.001 | 0.333 | 0.000 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | L23:attn_out | 48 | 0.354 | 0.500 | 0.466 | 0.034 | 0.020 | -0.010 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | L23:attn_out | 48 | 0.312 | 0.428 | 0.383 | 0.046 | 0.267 | -0.039 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | L24:mlp_out | 48 | 0.292 | 0.428 | 0.388 | 0.040 | 0.067 | -0.036 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L23:mlp_out | 48 | 0.250 | 0.385 | 0.333 | 0.052 | 0.197 | -0.044 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L23:attn_out | 48 | 0.250 | 0.385 | 0.346 | 0.039 | 0.098 | -0.018 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L24:mlp_out | 48 | 0.229 | 0.385 | 0.342 | 0.043 | 0.119 | 0.008 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L24:mlp_out | 48 | 0.208 | 0.480 | 0.426 | 0.055 | 0.161 | -0.009 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | L23:mlp_out | 48 | 0.188 | 0.428 | 0.423 | 0.005 | 0.040 | 0.004 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L24:attn_out | 48 | 0.167 | 0.385 | 0.348 | 0.038 | 0.093 | -0.031 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L23:mlp_out | 48 | 0.167 | 0.480 | 0.436 | 0.044 | 0.084 | -0.018 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | L24:mlp_out | 48 | 0.167 | 0.500 | 0.536 | -0.036 | -0.183 | 0.031 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | L24:attn_out | 48 | 0.146 | 0.428 | 0.440 | -0.012 | 0.024 | -0.033 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L24:attn_out | 48 | 0.125 | 0.480 | 0.447 | 0.034 | 0.078 | -0.043 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L23:attn_out | 48 | 0.125 | 0.480 | 0.443 | 0.038 | 0.022 | -0.040 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | L23:mlp_out | 48 | 0.125 | 0.500 | 0.531 | -0.031 | -0.157 | 0.042 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | target_record_line | L24:attn_out | 48 | 0.104 | 0.005 | -0.012 | 0.017 | 0.750 | 0.004 | `weak_or_unclear` |

## Strict Interpretation

- A candidate stronger than same-layer controls supports specificity, not universality.
- Downstream restore replaces the whole downstream component output at the answer position; it localizes a coarse carrier, not a neuron-level code.
- If qwen3 / GLM4 remain weak, DS7B results must stay model-local.
