# Phase 756 Cross-Domain Writer Control and Downstream Carrier Test (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: fixed source removal vs same-layer controls, then downstream component restoration under the same removal.

## Candidate vs Control Baseline

| model | candidate kind | groups | mean support | mean drop | mean guard | mean release | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | 4 | 0.177 | 0.062 | 0.271 | 0.132 | `{'control_or_weak': 4}` |
| qwen3 | `same_layer_control_head` | 4 | 0.073 | 0.034 | 0.177 | 0.079 | `{'control_or_weak': 4}` |
| glm4 | `phase755_top_candidate` | 4 | 0.042 | -0.007 | 0.104 | 0.106 | `{'control_or_weak': 4}` |
| glm4 | `same_layer_control_head` | 4 | 0.000 | 0.010 | 0.000 | 0.023 | `{'control_or_weak': 4}` |
| deepseek7b | `phase755_top_candidate` | 4 | 0.750 | 0.465 | 0.250 | 0.155 | `{'cross_domain_writer_candidate': 3, 'cross_domain_writer_guard_candidate': 1}` |
| deepseek7b | `same_layer_control_head` | 4 | 0.083 | 0.001 | 0.271 | 0.184 | `{'control_or_weak': 4}` |

## Top Controlled Writer / Guard Candidates

| model | kind | site | head | source | n | domains | support | drop | guard | release | top1 loss | guess |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out | 23 | records_all | 24 | 6 | 0.292 | 0.156 | 0.167 | 0.083 | 0.042 | `control_or_weak` |
| qwen3 | `phase755_top_candidate` | L33:attn_out | 23 | target_record_line | 24 | 6 | 0.250 | 0.130 | 0.125 | 0.073 | 0.042 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 4 | records_all | 24 | 6 | 0.125 | 0.031 | 0.125 | 0.057 | 0.000 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 28 | records_all | 24 | 6 | 0.083 | 0.052 | 0.208 | 0.089 | 0.042 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 28 | target_record_line | 24 | 6 | 0.083 | 0.031 | 0.208 | 0.099 | 0.042 | `control_or_weak` |
| qwen3 | `phase755_top_candidate` | L33:attn_out | 15 | records_all | 24 | 6 | 0.083 | -0.016 | 0.417 | 0.188 | 0.000 | `control_or_weak` |
| qwen3 | `phase755_top_candidate` | L33:attn_out | 15 | target_record_line | 24 | 6 | 0.083 | -0.021 | 0.375 | 0.182 | 0.000 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 4 | target_record_line | 24 | 6 | 0.000 | 0.021 | 0.167 | 0.073 | 0.042 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L35:attn_out | 29 | records_all | 24 | 6 | 0.042 | 0.034 | 0.083 | 0.061 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L35:attn_out | 29 | target_record_line | 24 | 6 | 0.042 | 0.023 | 0.042 | 0.048 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L34:attn_out | 4 | target_record_line | 24 | 6 | 0.042 | -0.042 | 0.167 | 0.152 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L34:attn_out | 4 | records_all | 24 | 6 | 0.042 | -0.044 | 0.125 | 0.164 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L34:attn_out | 17 | target_record_line | 24 | 6 | 0.000 | 0.013 | 0.000 | 0.030 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L34:attn_out | 17 | records_all | 24 | 6 | 0.000 | 0.013 | 0.000 | 0.029 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L35:attn_out | 10 | target_record_line | 24 | 6 | 0.000 | 0.010 | 0.000 | 0.010 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L35:attn_out | 10 | records_all | 24 | 6 | 0.000 | 0.003 | 0.000 | 0.022 | 0.000 | `control_or_weak` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 1 | records_all | 24 | 6 | 0.792 | 0.523 | 0.208 | 0.130 | 0.125 | `cross_domain_writer_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 24 | records_all | 24 | 6 | 0.792 | 0.495 | 0.250 | 0.169 | 0.208 | `cross_domain_writer_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 24 | target_record_line | 24 | 6 | 0.750 | 0.419 | 0.208 | 0.138 | 0.208 | `cross_domain_writer_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 1 | target_record_line | 24 | 6 | 0.667 | 0.424 | 0.333 | 0.182 | 0.167 | `cross_domain_writer_guard_candidate` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 9 | records_all | 24 | 6 | 0.125 | -0.016 | 0.250 | 0.177 | 0.000 | `control_or_weak` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 14 | records_all | 24 | 6 | 0.083 | 0.010 | 0.208 | 0.185 | 0.000 | `control_or_weak` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 9 | target_record_line | 24 | 6 | 0.083 | -0.003 | 0.333 | 0.214 | 0.000 | `control_or_weak` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 14 | target_record_line | 24 | 6 | 0.042 | 0.010 | 0.292 | 0.161 | 0.000 | `control_or_weak` |

## Top Downstream Carrier Restores

| model | kind | writer | source | downstream | n | restore rate | erase drop | restored drop | recovered | recovery frac | release reduced | guess |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L35:attn_out | 24 | 0.208 | 0.156 | 0.099 | 0.057 | 0.532 | -0.047 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | target_record_line | L35:attn_out | 24 | 0.208 | 0.130 | 0.094 | 0.036 | 0.266 | -0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L35:attn_out | 24 | 0.125 | 0.031 | -0.021 | 0.052 | 0.786 | -0.068 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L35:mlp_out | 24 | 0.125 | 0.031 | 0.005 | 0.026 | 0.714 | 0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L34:attn_out | 24 | 0.125 | 0.031 | 0.042 | -0.010 | 0.357 | -0.026 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | L35:mlp_out | 24 | 0.083 | -0.016 | -0.036 | 0.021 | 1.167 | 0.073 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | L34:attn_out | 24 | 0.083 | 0.031 | 0.010 | 0.021 | 0.900 | 0.016 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | L35:mlp_out | 24 | 0.083 | -0.021 | -0.031 | 0.010 | 0.750 | 0.068 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | L34:mlp_out | 24 | 0.083 | 0.031 | 0.031 | 0.000 | 0.571 | -0.031 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L35:mlp_out | 24 | 0.083 | 0.052 | -0.005 | 0.057 | 0.417 | -0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | L34:mlp_out | 24 | 0.083 | 0.031 | 0.031 | 0.000 | 0.400 | 0.000 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L34:mlp_out | 24 | 0.083 | 0.052 | 0.047 | 0.005 | 0.333 | -0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | L35:mlp_out | 24 | 0.083 | 0.031 | 0.010 | 0.021 | 0.300 | 0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L35:attn_out | 24 | 0.083 | 0.052 | 0.062 | -0.010 | 0.208 | 0.031 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L34:attn_out | 24 | 0.083 | 0.156 | 0.135 | 0.021 | 0.148 | -0.026 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | L34:mlp_out | 24 | 0.083 | 0.156 | 0.359 | -0.203 | -1.925 | -0.026 | `anti_restore_or_off_path` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | L36:attn_out | 24 | 0.042 | -0.044 | -0.036 | -0.008 | 0.333 | 0.010 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L36:mlp_out | 24 | 0.000 | 0.003 | 0.003 | 0.000 | 1.000 | 0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L37:attn_out | 24 | 0.000 | 0.013 | -0.003 | 0.016 | 0.667 | -0.013 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L36:mlp_out | 24 | 0.000 | 0.013 | 0.005 | 0.008 | 0.667 | -0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | L36:attn_out | 24 | 0.000 | 0.013 | 0.010 | 0.003 | 0.667 | 0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | L37:attn_out | 24 | 0.000 | 0.010 | 0.008 | 0.003 | 0.667 | -0.022 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | L36:mlp_out | 24 | 0.000 | 0.010 | 0.013 | -0.003 | 0.667 | -0.007 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | L36:mlp_out | 24 | 0.000 | 0.013 | 0.010 | 0.003 | 0.600 | 0.010 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | L37:mlp_out | 24 | 0.000 | 0.013 | 0.016 | -0.003 | 0.600 | 0.000 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L36:attn_out | 24 | 0.000 | 0.003 | 0.003 | 0.000 | 0.500 | -0.018 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | target_record_line | L36:attn_out | 24 | 0.000 | 0.023 | 0.026 | -0.003 | 0.467 | 0.023 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | L36:attn_out | 24 | 0.000 | 0.010 | 0.003 | 0.008 | 0.333 | -0.020 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | L37:mlp_out | 24 | 0.000 | 0.010 | 0.013 | -0.003 | 0.333 | -0.022 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | target_record_line | L37:attn_out | 24 | 0.000 | -0.042 | -0.031 | -0.010 | 0.333 | 0.018 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | L37:mlp_out | 24 | 0.000 | 0.034 | 0.029 | 0.005 | 0.271 | 0.005 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | L37:attn_out | 24 | 0.000 | 0.034 | 0.036 | -0.003 | 0.250 | 0.021 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | L24:mlp_out | 24 | 0.333 | 0.419 | 0.380 | 0.039 | 0.118 | -0.049 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | L23:attn_out | 24 | 0.333 | 0.495 | 0.466 | 0.029 | 0.012 | -0.049 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L24:mlp_out | 24 | 0.292 | 0.523 | 0.435 | 0.089 | 0.297 | -0.049 | `partial_downstream_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | L23:attn_out | 24 | 0.292 | 0.419 | 0.419 | 0.000 | 0.139 | -0.044 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L23:mlp_out | 24 | 0.250 | 0.424 | 0.362 | 0.062 | 0.196 | -0.029 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L23:attn_out | 24 | 0.250 | 0.424 | 0.396 | 0.029 | 0.012 | -0.010 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L24:mlp_out | 24 | 0.208 | 0.424 | 0.354 | 0.070 | 0.178 | 0.013 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L23:mlp_out | 24 | 0.167 | 0.523 | 0.464 | 0.060 | 0.163 | -0.031 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | L24:attn_out | 24 | 0.167 | 0.424 | 0.380 | 0.044 | 0.084 | -0.047 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | L23:mlp_out | 24 | 0.125 | -0.016 | -0.016 | 0.000 | 0.500 | -0.018 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | L23:attn_out | 24 | 0.125 | -0.016 | 0.000 | -0.016 | 0.200 | 0.021 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | L24:attn_out | 24 | 0.125 | 0.523 | 0.510 | 0.013 | 0.020 | -0.021 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | target_record_line | L23:mlp_out | 24 | 0.083 | -0.003 | 0.003 | -0.005 | 0.938 | 0.070 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H14 | records_all | L23:mlp_out | 24 | 0.083 | 0.010 | 0.026 | -0.016 | 0.812 | 0.039 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H14 | records_all | L24:attn_out | 24 | 0.083 | 0.010 | -0.026 | 0.036 | 0.750 | -0.060 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H14 | records_all | L23:attn_out | 24 | 0.083 | 0.010 | -0.005 | 0.016 | 0.625 | -0.029 | `weak_or_unclear` |

## Strict Interpretation

- A candidate stronger than same-layer controls supports specificity, not universality.
- Downstream restore replaces the whole downstream component output at the answer position; it localizes a coarse carrier, not a neuron-level code.
- If qwen3 / GLM4 remain weak, DS7B results must stay model-local.
