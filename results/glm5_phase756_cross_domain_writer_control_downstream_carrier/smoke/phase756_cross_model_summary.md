# Phase 756 Cross-Domain Writer Control and Downstream Carrier Test (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: fixed source removal vs same-layer controls, then downstream component restoration under the same removal.

## Candidate vs Control Baseline

| model | candidate kind | groups | mean support | mean drop | mean guard | mean release | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | 1 | 0.500 | 0.188 | 0.000 | 0.062 | `{'control_or_weak': 1}` |
| qwen3 | `same_layer_control_head` | 1 | 0.000 | 0.000 | 0.500 | 0.125 | `{'control_or_weak': 1}` |
| glm4 | `phase755_top_candidate` | 1 | 0.000 | 0.062 | 0.000 | 0.031 | `{'control_or_weak': 1}` |
| glm4 | `same_layer_control_head` | 1 | 0.000 | 0.000 | 0.000 | 0.031 | `{'control_or_weak': 1}` |
| deepseek7b | `phase755_top_candidate` | 1 | 1.000 | 0.312 | 0.000 | 0.000 | `{'partial_writer_candidate': 1}` |
| deepseek7b | `same_layer_control_head` | 1 | 0.000 | -0.219 | 1.000 | 0.750 | `{'route_guard_candidate': 1}` |

## Top Controlled Writer / Guard Candidates

| model | kind | site | head | source | n | domains | support | drop | guard | release | top1 loss | guess |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out | 15 | records_all | 2 | 2 | 0.500 | 0.188 | 0.000 | 0.062 | 0.000 | `control_or_weak` |
| qwen3 | `same_layer_control_head` | L33:attn_out | 28 | records_all | 2 | 2 | 0.000 | 0.000 | 0.500 | 0.125 | 0.000 | `control_or_weak` |
| glm4 | `phase755_top_candidate` | L35:attn_out | 29 | records_all | 2 | 2 | 0.000 | 0.062 | 0.000 | 0.031 | 0.000 | `control_or_weak` |
| glm4 | `same_layer_control_head` | L35:attn_out | 10 | records_all | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `control_or_weak` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out | 24 | records_all | 2 | 2 | 1.000 | 0.312 | 0.000 | 0.000 | 0.000 | `partial_writer_candidate` |
| deepseek7b | `same_layer_control_head` | L22:attn_out | 9 | records_all | 2 | 2 | 0.000 | -0.219 | 1.000 | 0.750 | 0.000 | `route_guard_candidate` |

## Top Downstream Carrier Restores

| model | kind | writer | source | downstream | n | restore rate | erase drop | restored drop | recovered | recovery frac | release reduced | guess |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L34:attn_out | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.125 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | L34:mlp_out | 2 | 0.000 | 0.000 | 0.062 | -0.062 | 1.000 | 0.125 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | L34:attn_out | 2 | 0.000 | 0.188 | 0.125 | 0.062 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | L34:mlp_out | 2 | 0.000 | 0.188 | 0.875 | -0.688 | -3.667 | -0.062 | `anti_restore_or_off_path` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | L36:mlp_out | 2 | 0.000 | 0.062 | 0.000 | 0.062 | 1.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L36:attn_out | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | L36:attn_out | 2 | 0.000 | 0.062 | 0.062 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | L36:mlp_out | 2 | 0.000 | 0.000 | 0.031 | -0.031 | 0.000 | -0.062 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | L23:attn_out | 2 | 1.000 | 0.312 | 0.156 | 0.156 | 0.500 | -0.156 | `downstream_target_carrier_candidate` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | L23:mlp_out | 2 | 0.000 | -0.219 | -0.188 | -0.031 | 0.000 | 0.250 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | L23:attn_out | 2 | 0.000 | -0.219 | -0.031 | -0.188 | 0.000 | 0.469 | `anti_restore_or_off_path` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | L23:mlp_out | 2 | 0.000 | 0.312 | 0.375 | -0.062 | -0.250 | -0.031 | `anti_restore_or_off_path` |

## Strict Interpretation

- A candidate stronger than same-layer controls supports specificity, not universality.
- Downstream restore replaces the whole downstream component output at the answer position; it localizes a coarse carrier, not a neuron-level code.
- If qwen3 / GLM4 remain weak, DS7B results must stay model-local.
