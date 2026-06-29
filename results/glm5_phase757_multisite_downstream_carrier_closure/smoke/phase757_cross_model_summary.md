# Phase 757 Multi-Site Downstream Carrier Closure Test (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source removal followed by single-site, primary multi-site, and off-path multi-site component restores.

## Combo Kind Baseline

| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `off_path_control` | 2 | 0.250 | 0.000 | 0.167 | 0.062 | `{'anti_restore_or_off_path': 1, 'off_path_control_suspicious': 1}` |
| qwen3 | `same_layer_primary_pair` | 2 | 0.000 | -0.344 | -2.000 | 0.000 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 1}` |
| qwen3 | `single_primary_site` | 4 | 0.000 | -0.172 | -0.417 | 0.031 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 2}` |
| glm4 | `single_primary_site` | 8 | 0.000 | 0.004 | 0.250 | 0.008 | `{'weak_or_unclear': 8}` |
| deepseek7b | `single_primary_site` | 8 | 0.125 | -0.035 | 0.021 | 0.164 | `{'anti_restore_or_off_path': 4, 'multisite_target_carrier_candidate': 1, 'weak_or_unclear': 3}` |

## Top Multi-Site Restores

| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 2 | 0.500 | 0.188 | 0.062 | 0.333 | 0.000 | `off_path_control_suspicious` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L34:attn_out` | `['L34:attn_out']` | 2 | 0.000 | 0.188 | 0.062 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn_out` | `['L34:attn_out']` | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.125 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 2 | 0.000 | 0.000 | -0.062 | 1.000 | 0.125 | `anti_restore_or_off_path` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 2 | 0.000 | 0.000 | -0.062 | 0.000 | 0.125 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 2 | 0.000 | 0.188 | -0.688 | -3.667 | -0.062 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 2 | 0.000 | 0.188 | -0.688 | -4.000 | -0.062 | `anti_restore_or_off_path` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L36:mlp_out` | `['L36:mlp_out']` | 2 | 0.000 | 0.062 | 0.062 | 1.000 | 0.031 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:attn_out` | `['L37:attn_out']` | 2 | 0.000 | 0.062 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L36:attn_out` | `['L36:attn_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L37:attn_out` | `['L37:attn_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L36:attn_out` | `['L36:attn_out']` | 2 | 0.000 | 0.062 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:mlp_out` | `['L37:mlp_out']` | 2 | 0.000 | 0.062 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L37:mlp_out` | `['L37:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L36:mlp_out` | `['L36:mlp_out']` | 2 | 0.000 | 0.000 | -0.031 | 0.000 | -0.062 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn_out` | `['L23:attn_out']` | 2 | 1.000 | 0.312 | 0.156 | 0.500 | -0.156 | `multisite_target_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L24:mlp_out` | `['L24:mlp_out']` | 2 | 0.000 | 0.312 | 0.031 | 0.083 | -0.031 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `L24:mlp_out` | `['L24:mlp_out']` | 2 | 0.000 | -0.219 | -0.031 | 0.000 | 0.281 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `L23:mlp_out` | `['L23:mlp_out']` | 2 | 0.000 | -0.219 | -0.031 | 0.000 | 0.250 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:mlp_out` | `['L23:mlp_out']` | 2 | 0.000 | 0.312 | -0.062 | -0.250 | -0.031 | `anti_restore_or_off_path` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L24:attn_out` | `['L24:attn_out']` | 2 | 0.000 | 0.312 | -0.062 | -0.250 | -0.031 | `anti_restore_or_off_path` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `L24:attn_out` | `['L24:attn_out']` | 2 | 0.000 | -0.219 | -0.094 | 0.000 | 0.562 | `anti_restore_or_off_path` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `L23:attn_out` | `['L23:attn_out']` | 2 | 0.000 | -0.219 | -0.188 | 0.000 | 0.469 | `anti_restore_or_off_path` |

## Strict Interpretation

- Multi-site restore is stronger than Phase 756 only if primary path combos beat off-path controls.
- If off-path controls recover similarly, the result is not a localized carrier.
- Weak multi-site restore points the next bottleneck toward readout threshold / phrase likelihood / generation closure.
