# Phase 758 Late Carrier Rewrite Relabel Test (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source removal followed by primary, late-candidate, joint, and true-late-control component restores.

## Combo Kind Baseline

| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `late_candidate_all` | 2 | 0.250 | 0.000 | 0.167 | 0.062 | `{'anti_restore_or_off_path': 1, 'partial_late_rewrite_candidate': 1}` |
| qwen3 | `primary_multisite_all` | 2 | 0.000 | -0.344 | -2.000 | 0.000 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 1}` |
| qwen3 | `primary_plus_late_all` | 2 | 0.000 | -0.344 | -1.333 | -0.031 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 1}` |
| qwen3 | `single_late_candidate_site` | 2 | 0.000 | 0.000 | 0.500 | 0.031 | `{'weak_or_unclear': 2}` |
| qwen3 | `single_primary_site` | 4 | 0.000 | -0.172 | -0.417 | 0.031 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | 2 | 0.000 | 0.031 | 1.000 | 0.016 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | 2 | 0.000 | 0.031 | 1.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_late_candidate_pair` | 2 | 0.000 | 0.031 | 1.000 | 0.016 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | 4 | 0.000 | 0.016 | 1.000 | 0.008 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `late_candidate_all` | 2 | 0.250 | -0.047 | 0.333 | 0.312 | `{'anti_restore_or_off_path': 1, 'partial_late_rewrite_candidate': 1}` |
| deepseek7b | `primary_multisite_all` | 2 | 0.250 | -0.016 | -0.167 | 0.062 | `{'weak_or_unclear': 2}` |
| deepseek7b | `primary_plus_late_all` | 2 | 0.500 | 0.078 | 1.000 | 0.141 | `{'anti_restore_or_off_path': 1, 'primary_late_joint_target_candidate': 1}` |
| deepseek7b | `same_layer_primary_pair` | 4 | 0.125 | -0.039 | -0.125 | 0.164 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `true_late_control` | 2 | 0.500 | 0.016 | 0.500 | 0.312 | `{'anti_restore_or_off_path': 1, 'true_late_control_suspicious': 1}` |

## Top Multi-Site Restores

| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 2 | 0.500 | 0.188 | 0.062 | 0.333 | 0.000 | `partial_late_rewrite_candidate` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L34:attn_out` | `['L34:attn_out']` | 2 | 0.000 | 0.188 | 0.062 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn_out` | `['L34:attn_out']` | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.125 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L35:attn_out` | `['L35:attn_out']` | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.125 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.062 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L35:attn_out` | `['L35:attn_out']` | 2 | 0.000 | 0.188 | 0.000 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 2 | 0.000 | 0.000 | -0.062 | 1.000 | 0.125 | `anti_restore_or_off_path` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 2 | 0.000 | 0.000 | -0.062 | 0.000 | 0.125 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 2 | 0.000 | 0.188 | -0.688 | -3.667 | -0.062 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 2 | 0.000 | 0.188 | -0.688 | -3.667 | -0.125 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 2 | 0.000 | 0.188 | -0.688 | -4.000 | -0.062 | `anti_restore_or_off_path` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 2 | 0.000 | 0.062 | 0.062 | 1.000 | 0.031 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L36:attn+mlp` | `['L36:attn_out', 'L36:mlp_out']` | 2 | 0.000 | 0.062 | 0.062 | 1.000 | 0.031 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 2 | 0.000 | 0.062 | 0.062 | 1.000 | 0.000 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L38:late_attn+mlp` | `['L38:attn_out', 'L38:mlp_out']` | 2 | 0.000 | 0.062 | 0.062 | 1.000 | 0.000 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 2 | 0.000 | 0.062 | 0.062 | 1.000 | -0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L36:attn+mlp` | `['L36:attn_out', 'L36:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L38:late_attn+mlp` | `['L38:attn_out', 'L38:mlp_out']` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 2 | 0.000 | 0.062 | 0.000 | 0.000 | -0.031 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 2 | 0.000 | 0.000 | -0.062 | 0.000 | 0.000 | `anti_restore_or_off_path` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 2 | 1.000 | 0.312 | 0.312 | 1.000 | -0.344 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `late_control_same_count` | `['L27:attn_out', 'L27:mlp_out']` | 2 | 1.000 | 0.312 | 0.156 | 0.500 | -0.094 | `true_late_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 2 | 0.500 | 0.312 | 0.094 | 0.333 | -0.125 | `partial_late_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 2 | 0.500 | 0.312 | 0.031 | 0.000 | -0.031 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 2 | 0.500 | 0.312 | 0.000 | -0.167 | -0.094 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 2 | 0.000 | -0.219 | -0.031 | 0.000 | 0.219 | `weak_or_unclear` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `L24:attn+mlp` | `['L24:attn_out', 'L24:mlp_out']` | 2 | 0.000 | -0.219 | -0.031 | 0.000 | 0.156 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L24:attn+mlp` | `['L24:attn_out', 'L24:mlp_out']` | 2 | 0.000 | 0.312 | -0.062 | -0.250 | 0.000 | `anti_restore_or_off_path` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 2 | 0.000 | -0.219 | -0.094 | 0.000 | 0.531 | `anti_restore_or_off_path` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `late_control_same_count` | `['L27:attn_out', 'L27:mlp_out']` | 2 | 0.000 | -0.219 | -0.125 | 0.000 | 0.719 | `anti_restore_or_off_path` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 2 | 0.000 | -0.219 | -0.156 | 0.000 | 0.625 | `anti_restore_or_off_path` |
| deepseek7b | `same_layer_control_head` | L22:attn_out:H9 | records_all | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 2 | 0.000 | -0.219 | -0.188 | 0.000 | 0.750 | `anti_restore_or_off_path` |

## Strict Interpretation

- Phase 758 relabels Phase 757 off-path recovery as a late carrier / rewrite candidate.
- Strong evidence requires late_candidate groups to beat primary path and true_late_control groups.
- If target recovery rises but route release is not reduced, the mechanism is target rewrite rather than route closure.
