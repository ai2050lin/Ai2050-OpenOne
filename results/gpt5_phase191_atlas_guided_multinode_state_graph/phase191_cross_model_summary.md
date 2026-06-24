# Phase191 Cross-Model Summary

Objective: test whether atlas-selected multi-node MLP z-state graph interventions upgrade candidate-ranking evidence beyond Phase190 single-node/channel results.

All models used 128 confirm cases. Rows are target cases where base was wrong and repair prompt was correct.

## Model Overview

| model | target rows | nodes | time min | max signal switch | max control switch | max signal margin | max control margin | evidence update |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 11 | prompt_last/L34, query_category/L32, prompt_last/L32 | 1.88 | 1 | 1 | 0.1249 | 0.0682 | downgrade_or_hold_due_to_control_pollution |
| glm4 | 22 | prompt_last/L38, prompt_last/L37, prompt_last/L39 | 5.06 | 1 | 0 | 0.0000 | 0.0000 | hold_due_to_weak_margin |
| deepseek7b | 49 | rule_value/L26, query_relation/L19, prompt_last/L26 | 8.10 | 1 | 2 | 0.0197 | 0.0101 | downgrade_or_hold_due_to_control_pollution |

## qwen3 Best Signal

| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_last@L34+prompt_last@L32|combo_margin_top128|signal|a2` | signal | 2 | 1/11 | 0.1022 | -1.1708 | 0.8182 |
| `prompt_last@L34+query_category@L32|combo_margin_top128|signal|a2` | signal | 2 | 1/11 | 0.0681 | -0.9291 | 0.7273 |
| `prompt_last@L32|combo_correct_minus_old_top128|signal|a2` | signal | 1 | 1/11 | 0.0568 | 0.0133 | 0.7273 |
| `prompt_last@L34+query_category@L32|combo_correct_minus_old_top128|signal|a1` | signal | 2 | 1/11 | 0.0567 | -0.1600 | 0.6364 |
| `prompt_last@L34+prompt_last@L32|combo_margin_top128|signal|a1` | signal | 2 | 1/11 | 0.0567 | -0.2600 | 0.7273 |
| `query_category@L32+prompt_last@L32|combo_correct_minus_old_top128|signal|a2` | signal | 2 | 1/11 | 0.0455 | -0.0147 | 0.7273 |
| `query_category@L32+prompt_last@L32|combo_margin_top128|signal|a2` | signal | 2 | 1/11 | 0.0454 | -0.0976 | 0.4545 |
| `prompt_last@L32|combo_correct_minus_old_top128|signal|a1` | signal | 1 | 1/11 | 0.0454 | 0.0037 | 0.5455 |

## qwen3 Best Control

| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `query_category@L32+prompt_last@L32|shuffled_combo_margin_top128|shuffled_node_control|a2` | shuffled_node_control | 2 | 1/11 | 0.0682 | -0.2268 | 0.7273 |
| `prompt_last@L32|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 1 | 1/11 | 0.0682 | 0.0536 | 0.6364 |
| `query_category@L32+prompt_last@L32|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 2 | 1/11 | 0.0680 | 0.0623 | 0.6364 |
| `prompt_last@L34+query_category@L32+prompt_last@L32|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 3 | 1/11 | 0.0453 | -1.6197 | 0.5455 |
| `query_category@L32+prompt_last@L32|random_same_norm_top128|random_control|a2` | random_control | 2 | 1/11 | 0.0342 | -0.1046 | 0.8182 |
| `query_category@L32|wrong_relation_top128|wrong_relation_control|a1` | wrong_relation_control | 1 | 1/11 | 0.0341 | -0.0033 | 0.4545 |
| `prompt_last@L34+prompt_last@L32|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 2 | 1/11 | 0.0340 | -1.6356 | 0.6364 |
| `query_category@L32|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 1 | 1/11 | 0.0340 | -0.0004 | 0.5455 |

## qwen3 By Set Size

| set size | best signal switch | best signal margin | best control switch | best control margin |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1/11 | 0.0568 | 1/11 | 0.0682 |
| 2 | 1/11 | 0.1022 | 1/11 | 0.0682 |
| 3 | 1/11 | 0.0454 | 1/11 | 0.0453 |

## glm4 Best Signal

| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_last@L38+prompt_last@L37+prompt_last@L39|combo_margin_top128|signal|a2` | signal | 3 | 1/22 | -0.0000 | -0.2215 | 0.4091 |
| `prompt_last@L38+prompt_last@L37|combo_margin_top128|signal|a2` | signal | 2 | 1/22 | -0.0000 | -0.0868 | 0.4091 |
| `prompt_last@L38+prompt_last@L39|raw_delta|signal|a2` | signal | 2 | 1/22 | -0.0057 | -0.1675 | 0.5000 |
| `prompt_last@L38|raw_delta|signal|a2` | signal | 1 | 1/22 | -0.0057 | 0.2728 | 0.5000 |
| `prompt_last@L38+prompt_last@L37+prompt_last@L39|combo_correct_minus_old_top128|signal|a2` | signal | 3 | 1/22 | -0.0114 | -0.1916 | 0.2727 |
| `prompt_last@L38+prompt_last@L37|combo_correct_minus_old_top128|signal|a2` | signal | 2 | 1/22 | -0.0114 | 0.0141 | 0.2727 |
| `prompt_last@L39|combo_correct_minus_old_top128|signal|a1` | signal | 1 | 0/22 | 0.0000 | 0.0486 | 0.0000 |
| `prompt_last@L39|combo_correct_minus_old_top128|signal|a2` | signal | 1 | 0/22 | 0.0000 | -0.1686 | 0.0000 |

## glm4 Best Control

| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_last@L37+prompt_last@L39|random_same_norm_top128|random_control|a2` | random_control | 2 | 0/22 | 0.0000 | -0.0466 | 0.2727 |
| `prompt_last@L37|random_same_norm_top128|random_control|a2` | random_control | 1 | 0/22 | 0.0000 | 0.0007 | 0.2727 |
| `prompt_last@L39|random_same_norm_top128|random_control|a1` | random_control | 1 | 0/22 | 0.0000 | 0.0021 | 0.0000 |
| `prompt_last@L39|random_same_norm_top128|random_control|a2` | random_control | 1 | 0/22 | 0.0000 | -0.0414 | 0.0000 |
| `prompt_last@L39|wrong_relation_top128|wrong_relation_control|a1` | wrong_relation_control | 1 | 0/22 | 0.0000 | 0.0620 | 0.0000 |
| `prompt_last@L39|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 1 | 0/22 | 0.0000 | -0.1054 | 0.0000 |
| `prompt_last@L38+prompt_last@L37+prompt_last@L39|random_same_norm_top128|random_control|a1` | random_control | 3 | 0/22 | -0.0000 | 0.0163 | 0.3636 |
| `prompt_last@L38+prompt_last@L37|random_same_norm_top128|random_control|a1` | random_control | 2 | 0/22 | -0.0000 | 0.0134 | 0.3636 |

## glm4 By Set Size

| set size | best signal switch | best signal margin | best control switch | best control margin |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1/22 | -0.0057 | 0/22 | 0.0000 |
| 2 | 1/22 | -0.0000 | 0/22 | 0.0000 |
| 3 | 1/22 | -0.0000 | 0/22 | -0.0000 |

## deepseek7b Best Signal

| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rule_value@L26+query_relation@L19|combo_margin_top128|signal|a2` | signal | 2 | 1/49 | 0.0197 | -0.1435 | 0.6327 |
| `rule_value@L26+query_relation@L19|raw_delta|signal|a1` | signal | 2 | 1/49 | 0.0130 | 0.2685 | 0.6122 |
| `rule_value@L26+query_relation@L19+prompt_last@L26|raw_delta|signal|a1` | signal | 3 | 1/49 | 0.0100 | 0.2409 | 0.5714 |
| `query_relation@L19|raw_delta|signal|a1` | signal | 1 | 1/49 | 0.0070 | 0.2686 | 0.5510 |
| `rule_value@L26+prompt_last@L26|raw_delta|signal|a1` | signal | 2 | 1/49 | 0.0009 | -0.0312 | 0.4898 |
| `query_relation@L19+prompt_last@L26|raw_delta|signal|a1` | signal | 2 | 1/49 | -0.0014 | 0.2389 | 0.5510 |
| `rule_value@L26|raw_delta|signal|a2` | signal | 1 | 1/49 | -0.0024 | -0.0021 | 0.5510 |
| `prompt_last@L26|raw_delta|signal|a1` | signal | 1 | 1/49 | -0.0024 | -0.0330 | 0.5102 |

## deepseek7b Best Control

| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `query_relation@L19|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 1 | 2/49 | 0.0070 | 0.0251 | 0.5306 |
| `query_relation@L19+prompt_last@L26|wrong_relation_top128|wrong_relation_control|a2` | wrong_relation_control | 2 | 2/49 | 0.0026 | 0.6332 | 0.4694 |
| `query_relation@L19|random_same_norm_top128|random_control|a1` | random_control | 1 | 1/49 | 0.0035 | -0.0553 | 0.4490 |
| `query_relation@L19|wrong_relation_top128|wrong_relation_control|a1` | wrong_relation_control | 1 | 1/49 | 0.0018 | 0.0255 | 0.5510 |
| `rule_value@L26+query_relation@L19|random_same_norm_top128|random_control|a1` | random_control | 2 | 1/49 | 0.0004 | -0.0538 | 0.4490 |
| `query_relation@L19+prompt_last@L26|random_same_norm_top128|random_control|a2` | random_control | 2 | 1/49 | -0.0016 | -0.1646 | 0.5510 |
| `query_relation@L19+prompt_last@L26|random_same_norm_top128|random_control|a1` | random_control | 2 | 1/49 | -0.0018 | -0.0594 | 0.4082 |
| `rule_value@L26+query_relation@L19+prompt_last@L26|random_same_norm_top128|random_control|a1` | random_control | 3 | 1/49 | -0.0039 | -0.0589 | 0.4490 |

## deepseek7b By Set Size

| set size | best signal switch | best signal margin | best control switch | best control margin |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1/49 | 0.0070 | 2/49 | 0.0070 |
| 2 | 1/49 | 0.0197 | 2/49 | 0.0026 |
| 3 | 1/49 | 0.0100 | 1/49 | -0.0039 |

## Objective Reading

- The uploaded interpretation is correct: Phase191 is not a retreat from the atlas, but atlas-guided causal drilling with explicit evidence update.
- Multi-node signal did not cleanly beat controls across models. Qwen3 and GLM4 had signal switches, but controls remained comparable; DS7B's best switch was a wrong-relation control.
- Evidence should not upgrade to Level5. The candidate-ranking edge should hold as weak Level4 candidate at best, with control pollution and static z-patch insufficiency recorded as failure types.
- The next gap is not more static node/channel patching; it is forward dynamics/state-transition testing where downstream winner selection is measured over the full autoregressive trajectory.
