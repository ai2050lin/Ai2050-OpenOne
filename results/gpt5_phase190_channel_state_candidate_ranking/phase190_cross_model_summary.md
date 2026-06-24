# Phase190 Cross-Model Summary

Objective: test whether MLP z-channel groups at Phase594 candidate nodes can causally improve correct-vs-old-top-wrong candidate ranking.

All three models used confirm mode with 128 generated cases; rows below are the target subset where the base prompt was wrong and the repair prompt was correct.

## Model Overview

| model | target rows | nodes | time min | max non-control switch | max control switch | max non-control margin | max control margin |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | 11 | prompt_last/L34, query_category/L32, prompt_last/L32 | 4.56 | 1 | 1 | 0.1022 | 0.0681 |
| glm4 | 22 | prompt_last/L38, prompt_last/L37, prompt_last/L39 | 13.54 | 1 | 1 | 0.0085 | 0.0085 |
| deepseek7b | 49 | rule_value/L26, query_relation/L19, prompt_last/L26 | 23.66 | 2 | 2 | 0.0307 | 0.0126 |

## qwen3 Best Non-Control

| key | switch | margin_gain | common_delta | positive_margin_rate |
| --- | ---: | ---: | ---: | ---: |
| `prompt_last|L34|combo_margin_top256|a2` | 1/11 | 0.1022 | -0.8264 | 0.8182 |
| `prompt_last|L32|combo_margin_top256|a2` | 1/11 | 0.0909 | -0.1245 | 0.8182 |
| `prompt_last|L34|boost_margin_top256|a2` | 1/11 | 0.0680 | -1.0011 | 0.7273 |
| `prompt_last|L32|boost_margin_top64|a2` | 1/11 | 0.0569 | 0.0760 | 0.8182 |
| `prompt_last|L32|combo_correct_minus_old_top64|a1` | 1/11 | 0.0568 | -0.0039 | 0.7273 |
| `prompt_last|L32|combo_correct_minus_old_top256|a2` | 1/11 | 0.0568 | -0.0629 | 0.6364 |
| `prompt_last|L34|combo_correct_minus_old_top64|a2` | 1/11 | 0.0567 | -0.6448 | 0.6364 |
| `prompt_last|L34|combo_correct_minus_old_top256|a2` | 1/11 | 0.0567 | -0.8619 | 0.6364 |

## qwen3 Best Controls

| key | switch | margin_gain | common_delta | positive_margin_rate |
| --- | ---: | ---: | ---: | ---: |
| `prompt_last|L32|wrong_relation_delta_top64|a2` | 1/11 | 0.0681 | 0.0794 | 0.6364 |
| `prompt_last|L32|wrong_relation_delta_top256|a2` | 1/11 | 0.0567 | 0.0850 | 0.6364 |
| `query_category|L32|common_control_top16|a2` | 1/11 | 0.0454 | -0.0122 | 0.5455 |
| `query_category|L32|wrong_relation_delta_top64|a1` | 1/11 | 0.0341 | -0.0130 | 0.7273 |
| `prompt_last|L32|wrong_relation_delta_top64|a1` | 1/11 | 0.0341 | 0.0443 | 0.7273 |
| `prompt_last|L34|common_control_top256|a0.5` | 1/11 | 0.0340 | -0.4255 | 0.7273 |

## glm4 Best Non-Control

| key | switch | margin_gain | common_delta | positive_margin_rate |
| --- | ---: | ---: | ---: | ---: |
| `prompt_last|L37|boost_margin_top256|a2` | 1/22 | 0.0000 | -0.0025 | 0.3182 |
| `prompt_last|L37|combo_margin_top16|a2` | 0/22 | 0.0085 | 0.0188 | 0.5000 |
| `prompt_last|L37|combo_margin_top64|a0.5` | 0/22 | 0.0085 | 0.0240 | 0.4091 |
| `prompt_last|L37|remove_bad_margin_top16|a1` | 0/22 | 0.0085 | 0.0332 | 0.3182 |
| `prompt_last|L37|boost_margin_top256|a0.5` | 0/22 | 0.0057 | 0.0042 | 0.3636 |
| `prompt_last|L37|suppress_old_top16|a1` | 0/22 | 0.0057 | 0.0387 | 0.4091 |
| `prompt_last|L37|boost_correct_top64|a0.5` | 0/22 | 0.0057 | 0.0019 | 0.5000 |
| `prompt_last|L37|combo_correct_minus_old_top256|a0.5` | 0/22 | 0.0057 | 0.0355 | 0.4091 |

## glm4 Best Controls

| key | switch | margin_gain | common_delta | positive_margin_rate |
| --- | ---: | ---: | ---: | ---: |
| `prompt_last|L37|wrong_relation_delta_top64|a1` | 1/22 | 0.0000 | -0.1233 | 0.3636 |
| `prompt_last|L37|wrong_relation_delta_top256|a2` | 1/22 | -0.0028 | -0.3530 | 0.3182 |
| `prompt_last|L37|wrong_relation_delta_top16|a2` | 1/22 | -0.0057 | -0.1521 | 0.3636 |
| `prompt_last|L37|common_control_top64|a2` | 1/22 | -0.0057 | -0.1796 | 0.3636 |
| `prompt_last|L38|common_control_top16|a2` | 0/22 | 0.0085 | 0.2047 | 0.4091 |
| `prompt_last|L37|common_control_top64|a0.5` | 0/22 | 0.0057 | -0.0353 | 0.4545 |

## deepseek7b Best Non-Control

| key | switch | margin_gain | common_delta | positive_margin_rate |
| --- | ---: | ---: | ---: | ---: |
| `query_relation|L19|combo_correct_minus_old_top64|a1` | 2/49 | 0.0017 | -0.0217 | 0.5306 |
| `query_relation|L19|boost_margin_top256|a2` | 1/49 | 0.0307 | 0.0844 | 0.5918 |
| `query_relation|L19|combo_margin_top64|a2` | 1/49 | 0.0083 | -0.1212 | 0.5102 |
| `query_relation|L19|combo_margin_top256|a2` | 1/49 | 0.0082 | -0.1397 | 0.5306 |
| `rule_value|L26|suppress_old_top64|a0.5` | 1/49 | 0.0064 | -0.0012 | 0.5714 |
| `rule_value|L26|combo_margin_top16|a2` | 1/49 | 0.0063 | 0.0053 | 0.5918 |
| `prompt_last|L26|boost_margin_top16|a1` | 1/49 | 0.0054 | 0.1378 | 0.6327 |
| `prompt_last|L26|suppress_old_top64|a0.5` | 1/49 | 0.0030 | 0.0616 | 0.5306 |

## deepseek7b Best Controls

| key | switch | margin_gain | common_delta | positive_margin_rate |
| --- | ---: | ---: | ---: | ---: |
| `query_relation|L19|wrong_relation_delta_top16|a2` | 2/49 | 0.0126 | 0.0342 | 0.5510 |
| `query_relation|L19|random_margin_norm_top256|a2` | 2/49 | -0.0042 | -0.0711 | 0.5306 |
| `query_relation|L19|random_margin_norm_top16|a2` | 2/49 | -0.0053 | -0.0275 | 0.5510 |
| `query_relation|L19|common_control_top64|a2` | 2/49 | -0.0085 | 0.0982 | 0.4694 |
| `query_relation|L19|wrong_relation_delta_top64|a2` | 1/49 | 0.0064 | 0.0399 | 0.5306 |
| `query_relation|L19|wrong_relation_delta_top256|a2` | 1/49 | 0.0060 | -0.0259 | 0.4898 |

## Objective Reading

- Positive but weak causal signal: every model had at least one intervention that switched a target case, but switch rates stayed low.
- Control contamination is significant: random/wrong-relation/common controls also produced switches, especially DS7B query_relation L19. This prevents a strong claim that the selected channel groups are a clean candidate-ranking mechanism.
- The result supports a stricter interpretation of Phase189: Phase594 MLP nodes contain ranking-relevant channel directions, but channel selection by simple readout/top-k is not yet a stable repair rule.
