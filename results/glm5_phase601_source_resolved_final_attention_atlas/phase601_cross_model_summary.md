# Phase601 Cross-Model Summary

Source-resolved final attention atlas.

## qwen3

cases=128, rows=11, target_cases_seen=11, probe_layer=35, alpha=2.0, time_min=0.62

### Largest Attention Deltas

| key | trajectory | n | max_group | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other | entropy | top_mass |
|---|---|---:|---||---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_category|L32|natural_correct` | natural_correct | 11 | `punct_newline` | 0.0043 | -0.0009 | 0.0149 | 0.0009 | 0.0000 | 0.0218 | 0.0000 | 0.0453 | -0.0411 | 1.3861 | 0.6336 |
| `query_category|L32|natural_wrong` | natural_wrong | 11 | `punct_newline` | 0.0040 | -0.0010 | 0.0149 | 0.0011 | 0.0000 | 0.0208 | 0.0000 | 0.0422 | -0.0398 | 1.3783 | 0.6357 |
| `query_category|L32|artificial_repair` | artificial_repair | 11 | `other` | -0.0004 | -0.0010 | 0.0000 | -0.0004 | 0.0000 | -0.0230 | 0.0000 | -0.0012 | 0.0246 | 1.3640 | 0.6830 |
| `query_category|L32|artificial_wrong` | artificial_wrong | 11 | `other` | -0.0004 | -0.0010 | 0.0000 | -0.0004 | 0.0000 | -0.0228 | 0.0000 | -0.0021 | 0.0245 | 1.3525 | 0.6847 |
| `prompt_last|L32|natural_correct` | natural_correct | 11 | `other` | 0.0021 | 0.0011 | 0.0026 | 0.0021 | -0.0005 | 0.0073 | 0.0020 | 0.0157 | -0.0168 | 1.2632 | 0.6722 |
| `prompt_last|L34|natural_correct` | natural_correct | 11 | `other` | 0.0021 | 0.0011 | 0.0026 | 0.0021 | -0.0005 | 0.0073 | 0.0020 | 0.0157 | -0.0168 | 1.2632 | 0.6722 |
| `prompt_last|L32|natural_wrong` | natural_wrong | 11 | `punct_newline` | 0.0025 | 0.0016 | 0.0042 | 0.0020 | -0.0004 | 0.0064 | -0.0014 | 0.0153 | -0.0150 | 1.2993 | 0.6638 |
| `prompt_last|L34|natural_wrong` | natural_wrong | 11 | `punct_newline` | 0.0025 | 0.0016 | 0.0042 | 0.0020 | -0.0004 | 0.0064 | -0.0014 | 0.0153 | -0.0150 | 1.2993 | 0.6638 |
| `query_category|L32|artificial_random` | artificial_random | 11 | `other` | 0.0001 | -0.0006 | 0.0000 | -0.0006 | 0.0000 | -0.0077 | 0.0000 | -0.0044 | 0.0087 | 1.4565 | 0.6697 |
| `prompt_last|L34|artificial_wrong` | artificial_wrong | 11 | `prompt_last` | 0.0000 | 0.0001 | -0.0001 | 0.0000 | 0.0010 | -0.0001 | -0.0087 | -0.0073 | 0.0077 | 1.2670 | 0.6963 |
| `prompt_last|L34|artificial_repair` | artificial_repair | 11 | `prompt_last` | 0.0001 | 0.0001 | 0.0009 | 0.0000 | 0.0005 | 0.0003 | -0.0074 | 0.0010 | 0.0056 | 1.3178 | 0.6864 |
| `prompt_last|L32|artificial_repair` | artificial_repair | 11 | `punct_newline` | -0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0012 | 0.0000 | -0.0027 | -0.0060 | 0.0014 | 1.2639 | 0.6895 |
| `prompt_last|L32|artificial_wrong` | artificial_wrong | 11 | `punct_newline` | 0.0000 | 0.0001 | 0.0003 | 0.0000 | 0.0015 | 0.0000 | -0.0038 | -0.0056 | 0.0017 | 1.2912 | 0.6873 |
| `prompt_last|L34|artificial_random` | artificial_random | 11 | `punct_newline` | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.0002 | 0.0001 | -0.0050 | -0.0056 | 0.0045 | 1.2629 | 0.6961 |
| `prompt_last|L32|artificial_random` | artificial_random | 11 | `prompt_last` | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0003 | 0.0000 | -0.0005 | -0.0004 | -0.0002 | 1.2663 | 0.6952 |

### Natural Correct Minus Artificial Repair

| key | n | max_group | l1 | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other |
|---|---:|---|---:||---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_category|L32` | 11 | `other` | 0.1780 | 0.0047 | 0.0000 | 0.0149 | 0.0013 | 0.0000 | 0.0448 | 0.0000 | 0.0466 | -0.0657 |
| `prompt_last|L32` | 11 | `punct_newline` | 0.0614 | 0.0021 | 0.0011 | 0.0024 | 0.0021 | -0.0017 | 0.0073 | 0.0048 | 0.0217 | -0.0182 |
| `prompt_last|L34` | 11 | `other` | 0.0612 | 0.0020 | 0.0010 | 0.0017 | 0.0021 | -0.0009 | 0.0070 | 0.0094 | 0.0146 | -0.0224 |

## glm4

cases=128, rows=22, target_cases_seen=22, probe_layer=39, alpha=2.0, time_min=1.10

### Largest Attention Deltas

| key | trajectory | n | max_group | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other | entropy | top_mass |
|---|---|---:|---||---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L37|natural_correct` | natural_correct | 22 | `other` | 0.0018 | 0.0024 | 0.0109 | 0.0049 | 0.0063 | 0.0096 | 0.0373 | 0.0065 | -0.0731 | 1.8485 | 0.4810 |
| `prompt_last|L38|natural_correct` | natural_correct | 22 | `other` | 0.0018 | 0.0024 | 0.0109 | 0.0049 | 0.0063 | 0.0096 | 0.0373 | 0.0065 | -0.0731 | 1.8485 | 0.4810 |
| `prompt_last|L39|natural_correct` | natural_correct | 22 | `other` | 0.0018 | 0.0024 | 0.0109 | 0.0049 | 0.0063 | 0.0096 | 0.0373 | 0.0065 | -0.0731 | 1.8485 | 0.4810 |
| `prompt_last|L37|natural_wrong` | natural_wrong | 22 | `other` | 0.0015 | 0.0011 | 0.0096 | 0.0049 | 0.0047 | 0.0067 | 0.0327 | 0.0321 | -0.0611 | 1.8155 | 0.4912 |
| `prompt_last|L38|natural_wrong` | natural_wrong | 22 | `other` | 0.0015 | 0.0011 | 0.0096 | 0.0049 | 0.0047 | 0.0067 | 0.0327 | 0.0321 | -0.0611 | 1.8155 | 0.4912 |
| `prompt_last|L39|natural_wrong` | natural_wrong | 22 | `other` | 0.0015 | 0.0011 | 0.0096 | 0.0049 | 0.0047 | 0.0067 | 0.0327 | 0.0321 | -0.0611 | 1.8155 | 0.4912 |
| `prompt_last|L38|artificial_repair` | artificial_repair | 22 | `punct_newline` | 0.0001 | -0.0000 | 0.0017 | -0.0000 | 0.0071 | 0.0018 | -0.0175 | -0.0389 | 0.0069 | 2.1778 | 0.4322 |
| `prompt_last|L38|artificial_random` | artificial_random | 22 | `prompt_last` | 0.0000 | 0.0001 | 0.0017 | 0.0002 | 0.0050 | 0.0020 | -0.0371 | -0.0257 | 0.0281 | 2.1262 | 0.4549 |
| `prompt_last|L38|artificial_wrong` | artificial_wrong | 22 | `other` | -0.0002 | 0.0002 | 0.0010 | -0.0000 | 0.0004 | 0.0012 | 0.0273 | 0.0255 | -0.0299 | 2.0609 | 0.4773 |
| `prompt_last|L37|artificial_wrong` | artificial_wrong | 22 | `punct_newline` | -0.0003 | -0.0001 | -0.0004 | -0.0001 | -0.0016 | -0.0000 | 0.0042 | 0.0211 | -0.0018 | 1.9626 | 0.5060 |
| `prompt_last|L37|artificial_repair` | artificial_repair | 22 | `punct_newline` | -0.0002 | -0.0001 | -0.0000 | -0.0000 | -0.0023 | -0.0002 | 0.0077 | 0.0205 | -0.0048 | 1.9750 | 0.5026 |
| `prompt_last|L37|artificial_random` | artificial_random | 22 | `prompt_last` | -0.0002 | 0.0000 | 0.0003 | 0.0000 | -0.0013 | 0.0011 | 0.0112 | 0.0061 | -0.0111 | 2.0065 | 0.4924 |
| `prompt_last|L39|artificial_random` | artificial_random | 22 | `rule_relation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0857 | 0.4731 |
| `prompt_last|L39|artificial_repair` | artificial_repair | 22 | `rule_relation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0857 | 0.4731 |
| `prompt_last|L39|artificial_wrong` | artificial_wrong | 22 | `rule_relation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0857 | 0.4731 |

### Natural Correct Minus Artificial Repair

| key | n | max_group | l1 | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other |
|---|---:|---|---:||---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L38` | 22 | `other` | 0.2070 | 0.0017 | 0.0025 | 0.0092 | 0.0049 | -0.0008 | 0.0078 | 0.0547 | 0.0455 | -0.0800 |
| `prompt_last|L39` | 22 | `other` | 0.1527 | 0.0018 | 0.0024 | 0.0109 | 0.0049 | 0.0063 | 0.0096 | 0.0373 | 0.0065 | -0.0731 |
| `prompt_last|L37` | 22 | `other` | 0.1506 | 0.0020 | 0.0026 | 0.0109 | 0.0049 | 0.0086 | 0.0097 | 0.0296 | -0.0140 | -0.0683 |

## deepseek7b

cases=128, rows=49, target_cases_seen=49, probe_layer=27, alpha=2.0, time_min=1.43

### Largest Attention Deltas

| key | trajectory | n | max_group | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other | entropy | top_mass |
|---|---|---:|---||---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L26|natural_correct` | natural_correct | 49 | `other` | 0.0037 | 0.0091 | 0.0174 | 0.0150 | 0.0019 | 0.0125 | 0.0269 | 0.0481 | -0.0865 | 2.5096 | 0.3788 |
| `rule_value|L26|natural_correct` | natural_correct | 25 | `other` | 0.0277 | -0.0018 | 0.0000 | 0.0465 | 0.0000 | 0.0000 | 0.0000 | 0.0113 | -0.0724 | 1.8928 | 0.4800 |
| `query_relation|L19|natural_wrong` | natural_wrong | 49 | `other` | 0.0071 | -0.0026 | 0.0191 | 0.0037 | 0.0336 | 0.0086 | 0.0000 | 0.0353 | -0.0694 | 1.9633 | 0.5535 |
| `prompt_last|L26|natural_wrong` | natural_wrong | 49 | `other` | 0.0030 | -0.0020 | 0.0147 | 0.0085 | 0.0109 | 0.0083 | 0.0213 | 0.0391 | -0.0648 | 2.3244 | 0.4442 |
| `prompt_last|L26|artificial_random` | artificial_random | 49 | `punct_newline` | -0.0001 | 0.0002 | 0.0009 | 0.0001 | -0.0013 | -0.0002 | -0.0562 | -0.0632 | 0.0567 | 2.7349 | 0.4368 |
| `query_relation|L19|natural_correct` | natural_correct | 49 | `other` | 0.0048 | 0.0012 | 0.0248 | 0.0030 | 0.0163 | 0.0092 | 0.0000 | 0.0315 | -0.0593 | 1.9826 | 0.5455 |
| `prompt_last|L26|artificial_repair` | artificial_repair | 49 | `prompt_last` | 0.0004 | 0.0006 | -0.0003 | 0.0012 | -0.0020 | 0.0007 | -0.0503 | -0.0450 | 0.0497 | 2.8665 | 0.4136 |
| `rule_value|L26|artificial_repair` | artificial_repair | 25 | `other` | 0.0010 | 0.0385 | 0.0000 | -0.0005 | 0.0000 | 0.0000 | 0.0000 | -0.0109 | -0.0390 | 2.3988 | 0.4621 |
| `rule_value|L26|natural_wrong` | natural_wrong | 25 | `other` | 0.0264 | -0.0092 | 0.0000 | 0.0173 | 0.0000 | 0.0000 | 0.0000 | 0.0129 | -0.0345 | 1.9242 | 0.4769 |
| `rule_value|L26|artificial_wrong` | artificial_wrong | 25 | `other` | 0.0009 | 0.0334 | 0.0000 | -0.0005 | 0.0000 | 0.0000 | 0.0000 | -0.0099 | -0.0338 | 2.4151 | 0.4585 |
| `prompt_last|L26|artificial_wrong` | artificial_wrong | 49 | `punct_newline` | -0.0000 | 0.0008 | -0.0018 | 0.0000 | -0.0012 | -0.0005 | -0.0214 | -0.0287 | 0.0241 | 2.7713 | 0.4342 |
| `query_relation|L19|artificial_wrong` | artificial_wrong | 49 | `other` | -0.0006 | -0.0016 | 0.0000 | -0.0006 | -0.0131 | -0.0010 | 0.0000 | -0.0016 | 0.0169 | 2.4953 | 0.5109 |
| `query_relation|L19|artificial_random` | artificial_random | 49 | `punct_newline` | -0.0006 | -0.0012 | 0.0044 | -0.0005 | -0.0002 | 0.0007 | 0.0000 | 0.0115 | -0.0026 | 2.7791 | 0.4431 |
| `rule_value|L26|artificial_random` | artificial_random | 25 | `punct_newline` | -0.0009 | 0.0004 | 0.0000 | -0.0002 | 0.0000 | 0.0000 | 0.0000 | -0.0061 | 0.0006 | 2.4456 | 0.4673 |
| `query_relation|L19|artificial_repair` | artificial_repair | 49 | `other` | 0.0007 | -0.0011 | -0.0007 | -0.0001 | -0.0016 | 0.0002 | 0.0000 | 0.0016 | 0.0025 | 2.5753 | 0.4993 |

### Natural Correct Minus Artificial Repair

| key | n | max_group | l1 | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other |
|---|---:|---|---:||---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L26` | 49 | `other` | 0.3654 | 0.0034 | 0.0085 | 0.0177 | 0.0138 | 0.0039 | 0.0118 | 0.0772 | 0.0931 | -0.1362 |
| `rule_value|L26` | 25 | `category_first` | 0.1697 | 0.0267 | -0.0404 | 0.0000 | 0.0470 | 0.0000 | 0.0000 | 0.0000 | 0.0222 | -0.0334 |
| `query_relation|L19` | 49 | `other` | 0.1536 | 0.0041 | 0.0022 | 0.0255 | 0.0031 | 0.0178 | 0.0090 | 0.0000 | 0.0300 | -0.0618 |

### DS7B watched source deltas

| key | trajectory | n | max_group | rule_relation | rule_value | object | category_first | query_relation | query_category | prompt_last | punct_newline | other | entropy | top_mass |
|---|---|---:|---||---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|natural_correct` | natural_correct | 25 | `other` | 0.0277 | -0.0018 | 0.0000 | 0.0465 | 0.0000 | 0.0000 | 0.0000 | 0.0113 | -0.0724 | 1.8928 | 0.4800 |
| `rule_value|L26|natural_wrong` | natural_wrong | 25 | `other` | 0.0264 | -0.0092 | 0.0000 | 0.0173 | 0.0000 | 0.0000 | 0.0000 | 0.0129 | -0.0345 | 1.9242 | 0.4769 |
| `rule_value|L26|artificial_repair` | artificial_repair | 25 | `other` | 0.0010 | 0.0385 | 0.0000 | -0.0005 | 0.0000 | 0.0000 | 0.0000 | -0.0109 | -0.0390 | 2.3988 | 0.4621 |
| `rule_value|L26|artificial_random` | artificial_random | 25 | `punct_newline` | -0.0009 | 0.0004 | 0.0000 | -0.0002 | 0.0000 | 0.0000 | 0.0000 | -0.0061 | 0.0006 | 2.4456 | 0.4673 |
| `rule_value|L26|artificial_wrong` | artificial_wrong | 25 | `other` | 0.0009 | 0.0334 | 0.0000 | -0.0005 | 0.0000 | 0.0000 | 0.0000 | -0.0099 | -0.0338 | 2.4151 | 0.4585 |
| `prompt_last|L26|natural_correct` | natural_correct | 49 | `other` | 0.0037 | 0.0091 | 0.0174 | 0.0150 | 0.0019 | 0.0125 | 0.0269 | 0.0481 | -0.0865 | 2.5096 | 0.3788 |
| `prompt_last|L26|natural_wrong` | natural_wrong | 49 | `other` | 0.0030 | -0.0020 | 0.0147 | 0.0085 | 0.0109 | 0.0083 | 0.0213 | 0.0391 | -0.0648 | 2.3244 | 0.4442 |
| `prompt_last|L26|artificial_repair` | artificial_repair | 49 | `prompt_last` | 0.0004 | 0.0006 | -0.0003 | 0.0012 | -0.0020 | 0.0007 | -0.0503 | -0.0450 | 0.0497 | 2.8665 | 0.4136 |
| `prompt_last|L26|artificial_random` | artificial_random | 49 | `punct_newline` | -0.0001 | 0.0002 | 0.0009 | 0.0001 | -0.0013 | -0.0002 | -0.0562 | -0.0632 | 0.0567 | 2.7349 | 0.4368 |
| `prompt_last|L26|artificial_wrong` | artificial_wrong | 49 | `punct_newline` | -0.0000 | 0.0008 | -0.0018 | 0.0000 | -0.0012 | -0.0005 | -0.0214 | -0.0287 | 0.0241 | 2.7713 | 0.4342 |
| `query_relation|L19|natural_correct` | natural_correct | 49 | `other` | 0.0048 | 0.0012 | 0.0248 | 0.0030 | 0.0163 | 0.0092 | 0.0000 | 0.0315 | -0.0593 | 1.9826 | 0.5455 |
| `query_relation|L19|natural_wrong` | natural_wrong | 49 | `other` | 0.0071 | -0.0026 | 0.0191 | 0.0037 | 0.0336 | 0.0086 | 0.0000 | 0.0353 | -0.0694 | 1.9633 | 0.5535 |
| `query_relation|L19|artificial_repair` | artificial_repair | 49 | `other` | 0.0007 | -0.0011 | -0.0007 | -0.0001 | -0.0016 | 0.0002 | 0.0000 | 0.0016 | 0.0025 | 2.5753 | 0.4993 |
| `query_relation|L19|artificial_random` | artificial_random | 49 | `punct_newline` | -0.0006 | -0.0012 | 0.0044 | -0.0005 | -0.0002 | 0.0007 | 0.0000 | 0.0115 | -0.0026 | 2.7791 | 0.4431 |
| `query_relation|L19|artificial_wrong` | artificial_wrong | 49 | `other` | -0.0006 | -0.0016 | 0.0000 | -0.0006 | -0.0131 | -0.0010 | 0.0000 | -0.0016 | 0.0169 | 2.4953 | 0.5109 |

