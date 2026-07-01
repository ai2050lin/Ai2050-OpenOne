# Phase 808 Readout Closer Source Localization (confirm)

- Status: `complete`
- Boundary: component-level source localization for the format-only closer candidate.
- Success requires old-blocker reduction and low new-blocker emergence; target-logit gain alone is not enough.

## By Component

| model | component | rows | cases | full net | full resolved | full emerged | single net | single resolved | single emerged | single emergence rate | single bias | single fmt supp | loo net loss | loo resolved loss | loo emerged delta | loo bias loss | single closure | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `mlp:L34` | 10 | 5 | -63.900 | 68.000 | 4.100 | -25.500 | 26.300 | 0.800 | 0.005 | -0.109 | 0.154 | 24.700 | 24.600 | 0.100 | 0.128 | 0.000 | `{"neutral_or_weak": 4, "new_blocker_source_or_field_deformer": 2, "source_closer_candidate_no_closure": 4}` |
| qwen3 | `mlp:L35` | 10 | 5 | -63.900 | 68.000 | 4.100 | -25.400 | 28.300 | 2.900 | 0.025 | -0.234 | 0.375 | 20.800 | 23.200 | -2.400 | 0.203 | 0.000 | `{"neutral_or_weak": 2, "new_blocker_source_or_field_deformer": 2, "source_closer_candidate_no_closure": 5, "weak_local_reducer": 1}` |
| qwen3 | `attn:L34` | 5 | 5 | -70.800 | 71.200 | 0.400 | -17.200 | 17.400 | 0.200 | 0.004 | 0.013 | 0.297 | 13.400 | 13.200 | 0.200 | 0.113 | 0.000 | `{"neutral_or_weak": 2, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| qwen3 | `attn:L35` | 5 | 5 | -57.000 | 64.800 | 7.800 | -17.400 | 19.000 | 1.600 | 0.010 | -0.163 | 0.290 | 9.600 | 11.200 | -1.600 | 0.150 | 0.000 | `{"source_closer_candidate_no_closure": 2, "weak_local_reducer": 3}` |
| qwen3 | `mlp:L33` | 5 | 5 | -57.000 | 64.800 | 7.800 | -4.800 | 7.000 | 2.200 | 0.002 | -0.044 | 0.077 | 5.800 | 6.000 | -0.200 | 0.006 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 2}` |
| qwen3 | `attn:L31` | 5 | 5 | -70.800 | 71.200 | 0.400 | -2.600 | 4.000 | 1.400 | 0.049 | -0.013 | 0.023 | 4.800 | 5.200 | -0.400 | 0.006 | 0.000 | `{"new_blocker_source_or_field_deformer": 3, "source_closer_candidate_no_closure": 2}` |
| glm4 | `mlp:L34` | 3 | 3 | 6.667 | 13.000 | 19.667 | 1.667 | 1.000 | 2.667 | 0.014 | 0.026 | -0.009 | -2.000 | 0.333 | -2.333 | -0.026 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 2}` |
| glm4 | `mlp:L27` | 3 | 3 | 6.667 | 13.000 | 19.667 | 0.667 | 0.333 | 1.000 | 0.009 | -0.010 | 0.015 | -2.667 | -1.000 | -1.667 | -0.005 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `attn:L29` | 3 | 3 | 15.333 | 0.333 | 15.667 | 1.667 | 0.333 | 2.000 | 0.028 | 0.005 | 0.008 | -3.333 | 0.333 | -3.667 | 0.010 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `attn:L35` | 3 | 3 | 15.333 | 0.333 | 15.667 | 2.000 | 0.333 | 2.333 | 0.030 | 0.021 | -0.011 | -3.333 | 0.333 | -3.667 | -0.005 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `attn:L33` | 3 | 3 | 15.333 | 0.333 | 15.667 | 1.667 | 0.000 | 1.667 | 0.015 | 0.036 | -0.015 | -4.000 | 0.000 | -4.000 | -0.026 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 2}` |
| glm4 | `mlp:L39` | 3 | 3 | 6.667 | 13.000 | 19.667 | 4.333 | 12.000 | 16.333 | 0.109 | 0.151 | -0.038 | -5.000 | 12.000 | -17.000 | -0.161 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `mlp:L38` | 6 | 3 | 11.000 | 6.667 | 17.667 | 4.667 | 0.500 | 5.167 | 0.041 | 0.023 | -0.046 | -5.500 | 0.333 | -5.833 | -0.036 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 4, "weak_local_reducer": 1}` |
| deepseek7b | `attn:L27` | 2 | 2 | -46.500 | 48.000 | 1.500 | -10.500 | 10.500 | 0.000 | 0.000 | -0.375 | 0.196 | 9.500 | 10.000 | -0.500 | 0.312 | 0.000 | `{"source_closer_candidate_no_closure": 2}` |
| deepseek7b | `attn:L19` | 2 | 2 | 1067.500 | 159.500 | 1227.000 | 1.500 | 6.500 | 8.000 | 0.006 | -0.094 | 0.011 | 6.500 | 2.500 | 4.000 | 0.008 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1}` |
| deepseek7b | `attn:L26` | 2 | 2 | -46.500 | 48.000 | 1.500 | -7.000 | 7.000 | 0.000 | 0.000 | -0.359 | 0.168 | 2.500 | 3.000 | -0.500 | 0.219 | 0.000 | `{"source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| deepseek7b | `attn:L25` | 2 | 2 | -46.500 | 48.000 | 1.500 | 0.000 | 0.500 | 0.500 | 0.001 | 0.016 | 0.021 | 0.500 | 1.000 | -0.500 | -0.008 | 0.000 | `{"neutral_or_weak": 2}` |
| deepseek7b | `mlp:L24` | 2 | 2 | 1067.500 | 159.500 | 1227.000 | 13.000 | 2.500 | 15.500 | 0.006 | -0.059 | 0.027 | -12.000 | 1.000 | -13.000 | 0.020 | 0.000 | `{"new_blocker_source_or_field_deformer": 1, "weak_local_reducer": 1}` |
| deepseek7b | `mlp:L26` | 2 | 2 | 1067.500 | 159.500 | 1227.000 | 143.000 | 13.500 | 156.500 | 0.067 | -0.395 | 0.461 | -353.500 | -3.000 | -350.500 | 0.172 | 0.000 | `{"new_blocker_source_or_field_deformer": 1, "weak_local_reducer": 1}` |
| deepseek7b | `mlp:L27` | 4 | 2 | 510.500 | 103.750 | 614.250 | 330.750 | 94.250 | 425.000 | 0.216 | -1.859 | 1.520 | -441.250 | 88.500 | -529.750 | 1.686 | 0.000 | `{"distributed_closer_contributor": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 2}` |

## Top Source Candidates

| model | component | single net | single emerged | single bias | loo net loss | label counts |
|---|---|---:|---:|---:|---:|---|
| qwen3 | `mlp:L34` | -25.500 | 0.800 | -0.109 | 24.700 | `{"neutral_or_weak": 4, "new_blocker_source_or_field_deformer": 2, "source_closer_candidate_no_closure": 4}` |
| qwen3 | `mlp:L35` | -25.400 | 2.900 | -0.234 | 20.800 | `{"neutral_or_weak": 2, "new_blocker_source_or_field_deformer": 2, "source_closer_candidate_no_closure": 5, "weak_local_reducer": 1}` |
| qwen3 | `attn:L34` | -17.200 | 0.200 | 0.013 | 13.400 | `{"neutral_or_weak": 2, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| qwen3 | `attn:L35` | -17.400 | 1.600 | -0.163 | 9.600 | `{"source_closer_candidate_no_closure": 2, "weak_local_reducer": 3}` |
| qwen3 | `mlp:L33` | -4.800 | 2.200 | -0.044 | 5.800 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 2}` |
| qwen3 | `attn:L31` | -2.600 | 1.400 | -0.013 | 4.800 | `{"new_blocker_source_or_field_deformer": 3, "source_closer_candidate_no_closure": 2}` |
| deepseek7b | `attn:L27` | -10.500 | 0.000 | -0.375 | 9.500 | `{"source_closer_candidate_no_closure": 2}` |
| deepseek7b | `attn:L19` | 1.500 | 8.000 | -0.094 | 6.500 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1}` |
| deepseek7b | `attn:L26` | -7.000 | 0.000 | -0.359 | 2.500 | `{"source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| deepseek7b | `attn:L25` | 0.000 | 0.500 | 0.016 | 0.500 | `{"neutral_or_weak": 2}` |
