# Phase 808 Readout Closer Source Localization (main)

- Status: `complete`
- Boundary: component-level source localization for the format-only closer candidate.
- Success requires old-blocker reduction and low new-blocker emergence; target-logit gain alone is not enough.

## By Component

| model | component | rows | cases | full net | full resolved | full emerged | single net | single resolved | single emerged | single emergence rate | single bias | single fmt supp | loo net loss | loo resolved loss | loo emerged delta | loo bias loss | single closure | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `mlp:L34` | 6 | 3 | -103.167 | 110.000 | 6.833 | -44.500 | 45.500 | 1.000 | 0.004 | -0.151 | 0.124 | 38.000 | 38.667 | -0.667 | 0.182 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 4}` |
| qwen3 | `mlp:L35` | 6 | 3 | -103.167 | 110.000 | 6.833 | -44.167 | 48.167 | 4.000 | 0.007 | -0.370 | 0.593 | 31.667 | 35.500 | -3.833 | 0.380 | 0.000 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 4, "weak_local_reducer": 1}` |
| qwen3 | `attn:L34` | 3 | 3 | -117.000 | 117.000 | 0.000 | -26.667 | 27.000 | 0.333 | 0.001 | -0.021 | 0.150 | 19.667 | 19.333 | 0.333 | 0.083 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1}` |
| qwen3 | `attn:L35` | 3 | 3 | -89.333 | 103.000 | 13.667 | -31.000 | 34.333 | 3.333 | 0.003 | -0.281 | 0.337 | 14.000 | 18.667 | -4.667 | 0.240 | 0.000 | `{"source_closer_candidate_no_closure": 2, "weak_local_reducer": 1}` |
| qwen3 | `attn:L31` | 3 | 3 | -117.000 | 117.000 | 0.000 | -9.333 | 10.000 | 0.667 | 0.013 | -0.094 | 0.018 | 3.333 | 3.333 | 0.000 | 0.052 | 0.000 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| qwen3 | `mlp:L33` | 3 | 3 | -89.333 | 103.000 | 13.667 | -7.333 | 8.333 | 1.000 | 0.001 | -0.052 | 0.085 | -5.667 | -3.000 | -2.667 | 0.010 | 0.000 | `{"weak_local_reducer": 3}` |
| glm4 | `mlp:L34` | 3 | 3 | 6.667 | 13.333 | 20.000 | 2.000 | 1.000 | 3.000 | 0.015 | 0.042 | -0.009 | -2.000 | 0.000 | -2.000 | -0.021 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 2}` |
| glm4 | `attn:L29` | 3 | 3 | 15.333 | 0.333 | 15.667 | 1.667 | 0.333 | 2.000 | 0.028 | 0.005 | 0.008 | -3.333 | 0.333 | -3.667 | 0.010 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `attn:L35` | 3 | 3 | 15.333 | 0.333 | 15.667 | 2.000 | 0.333 | 2.333 | 0.030 | 0.021 | -0.011 | -3.333 | 0.333 | -3.667 | -0.005 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `mlp:L39` | 3 | 3 | 6.667 | 13.333 | 20.000 | 3.333 | 12.667 | 16.000 | 0.108 | 0.146 | -0.030 | -3.667 | 12.333 | -16.000 | -0.135 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "weak_local_reducer": 1}` |
| glm4 | `attn:L33` | 3 | 3 | 15.333 | 0.333 | 15.667 | 1.667 | 0.000 | 1.667 | 0.015 | 0.036 | -0.015 | -4.000 | 0.000 | -4.000 | -0.026 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 2}` |
| glm4 | `mlp:L27` | 3 | 3 | 6.667 | 13.333 | 20.000 | 1.000 | 0.333 | 1.333 | 0.010 | -0.010 | 0.013 | -4.667 | -2.000 | -2.667 | -0.005 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 2}` |
| glm4 | `mlp:L38` | 6 | 3 | 11.000 | 6.833 | 17.833 | 4.333 | 0.167 | 4.500 | 0.039 | 0.021 | -0.051 | -5.667 | -0.167 | -5.500 | -0.036 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 4, "weak_local_reducer": 1}` |
| deepseek7b | `attn:L27` | 3 | 3 | -28.000 | 29.667 | 1.667 | -5.000 | 6.000 | 1.000 | 0.019 | -0.255 | 0.088 | 10.333 | 10.333 | 0.000 | 0.214 | 0.000 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 2}` |
| deepseek7b | `attn:L26` | 3 | 3 | -28.000 | 29.667 | 1.667 | -1.333 | 1.667 | 0.333 | 0.001 | -0.234 | 0.113 | 2.333 | 2.667 | -0.333 | 0.193 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1}` |
| deepseek7b | `attn:L25` | 3 | 3 | -28.000 | 29.667 | 1.667 | -0.667 | 1.000 | 0.333 | 0.001 | -0.042 | 0.020 | -0.333 | -0.333 | 0.000 | 0.036 | 0.000 | `{"neutral_or_weak": 2, "weak_local_reducer": 1}` |
| deepseek7b | `attn:L19` | 3 | 3 | 785.000 | 98.667 | 883.667 | -13.667 | 15.333 | 1.667 | 0.001 | -0.026 | 0.002 | -37.667 | -6.333 | -31.333 | 0.013 | 0.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "weak_local_reducer": 1}` |
| deepseek7b | `mlp:L26` | 3 | 3 | 785.000 | 98.667 | 883.667 | 88.333 | 9.333 | 97.667 | 0.039 | -0.242 | 0.441 | -280.333 | -2.333 | -278.000 | 0.138 | 0.000 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| deepseek7b | `mlp:L27` | 6 | 3 | 378.500 | 64.167 | 442.667 | 239.833 | 57.667 | 297.500 | 0.164 | -1.089 | 1.129 | -334.500 | 53.833 | -388.333 | 0.996 | 0.000 | `{"new_blocker_source_or_field_deformer": 2, "source_closer_candidate_no_closure": 3, "weak_local_reducer": 1}` |
| deepseek7b | `mlp:L24` | 3 | 3 | 785.000 | 98.667 | 883.667 | -2.000 | 4.333 | 2.333 | 0.001 | -0.086 | -0.017 | 0.000 | -1.000 | 1.000 | -0.005 | 0.000 | `{"neutral_or_weak": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |

## Top Source Candidates

| model | component | single net | single emerged | single bias | loo net loss | label counts |
|---|---|---:|---:|---:|---:|---|
| qwen3 | `mlp:L34` | -44.500 | 1.000 | -0.151 | 38.000 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 4}` |
| qwen3 | `mlp:L35` | -44.167 | 4.000 | -0.370 | 31.667 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 4, "weak_local_reducer": 1}` |
| qwen3 | `attn:L34` | -26.667 | 0.333 | -0.021 | 19.667 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1}` |
| qwen3 | `attn:L35` | -31.000 | 3.333 | -0.281 | 14.000 | `{"source_closer_candidate_no_closure": 2, "weak_local_reducer": 1}` |
| qwen3 | `attn:L31` | -9.333 | 0.667 | -0.094 | 3.333 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
| qwen3 | `mlp:L33` | -7.333 | 1.000 | -0.052 | -5.667 | `{"weak_local_reducer": 3}` |
| deepseek7b | `attn:L27` | -5.000 | 1.000 | -0.255 | 10.333 | `{"new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 2}` |
| deepseek7b | `attn:L26` | -1.333 | 0.333 | -0.234 | 2.333 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "source_closer_candidate_no_closure": 1}` |
| deepseek7b | `attn:L25` | -0.667 | 0.333 | -0.042 | -0.333 | `{"neutral_or_weak": 2, "weak_local_reducer": 1}` |
| deepseek7b | `attn:L19` | -13.667 | 1.667 | -0.026 | -37.667 | `{"neutral_or_weak": 1, "new_blocker_source_or_field_deformer": 1, "weak_local_reducer": 1}` |
| deepseek7b | `mlp:L24` | -2.000 | 2.333 | -0.086 | 0.000 | `{"neutral_or_weak": 1, "source_closer_candidate_no_closure": 1, "weak_local_reducer": 1}` |
