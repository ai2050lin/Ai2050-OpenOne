# Phase 808 Readout Closer Source Localization (smoke)

- Status: `complete`
- Boundary: component-level source localization for the format-only closer candidate.
- Success requires old-blocker reduction and low new-blocker emergence; target-logit gain alone is not enough.

## By Component

| model | component | rows | cases | full net | full resolved | full emerged | single net | single resolved | single emerged | single emergence rate | single bias | single fmt supp | loo net loss | loo resolved loss | loo emerged delta | loo bias loss | single closure | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `mlp:L35` | 1 | 1 | -6.000 | 7.000 | 1.000 | -6.000 | 7.000 | 1.000 | 0.037 | -0.062 | 0.924 | 5.000 | 6.000 | -1.000 | 0.125 | 0.000 | `{"source_closer_candidate_no_closure": 1}` |
| qwen3 | `attn:L35` | 1 | 1 | -6.000 | 7.000 | 1.000 | -2.000 | 2.000 | 0.000 | 0.000 | -0.062 | 0.236 | 1.000 | 1.000 | 0.000 | -0.062 | 0.000 | `{"weak_local_reducer": 1}` |
| qwen3 | `mlp:L34` | 1 | 1 | -6.000 | 7.000 | 1.000 | 2.000 | 0.000 | 2.000 | 0.074 | 0.062 | -0.247 | -2.000 | -2.000 | 0.000 | 0.000 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| qwen3 | `mlp:L33` | 1 | 1 | -6.000 | 7.000 | 1.000 | -1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.094 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"weak_local_reducer": 1}` |
| glm4 | `mlp:L38` | 1 | 1 | 23.000 | 0.000 | 23.000 | 5.000 | 0.000 | 5.000 | 0.053 | 0.000 | -0.144 | -8.000 | 0.000 | -8.000 | -0.094 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| glm4 | `mlp:L39` | 1 | 1 | 23.000 | 0.000 | 23.000 | 16.000 | 0.000 | 16.000 | 0.170 | 0.125 | -0.513 | -19.000 | 0.000 | -19.000 | -0.156 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| glm4 | `mlp:L27` | 1 | 1 | 23.000 | 0.000 | 23.000 | 1.000 | 0.000 | 1.000 | 0.011 | 0.000 | 0.019 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| glm4 | `mlp:L34` | 1 | 1 | 23.000 | 0.000 | 23.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"neutral_or_weak": 1}` |
| deepseek7b | `mlp:L26` | 1 | 1 | 10.000 | 67.000 | 77.000 | -2.000 | 6.000 | 4.000 | 0.012 | -0.031 | 0.195 | 4.000 | 9.000 | -5.000 | 0.125 | 0.000 | `{"source_closer_candidate_no_closure": 1}` |
| deepseek7b | `attn:L19` | 1 | 1 | 10.000 | 67.000 | 77.000 | 5.000 | 0.000 | 5.000 | 0.015 | 0.031 | 0.008 | 2.000 | 3.000 | -1.000 | 0.031 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| deepseek7b | `mlp:L24` | 1 | 1 | 10.000 | 67.000 | 77.000 | 1.000 | 1.000 | 2.000 | 0.006 | 0.000 | 0.010 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| deepseek7b | `mlp:L27` | 1 | 1 | 10.000 | 67.000 | 77.000 | 5.000 | 64.000 | 69.000 | 0.205 | -0.969 | 1.316 | -11.000 | 62.000 | -73.000 | 1.031 | 0.000 | `{"new_blocker_source_or_field_deformer": 1}` |

## Top Source Candidates

| model | component | single net | single emerged | single bias | loo net loss | label counts |
|---|---|---:|---:|---:|---:|---|
| qwen3 | `mlp:L35` | -6.000 | 1.000 | -0.062 | 5.000 | `{"source_closer_candidate_no_closure": 1}` |
| qwen3 | `attn:L35` | -2.000 | 0.000 | -0.062 | 1.000 | `{"weak_local_reducer": 1}` |
| qwen3 | `mlp:L33` | -1.000 | 0.000 | 0.000 | 0.000 | `{"weak_local_reducer": 1}` |
| deepseek7b | `mlp:L26` | -2.000 | 4.000 | -0.031 | 4.000 | `{"source_closer_candidate_no_closure": 1}` |
| deepseek7b | `attn:L19` | 5.000 | 5.000 | 0.031 | 2.000 | `{"new_blocker_source_or_field_deformer": 1}` |
| deepseek7b | `mlp:L24` | 1.000 | 2.000 | 0.000 | 1.000 | `{"new_blocker_source_or_field_deformer": 1}` |
