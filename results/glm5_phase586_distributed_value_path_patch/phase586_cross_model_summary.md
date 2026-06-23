# Phase 586 Distributed Value Path Patch Summary

Confirm setting: 24 value cases per model, 5 positions, 4 layers, add/replace repair plus wrong-relation and random controls.

| model | target cases | best repair position | layer | mode | patch acc | target patch | mean correct-logprob gain | best wrong-rel gain |
|---|---:|---|---:|---|---:|---:|---:|---:|
| qwen3 | 2 | prompt_last | L34 | replace_repair | 95.8% | 1/2 (50.0%) | 1.075 | 0.010 |
| glm4 | 1 | query_relation | L10 | add_repair | 100.0% | 1/1 (100.0%) | 0.193 | 0.116 |
| deepseek7b | 9 | prompt_last | L26 | replace_repair | 62.5% | 0/9 (0.0%) | 5.385 | 2.105 |

## Objective Facts

- Qwen3 and GLM4 have too few failed target cases in this setup, so they are weak controls for value-gate localization.
- DS7B is the main diagnostic model: repair prompt fixes the task, but distributed single-position patch still repairs 0/9 target cases.
- DS7B prompt_last late-layer repair greatly increases correct-value logprob (L26 gain about +5.38), yet does not flip the final candidate winner.
- Wrong-relation control also increases DS7B correct-value logprob less strongly (about +2.10), so the gain is not purely direction-specific enough to close the gate.
- Phase586 therefore does not find the value gate location; it shows the value gate is not solved by single-position residual patch, even at rule/query/prompt positions.
