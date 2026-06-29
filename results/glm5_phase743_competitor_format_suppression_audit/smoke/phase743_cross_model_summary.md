# Phase 743 Competitor and Format Suppression Audit (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: top-k vocabulary competitor classes and final-norm suppression of current top competitor.

| model | condition | scale | donor top1 | mean donor rank | margin donor vs top | top classes | suppressed classes |
|---|---|---:|---:|---:|---:|---|---|
| qwen3 | joint_add_topK | 0.00 | 0.000 | 8.00 | -7.250 | `{"recipient_answer": 1}` | `{"recipient_answer": 1}` |
| qwen3 | suppress_current_top | 1.00 | 0.000 | 3.00 | -1.750 | `{"format_or_schema": 1}` | `{"recipient_answer": 1}` |
| qwen3 | suppress_current_top | 1.25 | 0.000 | 2.00 | -0.875 | `{"format_or_schema": 1}` | `{"recipient_answer": 1}` |
| glm4 | joint_add_topK | 0.00 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | `{"donor_answer": 1}` |
| deepseek7b | joint_add_topK | 0.00 | 0.000 | 10.00 | -3.250 | `{"echo_object_or_relation": 1}` | `{"echo_object_or_relation": 1}` |
| deepseek7b | suppress_current_top | 1.00 | 0.000 | 4.00 | -0.625 | `{"punctuation_or_stop": 1}` | `{"echo_object_or_relation": 1}` |
| deepseek7b | suppress_current_top | 1.25 | 0.000 | 2.00 | -0.125 | `{"punctuation_or_stop": 1}` | `{"echo_object_or_relation": 1}` |

## Suppressed Class Summary

| model | class | n | mean alpha needed | mean margin donor vs top | top tokens |
|---|---|---:|---:|---:|---|
| qwen3 | recipient_answer | 1 | 4.880 | -7.250 | `{" fruit": 1}` |
| glm4 | donor_answer | 1 | 0.000 | 0.000 | `{" fruit": 1}` |
| deepseek7b | echo_object_or_relation | 1 | 2.265 | -3.250 | `{" category": 1}` |

## Strict Interpretation

- If suppressing only the current top competitor does not make donor top1, the failure is multi-competitor or global readout geometry, not a single blocking token.
- If it does make donor top1, Phase 742 near-closure was mainly blocked by a local competitor class.
- This is still a final readout intervention; it does not prove the natural circuit that performs suppression.

Atlas graph: nodes=12 edges=12
