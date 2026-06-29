# Phase 743 Competitor and Format Suppression Audit (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: top-k vocabulary competitor classes and final-norm suppression of current top competitor.

| model | condition | scale | donor top1 | mean donor rank | margin donor vs top | top classes | suppressed classes |
|---|---|---:|---:|---:|---:|---|---|
| qwen3 | joint_add_topK | 0.00 | 0.000 | 13.08 | -7.120 | `{"recipient_answer": 12}` | `{"recipient_answer": 12}` |
| qwen3 | suppress_current_top | 1.00 | 0.167 | 2.42 | -1.510 | `{"donor_answer": 1, "format_or_schema": 8, "recipient_answer": 3}` | `{"recipient_answer": 12}` |
| qwen3 | suppress_current_top | 1.25 | 0.333 | 2.00 | -0.896 | `{"donor_answer": 4, "format_or_schema": 8}` | `{"recipient_answer": 12}` |
| glm4 | joint_add_topK | 0.00 | 0.500 | 3.50 | -1.089 | `{"donor_answer": 6, "echo_object_or_relation": 4, "other_vocab": 2}` | `{"donor_answer": 6, "echo_object_or_relation": 4, "other_vocab": 2}` |
| glm4 | suppress_current_top | 1.00 | 0.000 | 2.00 | -0.302 | `{"echo_object_or_relation": 2, "format_or_schema": 2, "other_vocab": 2}` | `{"echo_object_or_relation": 4, "other_vocab": 2}` |
| glm4 | suppress_current_top | 1.25 | 0.667 | 1.33 | -0.188 | `{"donor_answer": 4, "echo_object_or_relation": 2}` | `{"echo_object_or_relation": 4, "other_vocab": 2}` |
| deepseek7b | joint_add_topK | 0.00 | 0.083 | 12.83 | -2.760 | `{"donor_answer": 1, "echo_object_or_relation": 4, "format_or_schema": 5, "other_vocab": 1, "recipient_answer": 1}` | `{"donor_answer": 1, "echo_object_or_relation": 4, "format_or_schema": 5, "other_vocab": 1, "recipient_answer": 1}` |
| deepseek7b | suppress_current_top | 1.00 | 0.091 | 3.82 | -0.807 | `{"donor_answer": 1, "echo_object_or_relation": 3, "format_or_schema": 4, "other_semantic_value": 1, "punctuation_or_stop": 2}` | `{"echo_object_or_relation": 4, "format_or_schema": 5, "other_vocab": 1, "recipient_answer": 1}` |
| deepseek7b | suppress_current_top | 1.25 | 0.273 | 2.55 | -0.511 | `{"donor_answer": 3, "echo_object_or_relation": 3, "format_or_schema": 2, "other_semantic_value": 1, "punctuation_or_stop": 2}` | `{"echo_object_or_relation": 4, "format_or_schema": 5, "other_vocab": 1, "recipient_answer": 1}` |

## Suppressed Class Summary

| model | class | n | mean alpha needed | mean margin donor vs top | top tokens |
|---|---|---:|---:|---:|---|
| qwen3 | recipient_answer | 12 | 5.189 | -7.120 | `{" fruit": 2, " none": 2, " object": 2, " orange": 2, " sweet": 2, " yellow": 2}` |
| glm4 | donor_answer | 6 | 0.000 | 0.000 | `{" fruit": 2, " orange": 2, " yellow": 2}` |
| glm4 | echo_object_or_relation | 4 | 3.289 | -3.047 | `{" stone": 3, " taste": 1}` |
| glm4 | other_vocab | 2 | 0.460 | -0.438 | `{" B": 2}` |
| deepseek7b | format_or_schema | 5 | 1.450 | -1.887 | `{" The": 5}` |
| deepseek7b | echo_object_or_relation | 4 | 2.332 | -3.234 | `{" category": 2, " stone": 1, " taste": 1}` |
| deepseek7b | donor_answer | 1 | 0.000 | 0.000 | `{" orange": 1}` |
| deepseek7b | recipient_answer | 1 | 3.668 | -5.188 | `{" object": 1}` |
| deepseek7b | other_vocab | 1 | 3.968 | -5.562 | `{" __": 1}` |

## Strict Interpretation

- If suppressing only the current top competitor does not make donor top1, the failure is multi-competitor or global readout geometry, not a single blocking token.
- If it does make donor top1, Phase 742 near-closure was mainly blocked by a local competitor class.
- This is still a final readout intervention; it does not prove the natural circuit that performs suppression.

Atlas graph: nodes=18 edges=24
