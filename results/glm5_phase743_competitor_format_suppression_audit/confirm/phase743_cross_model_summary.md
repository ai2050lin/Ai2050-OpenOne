# Phase 743 Competitor and Format Suppression Audit (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: top-k vocabulary competitor classes and final-norm suppression of current top competitor.

| model | condition | scale | donor top1 | mean donor rank | margin donor vs top | top classes | suppressed classes |
|---|---|---:|---:|---:|---:|---|---|
| qwen3 | joint_add_topK | 0.00 | 0.000 | 14.45 | -7.534 | `{"recipient_answer": 20}` | `{"recipient_answer": 20}` |
| qwen3 | suppress_current_top | 1.00 | 0.100 | 2.70 | -1.556 | `{"donor_answer": 1, "format_or_schema": 16, "recipient_answer": 3}` | `{"recipient_answer": 20}` |
| qwen3 | suppress_current_top | 1.25 | 0.300 | 2.15 | -0.838 | `{"donor_answer": 7, "format_or_schema": 13}` | `{"recipient_answer": 20}` |
| glm4 | joint_add_topK | 0.00 | 0.500 | 2.75 | -0.706 | `{"donor_answer": 10, "echo_object_or_relation": 6, "other_vocab": 4}` | `{"donor_answer": 10, "echo_object_or_relation": 6, "other_vocab": 4}` |
| glm4 | suppress_current_top | 1.00 | 0.000 | 1.89 | -0.215 | `{"donor_answer": 1, "echo_object_or_relation": 3, "format_or_schema": 2, "other_vocab": 3}` | `{"echo_object_or_relation": 6, "other_vocab": 3}` |
| glm4 | suppress_current_top | 1.25 | 0.667 | 1.22 | -0.125 | `{"donor_answer": 6, "echo_object_or_relation": 2, "other_vocab": 1}` | `{"echo_object_or_relation": 6, "other_vocab": 3}` |
| deepseek7b | joint_add_topK | 0.00 | 0.050 | 9.30 | -2.356 | `{"donor_answer": 1, "echo_object_or_relation": 7, "format_or_schema": 9, "other_vocab": 1, "punctuation_or_stop": 1, "recipient_answer": 1}` | `{"donor_answer": 1, "echo_object_or_relation": 7, "format_or_schema": 9, "other_vocab": 1, "punctuation_or_stop": 1, "recipient_answer": 1}` |
| deepseek7b | suppress_current_top | 1.00 | 0.111 | 3.11 | -0.569 | `{"donor_answer": 3, "echo_object_or_relation": 4, "format_or_schema": 6, "other_semantic_value": 1, "other_vocab": 2, "punctuation_or_stop": 2}` | `{"echo_object_or_relation": 7, "format_or_schema": 8, "other_vocab": 1, "punctuation_or_stop": 1, "recipient_answer": 1}` |
| deepseek7b | suppress_current_top | 1.25 | 0.500 | 2.06 | -0.347 | `{"donor_answer": 9, "echo_object_or_relation": 3, "format_or_schema": 2, "other_semantic_value": 1, "other_vocab": 1, "punctuation_or_stop": 2}` | `{"echo_object_or_relation": 7, "format_or_schema": 8, "other_vocab": 1, "punctuation_or_stop": 1, "recipient_answer": 1}` |

## Suppressed Class Summary

| model | class | n | mean alpha needed | mean margin donor vs top | top tokens |
|---|---|---:|---:|---:|---|
| qwen3 | recipient_answer | 20 | 5.597 | -7.534 | `{" fruit": 2, " gray": 2, " none": 2, " object": 2, " orange": 2, " red": 2, " sweet": 4, " vegetable": 2, " yellow": 2}` |
| glm4 | donor_answer | 10 | 0.000 | 0.000 | `{" fruit": 2, " orange": 2, " red": 2, " vegetable": 2, " yellow": 2}` |
| glm4 | echo_object_or_relation | 6 | 2.364 | -2.188 | `{" stone": 5, " taste": 1}` |
| glm4 | other_vocab | 4 | 0.263 | -0.250 | `{" B": 4}` |
| deepseek7b | format_or_schema | 9 | 1.160 | -1.493 | `{" The": 9}` |
| deepseek7b | echo_object_or_relation | 7 | 1.984 | -2.812 | `{" carrot": 2, " category": 2, " stone": 1, " taste": 2}` |
| deepseek7b | punctuation_or_stop | 1 | 2.434 | -3.250 | `{" ?\n\n": 1}` |
| deepseek7b | donor_answer | 1 | 0.000 | 0.000 | `{" orange": 1}` |
| deepseek7b | recipient_answer | 1 | 3.668 | -5.188 | `{" object": 1}` |
| deepseek7b | other_vocab | 1 | 3.968 | -5.562 | `{" __": 1}` |

## Strict Interpretation

- If suppressing only the current top competitor does not make donor top1, the failure is multi-competitor or global readout geometry, not a single blocking token.
- If it does make donor top1, Phase 742 near-closure was mainly blocked by a local competitor class.
- This is still a final readout intervention; it does not prove the natural circuit that performs suppression.

Atlas graph: nodes=19 edges=26
