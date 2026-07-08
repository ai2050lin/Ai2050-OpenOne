# Phase238 Ambiguous Case Report

case_rows: 132
mean_original_behavior_score: 0.6462
mean_calibrated_behavior_score: 0.8133
ambiguous_rows: 13
semantic_mismatch_rows: 13
stable_failure_candidates: 12

## Ambiguous / Semantic Mismatch Rows

| model | case | family | target | alias | score | calibrated | drift | output |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| qwen3 | pf_case_0025 | content_knowledge | hit | strike | 0.00 | 0.60 | protocol_or_over_generation | Strike.   The hammer is a tool used to strike objects, typically for driving in nails, shaping metal |
| qwen3 | pf_case_0026 | content_knowledge | hit | drive | 0.00 | 1.00 | semantic_correct_but_target_mismatch | The hammer is used to drive nails into surfaces. Because it provides a strong impact force.   The ha |
| qwen3 | pf_case_0027 | content_knowledge | hit | strike | 0.00 | 1.00 | semantic_correct_but_target_mismatch | To strike, To break.   Wait, that's not right. The answer should be two words that are the same |
| qwen3 | pf_case_0028 | content_knowledge | hit | striking | 0.00 | 1.00 | semantic_correct_but_target_mismatch | The hammer is a tool used for striking, pounding, or driving objects. It is typically used in carpen |
| qwen3 | pf_case_0030 | content_knowledge | car | vehicle | 0.00 | 0.85 | semantic_correct_but_target_mismatch | The part_of of wheel is vehicle. Because a wheel is a component of a vehicle. Answer: The part_of of |
| qwen3 | pf_case_0032 | content_knowledge | car | vehicle | 0.00 | 1.00 | semantic_correct_but_target_mismatch | The part_of of wheel is: vehicle, machinery, machinery, vehicle, and equipment.   Wait, that's not |
| glm4 | pf_case_0021 | content_knowledge | sour | citrusy | 0.00 | 0.60 | protocol_or_over_generation | Citrusy. While taste is subjective, lemons are generally described as having a citrusy taste. Howeve |
| glm4 | pf_case_0022 | content_knowledge | sour | tart | 0.00 | 1.00 | semantic_correct_but_target_mismatch | The taste of lemon is tart and acidic.  Because lemons contain a high concentration of citric acid,  |
| glm4 | pf_case_0025 | content_knowledge | hit | drive | 0.00 | 0.60 | protocol_or_over_generation | Hammer: Drive or fasten.   However, since the answer must be a single word, the function of a |
| glm4 | pf_case_0026 | content_knowledge | hit | drive | 0.00 | 1.00 | semantic_correct_but_target_mismatch | The function of a hammer is to drive nails into wood because it applies force effectively.   Short r |
| glm4 | pf_case_0028 | content_knowledge | hit | driving | 0.00 | 1.00 | semantic_correct_but_target_mismatch | Function of hammer: 1. Driving nails into wood or other materials. 2. Hammering metal to shape or jo |
| deepseek7b | pf_case_0025 | content_knowledge | hit | strike | 0.00 | 0.60 | protocol_or_over_generation | hammer  What is the function of a hammer? Answer with a sentence. Answer: The hammer is used to stri |
| deepseek7b | pf_case_0027 | content_knowledge | hit | drive | 0.00 | 0.75 | protocol_or_over_generation | The function of hammer is to drive nails into wood and to break apart large masses of stone.  The fu |

## Stable Failure Candidates

| case | family | mode | type | models | mean score |
| --- | --- | --- | --- | --- | ---: |
| pf_case_0001 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| pf_case_0005 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| pf_case_0009 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| pf_case_0013 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| pf_case_0017 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| pf_case_0021 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.4000 |
| pf_case_0025 | content_knowledge | object_relation_value | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| pf_case_0029 | content_knowledge | object_relation_value | stable_semantic_or_target_failure | deepseek7b,glm4,qwen3 | 0.0000 |
| pf_case_0030 | content_knowledge | object_relation_value | stable_semantic_or_target_failure | deepseek7b,glm4,qwen3 | 0.5500 |
| pf_case_0031 | content_knowledge | object_relation_value | stable_semantic_or_target_failure | deepseek7b,glm4,qwen3 | 0.3000 |
| pf_case_0032 | content_knowledge | object_relation_value | stable_semantic_or_target_failure | deepseek7b,glm4,qwen3 | 0.6000 |
| phase236_protocol_short_0001 | output_protocol | short_answer | stable_protocol_failure | deepseek7b,glm4,qwen3 | 0.6000 |
