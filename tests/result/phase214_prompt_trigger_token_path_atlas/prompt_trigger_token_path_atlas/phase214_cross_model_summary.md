# Phase 214 prompt trigger token path atlas

Selected trajectory rows: 340
Trigger token rows: 14794
Path rows: 88764
Success/drift delta rows: 2754

| model | pattern | trigger | anchor | layer | success rows | drift rows | cosine delta | l2 delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | answer_list | list_three | gen_after_step_1 | 27 | 6 | 2 | 0.869056 | -143.338618 |
| deepseek7b | answer_list | list_plausible | gen_after_step_1 | 27 | 6 | 2 | 0.807151 | -114.163239 |
| deepseek7b | answer_explain | explain_because | gen_after_step_1 | 27 | 6 | 2 | 0.799727 | -108.877192 |
| deepseek7b | answer_explain | explain_answer_first | gen_after_step_1 | 27 | 6 | 2 | 0.785277 | -88.347158 |
| deepseek7b | answer_explain | explain_reason | gen_after_step_1 | 27 | 6 | 2 | 0.778157 | -100.404526 |
| deepseek7b | answer_list | list_short_answers | gen_after_step_1 | 27 | 6 | 2 | 0.698516 | -85.158386 |
| deepseek7b | answer_list | list_commas | gen_after_step_1 | 27 | 6 | 2 | 0.689743 | -85.066508 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 7 | 1 | 29 | -0.682425 | 3.021883 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 12 | 1 | 29 | -0.633153 | 4.752852 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 7 | 1 | 29 | -0.632177 | 3.114547 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 20 | 1 | 29 | -0.611031 | 11.011195 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 3 | 1 | 29 | -0.589193 | 1.677702 |
| glm4 | answer_list | list_plausible | gen_after_step_2 | 7 | 1 | 29 | -0.572477 | 3.077666 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 3 | 1 | 29 | -0.556973 | 1.545341 |
| deepseek7b | answer_list | answer_slot | gen_after_step_1 | 27 | 6 | 2 | 0.541172 | -55.035817 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 27 | 1 | 29 | -0.532583 | 42.413443 |
| glm4 | answer_list | list_three | gen_after_step_1 | 7 | 1 | 29 | 0.529849 | -2.751492 |
| glm4 | answer_list | list_three | gen_after_step_1 | 3 | 1 | 29 | 0.528291 | -1.625383 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 28 | 1 | 29 | -0.526250 | 47.824473 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 12 | 1 | 29 | -0.513800 | 4.563952 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 29 | 1 | 29 | -0.511937 | 52.362331 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 30 | 1 | 29 | -0.500831 | 57.606482 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 20 | 1 | 29 | -0.496561 | 10.212054 |
| glm4 | answer_list | list_plausible | gen_after_step_2 | 3 | 1 | 29 | -0.494481 | 1.612806 |
| glm4 | answer_list | list_three | gen_after_step_1 | 12 | 1 | 29 | 0.468741 | -4.342129 |
| glm4 | answer_list | list_three | gen_after_step_1 | 20 | 1 | 29 | 0.467019 | -10.844883 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 28 | 1 | 29 | -0.462800 | 47.469567 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 27 | 1 | 29 | -0.460999 | 41.544010 |
| deepseek7b | answer_list | list_commas | gen_after_step_6 | 27 | 6 | 2 | 0.457564 | -110.783445 |
| glm4 | answer_list | list_short_answers | gen_after_step_3 | 32 | 1 | 29 | -0.449598 | 63.572517 |
| glm4 | answer_list | list_plausible | gen_after_step_2 | 12 | 1 | 29 | -0.441722 | 4.229373 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 29 | 1 | 29 | -0.433302 | 53.306090 |
| deepseek7b | answer_list | list_short_answers | gen_after_step_6 | 27 | 6 | 2 | 0.431231 | -103.261561 |
| glm4 | answer_list | list_three | gen_after_step_1 | 27 | 1 | 29 | 0.424913 | -46.357436 |
| glm4 | answer_list | list_commas | gen_after_step_6 | 7 | 1 | 29 | 0.423587 | -0.715381 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 30 | 1 | 29 | -0.408992 | 58.849932 |
| glm4 | answer_list | list_three | gen_after_step_1 | 28 | 1 | 29 | 0.406034 | -53.186233 |
| glm4 | answer_list | answer_slot | gen_after_step_6 | 32 | 1 | 29 | -0.396613 | 72.399370 |
| glm4 | answer_list | list_commas | gen_after_step_6 | 3 | 1 | 29 | 0.395175 | -0.569436 |
| deepseek7b | answer_explain | answer_slot | gen_after_step_1 | 27 | 6 | 2 | 0.391504 | -32.123322 |
