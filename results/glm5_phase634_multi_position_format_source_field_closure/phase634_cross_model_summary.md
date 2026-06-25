# Phase 634 Cross-Model Summary

目标：测试多位置 source/format field 是否能补上 Phase633 排除 prompt_last 后剩余的 token0 prefix gate 缺口。

## deepseek7b

- rows: 82 / raw_cases: 256 / target_seen: 82
- layer_map: {'prompt_last': 25, 'answer_label': 21, 'question_mark_answer': 21, 'relation_tail': 23, 'question_subject': 21, 'question_all': 20}
- set_defs: {'single_prompt_last': ['prompt_last'], 'single_answer_label': ['answer_label'], 'single_question_mark_answer': ['question_mark_answer'], 'single_relation_tail': ['relation_tail'], 'single_question_all': ['question_all'], 'answer_prompt': ['answer_label', 'prompt_last'], 'qma_prompt': ['question_mark_answer', 'prompt_last'], 'relation_answer_prompt': ['relation_tail', 'answer_label', 'prompt_last'], 'question_all_answer_prompt': ['question_all', 'answer_label', 'prompt_last'], 'answer_qma_relation_prompt': ['answer_label', 'question_mark_answer', 'relation_tail', 'prompt_last'], 'all6': ['prompt_last', 'answer_label', 'question_mark_answer', 'relation_tail', 'question_subject', 'question_all']}
- downstream_layers: [22, 23, 24, 25, 26, 27]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |
|---|---:|---:|---:|---:|---|
| base | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| repair_prompt | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| semantic_cumulative | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| all6_random_semantic | 0/82 | 0/82 | 0/82 | -6.248 |  ?

:77,  No:2,  :2,  c:1 |
| all6_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| all6_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| all6_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| answer_prompt_random_semantic | 0/82 | 0/82 | 0/82 | -6.183 |  ?

:76,  :5,  No:1 |
| answer_prompt_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| answer_prompt_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| answer_prompt_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| answer_qma_relation_prompt_random_semantic | 0/82 | 0/82 | 0/82 | -6.298 |  ?

:77,  No:2,  :2,  c:1 |
| answer_qma_relation_prompt_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| answer_qma_relation_prompt_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| answer_qma_relation_prompt_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| qma_prompt_random_semantic | 0/82 | 0/82 | 0/82 | -6.158 |  ?

:80,  :2 |
| qma_prompt_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| qma_prompt_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| qma_prompt_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| question_all_answer_prompt_random_semantic | 0/82 | 0/82 | 0/82 | -6.286 |  ?

:78,  :3,  c:1 |
| question_all_answer_prompt_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| question_all_answer_prompt_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| question_all_answer_prompt_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| relation_answer_prompt_random_semantic | 0/82 | 0/82 | 0/82 | -6.343 |  ?

:79,  :2,  No:1 |
| relation_answer_prompt_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| relation_answer_prompt_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| relation_answer_prompt_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| single_answer_label_random_semantic | 0/82 | 0/82 | 0/82 | -6.132 |  ?

:69,  :11,  c:2 |
| single_answer_label_remove_from_repair | 0/82 | 0/82 | 0/82 | -5.832 |  ?

:73,  c:7,  No:1,  :1 |
| single_answer_label_restore_semantic | 21/82 | 21/82 | 0/82 | -2.158 |  ?

:57,  v:21,  o:2,  c:1,  yes:1 |
| single_answer_label_reverse_semantic | 0/82 | 0/82 | 0/82 | -9.724 |  ?

:63,  :11,  No:4,  c:4 |
| single_prompt_last_random_semantic | 0/82 | 0/82 | 0/82 | -6.394 |  ?

:74,  :4,  No:2,  c:2 |
| single_prompt_last_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |  ?

:81,  c:1 |
| single_prompt_last_restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| single_prompt_last_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.314 |  ?

:69,  No:5,  :5,  c:3 |
| single_question_all_random_semantic | 1/82 | 1/82 | 0/82 | -6.421 |  ?

:73,  :7,  No:1,  v:1 |
| single_question_all_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| single_question_all_restore_semantic | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| single_question_all_reverse_semantic | 0/82 | 0/82 | 0/82 | -9.607 |  ?

:66,  :13,  c:2,  No:1 |
| single_question_mark_answer_random_semantic | 0/82 | 0/82 | 0/82 | -5.920 |  ?

:75,  :5,  No:1,  c:1 |
| single_question_mark_answer_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.231 |  ?

:77,  c:5 |
| single_question_mark_answer_restore_semantic | 21/82 | 21/82 | 0/82 | -1.979 |  ?

:55,  v:21,  o:4,  c:1,  yes:1 |
| single_question_mark_answer_reverse_semantic | 0/82 | 0/82 | 0/82 | -9.898 |  ?

:70,  :7,  No:4,  c:1 |
| single_relation_tail_random_semantic | 0/82 | 0/82 | 0/82 | -6.322 |  ?

:78,  :2,  c:1,  No:1 |
| single_relation_tail_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.386 |  ?

:82 |
| single_relation_tail_restore_semantic | 21/82 | 21/82 | 0/82 | -1.806 |  ?

:57,  v:21,  o:2,  c:1,  yes:1 |
| single_relation_tail_reverse_semantic | 0/82 | 0/82 | 0/82 | -10.494 |  ?

:70,  :7,  No:3,  c:2 |
| single_question_all_restore | 20/82 | 8/82 | 12/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| question_all_answer_prompt_restore | 21/82 | 4/82 | 16/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| all6_restore | 21/82 | 4/82 | 16/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| single_prompt_last_restore | 21/82 | 3/82 | 17/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| answer_prompt_restore | 21/82 | 3/82 | 18/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| qma_prompt_restore | 21/82 | 3/82 | 18/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| relation_answer_prompt_restore | 21/82 | 3/82 | 18/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| answer_qma_relation_prompt_restore | 21/82 | 3/82 | 18/82 | -1.755 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| single_question_mark_answer_restore | 21/82 | 3/82 | 18/82 | -1.979 |  ?

:55,  v:21,  o:4,  c:1,  yes:1 |
| single_answer_label_restore | 21/82 | 3/82 | 18/82 | -2.158 |  ?

:57,  v:21,  o:2,  c:1,  yes:1 |
| single_relation_tail_restore | 21/82 | 2/82 | 19/82 | -1.806 |  ?

:57,  v:21,  o:2,  c:1,  yes:1 |
| single_question_all_random | 1/82 | 0/82 | 1/82 | -6.421 |  ?

:73,  :7,  No:1,  v:1 |
| single_question_mark_answer_random | 0/82 | 0/82 | 0/82 | -5.920 |  ?

:75,  :5,  No:1,  c:1 |
| single_answer_label_random | 0/82 | 0/82 | 0/82 | -6.132 |  ?

:69,  :11,  c:2 |
| qma_prompt_random | 0/82 | 0/82 | 0/82 | -6.158 |  ?

:80,  :2 |
| answer_prompt_random | 0/82 | 0/82 | 0/82 | -6.183 |  ?

:76,  :5,  No:1 |

### Examples

- sample=0 mode=base tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\nTo solve'
- sample=0 mode=semantic_cumulative tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\n2\n'
- sample=0 mode=single_prompt_last_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=answer_prompt_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=relation_answer_prompt_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=question_all_answer_prompt_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=all6_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=all6_random_semantic tok0=' ?\n\n' exact=False wrong=False margin=-7.062 text=' ?\n\n2\n'
- sample=0 mode=all6_reverse_semantic tok0=' ?\n\n' exact=False wrong=False margin=-10.062 text=' ?\n\n2\n'
- sample=2 mode=base tok0=' ?\n\n' exact=False wrong=False margin=-3.438 text=' ?\n\nTo solve'

## glm4

- rows: 31 / raw_cases: 256 / target_seen: 31
- layer_map: {'prompt_last': 32, 'answer_label': 32, 'question_mark_answer': 32, 'relation_tail': 32, 'question_subject': 32, 'question_all': 32}
- set_defs: {'single_prompt_last': ['prompt_last'], 'single_answer_label': ['answer_label'], 'single_question_mark_answer': ['question_mark_answer'], 'single_relation_tail': ['relation_tail'], 'single_question_all': ['question_all'], 'answer_prompt': ['answer_label', 'prompt_last'], 'qma_prompt': ['question_mark_answer', 'prompt_last'], 'relation_answer_prompt': ['relation_tail', 'answer_label', 'prompt_last'], 'question_all_answer_prompt': ['question_all', 'answer_label', 'prompt_last'], 'answer_qma_relation_prompt': ['answer_label', 'question_mark_answer', 'relation_tail', 'prompt_last'], 'all6': ['prompt_last', 'answer_label', 'question_mark_answer', 'relation_tail', 'question_subject', 'question_all']}
- downstream_layers: [34, 35, 36, 37, 38, 39]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |
|---|---:|---:|---:|---:|---|
| base | 11/31 | 2/31 | 9/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| repair_prompt | 29/31 | 28/31 | 1/31 | 1.710 |  v:29,  c:2 |
| semantic_cumulative | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| all6_random_semantic | 13/31 | 13/31 | 0/31 | -0.190 |  v:13,  o:9,  c:4,  Yes:3,  No:2 |
| all6_remove_from_repair | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| all6_restore_semantic | 29/31 | 29/31 | 0/31 | 1.710 |  v:29,  c:2 |
| all6_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.823 |  o:12,  No:8,  c:6,  Yes:3,  v:2 |
| answer_prompt_random_semantic | 12/31 | 12/31 | 0/31 | -0.208 |  v:12,  o:12,  c:4,  Yes:2,  No:1 |
| answer_prompt_remove_from_repair | 10/31 | 10/31 | 0/31 | -0.103 |  c:13,  v:10,  o:4,  No:2,  Yes:2 |
| answer_prompt_restore_semantic | 30/31 | 30/31 | 0/31 | 1.442 |  v:30,  No:1 |
| answer_prompt_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.946 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| answer_qma_relation_prompt_random_semantic | 13/31 | 13/31 | 0/31 | -0.244 |  v:13,  o:13,  Yes:3,  c:1,  No:1 |
| answer_qma_relation_prompt_remove_from_repair | 9/31 | 9/31 | 0/31 | -0.060 |  c:14,  v:9,  o:4,  No:2,  Yes:2 |
| answer_qma_relation_prompt_restore_semantic | 30/31 | 30/31 | 0/31 | 1.431 |  v:30,  No:1 |
| answer_qma_relation_prompt_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.903 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| qma_prompt_random_semantic | 11/31 | 11/31 | 0/31 | -0.224 |  o:14,  v:11,  Yes:3,  c:2,  No:1 |
| qma_prompt_remove_from_repair | 9/31 | 9/31 | 0/31 | -0.109 |  c:14,  v:9,  o:4,  No:2,  Yes:2 |
| qma_prompt_restore_semantic | 30/31 | 30/31 | 0/31 | 1.435 |  v:30,  No:1 |
| qma_prompt_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.973 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| question_all_answer_prompt_random_semantic | 11/31 | 11/31 | 0/31 | -0.220 |  v:11,  o:9,  c:4,  Yes:4,  No:3 |
| question_all_answer_prompt_remove_from_repair | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| question_all_answer_prompt_restore_semantic | 29/31 | 29/31 | 0/31 | 1.710 |  v:29,  c:2 |
| question_all_answer_prompt_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.823 |  o:12,  No:8,  c:6,  Yes:3,  v:2 |
| relation_answer_prompt_random_semantic | 11/31 | 11/31 | 0/31 | -0.202 |  v:11,  o:10,  c:6,  No:3,  Yes:1 |
| relation_answer_prompt_remove_from_repair | 9/31 | 9/31 | 0/31 | -0.060 |  c:14,  v:9,  o:4,  No:2,  Yes:2 |
| relation_answer_prompt_restore_semantic | 30/31 | 30/31 | 0/31 | 1.431 |  v:30,  No:1 |
| relation_answer_prompt_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.903 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| single_answer_label_random_semantic | 12/31 | 12/31 | 0/31 | -0.264 |  o:13,  v:12,  c:2,  No:2,  Yes:2 |
| single_answer_label_remove_from_repair | 10/31 | 10/31 | 0/31 | -0.103 |  c:13,  v:10,  o:4,  No:2,  Yes:2 |
| single_answer_label_restore_semantic | 30/31 | 30/31 | 0/31 | 1.442 |  v:30,  No:1 |
| single_answer_label_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.946 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| single_prompt_last_random_semantic | 10/31 | 10/31 | 0/31 | -0.278 |  o:12,  v:10,  c:5,  No:2,  Yes:2 |
| single_prompt_last_remove_from_repair | 11/31 | 10/31 | 1/31 | -0.056 |  c:14,  v:11,  o:4,  No:2 |
| single_prompt_last_restore_semantic | 30/31 | 30/31 | 0/31 | 1.409 |  v:30,  c:1 |
| single_prompt_last_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.912 |  o:15,  No:8,  c:4,  Yes:2,  v:2 |
| single_question_all_random_semantic | 11/31 | 11/31 | 0/31 | -0.204 |  v:11,  o:11,  Yes:4,  c:3,  No:2 |
| single_question_all_remove_from_repair | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| single_question_all_restore_semantic | 29/31 | 29/31 | 0/31 | 1.710 |  v:29,  c:2 |
| single_question_all_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.823 |  o:12,  No:8,  c:6,  Yes:3,  v:2 |
| single_question_mark_answer_random_semantic | 12/31 | 12/31 | 0/31 | -0.258 |  o:14,  v:12,  c:2,  Yes:2,  No:1 |
| single_question_mark_answer_remove_from_repair | 9/31 | 9/31 | 0/31 | -0.109 |  c:14,  v:9,  o:4,  No:2,  Yes:2 |
| single_question_mark_answer_restore_semantic | 30/31 | 30/31 | 0/31 | 1.435 |  v:30,  No:1 |
| single_question_mark_answer_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.973 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| single_relation_tail_random_semantic | 10/31 | 10/31 | 0/31 | -0.296 |  o:13,  v:10,  c:4,  Yes:2,  True:1,  No:1 |
| single_relation_tail_remove_from_repair | 9/31 | 9/31 | 0/31 | -0.060 |  c:14,  v:9,  o:4,  No:2,  Yes:2 |
| single_relation_tail_restore_semantic | 30/31 | 30/31 | 0/31 | 1.431 |  v:30,  No:1 |
| single_relation_tail_reverse_semantic | 2/31 | 2/31 | 0/31 | -1.903 |  o:14,  No:8,  c:5,  Yes:2,  v:2 |
| single_answer_label_restore | 30/31 | 5/31 | 25/31 | 1.442 |  v:30,  No:1 |
| answer_prompt_restore | 30/31 | 5/31 | 25/31 | 1.442 |  v:30,  No:1 |
| single_question_mark_answer_restore | 30/31 | 5/31 | 25/31 | 1.435 |  v:30,  No:1 |
| qma_prompt_restore | 30/31 | 5/31 | 25/31 | 1.435 |  v:30,  No:1 |
| single_relation_tail_restore | 30/31 | 5/31 | 25/31 | 1.431 |  v:30,  No:1 |
| relation_answer_prompt_restore | 30/31 | 5/31 | 25/31 | 1.431 |  v:30,  No:1 |
| answer_qma_relation_prompt_restore | 30/31 | 5/31 | 25/31 | 1.431 |  v:30,  No:1 |
| single_prompt_last_restore | 30/31 | 5/31 | 25/31 | 1.409 |  v:30,  c:1 |
| single_question_all_restore | 29/31 | 5/31 | 24/31 | 1.710 |  v:29,  c:2 |
| question_all_answer_prompt_restore | 29/31 | 5/31 | 24/31 | 1.710 |  v:29,  c:2 |
| all6_restore | 29/31 | 5/31 | 24/31 | 1.710 |  v:29,  c:2 |
| all6_random | 13/31 | 2/31 | 11/31 | -0.190 |  v:13,  o:9,  c:4,  Yes:3,  No:2 |
| answer_qma_relation_prompt_random | 13/31 | 2/31 | 11/31 | -0.244 |  v:13,  o:13,  Yes:3,  c:1,  No:1 |
| single_question_mark_answer_random | 12/31 | 2/31 | 10/31 | -0.258 |  o:14,  v:12,  c:2,  Yes:2,  No:1 |
| single_answer_label_random | 12/31 | 2/31 | 10/31 | -0.264 |  o:13,  v:12,  c:2,  No:2,  Yes:2 |
| relation_answer_prompt_random | 11/31 | 2/31 | 9/31 | -0.202 |  v:11,  o:10,  c:6,  No:3,  Yes:1 |

### Examples

- sample=20 mode=base tok0=' v' exact=False wrong=True margin=0.500 text=' v22'
- sample=20 mode=semantic_cumulative tok0=' v' exact=True wrong=False margin=0.500 text=' v05'
- sample=20 mode=single_prompt_last_restore_semantic tok0=' v' exact=True wrong=False margin=2.312 text=' v05'
- sample=20 mode=answer_prompt_restore_semantic tok0=' v' exact=True wrong=False margin=2.312 text=' v05'
- sample=20 mode=relation_answer_prompt_restore_semantic tok0=' v' exact=True wrong=False margin=2.125 text=' v05'
- sample=20 mode=question_all_answer_prompt_restore_semantic tok0=' v' exact=True wrong=False margin=2.812 text=' v05'
- sample=20 mode=all6_restore_semantic tok0=' v' exact=True wrong=False margin=2.812 text=' v05'
- sample=20 mode=all6_random_semantic tok0=' v' exact=True wrong=False margin=0.375 text=' v05'
- sample=20 mode=all6_reverse_semantic tok0=' o' exact=False wrong=False margin=-0.438 text=' o05'
- sample=29 mode=base tok0=' o' exact=False wrong=False margin=-0.125 text=' o95'

## qwen3

- rows: 17 / raw_cases: 256 / target_seen: 17
- layer_map: {'prompt_last': 27, 'answer_label': 27, 'question_mark_answer': 27, 'relation_tail': 27, 'question_subject': 27, 'question_all': 27}
- set_defs: {'single_prompt_last': ['prompt_last'], 'single_answer_label': ['answer_label'], 'single_question_mark_answer': ['question_mark_answer'], 'single_relation_tail': ['relation_tail'], 'single_question_all': ['question_all'], 'answer_prompt': ['answer_label', 'prompt_last'], 'qma_prompt': ['question_mark_answer', 'prompt_last'], 'relation_answer_prompt': ['relation_tail', 'answer_label', 'prompt_last'], 'question_all_answer_prompt': ['question_all', 'answer_label', 'prompt_last'], 'answer_qma_relation_prompt': ['answer_label', 'question_mark_answer', 'relation_tail', 'prompt_last'], 'all6': ['prompt_last', 'answer_label', 'question_mark_answer', 'relation_tail', 'question_subject', 'question_all']}
- downstream_layers: [29, 30, 31, 32, 33, 34, 35]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |
|---|---:|---:|---:|---:|---|
| base | 10/17 | 1/17 | 9/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| repair_prompt | 14/17 | 11/17 | 3/17 | 1.110 |  v:14,  :3 |
| semantic_cumulative | 10/17 | 10/17 | 0/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| all6_random_semantic | 8/17 | 8/17 | 0/17 | 0.221 |  v:8,  ?

:6,  o:2,  :1 |
| all6_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| all6_restore_semantic | 14/17 | 14/17 | 0/17 | 1.110 |  v:14,  :3 |
| all6_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.235 |  ?

:9,  v:6,  o:2 |
| answer_prompt_random_semantic | 7/17 | 7/17 | 0/17 | 0.184 |  v:7,  ?

:7,  o:2,  :1 |
| answer_prompt_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.691 |  v:11,  ?

:5,  :1 |
| answer_prompt_restore_semantic | 13/17 | 13/17 | 0/17 | 0.868 |  v:13,  :3,  o:1 |
| answer_prompt_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.360 |  ?

:9,  v:6,  o:2 |
| answer_qma_relation_prompt_random_semantic | 9/17 | 9/17 | 0/17 | 0.132 |  v:9,  ?

:6,  :2 |
| answer_qma_relation_prompt_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.669 |  v:11,  ?

:5,  :1 |
| answer_qma_relation_prompt_restore_semantic | 13/17 | 13/17 | 0/17 | 0.919 |  v:13,  :3,  

:1 |
| answer_qma_relation_prompt_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.294 |  ?

:9,  v:6,  o:2 |
| qma_prompt_random_semantic | 10/17 | 10/17 | 0/17 | 0.169 |  v:10,  ?

:4,  o:2,  :1 |
| qma_prompt_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.662 |  v:11,  ?

:5,  :1 |
| qma_prompt_restore_semantic | 14/17 | 14/17 | 0/17 | 0.897 |  v:14,  :3 |
| qma_prompt_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.324 |  ?

:9,  v:6,  o:2 |
| question_all_answer_prompt_random_semantic | 12/17 | 12/17 | 0/17 | 0.426 |  v:12,  ?

:3,  :2 |
| question_all_answer_prompt_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| question_all_answer_prompt_restore_semantic | 14/17 | 14/17 | 0/17 | 1.110 |  v:14,  :3 |
| question_all_answer_prompt_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.235 |  ?

:9,  v:6,  o:2 |
| relation_answer_prompt_random_semantic | 8/17 | 8/17 | 0/17 | 0.118 |  v:8,  ?

:6,  :2,  o:1 |
| relation_answer_prompt_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.669 |  v:11,  ?

:5,  :1 |
| relation_answer_prompt_restore_semantic | 13/17 | 13/17 | 0/17 | 0.919 |  v:13,  :3,  

:1 |
| relation_answer_prompt_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.294 |  ?

:9,  v:6,  o:2 |
| single_answer_label_random_semantic | 12/17 | 12/17 | 0/17 | 0.353 |  v:12,  ?

:3,  o:2 |
| single_answer_label_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.691 |  v:11,  ?

:5,  :1 |
| single_answer_label_restore_semantic | 13/17 | 13/17 | 0/17 | 0.868 |  v:13,  :3,  o:1 |
| single_answer_label_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.360 |  ?

:9,  v:6,  o:2 |
| single_prompt_last_random_semantic | 9/17 | 9/17 | 0/17 | 0.154 |  v:9,  ?

:6,  o:1,  :1 |
| single_prompt_last_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.728 |  v:11,  ?

:5,  :1 |
| single_prompt_last_restore_semantic | 13/17 | 13/17 | 0/17 | 0.824 |  v:13,  :3,  o:1 |
| single_prompt_last_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.368 |  ?

:9,  v:6,  o:2 |
| single_question_all_random_semantic | 9/17 | 9/17 | 0/17 | 0.228 |  v:9,  ?

:7,  o:1 |
| single_question_all_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| single_question_all_restore_semantic | 14/17 | 14/17 | 0/17 | 1.110 |  v:14,  :3 |
| single_question_all_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.235 |  ?

:9,  v:6,  o:2 |
| single_question_mark_answer_random_semantic | 7/17 | 7/17 | 0/17 | -0.037 |  ?

:8,  v:7,  o:1,  :1 |
| single_question_mark_answer_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.662 |  v:11,  ?

:5,  :1 |
| single_question_mark_answer_restore_semantic | 14/17 | 14/17 | 0/17 | 0.897 |  v:14,  :3 |
| single_question_mark_answer_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.324 |  ?

:9,  v:6,  o:2 |
| single_relation_tail_random_semantic | 10/17 | 10/17 | 0/17 | 0.154 |  v:10,  ?

:6,  :1 |
| single_relation_tail_remove_from_repair | 11/17 | 8/17 | 3/17 | 0.669 |  v:11,  ?

:5,  :1 |
| single_relation_tail_restore_semantic | 13/17 | 13/17 | 0/17 | 0.919 |  v:13,  :3,  

:1 |
| single_relation_tail_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.294 |  ?

:9,  v:6,  o:2 |
| single_question_all_restore | 14/17 | 4/17 | 10/17 | 1.110 |  v:14,  :3 |
| question_all_answer_prompt_restore | 14/17 | 4/17 | 10/17 | 1.110 |  v:14,  :3 |
| all6_restore | 14/17 | 4/17 | 10/17 | 1.110 |  v:14,  :3 |
| single_question_mark_answer_restore | 14/17 | 4/17 | 10/17 | 0.897 |  v:14,  :3 |
| qma_prompt_restore | 14/17 | 4/17 | 10/17 | 0.897 |  v:14,  :3 |
| single_relation_tail_restore | 13/17 | 4/17 | 9/17 | 0.919 |  v:13,  :3,  

:1 |
| relation_answer_prompt_restore | 13/17 | 4/17 | 9/17 | 0.919 |  v:13,  :3,  

:1 |
| answer_qma_relation_prompt_restore | 13/17 | 4/17 | 9/17 | 0.919 |  v:13,  :3,  

:1 |
| single_answer_label_restore | 13/17 | 4/17 | 9/17 | 0.868 |  v:13,  :3,  o:1 |
| answer_prompt_restore | 13/17 | 4/17 | 9/17 | 0.868 |  v:13,  :3,  o:1 |
| single_prompt_last_restore | 13/17 | 3/17 | 10/17 | 0.824 |  v:13,  :3,  o:1 |
| question_all_answer_prompt_random | 12/17 | 2/17 | 10/17 | 0.426 |  v:12,  ?

:3,  :2 |
| single_answer_label_random | 12/17 | 2/17 | 10/17 | 0.353 |  v:12,  ?

:3,  o:2 |
| qma_prompt_random | 10/17 | 2/17 | 8/17 | 0.169 |  v:10,  ?

:4,  o:2,  :1 |
| answer_qma_relation_prompt_random | 9/17 | 2/17 | 7/17 | 0.132 |  v:9,  ?

:6,  :2 |
| answer_prompt_random | 7/17 | 2/17 | 5/17 | 0.184 |  v:7,  ?

:7,  o:2,  :1 |

### Examples

- sample=22 mode=base tok0=' v' exact=False wrong=True margin=2.000 text=' v22'
- sample=22 mode=semantic_cumulative tok0=' v' exact=True wrong=False margin=2.000 text=' v05'
- sample=22 mode=single_prompt_last_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=answer_prompt_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=relation_answer_prompt_restore_semantic tok0=' v' exact=True wrong=False margin=1.500 text=' v05'
- sample=22 mode=question_all_answer_prompt_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=all6_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=all6_random_semantic tok0=' v' exact=True wrong=False margin=2.000 text=' v05'
- sample=22 mode=all6_reverse_semantic tok0=' v' exact=True wrong=False margin=2.750 text=' v05'
- sample=29 mode=base tok0=' v' exact=True wrong=False margin=1.125 text=' v05'
