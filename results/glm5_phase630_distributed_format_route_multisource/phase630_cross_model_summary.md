# Phase 630 Cross-Model Summary

Distributed format route multi-source sweep.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- format layers: [20, 21, 22, 23, 24, 25]
- downstream layers: [22, 23, 24, 25, 26, 27]
- groups: ['prompt_last', 'answer_label', 'question_mark_answer', 'relation_tail', 'question_subject', 'question_all']
- components: ['layer_out']
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

### Baselines

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.000, tok2:0.000 |
| repair_prompt | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.256, tok2:0.256 |
| semantic_cumulative_only | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.988, tok2:0.024 |
| semantic_cumulative_random | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.098, tok2:0.061 |

### Best Exact

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| answer_label_L21_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L22_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L23_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L24_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| answer_label_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| prompt_last_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| question_mark_answer_L21_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| question_mark_answer_L22_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| question_mark_answer_L23_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.293 |
| question_mark_answer_L24_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| question_mark_answer_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| relation_tail_L23_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| prompt_last_L23_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| question_all_L20_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.293 |
| question_all_L21_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.280 |
| question_all_L22_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.280 |
| question_all_L23_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.280 |
| question_all_L24_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.280 |
| question_all_L25_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.280 |
| relation_tail_L20_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| relation_tail_L21_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| relation_tail_L22_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| relation_tail_L24_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.317 |
| relation_tail_L25_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.280 |

### Best Tok0

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| answer_label_L21_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L22_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L23_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L24_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| answer_label_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| prompt_last_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| question_mark_answer_L21_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| question_mark_answer_L22_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| question_mark_answer_L23_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.293 |
| question_mark_answer_L24_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| question_mark_answer_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| relation_tail_L23_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.280 |
| answer_label_L21_layer_out | 3/82 | 18/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.037 |
| answer_label_L23_layer_out | 3/82 | 18/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.037 |
| answer_label_L25_layer_out | 3/82 | 18/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.049 |
| prompt_last_L25_layer_out | 3/82 | 17/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.049 |
| question_mark_answer_L21_layer_out | 3/82 | 18/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.037 |
| question_mark_answer_L22_layer_out | 3/82 | 18/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.037 |
| question_mark_answer_L23_layer_out | 3/82 | 18/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.037 |
| question_mark_answer_L25_layer_out | 3/82 | 17/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.049 |
| answer_label_L22_layer_out | 2/82 | 19/82 | 0.305 | tok0:0.256, tok1:0.024, tok2:0.024 |
| answer_label_L24_layer_out | 2/82 | 18/82 | 0.305 | tok0:0.256, tok1:0.024, tok2:0.037 |
| question_mark_answer_L24_layer_out | 2/82 | 18/82 | 0.305 | tok0:0.256, tok1:0.024, tok2:0.024 |
| relation_tail_L23_layer_out | 2/82 | 19/82 | 0.305 | tok0:0.256, tok1:0.024, tok2:0.024 |

### Examples

- o17 / r31 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - semantic_cumulative_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L21_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L22_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L24_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o29 / r31 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

I think` [' ?\n\n', 'I', ' think']
  - semantic_cumulative_only: ` ?

21` [' ?\n\n', '2', '1']
  - answer_label_L21_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L22_layer_out_semantic: ` ?

21` [' ?\n\n', '2', '1']
  - answer_label_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L24_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o95 / r64 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - semantic_cumulative_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L21_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L22_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L24_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o06 / r64 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - semantic_cumulative_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L21_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L22_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - answer_label_L24_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o17 / r64 correct=v91 old_wrong=v05
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` c77` [' c', '7', '7']
  - semantic_cumulative_only: ` ?

99` [' ?\n\n', '9', '9']
  - answer_label_L21_layer_out_semantic: ` c91` [' c', '9', '1']
  - answer_label_L22_layer_out_semantic: ` c91` [' c', '9', '1']
  - answer_label_L23_layer_out_semantic: ` c91` [' c', '9', '1']
  - answer_label_L24_layer_out_semantic: ` c91` [' c', '9', '1']

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- format layers: [32, 33, 34, 35, 36, 37]
- downstream layers: [34, 35, 36, 37, 38, 39]
- groups: ['prompt_last', 'answer_label', 'question_mark_answer', 'relation_tail', 'question_subject', 'question_all']
- components: ['layer_out']
- tokenization: `{'v05': {'ids': [348, 100002], 'tokens': [' v', '05']}, 'v91': {'ids': [348, 104327], 'tokens': [' v', '91']}, 'v22': {'ids': [348, 99241], 'tokens': [' v', '22']}, 'v48': {'ids': [348, 100933], 'tokens': [' v', '48']}}`

### Baselines

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 2/31 | 9/31 | 0.419 | tok0:0.355, tok1:0.065 |
| repair_prompt | 28/31 | 1/31 | 1.839 | tok0:0.935, tok1:0.903 |
| semantic_cumulative_only | 11/31 | 0/31 | 0.710 | tok0:0.355, tok1:1.000 |
| semantic_cumulative_random | 2/31 | 9/31 | 0.419 | tok0:0.355, tok1:0.226 |

### Best Exact

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| answer_label_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| answer_label_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| answer_label_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L35_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| question_mark_answer_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| question_mark_answer_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| question_mark_answer_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| relation_tail_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| relation_tail_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| relation_tail_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| answer_label_L35_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| answer_label_L36_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| answer_label_L37_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| prompt_last_L36_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| prompt_last_L37_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| question_all_L32_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| question_all_L33_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| question_all_L34_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| question_all_L35_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| question_all_L36_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| question_all_L37_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |

### Best Tok0

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| answer_label_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| answer_label_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| answer_label_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| prompt_last_L35_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| question_mark_answer_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| question_mark_answer_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| question_mark_answer_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| relation_tail_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| relation_tail_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| relation_tail_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| answer_label_L32_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| answer_label_L33_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| answer_label_L34_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| prompt_last_L32_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| prompt_last_L33_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| prompt_last_L34_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| prompt_last_L35_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| question_mark_answer_L32_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| question_mark_answer_L33_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| question_mark_answer_L34_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| relation_tail_L32_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |

### Examples

- o43 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '22']
  - repair_prompt: ` v05` [' v', '05']
  - semantic_cumulative_only: ` v05` [' v', '05']
  - answer_label_L32_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L33_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L34_layer_out_semantic: ` v05` [' v', '05']
  - prompt_last_L32_layer_out_semantic: ` v05` [' v', '05']
- o95 / r64 correct=v05 old_wrong=v91
  - base: ` o95` [' o', '95']
  - repair_prompt: ` v91` [' v', '91']
  - semantic_cumulative_only: ` o05` [' o', '05']
  - answer_label_L32_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L33_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L34_layer_out_semantic: ` v05` [' v', '05']
  - prompt_last_L32_layer_out_semantic: ` v05` [' v', '05']
- o43 / r31 correct=v05 old_wrong=v48
  - base: ` v48` [' v', '48']
  - repair_prompt: ` c77` [' c', '77']
  - semantic_cumulative_only: ` v05` [' v', '05']
  - answer_label_L32_layer_out_semantic: ` No05` [' No', '05']
  - answer_label_L33_layer_out_semantic: ` No05` [' No', '05']
  - answer_label_L34_layer_out_semantic: ` c05` [' c', '05']
  - prompt_last_L32_layer_out_semantic: ` c05` [' c', '05']
- o17 / r64 correct=v05 old_wrong=v91
  - base: ` o17` [' o', '17']
  - repair_prompt: ` v05` [' v', '05']
  - semantic_cumulative_only: ` o05` [' o', '05']
  - answer_label_L32_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L33_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L34_layer_out_semantic: ` v05` [' v', '05']
  - prompt_last_L32_layer_out_semantic: ` v05` [' v', '05']
- o43 / r64 correct=v05 old_wrong=v91
  - base: ` o43` [' o', '43']
  - repair_prompt: ` v05` [' v', '05']
  - semantic_cumulative_only: ` o05` [' o', '05']
  - answer_label_L32_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L33_layer_out_semantic: ` v05` [' v', '05']
  - answer_label_L34_layer_out_semantic: ` v05` [' v', '05']
  - prompt_last_L32_layer_out_semantic: ` v05` [' v', '05']

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- format layers: [27, 28, 29, 30, 31, 32]
- downstream layers: [29, 30, 31, 32, 33, 34, 35]
- groups: ['prompt_last', 'answer_label', 'question_mark_answer', 'relation_tail', 'question_subject', 'question_all']
- components: ['layer_out']
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

### Baselines

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 1/17 | 9/17 | 0.706 | tok0:0.588, tok1:0.059, tok2:0.059 |
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| semantic_cumulative_only | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| semantic_cumulative_random | 2/17 | 8/17 | 0.824 | tok0:0.588, tok1:0.235, tok2:0.176 |

### Best Exact

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| question_all_L27_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L28_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L29_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L30_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L31_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L32_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_mark_answer_L27_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| answer_label_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| prompt_last_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| relation_tail_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| question_all_L31_layer_out_random_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:0.824 |
| relation_tail_L27_layer_out_random_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:0.824 |
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| answer_label_L28_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| answer_label_L29_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| answer_label_L30_layer_out_random_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| prompt_last_L28_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| prompt_last_L29_layer_out_random_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.765 |
| prompt_last_L32_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| question_subject_L29_layer_out_random_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| relation_tail_L28_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| answer_label_L27_layer_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| answer_label_L29_layer_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| answer_label_L31_layer_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |

### Best Tok0

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| question_all_L27_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L28_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L29_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L30_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L31_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_all_L32_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| question_mark_answer_L27_layer_out_semantic | 14/17 | 0/17 | 2.471 | tok0:0.824, tok1:1.000, tok2:1.000 |
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| question_all_L27_layer_out | 4/17 | 10/17 | 1.294 | tok0:0.824, tok1:0.294, tok2:0.294 |
| question_all_L28_layer_out | 4/17 | 10/17 | 1.294 | tok0:0.824, tok1:0.294, tok2:0.294 |
| question_mark_answer_L27_layer_out | 4/17 | 10/17 | 1.294 | tok0:0.824, tok1:0.294, tok2:0.294 |
| question_all_L29_layer_out | 3/17 | 11/17 | 1.176 | tok0:0.824, tok1:0.235, tok2:0.235 |
| question_all_L32_layer_out | 3/17 | 11/17 | 1.176 | tok0:0.824, tok1:0.235, tok2:0.235 |
| question_all_L30_layer_out | 2/17 | 12/17 | 1.059 | tok0:0.824, tok1:0.176, tok2:0.176 |
| question_all_L31_layer_out | 2/17 | 12/17 | 1.059 | tok0:0.824, tok1:0.176, tok2:0.176 |
| answer_label_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| prompt_last_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| relation_tail_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| answer_label_L27_layer_out | 4/17 | 9/17 | 1.235 | tok0:0.765, tok1:0.294, tok2:0.294 |
| relation_tail_L27_layer_out | 4/17 | 9/17 | 1.235 | tok0:0.765, tok1:0.294, tok2:0.294 |
| prompt_last_L27_layer_out | 3/17 | 10/17 | 1.118 | tok0:0.765, tok1:0.235, tok2:0.235 |
| question_all_L31_layer_out_random_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:0.824 |
| relation_tail_L27_layer_out_random_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:0.824 |
| answer_label_L28_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |

### Examples

- o58 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '2', '2']
  - repair_prompt: ` v05` [' v', '0', '5']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - question_all_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L28_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L29_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L30_layer_out_semantic: ` v05` [' v', '0', '5']
- o95 / r64 correct=v05 old_wrong=v91
  - base: ` v05` [' v', '0', '5']
  - repair_prompt: ` v05` [' v', '0', '5']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - question_all_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L28_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L29_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L30_layer_out_semantic: ` v05` [' v', '0', '5']
- o06 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '2', '2']
  - repair_prompt: ` v05` [' v', '0', '5']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - question_all_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L28_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L29_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L30_layer_out_semantic: ` v05` [' v', '0', '5']
- o58 / r31 correct=v22 old_wrong=v48
  - base: ` v48` [' v', '4', '8']
  - repair_prompt: ` 22` [' ', '2', '2']
  - semantic_cumulative_only: ` v22` [' v', '2', '2']
  - question_all_L27_layer_out_semantic: ` 22` [' ', '2', '2']
  - question_all_L28_layer_out_semantic: ` 22` [' ', '2', '2']
  - question_all_L29_layer_out_semantic: ` 22` [' ', '2', '2']
  - question_all_L30_layer_out_semantic: ` 22` [' ', '2', '2']
- o06 / r31 correct=v05 old_wrong=v48
  - base: ` v48` [' v', '4', '8']
  - repair_prompt: ` v48` [' v', '4', '8']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - question_all_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L28_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L29_layer_out_semantic: ` v05` [' v', '0', '5']
  - question_all_L30_layer_out_semantic: ` v05` [' v', '0', '5']
