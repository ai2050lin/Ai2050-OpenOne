# Phase 636 Cross-Model Summary

目标：拆解 token0 correct prefix 被哪些 competitor token 类别压制。

## qwen3

- rows: 17 / raw_cases: 256 / target_seen: 17
- top_k: 20 / readout_scale: 0.25
- source_layer_map: {'prompt_last': 27, 'answer_label': 27, 'question_mark_answer': 27, 'relation_tail': 27, 'question_subject': 27, 'question_all': 27}

### Mode Ladder

| mode | tok0 | mean_rank | margin_vs_top | top0_category | top0_text |
|---|---:|---:|---:|---|---|
| base | 11/17 | 1.9 | -0.346 | correct_prefix:11, newline:5, space:1 |  v:11,  ?\n\n:5,  :1 |
| repair_prompt | 14/17 | 1.2 | -0.118 | correct_prefix:14, space:3 |  v:14,  :3 |
| source_all6 | 14/17 | 1.2 | -0.118 | correct_prefix:14, space:3 |  v:14,  :3 |
| final_output_repair | 14/17 | 1.2 | -0.118 | correct_prefix:14, space:3 |  v:14,  :3 |
| final_output_source | 14/17 | 1.2 | -0.118 | correct_prefix:14, space:3 |  v:14,  :3 |
| readout_delta | 17/17 | 1.0 | 0.000 | correct_prefix:17 |  v:17 |

### Category Margins

| mode | category | seen_rate | winner_rate | mean_best_rank | prefix_minus_group_max | max_tokens |
|---|---|---:|---:|---:|---:|---|
| base | correct_prefix | 1.00 | 0.65 | 1.9 | 0.000 |  v:17 |
| base | newline | 1.00 | 0.29 | 2.3 | 0.316 |  ?\n\n:17 |
| base | punctuation | 1.00 | 0.00 | 7.2 | 2.037 |  ?:17 |
| base | explanation | 1.00 | 0.00 | 7.8 | 2.272 |  The:17 |
| base | word | 1.00 | 0.00 | 3.2 | 0.699 |  o:17 |
| base | space | 1.00 | 0.06 | 3.9 | 0.875 |  :17 |
| base | symbol | 1.00 | 0.00 | 15.9 | 4.391 |  \:13,  ??:3 |
| repair_prompt | correct_prefix | 1.00 | 0.82 | 1.2 | 0.000 |  v:17 |
| repair_prompt | newline | 1.00 | 0.00 | 3.0 | 1.272 |  \n\n:13,  ?\n\n:4 |
| repair_prompt | punctuation | 1.00 | 0.00 | 8.2 | 3.301 |  ?:17 |
| repair_prompt | explanation | 1.00 | 0.00 | 6.5 | 2.544 |  The:17 |
| repair_prompt | word | 1.00 | 0.00 | 4.3 | 1.588 |  o:17 |
| repair_prompt | space | 1.00 | 0.18 | 2.8 | 0.978 |  :17 |
| repair_prompt | symbol | 1.00 | 0.00 | 12.0 | 4.081 |  \:17 |
| source_all6 | correct_prefix | 1.00 | 0.82 | 1.2 | 0.000 |  v:17 |
| source_all6 | newline | 1.00 | 0.00 | 3.0 | 1.272 |  \n\n:13,  ?\n\n:4 |
| source_all6 | punctuation | 1.00 | 0.00 | 8.2 | 3.301 |  ?:17 |
| source_all6 | explanation | 1.00 | 0.00 | 6.5 | 2.544 |  The:17 |
| source_all6 | word | 1.00 | 0.00 | 4.3 | 1.588 |  o:17 |
| source_all6 | space | 1.00 | 0.18 | 2.8 | 0.978 |  :17 |
| source_all6 | symbol | 1.00 | 0.00 | 12.0 | 4.081 |  \:17 |
| final_output_repair | correct_prefix | 1.00 | 0.82 | 1.2 | 0.000 |  v:17 |
| final_output_repair | newline | 1.00 | 0.00 | 3.0 | 1.272 |  \n\n:13,  ?\n\n:4 |
| final_output_repair | punctuation | 1.00 | 0.00 | 8.2 | 3.301 |  ?:17 |
| final_output_repair | explanation | 1.00 | 0.00 | 6.5 | 2.544 |  The:17 |
| final_output_repair | word | 1.00 | 0.00 | 4.3 | 1.588 |  o:17 |
| final_output_repair | space | 1.00 | 0.18 | 2.8 | 0.978 |  :17 |
| final_output_repair | symbol | 1.00 | 0.00 | 12.0 | 4.081 |  \:17 |
| final_output_source | correct_prefix | 1.00 | 0.82 | 1.2 | 0.000 |  v:17 |
| final_output_source | newline | 1.00 | 0.00 | 3.0 | 1.272 |  \n\n:13,  ?\n\n:4 |
| final_output_source | punctuation | 1.00 | 0.00 | 8.2 | 3.301 |  ?:17 |
| final_output_source | explanation | 1.00 | 0.00 | 6.5 | 2.544 |  The:17 |
| final_output_source | word | 1.00 | 0.00 | 4.3 | 1.588 |  o:17 |
| final_output_source | space | 1.00 | 0.18 | 2.8 | 0.978 |  :17 |
| final_output_source | symbol | 1.00 | 0.00 | 12.0 | 4.081 |  \:17 |
| readout_delta | correct_prefix | 1.00 | 1.00 | 1.0 | 0.000 |  v:17 |
| readout_delta | newline | 1.00 | 0.00 | 10.0 | 26.176 |  ?\n\n:10,  \n:7 |
| readout_delta | punctuation | 1.00 | 0.00 | 9.8 | 26.427 |  (:7,  ?:5 |
| readout_delta | explanation | 1.00 | 0.00 | 12.5 | 28.614 |  The:9,  the:2 |
| readout_delta | word | 1.00 | 0.00 | 2.4 | 20.860 | \tv:6,  o:5,  vX:4,  c:2 |
| readout_delta | space | 1.00 | 0.00 | 4.7 | 21.731 |  :13 |
| readout_delta | symbol | 1.00 | 0.00 | 13.7 | 28.562 |  \:6 |
| readout_delta | other | 1.00 | 0.00 | 4.5 | 22.375 | *v:6, $v:4 |

### Examples

- sample=22 mode=base prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: o[word], 3: ?\n\n[newline], 4: [space], 5: ?\n[newline], 6: \n\n[newline], 7: Based[word], 8: ?[punctuation]
- sample=22 mode=repair_prompt prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: o[word], 3: \n\n[newline], 4: ?\n\n[newline], 5: [space], 6: ?\n[newline], 7: ?[punctuation], 8: The[explanation]
- sample=22 mode=source_all6 prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: o[word], 3: \n\n[newline], 4: ?\n\n[newline], 5: [space], 6: ?\n[newline], 7: ?[punctuation], 8: The[explanation]
- sample=22 mode=final_output_repair prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: o[word], 3: \n\n[newline], 4: ?\n\n[newline], 5: [space], 6: ?\n[newline], 7: ?[punctuation], 8: The[explanation]
- sample=22 mode=final_output_source prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: o[word], 3: \n\n[newline], 4: ?\n\n[newline], 5: [space], 6: ?\n[newline], 7: ?[punctuation], 8: The[explanation]
- sample=22 mode=readout_delta prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2:\tv[word], 3:v[word], 4: V[word], 5:*v[other], 6:(v[other], 7:$v[other], 8:.v[other]
- sample=29 mode=base prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: o[word], 4: ?\n\n[newline], 5: \n\n[newline], 6: ?\n[newline], 7: ?[punctuation], 8: The[explanation]
- sample=29 mode=repair_prompt prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: o[word], 4: \n\n[newline], 5: ?\n\n[newline], 6: ?\n[newline], 7: The[explanation], 8: ?[punctuation]
- sample=29 mode=source_all6 prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: o[word], 4: \n\n[newline], 5: ?\n\n[newline], 6: ?\n[newline], 7: The[explanation], 8: ?[punctuation]
- sample=29 mode=final_output_repair prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: o[word], 4: \n\n[newline], 5: ?\n\n[newline], 6: ?\n[newline], 7: The[explanation], 8: ?[punctuation]
- sample=29 mode=final_output_source prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: o[word], 4: \n\n[newline], 5: ?\n\n[newline], 6: ?\n[newline], 7: The[explanation], 8: ?[punctuation]
- sample=29 mode=readout_delta prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: vX[word], 3:\tv[word], 4:$v[other], 5:)v[other], 6:*v[other], 7::v[other], 8:ｖ[other]

## glm4

- rows: 31 / raw_cases: 256 / target_seen: 31
- top_k: 20 / readout_scale: 0.25
- source_layer_map: {'prompt_last': 32, 'answer_label': 32, 'question_mark_answer': 32, 'relation_tail': 32, 'question_subject': 32, 'question_all': 32}

### Mode Ladder

| mode | tok0 | mean_rank | margin_vs_top | top0_category | top0_text |
|---|---:|---:|---:|---|---|
| base | 11/31 | 2.7 | -0.363 | word:17, correct_prefix:11, explanation:3 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| repair_prompt | 29/31 | 1.1 | -0.020 | correct_prefix:29, word:2 |  v:29,  c:2 |
| source_all6 | 29/31 | 1.1 | -0.020 | correct_prefix:29, word:2 |  v:29,  c:2 |
| final_output_repair | 29/31 | 1.1 | -0.020 | correct_prefix:29, word:2 |  v:29,  c:2 |
| final_output_source | 29/31 | 1.1 | -0.020 | correct_prefix:29, word:2 |  v:29,  c:2 |
| readout_delta | 31/31 | 1.0 | 0.000 | correct_prefix:31 |  v:31 |

### Category Margins

| mode | category | seen_rate | winner_rate | mean_best_rank | prefix_minus_group_max | max_tokens |
|---|---|---:|---:|---:|---:|---|
| base | correct_prefix | 1.00 | 0.35 | 2.8 | 0.000 |  v:31 |
| base | newline | 1.00 | 0.00 | 17.0 | 3.938 |  ?\n:4 |
| base | punctuation | 1.00 | 0.00 | 14.7 | 3.966 |  ?:8,  (:5 |
| base | explanation | 1.00 | 0.10 | 3.9 | 0.851 |  Yes:26,  No:5 |
| base | word | 1.00 | 0.55 | 1.5 | -0.183 |  o:23,  c:7,  False:1 |
| base | space | 1.00 | 0.00 | 9.7 | 2.264 |  :31 |
| base | symbol | 1.00 | 0.00 | 15.0 | 4.219 |  <:1 |
| repair_prompt | correct_prefix | 1.00 | 0.94 | 1.1 | 0.000 |  v:31 |
| repair_prompt | newline | 1.00 | 0.00 | 14.8 | 4.562 |  ?\n:6 |
| repair_prompt | punctuation | 1.00 | 0.00 | 11.8 | 4.272 |  {:10,  ?:10,  [:2,  (:1 |
| repair_prompt | explanation | 1.00 | 0.00 | 4.5 | 2.460 |  Yes:21,  The:7,  No:3 |
| repair_prompt | word | 1.00 | 0.06 | 1.9 | 0.623 |  c:30,  o:1 |
| repair_prompt | space | 1.00 | 0.00 | 6.4 | 2.770 |  :31 |
| source_all6 | correct_prefix | 1.00 | 0.94 | 1.1 | 0.000 |  v:31 |
| source_all6 | newline | 1.00 | 0.00 | 14.8 | 4.562 |  ?\n:6 |
| source_all6 | punctuation | 1.00 | 0.00 | 11.8 | 4.272 |  {:10,  ?:10,  [:2,  (:1 |
| source_all6 | explanation | 1.00 | 0.00 | 4.5 | 2.460 |  Yes:21,  The:7,  No:3 |
| source_all6 | word | 1.00 | 0.06 | 1.9 | 0.623 |  c:30,  o:1 |
| source_all6 | space | 1.00 | 0.00 | 6.4 | 2.770 |  :31 |
| final_output_repair | correct_prefix | 1.00 | 0.94 | 1.1 | 0.000 |  v:31 |
| final_output_repair | newline | 1.00 | 0.00 | 14.8 | 4.562 |  ?\n:6 |
| final_output_repair | punctuation | 1.00 | 0.00 | 11.8 | 4.272 |  {:10,  ?:10,  [:2,  (:1 |
| final_output_repair | explanation | 1.00 | 0.00 | 4.5 | 2.460 |  Yes:21,  The:7,  No:3 |
| final_output_repair | word | 1.00 | 0.06 | 1.9 | 0.623 |  c:30,  o:1 |
| final_output_repair | space | 1.00 | 0.00 | 6.4 | 2.770 |  :31 |
| final_output_source | correct_prefix | 1.00 | 0.94 | 1.1 | 0.000 |  v:31 |
| final_output_source | newline | 1.00 | 0.00 | 14.8 | 4.562 |  ?\n:6 |
| final_output_source | punctuation | 1.00 | 0.00 | 11.8 | 4.272 |  {:10,  ?:10,  [:2,  (:1 |
| final_output_source | explanation | 1.00 | 0.00 | 4.5 | 2.460 |  Yes:21,  The:7,  No:3 |
| final_output_source | word | 1.00 | 0.06 | 1.9 | 0.623 |  c:30,  o:1 |
| final_output_source | space | 1.00 | 0.00 | 6.4 | 2.770 |  :31 |
| readout_delta | correct_prefix | 1.00 | 1.00 | 1.0 | 0.000 |  v:31 |
| readout_delta | punctuation | 1.00 | 0.00 | 12.5 | 24.844 |  (:2 |
| readout_delta | explanation | 1.00 | 0.00 | 10.3 | 24.171 |  Yes:28,  no:2 |
| readout_delta | word | 1.00 | 0.00 | 2.9 | 20.871 | v:15,  V:7, \tv:6,  r:2,  c:1 |
| readout_delta | space | 1.00 | 0.00 | 15.9 | 25.549 |  :19 |
| readout_delta | other | 1.00 | 0.00 | 2.3 | 19.845 | (v:31 |

### Examples

- sample=20 mode=base prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: o[word], 3: r[word], 4: c[word], 5: [space], 6: Yes[explanation], 7: The[explanation], 8: No[explanation]
- sample=20 mode=repair_prompt prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: r[word], 7: Yes[explanation], 8: V[word]
- sample=20 mode=source_all6 prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: r[word], 7: Yes[explanation], 8: V[word]
- sample=20 mode=final_output_repair prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: r[word], 7: Yes[explanation], 8: V[word]
- sample=20 mode=final_output_source prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: r[word], 7: Yes[explanation], 8: V[word]
- sample=20 mode=readout_delta prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2:(v[other], 3:v[word], 4:\tv[word], 5: V[word], 6: r[word], 7:.v[other], 8:=v[other]
- sample=29 mode=base prefix_rank=2 top0=' o'/word ladder=1: o[word], 2: v[correct_prefix], 3: c[word], 4: r[word], 5: Yes[explanation], 6: No[explanation], 7: True[word], 8: [space]
- sample=29 mode=repair_prompt prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: {[punctuation], 7: r[word], 8: None[word]
- sample=29 mode=source_all6 prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: {[punctuation], 7: r[word], 8: None[word]
- sample=29 mode=final_output_repair prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: {[punctuation], 7: r[word], 8: None[word]
- sample=29 mode=final_output_source prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: c[word], 3: [space], 4: o[word], 5: The[explanation], 6: {[punctuation], 7: r[word], 8: None[word]
- sample=29 mode=readout_delta prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2:(v[other], 3:v[word], 4:\tv[word], 5: V[word], 6: r[word], 7: c[word], 8: Yes[explanation]

## deepseek7b

- rows: 82 / raw_cases: 256 / target_seen: 82
- top_k: 20 / readout_scale: 0.25
- source_layer_map: {'prompt_last': 25, 'answer_label': 21, 'question_mark_answer': 21, 'relation_tail': 23, 'question_subject': 21, 'question_all': 20}

### Mode Ladder

| mode | tok0 | mean_rank | margin_vs_top | top0_category | top0_text |
|---|---:|---:|---:|---|---|
| base | 0/82 | 92.8 | -6.356 | newline:81, word:1 |  ?\n\n:81,  c:1 |
| repair_prompt | 20/82 | 9.4 | -1.970 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |
| source_all6 | 20/82 | 9.9 | -2.008 | newline:57, correct_prefix:20, word:2, space:2, explanation:1 |  ?\n\n:57,  v:20,  :2,  c:1,  o:1,  yes:1 |
| final_output_repair | 20/82 | 9.4 | -1.970 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |
| final_output_source | 20/82 | 9.9 | -2.008 | newline:57, correct_prefix:20, word:2, space:2, explanation:1 |  ?\n\n:57,  v:20,  :2,  c:1,  o:1,  yes:1 |
| readout_delta | 82/82 | 1.0 | 0.000 | correct_prefix:82 |  v:82 |

### Category Margins

| mode | category | seen_rate | winner_rate | mean_best_rank | prefix_minus_group_max | max_tokens |
|---|---|---:|---:|---:|---:|---|
| base | correct_prefix | 1.00 | 0.00 | 12.6 | 0.000 |  v:28 |
| base | newline | 1.00 | 0.99 | 1.0 | -6.354 |  ?\n\n:82 |
| base | punctuation | 1.00 | 0.00 | 9.3 | -3.078 |  ?:82 |
| base | explanation | 1.00 | 0.00 | 4.3 | -4.232 |  yes:42,  No:40 |
| base | word | 1.00 | 0.01 | 6.6 | -3.750 |  c:49,  o:32,  r:1 |
| base | number | 1.00 | 0.00 | 18.0 | -2.042 | 1:18 |
| base | space | 1.00 | 0.00 | 2.3 | -5.119 |  :82 |
| base | symbol | 1.00 | 0.00 | 14.5 | -2.234 |  ...:46,  ??:25 |
| repair_prompt | correct_prefix | 1.00 | 0.27 | 5.6 | 0.000 |  v:75 |
| repair_prompt | newline | 1.00 | 0.70 | 1.5 | -1.704 |  ?\n\n:82 |
| repair_prompt | punctuation | 1.00 | 0.00 | 11.5 | 1.617 |  ?:53,  [:29 |
| repair_prompt | explanation | 1.00 | 0.01 | 5.8 | 0.344 |  yes:66,  The:12,  No:4 |
| repair_prompt | word | 1.00 | 0.04 | 4.0 | -0.489 |  c:59,  o:23 |
| repair_prompt | number | 1.00 | 0.00 | 12.9 | 4.527 | 4:7 |
| repair_prompt | space | 1.00 | 0.01 | 2.7 | -0.953 |  :82 |
| repair_prompt | symbol | 1.00 | 0.00 | 15.7 | 1.175 |  ...:37,  ??:17 |
| source_all6 | correct_prefix | 1.00 | 0.27 | 5.9 | 0.000 |  v:75 |
| source_all6 | newline | 1.00 | 0.70 | 1.5 | -1.765 |  ?\n\n:82 |
| source_all6 | punctuation | 1.00 | 0.00 | 11.6 | 1.535 |  ?:51,  [:31 |
| source_all6 | explanation | 1.00 | 0.01 | 5.7 | 0.234 |  yes:67,  The:11,  No:4 |
| source_all6 | word | 1.00 | 0.02 | 4.0 | -0.517 |  c:47,  o:35 |
| source_all6 | number | 1.00 | 0.00 | 14.0 | 4.482 | 4:7 |
| source_all6 | space | 1.00 | 0.02 | 2.6 | -1.065 |  :82 |
| source_all6 | symbol | 1.00 | 0.00 | 16.4 | 1.126 |  ...:43,  ??:11 |
| final_output_repair | correct_prefix | 1.00 | 0.27 | 5.6 | 0.000 |  v:75 |
| final_output_repair | newline | 1.00 | 0.70 | 1.5 | -1.704 |  ?\n\n:82 |
| final_output_repair | punctuation | 1.00 | 0.00 | 11.5 | 1.617 |  ?:53,  [:29 |
| final_output_repair | explanation | 1.00 | 0.01 | 5.8 | 0.344 |  yes:66,  The:12,  No:4 |
| final_output_repair | word | 1.00 | 0.04 | 4.0 | -0.489 |  c:59,  o:23 |
| final_output_repair | number | 1.00 | 0.00 | 12.9 | 4.527 | 4:7 |
| final_output_repair | space | 1.00 | 0.01 | 2.7 | -0.953 |  :82 |
| final_output_repair | symbol | 1.00 | 0.00 | 15.7 | 1.175 |  ...:37,  ??:17 |
| final_output_source | correct_prefix | 1.00 | 0.27 | 5.9 | 0.000 |  v:75 |
| final_output_source | newline | 1.00 | 0.70 | 1.5 | -1.765 |  ?\n\n:82 |
| final_output_source | punctuation | 1.00 | 0.00 | 11.6 | 1.535 |  ?:51,  [:31 |
| final_output_source | explanation | 1.00 | 0.01 | 5.7 | 0.234 |  yes:67,  The:11,  No:4 |
| final_output_source | word | 1.00 | 0.02 | 4.0 | -0.517 |  c:47,  o:35 |
| final_output_source | number | 1.00 | 0.00 | 14.0 | 4.482 | 4:7 |
| final_output_source | space | 1.00 | 0.02 | 2.6 | -1.065 |  :82 |
| final_output_source | symbol | 1.00 | 0.00 | 16.4 | 1.126 |  ...:43,  ??:11 |
| readout_delta | correct_prefix | 1.00 | 1.00 | 1.0 | 0.000 |  v:82 |
| readout_delta | newline | 1.00 | 0.00 | 17.5 | 22.896 |  \n:34,  ?\n\n:1 |
| readout_delta | punctuation | 1.00 | 0.00 | 3.1 | 17.980 |  (:81 |
| readout_delta | explanation | 1.00 | 0.00 | 16.4 | 21.788 |  no:19,  yes:1 |
| readout_delta | word | 1.00 | 0.00 | 3.8 | 19.189 |  c:69, v:13 |
| readout_delta | number | 1.00 | 0.00 | 8.0 | 22.145 | 1:81 |
| readout_delta | space | 1.00 | 0.00 | 2.0 | 11.573 |  :81 |
| readout_delta | symbol | 1.00 | 0.00 | 15.2 | 22.966 |  \:41 |
| readout_delta | other | 1.00 | 0.00 | 3.0 | 19.625 | *v:1 |

### Examples

- sample=0 mode=base prefix_rank=22 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: yes[explanation], 5: c[word], 6: Yes[explanation], 7: ?[punctuation], 8: \n\n[newline]
- sample=0 mode=repair_prompt prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: c[word], 6: o[word], 7: \n\n[newline], 8: ?[punctuation]
- sample=0 mode=source_all6 prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: o[word], 6: c[word], 7: \n\n[newline], 8: ?[punctuation]
- sample=0 mode=final_output_repair prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: c[word], 6: o[word], 7: \n\n[newline], 8: ?[punctuation]
- sample=0 mode=final_output_source prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: o[word], 6: c[word], 7: \n\n[newline], 8: ?[punctuation]
- sample=0 mode=readout_delta prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: ([punctuation], 4: c[word], 5:v[word], 6: V[word], 7: r[word], 8:1[number]
- sample=2 mode=base prefix_rank=10 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: yes[explanation], 5: c[word], 6: Yes[explanation], 7: \n\n[newline], 8: r[word]
- sample=2 mode=repair_prompt prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: v[correct_prefix], 5: c[word], 6: \n\n[newline], 7: ?[punctuation], 8: o[word]
- sample=2 mode=source_all6 prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: v[correct_prefix], 5: c[word], 6: o[word], 7: \n\n[newline], 8: ?[punctuation]
- sample=2 mode=final_output_repair prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: v[correct_prefix], 5: c[word], 6: \n\n[newline], 7: ?[punctuation], 8: o[word]
- sample=2 mode=final_output_source prefix_rank=4 top0=' ?\\n\\n'/newline ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: v[correct_prefix], 5: c[word], 6: o[word], 7: \n\n[newline], 8: ?[punctuation]
- sample=2 mode=readout_delta prefix_rank=1 top0=' v'/correct_prefix ladder=1: v[correct_prefix], 2: [space], 3: ([punctuation], 4:v[word], 5: c[word], 6: V[word], 7: r[word], 8: [[punctuation]
