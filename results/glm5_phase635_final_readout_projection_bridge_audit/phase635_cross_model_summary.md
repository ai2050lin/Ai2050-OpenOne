# Phase 635 Cross-Model Summary

目标：审计 source/format state 到 final_norm/lm_head 读出竞争之间的 final readout bridge。

## qwen3

- rows: 17 / raw_cases: 256 / target_seen: 17
- readout_scale: 0.25
- downstream_layers: [29, 30, 31, 32, 33, 34, 35]
- source_layer_map: {'prompt_last': 27, 'answer_label': 27, 'question_mark_answer': 27, 'relation_tail': 27, 'question_subject': 27, 'question_all': 27}

| mode | tok0 | exact | wrong_exact | mean_rank | mean_margin | out_proj | out_cos | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 10/17 | 1/17 | 9/17 | 1.9 | 0.213 | 0.000 | 0.000 |  v:10,  ?

:5,  :1,  o:1 |
| repair_prompt | 14/17 | 11/17 | 3/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| semantic_cumulative | 10/17 | 10/17 | 0/17 | 1.9 | 0.213 | 0.000 | 0.000 |  v:10,  ?

:5,  :1,  o:1 |
| source_all6 | 14/17 | 4/17 | 10/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| source_all6_semantic | 14/17 | 14/17 | 0/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| final_input_repair | 14/17 | 3/17 | 11/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| final_input_repair_semantic | 14/17 | 14/17 | 0/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| final_output_repair | 14/17 | 3/17 | 11/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| final_output_repair_semantic | 14/17 | 14/17 | 0/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| final_output_source | 14/17 | 3/17 | 11/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| final_output_source_semantic | 14/17 | 14/17 | 0/17 | 1.2 | 1.110 | 0.581 | 0.023 |  v:14,  :3 |
| readout_delta | 17/17 | 3/17 | 14/17 | 1.0 | 56.974 | 39.638 | 1.000 |  v:17 |
| readout_delta_semantic | 17/17 | 17/17 | 0/17 | 1.0 | 56.974 | 39.638 | 1.000 |  v:17 |

### Examples

- sample=22 mode=base tok0=' v' rank=1 exact=False margin=2.000 text=' v22'
- sample=22 mode=repair_prompt tok0=' v' rank=1 exact=True margin=1.375 text=' v05'
- sample=22 mode=semantic_cumulative tok0=' v' rank=1 exact=True margin=2.000 text=' v05'
- sample=22 mode=source_all6 tok0=' v' rank=1 exact=True margin=1.375 text=' v05'
- sample=22 mode=source_all6_semantic tok0=' v' rank=1 exact=True margin=1.375 text=' v05'
- sample=22 mode=final_input_repair tok0=' v' rank=1 exact=False margin=1.375 text=' v22'
- sample=22 mode=final_input_repair_semantic tok0=' v' rank=1 exact=True margin=1.375 text=' v05'
- sample=22 mode=final_output_repair tok0=' v' rank=1 exact=False margin=1.375 text=' v22'
- sample=22 mode=final_output_repair_semantic tok0=' v' rank=1 exact=True margin=1.375 text=' v05'
- sample=22 mode=final_output_source tok0=' v' rank=1 exact=False margin=1.375 text=' v22'

## glm4

- rows: 31 / raw_cases: 256 / target_seen: 31
- readout_scale: 0.25
- downstream_layers: [34, 35, 36, 37, 38, 39]
- source_layer_map: {'prompt_last': 32, 'answer_label': 32, 'question_mark_answer': 32, 'relation_tail': 32, 'question_subject': 32, 'question_all': 32}

| mode | tok0 | exact | wrong_exact | mean_rank | mean_margin | out_proj | out_cos | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 11/31 | 2/31 | 9/31 | 2.7 | -0.226 | 0.000 | 0.000 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| repair_prompt | 29/31 | 28/31 | 1/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| semantic_cumulative | 11/31 | 11/31 | 0/31 | 2.7 | -0.226 | 0.000 | 0.000 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| source_all6 | 29/31 | 5/31 | 24/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| source_all6_semantic | 29/31 | 29/31 | 0/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| final_input_repair | 29/31 | 5/31 | 24/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| final_input_repair_semantic | 29/31 | 29/31 | 0/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| final_output_repair | 29/31 | 5/31 | 24/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| final_output_repair_semantic | 29/31 | 29/31 | 0/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| final_output_source | 29/31 | 5/31 | 24/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| final_output_source_semantic | 29/31 | 29/31 | 0/31 | 1.1 | 1.710 | 2.148 | 0.046 |  v:29,  c:2 |
| readout_delta | 31/31 | 5/31 | 26/31 | 1.0 | 41.673 | 46.475 | 1.000 |  v:31 |
| readout_delta_semantic | 31/31 | 31/31 | 0/31 | 1.0 | 41.673 | 46.475 | 1.000 |  v:31 |

### Examples

- sample=20 mode=base tok0=' v' rank=1 exact=False margin=0.500 text=' v22'
- sample=20 mode=repair_prompt tok0=' v' rank=1 exact=True margin=2.812 text=' v05'
- sample=20 mode=semantic_cumulative tok0=' v' rank=1 exact=True margin=0.500 text=' v05'
- sample=20 mode=source_all6 tok0=' v' rank=1 exact=False margin=2.812 text=' v22'
- sample=20 mode=source_all6_semantic tok0=' v' rank=1 exact=True margin=2.812 text=' v05'
- sample=20 mode=final_input_repair tok0=' v' rank=1 exact=False margin=2.812 text=' v22'
- sample=20 mode=final_input_repair_semantic tok0=' v' rank=1 exact=True margin=2.812 text=' v05'
- sample=20 mode=final_output_repair tok0=' v' rank=1 exact=False margin=2.812 text=' v22'
- sample=20 mode=final_output_repair_semantic tok0=' v' rank=1 exact=True margin=2.812 text=' v05'
- sample=20 mode=final_output_source tok0=' v' rank=1 exact=False margin=2.812 text=' v22'

## deepseek7b

- rows: 82 / raw_cases: 256 / target_seen: 82
- readout_scale: 0.25
- downstream_layers: [22, 23, 24, 25, 26, 27]
- source_layer_map: {'prompt_last': 25, 'answer_label': 21, 'question_mark_answer': 21, 'relation_tail': 23, 'question_subject': 21, 'question_all': 20}

| mode | tok0 | exact | wrong_exact | mean_rank | mean_margin | out_proj | out_cos | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 0/82 | 0/82 | 0/82 | 92.8 | -6.356 | 0.000 | 0.000 |  ?

:81,  c:1 |
| repair_prompt | 20/82 | 20/82 | 0/82 | 9.4 | -1.699 | 3.618 | 0.078 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| semantic_cumulative | 0/82 | 0/82 | 0/82 | 92.8 | -6.356 | 0.000 | 0.000 |  ?

:81,  c:1 |
| source_all6 | 21/82 | 4/82 | 16/82 | 9.9 | -1.755 | 3.578 | 0.078 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| source_all6_semantic | 21/82 | 21/82 | 0/82 | 9.9 | -1.755 | 3.578 | 0.078 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| final_input_repair | 20/82 | 2/82 | 18/82 | 9.4 | -1.699 | 3.618 | 0.078 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| final_input_repair_semantic | 20/82 | 20/82 | 0/82 | 9.4 | -1.699 | 3.618 | 0.078 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| final_output_repair | 20/82 | 2/82 | 18/82 | 9.4 | -1.699 | 3.618 | 0.078 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| final_output_repair_semantic | 20/82 | 20/82 | 0/82 | 9.4 | -1.699 | 3.618 | 0.078 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| final_output_source | 21/82 | 2/82 | 19/82 | 9.9 | -1.755 | 3.578 | 0.078 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| final_output_source_semantic | 21/82 | 21/82 | 0/82 | 9.9 | -1.755 | 3.578 | 0.078 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| readout_delta | 82/82 | 3/82 | 79/82 | 1.0 | 55.759 | 48.258 | 1.000 |  v:82 |
| readout_delta_semantic | 82/82 | 81/82 | 1/82 | 1.0 | 55.759 | 48.258 | 1.000 |  v:82 |

### Examples

- sample=0 mode=base tok0=' ?\n\n' rank=22 exact=False margin=-5.812 text=' ?\n\nTo solve'
- sample=0 mode=repair_prompt tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\nTo solve'
- sample=0 mode=semantic_cumulative tok0=' ?\n\n' rank=22 exact=False margin=-5.812 text=' ?\n\n2\n'
- sample=0 mode=source_all6 tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\nTo solve'
- sample=0 mode=source_all6_semantic tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=final_input_repair tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\nTo solve'
- sample=0 mode=final_input_repair_semantic tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=final_output_repair tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\nTo solve'
- sample=0 mode=final_output_repair_semantic tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=final_output_source tok0=' ?\n\n' rank=4 exact=False margin=-2.562 text=' ?\n\nTo solve'
