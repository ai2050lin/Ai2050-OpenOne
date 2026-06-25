# Phase 633 Cross-Model Summary

目标：对 Phase632 的自然 prefix writer 做去重后的多节点 cumulative restore，检查 prompt_last residual writer field 能否闭合 token0 prefix gate。

## deepseek7b

- rows: 82 / raw_cases: 256 / target_seen: 82
- candidate_nodes: ['L26_layer_out', 'L27_layer_out', 'L25_layer_out', 'L24_layer_out', 'L26_attn_out', 'L23_layer_out', 'L26_mlp_out', 'L25_attn_out', 'L24_attn_out', 'L22_layer_out', 'L22_attn_out', 'L24_mlp_out']
- set_defs: {'top1': ['L26_layer_out'], 'top2': ['L26_layer_out', 'L27_layer_out'], 'top4': ['L26_layer_out', 'L27_layer_out', 'L25_layer_out', 'L24_layer_out'], 'top8': ['L26_layer_out', 'L27_layer_out', 'L25_layer_out', 'L24_layer_out', 'L26_attn_out', 'L23_layer_out', 'L26_mlp_out', 'L25_attn_out'], 'top12': ['L26_layer_out', 'L27_layer_out', 'L25_layer_out', 'L24_layer_out', 'L26_attn_out', 'L23_layer_out', 'L26_mlp_out', 'L25_attn_out', 'L24_attn_out', 'L22_layer_out', 'L22_attn_out', 'L24_mlp_out']}
- downstream_layers: [22, 23, 24, 25, 26, 27]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |
|---|---:|---:|---:|---:|---|
| repair_prompt | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| base | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| semantic_cumulative | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| top1_random_semantic | 1/82 | 1/82 | 0/82 | -6.189 |  ?

:77,  :2,  No:1,  c:1,  v:1 |
| top1_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.409 |  ?

:81,  c:1 |
| top1_restore_semantic | 21/82 | 21/82 | 0/82 | -1.662 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| top1_reverse_semantic | 0/82 | 0/82 | 0/82 | -11.152 |  ?

:68,  No:6,  :5,  c:3 |
| top12_random_semantic | 0/82 | 0/82 | 0/82 | -5.989 |  ?

:59,  :19,  No:2,  c:2 |
| top12_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| top12_restore_semantic | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top12_reverse_semantic | 0/82 | 0/82 | 0/82 | -10.592 |  ?

:60,  No:16,  c:3,  r:2,  :1 |
| top2_random_semantic | 0/82 | 0/82 | 0/82 | -6.312 |  ?

:70,  :8,  No:3,  c:1 |
| top2_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| top2_restore_semantic | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top2_reverse_semantic | 0/82 | 0/82 | 0/82 | -10.592 |  ?

:60,  No:16,  c:3,  r:2,  :1 |
| top4_random_semantic | 0/82 | 0/82 | 0/82 | -6.340 |  ?

:65,  :13,  ?
:1,  yes:1,  c:1,  No:1 |
| top4_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| top4_restore_semantic | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top4_reverse_semantic | 0/82 | 0/82 | 0/82 | -10.592 |  ?

:60,  No:16,  c:3,  r:2,  :1 |
| top8_random_semantic | 0/82 | 0/82 | 0/82 | -6.190 |  ?

:63,  :15,  no:1,  r:1,  yes:1,  c:1 |
| top8_remove_from_repair | 0/82 | 0/82 | 0/82 | -6.356 |  ?

:81,  c:1 |
| top8_restore_semantic | 20/82 | 20/82 | 0/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top8_reverse_semantic | 0/82 | 0/82 | 0/82 | -10.592 |  ?

:60,  No:16,  c:3,  r:2,  :1 |
| top4_restore | 20/82 | 4/82 | 15/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top1_restore | 21/82 | 3/82 | 18/82 | -1.662 |  ?

:54,  v:21,  o:3,  :2,  c:1,  yes:1 |
| top2_restore | 20/82 | 3/82 | 17/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top8_restore | 20/82 | 3/82 | 17/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |
| top12_restore | 20/82 | 3/82 | 17/82 | -1.699 |  ?

:56,  v:20,  o:3,  c:1,  :1,  yes:1 |

### Examples

- sample=0 mode=base tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\nTo solve'
- sample=0 mode=semantic_cumulative tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\n2\n'
- sample=0 mode=top1_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.500 text=' ?\n\n2\n'
- sample=0 mode=top4_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=top8_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=top12_restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.562 text=' ?\n\n2\n'
- sample=0 mode=top12_random_semantic tok0=' ?\n\n' exact=False wrong=False margin=-4.938 text=' ?\n\n2\n'
- sample=0 mode=top12_reverse_semantic tok0=' ?\n\n' exact=False wrong=False margin=-8.750 text=' ?\n\n2\n'

## glm4

- rows: 31 / raw_cases: 256 / target_seen: 31
- candidate_nodes: ['L38_layer_out', 'L37_layer_out', 'L36_layer_out', 'L39_layer_out', 'L35_layer_out', 'L34_layer_out', 'L33_layer_out', 'L32_layer_out', 'L32_attn_out', 'L38_mlp_out', 'L31_layer_out', 'L29_layer_out']
- set_defs: {'top1': ['L38_layer_out'], 'top2': ['L38_layer_out', 'L37_layer_out'], 'top4': ['L38_layer_out', 'L37_layer_out', 'L36_layer_out', 'L39_layer_out'], 'top8': ['L38_layer_out', 'L37_layer_out', 'L36_layer_out', 'L39_layer_out', 'L35_layer_out', 'L34_layer_out', 'L33_layer_out', 'L32_layer_out'], 'top12': ['L38_layer_out', 'L37_layer_out', 'L36_layer_out', 'L39_layer_out', 'L35_layer_out', 'L34_layer_out', 'L33_layer_out', 'L32_layer_out', 'L32_attn_out', 'L38_mlp_out', 'L31_layer_out', 'L29_layer_out']}
- downstream_layers: [34, 35, 36, 37, 38, 39]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |
|---|---:|---:|---:|---:|---|
| repair_prompt | 29/31 | 28/31 | 1/31 | 1.710 |  v:29,  c:2 |
| semantic_cumulative | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| base | 11/31 | 2/31 | 9/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| top1_random_semantic | 10/31 | 10/31 | 0/31 | -0.220 |  o:14,  v:10,  c:3,  r:1,  Yes:1,  False:1 |
| top1_remove_from_repair | 10/31 | 10/31 | 0/31 | -0.238 |  o:14,  v:10,  c:4,  Yes:2,  No:1 |
| top1_restore_semantic | 29/31 | 29/31 | 0/31 | 1.712 |  v:29,  c:2 |
| top1_reverse_semantic | 0/31 | 0/31 | 0/31 | -1.979 |  o:18,  True:5,  No:5,  c:2,  Yes:1 |
| top12_random_semantic | 11/31 | 11/31 | 0/31 | -0.228 |  v:11,  o:11,  c:4,  Yes:2,  True:2,  No:1 |
| top12_remove_from_repair | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| top12_restore_semantic | 29/31 | 29/31 | 0/31 | 1.710 |  v:29,  c:2 |
| top12_reverse_semantic | 0/31 | 0/31 | 0/31 | -2.064 |  o:17,  No:8,  Yes:2,  True:2,  c:2 |
| top2_random_semantic | 11/31 | 11/31 | 0/31 | -0.349 |  o:15,  v:11,  c:2,  True:1,  No:1,  Yes:1 |
| top2_remove_from_repair | 10/31 | 10/31 | 0/31 | -0.238 |  o:14,  v:10,  c:4,  Yes:2,  No:1 |
| top2_restore_semantic | 29/31 | 29/31 | 0/31 | 1.712 |  v:29,  c:2 |
| top2_reverse_semantic | 0/31 | 0/31 | 0/31 | -1.979 |  o:18,  True:5,  No:5,  c:2,  Yes:1 |
| top4_random_semantic | 6/31 | 6/31 | 0/31 | -0.419 |  o:15,  v:6,  Yes:4,  c:3,  No:2,  r:1 |
| top4_remove_from_repair | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| top4_restore_semantic | 29/31 | 29/31 | 0/31 | 1.710 |  v:29,  c:2 |
| top4_reverse_semantic | 0/31 | 0/31 | 0/31 | -2.064 |  o:17,  No:8,  Yes:2,  True:2,  c:2 |
| top8_random_semantic | 9/31 | 9/31 | 0/31 | -0.194 |  o:12,  v:9,  c:4,  No:3,  Yes:2,  True:1 |
| top8_remove_from_repair | 11/31 | 11/31 | 0/31 | -0.226 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| top8_restore_semantic | 29/31 | 29/31 | 0/31 | 1.710 |  v:29,  c:2 |
| top8_reverse_semantic | 0/31 | 0/31 | 0/31 | -2.064 |  o:17,  No:8,  Yes:2,  True:2,  c:2 |
| top1_restore | 29/31 | 5/31 | 24/31 | 1.712 |  v:29,  c:2 |
| top2_restore | 29/31 | 5/31 | 24/31 | 1.712 |  v:29,  c:2 |
| top4_restore | 29/31 | 5/31 | 24/31 | 1.710 |  v:29,  c:2 |
| top8_restore | 29/31 | 5/31 | 24/31 | 1.710 |  v:29,  c:2 |
| top12_restore | 29/31 | 5/31 | 24/31 | 1.710 |  v:29,  c:2 |

### Examples

- sample=20 mode=base tok0=' v' exact=False wrong=True margin=0.500 text=' v22'
- sample=20 mode=semantic_cumulative tok0=' v' exact=True wrong=False margin=0.500 text=' v05'
- sample=20 mode=top1_restore_semantic tok0=' v' exact=True wrong=False margin=2.875 text=' v05'
- sample=20 mode=top4_restore_semantic tok0=' v' exact=True wrong=False margin=2.812 text=' v05'
- sample=20 mode=top8_restore_semantic tok0=' v' exact=True wrong=False margin=2.812 text=' v05'
- sample=20 mode=top12_restore_semantic tok0=' v' exact=True wrong=False margin=2.812 text=' v05'
- sample=20 mode=top12_random_semantic tok0=' v' exact=True wrong=False margin=0.250 text=' v05'
- sample=20 mode=top12_reverse_semantic tok0=' o' exact=False wrong=False margin=-1.688 text=' o05'

## qwen3

- rows: 17 / raw_cases: 256 / target_seen: 17
- candidate_nodes: ['L34_layer_out', 'L33_layer_out', 'L32_layer_out', 'L35_layer_out', 'L34_attn_out', 'L32_attn_out', 'L30_layer_out', 'L31_layer_out', 'L28_layer_out', 'L29_layer_out', 'L33_attn_out', 'L25_layer_out']
- set_defs: {'top1': ['L34_layer_out'], 'top2': ['L34_layer_out', 'L33_layer_out'], 'top4': ['L34_layer_out', 'L33_layer_out', 'L32_layer_out', 'L35_layer_out'], 'top8': ['L34_layer_out', 'L33_layer_out', 'L32_layer_out', 'L35_layer_out', 'L34_attn_out', 'L32_attn_out', 'L30_layer_out', 'L31_layer_out'], 'top12': ['L34_layer_out', 'L33_layer_out', 'L32_layer_out', 'L35_layer_out', 'L34_attn_out', 'L32_attn_out', 'L30_layer_out', 'L31_layer_out', 'L28_layer_out', 'L29_layer_out', 'L33_attn_out', 'L25_layer_out']}
- downstream_layers: [29, 30, 31, 32, 33, 34, 35]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |
|---|---:|---:|---:|---:|---|
| repair_prompt | 14/17 | 11/17 | 3/17 | 1.110 |  v:14,  :3 |
| semantic_cumulative | 10/17 | 10/17 | 0/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| base | 10/17 | 1/17 | 9/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| top1_random_semantic | 9/17 | 9/17 | 0/17 | 0.140 |  v:9,  ?

:4,  :3,  o:1 |
| top1_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.272 |  v:10,  ?

:6,  :1 |
| top1_restore_semantic | 14/17 | 14/17 | 0/17 | 1.044 |  v:14,  :3 |
| top1_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.596 |  ?

:8,  v:6,  o:3 |
| top12_random_semantic | 6/17 | 6/17 | 0/17 | -0.110 |  v:6,  ?

:5,  o:4,  :2 |
| top12_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| top12_restore_semantic | 14/17 | 14/17 | 0/17 | 1.110 |  v:14,  :3 |
| top12_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.537 |  ?

:7,  v:6,  o:4 |
| top2_random_semantic | 10/17 | 10/17 | 0/17 | 0.221 |  v:10,  ?

:5,  :2 |
| top2_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.272 |  v:10,  ?

:6,  :1 |
| top2_restore_semantic | 14/17 | 14/17 | 0/17 | 1.044 |  v:14,  :3 |
| top2_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.596 |  ?

:8,  v:6,  o:3 |
| top4_random_semantic | 11/17 | 11/17 | 0/17 | 0.140 |  v:11,  ?

:3,  :2,  o:1 |
| top4_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| top4_restore_semantic | 14/17 | 14/17 | 0/17 | 1.110 |  v:14,  :3 |
| top4_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.537 |  ?

:7,  v:6,  o:4 |
| top8_random_semantic | 10/17 | 10/17 | 0/17 | 0.485 |  v:10,  :3,  ?

:2,  o:2 |
| top8_remove_from_repair | 10/17 | 7/17 | 3/17 | 0.213 |  v:10,  ?

:5,  :1,  o:1 |
| top8_restore_semantic | 14/17 | 14/17 | 0/17 | 1.110 |  v:14,  :3 |
| top8_reverse_semantic | 6/17 | 6/17 | 0/17 | -0.537 |  ?

:7,  v:6,  o:4 |
| top4_restore | 14/17 | 4/17 | 10/17 | 1.110 |  v:14,  :3 |
| top12_restore | 14/17 | 4/17 | 10/17 | 1.110 |  v:14,  :3 |
| top8_restore | 14/17 | 3/17 | 11/17 | 1.110 |  v:14,  :3 |
| top1_restore | 14/17 | 3/17 | 11/17 | 1.044 |  v:14,  :3 |
| top2_restore | 14/17 | 3/17 | 11/17 | 1.044 |  v:14,  :3 |

### Examples

- sample=22 mode=base tok0=' v' exact=False wrong=True margin=2.000 text=' v22'
- sample=22 mode=semantic_cumulative tok0=' v' exact=True wrong=False margin=2.000 text=' v05'
- sample=22 mode=top1_restore_semantic tok0=' v' exact=True wrong=False margin=1.500 text=' v05'
- sample=22 mode=top4_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=top8_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=top12_restore_semantic tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
- sample=22 mode=top12_random_semantic tok0=' v' exact=True wrong=False margin=1.250 text=' v05'
- sample=22 mode=top12_reverse_semantic tok0=' v' exact=True wrong=False margin=2.500 text=' v05'
