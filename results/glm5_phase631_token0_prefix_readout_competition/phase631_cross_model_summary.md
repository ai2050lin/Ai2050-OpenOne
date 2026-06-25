# Phase 631 Cross-Model Summary

目标：直接审计第一个生成词元的 prefix/readout competition，并测试 final_norm unembedding 方向注入是否能替代缺失的格式/前缀门。

## deepseek7b

- rows: 82 / raw_cases: 256 / target_seen: 82
- source: {'group': 'answer_label', 'layer': 21, 'component': 'layer_out'}
- downstream_layers: [22, 23, 24, 25, 26, 27]
- scales: [0.125, 0.25, 0.5, 1.0]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | mean_prefix_logit | mean_competitor_logit |
|---|---:|---:|---:|---:|---:|---:|
| base | 0/82 | 0/82 | 0/82 | -6.356 | 11.710 | 18.066 |
| repair_prompt | 20/82 | 20/82 | 0/82 | -1.699 | 15.086 | 16.785 |
| semantic_cumulative | 0/82 | 0/82 | 0/82 | -6.356 | 11.710 | 18.066 |
| best_source | 21/82 | 3/82 | 18/82 | -2.158 | 14.726 | 16.883 |
| best_source_semantic | 21/82 | 21/82 | 0/82 | -2.158 | 14.726 | 16.883 |
| readout_scale1_semantic | 82/82 | 81/82 | 1/82 | 242.183 | 141.329 | -100.854 |
| readout_scale0.5_semantic | 82/82 | 81/82 | 1/82 | 117.930 | 76.549 | -41.381 |
| readout_scale0.25_semantic | 82/82 | 81/82 | 1/82 | 55.759 | 44.095 | -11.665 |
| readout_scale1 | 82/82 | 3/82 | 79/82 | 242.183 | 141.329 | -100.854 |
| readout_scale0.5 | 82/82 | 3/82 | 79/82 | 117.930 | 76.549 | -41.381 |
| readout_scale0.25 | 82/82 | 3/82 | 79/82 | 55.759 | 44.095 | -11.665 |

### Examples

- sample 0 object=o17 relation=r31 correct=v22 prefix=' v' competitor=' ?\n\n'
  - base: tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\nTo solve'
  - semantic_cumulative: tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\n2\n'
  - best_source_semantic: tok0=' ?\n\n' exact=False wrong=False margin=-3.062 text=' ?\n\n2\n'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=244.000 text=' v22'
- sample 2 object=o29 relation=r31 correct=v22 prefix=' v' competitor=' ?\n\n'
  - base: tok0=' ?\n\n' exact=False wrong=False margin=-3.438 text=' ?\n\nTo solve'
  - semantic_cumulative: tok0=' ?\n\n' exact=False wrong=False margin=-3.438 text=' ?\n\n21'
  - best_source_semantic: tok0=' ?\n\n' exact=False wrong=False margin=-2.188 text=' ?\n\n2\n'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=246.500 text=' v22'
- sample 13 object=o95 relation=r64 correct=v22 prefix=' v' competitor=' ?\n\n'
  - base: tok0=' ?\n\n' exact=False wrong=False margin=-7.500 text=' ?\n\nTo solve'
  - semantic_cumulative: tok0=' ?\n\n' exact=False wrong=False margin=-7.500 text=' ?\n\n2\n'
  - best_source_semantic: tok0=' ?\n\n' exact=False wrong=False margin=-2.750 text=' ?\n\n2\n'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=241.500 text=' v22'
- sample 15 object=o06 relation=r64 correct=v22 prefix=' v' competitor=' ?\n\n'
  - base: tok0=' ?\n\n' exact=False wrong=False margin=-5.062 text=' ?\n\nTo solve'
  - semantic_cumulative: tok0=' ?\n\n' exact=False wrong=False margin=-5.062 text=' ?\n\n2\n'
  - best_source_semantic: tok0=' ?\n\n' exact=False wrong=False margin=-3.875 text=' ?\n\n2\n'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=251.000 text=' v22'

## glm4

- rows: 31 / raw_cases: 256 / target_seen: 31
- source: {'group': 'answer_label', 'layer': 32, 'component': 'layer_out'}
- downstream_layers: [34, 35, 36, 37, 38, 39]
- scales: [0.125, 0.25, 0.5, 1.0]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | mean_prefix_logit | mean_competitor_logit |
|---|---:|---:|---:|---:|---:|---:|
| base | 11/31 | 2/31 | 9/31 | -0.226 | 11.498 | 11.724 |
| repair_prompt | 29/31 | 28/31 | 1/31 | 1.710 | 12.639 | 10.929 |
| semantic_cumulative | 11/31 | 11/31 | 0/31 | -0.226 | 11.498 | 11.724 |
| best_source | 30/31 | 5/31 | 25/31 | 1.442 | 12.653 | 11.212 |
| best_source_semantic | 30/31 | 30/31 | 0/31 | 1.442 | 12.653 | 11.212 |
| readout_scale1_semantic | 31/31 | 31/31 | 0/31 | 167.435 | 106.790 | -60.645 |
| readout_scale0.5_semantic | 31/31 | 31/31 | 0/31 | 83.601 | 59.137 | -24.464 |
| readout_scale0.25_semantic | 31/31 | 31/31 | 0/31 | 41.673 | 35.306 | -6.367 |
| readout_scale0.125_semantic | 31/31 | 31/31 | 0/31 | 20.704 | 23.383 | 2.679 |
| readout_scale1 | 31/31 | 5/31 | 26/31 | 167.435 | 106.790 | -60.645 |
| readout_scale0.5 | 31/31 | 5/31 | 26/31 | 83.601 | 59.137 | -24.464 |

### Examples

- sample 20 object=o43 relation=r31 correct=v05 prefix=' v' competitor=' o'
  - base: tok0=' v' exact=False wrong=True margin=0.500 text=' v22'
  - semantic_cumulative: tok0=' v' exact=True wrong=False margin=0.500 text=' v05'
  - best_source_semantic: tok0=' v' exact=True wrong=False margin=2.312 text=' v05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=163.500 text=' v05'
- sample 29 object=o95 relation=r64 correct=v05 prefix=' v' competitor=' o'
  - base: tok0=' o' exact=False wrong=False margin=-0.125 text=' o95'
  - semantic_cumulative: tok0=' o' exact=False wrong=False margin=-0.125 text=' o05'
  - best_source_semantic: tok0=' v' exact=True wrong=False margin=1.938 text=' v05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=164.000 text=' v05'
- sample 36 object=o43 relation=r31 correct=v05 prefix=' v' competitor=' o'
  - base: tok0=' v' exact=False wrong=True margin=0.188 text=' v48'
  - semantic_cumulative: tok0=' v' exact=True wrong=False margin=0.188 text=' v05'
  - best_source_semantic: tok0=' No' exact=False wrong=False margin=2.000 text=' No05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=173.250 text=' v05'
- sample 65 object=o17 relation=r64 correct=v05 prefix=' v' competitor=' o'
  - base: tok0=' o' exact=False wrong=False margin=-0.562 text=' o17'
  - semantic_cumulative: tok0=' o' exact=False wrong=False margin=-0.562 text=' o05'
  - best_source_semantic: tok0=' v' exact=True wrong=False margin=1.188 text=' v05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=162.500 text=' v05'

## qwen3

- rows: 17 / raw_cases: 256 / target_seen: 17
- source: {'group': 'question_all', 'layer': 27, 'component': 'layer_out'}
- downstream_layers: [29, 30, 31, 32, 33, 34, 35]
- scales: [0.125, 0.25, 0.5, 1.0]

| mode | tok0 | exact | wrong_exact | mean_prefix_margin | mean_prefix_logit | mean_competitor_logit |
|---|---:|---:|---:|---:|---:|---:|
| base | 10/17 | 1/17 | 9/17 | 0.213 | 23.478 | 23.265 |
| repair_prompt | 14/17 | 11/17 | 3/17 | 1.110 | 24.331 | 23.221 |
| semantic_cumulative | 10/17 | 10/17 | 0/17 | 0.213 | 23.478 | 23.265 |
| best_source | 14/17 | 4/17 | 10/17 | 1.110 | 24.331 | 23.221 |
| best_source_semantic | 14/17 | 14/17 | 0/17 | 1.110 | 24.331 | 23.221 |
| readout_scale1_semantic | 17/17 | 17/17 | 0/17 | 227.147 | 130.618 | -96.529 |
| readout_scale0.5_semantic | 17/17 | 17/17 | 0/17 | 113.713 | 77.059 | -36.654 |
| readout_scale0.25_semantic | 17/17 | 17/17 | 0/17 | 56.974 | 50.279 | -6.695 |
| readout_scale0.125_semantic | 17/17 | 17/17 | 0/17 | 28.592 | 36.868 | 8.276 |
| readout_scale1 | 17/17 | 3/17 | 14/17 | 227.147 | 130.618 | -96.529 |
| readout_scale0.5 | 17/17 | 3/17 | 14/17 | 113.713 | 77.059 | -36.654 |

### Examples

- sample 22 object=o58 relation=r31 correct=v05 prefix=' v' competitor=' o'
  - base: tok0=' v' exact=False wrong=True margin=2.000 text=' v22'
  - semantic_cumulative: tok0=' v' exact=True wrong=False margin=2.000 text=' v05'
  - best_source_semantic: tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=205.000 text=' v05'
- sample 29 object=o95 relation=r64 correct=v05 prefix=' v' competitor=' '
  - base: tok0=' v' exact=True wrong=False margin=1.125 text=' v05'
  - semantic_cumulative: tok0=' v' exact=True wrong=False margin=1.125 text=' v05'
  - best_source_semantic: tok0=' v' exact=True wrong=False margin=1.125 text=' v05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=216.500 text=' v05'
- sample 30 object=o06 relation=r31 correct=v05 prefix=' v' competitor=' o'
  - base: tok0=' v' exact=False wrong=True margin=1.125 text=' v22'
  - semantic_cumulative: tok0=' v' exact=True wrong=False margin=1.125 text=' v05'
  - best_source_semantic: tok0=' v' exact=True wrong=False margin=1.375 text=' v05'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=207.500 text=' v05'
- sample 38 object=o58 relation=r31 correct=v22 prefix=' v' competitor=' o'
  - base: tok0=' v' exact=False wrong=True margin=0.375 text=' v48'
  - semantic_cumulative: tok0=' v' exact=True wrong=False margin=0.375 text=' v22'
  - best_source_semantic: tok0=' ' exact=False wrong=False margin=1.750 text=' 22'
  - readout_scale1_semantic: tok0=' v' exact=True wrong=False margin=216.000 text=' v22'
