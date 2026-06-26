# Phase 686 Late Attention Value-Readout Writer Restore

- generated: `2026-06-26 12:38:01`

| model | pairs | candidate_sites | best_condition | repair_rate | patched_top1 | rank_delta | patched_pmv | patched_best_other |
|---|---:|---|---|---:|---:|---:|---:|---|
| deepseek7b | 72 | L26_attn_out, L23_attn_out, L27_layer_out, L26_layer_out | L27_layer_out|add_delta | 1.000 | 1.000 | 166.69 | -1.884 | {'prose': 72} |
| glm4 | 5 | L32_attn_out, L29_attn_out, L38_layer_out, L39_layer_out | L38_layer_out|add_delta | 1.000 | 1.000 | 1.00 | -3.750 | {'prose': 5} |
| qwen3 | 3 | L31_attn_out, L34_attn_out, L34_layer_out, L33_layer_out | L34_layer_out|add_delta | 1.000 | 1.000 | 1.00 | -3.875 | {'continuation': 2, 'prose': 1} |

## Best Conditions

### deepseek7b

| condition | repair_rate | patched_top1 | patched_rank | rank_delta | patched_pmv | best_other |
|---|---:|---:|---:|---:|---:|---|
| L27_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 166.69 | -1.884 | {'prose': 72} |
| L27_layer_out|replace | 1.000 | 1.000 | 1.00 | 166.69 | -1.886 | {'prose': 72} |
| L26_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 166.69 | -1.975 | {'prose': 72} |
| L26_layer_out|replace | 1.000 | 1.000 | 1.00 | 166.69 | -1.972 | {'prose': 72} |
| top2_layer_out|replace | 1.000 | 1.000 | 1.00 | 166.69 | -1.886 | {'prose': 72} |
| best_attn_plus_best_layer|replace | 1.000 | 1.000 | 1.00 | 166.69 | -1.886 | {'prose': 72} |
| best_attn_plus_best_layer|add_delta | 0.986 | 0.986 | 1.01 | 166.68 | -3.696 | {'prose': 72} |
| top2_layer_out|add_delta | 0.986 | 0.986 | 1.03 | 166.67 | -5.813 | {'json': 11, 'prose': 61} |
| top2_attn_out|add_delta | 0.292 | 0.292 | 5.14 | 162.56 | 0.940 | {'prose': 72} |
| top2_attn_out|replace | 0.250 | 0.250 | 5.49 | 162.21 | 0.940 | {'prose': 72} |

### glm4

| condition | repair_rate | patched_top1 | patched_rank | rank_delta | patched_pmv | best_other |
|---|---:|---:|---:|---:|---:|---|
| L38_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 | {'prose': 5} |
| L38_layer_out|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.737 | {'prose': 5} |
| L39_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 | {'prose': 5} |
| L39_layer_out|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 | {'prose': 5} |
| top2_layer_out|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 | {'prose': 5} |
| best_attn_plus_best_layer|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -3.925 | {'prose': 5} |
| best_attn_plus_best_layer|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.737 | {'prose': 5} |
| top2_attn_out|add_delta | 0.800 | 0.800 | 1.20 | 0.80 | -2.587 | {'continuation': 5} |
| top2_attn_out|replace | 0.800 | 0.800 | 1.20 | 0.80 | -2.612 | {'continuation': 5} |
| top2_layer_out|add_delta | 0.800 | 0.800 | 1.20 | 0.80 | -4.562 | {'json': 2, 'prose': 3} |

### qwen3

| condition | repair_rate | patched_top1 | patched_rank | rank_delta | patched_pmv | best_other |
|---|---:|---:|---:|---:|---:|---|
| L34_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 | {'continuation': 2, 'prose': 1} |
| L34_layer_out|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 | {'continuation': 2, 'prose': 1} |
| L33_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 | {'continuation': 2, 'prose': 1} |
| L33_layer_out|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.917 | {'continuation': 2, 'prose': 1} |
| top2_layer_out|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -6.708 | {'continuation': 2, 'prose': 1} |
| top2_layer_out|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 | {'continuation': 2, 'prose': 1} |
| best_attn_plus_best_layer|add_delta | 1.000 | 1.000 | 1.00 | 1.00 | -4.208 | {'continuation': 2, 'prose': 1} |
| best_attn_plus_best_layer|replace | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 | {'continuation': 2, 'prose': 1} |
| L31_attn_out|add_delta | 0.333 | 0.333 | 1.67 | 0.33 | -1.250 | {'continuation': 2, 'prose': 1} |
| L31_attn_out|replace | 0.333 | 0.333 | 1.67 | 0.33 | -1.333 | {'continuation': 2, 'prose': 1} |

