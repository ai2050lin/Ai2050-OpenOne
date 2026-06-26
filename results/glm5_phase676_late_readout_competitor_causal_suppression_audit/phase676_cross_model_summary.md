# Phase 676 Late Readout Competitor Causal Suppression

- generated: `2026-06-26 11:14:39`

| model | condition | top1_rate | mean_rank | mean_gap | gap_delta | switch_to_expected | damage_success |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | baseline | 0.125 | 469.79 | 6.301 | 0.000 | 0.000 | 0.000 |
| deepseek7b | final_output_remove_comp_a1 | 0.083 | 8.81 | -0.003 | -6.304 | 0.000 | 0.042 |
| deepseek7b | final_output_remove_random_a1 | 0.125 | 466.06 | 6.289 | -0.012 | 0.000 | 0.000 |
| deepseek7b | final_output_cancel_gap_a1 | 0.472 | 2.90 | -2.455 | -8.756 | 0.472 | 0.125 |
| deepseek7b | final_input_remove_comp_a1 | 0.000 | 52810.07 | 16.405 | 10.104 | 0.000 | 0.125 |
| deepseek7b | attn_last_remove_comp_a1 | 0.000 | 18337.64 | 12.855 | 6.554 | 0.000 | 0.125 |
| deepseek7b | attn_prev_remove_comp_a1 | 0.000 | 782.47 | 8.018 | 1.717 | 0.000 | 0.125 |
| deepseek7b | attn_last_zero_a1 | 0.000 | 10186.92 | 6.921 | 0.620 | 0.000 | 0.125 |
| glm4 | baseline | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 | 0.000 |
| glm4 | final_output_remove_comp_a1 | 0.083 | 1.04 | -0.001 | 3.328 | 0.000 | 0.917 |
| glm4 | final_output_remove_random_a1 | 1.000 | 1.00 | -3.332 | -0.003 | 0.000 | 0.000 |
| glm4 | final_output_cancel_gap_a1 | 0.292 | 1.62 | 0.010 | 3.339 | 0.000 | 0.708 |
| glm4 | final_input_remove_comp_a1 | 0.083 | 20377.78 | 19.914 | 23.243 | 0.000 | 0.917 |
| glm4 | attn_last_remove_comp_a1 | 1.000 | 1.00 | -3.026 | 0.303 | 0.000 | 0.000 |
| glm4 | attn_prev_remove_comp_a1 | 0.542 | 2.28 | -0.431 | 2.898 | 0.000 | 0.458 |
| glm4 | attn_last_zero_a1 | 1.000 | 1.00 | -3.457 | -0.128 | 0.000 | 0.000 |
| qwen3 | baseline | 0.931 | 1.49 | -5.341 | 0.000 | 0.000 | 0.000 |
| qwen3 | final_output_remove_comp_a1 | 0.069 | 1.03 | -0.005 | 5.336 | 0.000 | 0.861 |
| qwen3 | final_output_remove_random_a1 | 0.931 | 1.44 | -5.366 | -0.025 | 0.000 | 0.000 |
| qwen3 | final_output_cancel_gap_a1 | 0.056 | 2.61 | 3.243 | 8.584 | 0.056 | 0.931 |
| qwen3 | final_input_remove_comp_a1 | 0.000 | 9.21 | 8.963 | 14.304 | 0.000 | 0.931 |
| qwen3 | attn_last_remove_comp_a1 | 0.931 | 1.51 | -5.429 | -0.088 | 0.000 | 0.000 |
| qwen3 | attn_prev_remove_comp_a1 | 0.903 | 1.47 | -3.564 | 1.777 | 0.000 | 0.028 |
| qwen3 | attn_last_zero_a1 | 0.931 | 1.68 | -4.156 | 1.185 | 0.000 | 0.000 |
