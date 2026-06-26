# Phase 677 Readout Intervention Strength Scan

- generated: `2026-06-26 11:18:42`

| model | condition | top1_rate | mean_rank | mean_gap | gap_delta | failure_switch | success_damage |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | baseline | 0.125 | 469.79 | 6.301 | 0.000 | 0.000 | 0.000 |
| deepseek7b | final_cancel_gap_a0p25 | 0.125 | 95.03 | 4.118 | -2.183 | 0.000 | 0.000 |
| deepseek7b | final_cancel_gap_a0p5 | 0.125 | 26.42 | 1.924 | -4.377 | 0.000 | 0.000 |
| deepseek7b | final_cancel_gap_a0p75 | 0.125 | 8.40 | -0.276 | -6.577 | 0.095 | 0.667 |
| deepseek7b | final_cancel_gap_a1p0 | 0.472 | 2.90 | -2.455 | -8.756 | 0.540 | 1.000 |
| deepseek7b | final_cancel_gap_a1p25 | 0.708 | 1.79 | -4.659 | -10.960 | 0.810 | 1.000 |
| deepseek7b | final_cancel_gap_a1p5 | 0.722 | 1.57 | -6.845 | -13.146 | 0.825 | 1.000 |
| deepseek7b | final_cancel_gap_a2p0 | 0.833 | 1.61 | -11.224 | -17.525 | 0.952 | 1.000 |
| deepseek7b | final_cancel_gap_a3p0 | 0.833 | 3.10 | -19.984 | -26.285 | 0.952 | 1.000 |
| deepseek7b | final_remove_comp_a0p5 | 0.125 | 49.78 | 3.159 | -3.142 | 0.000 | 0.000 |
| deepseek7b | final_remove_comp_a1p0 | 0.083 | 8.81 | -0.003 | -6.304 | 0.000 | 0.333 |
| deepseek7b | final_remove_comp_a1p5 | 0.569 | 1.92 | -3.155 | -9.456 | 0.635 | 0.889 |
| deepseek7b | final_remove_comp_a2p0 | 0.819 | 1.43 | -6.300 | -12.601 | 0.937 | 1.000 |
| deepseek7b | final_remove_random_a1p0 | 0.125 | 471.81 | 6.291 | -0.010 | 0.000 | 0.000 |
| glm4 | baseline | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a0p25 | 1.000 | 1.00 | -2.516 | 0.812 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a0p5 | 1.000 | 1.00 | -1.671 | 1.658 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a0p75 | 1.000 | 1.00 | -0.832 | 2.497 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a1p0 | 0.292 | 1.62 | 0.010 | 3.339 | 0.000 | 0.708 |
| glm4 | final_cancel_gap_a1p25 | 0.000 | 1.99 | 0.838 | 4.167 | 0.000 | 1.000 |
| glm4 | final_cancel_gap_a1p5 | 0.000 | 2.00 | 1.665 | 4.994 | 0.000 | 1.000 |
| glm4 | final_cancel_gap_a2p0 | 0.000 | 2.38 | 3.336 | 6.665 | 0.000 | 1.000 |
| glm4 | final_cancel_gap_a3p0 | 0.000 | 6.93 | 6.661 | 9.990 | 0.000 | 1.000 |
| glm4 | final_remove_comp_a0p5 | 1.000 | 1.00 | -1.672 | 1.657 | 0.000 | 0.000 |
| glm4 | final_remove_comp_a1p0 | 0.083 | 1.04 | -0.001 | 3.328 | 0.000 | 0.917 |
| glm4 | final_remove_comp_a1p5 | 0.000 | 2.03 | 1.661 | 4.990 | 0.000 | 1.000 |
| glm4 | final_remove_comp_a2p0 | 0.000 | 2.60 | 3.325 | 6.654 | 0.000 | 1.000 |
| glm4 | final_remove_random_a1p0 | 1.000 | 1.00 | -3.321 | 0.008 | 0.000 | 0.000 |
| qwen3 | baseline | 0.931 | 1.49 | -5.341 | 0.000 | 0.000 | 0.000 |
| qwen3 | final_cancel_gap_a0p25 | 0.931 | 1.18 | -3.203 | 2.138 | 0.000 | 0.000 |
| qwen3 | final_cancel_gap_a0p5 | 0.931 | 1.12 | -1.052 | 4.289 | 0.000 | 0.000 |
| qwen3 | final_cancel_gap_a0p75 | 0.042 | 1.93 | 1.095 | 6.437 | 0.600 | 1.000 |
| qwen3 | final_cancel_gap_a1p0 | 0.056 | 2.61 | 3.243 | 8.584 | 0.800 | 1.000 |
| qwen3 | final_cancel_gap_a1p25 | 0.056 | 3.53 | 5.403 | 10.744 | 0.800 | 1.000 |
| qwen3 | final_cancel_gap_a1p5 | 0.056 | 4.97 | 7.546 | 12.887 | 0.800 | 1.000 |
| qwen3 | final_cancel_gap_a2p0 | 0.069 | 11.76 | 11.865 | 17.207 | 1.000 | 1.000 |
| qwen3 | final_cancel_gap_a3p0 | 0.069 | 127.69 | 20.472 | 25.813 | 1.000 | 1.000 |
| qwen3 | final_remove_comp_a0p5 | 0.931 | 1.14 | -2.689 | 2.652 | 0.000 | 0.000 |
| qwen3 | final_remove_comp_a1p0 | 0.069 | 1.03 | -0.005 | 5.336 | 0.000 | 0.925 |
| qwen3 | final_remove_comp_a1p5 | 0.042 | 2.26 | 2.682 | 8.023 | 0.600 | 1.000 |
| qwen3 | final_remove_comp_a2p0 | 0.069 | 3.47 | 5.337 | 10.678 | 1.000 | 1.000 |
| qwen3 | final_remove_random_a1p0 | 0.931 | 1.51 | -5.357 | -0.016 | 0.000 | 0.000 |
