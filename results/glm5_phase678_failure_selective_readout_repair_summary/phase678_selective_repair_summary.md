# Phase 678 Failure-Selective Readout Repair Summary

- generated: `2026-06-26 11:21:03`

| model | condition | selective_top1 | mean_rank | mean_gap | failure_repair | success_damage |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | baseline | 0.931 | 1.49 | -5.341 | 0.000 | 0.000 |
| qwen3 | final_cancel_gap_a1p0 | 0.986 | 1.01 | -5.674 | 0.800 | 0.000 |
| qwen3 | final_cancel_gap_a1p25 | 0.986 | 1.00 | -5.755 | 0.800 | 0.000 |
| qwen3 | final_cancel_gap_a1p5 | 0.986 | 1.00 | -5.839 | 0.800 | 0.000 |
| qwen3 | final_cancel_gap_a2p0 | 1.000 | 1.00 | -6.007 | 1.000 | 0.000 |
| qwen3 | final_remove_comp_a1p5 | 0.972 | 1.00 | -5.665 | 0.600 | 0.000 |
| qwen3 | final_remove_comp_a2p0 | 1.000 | 1.00 | -5.774 | 1.000 | 0.000 |
| glm4 | baseline | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a1p0 | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a1p25 | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a1p5 | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| glm4 | final_cancel_gap_a2p0 | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| glm4 | final_remove_comp_a1p5 | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| glm4 | final_remove_comp_a2p0 | 1.000 | 1.00 | -3.329 | 0.000 | 0.000 |
| deepseek7b | baseline | 0.125 | 469.79 | 6.301 | 0.000 | 0.000 |
| deepseek7b | final_cancel_gap_a1p0 | 0.597 | 2.79 | -2.660 | 0.540 | 0.000 |
| deepseek7b | final_cancel_gap_a1p25 | 0.833 | 1.61 | -4.913 | 0.810 | 0.000 |
| deepseek7b | final_cancel_gap_a1p5 | 0.847 | 1.26 | -7.155 | 0.825 | 0.000 |
| deepseek7b | final_cancel_gap_a2p0 | 0.958 | 1.04 | -11.634 | 0.952 | 0.000 |
| deepseek7b | final_remove_comp_a1p5 | 0.681 | 1.81 | -3.372 | 0.635 | 0.000 |
| deepseek7b | final_remove_comp_a2p0 | 0.944 | 1.15 | -6.595 | 0.937 | 0.000 |
