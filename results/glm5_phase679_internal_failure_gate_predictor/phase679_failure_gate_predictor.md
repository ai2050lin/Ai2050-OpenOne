# Phase 679 Internal Failure-Gate Predictor

- generated: `2026-06-26 11:27:09`

| model | gate | kind | repair | pred_rate | fail_capture | false_pos | selective_top1 | fail_repair | damage |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | top1_category_not_expected | near_readout | final_cancel_gap_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | top1_category_word_or_newline_or_other | near_readout | final_cancel_gap_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | expected_rank_gt_1 | near_readout_upper_bound | final_cancel_gap_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | final_gap_gt_-0.25 | readout_gap | final_cancel_gap_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | final_gap_gt_0 | readout_gap | final_cancel_gap_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | top1_category_not_expected | near_readout | final_remove_comp_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | top1_category_word_or_newline_or_other | near_readout | final_remove_comp_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | expected_rank_gt_1 | near_readout_upper_bound | final_remove_comp_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | final_gap_gt_-0.25 | readout_gap | final_remove_comp_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | final_gap_gt_0 | readout_gap | final_remove_comp_a2p0 | 0.069 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | top1_category_not_expected | near_readout | final_cancel_gap_a1p25 | 0.069 | 1.000 | 0.000 | 0.986 | 0.800 | 0.000 |
| qwen3 | top1_category_word_or_newline_or_other | near_readout | final_cancel_gap_a1p25 | 0.069 | 1.000 | 0.000 | 0.986 | 0.800 | 0.000 |
| glm4 | top1_category_not_expected | near_readout | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | top1_category_word_or_newline_or_other | near_readout | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | expected_rank_gt_1 | near_readout_upper_bound | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | expected_rank_gt_10 | near_readout | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | final_gap_lt_-5.875 | readout_gap | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | final_gap_gt_-0.25 | readout_gap | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | final_gap_gt_0 | readout_gap | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | pre_gap_lt_-71.25 | pre_final_gap | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | pre_gap_gt_4.25 | pre_final_gap | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | post_unit_gap_lt_-7.418 | readout_geometry | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | post_unit_gap_gt_-1.342 | readout_geometry | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | post_unit_gap_gt_0 | readout_geometry | final_cancel_gap_a1p25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| deepseek7b | top1_category_not_expected | near_readout | final_cancel_gap_a2p0 | 0.875 | 1.000 | 0.000 | 0.958 | 0.952 | 0.000 |
| deepseek7b | top1_category_word_or_newline_or_other | near_readout | final_cancel_gap_a2p0 | 0.875 | 1.000 | 0.000 | 0.958 | 0.952 | 0.000 |
| deepseek7b | expected_rank_gt_1 | near_readout_upper_bound | final_cancel_gap_a2p0 | 0.875 | 1.000 | 0.000 | 0.958 | 0.952 | 0.000 |
| deepseek7b | final_gap_gt_0 | readout_gap | final_cancel_gap_a2p0 | 0.875 | 1.000 | 0.000 | 0.958 | 0.952 | 0.000 |
| deepseek7b | final_gap_gt_0.375 | readout_gap | final_cancel_gap_a2p0 | 0.875 | 1.000 | 0.000 | 0.958 | 0.952 | 0.000 |
| deepseek7b | top1_category_not_expected | near_readout | final_remove_comp_a2p0 | 0.875 | 1.000 | 0.000 | 0.944 | 0.937 | 0.000 |
| deepseek7b | top1_category_word_or_newline_or_other | near_readout | final_remove_comp_a2p0 | 0.875 | 1.000 | 0.000 | 0.944 | 0.937 | 0.000 |
| deepseek7b | expected_rank_gt_1 | near_readout_upper_bound | final_remove_comp_a2p0 | 0.875 | 1.000 | 0.000 | 0.944 | 0.937 | 0.000 |
| deepseek7b | final_gap_gt_0 | readout_gap | final_remove_comp_a2p0 | 0.875 | 1.000 | 0.000 | 0.944 | 0.937 | 0.000 |
| deepseek7b | final_gap_gt_0.375 | readout_gap | final_remove_comp_a2p0 | 0.875 | 1.000 | 0.000 | 0.944 | 0.937 | 0.000 |
| deepseek7b | expected_rank_gt_10 | near_readout | final_cancel_gap_a2p0 | 0.806 | 0.921 | 0.000 | 0.889 | 0.873 | 0.000 |
| deepseek7b | expected_rank_gt_10 | near_readout | final_remove_comp_a2p0 | 0.806 | 0.921 | 0.000 | 0.875 | 0.857 | 0.000 |
