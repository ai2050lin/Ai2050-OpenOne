# Phase 224 multilayer activation propagation

spec_count: 3
filter_rows: 56
reproducible_success_rows: 17
reproducible_drift_rows: 20
propagation_rows: 1296
channel_score_rows: 1152
total_top_token_changed: 141
total_target_rank_improved: 219

| spec | group | condition | layer | rows | mean shift | mean cos | top changed | rank improved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K64 | 33 | 12 | 25.059296 | 0.352971 | 3 | 5 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K64 | 33 | 12 | -25.924016 | -0.353473 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K64 | 31 | 12 | 22.752242 | 0.389843 | 3 | 5 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K64 | 29 | 12 | 19.033755 | 0.393239 | 3 | 5 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K64 | 31 | 12 | -21.881710 | -0.368788 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K64 | 33 | 12 | 18.017534 | 0.269322 | 3 | 7 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K64 | 29 | 12 | -18.451653 | -0.384015 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K64 | 31 | 12 | 15.167239 | 0.271522 | 3 | 7 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K16 | 33 | 12 | -17.563278 | -0.284845 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K16 | 33 | 12 | 16.549056 | 0.293035 | 0 | 7 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K16 | 31 | 12 | -15.531940 | -0.318302 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K16 | 31 | 12 | 14.831549 | 0.321349 | 0 | 7 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K16 | 33 | 12 | 11.806707 | 0.220924 | 3 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K64 | 29 | 12 | 11.499135 | 0.256345 | 3 | 7 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K16 | 31 | 12 | 9.215590 | 0.206414 | 3 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K16 | 29 | 12 | 12.154827 | 0.316636 | 0 | 7 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K16 | 29 | 12 | -11.700416 | -0.298970 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K16 | 29 | 12 | 6.407815 | 0.187621 | 3 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K4 | 33 | 12 | -9.263155 | -0.219479 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K4 | 31 | 12 | -8.864332 | -0.258593 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K4 | 33 | 12 | 7.972813 | 0.199628 | 0 | 6 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K4 | 31 | 12 | 7.731731 | 0.233489 | 0 | 6 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K64 | 33 | 12 | -7.506846 | -0.098517 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K64 | 29 | 12 | -7.467288 | -0.170336 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K64 | 31 | 12 | -7.021868 | -0.121891 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K16 | 31 | 12 | -6.498766 | -0.129874 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_drift_L29_K4 | 29 | 12 | -6.447455 | -0.234314 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K16 | 33 | 12 | -6.424109 | -0.086230 | 0 | 4 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K4 | 29 | 12 | 6.399459 | 0.237517 | 0 | 6 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K4 | 33 | 12 | 6.343352 | 0.163778 | 0 | 3 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K64 | 32 | 12 | -3.967513 | -0.315842 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K64 | 32 | 12 | -3.795964 | -0.377506 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K64 | 32 | 12 | -1.615933 | -0.120001 | 4 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K64 | 31 | 12 | -3.601269 | -0.383143 | 2 | 0 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K16 | 29 | 12 | -5.592684 | -0.146365 | 0 | 4 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K64 | 31 | 12 | -3.575477 | -0.311458 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K64 | 30 | 12 | -3.513865 | -0.410339 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K64 | 30 | 12 | -3.488126 | -0.339176 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K64 | 30 | 12 | -1.461441 | -0.133163 | 4 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K16 | 32 | 12 | -3.289355 | -0.296111 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K16 | 32 | 12 | -1.258906 | -0.106647 | 4 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K16 | 30 | 12 | -1.220485 | -0.121156 | 4 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K64 | 31 | 12 | -1.205184 | -0.097827 | 4 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K4 | 32 | 12 | -1.999225 | -0.277361 | 3 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K16 | 31 | 12 | -2.970105 | -0.293029 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K16 | 30 | 12 | -2.960615 | -0.317765 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K4 | 30 | 12 | -0.945797 | -0.115356 | 4 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K4 | 32 | 12 | -0.934607 | -0.093596 | 4 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K16 | 31 | 12 | -0.913349 | -0.085001 | 4 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K4 | 30 | 12 | -1.902819 | -0.300311 | 3 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K4 | 31 | 12 | -1.893681 | -0.278668 | 3 | 0 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K4 | 31 | 12 | -4.883330 | -0.080597 | 0 | 2 |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K4 | 31 | 12 | 4.808185 | 0.142738 | 0 | 3 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K4 | 31 | 12 | -0.686030 | -0.073353 | 4 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K64 | 30 | 12 | -1.488368 | -0.221083 | 3 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K64 | 32 | 12 | -1.455333 | -0.179179 | 3 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K4 | 32 | 12 | -2.340266 | -0.269443 | 2 | 0 |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K64 | 31 | 12 | -1.242414 | -0.172062 | 3 | 1 |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K4 | 30 | 12 | -2.229645 | -0.289591 | 2 | 0 |
| qwen3_explain_l29_to_l31_l33_propagation | success_repro | mlpchan_pos_zero_L29_K4 | 33 | 12 | -4.189954 | -0.037336 | 0 | 2 |
