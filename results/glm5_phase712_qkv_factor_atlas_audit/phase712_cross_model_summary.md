# Phase 712 QK-V Factor Atlas Audit

- generated: `2026-06-27 12:21:52`

| model | cases | condition | n | dominant | abs_qk | abs_v | abs_interaction | sum_total | sum_qk | sum_v | sum_interaction |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | 72 | source_top_channel | 512 | qk_addressing | 0.505 | 0.247 | 0.248 | 34.536267 | 34.375971 | 13.263107 | -13.102811 |
| deepseek7b | 72 | all_positive_source_channels | 2647 | qk_addressing | 0.449 | 0.275 | 0.275 | 48.043054 | 47.821111 | 15.571635 | -15.349692 |
| deepseek7b | 72 | source_random_channel | 512 | qk_addressing | 0.432 | 0.284 | 0.285 | 3.879545 | 3.872640 | 0.040858 | -0.033954 |
| glm4 | 5 | source_top_channel | 512 | qk_addressing | 0.291 | 0.356 | 0.353 | 3.135055 | 3.119839 | 2.118777 | -2.103561 |
| glm4 | 5 | all_positive_source_channels | 2130 | qk_addressing | 0.248 | 0.378 | 0.375 | 4.112758 | 4.085830 | 2.816152 | -2.789223 |
| glm4 | 5 | source_random_channel | 512 | qk_addressing | 0.273 | 0.365 | 0.362 | 0.189427 | 0.187510 | 0.087482 | -0.085565 |
| qwen3 | 3 | source_top_channel | 512 | mixed_coupled | 0.310 | 0.345 | 0.344 | 33.323272 | 32.783493 | 24.837520 | -24.297741 |
| qwen3 | 3 | all_positive_source_channels | 2151 | qk_addressing | 0.287 | 0.357 | 0.355 | 41.876746 | 40.800191 | 29.916179 | -28.839624 |
| qwen3 | 3 | source_random_channel | 512 | qk_addressing | 0.305 | 0.348 | 0.347 | 0.962729 | 0.935668 | -0.408560 | 0.435621 |

## deepseek7b top heads

| layer | head | dominant | abs_qk | abs_v | abs_interaction | sum_total |
|---:|---:|---|---:|---:|---:|---:|
| 26 | 15 | qk_addressing | 0.948 | 0.026 | 0.026 | 2.099944 |
| 26 | 19 | qk_addressing | 0.977 | 0.012 | 0.011 | 1.516378 |
| 25 | 14 | qk_addressing | 0.380 | 0.310 | 0.310 | 1.428270 |
| 26 | 25 | qk_addressing | 0.416 | 0.292 | 0.292 | 1.382792 |
| 26 | 26 | mixed_coupled | 0.378 | 0.312 | 0.310 | 1.214273 |
| 26 | 11 | mixed_coupled | 0.391 | 0.305 | 0.305 | 1.086769 |
| 26 | 24 | mixed_coupled | 0.323 | 0.339 | 0.339 | 1.028882 |
| 23 | 11 | qk_addressing | 0.844 | 0.078 | 0.078 | 0.935820 |
| 27 | 2 | qk_addressing | 0.900 | 0.051 | 0.049 | 0.909941 |
| 23 | 19 | qk_addressing | 0.813 | 0.084 | 0.103 | 0.818594 |
| 27 | 17 | qk_addressing | 0.830 | 0.085 | 0.085 | 0.814272 |
| 26 | 23 | mixed_coupled | 0.365 | 0.318 | 0.317 | 0.642494 |

## glm4 top heads

| layer | head | dominant | abs_qk | abs_v | abs_interaction | sum_total |
|---:|---:|---|---:|---:|---:|---:|
| 39 | 21 | mixed_coupled | 0.200 | 0.400 | 0.400 | 0.175419 |
| 39 | 11 | qk_addressing | 0.329 | 0.337 | 0.334 | 0.153843 |
| 39 | 24 | qk_addressing | 0.351 | 0.324 | 0.324 | 0.136120 |
| 39 | 22 | mixed_coupled | 0.313 | 0.350 | 0.337 | 0.109887 |
| 38 | 15 | mixed_coupled | 0.393 | 0.304 | 0.303 | 0.103318 |
| 36 | 27 | mixed_coupled | 0.331 | 0.335 | 0.335 | 0.102284 |
| 38 | 5 | qk_addressing | 0.405 | 0.298 | 0.297 | 0.096510 |
| 39 | 23 | mixed_coupled | 0.229 | 0.385 | 0.385 | 0.092919 |
| 38 | 11 | mixed_coupled | 0.373 | 0.314 | 0.313 | 0.090513 |
| 39 | 9 | qk_addressing | 0.353 | 0.324 | 0.323 | 0.072313 |
| 37 | 11 | qk_addressing | 0.339 | 0.333 | 0.327 | 0.063773 |
| 35 | 10 | qk_addressing | 0.301 | 0.350 | 0.349 | 0.061394 |

## qwen3 top heads

| layer | head | dominant | abs_qk | abs_v | abs_interaction | sum_total |
|---:|---:|---|---:|---:|---:|---:|
| 35 | 25 | qk_addressing | 0.413 | 0.294 | 0.293 | 2.019804 |
| 35 | 1 | mixed_coupled | 0.296 | 0.352 | 0.352 | 1.810951 |
| 35 | 15 | mixed_coupled | 0.320 | 0.340 | 0.340 | 1.723543 |
| 32 | 25 | mixed_coupled | 0.382 | 0.309 | 0.309 | 1.574426 |
| 34 | 28 | qk_addressing | 0.337 | 0.331 | 0.331 | 1.310166 |
| 34 | 19 | mixed_coupled | 0.321 | 0.339 | 0.339 | 1.109701 |
| 34 | 9 | qk_addressing | 0.343 | 0.329 | 0.329 | 1.065018 |
| 35 | 2 | qk_addressing | 0.257 | 0.372 | 0.371 | 1.005866 |
| 31 | 19 | qk_addressing | 0.368 | 0.318 | 0.314 | 0.891512 |
| 35 | 26 | qk_addressing | 0.403 | 0.298 | 0.298 | 0.828040 |
| 33 | 7 | qk_addressing | 0.353 | 0.323 | 0.324 | 0.629366 |
| 34 | 20 | mixed_coupled | 0.392 | 0.304 | 0.304 | 0.569728 |
