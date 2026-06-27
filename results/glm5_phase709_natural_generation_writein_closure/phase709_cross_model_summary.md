# Phase 709 Natural Generation and Write-In Closure Audit

- generated: `2026-06-27 08:39:38`

| model | pairs | donor_pairs | best_restore | n | target_value | donor_value | prose_target | prose_donor | continuation | other |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | 72 | 80 | unrelated|restore|all_positive_source_channels | 72 | 0.542 | 0.097 | 0.264 | 0.000 | 0.000 | 0.097 |
| glm4 | 5 | 5 | unrelated|restore|all_positive_source_channels | 5 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.800 |
| qwen3 | 3 | 3 | unrelated|restore|all_positive_source_channels | 3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## deepseek7b

| condition | n | target_value | donor_value | prose_target | prose_donor | continuation | other | mean_direct | mean_combo_delta | mean_output_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unrelated|restore|all_positive_source_channels | 72 | 0.542 | 0.097 | 0.264 | 0.000 | 0.000 | 0.097 | 0.015483 | -0.012966 | -0.000221 |
| unrelated|restore|source_top_channel_512 | 72 | 0.389 | 0.000 | 0.417 | 0.000 | 0.000 | 0.194 | 0.060397 | -0.058346 | -0.001006 |
| same_value|restore|all_positive_source_channels | 8 | 0.375 | 0.000 | 0.625 | 0.000 | 0.000 | 0.000 | 0.015483 | -0.012966 | -0.000221 |
| same_value|restore|source_top_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.125 | 0.060397 | -0.058346 | -0.001006 |
| same_value|restore|source_random_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.125 | 0.007547 | -0.036688 | -0.000516 |
| unrelated|restore|source_random_channel_512 | 72 | 0.000 | 0.000 | 0.806 | 0.000 | 0.000 | 0.194 | 0.007547 | -0.036688 | -0.000516 |
| unrelated|degradation|source_top_channel_512 | 72 | 0.347 | 0.000 | 0.278 | 0.000 | 0.000 | 0.375 | 0.060397 | -0.058346 | -0.001006 |
| unrelated|degradation|all_positive_source_channels | 72 | 0.042 | 0.000 | 0.639 | 0.000 | 0.000 | 0.319 | 0.015483 | -0.012966 | -0.000221 |
| unrelated|degradation|source_random_channel_512 | 72 | 0.736 | 0.000 | 0.097 | 0.000 | 0.000 | 0.167 | 0.007547 | -0.036688 | -0.000516 |
| same_value|degradation|all_positive_source_channels | 8 | 0.125 | 0.000 | 0.750 | 0.000 | 0.000 | 0.125 | 0.015483 | -0.012966 | -0.000221 |
| same_value|degradation|source_top_channel_512 | 8 | 0.750 | 0.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.060397 | -0.058346 | -0.001006 |
| same_value|degradation|source_random_channel_512 | 8 | 0.875 | 0.000 | 0.125 | 0.000 | 0.000 | 0.000 | 0.007547 | -0.036688 | -0.000516 |

## glm4

| condition | n | target_value | donor_value | prose_target | prose_donor | continuation | other | mean_direct | mean_combo_delta | mean_output_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unrelated|restore|all_positive_source_channels | 5 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.800 | 0.001234 | 0.011820 | 0.000041 |
| unrelated|restore|source_random_channel_512 | 5 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.800 | 0.000210 | -0.023708 | -0.000020 |
| unrelated|restore|source_top_channel_512 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.003889 | 0.039344 | 0.000133 |
| unrelated|degradation|all_positive_source_channels | 5 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001234 | 0.011820 | 0.000041 |
| unrelated|degradation|source_top_channel_512 | 5 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003889 | 0.039344 | 0.000133 |
| unrelated|degradation|source_random_channel_512 | 5 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000210 | -0.023708 | -0.000020 |

## qwen3

| condition | n | target_value | donor_value | prose_target | prose_donor | continuation | other | mean_direct | mean_combo_delta | mean_output_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unrelated|restore|all_positive_source_channels | 3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.009746 | 0.001090 | -0.000751 |
| unrelated|restore|source_top_channel_512 | 3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.032166 | -0.002962 | -0.002274 |
| unrelated|restore|source_random_channel_512 | 3 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.667 | 0.002117 | 0.018242 | -0.000032 |
| unrelated|degradation|source_top_channel_512 | 3 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 | 0.032166 | -0.002962 | -0.002274 |
| unrelated|degradation|all_positive_source_channels | 3 | 0.333 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | 0.009746 | 0.001090 | -0.000751 |
| unrelated|degradation|source_random_channel_512 | 3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002117 | 0.018242 | -0.000032 |
