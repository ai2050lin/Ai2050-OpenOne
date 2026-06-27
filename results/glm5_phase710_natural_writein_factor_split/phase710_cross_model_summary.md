# Phase 710 Natural Write-In Factor Split Audit

- generated: `2026-06-27 08:49:26`

| model | condition | n | target_value | donor_value | prose_target | other |
|---|---|---:|---:|---:|---:|---:|
| deepseek7b | post_layer_output|unrelated|restore|source_top_channel_512 | 72 | 0.472 | 0.000 | 0.375 | 0.153 |
| deepseek7b | post_o_output|unrelated|restore|source_top_channel_512 | 72 | 0.361 | 0.000 | 0.458 | 0.181 |
| deepseek7b | pre_o_input|unrelated|restore|source_top_channel_512 | 72 | 0.347 | 0.000 | 0.472 | 0.181 |
| deepseek7b | post_o_output|unrelated|restore|source_random_channel_512 | 72 | 0.028 | 0.000 | 0.750 | 0.222 |
| deepseek7b | pre_o_input|unrelated|restore|source_random_channel_512 | 72 | 0.014 | 0.000 | 0.764 | 0.222 |
| deepseek7b | post_layer_output|unrelated|restore|source_random_channel_512 | 72 | 0.014 | 0.000 | 0.736 | 0.250 |
| deepseek7b | pre_o_input|same_value|restore|source_top_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.125 |
| deepseek7b | post_o_output|same_value|restore|source_top_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.125 |
| deepseek7b | post_layer_output|same_value|restore|source_top_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.125 |
| deepseek7b | pre_o_input|same_value|restore|source_random_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.125 |
| deepseek7b | post_o_output|same_value|restore|source_random_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.125 |
| deepseek7b | post_layer_output|same_value|restore|source_random_channel_512 | 8 | 0.000 | 0.000 | 0.875 | 0.125 |
| glm4 | post_layer_output|unrelated|restore|source_top_channel_512 | 5 | 0.200 | 0.000 | 0.000 | 0.800 |
| glm4 | pre_o_input|unrelated|restore|source_top_channel_512 | 5 | 0.000 | 0.000 | 0.000 | 1.000 |
| glm4 | post_o_output|unrelated|restore|source_top_channel_512 | 5 | 0.000 | 0.000 | 0.000 | 1.000 |
| glm4 | pre_o_input|unrelated|restore|source_random_channel_512 | 5 | 0.000 | 0.000 | 0.000 | 1.000 |
| glm4 | post_o_output|unrelated|restore|source_random_channel_512 | 5 | 0.000 | 0.000 | 0.000 | 1.000 |
| glm4 | post_layer_output|unrelated|restore|source_random_channel_512 | 5 | 0.000 | 0.000 | 0.000 | 1.000 |
| qwen3 | pre_o_input|unrelated|restore|source_top_channel_512 | 3 | 1.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | post_o_output|unrelated|restore|source_top_channel_512 | 3 | 1.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | post_layer_output|unrelated|restore|source_top_channel_512 | 3 | 1.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | pre_o_input|unrelated|restore|source_random_channel_512 | 3 | 0.333 | 0.000 | 0.000 | 0.667 |
| qwen3 | post_o_output|unrelated|restore|source_random_channel_512 | 3 | 0.333 | 0.000 | 0.000 | 0.667 |
| qwen3 | post_layer_output|unrelated|restore|source_random_channel_512 | 3 | 0.000 | 0.000 | 0.000 | 1.000 |
