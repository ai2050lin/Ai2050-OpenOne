# Phase 730 Downstream Propagation Node Cancellation

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: downstream propagation node cancellation.

| model | condition | mean_delta | recovery | hit_rate | changed | hit_drop | rank_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | category_cluster|cancel_top_layer_out | 0.0000 | 1.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_cluster|cancel_top_mlp_out | 0.0032 | 0.829 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_cluster|upstream_only | 0.0184 | 0.000 | 0.955 | 0.000 | 0.000 | -0.05 |
| qwen3 | category_full_head|cancel_top_layer_out | 0.0000 | 1.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_full_head|cancel_top_mlp_out | -0.0209 | -0.318 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_full_head|upstream_only | -0.0159 | 0.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_cluster|cancel_top_layer_out | 0.0000 | 1.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_cluster|cancel_top_mlp_out | 0.0047 | -6.311 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_cluster|upstream_only | 0.0006 | 0.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_full_head|cancel_top_layer_out | 0.0000 | 1.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_full_head|cancel_top_mlp_out | 0.0050 | -2.051 | 0.955 | 0.000 | 0.000 | 0.05 |
| glm4 | category_full_head|upstream_only | 0.0016 | 0.000 | 0.955 | 0.000 | 0.000 | 0.00 |
| deepseek7b | category_cluster|cancel_top_layer_out | 0.0000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.00 |
| deepseek7b | category_cluster|cancel_top_mlp_out | -0.3281 | -0.256 | 0.500 | 0.000 | 0.000 | 1.23 |
| deepseek7b | category_cluster|upstream_only | -0.2613 | 0.000 | 0.500 | 0.000 | 0.000 | 0.68 |
| deepseek7b | category_full_head|cancel_top_layer_out | 0.0000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.00 |
| deepseek7b | category_full_head|cancel_top_mlp_out | -0.9339 | 0.117 | 0.455 | 0.273 | 0.091 | 4.41 |
| deepseek7b | category_full_head|upstream_only | -1.0575 | 0.000 | 0.500 | 0.409 | 0.091 | 4.50 |

## Strict Interpretation

- Cancellation toward baseline is a mediation test, not a complete circuit proof.
- Positive recovery means the downstream site carries part of the upstream perturbation effect.
- No recovery means the site is visible in propagation but not sufficient as a bottleneck.
