# Phase 787 Source Unit Causal Validation (smoke)

- Status: `complete`
- Test: patch donor source units into baseline, with random controls.
- Attention source units are o_proj input head slices.
- MLP source units are down_proj input activation channels.

## Cross-Model Intervention Summary

| model | source | subspace | selection | intervention | cases | strict gain | strict loss | delta margin | source signed | top1 classes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head_set` | `negative` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.125 | -0.363 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `negative` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 1.625 | -4.693 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `negative` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -1.125 | -4.693 | `{"target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.250 | 0.694 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 1.625 | 6.728 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -1.125 | 6.728 | `{"target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.034 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.500 | -4.937 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.125 | -4.937 | `{"target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.016 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 1.875 | 6.810 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -1.750 | 6.810 | `{"target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.003 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -1.375 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.000 | -1.375 | `{"target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 1.491 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.000 | 1.491 | `{"target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | -0.062 | 0.011 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 2.250 | -0.316 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -3.188 | -0.316 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.062 | 0.042 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 2.250 | 0.482 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -3.188 | 0.482 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.002 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.125 | -11.788 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.125 | -11.788 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.027 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.688 | 8.944 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.562 | 8.944 | `{"case_variant_target_value": 1}` |

## Top Sufficiency Components

| model | route | component | source | subspace | selection | cases | strict gain | delta margin | source signed |
|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `top` | 1 | 0.000 | 1.875 | 6.810 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `top` | 1 | 0.000 | 1.625 | -4.693 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `top` | 1 | 0.000 | 1.625 | 6.728 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `top` | 1 | 0.000 | 0.500 | -4.937 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `random` | 1 | 0.000 | 0.250 | 0.694 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `random` | 1 | 0.000 | 0.125 | -0.363 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `random` | 1 | 0.000 | 0.000 | -0.034 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `random` | 1 | 0.000 | 0.000 | -0.016 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `random` | 1 | 0.000 | 0.000 | -0.003 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `top` | 1 | 0.000 | 0.000 | -1.375 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `random` | 1 | 0.000 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `top` | 1 | 0.000 | 0.000 | 1.491 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `top` | 1 | 0.000 | 2.250 | -0.316 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `top` | 1 | 0.000 | 2.250 | 0.482 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `top` | 1 | 0.000 | 0.688 | 8.944 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `top` | 1 | 0.000 | 0.125 | -11.788 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `random` | 1 | 0.000 | 0.062 | 0.042 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `random` | 1 | 0.000 | 0.000 | 0.002 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `random` | 1 | 0.000 | 0.000 | 0.027 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `random` | 1 | 0.000 | -0.062 | 0.011 |

## Interpretation Boundary

- This validates answer-site source-unit effects, not full Q/K/V path or cross-position semantic fibers.
- Random controls are matched by unit count but not by activation norm.
- MLP channel sets are activation channels, closer to neuron-level than residual channels but still not biological neurons.
