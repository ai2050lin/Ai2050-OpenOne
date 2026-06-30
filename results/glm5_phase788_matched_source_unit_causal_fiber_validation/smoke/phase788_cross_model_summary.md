# Phase 788 Matched Source Unit Causal Fiber Validation (smoke)

- Status: `complete`
- Test: patch donor source units into baseline, with matched and random controls.
- Attention source units are o_proj input head slices.
- MLP source units are down_proj input activation channels.
- Matched controls are selected inside the same layer/component by source magnitude and activation-delta norm.

## Cross-Model Intervention Summary

| model | source | subspace | selection | intervention | cases | strict gain | strict loss | delta margin | source signed | top1 classes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head_set` | `negative` | `matched3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.250 | -1.136 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `negative` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.125 | -0.363 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `negative` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 1.625 | -4.693 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `negative` | `matched3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.250 | -1.136 | `{"target_value": 1}` |
| qwen3 | `attention_head_set` | `negative` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -1.125 | -4.693 | `{"target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `matched3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.250 | 1.766 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.250 | 0.694 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 1.625 | 6.728 | `{"case_variant_target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `matched3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.125 | 1.766 | `{"target_value": 1}` |
| qwen3 | `attention_head_set` | `positive` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -1.125 | 6.728 | `{"target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `matched8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.750 | -2.107 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.034 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.500 | -4.937 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `matched8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.625 | -2.107 | `{"target_value": 1}` |
| qwen3 | `mlp_channel_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.125 | -4.937 | `{"target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `matched8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.375 | 2.472 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.016 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 1.875 | 6.810 | `{"case_variant_target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `matched8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.375 | 2.472 | `{"target_value": 1}` |
| qwen3 | `mlp_channel_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -1.750 | 6.810 | `{"target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `matched8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.703 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -0.003 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | -1.375 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `matched8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.062 | -0.703 | `{"target_value": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.000 | -1.375 | `{"target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `matched8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.741 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 1.491 | `{"case_variant_target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `matched8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.062 | 0.741 | `{"target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | 0.000 | 1.491 | `{"target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `matched3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.312 | -0.057 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | -0.062 | 0.011 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 2.250 | -0.316 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `matched3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.438 | -0.057 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -3.188 | -0.316 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `matched3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.312 | 0.067 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `random3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.062 | 0.042 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `top3` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 2.250 | 0.482 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `matched3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.438 | 0.067 | `{"case_variant_target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `top3` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -3.188 | 0.482 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `matched8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.250 | -6.499 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.002 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.125 | -11.788 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `matched8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.250 | -6.499 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.125 | -11.788 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `matched8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.625 | 5.852 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.000 | 0.027 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 1 | 0.000 | 0.000 | 0.688 | 8.944 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `matched8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.500 | 5.852 | `{"case_variant_target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 1 | 0.000 | 0.000 | -0.562 | 8.944 | `{"case_variant_target_value": 1}` |

## Top-Minus-Matched Specificity

| model | source | subspace | intervention | set | top gain | matched gain | top delta | matched delta | gap | matched distance |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `mlp_channel_set` | `positive` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 1.875 | 0.375 | 1.500 | 1155.057 |
| qwen3 | `attention_head_set` | `negative` | `patch_baseline_from_donor_source_units` | 3 | 0.000 | 0.000 | 1.625 | 0.250 | 1.375 | 8.804 |
| qwen3 | `attention_head_set` | `positive` | `patch_baseline_from_donor_source_units` | 3 | 0.000 | 0.000 | 1.625 | 0.250 | 1.375 | 7.698 |
| qwen3 | `mlp_channel_set` | `positive` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -1.750 | -0.375 | -1.375 | 1155.057 |
| qwen3 | `attention_head_set` | `positive` | `replace_donor_source_units_with_baseline` | 3 | 0.000 | 0.000 | -1.125 | 0.125 | -1.250 | 7.698 |
| qwen3 | `attention_head_set` | `negative` | `replace_donor_source_units_with_baseline` | 3 | 0.000 | 0.000 | -1.125 | -0.250 | -0.875 | 8.804 |
| qwen3 | `mlp_channel_set` | `negative` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -0.125 | -0.625 | 0.500 | 497.469 |
| qwen3 | `mlp_channel_set` | `negative` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 0.500 | 0.750 | -0.250 | 497.469 |
| glm4 | `mlp_channel_set` | `negative` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | 0.000 | 0.062 | -0.062 | 720.269 |
| glm4 | `mlp_channel_set` | `positive` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | 0.000 | 0.062 | -0.062 | 959.730 |
| glm4 | `mlp_channel_set` | `negative` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 720.269 |
| glm4 | `mlp_channel_set` | `positive` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 959.730 |
| deepseek7b | `attention_head_set` | `negative` | `replace_donor_source_units_with_baseline` | 3 | 0.000 | 0.000 | -3.188 | -0.438 | -2.750 | 16.426 |
| deepseek7b | `attention_head_set` | `positive` | `replace_donor_source_units_with_baseline` | 3 | 0.000 | 0.000 | -3.188 | -0.438 | -2.750 | 23.874 |
| deepseek7b | `attention_head_set` | `negative` | `patch_baseline_from_donor_source_units` | 3 | 0.000 | 0.000 | 2.250 | 0.312 | 1.938 | 16.426 |
| deepseek7b | `attention_head_set` | `positive` | `patch_baseline_from_donor_source_units` | 3 | 0.000 | 0.000 | 2.250 | 0.312 | 1.938 | 23.874 |
| deepseek7b | `mlp_channel_set` | `positive` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 0.688 | 0.625 | 0.062 | 656.813 |
| deepseek7b | `mlp_channel_set` | `positive` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -0.562 | -0.500 | -0.062 | 656.813 |
| deepseek7b | `mlp_channel_set` | `negative` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 0.125 | 0.250 | -0.125 | 960.148 |
| deepseek7b | `mlp_channel_set` | `negative` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -0.125 | -0.250 | 0.125 | 960.148 |

## Top Sufficiency Components

| model | route | component | source | subspace | selection | cases | strict gain | delta margin | source signed |
|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `top8` | 1 | 0.000 | 1.875 | 6.810 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `top3` | 1 | 0.000 | 1.625 | -4.693 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `top3` | 1 | 0.000 | 1.625 | 6.728 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `matched8` | 1 | 0.000 | 0.750 | -2.107 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `top8` | 1 | 0.000 | 0.500 | -4.937 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `matched8` | 1 | 0.000 | 0.375 | 2.472 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `matched3` | 1 | 0.000 | 0.250 | -1.136 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `matched3` | 1 | 0.000 | 0.250 | 1.766 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `random3` | 1 | 0.000 | 0.250 | 0.694 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `random3` | 1 | 0.000 | 0.125 | -0.363 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `random8` | 1 | 0.000 | 0.000 | -0.034 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `random8` | 1 | 0.000 | 0.000 | -0.016 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `matched8` | 1 | 0.000 | 0.000 | -0.703 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `random8` | 1 | 0.000 | 0.000 | -0.003 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `top8` | 1 | 0.000 | 0.000 | -1.375 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `matched8` | 1 | 0.000 | 0.000 | 0.741 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `random8` | 1 | 0.000 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `top8` | 1 | 0.000 | 0.000 | 1.491 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `top3` | 1 | 0.000 | 2.250 | -0.316 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `top3` | 1 | 0.000 | 2.250 | 0.482 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `top8` | 1 | 0.000 | 0.688 | 8.944 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `matched8` | 1 | 0.000 | 0.625 | 5.852 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `matched3` | 1 | 0.000 | 0.312 | -0.057 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `matched3` | 1 | 0.000 | 0.312 | 0.067 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `matched8` | 1 | 0.000 | 0.250 | -6.499 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `top8` | 1 | 0.000 | 0.125 | -11.788 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `random3` | 1 | 0.000 | 0.062 | 0.042 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `random8` | 1 | 0.000 | 0.000 | 0.002 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `random8` | 1 | 0.000 | 0.000 | 0.027 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `random3` | 1 | 0.000 | -0.062 | 0.011 |

## Interpretation Boundary

- This validates answer-site source-unit specificity against matched controls, not full Q/K/V path or cross-position semantic fibers.
- Matched controls are still approximate: they do not fully match token source, attention pattern, or upstream causal history.
- MLP channel sets are activation channels, closer to neuron-level than residual channels but still not biological neurons.
