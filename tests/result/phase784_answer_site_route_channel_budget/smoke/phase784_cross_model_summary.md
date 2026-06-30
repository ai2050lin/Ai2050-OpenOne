# Phase 784 Answer-Site Route Channel Budget (smoke)

- Status: `complete`
- Test: answer-site route channel/subspace budget.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-output dimension budget, not final head/neuron atlas.

## Routes

| model | route | compare | size | components |
|---|---|---|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `attn:L35, mlp:L35, mlp:L34, mlp:L33, mlp:L32, mlp:L26` |
| glm4 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `mlp:L38, mlp:L39, mlp:L34, mlp:L27, mlp:L36, mlp:L31` |
| deepseek7b | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `mlp:L27, mlp:L26, mlp:L24, attn:L19, mlp:L22, mlp:L21` |

## Budget Intervention Summary

| model | route | budget | intervention | cases | dims | frac | score cover | strict gain | delta margin | gain/full | margin/full | top1 classes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `32` | `patch_baseline_from_donor_channel_budget` | 1 | 32.000 | 0.002 | 0.081 | 0.000 | 3.125 | 0.000 | 0.312 | `{"case_variant_target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 1 | 15360.000 | 1.000 | 1.000 | 1.000 | 10.000 | 1.000 | 1.000 | `{"target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `32` | `replace_donor_channel_budget_with_baseline` | 1 | 32.000 | 0.002 | 0.081 | 0.000 | -3.125 | null | null | `{"target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 1 | 15360.000 | 1.000 | 1.000 | 0.000 | -9.750 | null | null | `{"case_variant_target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `32` | `patch_baseline_from_donor_channel_budget` | 1 | 32.000 | 0.001 | 0.103 | 0.000 | 1.500 | 0.000 | 0.632 | `{"case_variant_target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 1 | 24576.000 | 1.000 | 1.000 | 1.000 | 2.375 | 1.000 | 1.000 | `{"target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `32` | `replace_donor_channel_budget_with_baseline` | 1 | 32.000 | 0.001 | 0.103 | 0.000 | -1.562 | null | null | `{"case_variant_target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 1 | 24576.000 | 1.000 | 1.000 | 0.000 | -2.438 | null | null | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `32` | `patch_baseline_from_donor_channel_budget` | 1 | 32.000 | 0.001 | 0.070 | 0.000 | 3.000 | null | 0.632 | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 1 | 21504.000 | 1.000 | 1.000 | 0.000 | 4.750 | null | 1.000 | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `32` | `replace_donor_channel_budget_with_baseline` | 1 | 32.000 | 0.001 | 0.070 | 0.000 | -2.750 | null | null | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 1 | 21504.000 | 1.000 | 1.000 | 0.000 | -4.688 | null | null | `{"case_variant_target_value": 1}` |

## Low-Budget Successes

| model | route | budget | dims | frac | score cover | strict gain | delta margin |
|---|---|---|---:|---:|---:|---:|---:|

## Strict Interpretation

- `all` should approximate the Phase 782 full answer-site route patch.
- Small-budget success means the readout-side route has sparse/channel-like support under the current ranking rule.
- Small-budget failure means the route is distributed or the ranking rule is incomplete.
- This does not yet identify attention heads or biological neurons.
