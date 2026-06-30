# Phase 785 Positive-Negative Subspace Split (smoke)

- Status: `complete`
- Test: split answer-site route dimensions by signed readout contribution.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-output channel evidence, not final head/neuron atlas.

## Top Sufficiency Subspaces

| model | route | mode | budget | cases | dims | strict gain | delta margin | pos cover | neg cover | abs cover | top1 classes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `all` | `all` | 1 | 15360.000 | 1.000 | 10.000 | 1.000 | 1.000 | 1.000 | `{"target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `positive` | `positive_256` | 1 | 256.000 | 0.000 | 7.750 | 0.248 | 0.000 | 0.147 | `{"case_variant_target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `random` | `random_256` | 1 | 256.000 | 0.000 | 0.125 | 0.016 | 0.015 | 0.016 | `{"case_variant_target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `negative` | `negative_256` | 1 | 256.000 | 0.000 | -3.625 | 0.000 | 0.237 | 0.097 | `{"case_variant_target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `positive` | `positive_256` | 1 | 256.000 | 1.000 | 3.875 | 0.283 | 0.000 | 0.153 | `{"target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `all` | `all` | 1 | 24576.000 | 1.000 | 2.375 | 1.000 | 1.000 | 1.000 | `{"target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `random` | `random_256` | 1 | 256.000 | 0.000 | 0.062 | 0.010 | 0.012 | 0.011 | `{"case_variant_target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `negative` | `negative_256` | 1 | 256.000 | 0.000 | -2.781 | 0.000 | 0.236 | 0.108 | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `positive` | `positive_256` | 1 | 256.000 | 1.000 | 7.000 | 0.222 | 0.000 | 0.120 | `{"target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all` | `all` | 1 | 21504.000 | 0.000 | 4.750 | 1.000 | 1.000 | 1.000 | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `random` | `random_256` | 1 | 256.000 | 0.000 | 0.188 | 0.013 | 0.010 | 0.012 | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `negative` | `negative_256` | 1 | 256.000 | 0.000 | -4.750 | 0.000 | 0.198 | 0.091 | `{"case_variant_target_value": 1}` |

## Negative Patch Effects

| model | route | budget | dims | strict gain | delta margin | neg cover | top1 classes |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `negative_256` | 256.000 | 0.000 | -3.625 | 0.237 | `{"case_variant_target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `negative_256` | 256.000 | 0.000 | -2.781 | 0.236 | `{"case_variant_target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `negative_256` | 256.000 | 0.000 | -4.750 | 0.198 | `{"case_variant_target_value": 1}` |

## Random Controls

| model | route | budget | dims | strict gain | delta margin | abs cover |
|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `random_256` | 256.000 | 0.000 | 0.125 | 0.016 |
| glm4 | `lowercase_short_value:route_k6` | `random_256` | 256.000 | 0.000 | 0.062 | 0.011 |
| deepseek7b | `lowercase_short_value:route_k6` | `random_256` | 256.000 | 0.000 | 0.188 | 0.012 |

## Strict Interpretation

- Positive subspace success supports a readout-supporting subspace.
- Negative-only patch with negative margin supports an interfering subspace.
- Random/neutral controls estimate whether success comes from score ranking rather than arbitrary channel count.
- This phase still does not split attention heads or MLP activation neurons.
