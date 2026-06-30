# Phase 783 Token-Position Surface Route Patch (smoke)

- Status: `complete`
- Test: patch/replace Phase 782 route components over token-position scopes.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-level position fiber test, not head/channel/neuron-level proof.

## Routes And Scopes

| model | route | compare | size | scopes | components |
|---|---|---|---:|---|---|
| qwen3 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `answer_site, format_cue, semantic_pair` | `attn:L35, mlp:L35, mlp:L34, mlp:L33, mlp:L32, mlp:L26` |
| glm4 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `answer_site, format_cue, semantic_pair` | `mlp:L38, mlp:L39, mlp:L34, mlp:L27, mlp:L36, mlp:L31` |
| deepseek7b | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `answer_site, format_cue, semantic_pair` | `mlp:L27, mlp:L26, mlp:L24, attn:L19, mlp:L22, mlp:L21` |

## Top Sufficiency Fibers

| model | route | scope | size | strict gain | delta margin | gain vs answer | margin vs answer | score | alignment |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `answer_site` | 6 | 1.000 | 10.000 | 0.000 | 0.000 | 10.000 | `{"same_count": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `semantic_pair` | 6 | 0.000 | 0.000 | -1.000 | -10.000 | 0.000 | `{"same_count": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `answer_site` | 6 | 1.000 | 2.375 | 0.000 | 0.000 | 2.375 | `{"same_count": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `semantic_pair` | 6 | 0.000 | -0.062 | -1.000 | -2.438 | -0.000 | `{"same_count": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `answer_site` | 6 | 0.000 | 4.750 | 0.000 | 0.000 | 0.000 | `{"same_count": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `semantic_pair` | 6 | 0.000 | -0.062 | 0.000 | -4.812 | -0.000 | `{"same_count": 1}` |

## Top Answer-Site Advantages

| model | route | scope | strict gain vs answer | margin gain vs answer | strict gain | delta margin |
|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `semantic_pair` | -1.000 | -10.000 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `semantic_pair` | -1.000 | -2.438 | 0.000 | -0.062 |
| deepseek7b | `lowercase_short_value:route_k6` | `semantic_pair` | 0.000 | -4.812 | 0.000 | -0.062 |

## Top Necessity Fibers

| model | route | scope | intervention | size | strict loss | semantic loss | delta margin | score |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 1.000 | 0.000 | -9.750 | 9.750 |
| qwen3 | `lowercase_short_value:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.000 | -0.000 |
| glm4 | `lowercase_short_value:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 1.000 | 0.000 | -2.438 | 2.438 |
| glm4 | `lowercase_short_value:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.000 | -0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -4.688 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.000 | -0.000 |

## Strict Interpretation

- If non-answer scopes beat answer_site, the route should be treated as a position-component fiber.
- If answer_site remains best, Phase 782 likely captured a readout-side route.
- Mean-broadcast rows are useful boundary probes, but same-count rows are cleaner causal evidence.
