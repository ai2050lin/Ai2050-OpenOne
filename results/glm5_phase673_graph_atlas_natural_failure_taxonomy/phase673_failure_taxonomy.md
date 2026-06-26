# Phase 673 Graph Atlas Natural Failure Taxonomy

- generated: `2026-06-26 10:39:14`

## Model Failure Classes

### qwen3

- success_rate: `0.568`
- dominant_failure: `value_binding_failure`

| class | count |
|---|---:|
| success | 358 |
| value_binding_failure | 166 |
| other_generation_failure | 83 |
| readout_other_failure | 22 |
| format_surface_failure | 1 |

### glm4

- success_rate: `0.543`
- dominant_failure: `value_binding_failure`

| class | count |
|---|---:|
| success | 342 |
| value_binding_failure | 160 |
| other_generation_failure | 66 |
| readout_competitor_failure | 30 |
| readout_other_failure | 22 |
| protocol_route_failure | 7 |
| format_surface_failure | 2 |
| continuation_transition_failure | 1 |

### deepseek7b

- success_rate: `0.175`
- dominant_failure: `readout_competitor_failure`

| class | count |
|---|---:|
| readout_competitor_failure | 281 |
| value_binding_failure | 155 |
| success | 110 |
| other_generation_failure | 42 |
| readout_other_failure | 36 |
| format_surface_failure | 6 |

## Internal Entry Points

| priority | model | target | failure_class | next_internal_test |
|---:|---|---|---|---|
| 1 | deepseek7b | same_format_random_value | readout_competitor_failure | trace short-value first-token readout and compare word/explanation competitors at final residual. |
| 2 | deepseek7b | same_prefix_different_continuation | readout_competitor_failure / continuation_transition_failure | first fix or localize readout/protocol entry before token1 transition patching. |
| 3 | deepseek7b | list format | protocol_route_failure | compare list marker '-' against explanation word competitors at protocol field layers. |
| 4 | qwen3 | same_value_different_format | protocol_route_failure / format_surface_failure | protocol surface formation after first expected token, especially explanation/list/json formatting. |
| 5 | glm4 | different_value_same_format | readout_competitor_failure | space/newline readout source under synthetic in-context value binding. |

## Interpretation

- Phase 672 becomes useful only after failures are separated by class.
- DS7B should not immediately enter token1 writer patching; its natural failures often happen before the value route is entered.
- qwen3 and GLM4 are better candidates for protocol/format continuation studies because first-token readout is mostly closed.
