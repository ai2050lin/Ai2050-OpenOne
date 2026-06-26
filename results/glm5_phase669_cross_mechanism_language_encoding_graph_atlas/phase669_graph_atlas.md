# Phase 669 Cross-Mechanism Language Encoding Graph Atlas

- generated: `2026-06-26 10:04:12`
- source_phase_range: `[626, 668]`
- available_phase_count: `40`
- available_phases: `[626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 645, 647, 648, 649, 650, 651, 652, 653, 654, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668]`

## Nodes

| id | label | system | status | phases | sufficiency | necessity |
|---|---|---|---|---|---|---|
| semantic_value_support | Semantic Value Support | knowledge_network | partial_confirmed | 626,628,651,653,654 | Restore and protocol patches can raise correct-prefix rank and sometimes exact generation. | Necessary path not fully isolated; support alone is insufficient. |
| task_intent_gate | Task Intent Gate | reasoning_route | partial_confirmed | 651,652,653 | Value-to-task restore can pull non-value prompts toward value answers. | Task-to-value suppression shows intent can block short-answer support, but necessity is position dependent. |
| protocol_execution_field | Protocol Execution Field | grammar_format_protocol | strong_partial | 638,639,640,641,642,643,645,647,648,649,650 | Protocol trajectory restores can flip newline/explanation vs short-value generation in several models. | Necessary intervals exist, but side effects and model-specific boundaries remain. |
| first_token_readout_closure | First Token Readout Closure | readout_competition | confirmed_but_insufficient | 631,632,633,635,636,659,660,661,662 | Readout/last-writer combinations can make correct_prefix top1. | Not sufficient for exact answer sequence. |
| multi_competitor_readout | Multi-Competitor Readout | readout_competition | confirmed | 636,659,662,663,664 | Multi-competitor correction identifies hidden blockers missed by pairwise tests. | Not a standalone generative mechanism. |
| format_continuation_state | Format Continuation State | grammar_format_protocol | confirmed_partial | 665,666 | Correct and mismatch restores can both repair some early boundaries, showing general continuation state. | Zero removal can be destructive but not semantically specific. |
| value_specific_token1_transition_state | Value-Specific Token1 Transition State | knowledge_network | strong_model_specific | 665,666,667,668 | Correct_restore beats mismatch_restore on key boundaries, especially DS7B and qwen3 L23/L22 boundary. | Zero_remove often destroys token1, but this also affects scale/format/position. |
| writer_topology | Writer Topology | mechanism_graph | model_specific_partial | 667,668 | qwen3 head10+head11 ensemble shows strong semantic subchannel; DS7B requires full boundary state. | Single-head closure rejected as a universal explanation. |
| residual_boundary_integrated_state | Residual Boundary Integrated State | mechanism_graph | strong_in_ds7b_partial_elsewhere | 666,667,668 | DS7B full L21 layer_out / L22 layer_input restores token1 strongly and specifically. | Small head/component ensembles fail to close DS7B. |
| continuation_controller | Continuation Controller | generation_route | partial_confirmed | 664,665 | If token1 is forced correct, token2 is usually top1 in selected failures. | Later continuation beyond token2 is not yet fully audited. |

## Edges

| from | to | relation | confidence | phases | note |
|---|---|---|---|---|---|
| semantic_value_support | task_intent_gate | permission_edge | medium | 651,652,653 | Intent can permit or suppress the value support route. |
| task_intent_gate | protocol_execution_field | protocol_edge | medium | 651,653,650 | Intent and protocol jointly choose short answer vs explanation/full sentence behavior. |
| protocol_execution_field | first_token_readout_closure | support_edge | high | 643,651,661 | Protocol trajectory can make correct prefix accessible to first-token readout. |
| first_token_readout_closure | multi_competitor_readout | competition_edge | high | 636,659,662 | Correct prefix must beat multiple policy competitors. |
| multi_competitor_readout | format_continuation_state | transition_edge | medium | 664,665,666 | Top1 prefix does not guarantee continuation route; format continuation state is separate. |
| format_continuation_state | value_specific_token1_transition_state | transition_edge | high | 666 | Phase 666 separates general continuation from value-specific token1 transition. |
| value_specific_token1_transition_state | writer_topology | writer_edge | medium | 667,668 | Writer topology is model-specific; qwen3 small head ensemble, DS7B residual boundary. |
| writer_topology | residual_boundary_integrated_state | integration_edge | medium | 667,668 | Full residual boundaries can dominate small writer ensembles. |
| value_specific_token1_transition_state | continuation_controller | continuation_edge | medium | 665,666 | Once token1 is correct, token2 tends to stabilize in current failures. |

## Model-Specific Token1 Transition Evidence

### qwen3

| ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |
|---|---|---:|---:|---:|---:|---:|
| L22_heads10_11 | head_set | 2 | 0.500 | 0.000 | 4.688 | 1.375 |
| full_L22_layer_out | component_set | 2 | 0.500 | 0.000 | 3.562 | 4.938 |
| full_L23_layer_input | component_set | 2 | 0.500 | 0.000 | 3.562 | 4.938 |
| L22_heads10_11 | head_set | 3 | 0.667 | 0.333 | 3.000 | 0.792 |
| full_L22_layer_out | component_set | 3 | 0.667 | 0.333 | 2.250 | 5.354 |
| full_L23_layer_input | component_set | 3 | 0.667 | 0.333 | 2.250 | 5.354 |

### glm4

| ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |
|---|---|---:|---:|---:|---:|---:|
| L22_heads7_13 | head_set | 3 | 1.000 | 0.333 | 0.604 | 0.292 |
| full_L22_attn_out | component_set | 3 | 1.000 | 0.333 | 0.500 | 0.000 |
| full_L22_layer_input | component_set | 3 | 1.000 | 0.667 | 0.417 | 3.729 |
| L21_layer_out_L22_attn_mlp | component_set | 1 | 1.000 | 1.000 | 0.000 | 9.695 |
| L21_layer_out_L22_attn_mlp | component_set | 3 | 1.000 | 1.000 | 0.000 | 11.422 |
| L22_heads7_13 | head_set | 1 | 1.000 | 1.000 | 0.000 | 0.000 |

### deepseek7b

| ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |
|---|---|---:|---:|---:|---:|---:|
| full_L21_layer_out | component_set | 3 | 1.000 | 0.000 | 3.312 | 8.729 |
| full_L22_layer_input | component_set | 3 | 1.000 | 0.000 | 3.312 | 8.729 |
| full_L21_layer_out | component_set | 3 | 1.000 | 0.000 | 3.250 | 8.604 |
| full_L22_layer_input | component_set | 3 | 1.000 | 0.000 | 3.250 | 8.604 |
| L21_attn_mlp | component_set | 3 | 0.333 | 0.000 | 0.542 | 0.271 |
| L21_attn_mlp | component_set | 3 | 0.333 | 0.000 | 0.479 | 0.312 |

## Global Findings

- The token1 transition gate local phase is complete enough to stop single-head chasing.
- The current atlas has stable functional nodes but model-specific implementation topology.
- The largest open gap is not another local patch, but clean graph-scale controls and purer semantic/format counterfactuals.

## Hard Limits

- Several nodes are based on ORV short-value tasks and require format/generalization tests.
- Mismatch restore is useful but not a pure semantic intervention.
- Zero removal proves necessity only in a broad state-distribution sense.
- Natural writer paths are still partly unresolved because head-slice tests do not decompose Q/K/V and attention pattern.

## Next Phase

- phase: `670`
- title: `Graph Atlas Counterfactual Control Set`
- goal: Build clean same-value/different-format and different-value/same-format counterfactuals for the atlas nodes before running new model tests.
- reason: The atlas shows the main bottleneck is control purity, not absence of local candidates.
