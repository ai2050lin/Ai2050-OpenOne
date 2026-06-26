# Phase 670 Graph Atlas Counterfactual Control Set

- generated: `2026-06-26 10:20:24`
- source_atlas: `results/glm5_phase669_cross_mechanism_language_encoding_graph_atlas/phase669_graph_atlas.json`
- n_cases: `630`
- n_pairs: `462`

## Principle

This phase does not run model inference. It creates clean counterfactual controls for the Phase 669 graph atlas.

Each pair changes one intended factor while holding the others as stable as possible:

- same value / different format
- different value / same format
- same prefix / different continuation
- same format / random value
- value-only / intent-only / protocol-only factor isolation

## Case Families

| family | count |
|---|---:|
| different_value_same_format | 48 |
| factor_isolation | 54 |
| same_format_random_value | 72 |
| same_prefix_different_continuation | 24 |
| same_value_different_format | 432 |

## Pair Families

| family | count |
|---|---:|
| different_value_same_format | 48 |
| factor_isolation | 36 |
| same_prefix_different_continuation | 18 |
| same_value_different_format | 360 |

## Target Nodes

| node | case_count |
|---|---:|
| continuation_controller | 24 |
| first_token_readout_closure | 96 |
| format_continuation_state | 240 |
| multi_competitor_readout | 72 |
| protocol_execution_field | 468 |
| semantic_value_support | 570 |
| task_intent_gate | 180 |
| value_specific_token1_transition_state | 144 |

## Nodes Not Covered By Prompt-Level Controls

- `residual_boundary_integrated_state`: Requires boundary restore/remove or trajectory capture.
- `writer_topology`: Requires internal activation/component tests, not only prompt-level counterfactuals.

## Future Model Test Command Shape

```bash
python tests/gpt5/phase671_graph_atlas_counterfactual_model_audit.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase671_graph_atlas_counterfactual_model_audit.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase671_graph_atlas_counterfactual_model_audit.py --model deepseek7b --hard-exit-after-model
```

## Stop Condition

The next model phase should not start until the control set passes tokenizer validation for all three models.
