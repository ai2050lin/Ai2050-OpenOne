# Pattern Family Atlas Data Contract

schema_version: 1.0.0
phase: Phase235
families: 9
modes: 72
test_cases: 36

## Required Files

- schema: `schema.json`
- client_index: `client_index.json`
- families: `families.jsonl`
- modes: `modes.jsonl`
- test_cases: `test_cases.jsonl`
- runs: `runs.jsonl`
- observations: `observations.jsonl`
- metrics: `metrics.jsonl`
- graph_nodes: `graph_nodes.jsonl`
- graph_edges: `graph_edges.jsonl`
- progress: `progress.json`
- summary: `summary.md`

## Progress

- pattern_family_atlas: 0.34
- model_internal_closure: 0.46
- general_language_mechanism_confidence: 0.43

## Known Evidence Edges

| edge | model | type | status | confidence |
| --- | --- | --- | --- | ---: |
| edge_glm4_no_answer_anchor_for_continuation | glm4 | prompt_anchor_to_regime_switch | hook_supported | 0.72 |
| edge_glm4_explain_instruction_because | glm4 | instruction_to_competitor_pressure | hook_supported | 0.60 |
| edge_qwen3_no_answer_anchor_because_period | qwen3 | prompt_anchor_to_competitor_pressure | hook_supported | 0.56 |
| edge_qwen3_period_second_takeover | qwen3 | suppression_to_takeover | readout_mapped | 0.50 |
| edge_deepseek7b_be_continuation_candidate | deepseek7b | weak_product_coupling | source_candidate | 0.32 |

## Phase236 Behavior Benchmark Update

- models: qwen3, glm4, deepseek7b
- case_rows: 132
- observation_rows: 1056
- mean_behavior_score: 0.6462
- pattern_match_rate: 0.6288
- drift_types: {'none': 83, 'wrong_or_missing_target': 30, 'over_generation': 19}
