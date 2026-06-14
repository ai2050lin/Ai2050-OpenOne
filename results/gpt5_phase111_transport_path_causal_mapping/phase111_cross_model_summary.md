# Phase 111 Cross-model Transport Path Causal Mapping

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, time, container, clothing, furniture, plant; train/test objects per category: 12/12; templates: 4; prompts per category: 48
- monitor layer: model-specific peak; patch layers: peak-3 ... peak
- sites: object_last, answer_last; modes: remove_target, amplify_target, wrong_inject_abs, random_remove; scales: [0.25, 0.5, 1.0, 1.5]

## Cross-model Table
| model | category | wrong cat | object remove | answer remove | object answer-proj down | wrong inject | random | class |
|---|---|---|---|---|---|---|---|---|
| qwen3 | number | time | L33 object_last s0.25 T-0.00 A-0.10 | L35 answer_last s1.5 T-3.43 A+0.00 | L34 object_last s1.5 T+0.01 A-0.35 | L35 answer_last s1.5 T-3.54 A+0.00 | L33 answer_last s1.5 T-0.05 A-0.23 | answer_site_only |
| qwen3 | time | number | L32 object_last s1.5 T-0.03 A+0.15 | L35 answer_last s1.5 T-1.84 A+0.00 | L34 object_last s1.0 T+0.01 A-0.31 | L35 answer_last s1.5 T-4.18 A+0.00 | L33 answer_last s1.5 T-0.09 A-0.97 | answer_site_only |
| qwen3 | container | clothing | L32 object_last s1.5 T-0.05 A-0.16 | L33 answer_last s1.5 T-2.59 A-530.79 | L34 object_last s1.0 T-0.03 A-0.36 | L35 answer_last s1.5 T-0.76 A+0.00 | L35 answer_last s1.5 T-0.08 A+0.00 | answer_site_only |
| qwen3 | clothing | tool | L35 object_last s0.25 T+0.01 A+0.00 | L35 answer_last s1.5 T-1.43 A+0.00 | L34 object_last s1.5 T+0.05 A-0.72 | L35 answer_last s1.5 T-4.51 A+0.00 | L35 object_last s1.5 T-0.00 A+0.00 | answer_site_only |
| qwen3 | furniture | clothing | L34 object_last s0.25 T+0.01 A-0.28 | L35 answer_last s1.5 T-0.55 A+0.00 | L34 object_last s1.0 T+0.06 A-0.71 | L35 answer_last s1.5 T-3.26 A+0.00 | L32 answer_last s1.5 T-0.02 A+0.17 | mixed |
| qwen3 | plant | fruit | L32 object_last s1.5 T-0.00 A+0.10 | L35 answer_last s1.5 T-5.97 A+0.00 | L34 object_last s1.5 T+0.01 A-0.28 | L35 answer_last s1.5 T-2.52 A+0.00 | L33 object_last s1.0 T-0.00 A+0.01 | answer_site_only |
| glm4 | number | time | L18 object_last s0.25 T-0.01 A+0.00 | L18 answer_last s1.5 T-0.09 A+0.00 | L16 object_last s0.5 T+0.01 A-0.00 | L18 answer_last s1.5 T-0.10 A+0.00 | L15 answer_last s0.25 T-0.02 A-0.00 | weak |
| glm4 | time | number | L16 object_last s0.25 T-0.00 A+0.00 | L16 answer_last s1.5 T-0.06 A+0.65 | L15 object_last s0.25 T+0.01 A-0.00 | L18 answer_last s1.5 T-0.07 A+0.00 | L15 answer_last s0.5 T-0.01 A-0.00 | weak |
| glm4 | container | clothing | L16 object_last s0.25 T-0.01 A+0.00 | L18 answer_last s1.5 T-0.07 A+0.00 | L17 object_last s0.5 T+0.01 A-0.00 | L18 answer_last s1.5 T-0.07 A+0.00 | L15 answer_last s1.5 T-0.03 A+0.00 | weak |
| glm4 | clothing | tool | L17 object_last s0.25 T-0.00 A-0.00 | L18 answer_last s1.5 T-0.07 A+0.00 | L16 object_last s0.5 T+0.04 A-0.00 | L15 object_last s1.5 T-0.22 A+0.01 | L15 object_last s1.5 T-0.02 A-0.00 | weak |
| glm4 | furniture | clothing | L18 object_last s0.25 T+0.00 A+0.00 | L15 answer_last s1.5 T-0.04 A+0.75 | L15 object_last s0.25 T+0.00 A-0.00 | L15 object_last s1.5 T-0.21 A+0.01 | L18 object_last s1.5 T-0.01 A+0.00 | weak |
| glm4 | plant | fruit | L17 object_last s1.5 T-0.03 A-0.01 | L17 answer_last s0.25 T-0.01 A-0.19 | L15 object_last s1.0 T-0.02 A-0.03 | L16 object_last s1.0 T-0.10 A+0.00 | L18 object_last s1.0 T-0.01 A+0.00 | weak |
| deepseek7b | number | time | L25 object_last s1.5 T-0.07 A+0.06 | L26 answer_last s0.25 T+0.69 A-96.88 | L24 object_last s1.5 T-0.04 A-0.05 | L26 answer_last s1.5 T-3.39 A+412.31 | L25 answer_last s0.25 T-0.12 A-0.04 | mixed |
| deepseek7b | time | number | L24 object_last s0.5 T-0.02 A+0.14 | L27 answer_last s1.5 T-0.56 A+0.00 | L25 object_last s1.5 T+0.10 A-0.20 | L26 answer_last s1.5 T-1.50 A+300.44 | L26 answer_last s1.5 T-0.06 A+0.09 | mixed |
| deepseek7b | container | clothing | L26 object_last s0.25 T-0.21 A-0.06 | L27 answer_last s1.5 T-5.50 A+0.00 | L24 object_last s1.5 T+0.08 A-1.70 | L27 answer_last s1.5 T-1.42 A+0.00 | L25 answer_last s1.5 T-0.38 A-0.33 | answer_site_only |
| deepseek7b | clothing | tool | L26 object_last s1.0 T-0.23 A-0.03 | L27 answer_last s1.5 T-5.04 A+0.00 | L24 object_last s1.5 T+0.05 A-2.23 | L27 answer_last s1.5 T-1.04 A+0.00 | L26 object_last s1.0 T-0.25 A-0.03 | answer_site_only |
| deepseek7b | furniture | clothing | L24 object_last s0.25 T-0.17 A-0.46 | L27 answer_last s1.5 T-3.82 A+0.00 | L24 object_last s1.5 T+0.11 A-2.16 | L24 answer_last s1.0 T-0.86 A+276.82 | L25 object_last s0.25 T-0.16 A-0.07 | answer_site_only |
| deepseek7b | plant | fruit | L24 object_last s1.5 T-0.15 A-0.75 | L27 answer_last s1.5 T-3.20 A+0.00 | L24 object_last s1.5 T-0.15 A-0.75 | L27 answer_last s1.5 T-2.11 A+0.00 | L25 object_last s1.5 T-0.10 A-0.01 | answer_site_only |

## Objective Reading Rules
- object_path_supported means object_last removal reduced target logits, reduced answer_last transport projection, and beat random control.
- answer_site_only means answer_last removal was strong while object_last removal was weak.
- logit_without_projection_sync means logits moved but monitored answer transport projection did not move together.
- control_sensitive means random control was too close or stronger, so the condition is not reliable.

## Hard Limits
- This phase monitors one peak-layer transport projection; hidden path changes outside that projection can be missed.
- wrong-category injection uses fixed neighbor choices, not an automatically learned release graph.
- Generation audit is still not included in this script.
