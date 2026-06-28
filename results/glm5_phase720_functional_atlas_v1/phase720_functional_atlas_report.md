# Phase 720 Functional Atlas v1 Readiness

## Core Judgment

The uploaded theory direction is basically correct: cracking the encoding mechanism should move from isolated patch tests to a functional atlas. However, the current evidence supports a head/channel-level route atlas first, not a full neuron-level global atlas.

## Objective Results

- Nodes built: `288`
- By model: `{'deepseek7b': 96, 'glm4': 96, 'qwen3': 96}`
- By unit type: `{'attention_head': 96, 'attention_channel': 192}`
- QK/V dominant factors: `{'qk_addressing': 157, 'mixed_coupled': 131}`
- Measured function families: `['object_relation_value_short_answer']`
- Not yet measured: `['fruit_identity_reuse_difference', 'color_value_reuse_difference', 'translation_language_route']`

## Feasibility

- `head_level_global_atlas`: `ready_to_start_v1`. cross-model head candidates, route roles, and QK/V factor labels already exist for one function family
- `channel_level_bridge`: `ready_for_targeted_drilldown`. source-restricted channel units exist, but only inside selected high-value routes
- `neuron_level_global_atlas`: `not_ready_as_full_global_project`. neuron/channel identity is not yet separable from head QK addressing, V content, W_O readout, and downstream MLP effects

## Top Attention Heads

| model | unit | role | factor | qk_share | v_share | total_direct | route_gain | identity | format/prose |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | qwen3_L35_H25 | short_value_route_carrier | qk_addressing | 0.413 | 0.294 | 2.020 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L35_H1 | short_value_route_carrier | mixed_coupled | 0.296 | 0.352 | 1.811 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L35_H15 | short_value_route_carrier | mixed_coupled | 0.320 | 0.340 | 1.724 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L32_H25 | short_value_route_carrier | mixed_coupled | 0.382 | 0.309 | 1.574 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L34_H28 | short_value_route_carrier | qk_addressing | 0.337 | 0.331 | 1.310 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L34_H19 | short_value_route_carrier | mixed_coupled | 0.321 | 0.339 | 1.110 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L34_H9 | short_value_route_carrier | qk_addressing | 0.343 | 0.329 | 1.065 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L35_H2 | short_value_route_carrier | qk_addressing | 0.257 | 0.372 | 1.006 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L31_H19 | short_value_route_carrier | qk_addressing | 0.368 | 0.318 | 0.892 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L35_H26 | short_value_route_carrier | qk_addressing | 0.403 | 0.298 | 0.828 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L33_H7 | short_value_route_carrier | qk_addressing | 0.353 | 0.323 | 0.629 | 1.000 | 1.000 | 0.000 |
| qwen3 | qwen3_L34_H20 | short_value_route_carrier | mixed_coupled | 0.392 | 0.304 | 0.570 | 1.000 | 1.000 | 0.000 |

## Top Attention Heads By Model

### deepseek7b

| unit | role | factor | qk_share | v_share | total_direct | route_gain | identity | format/prose |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b_L26_H15 | prose_or_format_route_carrier | qk_addressing | 0.948 | 0.026 | 2.100 | 0.806 | 0.389 | 0.417 |
| deepseek7b_L26_H19 | prose_or_format_route_carrier | qk_addressing | 0.977 | 0.012 | 1.516 | 0.806 | 0.389 | 0.417 |
| deepseek7b_L25_H14 | prose_or_format_route_carrier | qk_addressing | 0.380 | 0.310 | 1.428 | 0.806 | 0.389 | 0.417 |
| deepseek7b_L26_H25 | prose_or_format_route_carrier | qk_addressing | 0.416 | 0.292 | 1.383 | 0.806 | 0.389 | 0.417 |
| deepseek7b_L26_H26 | prose_or_format_route_carrier | mixed_coupled | 0.378 | 0.312 | 1.214 | 0.806 | 0.389 | 0.417 |
| deepseek7b_L26_H11 | prose_or_format_route_carrier | mixed_coupled | 0.391 | 0.305 | 1.087 | 0.806 | 0.389 | 0.417 |

### glm4

| unit | role | factor | qk_share | v_share | total_direct | route_gain | identity | format/prose |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| glm4_L39_H21 | unresolved_or_weak | mixed_coupled | 0.200 | 0.400 | 0.175 | 0.000 | 0.000 | 0.000 |
| glm4_L39_H11 | unresolved_or_weak | qk_addressing | 0.329 | 0.337 | 0.154 | 0.000 | 0.000 | 0.000 |
| glm4_L39_H24 | unresolved_or_weak | qk_addressing | 0.351 | 0.324 | 0.136 | 0.000 | 0.000 | 0.000 |
| glm4_L39_H22 | unresolved_or_weak | mixed_coupled | 0.313 | 0.350 | 0.110 | 0.000 | 0.000 | 0.000 |
| glm4_L38_H15 | unresolved_or_weak | mixed_coupled | 0.393 | 0.304 | 0.103 | 0.000 | 0.000 | 0.000 |
| glm4_L36_H27 | unresolved_or_weak | mixed_coupled | 0.331 | 0.335 | 0.102 | 0.000 | 0.000 | 0.000 |

### qwen3

| unit | role | factor | qk_share | v_share | total_direct | route_gain | identity | format/prose |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_L35_H25 | short_value_route_carrier | qk_addressing | 0.413 | 0.294 | 2.020 | 1.000 | 1.000 | 0.000 |
| qwen3_L35_H1 | short_value_route_carrier | mixed_coupled | 0.296 | 0.352 | 1.811 | 1.000 | 1.000 | 0.000 |
| qwen3_L35_H15 | short_value_route_carrier | mixed_coupled | 0.320 | 0.340 | 1.724 | 1.000 | 1.000 | 0.000 |
| qwen3_L32_H25 | short_value_route_carrier | mixed_coupled | 0.382 | 0.309 | 1.574 | 1.000 | 1.000 | 0.000 |
| qwen3_L34_H28 | short_value_route_carrier | qk_addressing | 0.337 | 0.331 | 1.310 | 1.000 | 1.000 | 0.000 |
| qwen3_L34_H19 | short_value_route_carrier | mixed_coupled | 0.321 | 0.339 | 1.110 | 1.000 | 1.000 | 0.000 |

## Strict Limits

- A head is not a semantic unit. It mixes QK addressing, V content, output projection, residual trajectory, and downstream nonlinear effects.
- The present atlas covers one measured micro-family: object relation value short-answer route. It does not yet prove apple/red/translation mechanisms.
- Current models are small; architecture-specific bias and scale effects must be treated as unresolved risks.
- Neuron-level global atlas should begin only after repeated head/channel patterns are stable across multiple function families.

## Next Phase

Phase 721 should stay in the same atlas-building stage and expand function families before drilling globally to neurons: fruit/category, color, translation, and grammar protocol. For each family, require observational contribution, QK/V split, causal patch on top units, and generation or phrase-likelihood closure.
