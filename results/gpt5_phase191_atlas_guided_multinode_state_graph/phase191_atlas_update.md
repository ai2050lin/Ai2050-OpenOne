# Phase191 Atlas Update

- edge: candidate-specific ranking repair
- old_level: Level3 strong transition / weak Level4 channel contribution
- new_level: hold at weak Level4 candidate; do not upgrade to Level5
- next_gap: forward_dynamics_state_transition_test
- stop_condition_triggered: static single-node/channel and static multi-node z-patch routes failed to cleanly beat controls

## Validated Nodes
- qwen3: prompt_last/L34, query_category/L32, prompt_last/L32
- glm4: prompt_last/L38, prompt_last/L37, prompt_last/L39
- deepseek7b: rule_value/L26, query_relation/L19, prompt_last/L26

## Failure Types
- control_pollution
- multi_node_static_z_patch_insufficiency
- candidate_ranking_not_portable_by_static_node_set
- winner_switch_rate_too_low
