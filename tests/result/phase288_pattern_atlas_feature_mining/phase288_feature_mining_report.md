# Phase288 Pattern Atlas Feature Mining

- signature_rows: 972
- component_summary_rows: 196
- causal_rows: 392
- closure_quality_rows: 36
- gap_rows: 972
- global_mlp_dominance_rate: 0.943878
- global_continue_win_rate: 1.0
- global_closure_closed_count: 0

## Model Matrix

- deepseek7b: mlp=0.984848 side_effect=0.416667 closure_reject=1.0 need_closure=222
- glm4: mlp=0.836066 side_effect=0.606557 closure_reject=1.0 need_closure=157
- qwen3: mlp=1.0 side_effect=0.507246 closure_reject=1.0 need_closure=178

## Top Gap Families

- content_knowledge: need_component=87 need_causal=88 need_closure=86
- syntax_structure: need_component=84 need_causal=85 need_closure=83
- readout_competition: need_component=83 need_causal=84 need_closure=75
- closure: need_component=86 need_causal=86 need_closure=59
- output_protocol: need_component=86 need_causal=87 need_closure=48
- language_action: need_component=83 need_causal=83 need_closure=54

## Layer Clusters

- late_mlp_strong_continue: 80
- middle_mlp_strong_continue: 67
- late_mlp_continue: 15
- middle_mlp_continue: 15
- early_attention_routed_continue: 6
- middle_attention_routed_continue: 5
- early_mlp_continue: 4
- early_mlp_strong_continue: 4
