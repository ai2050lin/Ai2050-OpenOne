# Phase 891 target-lift source pathway and projection subspace audit

## Overall

- models: qwen3, glm4, deepseek7b
- selected_sources: 35
- output_rows: 1200
- none_closure_from_open: 151
- multi_axis_none_closure: 93
- mlp_zero_closure_lost: 64
- attn_zero_closure_lost: 33
- mean_mlp_zero_lift_retention: 0.826
- mean_attn_zero_lift_retention: -0.724

## Candidate groups

| model | candidate | label | none closure | multi-axis closure | mlp lost | attn lost | mlp retention | attn retention | gear sets |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | L27C15369:subset0:zero | mixed_component_target_lift | 69 | 46 | 19 | 14 | -0.557 | -1.210 | {"candidate_axis": 23, "model_U": 23, "same_layer_U": 23} |
| deepseek7b | L27C16651:flip | mixed_component_target_lift | 57 | 38 | 36 | 11 | -0.346 | -0.771 | {"candidate_axis": 19, "model_U": 19, "same_layer_U": 19} |
| deepseek7b | L26C8587:subset1:zero | mixed_component_target_lift | 15 | 9 | 5 | 4 | 5.516 | -0.682 | {"candidate_axis": 6, "model_U": 9} |
| qwen3 | L31C2257:flip | mixed_component_target_lift | 10 | 0 | 4 | 4 | 2.305 | -1.197 | {"candidate_axis": 10} |
| glm4 | L31C6437:flip | negative_no_target_lift_pathway | 0 | 0 | 0 | 0 | 0.000 | 0.000 | {} |
| glm4 | L31C6437:zero | negative_no_target_lift_pathway | 0 | 0 | 0 | 0 | 0.000 | 0.000 | {} |
