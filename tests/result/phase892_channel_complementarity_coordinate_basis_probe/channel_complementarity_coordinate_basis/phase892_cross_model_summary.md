# Phase 892 channel complementarity and coordinate basis probe

## Overall

- models: qwen3, glm4, deepseek7b
- selected_sources: 29
- output_rows: 483
- closure_from_open: 238
- single_axis_closure: 73
- multi_axis_closure: 165
- positive_complementarity_rows: 47
- closure_without_single_axis_closure: 0
- mean_multi_complementarity_over_best: 0.159
- mean_interaction_residual_vs_additive: 0.126

## Candidate groups

| model | candidate | label | closures | single | multi | comp rows | no-single closures | mean comp | best subset | modes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| deepseek7b | L27C15369:subset0:zero | multi_axis_target_lift_complementarity | 104 | 29 | 75 | 31 | 0 | 0.698 | {"subset_key": "L26C8587+L27C15369", "mean_target_lift": 3.1494565217391304, "n": 23} | {"flip": 38, "half": 28, "zero": 38} |
| deepseek7b | L26C8587:subset1:zero | multi_axis_target_lift_complementarity | 48 | 15 | 33 | 16 | 0 | 0.818 | {"subset_key": "L26C8587+L27C15369", "mean_target_lift": 3.0902777777777777, "n": 9} | {"flip": 18, "half": 12, "zero": 18} |
| deepseek7b | L27C16651:flip | single_axis_dominant_target_lift | 76 | 19 | 57 | 0 | 0 | -0.010 | {"subset_key": "L27C16651", "mean_target_lift": 2.6414473684210527, "n": 19} | {"flip": 32, "half": 12, "zero": 32} |
| qwen3 | L31C2257:flip | single_axis_dominant_target_lift | 10 | 10 | 0 | 0 | 0 | 0.000 | {"subset_key": "L31C2257", "mean_target_lift": 0.5, "n": 10} | {"flip": 4, "half": 2, "zero": 4} |
| glm4 | L31C6437:flip | negative_no_channel_complementarity | 0 | 0 | 0 | 0 | 0 | 0.000 | {} | {} |
| glm4 | L31C6437:zero | negative_no_channel_complementarity | 0 | 0 | 0 | 0 | 0 | 0.000 | {} | {} |
