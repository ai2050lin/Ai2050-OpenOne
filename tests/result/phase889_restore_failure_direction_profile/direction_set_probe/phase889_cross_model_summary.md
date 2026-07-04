# Phase 889 restore failure and direction profile

## Overall

- source_rows: 236
- mode_closure_from_open: 88
- restored_reopens_boundary: 3
- no_restore_closure: 85

## Candidate groups

| model | candidate | label | closures | restore | no restore | mean class delta | mean cut delta | modes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | L31C2257:flip | internal_cut_token_coupling | 10 | 3 | 7 | 0.500 | -0.067 | {"flip": 4, "half": 2, "zero": 4} |
| deepseek7b | L27C16651:flip | distributed_target_lift_direction_set | 42 | 0 | 42 | 2.360 | -0.086 | {"flip": 17, "half": 8, "zero": 17} |
| deepseek7b | L27C15369:subset0:zero | distributed_target_lift_direction_set | 30 | 0 | 30 | 1.948 | 0.006 | {"flip": 11, "half": 8, "zero": 11} |
| deepseek7b | L26C8587:subset1:zero | distributed_target_lift_direction_set | 6 | 0 | 6 | 0.490 | 0.125 | {"flip": 3, "zero": 3} |
| glm4 | L31C6437:flip | negative_no_direction_profile | 0 | 0 | 0 | 0.000 | 0.000 | {} |
| glm4 | L31C6437:zero | negative_no_direction_profile | 0 | 0 | 0 | 0.000 | 0.000 | {} |
