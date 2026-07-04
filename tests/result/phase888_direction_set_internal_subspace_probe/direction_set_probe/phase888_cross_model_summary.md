# Phase 888 direction-set intervention and internal subspace probe

## Overall

- models: qwen3, glm4, deepseek7b
- source_rows: 59
- output_rows: 236
- mode_closure_from_open: 88
- restored_reopens_boundary: 3
- base_mask_boundary_closed: 140
- unique_base_mask_closed_cases: 35
- unique_multi_mode_closure_cases: 35

## Candidate groups

| model | candidate | label | cases | closures | restore reopen | mask closed | multi-mode | modes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | L27C16651:flip | direction_set_boundary_signal_no_restore | 21 | 42 | 0 | 17 | 17 | {"flip": 17, "half": 8, "zero": 17} |
| deepseek7b | L27C15369:subset0:zero | direction_set_boundary_signal_no_restore | 15 | 30 | 0 | 11 | 11 | {"flip": 11, "half": 8, "zero": 11} |
| qwen3 | L31C2257:flip | direction_set_internal_subspace_signal | 8 | 10 | 3 | 4 | 4 | {"flip": 4, "half": 2, "zero": 4} |
| deepseek7b | L26C8587:subset1:zero | direction_set_boundary_signal_no_restore | 7 | 6 | 0 | 3 | 3 | {"flip": 3, "zero": 3} |
| glm4 | L31C6437:flip | negative_no_direction_set_signal | 4 | 0 | 0 | 0 | 0 | {} |
| glm4 | L31C6437:zero | negative_no_direction_set_signal | 4 | 0 | 0 | 0 | 0 | {} |
