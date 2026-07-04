# Phase 890 distributed restore and projection-style subspace intervention

## Overall

- models: qwen3, glm4, deepseek7b
- source_rows: 59
- output_rows: 486
- mode_closure_from_open: 228
- restore_reopens_boundary: 5
- distributed_restore_reopens_boundary: 0
- projection_style_closure: 104
- unique_source_cases: 59
- unique_closure_cases: 35
- unique_restore_cases: 1
- unique_distributed_restore_cases: 0

## Candidate groups

| model | candidate | label | closures | restore | distributed restore | projection closure | set types | modes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| deepseek7b | L27C16651:flip | projection_equivalent_direction_signal | 130 | 0 | 0 | 60 | {} | {"flip": 30, "half": 10, "proj_out": 30, "proj_reflect": 30, "zero": 30} |
| deepseek7b | L27C15369:subset0:zero | projection_equivalent_direction_signal | 60 | 0 | 0 | 26 | {} | {"flip": 13, "half": 8, "proj_out": 13, "proj_reflect": 13, "zero": 13} |
| qwen3 | L31C2257:flip | projection_equivalent_direction_signal | 26 | 5 | 0 | 12 | {"exact_cut": 5} | {"flip": 6, "half": 2, "proj_out": 6, "proj_reflect": 6, "zero": 6} |
| deepseek7b | L26C8587:subset1:zero | projection_equivalent_direction_signal | 12 | 0 | 0 | 6 | {} | {"flip": 3, "proj_out": 3, "proj_reflect": 3, "zero": 3} |
| glm4 | L31C6437:flip | negative_no_distributed_restore | 0 | 0 | 0 | 0 | {} | {} |
| glm4 | L31C6437:zero | negative_no_distributed_restore | 0 | 0 | 0 | 0 | {} | {} |
