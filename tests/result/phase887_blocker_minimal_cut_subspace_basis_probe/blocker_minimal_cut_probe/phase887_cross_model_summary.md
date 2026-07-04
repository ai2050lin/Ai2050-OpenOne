# Phase 887 blocker-token minimal cut and subspace basis probe

## Overall

- paired_rows: 504
- same_boundary_closure: 36
- shared_complete_topk_cut: 35
- exact_single_blocker_cut: 23
- candidate_complete_topk_cut: 42
- opposite_complete_topk_cut: 45

## Candidate groups

| model | candidate | label | pairs | both | shared cut | single cut | exact clean | exact nonclean | mean cut |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | L27C16651:flip | single_token_minimal_cut_signal | 102 | 17 | 17 | 9 | 16 | 4 | 1.76 |
| deepseek7b | L27C15369:subset0:zero | single_token_minimal_cut_signal | 102 | 11 | 11 | 9 | 3 | 9 | 1.18 |
| qwen3 | L31C2257:flip | topk_complete_minimal_cut_signal | 102 | 4 | 4 | 2 | 2 | 3 | 1.75 |
| deepseek7b | L26C8587:subset1:zero | single_token_minimal_cut_signal | 102 | 4 | 3 | 3 | 0 | 3 | 1.00 |
| glm4 | L31C6437:flip | negative_no_min_cut | 48 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| glm4 | L31C6437:zero | negative_no_min_cut | 48 | 0 | 0 | 0 | 0 | 0 | 0.00 |
