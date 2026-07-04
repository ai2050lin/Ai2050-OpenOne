# Phase 886 local subspace boundary decomposition

## Overall

- source_rows: 2016
- paired_rows: 504
- candidate_closure: 45
- opposite_closure: 46
- both_closure: 36
- same_blocker_direction: 66
- mean_removed_jaccard: 0.3912281990985694
- random_closure: 0
- neighbor_closure: 0

## Candidate groups

| model | candidate | label | pairs | cand | opp | both | same blocker | removed J | random | neighbor |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | L27C16651:flip | same_blocker_local_subspace_boundary | 102 | 22 | 17 | 17 | 25 | 0.733 | 0 | 0 |
| qwen3 | L31C2257:flip | same_blocker_local_subspace_boundary | 102 | 8 | 4 | 4 | 12 | 0.549 | 0 | 0 |
| deepseek7b | L26C8587:subset1:zero | same_blocker_local_subspace_boundary | 102 | 4 | 8 | 4 | 13 | 0.525 | 0 | 0 |
| deepseek7b | L27C15369:subset0:zero | same_blocker_local_subspace_boundary | 102 | 11 | 17 | 11 | 16 | 0.580 | 0 | 0 |
| glm4 | L31C6437:flip | negative_no_local_boundary | 48 | 0 | 0 | 0 | 0 | 0.000 | 0 | 0 |
| glm4 | L31C6437:zero | negative_no_local_boundary | 48 | 0 | 0 | 0 | 0 | 0.000 | 0 | 0 |
