# Phase 881 Dominant Gear Robustness and Repair (dominant_l27c16651_repair)

- Boundary: single dominant gear robustness and candidate repair; not token-level minimal cut.
- qwen3/GLM4 are included sequentially; missing candidate sources are recorded explicitly.

## Models

| model | status | candidates | rows | closure from open | answer gain | clean-like | nonclean-like |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | complete | 8 | 192 | 0 | 0 | 0 | 0 |
| glm4 | no_candidate_sources | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | complete | 4 | 288 | 11 | 10 | 1 | 10 |

## Overall

- Overall summary: `{'n': 480, 'models': {'qwen3': 192, 'deepseek7b': 288}, 'domains': {'material': 288, 'animal': 96, 'color': 96}, 'candidate_counts': {'L31C2257:flip': 24, 'L31C2257:half': 24, 'L31C2257:zero': 24, 'L31C2257:scale_up': 24, 'L31C4800:flip': 24, 'L31C4800:half': 24, 'L31C4800:zero': 24, 'L31C4800:scale_up': 24, 'L27C16651:flip': 72, 'L27C16651:half': 72, 'L27C16651:zero': 72, 'L27C16651:scale_up': 72}, 'closure_from_open': 11, 'answer_gain': 10, 'clean_like_closure': 1, 'nonclean_like_closure': 10, 'intervened_boundary_closed': 95, 'mean_class_logit_delta': 0.07858072916666667, 'mean_blocker_reduction': 0.9041666666666667, 'mean_rank_improvement': 0.9041666666666667, 'mean_original_blocker_delta': -0.004304350517679599}`

## By Candidate

| candidate | n | domains | closure from open | answer gain | nonclean-like | mean blocker red. | mean rank improve |
|---|---:|---|---:|---:|---:|---:|---:|
| `L27C16651:flip` | 72 | `{'animal': 24, 'material': 24, 'color': 24}` | 4 | 4 | 3 | 0.5555555555555556 | 0.5555555555555556 |
| `L27C16651:half` | 72 | `{'animal': 24, 'material': 24, 'color': 24}` | 3 | 2 | 3 | 0.25 | 0.25 |
| `L27C16651:scale_up` | 72 | `{'animal': 24, 'material': 24, 'color': 24}` | 1 | 1 | 1 | -0.5833333333333334 | -0.5833333333333334 |
| `L27C16651:zero` | 72 | `{'animal': 24, 'material': 24, 'color': 24}` | 3 | 3 | 3 | 0.06944444444444445 | 0.06944444444444445 |
| `L31C2257:flip` | 24 | `{'material': 24}` | 0 | 0 | 0 | 9.958333333333334 | 9.958333333333334 |
| `L31C2257:half` | 24 | `{'material': 24}` | 0 | 0 | 0 | 2.7916666666666665 | 2.7916666666666665 |
| `L31C2257:scale_up` | 24 | `{'material': 24}` | 0 | 0 | 0 | -4.666666666666667 | -4.666666666666667 |
| `L31C2257:zero` | 24 | `{'material': 24}` | 0 | 0 | 0 | 5.458333333333333 | 5.458333333333333 |
| `L31C4800:flip` | 24 | `{'material': 24}` | 0 | 0 | 0 | 1.25 | 1.25 |
| `L31C4800:half` | 24 | `{'material': 24}` | 0 | 0 | 0 | 0.75 | 0.75 |
| `L31C4800:scale_up` | 24 | `{'material': 24}` | 0 | 0 | 0 | 0.7083333333333334 | 0.7083333333333334 |
| `L31C4800:zero` | 24 | `{'material': 24}` | 0 | 0 | 0 | 0.9583333333333334 | 0.9583333333333334 |
