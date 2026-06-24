# Phase188 Atlas Model Summary

Atlas v0 imports existing GLM5 result files. It does not rerun CUDA models.

## Imported Sample Summaries by Model

| model | sample summaries |
|---|---:|
| deepseek7b | 23 |
| glm4 | 23 |
| qwen3 | 23 |

## Imported Sample Summaries by Phase

| phase | count |
|---:|---:|
| 575 | 3 |
| 576 | 3 |
| 577 | 3 |
| 578 | 3 |
| 579 | 3 |
| 580 | 3 |
| 581 | 3 |
| 582 | 3 |
| 583 | 3 |
| 584 | 3 |
| 585 | 3 |
| 586 | 3 |
| 587 | 3 |
| 588 | 3 |
| 589 | 3 |
| 590 | 3 |
| 591 | 3 |
| 592 | 3 |
| 593 | 3 |
| 594 | 3 |
| 595 | 3 |
| 596 | 3 |
| 597 | 3 |

## Causal Level Coverage

| level | name | sample summaries |
|---:|---|---:|
| 1 | correlation | 0 |
| 2 | decodable_projection | 9 |
| 3 | transition_evidence | 27 |
| 4 | component_path_contribution | 30 |
| 5 | hidden_causal_repair | 3 |
| 6 | specific_repair | 0 |
| 7 | generation_compositional_closure | 0 |

## Highest-Priority Reading

- Level 2/3 evidence remains abundant for candidate-specific ranking.
- Level 4/5 evidence is sparse and often negative for value winner repair.
- Polarity-format gate is the strongest hidden-repair region; value/ranking gate remains the central gap.
