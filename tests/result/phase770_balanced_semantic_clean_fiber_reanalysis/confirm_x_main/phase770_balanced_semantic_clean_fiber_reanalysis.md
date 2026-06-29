# Phase 770 Balanced Semantic-Clean Fiber Reanalysis (confirm / main)

- Status: `complete`
- Input: Phase 765 causal-fiber effect rows and Phase 767 semantic/exact labels.
- This is an offline balanced reanalysis; no model was loaded.
- Balanced strata: `domain,relation,context_format`

## Label Counts

| model | exact clean | semantic clean | semantic only | semantic fail |
|---|---:|---:|---:|---:|
| qwen3 | 89 | 89 | 0 | 19 |
| glm4 | 66 | 81 | 15 | 27 |
| deepseek7b | 19 | 38 | 19 | 70 |

## Balanced Contrast Deltas

Delta means arm A minus arm B after matching counts inside each stratum.

| model | contrast | strata | cases each | delta sep | delta NN | delta object gap | delta domain gap |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `semantic_clean_vs_semantic_fail` | 11 | 11 | 0.310 | 0.000 | -0.569 | -0.459 |
| qwen3 | `exact_clean_vs_semantic_only` | 0 | 0 | null | null | null | null |
| qwen3 | `exact_clean_vs_semantic_fail` | 11 | 11 | 0.310 | 0.000 | -0.569 | -0.459 |
| qwen3 | `semantic_only_vs_semantic_fail` | 0 | 0 | null | null | null | null |
| glm4 | `semantic_clean_vs_semantic_fail` | 16 | 16 | 0.325 | 0.000 | 0.331 | 0.045 |
| glm4 | `exact_clean_vs_semantic_only` | 3 | 3 | -0.005 | 0.000 | null | null |
| glm4 | `exact_clean_vs_semantic_fail` | 15 | 15 | 0.230 | 0.000 | 0.548 | -0.183 |
| glm4 | `semantic_only_vs_semantic_fail` | 2 | 2 | 0.000 | 0.000 | null | null |
| deepseek7b | `semantic_clean_vs_semantic_fail` | 18 | 18 | 0.064 | 0.000 | -0.069 | -0.209 |
| deepseek7b | `exact_clean_vs_semantic_only` | 0 | 0 | null | null | null | null |
| deepseek7b | `exact_clean_vs_semantic_fail` | 8 | 8 | 0.006 | 0.000 | null | null |
| deepseek7b | `semantic_only_vs_semantic_fail` | 10 | 10 | 0.219 | 0.000 | null | null |

## Balanced Arm Metrics

| model | contrast | arm | cases | mean sep | mean NN | object gap | domain gap |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `semantic_clean_vs_semantic_fail` | `semantic_clean` | 11 | 0.614 | 0.000 | -0.070 | 0.222 |
| qwen3 | `semantic_clean_vs_semantic_fail` | `semantic_fail` | 11 | 0.304 | 0.000 | 0.499 | 0.681 |
| qwen3 | `exact_clean_vs_semantic_fail` | `exact_clean` | 11 | 0.614 | 0.000 | -0.070 | 0.222 |
| qwen3 | `exact_clean_vs_semantic_fail` | `semantic_fail` | 11 | 0.304 | 0.000 | 0.499 | 0.681 |
| glm4 | `semantic_clean_vs_semantic_fail` | `semantic_clean` | 16 | -0.324 | 0.000 | 1.374 | -0.329 |
| glm4 | `semantic_clean_vs_semantic_fail` | `semantic_fail` | 16 | -0.649 | 0.000 | 1.043 | -0.374 |
| glm4 | `exact_clean_vs_semantic_only` | `exact_clean` | 3 | 0.349 | 0.000 | null | null |
| glm4 | `exact_clean_vs_semantic_only` | `semantic_only` | 3 | 0.353 | 0.000 | null | null |
| glm4 | `exact_clean_vs_semantic_fail` | `exact_clean` | 15 | -0.300 | 0.000 | 1.388 | -0.440 |
| glm4 | `exact_clean_vs_semantic_fail` | `semantic_fail` | 15 | -0.530 | 0.000 | 0.840 | -0.257 |
| glm4 | `semantic_only_vs_semantic_fail` | `semantic_only` | 2 | 1.000 | 0.000 | null | null |
| glm4 | `semantic_only_vs_semantic_fail` | `semantic_fail` | 2 | 1.000 | 0.000 | null | null |
| deepseek7b | `semantic_clean_vs_semantic_fail` | `semantic_clean` | 18 | -0.451 | 0.000 | 0.333 | -0.359 |
| deepseek7b | `semantic_clean_vs_semantic_fail` | `semantic_fail` | 18 | -0.515 | 0.000 | 0.402 | -0.150 |
| deepseek7b | `exact_clean_vs_semantic_only` | `exact_clean` | 0 | null | null | null | null |
| deepseek7b | `exact_clean_vs_semantic_only` | `semantic_only` | 0 | null | null | null | null |
| deepseek7b | `exact_clean_vs_semantic_fail` | `exact_clean` | 8 | 0.264 | 0.000 | -1.000 | 0.000 |
| deepseek7b | `exact_clean_vs_semantic_fail` | `semantic_fail` | 8 | 0.258 | 0.000 | null | null |
| deepseek7b | `semantic_only_vs_semantic_fail` | `semantic_only` | 10 | -0.467 | 0.000 | null | null |
| deepseek7b | `semantic_only_vs_semantic_fail` | `semantic_fail` | 10 | -0.687 | 0.000 | null | null |

## Paired Context Stability

Each pair is the same object and same relation across `commonsense_question` and `commonsense_statement`.

| model | group type | group | pairs | mean context cosine |
|---|---|---|---:|---:|
| qwen3 | `by_semantic_pair` | `both_clean` | 40 | 0.951 |
| qwen3 | `by_semantic_pair` | `both_fail` | 5 | 0.959 |
| qwen3 | `by_semantic_pair` | `mixed` | 9 | 0.944 |
| qwen3 | `by_exact_pair` | `both_clean` | 40 | 0.951 |
| qwen3 | `by_exact_pair` | `both_fail` | 5 | 0.959 |
| qwen3 | `by_exact_pair` | `mixed` | 9 | 0.944 |
| qwen3 | `by_lexical_pair` | `exact__exact` | 40 | 0.951 |
| qwen3 | `by_lexical_pair` | `exact__semantic_fail` | 9 | 0.944 |
| qwen3 | `by_lexical_pair` | `semantic_fail__semantic_fail` | 5 | 0.959 |
| glm4 | `by_semantic_pair` | `both_clean` | 38 | 0.931 |
| glm4 | `by_semantic_pair` | `both_fail` | 11 | 0.918 |
| glm4 | `by_semantic_pair` | `mixed` | 5 | 0.933 |
| glm4 | `by_exact_pair` | `both_clean` | 25 | 0.949 |
| glm4 | `by_exact_pair` | `both_fail` | 13 | 0.918 |
| glm4 | `by_exact_pair` | `mixed` | 16 | 0.906 |
| glm4 | `by_lexical_pair` | `exact__exact` | 25 | 0.949 |
| glm4 | `by_lexical_pair` | `exact__semantic_fail` | 3 | 0.945 |
| glm4 | `by_lexical_pair` | `exact__semantic_only` | 13 | 0.897 |
| glm4 | `by_lexical_pair` | `semantic_fail__semantic_fail` | 11 | 0.918 |
| glm4 | `by_lexical_pair` | `semantic_fail__semantic_only` | 2 | 0.916 |
| deepseek7b | `by_semantic_pair` | `both_clean` | 10 | 0.636 |
| deepseek7b | `by_semantic_pair` | `both_fail` | 26 | 0.640 |
| deepseek7b | `by_semantic_pair` | `mixed` | 18 | 0.720 |
| deepseek7b | `by_exact_pair` | `both_clean` | 3 | 0.738 |
| deepseek7b | `by_exact_pair` | `both_fail` | 38 | 0.661 |
| deepseek7b | `by_exact_pair` | `mixed` | 13 | 0.662 |
| deepseek7b | `by_lexical_pair` | `exact__exact` | 3 | 0.738 |
| deepseek7b | `by_lexical_pair` | `exact__semantic_fail` | 6 | 0.744 |
| deepseek7b | `by_lexical_pair` | `exact__semantic_only` | 7 | 0.592 |
| deepseek7b | `by_lexical_pair` | `semantic_fail__semantic_fail` | 26 | 0.640 |
| deepseek7b | `by_lexical_pair` | `semantic_fail__semantic_only` | 12 | 0.708 |

## Strict Interpretation

- If balanced deltas differ from Phase 769, the previous subset result was partly caused by object/relation/context distribution.
- If semantic clean still fails to improve fiber metrics after balancing, output closure and internal fiber stability are genuinely separated.
- Paired context stability is stricter than subset filtering because the object and relation are held fixed.
- This audit is still offline and head/source-level; it does not replace new causal interventions or neuron/channel-level atlas work.
