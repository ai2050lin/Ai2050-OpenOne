# Phase 804 True Semantic Suppressor Projection Search (smoke)

- Status: `complete`
- Boundary: removes matched semantic blocker readout direction from route deltas.
- A candidate must lower matched semantic blocker logits; lower new-blocker rate alone is insufficient.

## By Target Alpha And Semantic Beta

| model | target alpha | semantic beta | rows | cases | target gain | target gain vs a0 | old suppress | new rate | true semantic suppress | still above | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 1 | 1 | -0.062 | 0.000 | 1.242 | 0.374 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 1}` |
| qwen3 | 0.000 | 1.000 | 1 | 1 | -0.188 | -0.125 | 2.237 | 0.283 | 2.673 | 0.094 | 0.000 | `{"true_semantic_suppressor_candidate_strict": 1}` |
| qwen3 | 0.750 | 0.000 | 1 | 1 | 2.812 | 2.875 | 1.143 | 0.121 | -0.174 | 0.031 | 0.000 | `{"below_target_without_true_semantic_suppression": 1}` |
| qwen3 | 0.750 | 1.000 | 1 | 1 | 2.688 | 2.750 | 2.123 | 0.107 | 2.462 | 0.000 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 1}` |
| glm4 | 0.000 | 0.000 | 1 | 1 | 0.344 | 0.000 | 0.790 | 0.242 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 1}` |
| glm4 | 0.000 | 1.000 | 1 | 1 | 0.406 | 0.062 | 1.166 | 0.068 | 1.571 | 0.000 | 0.000 | `{"true_semantic_suppressor_candidate_strict": 1}` |
| glm4 | 0.750 | 0.000 | 1 | 1 | 0.094 | -0.250 | 0.784 | 0.281 | -0.019 | 1.000 | 0.000 | `{"weak_or_mixed": 1}` |
| glm4 | 0.750 | 1.000 | 1 | 1 | 0.125 | -0.219 | 1.169 | 0.096 | 1.572 | 0.000 | 0.000 | `{"true_semantic_suppressor_candidate_strict": 1}` |
| deepseek7b | 0.000 | 0.000 | 1 | 1 | 0.938 | 0.000 | -1.100 | 0.501 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 1}` |
| deepseek7b | 0.000 | 1.000 | 1 | 1 | 0.766 | -0.172 | 0.007 | 0.217 | 2.695 | 0.500 | 0.000 | `{"semantic_logit_suppression_without_closure": 1}` |
| deepseek7b | 0.750 | 0.000 | 1 | 1 | 2.422 | 1.484 | -1.139 | 0.160 | -0.030 | 1.000 | 0.000 | `{"weak_or_mixed": 1}` |
| deepseek7b | 0.750 | 1.000 | 1 | 1 | 2.297 | 1.359 | -0.019 | 0.053 | 2.665 | 0.031 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 1}` |
