# Phase 804 True Semantic Suppressor Projection Search (confirm)

- Status: `complete`
- Boundary: removes matched semantic blocker readout direction from route deltas.
- A candidate must lower matched semantic blocker logits; lower new-blocker rate alone is insufficient.

## By Target Alpha And Semantic Beta

| model | target alpha | semantic beta | rows | cases | target gain | target gain vs a0 | old suppress | new rate | true semantic suppress | still above | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 10 | 5 | -0.400 | 0.000 | 0.802 | 0.363 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 10}` |
| qwen3 | 0.000 | 1.000 | 10 | 5 | -0.472 | -0.072 | 2.093 | 0.165 | 2.637 | 0.221 | 0.000 | `{"semantic_logit_suppression_without_closure": 3, "true_semantic_suppressor_candidate_strict": 7}` |
| qwen3 | 0.750 | 0.000 | 10 | 5 | 2.216 | 2.616 | 0.764 | 0.068 | -0.073 | 0.259 | 0.000 | `{"below_target_without_true_semantic_suppression": 7, "weak_or_mixed": 3}` |
| qwen3 | 0.750 | 1.000 | 10 | 5 | 2.153 | 2.553 | 2.059 | 0.036 | 2.552 | 0.009 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 10}` |
| glm4 | 0.000 | 0.000 | 6 | 3 | 0.195 | 0.000 | 0.470 | 0.202 | 0.000 | 1.000 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "weak_or_mixed": 5}` |
| glm4 | 0.000 | 1.000 | 6 | 3 | 0.107 | -0.089 | 0.612 | 0.151 | 0.795 | 0.318 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 1, "weak_or_mixed": 1}` |
| glm4 | 0.750 | 0.000 | 6 | 3 | 0.870 | 0.674 | 0.471 | 0.103 | 0.001 | 0.454 | 0.000 | `{"below_target_without_true_semantic_suppression": 3, "weak_or_mixed": 3}` |
| glm4 | 0.750 | 1.000 | 6 | 3 | 0.755 | 0.560 | 0.613 | 0.070 | 0.798 | 0.214 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 1, "true_semantic_suppressor_candidate_strict": 2, "weak_or_mixed": 1}` |
| deepseek7b | 0.000 | 0.000 | 4 | 2 | 0.299 | 0.000 | -0.741 | 0.545 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 4}` |
| deepseek7b | 0.000 | 1.000 | 4 | 2 | 0.146 | -0.152 | 1.493 | 0.245 | 3.035 | 0.508 | 0.000 | `{"semantic_logit_suppression_without_closure": 3, "true_semantic_suppressor_candidate_strict": 1}` |
| deepseek7b | 0.750 | 0.000 | 4 | 2 | 2.699 | 2.400 | -0.878 | 0.162 | -0.087 | 0.633 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "weak_or_mixed": 3}` |
| deepseek7b | 0.750 | 1.000 | 4 | 2 | 2.590 | 2.291 | 1.355 | 0.120 | 2.953 | 0.273 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 4}` |
