# Phase 804 True Semantic Suppressor Projection Search (main)

- Status: `complete`
- Boundary: removes matched semantic blocker readout direction from route deltas.
- A candidate must lower matched semantic blocker logits; lower new-blocker rate alone is insufficient.

## By Target Alpha And Semantic Beta

| model | target alpha | semantic beta | rows | cases | target gain | target gain vs a0 | old suppress | new rate | true semantic suppress | still above | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 6 | 3 | -0.521 | 0.000 | 0.851 | 0.380 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 6}` |
| qwen3 | 0.000 | 0.500 | 6 | 3 | -0.552 | -0.031 | 1.666 | 0.233 | 1.561 | 0.697 | 0.000 | `{"semantic_logit_suppression_without_closure": 5, "true_semantic_suppressor_candidate_strict": 1}` |
| qwen3 | 0.000 | 1.000 | 6 | 3 | -0.625 | -0.104 | 2.485 | 0.172 | 3.142 | 0.346 | 0.000 | `{"semantic_logit_suppression_without_closure": 3, "true_semantic_suppressor_candidate_strict": 3}` |
| qwen3 | 0.000 | 1.500 | 6 | 3 | -0.714 | -0.193 | 3.346 | 0.172 | 4.765 | 0.098 | 0.000 | `{"semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 4}` |
| qwen3 | 0.750 | 0.000 | 6 | 3 | 2.224 | 2.745 | 0.854 | 0.065 | -0.042 | 0.379 | 0.000 | `{"below_target_without_true_semantic_suppression": 4, "weak_or_mixed": 2}` |
| qwen3 | 0.750 | 0.500 | 6 | 3 | 2.198 | 2.719 | 1.668 | 0.039 | 1.513 | 0.081 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 6}` |
| qwen3 | 0.750 | 1.000 | 6 | 3 | 2.125 | 2.646 | 2.493 | 0.031 | 3.095 | 0.017 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 6}` |
| qwen3 | 0.750 | 1.500 | 6 | 3 | 2.057 | 2.578 | 3.338 | 0.036 | 4.705 | 0.010 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 6}` |
| glm4 | 0.000 | 0.000 | 6 | 3 | 0.195 | 0.000 | 0.470 | 0.202 | 0.000 | 1.000 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "weak_or_mixed": 5}` |
| glm4 | 0.000 | 0.500 | 6 | 3 | 0.151 | -0.044 | 0.542 | 0.159 | 0.403 | 0.499 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 1, "weak_or_mixed": 2}` |
| glm4 | 0.000 | 1.000 | 6 | 3 | 0.102 | -0.094 | 0.614 | 0.152 | 0.804 | 0.318 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 1, "weak_or_mixed": 1}` |
| glm4 | 0.000 | 1.500 | 6 | 3 | 0.065 | -0.130 | 0.689 | 0.147 | 1.213 | 0.259 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 1, "weak_or_mixed": 1}` |
| glm4 | 0.750 | 0.000 | 6 | 3 | 0.870 | 0.674 | 0.471 | 0.103 | 0.001 | 0.487 | 0.000 | `{"below_target_without_true_semantic_suppression": 3, "weak_or_mixed": 3}` |
| glm4 | 0.750 | 0.500 | 6 | 3 | 0.802 | 0.607 | 0.542 | 0.080 | 0.402 | 0.323 | 0.000 | `{"below_target_without_true_semantic_suppression": 2, "semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 1, "weak_or_mixed": 1}` |
| glm4 | 0.750 | 1.000 | 6 | 3 | 0.753 | 0.557 | 0.616 | 0.070 | 0.808 | 0.223 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 1, "true_semantic_suppressor_candidate_strict": 2, "weak_or_mixed": 1}` |
| glm4 | 0.750 | 1.500 | 6 | 3 | 0.719 | 0.523 | 0.692 | 0.064 | 1.215 | 0.205 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 1, "true_semantic_suppressor_candidate_strict": 2, "weak_or_mixed": 1}` |
| deepseek7b | 0.000 | 0.000 | 4 | 2 | 0.299 | 0.000 | -0.741 | 0.545 | 0.000 | 1.000 | 0.000 | `{"weak_or_mixed": 4}` |
| deepseek7b | 0.000 | 0.500 | 4 | 2 | 0.248 | -0.051 | 0.308 | 0.326 | 1.572 | 0.875 | 0.000 | `{"semantic_logit_suppression_without_closure": 4}` |
| deepseek7b | 0.000 | 1.000 | 4 | 2 | 0.170 | -0.129 | 1.381 | 0.245 | 3.180 | 0.469 | 0.000 | `{"semantic_logit_suppression_without_closure": 3, "true_semantic_suppressor_candidate_strict": 1}` |
| deepseek7b | 0.000 | 1.500 | 4 | 2 | 0.131 | -0.168 | 2.440 | 0.233 | 4.774 | 0.224 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 1, "semantic_logit_suppression_without_closure": 2, "true_semantic_suppressor_candidate_strict": 1}` |
| deepseek7b | 0.750 | 0.000 | 4 | 2 | 2.699 | 2.400 | -0.878 | 0.162 | -0.096 | 0.677 | 0.000 | `{"below_target_without_true_semantic_suppression": 1, "weak_or_mixed": 3}` |
| deepseek7b | 0.750 | 0.500 | 4 | 2 | 2.654 | 2.355 | 0.166 | 0.099 | 1.483 | 0.365 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 4}` |
| deepseek7b | 0.750 | 1.000 | 4 | 2 | 2.602 | 2.303 | 1.234 | 0.112 | 3.078 | 0.271 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 4}` |
| deepseek7b | 0.750 | 1.500 | 4 | 2 | 2.588 | 2.289 | 2.292 | 0.142 | 4.670 | 0.104 | 0.000 | `{"semantic_logit_suppression_but_target_shifted": 3, "semantic_logit_suppression_without_closure": 1}` |
