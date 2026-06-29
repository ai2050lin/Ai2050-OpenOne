# Phase 745 Route-Level Multi-Competitor Suppression Validation (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: final-norm route-level suppression on the Phase743/744 joint+topK near-closure state.

| model | condition | scale | n | donor top1 | mean donor rank | margin gain | top classes | route shift |
|---|---|---:|---:|---:|---:|---:|---|---:|
| qwen3 | suppress_route_representatives | 1.00 | 20 | 1.000 | 1.00 | 7.534 | `{"donor_answer": 20}` | 1.000 |
| qwen3 | suppress_route_representatives | 1.25 | 20 | 1.000 | 1.00 | 7.534 | `{"donor_answer": 20}` | 1.000 |
| qwen3 | suppress_route_centroids | 1.00 | 20 | 1.000 | 1.00 | 7.534 | `{"donor_answer": 20}` | 1.000 |
| qwen3 | suppress_route_centroids | 1.25 | 20 | 1.000 | 1.00 | 7.534 | `{"donor_answer": 20}` | 1.000 |
| qwen3 | suppress_all_topk_competitors | 1.00 | 20 | 1.000 | 1.00 | 7.534 | `{"donor_answer": 20}` | 1.000 |
| qwen3 | suppress_all_topk_competitors | 1.25 | 20 | 1.000 | 1.00 | 7.534 | `{"donor_answer": 20}` | 1.000 |
| qwen3 | suppress_current_top | 1.25 | 20 | 0.300 | 2.15 | 6.697 | `{"donor_answer": 7, "format_or_schema": 13}` | 1.000 |
| qwen3 | suppress_current_top_class | 1.25 | 20 | 0.300 | 2.15 | 6.697 | `{"donor_answer": 7, "format_or_schema": 13}` | 1.000 |
| qwen3 | suppress_class:recipient_answer | 1.25 | 20 | 0.300 | 2.15 | 6.697 | `{"donor_answer": 7, "format_or_schema": 13}` | 1.000 |
| qwen3 | suppress_class:format_or_schema | 1.25 | 20 | 0.150 | 1.85 | 4.797 | `{"donor_answer": 3, "recipient_answer": 17}` | 0.150 |
| qwen3 | suppress_current_top | 1.00 | 20 | 0.100 | 2.70 | 5.978 | `{"donor_answer": 1, "format_or_schema": 16, "recipient_answer": 3}` | 0.850 |
| qwen3 | suppress_current_top_class | 1.00 | 20 | 0.100 | 2.70 | 5.978 | `{"donor_answer": 1, "format_or_schema": 16, "recipient_answer": 3}` | 0.850 |
| glm4 | suppress_class:punctuation_or_stop | 1.00 | 4 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 4}` | 0.000 |
| glm4 | suppress_class:punctuation_or_stop | 1.25 | 4 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 4}` | 0.000 |
| glm4 | suppress_current_top_class | 1.25 | 20 | 0.900 | 1.00 | 0.706 | `{"donor_answer": 18, "other_vocab": 2}` | 0.400 |
| glm4 | suppress_route_representatives | 1.25 | 20 | 0.900 | 1.00 | 0.706 | `{"donor_answer": 18, "other_vocab": 2}` | 0.400 |
| glm4 | suppress_route_centroids | 1.25 | 20 | 0.900 | 1.00 | 0.706 | `{"donor_answer": 18, "other_vocab": 2}` | 0.400 |
| glm4 | suppress_all_topk_competitors | 1.25 | 20 | 0.900 | 1.00 | 0.706 | `{"donor_answer": 18, "other_vocab": 2}` | 0.400 |
| glm4 | suppress_route_centroids | 1.00 | 20 | 0.800 | 1.05 | 0.703 | `{"donor_answer": 16, "other_vocab": 4}` | 0.300 |
| glm4 | suppress_route_representatives | 1.00 | 20 | 0.800 | 1.10 | 0.700 | `{"donor_answer": 16, "other_vocab": 4}` | 0.300 |
| glm4 | suppress_all_topk_competitors | 1.00 | 20 | 0.800 | 1.10 | 0.700 | `{"donor_answer": 16, "other_vocab": 4}` | 0.300 |
| glm4 | suppress_current_top | 1.25 | 20 | 0.800 | 1.10 | 0.650 | `{"donor_answer": 16, "echo_object_or_relation": 2, "other_vocab": 2}` | 0.300 |
| glm4 | suppress_class:echo_object_or_relation | 1.25 | 15 | 0.733 | 1.20 | 0.875 | `{"donor_answer": 11, "other_vocab": 4}` | 0.400 |
| glm4 | suppress_current_top_class | 1.00 | 20 | 0.700 | 1.15 | 0.697 | `{"donor_answer": 15, "echo_object_or_relation": 1, "other_vocab": 4}` | 0.250 |
| deepseek7b | suppress_route_representatives | 1.25 | 20 | 0.900 | 1.05 | 2.350 | `{"donor_answer": 18, "format_or_schema": 2}` | 0.850 |
| deepseek7b | suppress_route_centroids | 1.25 | 20 | 0.900 | 1.05 | 2.350 | `{"donor_answer": 18, "format_or_schema": 2}` | 0.850 |
| deepseek7b | suppress_all_topk_competitors | 1.25 | 20 | 0.900 | 1.05 | 2.350 | `{"donor_answer": 18, "format_or_schema": 2}` | 0.850 |
| deepseek7b | suppress_route_representatives | 1.00 | 20 | 0.850 | 1.10 | 2.347 | `{"donor_answer": 17, "format_or_schema": 3}` | 0.800 |
| deepseek7b | suppress_route_centroids | 1.00 | 20 | 0.850 | 1.10 | 2.347 | `{"donor_answer": 17, "format_or_schema": 3}` | 0.800 |
| deepseek7b | suppress_all_topk_competitors | 1.00 | 20 | 0.850 | 1.10 | 2.347 | `{"donor_answer": 17, "format_or_schema": 3}` | 0.800 |
| deepseek7b | suppress_current_top_class | 1.25 | 20 | 0.650 | 1.55 | 2.178 | `{"donor_answer": 13, "echo_object_or_relation": 2, "format_or_schema": 2, "other_semantic_value": 1, "punctuation_or_stop": 2}` | 0.850 |
| deepseek7b | suppress_current_top | 1.25 | 20 | 0.500 | 1.95 | 2.044 | `{"donor_answer": 10, "echo_object_or_relation": 3, "format_or_schema": 3, "other_semantic_value": 1, "other_vocab": 1, "punctuation_or_stop": 2}` | 0.800 |
| deepseek7b | suppress_class:echo_object_or_relation | 1.25 | 16 | 0.438 | 2.69 | 1.512 | `{"donor_answer": 7, "format_or_schema": 6, "punctuation_or_stop": 2, "recipient_answer": 1}` | 0.438 |
| deepseek7b | suppress_current_top_class | 1.00 | 20 | 0.300 | 2.20 | 2.044 | `{"donor_answer": 7, "echo_object_or_relation": 3, "format_or_schema": 6, "other_semantic_value": 1, "other_vocab": 1, "punctuation_or_stop": 2}` | 0.650 |
| deepseek7b | suppress_class:format_or_schema | 1.25 | 20 | 0.300 | 2.80 | 1.294 | `{"donor_answer": 6, "echo_object_or_relation": 8, "format_or_schema": 2, "other_semantic_value": 1, "punctuation_or_stop": 2, "recipient_answer": 1}` | 0.400 |
| deepseek7b | suppress_class:echo_object_or_relation | 1.00 | 16 | 0.250 | 3.06 | 1.391 | `{"donor_answer": 4, "echo_object_or_relation": 1, "format_or_schema": 7, "other_vocab": 1, "punctuation_or_stop": 2, "recipient_answer": 1}` | 0.375 |

## By Base Top Class

| model | base class | condition | scale | n | donor top1 | margin gain | new top classes |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | suppress_route_representatives | 1.00 | 20 | 1.000 | 7.534 | `{"donor_answer": 20}` |
| qwen3 | recipient_answer | suppress_route_representatives | 1.25 | 20 | 1.000 | 7.534 | `{"donor_answer": 20}` |
| qwen3 | recipient_answer | suppress_route_centroids | 1.00 | 20 | 1.000 | 7.534 | `{"donor_answer": 20}` |
| qwen3 | recipient_answer | suppress_route_centroids | 1.25 | 20 | 1.000 | 7.534 | `{"donor_answer": 20}` |
| qwen3 | recipient_answer | suppress_all_topk_competitors | 1.00 | 20 | 1.000 | 7.534 | `{"donor_answer": 20}` |
| qwen3 | recipient_answer | suppress_all_topk_competitors | 1.25 | 20 | 1.000 | 7.534 | `{"donor_answer": 20}` |
| qwen3 | recipient_answer | suppress_current_top | 1.25 | 20 | 0.300 | 6.697 | `{"donor_answer": 7, "format_or_schema": 13}` |
| qwen3 | recipient_answer | suppress_current_top_class | 1.25 | 20 | 0.300 | 6.697 | `{"donor_answer": 7, "format_or_schema": 13}` |
| qwen3 | recipient_answer | suppress_class:recipient_answer | 1.25 | 20 | 0.300 | 6.697 | `{"donor_answer": 7, "format_or_schema": 13}` |
| qwen3 | recipient_answer | suppress_class:format_or_schema | 1.25 | 20 | 0.150 | 4.797 | `{"donor_answer": 3, "recipient_answer": 17}` |
| qwen3 | recipient_answer | suppress_current_top | 1.00 | 20 | 0.100 | 5.978 | `{"donor_answer": 1, "format_or_schema": 16, "recipient_answer": 3}` |
| qwen3 | recipient_answer | suppress_current_top_class | 1.00 | 20 | 0.100 | 5.978 | `{"donor_answer": 1, "format_or_schema": 16, "recipient_answer": 3}` |
| qwen3 | recipient_answer | suppress_class:recipient_answer | 1.00 | 20 | 0.100 | 5.978 | `{"donor_answer": 1, "format_or_schema": 16, "recipient_answer": 3}` |
| qwen3 | recipient_answer | suppress_class:format_or_schema | 1.00 | 20 | 0.000 | 3.916 | `{"recipient_answer": 20}` |
| qwen3 | recipient_answer | suppress_class:punctuation_or_stop | 1.25 | 20 | 0.000 | 3.728 | `{"recipient_answer": 20}` |
| qwen3 | recipient_answer | suppress_class:punctuation_or_stop | 1.00 | 20 | 0.000 | 2.984 | `{"recipient_answer": 20}` |
| qwen3 | recipient_answer | suppress_class:other_semantic_value | 1.25 | 14 | 0.000 | 2.478 | `{"format_or_schema": 6, "recipient_answer": 8}` |
| qwen3 | recipient_answer | suppress_class:other_semantic_value | 1.00 | 14 | 0.000 | 2.058 | `{"format_or_schema": 2, "recipient_answer": 12}` |
| qwen3 | recipient_answer | suppress_class:other_vocab | 1.25 | 14 | 0.000 | 0.808 | `{"recipient_answer": 14}` |
| qwen3 | recipient_answer | suppress_class:other_vocab | 1.00 | 14 | 0.000 | 0.647 | `{"recipient_answer": 14}` |
| qwen3 | recipient_answer | suppress_class:echo_object_or_relation | 1.25 | 1 | 0.000 | 0.250 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | suppress_class:echo_object_or_relation | 1.00 | 1 | 0.000 | 0.125 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | joint_add_topK | 0.00 | 20 | 0.000 | 0.000 | `{"recipient_answer": 20}` |
| glm4 | other_vocab | suppress_current_top | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_current_top_class | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_route_representatives | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_route_centroids | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_all_topk_competitors | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_class:other_vocab | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_route_centroids | 1.00 | 4 | 0.000 | 0.234 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_current_top | 1.00 | 4 | 0.000 | 0.219 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_current_top_class | 1.00 | 4 | 0.000 | 0.219 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_route_representatives | 1.00 | 4 | 0.000 | 0.219 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_all_topk_competitors | 1.00 | 4 | 0.000 | 0.219 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:other_vocab | 1.00 | 4 | 0.000 | 0.219 | `{"other_vocab": 4}` |
| glm4 | other_vocab | joint_add_topK | 0.00 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:other_semantic_value | 1.00 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:other_semantic_value | 1.25 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:recipient_answer | 1.00 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:recipient_answer | 1.25 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:format_or_schema | 1.00 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:format_or_schema | 1.25 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:echo_object_or_relation | 1.00 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | other_vocab | suppress_class:echo_object_or_relation | 1.25 | 4 | 0.000 | 0.000 | `{"other_vocab": 4}` |
| glm4 | echo_object_or_relation | suppress_current_top_class | 1.25 | 6 | 1.000 | 2.188 | `{"donor_answer": 6}` |
| glm4 | echo_object_or_relation | suppress_route_representatives | 1.00 | 6 | 1.000 | 2.188 | `{"donor_answer": 6}` |
| glm4 | echo_object_or_relation | suppress_route_representatives | 1.25 | 6 | 1.000 | 2.188 | `{"donor_answer": 6}` |
| deepseek7b | recipient_answer | suppress_route_representatives | 1.00 | 1 | 1.000 | 5.188 | `{"donor_answer": 1}` |
| deepseek7b | recipient_answer | suppress_route_representatives | 1.25 | 1 | 1.000 | 5.188 | `{"donor_answer": 1}` |
| deepseek7b | recipient_answer | suppress_route_centroids | 1.00 | 1 | 1.000 | 5.188 | `{"donor_answer": 1}` |
| deepseek7b | recipient_answer | suppress_route_centroids | 1.25 | 1 | 1.000 | 5.188 | `{"donor_answer": 1}` |
| deepseek7b | recipient_answer | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 5.188 | `{"donor_answer": 1}` |
| deepseek7b | recipient_answer | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 5.188 | `{"donor_answer": 1}` |
| deepseek7b | recipient_answer | suppress_current_top | 1.25 | 1 | 0.000 | 4.938 | `{"echo_object_or_relation": 1}` |
| deepseek7b | recipient_answer | suppress_current_top_class | 1.25 | 1 | 0.000 | 4.938 | `{"echo_object_or_relation": 1}` |
| deepseek7b | recipient_answer | suppress_class:recipient_answer | 1.25 | 1 | 0.000 | 4.938 | `{"echo_object_or_relation": 1}` |
| deepseek7b | recipient_answer | suppress_current_top | 1.00 | 1 | 0.000 | 4.312 | `{"echo_object_or_relation": 1}` |
| deepseek7b | recipient_answer | suppress_current_top_class | 1.00 | 1 | 0.000 | 4.312 | `{"echo_object_or_relation": 1}` |
| deepseek7b | recipient_answer | suppress_class:recipient_answer | 1.00 | 1 | 0.000 | 4.312 | `{"echo_object_or_relation": 1}` |
| deepseek7b | recipient_answer | suppress_class:echo_object_or_relation | 1.25 | 1 | 0.000 | 3.562 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:echo_object_or_relation | 1.00 | 1 | 0.000 | 2.812 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:other_vocab | 1.25 | 1 | 0.000 | 0.812 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:other_vocab | 1.00 | 1 | 0.000 | 0.625 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:format_or_schema | 1.25 | 1 | 0.000 | 0.625 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:format_or_schema | 1.00 | 1 | 0.000 | 0.500 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:punctuation_or_stop | 1.00 | 1 | 0.000 | 0.250 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | suppress_class:punctuation_or_stop | 1.25 | 1 | 0.000 | 0.250 | `{"recipient_answer": 1}` |
| deepseek7b | recipient_answer | joint_add_topK | 0.00 | 1 | 0.000 | 0.000 | `{"recipient_answer": 1}` |
| deepseek7b | punctuation_or_stop | suppress_current_top_class | 1.00 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | punctuation_or_stop | suppress_current_top_class | 1.25 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | punctuation_or_stop | suppress_route_representatives | 1.00 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |

## Strict Interpretation

- If route-level suppression beats current-top suppression, the failure is multi-token or multi-route competition.
- If all-topK suppression beats route representatives, route classes are internally multi-token rather than represented by one token.
- If donor still fails after multi-route suppression, the remaining bottleneck is donor boost, continuation policy, or a route outside the measured top-k window.
- This phase is a readout geometry validation, not yet a natural circuit proof.

Atlas graph: nodes=33 edges=30
