# Phase 745 Route-Level Multi-Competitor Suppression Validation (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: final-norm route-level suppression on the Phase743/744 joint+topK near-closure state.

| model | condition | scale | n | donor top1 | mean donor rank | margin gain | top classes | route shift |
|---|---|---:|---:|---:|---:|---:|---|---:|
| qwen3 | suppress_route_representatives | 1.00 | 1 | 1.000 | 1.00 | 7.250 | `{"donor_answer": 1}` | 1.000 |
| qwen3 | suppress_route_representatives | 1.25 | 1 | 1.000 | 1.00 | 7.250 | `{"donor_answer": 1}` | 1.000 |
| qwen3 | suppress_route_centroids | 1.00 | 1 | 1.000 | 1.00 | 7.250 | `{"donor_answer": 1}` | 1.000 |
| qwen3 | suppress_route_centroids | 1.25 | 1 | 1.000 | 1.00 | 7.250 | `{"donor_answer": 1}` | 1.000 |
| qwen3 | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 1.00 | 7.250 | `{"donor_answer": 1}` | 1.000 |
| qwen3 | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 1.00 | 7.250 | `{"donor_answer": 1}` | 1.000 |
| qwen3 | suppress_current_top | 1.25 | 1 | 0.000 | 2.00 | 6.375 | `{"format_or_schema": 1}` | 1.000 |
| qwen3 | suppress_current_top_class | 1.25 | 1 | 0.000 | 2.00 | 6.375 | `{"format_or_schema": 1}` | 1.000 |
| qwen3 | suppress_class:recipient_answer | 1.25 | 1 | 0.000 | 2.00 | 6.375 | `{"format_or_schema": 1}` | 1.000 |
| qwen3 | suppress_current_top | 1.00 | 1 | 0.000 | 3.00 | 5.500 | `{"format_or_schema": 1}` | 1.000 |
| qwen3 | suppress_current_top_class | 1.00 | 1 | 0.000 | 3.00 | 5.500 | `{"format_or_schema": 1}` | 1.000 |
| qwen3 | suppress_class:recipient_answer | 1.00 | 1 | 0.000 | 3.00 | 5.500 | `{"format_or_schema": 1}` | 1.000 |
| glm4 | joint_add_topK | 0.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_current_top | 1.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_current_top | 1.25 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_current_top_class | 1.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_current_top_class | 1.25 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_route_representatives | 1.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_route_representatives | 1.25 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_route_centroids | 1.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_route_centroids | 1.25 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| glm4 | suppress_class:format_or_schema | 1.00 | 1 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 1}` | 0.000 |
| deepseek7b | suppress_route_representatives | 1.00 | 1 | 1.000 | 1.00 | 3.250 | `{"donor_answer": 1}` | 1.000 |
| deepseek7b | suppress_route_representatives | 1.25 | 1 | 1.000 | 1.00 | 3.250 | `{"donor_answer": 1}` | 1.000 |
| deepseek7b | suppress_route_centroids | 1.00 | 1 | 1.000 | 1.00 | 3.250 | `{"donor_answer": 1}` | 1.000 |
| deepseek7b | suppress_route_centroids | 1.25 | 1 | 1.000 | 1.00 | 3.250 | `{"donor_answer": 1}` | 1.000 |
| deepseek7b | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 1.00 | 3.250 | `{"donor_answer": 1}` | 1.000 |
| deepseek7b | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 1.00 | 3.250 | `{"donor_answer": 1}` | 1.000 |
| deepseek7b | suppress_current_top | 1.25 | 1 | 0.000 | 2.00 | 3.125 | `{"punctuation_or_stop": 1}` | 1.000 |
| deepseek7b | suppress_current_top_class | 1.25 | 1 | 0.000 | 2.00 | 3.125 | `{"punctuation_or_stop": 1}` | 1.000 |
| deepseek7b | suppress_class:echo_object_or_relation | 1.25 | 1 | 0.000 | 2.00 | 3.125 | `{"punctuation_or_stop": 1}` | 1.000 |
| deepseek7b | suppress_class:punctuation_or_stop | 1.25 | 1 | 0.000 | 2.00 | 2.875 | `{"echo_object_or_relation": 1}` | 0.000 |
| deepseek7b | suppress_current_top | 1.00 | 1 | 0.000 | 4.00 | 2.625 | `{"punctuation_or_stop": 1}` | 1.000 |
| deepseek7b | suppress_current_top_class | 1.00 | 1 | 0.000 | 4.00 | 2.625 | `{"punctuation_or_stop": 1}` | 1.000 |

## By Base Top Class

| model | base class | condition | scale | n | donor top1 | margin gain | new top classes |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | suppress_route_representatives | 1.00 | 1 | 1.000 | 7.250 | `{"donor_answer": 1}` |
| qwen3 | recipient_answer | suppress_route_representatives | 1.25 | 1 | 1.000 | 7.250 | `{"donor_answer": 1}` |
| qwen3 | recipient_answer | suppress_route_centroids | 1.00 | 1 | 1.000 | 7.250 | `{"donor_answer": 1}` |
| qwen3 | recipient_answer | suppress_route_centroids | 1.25 | 1 | 1.000 | 7.250 | `{"donor_answer": 1}` |
| qwen3 | recipient_answer | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 7.250 | `{"donor_answer": 1}` |
| qwen3 | recipient_answer | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 7.250 | `{"donor_answer": 1}` |
| qwen3 | recipient_answer | suppress_current_top | 1.25 | 1 | 0.000 | 6.375 | `{"format_or_schema": 1}` |
| qwen3 | recipient_answer | suppress_current_top_class | 1.25 | 1 | 0.000 | 6.375 | `{"format_or_schema": 1}` |
| qwen3 | recipient_answer | suppress_class:recipient_answer | 1.25 | 1 | 0.000 | 6.375 | `{"format_or_schema": 1}` |
| qwen3 | recipient_answer | suppress_current_top | 1.00 | 1 | 0.000 | 5.500 | `{"format_or_schema": 1}` |
| qwen3 | recipient_answer | suppress_current_top_class | 1.00 | 1 | 0.000 | 5.500 | `{"format_or_schema": 1}` |
| qwen3 | recipient_answer | suppress_class:recipient_answer | 1.00 | 1 | 0.000 | 5.500 | `{"format_or_schema": 1}` |
| qwen3 | recipient_answer | suppress_class:format_or_schema | 1.25 | 1 | 0.000 | 3.750 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | suppress_class:punctuation_or_stop | 1.25 | 1 | 0.000 | 3.125 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | suppress_class:format_or_schema | 1.00 | 1 | 0.000 | 3.000 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | suppress_class:punctuation_or_stop | 1.00 | 1 | 0.000 | 2.625 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | joint_add_topK | 0.00 | 1 | 0.000 | 0.000 | `{"recipient_answer": 1}` |
| glm4 | donor_answer | joint_add_topK | 0.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_current_top | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_current_top | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_current_top_class | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_current_top_class | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_route_representatives | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_route_representatives | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_route_centroids | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_route_centroids | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_class:format_or_schema | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_class:format_or_schema | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_class:other_vocab | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_class:other_vocab | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_class:recipient_answer | 1.00 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| glm4 | donor_answer | suppress_class:recipient_answer | 1.25 | 1 | 1.000 | 0.000 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_route_representatives | 1.00 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_route_representatives | 1.25 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_route_centroids | 1.00 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_route_centroids | 1.25 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_all_topk_competitors | 1.00 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_all_topk_competitors | 1.25 | 1 | 1.000 | 3.250 | `{"donor_answer": 1}` |
| deepseek7b | echo_object_or_relation | suppress_current_top | 1.25 | 1 | 0.000 | 3.125 | `{"punctuation_or_stop": 1}` |
| deepseek7b | echo_object_or_relation | suppress_current_top_class | 1.25 | 1 | 0.000 | 3.125 | `{"punctuation_or_stop": 1}` |
| deepseek7b | echo_object_or_relation | suppress_class:echo_object_or_relation | 1.25 | 1 | 0.000 | 3.125 | `{"punctuation_or_stop": 1}` |
| deepseek7b | echo_object_or_relation | suppress_class:punctuation_or_stop | 1.25 | 1 | 0.000 | 2.875 | `{"echo_object_or_relation": 1}` |
| deepseek7b | echo_object_or_relation | suppress_current_top | 1.00 | 1 | 0.000 | 2.625 | `{"punctuation_or_stop": 1}` |
| deepseek7b | echo_object_or_relation | suppress_current_top_class | 1.00 | 1 | 0.000 | 2.625 | `{"punctuation_or_stop": 1}` |
| deepseek7b | echo_object_or_relation | suppress_class:echo_object_or_relation | 1.00 | 1 | 0.000 | 2.625 | `{"punctuation_or_stop": 1}` |
| deepseek7b | echo_object_or_relation | suppress_class:punctuation_or_stop | 1.00 | 1 | 0.000 | 2.250 | `{"echo_object_or_relation": 1}` |
| deepseek7b | echo_object_or_relation | suppress_class:format_or_schema | 1.25 | 1 | 0.000 | 1.750 | `{"echo_object_or_relation": 1}` |
| deepseek7b | echo_object_or_relation | suppress_class:format_or_schema | 1.00 | 1 | 0.000 | 1.375 | `{"echo_object_or_relation": 1}` |
| deepseek7b | echo_object_or_relation | joint_add_topK | 0.00 | 1 | 0.000 | 0.000 | `{"echo_object_or_relation": 1}` |

## Strict Interpretation

- If route-level suppression beats current-top suppression, the failure is multi-token or multi-route competition.
- If all-topK suppression beats route representatives, route classes are internally multi-token rather than represented by one token.
- If donor still fails after multi-route suppression, the remaining bottleneck is donor boost, continuation policy, or a route outside the measured top-k window.
- This phase is a readout geometry validation, not yet a natural circuit proof.

Atlas graph: nodes=33 edges=30
