# Phase 745 Route-Level Multi-Competitor Suppression Validation (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: final-norm route-level suppression on the Phase743/744 joint+topK near-closure state.

| model | condition | scale | n | donor top1 | mean donor rank | margin gain | top classes | route shift |
|---|---|---:|---:|---:|---:|---:|---|---:|
| qwen3 | suppress_route_representatives | 1.00 | 16 | 1.000 | 1.00 | 7.637 | `{"donor_answer": 16}` | 1.000 |
| qwen3 | suppress_route_representatives | 1.25 | 16 | 1.000 | 1.00 | 7.637 | `{"donor_answer": 16}` | 1.000 |
| qwen3 | suppress_route_centroids | 1.00 | 16 | 1.000 | 1.00 | 7.637 | `{"donor_answer": 16}` | 1.000 |
| qwen3 | suppress_route_centroids | 1.25 | 16 | 1.000 | 1.00 | 7.637 | `{"donor_answer": 16}` | 1.000 |
| qwen3 | suppress_all_topk_competitors | 1.00 | 16 | 1.000 | 1.00 | 7.637 | `{"donor_answer": 16}` | 1.000 |
| qwen3 | suppress_all_topk_competitors | 1.25 | 16 | 1.000 | 1.00 | 7.637 | `{"donor_answer": 16}` | 1.000 |
| qwen3 | suppress_class:format_or_schema | 1.25 | 16 | 0.188 | 1.81 | 5.988 | `{"donor_answer": 3, "recipient_answer": 13}` | 0.188 |
| qwen3 | suppress_current_top | 1.25 | 16 | 0.000 | 2.56 | 6.301 | `{"donor_answer": 1, "format_or_schema": 15}` | 1.000 |
| qwen3 | suppress_current_top_class | 1.25 | 16 | 0.000 | 2.56 | 6.301 | `{"donor_answer": 1, "format_or_schema": 15}` | 1.000 |
| qwen3 | suppress_class:recipient_answer | 1.25 | 16 | 0.000 | 2.56 | 6.301 | `{"donor_answer": 1, "format_or_schema": 15}` | 1.000 |
| qwen3 | suppress_current_top | 1.00 | 16 | 0.000 | 2.94 | 5.379 | `{"format_or_schema": 16}` | 1.000 |
| qwen3 | suppress_current_top_class | 1.00 | 16 | 0.000 | 2.94 | 5.379 | `{"format_or_schema": 16}` | 1.000 |
| glm4 | suppress_class:punctuation_or_stop | 1.00 | 6 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 6}` | 0.000 |
| glm4 | suppress_class:punctuation_or_stop | 1.25 | 6 | 1.000 | 1.00 | 0.000 | `{"donor_answer": 6}` | 0.000 |
| glm4 | suppress_route_representatives | 1.25 | 16 | 0.875 | 1.00 | 1.047 | `{"donor_answer": 14, "other_vocab": 2}` | 0.500 |
| glm4 | suppress_route_centroids | 1.25 | 16 | 0.875 | 1.00 | 1.047 | `{"donor_answer": 14, "other_vocab": 2}` | 0.500 |
| glm4 | suppress_all_topk_competitors | 1.25 | 16 | 0.875 | 1.00 | 1.047 | `{"donor_answer": 14, "other_vocab": 2}` | 0.500 |
| glm4 | suppress_route_centroids | 1.00 | 16 | 0.750 | 1.06 | 1.043 | `{"donor_answer": 12, "other_vocab": 4}` | 0.375 |
| glm4 | suppress_route_representatives | 1.00 | 16 | 0.750 | 1.12 | 1.039 | `{"donor_answer": 12, "other_vocab": 4}` | 0.375 |
| glm4 | suppress_all_topk_competitors | 1.00 | 16 | 0.750 | 1.12 | 1.039 | `{"donor_answer": 12, "other_vocab": 4}` | 0.375 |
| glm4 | suppress_current_top_class | 1.25 | 16 | 0.750 | 1.12 | 1.020 | `{"donor_answer": 12, "other_vocab": 4}` | 0.500 |
| glm4 | suppress_current_top_class | 1.00 | 16 | 0.625 | 1.25 | 0.984 | `{"donor_answer": 10, "other_vocab": 6}` | 0.375 |
| glm4 | suppress_current_top | 1.25 | 16 | 0.625 | 1.25 | 0.949 | `{"donor_answer": 10, "echo_object_or_relation": 2, "other_vocab": 4}` | 0.375 |
| glm4 | suppress_class:echo_object_or_relation | 1.25 | 13 | 0.538 | 1.54 | 0.990 | `{"donor_answer": 7, "other_vocab": 4, "recipient_answer": 2}` | 0.308 |
| deepseek7b | suppress_route_representatives | 1.25 | 16 | 1.000 | 1.00 | 3.438 | `{"donor_answer": 16}` | 1.000 |
| deepseek7b | suppress_route_centroids | 1.25 | 16 | 1.000 | 1.00 | 3.438 | `{"donor_answer": 16}` | 1.000 |
| deepseek7b | suppress_all_topk_competitors | 1.25 | 16 | 1.000 | 1.00 | 3.438 | `{"donor_answer": 16}` | 1.000 |
| deepseek7b | suppress_route_representatives | 1.00 | 16 | 0.938 | 1.06 | 3.434 | `{"donor_answer": 15, "format_or_schema": 1}` | 0.938 |
| deepseek7b | suppress_route_centroids | 1.00 | 16 | 0.938 | 1.06 | 3.434 | `{"donor_answer": 15, "format_or_schema": 1}` | 0.938 |
| deepseek7b | suppress_all_topk_competitors | 1.00 | 16 | 0.938 | 1.06 | 3.434 | `{"donor_answer": 15, "format_or_schema": 1}` | 0.938 |
| deepseek7b | suppress_current_top_class | 1.25 | 16 | 0.750 | 1.62 | 3.250 | `{"donor_answer": 12, "echo_object_or_relation": 1, "format_or_schema": 1, "punctuation_or_stop": 2}` | 1.000 |
| deepseek7b | suppress_class:echo_object_or_relation | 1.25 | 13 | 0.615 | 2.46 | 2.769 | `{"donor_answer": 8, "format_or_schema": 3, "punctuation_or_stop": 1, "recipient_answer": 1}` | 0.769 |
| deepseek7b | suppress_current_top | 1.25 | 16 | 0.500 | 2.19 | 3.082 | `{"donor_answer": 9, "echo_object_or_relation": 3, "format_or_schema": 2, "punctuation_or_stop": 2}` | 0.875 |
| deepseek7b | suppress_class:echo_object_or_relation | 1.00 | 13 | 0.385 | 3.15 | 2.587 | `{"donor_answer": 5, "echo_object_or_relation": 1, "format_or_schema": 4, "other_vocab": 1, "punctuation_or_stop": 1, "recipient_answer": 1}` | 0.692 |
| deepseek7b | suppress_current_top_class | 1.00 | 16 | 0.375 | 2.38 | 3.055 | `{"donor_answer": 6, "echo_object_or_relation": 2, "format_or_schema": 4, "other_vocab": 2, "punctuation_or_stop": 2}` | 0.812 |
| deepseek7b | suppress_class:punctuation_or_stop | 1.25 | 16 | 0.250 | 2.75 | 2.055 | `{"donor_answer": 4, "echo_object_or_relation": 8, "format_or_schema": 3, "recipient_answer": 1}` | 0.250 |

## By Base Top Class

| model | base class | condition | scale | n | donor top1 | margin gain | new top classes |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | suppress_route_representatives | 1.00 | 16 | 1.000 | 7.637 | `{"donor_answer": 16}` |
| qwen3 | recipient_answer | suppress_route_representatives | 1.25 | 16 | 1.000 | 7.637 | `{"donor_answer": 16}` |
| qwen3 | recipient_answer | suppress_route_centroids | 1.00 | 16 | 1.000 | 7.637 | `{"donor_answer": 16}` |
| qwen3 | recipient_answer | suppress_route_centroids | 1.25 | 16 | 1.000 | 7.637 | `{"donor_answer": 16}` |
| qwen3 | recipient_answer | suppress_all_topk_competitors | 1.00 | 16 | 1.000 | 7.637 | `{"donor_answer": 16}` |
| qwen3 | recipient_answer | suppress_all_topk_competitors | 1.25 | 16 | 1.000 | 7.637 | `{"donor_answer": 16}` |
| qwen3 | recipient_answer | suppress_class:format_or_schema | 1.25 | 16 | 0.188 | 5.988 | `{"donor_answer": 3, "recipient_answer": 13}` |
| qwen3 | recipient_answer | suppress_current_top | 1.25 | 16 | 0.000 | 6.301 | `{"donor_answer": 1, "format_or_schema": 15}` |
| qwen3 | recipient_answer | suppress_current_top_class | 1.25 | 16 | 0.000 | 6.301 | `{"donor_answer": 1, "format_or_schema": 15}` |
| qwen3 | recipient_answer | suppress_class:recipient_answer | 1.25 | 16 | 0.000 | 6.301 | `{"donor_answer": 1, "format_or_schema": 15}` |
| qwen3 | recipient_answer | suppress_current_top | 1.00 | 16 | 0.000 | 5.379 | `{"format_or_schema": 16}` |
| qwen3 | recipient_answer | suppress_current_top_class | 1.00 | 16 | 0.000 | 5.379 | `{"format_or_schema": 16}` |
| qwen3 | recipient_answer | suppress_class:recipient_answer | 1.00 | 16 | 0.000 | 5.379 | `{"format_or_schema": 16}` |
| qwen3 | recipient_answer | suppress_class:format_or_schema | 1.00 | 16 | 0.000 | 4.887 | `{"recipient_answer": 16}` |
| qwen3 | recipient_answer | suppress_class:punctuation_or_stop | 1.25 | 16 | 0.000 | 4.535 | `{"recipient_answer": 16}` |
| qwen3 | recipient_answer | suppress_class:other_semantic_value | 1.25 | 6 | 0.000 | 3.885 | `{"format_or_schema": 6}` |
| qwen3 | recipient_answer | suppress_class:punctuation_or_stop | 1.00 | 16 | 0.000 | 3.645 | `{"recipient_answer": 16}` |
| qwen3 | recipient_answer | suppress_class:other_semantic_value | 1.00 | 6 | 0.000 | 3.260 | `{"format_or_schema": 2, "recipient_answer": 4}` |
| qwen3 | recipient_answer | suppress_class:other_vocab | 1.25 | 12 | 0.000 | 1.391 | `{"recipient_answer": 12}` |
| qwen3 | recipient_answer | suppress_class:other_vocab | 1.00 | 12 | 0.000 | 1.109 | `{"recipient_answer": 12}` |
| qwen3 | recipient_answer | suppress_class:echo_object_or_relation | 1.25 | 1 | 0.000 | 0.250 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | suppress_class:echo_object_or_relation | 1.00 | 1 | 0.000 | 0.125 | `{"recipient_answer": 1}` |
| qwen3 | recipient_answer | joint_add_topK | 0.00 | 16 | 0.000 | 0.000 | `{"recipient_answer": 16}` |
| glm4 | recipient_answer | suppress_route_representatives | 1.00 | 2 | 1.000 | 1.781 | `{"donor_answer": 2}` |
| glm4 | recipient_answer | suppress_route_representatives | 1.25 | 2 | 1.000 | 1.781 | `{"donor_answer": 2}` |
| glm4 | recipient_answer | suppress_route_centroids | 1.00 | 2 | 1.000 | 1.781 | `{"donor_answer": 2}` |
| glm4 | recipient_answer | suppress_route_centroids | 1.25 | 2 | 1.000 | 1.781 | `{"donor_answer": 2}` |
| glm4 | recipient_answer | suppress_all_topk_competitors | 1.00 | 2 | 1.000 | 1.781 | `{"donor_answer": 2}` |
| glm4 | recipient_answer | suppress_all_topk_competitors | 1.25 | 2 | 1.000 | 1.781 | `{"donor_answer": 2}` |
| glm4 | recipient_answer | suppress_current_top | 1.25 | 2 | 0.000 | 1.562 | `{"other_vocab": 2}` |
| glm4 | recipient_answer | suppress_current_top_class | 1.25 | 2 | 0.000 | 1.562 | `{"other_vocab": 2}` |
| glm4 | recipient_answer | suppress_class:recipient_answer | 1.25 | 2 | 0.000 | 1.562 | `{"other_vocab": 2}` |
| glm4 | recipient_answer | suppress_current_top | 1.00 | 2 | 0.000 | 1.344 | `{"other_vocab": 2}` |
| glm4 | recipient_answer | suppress_current_top_class | 1.00 | 2 | 0.000 | 1.344 | `{"other_vocab": 2}` |
| glm4 | recipient_answer | suppress_class:recipient_answer | 1.00 | 2 | 0.000 | 1.344 | `{"other_vocab": 2}` |
| glm4 | recipient_answer | suppress_class:other_vocab | 1.25 | 2 | 0.000 | 0.875 | `{"recipient_answer": 2}` |
| glm4 | recipient_answer | suppress_class:other_vocab | 1.00 | 2 | 0.000 | 0.688 | `{"recipient_answer": 2}` |
| glm4 | recipient_answer | suppress_class:echo_object_or_relation | 1.25 | 2 | 0.000 | 0.344 | `{"recipient_answer": 2}` |
| glm4 | recipient_answer | suppress_class:echo_object_or_relation | 1.00 | 2 | 0.000 | 0.281 | `{"recipient_answer": 2}` |
| glm4 | recipient_answer | joint_add_topK | 0.00 | 2 | 0.000 | 0.000 | `{"recipient_answer": 2}` |
| glm4 | recipient_answer | suppress_class:format_or_schema | 1.00 | 2 | 0.000 | 0.000 | `{"recipient_answer": 2}` |
| glm4 | recipient_answer | suppress_class:format_or_schema | 1.25 | 2 | 0.000 | 0.000 | `{"recipient_answer": 2}` |
| glm4 | other_vocab | suppress_current_top | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_current_top_class | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_route_representatives | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_route_centroids | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
| glm4 | other_vocab | suppress_all_topk_competitors | 1.25 | 4 | 0.500 | 0.250 | `{"donor_answer": 2, "other_vocab": 2}` |
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
| deepseek7b | other_vocab | suppress_route_representatives | 1.00 | 1 | 1.000 | 5.562 | `{"donor_answer": 1}` |
| deepseek7b | other_vocab | suppress_route_representatives | 1.25 | 1 | 1.000 | 5.562 | `{"donor_answer": 1}` |
| deepseek7b | other_vocab | suppress_route_centroids | 1.00 | 1 | 1.000 | 5.562 | `{"donor_answer": 1}` |

## Strict Interpretation

- If route-level suppression beats current-top suppression, the failure is multi-token or multi-route competition.
- If all-topK suppression beats route representatives, route classes are internally multi-token rather than represented by one token.
- If donor still fails after multi-route suppression, the remaining bottleneck is donor boost, continuation policy, or a route outside the measured top-k window.
- This phase is a readout geometry validation, not yet a natural circuit proof.

Atlas graph: nodes=33 edges=30
