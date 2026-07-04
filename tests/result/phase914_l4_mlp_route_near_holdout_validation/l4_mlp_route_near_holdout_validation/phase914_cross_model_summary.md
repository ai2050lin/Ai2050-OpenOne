# Phase 914 GLM4 route-near L4 MLP channel group holdout validation

## Overall

- models: qwen3, glm4, deepseek7b
- route_eos_top10: 23
- route_eos_top5: 2
- route_eos_top50: 40
- route_near_route_rows: 40
- route_near_source_rows: 1400
- route_rows: 288
- rows: 1688
- source_eos_top1: 0
- source_eos_top10: 714
- source_eos_top5: 52
- source_eos_top50: 1400
- source_margin_nonnegative: 0
- source_promoted_top10_from_non_top10: 21
- source_promoted_top5_from_non_top5: 8
- source_promoted_top5_unique_eval_keys: 5
- source_rank_improved: 482
- source_rows: 1400
- source_strict_clean_candidate: 0
- source_top5_already_route_top5: 44
- strict_clean_candidate: 0
- strong_holdout_candidate: 0
- weak_holdout_candidate: 12

## Model Summaries

| model | eval items | route rows | route top5 | route top50 | near source rows | top5 | promoted top5 | top10 | margin>=0 | weak | strong | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 96 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_route_near_samples_for_l4_holdout |
| glm4 | 96 | 96 | 2 | 40 | 1400 | 52 | 8 | 714 | 0 | 12 | 0 | l4_holdout_reaches_eos_top5 |
| deepseek7b | 96 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_route_near_samples_for_l4_holdout |

## Top Groups

| model | group | factor | rows | top5 | top10 | margin>=0 | weak | strong | median band16 delta | median eos delta | blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| glm4 | top_abs_64 | 0.3 | 40 | 7 | 32 | 0 | 5 | 0 | -0.16015625 | 0.15625 | {} |
| glm4 | top_abs_64 | 0.4 | 40 | 5 | 27 | 0 | 7 | 0 | -0.146484375 | 0.125 | {} |
| glm4 | band16_support_32 | 0.4 | 40 | 2 | 18 | 0 | 0 | 0 | -0.119140625 | 0.0 | {} |
| glm4 | band16_support_64 | 0.4 | 40 | 2 | 18 | 0 | 0 | 0 | -0.107421875 | 0.0 | {} |
| glm4 | top_abs_64 | 0.6 | 40 | 2 | 25 | 0 | 0 | 0 | -0.0693359375 | 0.0625 | {} |
| glm4 | top_abs_64 | 0.7 | 40 | 2 | 23 | 0 | 0 | 0 | -0.052734375 | 0.03125 | {} |
| glm4 | band16_support_32 | 0.6 | 40 | 2 | 18 | 0 | 0 | 0 | -0.05078125 | 0.0 | {} |
| glm4 | top_abs_64 | 0.8 | 40 | 2 | 23 | 0 | 0 | 0 | -0.044921875 | 0.03125 | {} |
| glm4 | band16_support_32 | 0.7 | 40 | 2 | 18 | 0 | 0 | 0 | -0.03125 | 0.0 | {} |
| glm4 | band16_support_64 | 0.6 | 40 | 2 | 22 | 0 | 0 | 0 | -0.0283203125 | 0.03125 | {} |
| glm4 | band32_support_64 | 0.7 | 40 | 2 | 18 | 0 | 0 | 0 | -0.025390625 | 0.0 | {} |
| glm4 | band16_support_64 | 0.8 | 40 | 2 | 18 | 0 | 0 | 0 | -0.017578125 | 0.0 | {} |
| glm4 | band32_support_64 | 0.9 | 40 | 2 | 22 | 0 | 0 | 0 | -0.0107421875 | 0.0 | {} |
| glm4 | band16_support_32 | 0.9 | 40 | 2 | 19 | 0 | 0 | 0 | -0.005859375 | 0.0 | {} |
| glm4 | low_abs_64 | 0.7 | 40 | 2 | 23 | 0 | 0 | 0 | -0.00390625 | 0.015625 | {} |
| glm4 | low_abs_64 | 0.5 | 40 | 2 | 19 | 0 | 0 | 0 | 0.0048828125 | 0.0 | {} |
| glm4 | low_abs_64 | 0.3 | 40 | 2 | 23 | 0 | 0 | 0 | 0.005859375 | 0.0 | {} |
| glm4 | low_abs_64 | 0.6 | 40 | 2 | 23 | 0 | 0 | 0 | 0.0078125 | 0.0 | {} |
| glm4 | low_abs_64 | 0.4 | 40 | 2 | 23 | 0 | 0 | 0 | 0.0078125 | 0.0 | {} |
| glm4 | band16_support_64 | 0.9 | 40 | 2 | 18 | 0 | 0 | 0 | 0.0 | 0.0 | {} |
| glm4 | low_abs_64 | 0.9 | 40 | 2 | 23 | 0 | 0 | 0 | 0.0 | 0.0 | {} |
| glm4 | low_abs_64 | 0.8 | 40 | 2 | 23 | 0 | 0 | 0 | 0.0 | 0.0 | {} |
| glm4 | band32_support_64 | 0.3 | 40 | 0 | 14 | 0 | 0 | 0 | -0.208984375 | 0.0 | {} |
| glm4 | band16_support_32 | 0.3 | 40 | 0 | 18 | 0 | 0 | 0 | -0.18359375 | -0.046875 | {} |
| glm4 | band16_support_64 | 0.3 | 40 | 0 | 18 | 0 | 0 | 0 | -0.18359375 | -0.0625 | {} |
| glm4 | top_abs_64 | 0.5 | 40 | 0 | 25 | 0 | 0 | 0 | -0.123046875 | 0.09375 | {} |
| glm4 | band32_support_64 | 0.4 | 40 | 0 | 14 | 0 | 0 | 0 | -0.109375 | -0.0625 | {} |
| glm4 | band16_support_64 | 0.5 | 40 | 0 | 22 | 0 | 0 | 0 | -0.08984375 | 0.0 | {} |
| glm4 | band32_support_64 | 0.5 | 40 | 0 | 14 | 0 | 0 | 0 | -0.080078125 | -0.046875 | {} |
| glm4 | band16_support_32 | 0.5 | 40 | 0 | 18 | 0 | 0 | 0 | -0.076171875 | 0.03125 | {} |
| glm4 | band16_support_64 | 0.7 | 40 | 0 | 18 | 0 | 0 | 0 | -0.0380859375 | -0.015625 | {} |
| glm4 | band32_support_64 | 0.6 | 40 | 0 | 14 | 0 | 0 | 0 | -0.037109375 | 0.0 | {} |
| glm4 | band32_support_64 | 0.8 | 40 | 0 | 22 | 0 | 0 | 0 | -0.029296875 | 0.03125 | {} |
| glm4 | band16_support_32 | 0.8 | 40 | 0 | 18 | 0 | 0 | 0 | -0.0244140625 | -0.015625 | {} |
| glm4 | top_abs_64 | 0.9 | 40 | 0 | 23 | 0 | 0 | 0 | -0.013671875 | 0.03125 | {} |

## Top Monotonic

| model | case | prompt | group | weak | strong | band mono | eos nonneg all | best factor | best rank | best band16 delta | best eos delta |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| glm4 | p856_021_material_wood | natural_question | top_abs_64 | True | False | True | True | 0.3 | 7 | -0.353515625 | 0.0 |
| glm4 | p856_021_material_wood | natural_question | top_abs_64 | True | False | True | True | 0.3 | 7 | -0.353515625 | 0.0 |
| glm4 | p856_008_animal_bird | natural_question | top_abs_64 | True | False | False | True | 0.3 | 9 | -0.25 | 0.125 |
| glm4 | p856_008_animal_bird | natural_question | top_abs_64 | True | False | False | True | 0.3 | 9 | -0.25 | 0.125 |
| glm4 | p856_022_material_iron | natural_question | top_abs_64 | True | False | False | False | 0.3 | 11 | -0.306640625 | 0.09375 |
| glm4 | p856_009_animal_fish | natural_question | top_abs_64 | True | False | False | False | 0.3 | 12 | -0.24609375 | 0.21875 |
| glm4 | p856_009_animal_fish | natural_question | top_abs_64 | True | False | False | False | 0.3 | 12 | -0.24609375 | 0.21875 |
| glm4 | p856_009_animal_fish | natural_question | band16_support_32 | False | False | True | True | 0.3 | 15 | -0.240234375 | 0.03125 |
| glm4 | p856_009_animal_fish | natural_question | band16_support_32 | False | False | True | True | 0.3 | 15 | -0.240234375 | 0.03125 |
| glm4 | p856_023_material_plastic | natural_question | top_abs_64 | False | False | True | True | 0.3 | 9 | -0.201171875 | 0.125 |
| glm4 | p856_022_material_iron | natural_category | band16_support_64 | False | False | True | False | 0.3 | 14 | -0.400390625 | -0.3125 |
| glm4 | p856_009_animal_fish | natural_category | band16_support_64 | False | False | True | False | 0.3 | 12 | -0.3359375 | -0.21875 |
| glm4 | p856_009_animal_fish | natural_category | band16_support_64 | False | False | True | False | 0.3 | 12 | -0.3359375 | -0.21875 |
| glm4 | p856_009_animal_fish | natural_category | band16_support_32 | False | False | True | False | 0.3 | 12 | -0.294921875 | -0.21875 |
| glm4 | p856_009_animal_fish | natural_category | band16_support_32 | False | False | True | False | 0.3 | 12 | -0.294921875 | -0.21875 |
| glm4 | p856_036_object_car | question_plain | band16_support_64 | False | False | True | False | 0.5 | 10 | -0.2578125 | -0.28125 |
| glm4 | p856_036_object_car | question_plain | band16_support_64 | False | False | True | False | 0.5 | 10 | -0.2578125 | -0.28125 |
| glm4 | p856_036_object_car | question_plain | band16_support_64 | False | False | True | False | 0.5 | 10 | -0.2578125 | -0.28125 |
| glm4 | p856_036_object_car | question_plain | band16_support_64 | False | False | True | False | 0.5 | 10 | -0.2578125 | -0.28125 |
| glm4 | p856_022_material_iron | natural_category | band16_support_32 | False | False | True | False | 0.4 | 15 | -0.251953125 | -0.25 |
| glm4 | p856_009_animal_fish | question_plain | band16_support_64 | False | False | True | False | 0.5 | 7 | -0.244140625 | -0.25 |
| glm4 | p856_009_animal_fish | question_plain | band16_support_64 | False | False | True | False | 0.5 | 7 | -0.244140625 | -0.25 |
| glm4 | p856_009_animal_fish | question_plain | band32_support_64 | False | False | True | False | 0.5 | 7 | -0.203125 | -0.1875 |
| glm4 | p856_009_animal_fish | question_plain | band32_support_64 | False | False | True | False | 0.5 | 7 | -0.203125 | -0.1875 |
| glm4 | p856_010_animal_mammal | classification | band16_support_64 | False | False | True | False | 0.3 | 12 | -0.18359375 | -0.0625 |
| glm4 | p856_010_animal_mammal | classification | band16_support_64 | False | False | True | False | 0.3 | 12 | -0.18359375 | -0.0625 |
| glm4 | p856_009_animal_fish | question_plain | band16_support_32 | False | False | True | False | 0.6 | 7 | -0.1796875 | -0.15625 |
| glm4 | p856_009_animal_fish | question_plain | band16_support_32 | False | False | True | False | 0.6 | 7 | -0.1796875 | -0.15625 |
| glm4 | p856_022_material_iron | natural_category | top_abs_64 | False | False | True | False | 0.3 | 11 | -0.171875 | 0.125 |
| glm4 | p856_036_object_car | natural_category | band16_support_32 | False | False | True | False | 0.6 | 11 | -0.130859375 | -0.03125 |
| glm4 | p856_036_object_car | natural_category | band16_support_32 | False | False | True | False | 0.6 | 11 | -0.130859375 | -0.03125 |
| glm4 | p856_036_object_car | natural_category | band16_support_32 | False | False | True | False | 0.6 | 11 | -0.130859375 | -0.03125 |
| glm4 | p856_036_object_car | natural_category | band16_support_32 | False | False | True | False | 0.6 | 11 | -0.130859375 | -0.03125 |
| glm4 | p856_036_object_car | question_plain | band16_support_32 | False | False | True | False | 0.7 | 11 | -0.111328125 | -0.15625 |
| glm4 | p856_036_object_car | question_plain | band16_support_32 | False | False | True | False | 0.7 | 11 | -0.111328125 | -0.15625 |
| glm4 | p856_036_object_car | question_plain | band16_support_32 | False | False | True | False | 0.7 | 11 | -0.111328125 | -0.15625 |
| glm4 | p856_036_object_car | question_plain | band16_support_32 | False | False | True | False | 0.7 | 11 | -0.111328125 | -0.15625 |
| glm4 | p856_036_object_car | question_plain | band32_support_64 | False | False | True | False | 0.8 | 10 | -0.09765625 | -0.09375 |
| glm4 | p856_036_object_car | question_plain | band32_support_64 | False | False | True | False | 0.8 | 10 | -0.09765625 | -0.09375 |
| glm4 | p856_036_object_car | question_plain | band32_support_64 | False | False | True | False | 0.8 | 10 | -0.09765625 | -0.09375 |
| glm4 | p856_036_object_car | question_plain | band32_support_64 | False | False | True | False | 0.8 | 10 | -0.09765625 | -0.09375 |
| glm4 | p856_009_animal_fish | natural_question | band32_support_64 | False | False | False | True | 0.3 | 15 | -0.236328125 | 0.03125 |
| glm4 | p856_009_animal_fish | natural_question | band32_support_64 | False | False | False | True | 0.3 | 15 | -0.236328125 | 0.03125 |
| glm4 | p856_008_animal_bird | natural_question | band32_support_64 | False | False | False | True | 0.3 | 11 | -0.208984375 | 0.03125 |
| glm4 | p856_008_animal_bird | natural_question | band32_support_64 | False | False | False | True | 0.3 | 11 | -0.208984375 | 0.03125 |
| glm4 | p856_023_material_plastic | natural_question | band16_support_64 | False | False | False | True | 0.3 | 11 | -0.189453125 | 0.0 |
| glm4 | p856_008_animal_bird | natural_question | band16_support_32 | False | False | False | True | 0.3 | 11 | -0.18359375 | 0.03125 |
| glm4 | p856_008_animal_bird | natural_question | band16_support_32 | False | False | False | True | 0.3 | 11 | -0.18359375 | 0.03125 |
| glm4 | p856_008_animal_bird | natural_question | band16_support_64 | False | False | False | True | 0.3 | 11 | -0.171875 | 0.09375 |
| glm4 | p856_008_animal_bird | natural_question | band16_support_64 | False | False | False | True | 0.3 | 11 | -0.171875 | 0.09375 |
| glm4 | p856_038_object_object | natural_question | top_abs_64 | False | False | False | True | 0.3 | 6 | -0.1640625 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | band16_support_64 | False | False | False | True | 0.3 | 10 | -0.158203125 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | band16_support_64 | False | False | False | True | 0.3 | 10 | -0.158203125 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | band16_support_64 | False | False | False | True | 0.3 | 10 | -0.158203125 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | band16_support_64 | False | False | False | True | 0.3 | 10 | -0.158203125 | 0.0625 |
| glm4 | p856_009_animal_fish | natural_category | top_abs_64 | False | False | False | True | 0.3 | 10 | -0.15625 | 0.1875 |
| glm4 | p856_009_animal_fish | natural_category | top_abs_64 | False | False | False | True | 0.3 | 10 | -0.15625 | 0.1875 |
| glm4 | p856_036_object_car | natural_question | top_abs_64 | False | False | False | True | 0.3 | 8 | -0.13671875 | 0.1875 |
| glm4 | p856_036_object_car | natural_question | top_abs_64 | False | False | False | True | 0.3 | 8 | -0.13671875 | 0.1875 |
| glm4 | p856_036_object_car | natural_question | top_abs_64 | False | False | False | True | 0.3 | 8 | -0.13671875 | 0.1875 |
| glm4 | p856_036_object_car | natural_question | top_abs_64 | False | False | False | True | 0.3 | 8 | -0.13671875 | 0.1875 |
| glm4 | p856_036_object_car | natural_question | band16_support_32 | False | False | False | True | 0.3 | 10 | -0.134765625 | 0.0 |
| glm4 | p856_036_object_car | natural_question | band16_support_32 | False | False | False | True | 0.3 | 10 | -0.134765625 | 0.0 |
| glm4 | p856_036_object_car | natural_question | band16_support_32 | False | False | False | True | 0.3 | 10 | -0.134765625 | 0.0 |
| glm4 | p856_036_object_car | natural_question | band16_support_32 | False | False | False | True | 0.3 | 10 | -0.134765625 | 0.0 |
| glm4 | p856_010_animal_mammal | classification | band32_support_64 | False | False | False | True | 0.3 | 12 | -0.1171875 | 0.0 |
| glm4 | p856_010_animal_mammal | classification | band32_support_64 | False | False | False | True | 0.3 | 12 | -0.1171875 | 0.0 |
| glm4 | p856_036_object_car | natural_question | band32_support_64 | False | False | False | True | 0.3 | 10 | -0.10546875 | 0.03125 |
| glm4 | p856_036_object_car | natural_question | band32_support_64 | False | False | False | True | 0.3 | 10 | -0.10546875 | 0.03125 |
| glm4 | p856_036_object_car | natural_question | band32_support_64 | False | False | False | True | 0.3 | 10 | -0.10546875 | 0.03125 |
| glm4 | p856_036_object_car | natural_question | band32_support_64 | False | False | False | True | 0.3 | 10 | -0.10546875 | 0.03125 |
| glm4 | p856_009_animal_fish | question_plain | top_abs_64 | False | False | False | True | 0.3 | 5 | -0.076171875 | 0.25 |
| glm4 | p856_009_animal_fish | question_plain | top_abs_64 | False | False | False | True | 0.3 | 5 | -0.076171875 | 0.25 |
| glm4 | p856_038_object_object | natural_question | top_abs_64 | False | False | False | True | 0.3 | 5 | -0.041015625 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | top_abs_64 | False | False | False | True | 0.3 | 5 | -0.041015625 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | top_abs_64 | False | False | False | True | 0.3 | 5 | -0.041015625 | 0.1875 |
| glm4 | p856_023_material_plastic | natural_question | low_abs_64 | False | False | False | True | 0.3 | 12 | -0.029296875 | 0.0 |
| glm4 | p856_036_object_car | question_plain | top_abs_64 | False | False | False | True | 0.5 | 9 | -0.029296875 | 0.125 |
| glm4 | p856_036_object_car | question_plain | top_abs_64 | False | False | False | True | 0.5 | 9 | -0.029296875 | 0.125 |
| glm4 | p856_036_object_car | question_plain | top_abs_64 | False | False | False | True | 0.5 | 9 | -0.029296875 | 0.125 |
| glm4 | p856_036_object_car | question_plain | top_abs_64 | False | False | False | True | 0.5 | 9 | -0.029296875 | 0.125 |
| glm4 | p856_022_material_iron | question_plain | low_abs_64 | False | False | False | True | 0.3 | 10 | -0.01953125 | 0.03125 |
| glm4 | p856_038_object_object | natural_question | low_abs_64 | False | False | False | True | 0.3 | 8 | -0.009765625 | 0.0625 |
| glm4 | p856_010_animal_mammal | classification | low_abs_64 | False | False | False | True | 0.6 | 12 | 0.009765625 | 0.03125 |
| glm4 | p856_010_animal_mammal | classification | low_abs_64 | False | False | False | True | 0.6 | 12 | 0.009765625 | 0.03125 |
| glm4 | p856_036_object_car | natural_category | low_abs_64 | False | False | False | True | 0.5 | 12 | 0.037109375 | 0.0625 |
| glm4 | p856_036_object_car | natural_category | low_abs_64 | False | False | False | True | 0.5 | 12 | 0.037109375 | 0.0625 |
| glm4 | p856_036_object_car | natural_category | low_abs_64 | False | False | False | True | 0.5 | 12 | 0.037109375 | 0.0625 |
| glm4 | p856_036_object_car | natural_category | low_abs_64 | False | False | False | True | 0.5 | 12 | 0.037109375 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | low_abs_64 | False | False | False | True | 0.3 | 10 | 0.0390625 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | low_abs_64 | False | False | False | True | 0.3 | 10 | 0.0390625 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | low_abs_64 | False | False | False | True | 0.3 | 10 | 0.0390625 | 0.0625 |
| glm4 | p856_036_object_car | natural_question | low_abs_64 | False | False | False | True | 0.3 | 10 | 0.0390625 | 0.0625 |
| glm4 | p856_038_object_object | natural_question | band32_support_64 | False | False | False | True | 0.8 | 6 | 0.046875 | 0.125 |
| glm4 | p856_038_object_object | natural_question | band32_support_64 | False | False | False | True | 0.8 | 6 | 0.046875 | 0.125 |
| glm4 | p856_038_object_object | natural_question | band32_support_64 | False | False | False | True | 0.8 | 6 | 0.046875 | 0.125 |
| glm4 | p856_038_object_object | natural_question | band16_support_32 | False | False | False | True | 0.6 | 6 | 0.05078125 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | band16_support_32 | False | False | False | True | 0.6 | 6 | 0.05078125 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | band16_support_32 | False | False | False | True | 0.6 | 6 | 0.05078125 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | band16_support_64 | False | False | False | True | 0.8 | 6 | 0.07421875 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | band16_support_64 | False | False | False | True | 0.8 | 6 | 0.07421875 | 0.1875 |
| glm4 | p856_038_object_object | natural_question | band16_support_64 | False | False | False | True | 0.8 | 6 | 0.07421875 | 0.1875 |
| glm4 | p856_022_material_iron | question_plain | top_abs_64 | False | False | False | True | 0.3 | 6 | 0.1015625 | 0.4375 |
| glm4 | p856_035_object_chair | natural_question | top_abs_64 | False | False | False | True | 0.3 | 9 | 0.123046875 | 0.1875 |
| glm4 | p856_035_object_chair | natural_question | top_abs_64 | False | False | False | True | 0.3 | 9 | 0.123046875 | 0.1875 |
| glm4 | p856_035_object_chair | natural_question | top_abs_64 | False | False | False | True | 0.3 | 9 | 0.123046875 | 0.1875 |
| glm4 | p856_035_object_chair | natural_question | top_abs_64 | False | False | False | True | 0.3 | 9 | 0.123046875 | 0.1875 |
| glm4 | p856_008_animal_bird | classification | top_abs_64 | False | False | False | False | 0.3 | 14 | -0.392578125 | -0.09375 |
| glm4 | p856_008_animal_bird | classification | top_abs_64 | False | False | False | False | 0.3 | 14 | -0.392578125 | -0.09375 |
| glm4 | p856_009_animal_fish | natural_category | band32_support_64 | False | False | False | False | 0.3 | 12 | -0.373046875 | -0.28125 |
| glm4 | p856_009_animal_fish | natural_category | band32_support_64 | False | False | False | False | 0.3 | 12 | -0.373046875 | -0.28125 |
| glm4 | p856_008_animal_bird | classification | band16_support_64 | False | False | False | False | 0.4 | 14 | -0.3515625 | -0.125 |
| glm4 | p856_008_animal_bird | classification | band16_support_64 | False | False | False | False | 0.4 | 14 | -0.3515625 | -0.125 |
| glm4 | p856_022_material_iron | natural_category | band32_support_64 | False | False | False | False | 0.4 | 14 | -0.318359375 | -0.28125 |
| glm4 | p856_008_animal_bird | classification | band32_support_64 | False | False | False | False | 0.4 | 13 | -0.298828125 | -0.0625 |
| glm4 | p856_008_animal_bird | classification | band32_support_64 | False | False | False | False | 0.4 | 13 | -0.298828125 | -0.0625 |
| glm4 | p856_022_material_iron | natural_question | band16_support_32 | False | False | False | False | 0.3 | 13 | -0.287109375 | -0.09375 |
| glm4 | p856_036_object_car | natural_category | band32_support_64 | False | False | False | False | 0.5 | 11 | -0.265625 | -0.15625 |
| glm4 | p856_036_object_car | natural_category | band32_support_64 | False | False | False | False | 0.5 | 11 | -0.265625 | -0.15625 |
| glm4 | p856_036_object_car | natural_category | band32_support_64 | False | False | False | False | 0.5 | 11 | -0.265625 | -0.15625 |
