# Phase63 Same Class Reader Calibration Summary

## qwen3

cases=384, rows=3072

| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | different_letter | 0.9896 | 0.9583 | 0.9792 | 2.5648 | 2.5661 | yes |
| 2 | b_statement_true | 0.9271 | 0.7083 | 0.9010 | 2.9575 | 2.9985 | no |
| 3 | c_statement_true | 0.9010 | 0.7396 | 0.8021 | 2.5876 | 2.6696 | no |
| 4 | same_letter | 0.8802 | 0.5208 | 0.7865 | 3.2979 | 3.4899 | no |
| 5 | b_same_yesno | 0.8724 | 0.5625 | 0.7448 | 1.5853 | 1.7845 | no |
| 6 | c_same_yesno | 0.8698 | 0.5625 | 0.7396 | 1.8958 | 1.9831 | no |
| 7 | json_same | 0.6198 | 0.4479 | 0.2969 | 0.9100 | 2.0617 | no |
| 8 | same_object_label | 0.5807 | 0.5000 | 0.2656 | 2.2721 | 3.7572 | no |

## glm4

cases=384, rows=3072

| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | json_same | 0.8099 | 0.5729 | 0.6979 | 1.1681 | 1.4438 | no |
| 2 | c_statement_true | 0.7057 | 0.5208 | 0.4792 | 0.2670 | 0.4291 | no |
| 3 | same_object_label | 0.6510 | 0.5000 | 0.3021 | 0.4714 | 0.7959 | no |
| 4 | b_statement_true | 0.6198 | 0.5208 | 0.2552 | 0.2401 | 0.4956 | no |
| 5 | different_letter | 0.5052 | 0.5000 | 0.0104 | 0.0409 | 0.6808 | no |
| 6 | b_same_yesno | 0.5000 | 0.5000 | 0.0000 | 0.1955 | 1.2915 | no |
| 7 | c_same_yesno | 0.5000 | 0.5000 | 0.0000 | 0.1479 | 1.1789 | no |
| 8 | same_letter | 0.5000 | 0.5000 | 0.0000 | 0.2021 | 1.0117 | no |

## deepseek7b

cases=384, rows=3072

| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | c_same_yesno | 0.7552 | 0.5521 | 0.5938 | 0.5418 | 0.7476 | no |
| 2 | b_same_yesno | 0.7266 | 0.5312 | 0.5365 | 0.5094 | 0.7145 | no |
| 3 | b_statement_true | 0.6536 | 0.5208 | 0.4427 | 0.3537 | 0.6772 | no |
| 4 | c_statement_true | 0.6536 | 0.5104 | 0.4062 | 0.3424 | 0.7243 | no |
| 5 | same_object_label | 0.5990 | 0.5000 | 0.1979 | 0.8605 | 1.9539 | no |
| 6 | json_same | 0.5182 | 0.5000 | 0.0729 | 0.3189 | 1.7844 | no |
| 7 | different_letter | 0.5026 | 0.5000 | 0.0052 | 0.7223 | 2.3633 | no |
| 8 | same_letter | 0.5000 | 0.5000 | 0.0000 | 0.5246 | 3.7847 | no |

## Cross Model

| rank | reader | mean_acc | min_acc | min_ctx | min_variant | all_pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | c_statement_true | 0.7535 | 0.6536 | 0.5104 | 0.4062 | no |
| 2 | b_statement_true | 0.7335 | 0.6198 | 0.5208 | 0.2552 | no |
| 3 | same_object_label | 0.6102 | 0.5807 | 0.5000 | 0.1979 | no |
| 4 | json_same | 0.6493 | 0.5182 | 0.4479 | 0.0729 | no |
| 5 | different_letter | 0.6658 | 0.5026 | 0.5000 | 0.0052 | no |
| 6 | c_same_yesno | 0.7083 | 0.5000 | 0.5000 | 0.0000 | no |
| 7 | b_same_yesno | 0.6997 | 0.5000 | 0.5000 | 0.0000 | no |
| 8 | same_letter | 0.6267 | 0.5000 | 0.5000 | 0.0000 | no |
