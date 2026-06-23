# Phase592 Cross-Model Summary

Relation-specific ranking factor atlas. Metrics are projection-level evidence, not causal repair.

## qwen3

- cases=64, target_cases=5, n_layers=36, threshold=0.25

| rank | position | layer | bucket | spec_margin | correct_specific | old_top_wrong_specific | common | pos_rate |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | prompt_last | 34 | late | +3.740 | +1.292 | -2.448 | +4.026 | 0.80 |
| 2 | prompt_last | 33 | late | +3.169 | +1.201 | -1.967 | +3.377 | 0.80 |
| 3 | prompt_last | 32 | late | +3.001 | +1.175 | -1.826 | +1.217 | 0.80 |
| 4 | query_category | 32 | late | +1.956 | +0.767 | -1.188 | -2.837 | 0.80 |
| 5 | query_category | 34 | late | +1.864 | +0.668 | -1.196 | +4.717 | 0.80 |
| 6 | prompt_last | 30 | late | +1.810 | +0.789 | -1.021 | +1.356 | 0.80 |
| 7 | prompt_last | 31 | late | +1.762 | +0.716 | -1.047 | +0.952 | 0.80 |
| 8 | query_category | 33 | late | +1.705 | +0.587 | -1.118 | +1.574 | 0.80 |
| 9 | prompt_last | 29 | late | +1.512 | +0.611 | -0.902 | +1.222 | 0.80 |
| 10 | query_category | 31 | late | +1.445 | +0.441 | -1.004 | -2.553 | 0.80 |
| 11 | query_category | 30 | late | +1.170 | +0.214 | -0.955 | -3.536 | 0.80 |
| 12 | prompt_last | 28 | late | +1.010 | +0.367 | -0.643 | +1.084 | 0.80 |

First crossing by position:

- prompt_last: L24 late_mid spec_margin=0.574, cspec=0.141, wspec=-0.433
- query_category: L19 late_mid spec_margin=0.425, cspec=0.147, wspec=-0.277
- rule_relation: none
- rule_value: none
- query_relation: none

Atlas nodes=60, edges=62

## glm4

- cases=64, target_cases=4, n_layers=40, threshold=0.25

| rank | position | layer | bucket | spec_margin | correct_specific | old_top_wrong_specific | common | pos_rate |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | prompt_last | 38 | late | +0.821 | +0.383 | -0.437 | -0.725 | 0.50 |
| 2 | prompt_last | 39 | late | +0.704 | +0.364 | -0.339 | -0.755 | 0.75 |
| 3 | prompt_last | 37 | late | +0.468 | +0.253 | -0.215 | -0.438 | 0.50 |
| 4 | prompt_last | 36 | late | +0.369 | +0.181 | -0.189 | -0.469 | 0.50 |
| 5 | prompt_last | 35 | late | +0.361 | +0.163 | -0.198 | -0.469 | 0.50 |
| 6 | prompt_last | 34 | late | +0.314 | +0.138 | -0.177 | -0.208 | 0.50 |
| 7 | prompt_last | 33 | late | +0.289 | +0.125 | -0.163 | -0.107 | 0.50 |
| 8 | prompt_last | 32 | late | +0.185 | +0.060 | -0.124 | +0.365 | 1.00 |
| 9 | rule_value | 31 | late | +0.163 | +0.079 | -0.083 | -0.415 | 0.75 |
| 10 | rule_value | 30 | late | +0.162 | +0.082 | -0.080 | -0.286 | 0.75 |
| 11 | prompt_last | 30 | late | +0.131 | +0.048 | -0.084 | +0.316 | 1.00 |
| 12 | rule_value | 32 | late | +0.126 | +0.049 | -0.076 | -0.760 | 0.75 |

First crossing by position:

- prompt_last: L33 late spec_margin=0.289, cspec=0.125, wspec=-0.163
- rule_value: none
- query_category: none
- query_relation: none
- rule_relation: none

Atlas nodes=60, edges=61

## deepseek7b

- cases=64, target_cases=21, n_layers=28, threshold=0.25

| rank | position | layer | bucket | spec_margin | correct_specific | old_top_wrong_specific | common | pos_rate |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | rule_value | 26 | late | +1.210 | +0.718 | -0.492 | +0.753 | 0.43 |
| 2 | prompt_last | 26 | late | +1.054 | +2.296 | +1.241 | +10.725 | 0.38 |
| 3 | rule_relation | 18 | late_mid | +0.774 | +0.089 | -0.684 | +2.871 | 0.76 |
| 4 | prompt_last | 25 | late | +0.760 | +1.691 | +0.931 | +8.666 | 0.48 |
| 5 | rule_relation | 20 | late_mid | +0.714 | +0.117 | -0.597 | +3.816 | 0.57 |
| 6 | rule_value | 25 | late | +0.616 | +0.469 | -0.147 | -1.122 | 0.62 |
| 7 | rule_relation | 21 | late | +0.575 | +0.070 | -0.505 | +2.427 | 0.67 |
| 8 | rule_relation | 19 | late_mid | +0.544 | +0.113 | -0.430 | +3.228 | 0.76 |
| 9 | query_relation | 16 | late_mid | +0.509 | +0.281 | -0.228 | +1.361 | 0.81 |
| 10 | query_relation | 19 | late_mid | +0.498 | +0.314 | -0.184 | +1.551 | 0.67 |
| 11 | rule_relation | 9 | mid | +0.416 | +0.189 | -0.228 | +2.709 | 0.86 |
| 12 | query_relation | 9 | mid | +0.403 | +0.118 | -0.286 | +1.291 | 0.81 |

First crossing by position:

- rule_value: L25 late spec_margin=0.616, cspec=0.469, wspec=-0.147
- prompt_last: L25 late spec_margin=0.760, cspec=1.691, wspec=0.931
- rule_relation: L8 mid spec_margin=0.280, cspec=0.127, wspec=-0.154
- query_relation: L9 mid spec_margin=0.403, cspec=0.118, wspec=-0.286
- query_category: none

Atlas nodes=60, edges=64

## Objective facts

- Qwen3 ranking projection peaks at late prompt_last, with query_category also strong.
- GLM4 ranking projection is weaker and concentrated at late prompt_last.
- DS7B ranking projection is distributed: rule_value L26, prompt_last L26, rule_relation mid-late layers, and query_relation mid layers.
- DS7B has several non-prompt_last positions above threshold, supporting an atlas view rather than a single-point mechanism.
- These are Level 2 decodable projection edges. They locate candidates for causal testing; they do not yet prove repair.
