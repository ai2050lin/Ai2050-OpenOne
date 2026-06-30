# Phase 802 New-Blocker Stabilization Dose Response (main)

- Status: `complete`
- Boundary: alpha=0 is target-neutral, alpha=1 is raw route patch, alpha>1 over-injects the direct target direction.
- This phase tests whether adding controlled target-readout dose reduces new blockers while preserving old-blocker suppression.

## By Alpha

| model | alpha | rows | cases | target gain | old suppression | resolved | new rate | anchor | closure score | token gain | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 8 | 4 | -0.703 | 0.756 | 0.206 | 0.418 | 0.766 | 0.149 | 0.000 | `{"old_suppress_new_unstable": 7, "weak_or_mixed": 1}` |
| qwen3 | 0.250 | 8 | 4 | 0.188 | 0.747 | 0.354 | 0.268 | 1.188 | 0.322 | 0.000 | `{"old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 6, "weak_or_mixed": 1}` |
| qwen3 | 0.500 | 8 | 4 | 1.039 | 0.745 | 0.508 | 0.139 | 1.570 | 0.393 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 4, "weak_or_mixed": 1}` |
| qwen3 | 0.750 | 8 | 4 | 1.926 | 0.734 | 0.645 | 0.070 | 2.051 | 0.394 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 4, "old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 2, "weak_or_mixed": 1}` |
| qwen3 | 1.000 | 8 | 4 | 2.785 | 0.734 | 0.734 | 0.035 | 2.504 | 0.316 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 6, "old_suppress_new_stable_anchor_weak": 1, "weak_or_mixed": 1}` |
| glm4 | 0.000 | 8 | 4 | 0.043 | 0.395 | 0.298 | 0.238 | -0.660 | 0.140 | 0.000 | `{"old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 4, "weak_or_mixed": 3}` |
| glm4 | 0.250 | 8 | 4 | 0.246 | 0.393 | 0.346 | 0.185 | -0.480 | 0.151 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1, "old_suppress_new_unstable": 4, "weak_or_mixed": 3}` |
| glm4 | 0.500 | 8 | 4 | 0.453 | 0.399 | 0.396 | 0.148 | -0.273 | 0.162 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1, "old_suppress_new_unstable": 4, "weak_or_mixed": 3}` |
| glm4 | 0.750 | 8 | 4 | 0.664 | 0.395 | 0.457 | 0.119 | -0.086 | 0.170 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1, "old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 3, "weak_or_mixed": 3}` |
| glm4 | 1.000 | 8 | 4 | 0.834 | 0.400 | 0.496 | 0.098 | 0.092 | 0.181 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 2, "weak_or_mixed": 3}` |
| deepseek7b | 0.000 | 8 | 4 | 0.938 | -0.415 | 0.337 | 0.367 | 0.419 | 0.016 | 0.000 | `{"threshold_shift_without_suppression": 2, "weak_or_mixed": 6}` |
| deepseek7b | 0.250 | 8 | 4 | 1.606 | -0.447 | 0.432 | 0.251 | 0.806 | 0.021 | 0.000 | `{"threshold_shift_without_suppression": 3, "weak_or_mixed": 5}` |
| deepseek7b | 0.500 | 8 | 4 | 2.241 | -0.474 | 0.591 | 0.156 | 1.151 | 0.023 | 0.000 | `{"threshold_shift_without_suppression": 4, "weak_or_mixed": 4}` |
| deepseek7b | 0.750 | 8 | 4 | 2.896 | -0.513 | 0.683 | 0.129 | 1.482 | 0.019 | 0.000 | `{"threshold_shift_without_suppression": 5, "weak_or_mixed": 3}` |
| deepseek7b | 1.000 | 8 | 4 | 3.557 | -0.549 | 0.732 | 0.124 | 1.854 | 0.017 | 0.000 | `{"threshold_shift_without_suppression": 5, "weak_or_mixed": 3}` |

## Best Alpha Triplets

| model | case | route | best alpha | target gain | old suppress | new rate | anchor | score | label | new reduction vs a0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.250 | 0.938 | 1.206 | 0.264 | 3.438 | 0.986 | `old_suppress_new_unstable` | 0.111 |
| qwen3 | `p765_0058_commonsense_statement_object:chair:edible` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.500 | 1.500 | 0.964 | 0.037 | 4.375 | 0.740 | `old_suppress_new_stable_anchor_ok` | 0.341 |
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.750 | 1.062 | 0.700 | 0.048 | 6.312 | 0.652 | `old_suppress_new_stable_anchor_ok` | 0.429 |
| qwen3 | `p765_0006_commonsense_statement_fruit:apple:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.750 | 0.938 | 0.704 | 0.073 | 1.938 | 0.495 | `old_suppress_new_stable_anchor_ok` | 0.523 |
| qwen3 | `p765_0056_commonsense_statement_object:chair:category` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.750 | 2.812 | 0.983 | 0.009 | 0.812 | 0.406 | `old_suppress_new_stable_anchor_ok` | 0.363 |
| qwen3 | `p765_0058_commonsense_statement_object:chair:edible` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.500 | 0.875 | 0.748 | 0.306 | 0.500 | 0.348 | `old_suppress_new_unstable` | 0.270 |
| qwen3 | `p765_0056_commonsense_statement_object:chair:category` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.000 | 1.031 | 0.601 | 0.100 | -6.844 | 0.265 | `old_suppress_new_unstable` | 0.000 |
| qwen3 | `p765_0006_commonsense_statement_fruit:apple:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.500 | 0.500 | 0.070 | 0.225 | 3.000 | 0.020 | `weak_or_mixed` | 0.245 |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | 0.344 | 0.790 | 0.242 | 1.531 | 0.462 | `old_suppress_new_unstable` | 0.000 |
| glm4 | `p765_0055_commonsense_question_object:chair:category` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.000 | 0.688 | 0.757 | 0.067 | -1.625 | 0.454 | `old_suppress_new_stable_anchor_weak` | 0.172 |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 0.500 | 2.062 | 0.661 | 0.000 | 0.688 | 0.371 | `old_suppress_new_stable_anchor_ok` | 0.033 |
| glm4 | `p765_0056_commonsense_statement_object:chair:category` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.000 | 0.422 | 0.384 | 0.098 | -1.141 | 0.158 | `old_suppress_new_stable_anchor_weak` | 0.132 |
| glm4 | `p765_0024_commonsense_statement_animal:cat:grows_on_tree` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.000 | 2.266 | 0.138 | 0.000 | 0.703 | 0.072 | `weak_or_mixed` | 0.127 |
| glm4 | `p765_0055_commonsense_question_object:chair:category` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 1.000 | -0.422 | 0.476 | 0.152 | -0.172 | 0.049 | `old_suppress_new_unstable` | 0.342 |
| glm4 | `p765_0024_commonsense_statement_animal:cat:grows_on_tree` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.750 | 0.891 | 0.031 | 0.023 | 0.516 | 0.021 | `weak_or_mixed` | 0.052 |
| glm4 | `p765_0056_commonsense_statement_object:chair:category` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | -0.656 | -0.039 | 0.464 | -0.906 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.500 | 2.719 | 0.179 | 0.159 | 2.594 | 0.076 | `weak_or_mixed` | 0.004 |
| deepseek7b | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.500 | 1.344 | 0.098 | 0.000 | 0.969 | 0.064 | `weak_or_mixed` | 0.233 |
| deepseek7b | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 2.500 | 0.167 | 0.206 | 0.250 | 0.047 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0103_commonsense_question_abstract:justice:category` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.250 | -0.402 | 0.035 | 0.512 | -3.996 | 0.002 | `weak_or_mixed` | 0.343 |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 0.938 | -1.100 | 0.501 | 1.438 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.000 | -0.094 | -0.551 | 0.574 | 1.094 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0052_commonsense_statement_plant:wheat:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.000 | 1.438 | -0.854 | 0.156 | 0.062 | 0.000 | `threshold_shift_without_suppression` | 0.000 |
| deepseek7b | `p765_0103_commonsense_question_abstract:justice:category` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 3.301 | -1.531 | 0.249 | 3.113 | 0.000 | `threshold_shift_without_suppression` | 0.000 |
