# Phase 802 New-Blocker Stabilization Dose Response (confirm)

- Status: `complete`
- Boundary: alpha=0 is target-neutral, alpha=1 is raw route patch, alpha>1 over-injects the direct target direction.
- This phase tests whether adding controlled target-readout dose reduces new blockers while preserving old-blocker suppression.

## By Alpha

| model | alpha | rows | cases | target gain | old suppression | resolved | new rate | anchor | closure score | token gain | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 12 | 6 | -0.542 | 0.747 | 0.229 | 0.392 | 0.969 | 0.166 | 0.000 | `{"old_suppress_new_unstable": 10, "weak_or_mixed": 2}` |
| qwen3 | 0.250 | 12 | 6 | 0.318 | 0.729 | 0.374 | 0.238 | 1.359 | 0.326 | 0.000 | `{"old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 9, "threshold_shift_without_suppression": 1, "weak_or_mixed": 1}` |
| qwen3 | 0.500 | 12 | 6 | 1.156 | 0.716 | 0.523 | 0.136 | 1.688 | 0.381 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 3, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 5, "threshold_shift_without_suppression": 1, "weak_or_mixed": 1}` |
| qwen3 | 0.750 | 12 | 6 | 2.018 | 0.699 | 0.652 | 0.071 | 2.091 | 0.399 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 6, "old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 3, "threshold_shift_without_suppression": 1, "weak_or_mixed": 1}` |
| qwen3 | 1.000 | 12 | 6 | 2.852 | 0.691 | 0.728 | 0.041 | 2.477 | 0.337 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 8, "old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 1, "threshold_shift_without_suppression": 1, "weak_or_mixed": 1}` |
| qwen3 | 1.250 | 12 | 6 | 3.690 | 0.678 | 0.795 | 0.027 | 2.826 | 0.291 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 8, "old_suppress_new_stable_anchor_weak": 1, "old_suppress_new_unstable": 1, "threshold_shift_without_suppression": 1, "weak_or_mixed": 1}` |
| glm4 | 0.000 | 12 | 6 | 0.174 | 0.418 | 0.338 | 0.193 | -0.514 | 0.163 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 6, "weak_or_mixed": 3}` |
| glm4 | 0.250 | 12 | 6 | 0.424 | 0.415 | 0.406 | 0.142 | -0.305 | 0.189 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 3, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 4, "weak_or_mixed": 3}` |
| glm4 | 0.500 | 12 | 6 | 0.678 | 0.421 | 0.470 | 0.105 | -0.077 | 0.195 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 3, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 4, "weak_or_mixed": 3}` |
| glm4 | 0.750 | 12 | 6 | 0.931 | 0.418 | 0.535 | 0.083 | 0.134 | 0.200 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 4, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 3, "weak_or_mixed": 3}` |
| glm4 | 1.000 | 12 | 6 | 1.146 | 0.419 | 0.570 | 0.068 | 0.339 | 0.208 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 5, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 2, "weak_or_mixed": 3}` |
| glm4 | 1.250 | 12 | 6 | 1.355 | 0.420 | 0.604 | 0.059 | 0.522 | 0.215 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 5, "old_suppress_new_stable_anchor_weak": 2, "old_suppress_new_unstable": 2, "weak_or_mixed": 3}` |
| deepseek7b | 0.000 | 12 | 6 | 0.702 | -0.424 | 0.265 | 0.427 | 0.794 | 0.011 | 0.000 | `{"threshold_shift_without_suppression": 2, "weak_or_mixed": 10}` |
| deepseek7b | 0.250 | 12 | 6 | 1.292 | -0.451 | 0.355 | 0.306 | 1.128 | 0.014 | 0.000 | `{"threshold_shift_without_suppression": 4, "weak_or_mixed": 8}` |
| deepseek7b | 0.500 | 12 | 6 | 1.859 | -0.474 | 0.495 | 0.198 | 1.450 | 0.015 | 0.000 | `{"threshold_shift_without_suppression": 6, "weak_or_mixed": 6}` |
| deepseek7b | 0.750 | 12 | 6 | 2.436 | -0.501 | 0.599 | 0.143 | 1.749 | 0.012 | 0.000 | `{"threshold_shift_without_suppression": 9, "weak_or_mixed": 3}` |
| deepseek7b | 1.000 | 12 | 6 | 3.016 | -0.530 | 0.671 | 0.116 | 2.078 | 0.011 | 0.000 | `{"threshold_shift_without_suppression": 9, "weak_or_mixed": 3}` |
| deepseek7b | 1.250 | 12 | 6 | 3.591 | -0.556 | 0.727 | 0.100 | 2.399 | 0.010 | 0.000 | `{"threshold_shift_without_suppression": 9, "weak_or_mixed": 3}` |

## Best Alpha Triplets

| model | case | route | best alpha | target gain | old suppress | new rate | anchor | score | label | new reduction vs a0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.250 | 0.938 | 1.206 | 0.264 | 3.438 | 0.986 | `old_suppress_new_unstable` | 0.111 |
| qwen3 | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.750 | 0.875 | 0.943 | 0.056 | 5.875 | 0.887 | `old_suppress_new_stable_anchor_ok` | 0.242 |
| qwen3 | `p765_0058_commonsense_statement_object:chair:edible` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.500 | 1.500 | 0.964 | 0.037 | 4.375 | 0.740 | `old_suppress_new_stable_anchor_ok` | 0.341 |
| qwen3 | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.250 | 0.938 | 0.870 | 0.138 | 2.062 | 0.666 | `old_suppress_new_unstable` | 0.168 |
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.750 | 1.062 | 0.700 | 0.048 | 6.312 | 0.652 | `old_suppress_new_stable_anchor_ok` | 0.429 |
| qwen3 | `p765_0006_commonsense_statement_fruit:apple:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.750 | 0.938 | 0.704 | 0.073 | 1.938 | 0.495 | `old_suppress_new_stable_anchor_ok` | 0.523 |
| qwen3 | `p765_0002_commonsense_statement_fruit:apple:category` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 1.250 | 0.562 | 0.990 | 0.161 | 0.188 | 0.461 | `old_suppress_new_unstable` | 0.222 |
| qwen3 | `p765_0056_commonsense_statement_object:chair:category` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.750 | 2.812 | 0.983 | 0.009 | 0.812 | 0.406 | `old_suppress_new_stable_anchor_ok` | 0.363 |
| qwen3 | `p765_0058_commonsense_statement_object:chair:edible` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.500 | 0.875 | 0.748 | 0.306 | 0.500 | 0.348 | `old_suppress_new_unstable` | 0.270 |
| qwen3 | `p765_0056_commonsense_statement_object:chair:category` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.000 | 1.031 | 0.601 | 0.100 | -6.844 | 0.265 | `old_suppress_new_unstable` | 0.000 |
| qwen3 | `p765_0006_commonsense_statement_fruit:apple:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.500 | 0.500 | 0.070 | 0.225 | 3.000 | 0.020 | `weak_or_mixed` | 0.245 |
| qwen3 | `p765_0002_commonsense_statement_fruit:apple:category` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 0.000 | 0.312 | -0.029 | 0.368 | -1.438 | 0.000 | `weak_or_mixed` | 0.000 |
| glm4 | `p765_0055_commonsense_question_object:chair:category` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.250 | 0.875 | 0.756 | 0.066 | -1.438 | 0.486 | `old_suppress_new_stable_anchor_weak` | 0.173 |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | 0.344 | 0.790 | 0.242 | 1.531 | 0.462 | `old_suppress_new_unstable` | 0.000 |
| glm4 | `p765_0057_commonsense_question_object:chair:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.250 | 0.992 | 0.498 | 0.089 | 1.242 | 0.424 | `old_suppress_new_stable_anchor_ok` | 0.023 |
| glm4 | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 1.250 | 0.531 | 0.519 | 0.000 | 0.969 | 0.393 | `old_suppress_new_stable_anchor_ok` | 0.054 |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 0.500 | 2.062 | 0.661 | 0.000 | 0.688 | 0.371 | `old_suppress_new_stable_anchor_ok` | 0.033 |
| glm4 | `p765_0057_commonsense_question_object:chair:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 0.250 | 0.852 | 0.562 | 0.097 | -1.336 | 0.302 | `old_suppress_new_stable_anchor_weak` | 0.117 |
| glm4 | `p765_0056_commonsense_statement_object:chair:category` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.250 | 0.547 | 0.383 | 0.076 | -1.078 | 0.180 | `old_suppress_new_stable_anchor_weak` | 0.154 |
| glm4 | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.250 | 2.438 | 0.281 | 0.000 | 0.562 | 0.132 | `old_suppress_new_stable_anchor_ok` | 0.027 |
| glm4 | `p765_0024_commonsense_statement_animal:cat:grows_on_tree` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 1.000 | 2.266 | 0.138 | 0.000 | 0.703 | 0.072 | `weak_or_mixed` | 0.127 |
| glm4 | `p765_0055_commonsense_question_object:chair:category` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 1.250 | -0.266 | 0.475 | 0.112 | -0.016 | 0.067 | `old_suppress_new_unstable` | 0.382 |
| glm4 | `p765_0024_commonsense_statement_animal:cat:grows_on_tree` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.750 | 0.891 | 0.031 | 0.023 | 0.516 | 0.021 | `weak_or_mixed` | 0.052 |
| glm4 | `p765_0056_commonsense_statement_object:chair:category` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | -0.656 | -0.039 | 0.464 | -0.906 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.500 | 2.719 | 0.179 | 0.159 | 2.594 | 0.076 | `weak_or_mixed` | 0.004 |
| deepseek7b | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.500 | 1.344 | 0.098 | 0.000 | 0.969 | 0.064 | `weak_or_mixed` | 0.233 |
| deepseek7b | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 2.500 | 0.167 | 0.206 | 0.250 | 0.047 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0103_commonsense_question_abstract:justice:category` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.250 | -0.402 | 0.035 | 0.512 | -3.996 | 0.002 | `weak_or_mixed` | 0.343 |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 0.938 | -1.100 | 0.501 | 1.438 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.000 | -0.094 | -0.551 | 0.574 | 1.094 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0033_commonsense_question_animal:dog:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 0.906 | -0.133 | 0.458 | 3.344 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0033_commonsense_question_animal:dog:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.000 | 0.281 | -0.599 | 0.467 | 1.156 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0052_commonsense_statement_plant:wheat:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.000 | 1.438 | -0.854 | 0.156 | 0.062 | 0.000 | `threshold_shift_without_suppression` | 0.000 |
| deepseek7b | `p765_0101_commonsense_question_abstract:time:grows_on_tree` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | -0.344 | -0.212 | 0.635 | 0.719 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0101_commonsense_question_abstract:time:grows_on_tree` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 0.000 | 0.078 | -0.827 | 0.624 | 0.953 | 0.000 | `weak_or_mixed` | 0.000 |
| deepseek7b | `p765_0103_commonsense_question_abstract:justice:category` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 3.301 | -1.531 | 0.249 | 3.113 | 0.000 | `threshold_shift_without_suppression` | 0.000 |
