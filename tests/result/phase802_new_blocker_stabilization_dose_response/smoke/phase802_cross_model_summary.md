# Phase 802 New-Blocker Stabilization Dose Response (smoke)

- Status: `complete`
- Boundary: alpha=0 is target-neutral, alpha=1 is raw route patch, alpha>1 over-injects the direct target direction.
- This phase tests whether adding controlled target-readout dose reduces new blockers while preserving old-blocker suppression.

## By Alpha

| model | alpha | rows | cases | target gain | old suppression | resolved | new rate | anchor | closure score | token gain | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 1 | 1 | -0.062 | 1.242 | 0.402 | 0.374 | 3.188 | 0.550 | 0.000 | `{"old_suppress_new_unstable": 1}` |
| qwen3 | 0.500 | 1 | 1 | 1.875 | 1.166 | 0.782 | 0.133 | 3.625 | 0.752 | 0.000 | `{"old_suppress_new_unstable": 1}` |
| qwen3 | 1.000 | 1 | 1 | 3.750 | 1.117 | 0.888 | 0.048 | 4.250 | 0.456 | 0.000 | `{"old_suppress_new_stable_anchor_ok": 1}` |
| glm4 | 0.000 | 1 | 1 | 0.344 | 0.790 | 0.481 | 0.242 | 1.531 | 0.462 | 0.000 | `{"old_suppress_new_unstable": 1}` |
| glm4 | 0.500 | 1 | 1 | 0.188 | 0.796 | 0.439 | 0.255 | 1.500 | 0.416 | 0.000 | `{"old_suppress_new_unstable": 1}` |
| glm4 | 1.000 | 1 | 1 | 0.000 | 0.784 | 0.390 | 0.301 | 1.375 | 0.338 | 0.000 | `{"old_suppress_new_unstable": 1}` |
| deepseek7b | 0.000 | 1 | 1 | 0.938 | -1.100 | 0.114 | 0.501 | 1.438 | 0.000 | 0.000 | `{"weak_or_mixed": 1}` |
| deepseek7b | 0.500 | 1 | 1 | 1.953 | -1.131 | 0.384 | 0.237 | 2.078 | 0.000 | 0.000 | `{"threshold_shift_without_suppression": 1}` |
| deepseek7b | 1.000 | 1 | 1 | 2.922 | -1.157 | 0.624 | 0.101 | 2.672 | 0.000 | 0.000 | `{"threshold_shift_without_suppression": 1}` |

## Best Alpha Triplets

| model | case | route | best alpha | target gain | old suppress | new rate | anchor | score | label | new reduction vs a0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.500 | 1.875 | 1.166 | 0.133 | 3.625 | 0.752 | `old_suppress_new_unstable` | 0.241 |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | 0.344 | 0.790 | 0.242 | 1.531 | 0.462 | `old_suppress_new_unstable` | 0.000 |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 0.000 | 0.938 | -1.100 | 0.501 | 1.438 | 0.000 | `weak_or_mixed` | 0.000 |
