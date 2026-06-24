# Phase192 Cross-Model Summary

Objective: after static patch routes failed, measure base/repair/wrong forward trajectories and locate where candidate margins separate across layers.

The generator produced 192 valid cases under the confirm settings. Rows below are target cases where base was wrong and repair prompt was correct.

## Model Overview

| model | cases | target rows | layers | time min | best position | strict flip | over base | best rb | leak | evidence update |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| qwen3 | 192 | 13 | 36 | 0.84 | query_category | 13/13 | 13/13 | 1.4074 | 4.7206 | trajectory_level4_upgrade_candidate |
| glm4 | 192 | 32 | 40 | 1.35 | prompt_last | 31/32 | 32/32 | 1.2141 | 2.2762 | trajectory_signal_with_control_pollution |
| deepseek7b | 192 | 78 | 28 | 1.14 | query_category | 72/78 | 78/78 | 1.1086 | 22.3664 | trajectory_level4_upgrade_candidate |

## qwen3 Positions

| position | n | strict flip | over base | best rb | best transition | leak | mean strict layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query_category | 13 | 13/13 | 13/13 | 1.4074 | 0.2772 | 4.7206 | 6.5385 |
| prompt_last | 13 | 12/13 | 13/13 | 2.0240 | 0.4899 | 0.9403 | 22.8333 |
| rule_relation | 13 | 10/13 | 13/13 | 0.9373 | 0.5794 | 2.9455 | 11.7000 |
| query_relation | 13 | 9/13 | 13/13 | 0.8441 | 0.5841 | 124.1764 | 9.8889 |
| rule_value | 12 | 8/12 | 12/12 | 0.8266 | 0.5639 | 4.1927 | 10.0000 |

Low final-control-leak positions: prompt_last


## glm4 Positions

| position | n | strict flip | over base | best rb | best transition | leak | mean strict layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt_last | 32 | 31/32 | 32/32 | 1.2141 | 0.4884 | 2.2762 | 10.5484 |
| query_category | 32 | 28/32 | 32/32 | 2.0053 | 0.3581 | 5.4388 | 8.3214 |
| query_relation | 32 | 24/32 | 32/32 | 1.0418 | 0.7792 | 1.3636 | 17.6250 |
| rule_value | 25 | 17/25 | 25/25 | 1.0409 | 0.6921 | 2.0621 | 9.4706 |
| rule_relation | 32 | 19/32 | 32/32 | 1.3662 | 0.6874 | 2.6641 | 5.6316 |

Low final-control-leak positions: none


## deepseek7b Positions

| position | n | strict flip | over base | best rb | best transition | leak | mean strict layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query_category | 78 | 72/78 | 78/78 | 1.1086 | 0.2510 | 22.3664 | 9.4861 |
| rule_relation | 78 | 62/78 | 78/78 | 1.0684 | 0.5011 | 7.0090 | 5.1290 |
| query_relation | 78 | 55/78 | 77/78 | 0.7596 | 0.4927 | 2.4111 | 9.1273 |
| rule_value | 43 | 28/43 | 43/43 | 0.8831 | 0.5636 | 1.1860 | 13.8929 |
| prompt_last | 78 | 48/78 | 75/78 | 0.8172 | 0.3891 | 0.6719 | 16.6250 |

Low final-control-leak positions: prompt_last

## Objective Reading

- Phase191's interpretation is correct: static node/channel patch routes should be downgraded, and the next object is the forward trajectory.
- Phase192 shows strong repair-vs-base separation in the natural forward trajectory: repair_over_base is near-total for most positions in all three models.
- This is not hidden causal repair yet. The wrong trajectory often also diverges from base, so control leak remains large at many positions.
- The cleanest observed low-leak handle is Qwen3 prompt_last and DS7B prompt_last. Query/category positions often show strong trajectory signal but heavy control leak.
- Evidence should be updated from static weak Level4 to trajectory-Level4 candidate, not Level5 repair.
