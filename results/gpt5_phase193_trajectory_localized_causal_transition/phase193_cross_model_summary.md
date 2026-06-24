# Phase193 Cross-Model Summary

Objective: test whether Phase192 trajectory separation is caused by localized layer transitions.

Rows are target cases where base was wrong and repair prompt was correct. Each position uses the case-local best transition layer plus radius 1.

## Model Overview

| model | cases | target rows | layers | time min | best position | repair_gain | wrong_gain | ablation_loss | specificity | evidence update |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| qwen3 | 192 | 13 | 36 | 1.19 | prompt_last | 0.0867 | -0.0129 | 0.1377 | 6.7050 | weak_level5_candidate_for_selected_position |
| glm4 | 192 | 32 | 40 | 2.70 | query_relation | 0.1784 | 0.1328 | -0.0169 | 1.3431 | partial_transition_evidence_not_closed |
| deepseek7b | 192 | 78 | 28 | 4.49 | prompt_last | 0.0013 | -0.0263 | 0.1545 | 0.0499 | partial_transition_evidence_not_closed |

## qwen3 Positions

| position | repair_gain | wrong_gain | ablation_loss | specificity | repair positive | wrong positive | switch | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prompt_last | 0.0867 | -0.0129 | 0.1377 | 6.7050 | 0.7179 | 0.4872 | 3/39 | localized_transition_candidate |
| query_category | 0.0061 | 0.0544 | 0.1884 | 0.1126 | 0.4103 | 0.5897 | 2/39 | ablation_only |

## glm4 Positions

| position | repair_gain | wrong_gain | ablation_loss | specificity | repair positive | wrong positive | switch | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| query_relation | 0.1784 | 0.1328 | -0.0169 | 1.3431 | 0.5729 | 0.4792 | 21/96 | gain_without_ablation |
| prompt_last | 0.0195 | 0.0078 | -0.0241 | 2.4997 | 0.4479 | 0.4271 | 5/96 | weak_or_failed |

## deepseek7b Positions

| position | repair_gain | wrong_gain | ablation_loss | specificity | repair positive | wrong positive | switch | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prompt_last | 0.0013 | -0.0263 | 0.1545 | 0.0499 | 0.4316 | 0.3761 | 6/234 | ablation_only |
| query_relation | -0.0023 | -0.0486 | 0.0455 | -0.0469 | 0.5000 | 0.4274 | 5/234 | weak_or_failed |
| rule_value | -0.0029 | 0.0133 | 0.0049 | -0.2145 | 0.4651 | 0.4651 | 0/129 | weak_or_failed |

## Objective Reading

- The uploaded Phase192 interpretation is correct: trajectory signal required localized causal transition testing.
- Qwen3 prompt_last is the only position meeting the current weak localized-transition candidate rule: repair_gain positive, wrong_gain smaller/opposite, and repair ablation lowers margin.
- GLM4 query_relation has large repair_gain, but wrong_gain is also large and repair->base ablation does not lower repair margin; this is not closed causal evidence.
- DS7B does not convert its strong trajectory signal into localized transition gain. This is important because DS7B had the largest target set.
- Overall: Phase193 upgrades a narrow Qwen3 prompt_last handle, but does not close the cross-model candidate-ranking mechanism.
