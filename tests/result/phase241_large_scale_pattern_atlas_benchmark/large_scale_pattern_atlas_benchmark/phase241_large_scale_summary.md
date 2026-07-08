# Phase241 Large-Scale Pattern Atlas Benchmark

case_count: 288
behavior_rows: 5184
readout_rows: 5184
negative_rows: 4223
mean_score: 0.5386
semantic_match_rate: 0.7355
protocol_match_rate: 0.1854
negative_rate: 0.8146

## Negative Categories

- rollout_negative: 1863
- semantic_failure: 1371
- closure_negative: 398
- readout_negative: 363
- protocol_negative: 228

## Top Negative Modes

| model | family | mode | rows | score | negative | protocol | winners |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| deepseek7b | content_knowledge | location_fact | 24 | 0.1125 | 1.0 | 0.0 | {'comma_repeat': 12, 'period_stop': 8, 'the_continuation': 4} |
| qwen3 | content_knowledge | causal_fact | 24 | 0.1167 | 1.0 | 0.0 | {'answer_boundary': 2, 'be_continuation': 1, 'comma_repeat': 7, 'newline_boundary': 4, 'period_stop': 8, 'the_continuation': 2} |
| glm4 | content_knowledge | location_fact | 24 | 0.1167 | 1.0 | 0.0 | {'answer_boundary': 4, 'be_continuation': 4, 'newline_boundary': 1, 'period_stop': 4, 'the_continuation': 11} |
| qwen3 | content_knowledge | location_fact | 24 | 0.125 | 1.0 | 0.0 | {'answer_boundary': 2, 'be_continuation': 8, 'comma_repeat': 4, 'period_stop': 8, 'the_continuation': 2} |
| deepseek7b | content_knowledge | causal_fact | 24 | 0.125 | 1.0 | 0.0 | {'answer_boundary': 3, 'be_continuation': 1, 'comma_repeat': 9, 'period_stop': 7, 'the_continuation': 4} |
| glm4 | language_action | classify | 24 | 0.1917 | 1.0 | 0.0 | {'answer_boundary': 2, 'newline_boundary': 3, 'period_stop': 3, 'the_continuation': 16} |
| qwen3 | language_action | classify | 24 | 0.2375 | 1.0 | 0.0 | {'answer_boundary': 3, 'comma_repeat': 4, 'period_stop': 16, 'the_continuation': 1} |
| deepseek7b | readout_competition | be_continuation | 24 | 0.2375 | 1.0 | 0.0 | {'be_continuation': 1, 'comma_repeat': 19, 'period_stop': 1, 'the_continuation': 3} |
| glm4 | closure | done_state_stable | 24 | 0.2563 | 1.0 | 0.0 | {'answer_boundary': 3, 'be_continuation': 1, 'comma_repeat': 2, 'for_continuation': 1, 'newline_boundary': 1, 'period_stop': 4, 'the_continuation': 12} |
| glm4 | content_knowledge | causal_fact | 24 | 0.2604 | 1.0 | 0.0 | {'answer_boundary': 4, 'comma_repeat': 4, 'newline_boundary': 3, 'period_stop': 3, 'the_continuation': 10} |
| deepseek7b | language_action | classify | 24 | 0.3188 | 1.0 | 0.0 | {'comma_repeat': 18, 'period_stop': 2, 'the_continuation': 4} |
| qwen3 | cross_lingual | EN_to_FR | 24 | 0.3333 | 1.0 | 0.0 | {'answer_boundary': 2, 'comma_repeat': 2, 'newline_boundary': 4, 'period_stop': 8, 'the_continuation': 8} |
| deepseek7b | reasoning_constraint | double_negation | 24 | 0.3333 | 1.0 | 0.0 | {'be_continuation': 4, 'comma_repeat': 9, 'newline_boundary': 1, 'period_stop': 2, 'the_continuation': 8} |
| qwen3 | closure | done_state_stable | 24 | 0.3417 | 1.0 | 0.0 | {'answer_boundary': 3, 'be_continuation': 3, 'comma_repeat': 1, 'for_continuation': 1, 'newline_boundary': 4, 'period_stop': 6, 'the_continuation': 6} |
| deepseek7b | output_protocol | one_word | 24 | 0.3417 | 1.0 | 0.0 | {'answer_boundary': 1, 'be_continuation': 1, 'comma_repeat': 6, 'for_continuation': 2, 'period_stop': 6, 'the_continuation': 8} |
| deepseek7b | state_drift | boundary_takeover | 24 | 0.3458 | 1.0 | 0.0 | {'comma_repeat': 16, 'the_continuation': 8} |
| qwen3 | output_protocol | one_word | 24 | 0.3521 | 1.0 | 0.0 | {'answer_boundary': 3, 'be_continuation': 1, 'comma_repeat': 5, 'for_continuation': 1, 'newline_boundary': 4, 'period_stop': 3, 'the_continuation': 7} |
| qwen3 | closure | boundary_stable | 24 | 0.3521 | 1.0 | 0.0 | {'answer_boundary': 3, 'comma_repeat': 7, 'for_continuation': 1, 'newline_boundary': 2, 'period_stop': 5, 'the_continuation': 6} |
| qwen3 | closure | pattern_matched | 24 | 0.3563 | 1.0 | 0.0 | {'answer_boundary': 4, 'be_continuation': 1, 'comma_repeat': 4, 'for_continuation': 1, 'newline_boundary': 4, 'period_stop': 3, 'the_continuation': 7} |
| deepseek7b | closure | boundary_stable | 24 | 0.3604 | 1.0 | 0.0 | {'comma_repeat': 7, 'for_continuation': 1, 'period_stop': 6, 'the_continuation': 10} |
| deepseek7b | cross_lingual | EN_to_FR | 24 | 0.3688 | 1.0 | 0.0 | {'comma_repeat': 7, 'period_stop': 7, 'the_continuation': 10} |
| qwen3 | content_knowledge | part_whole | 24 | 0.3875 | 1.0 | 0.0 | {'answer_boundary': 4, 'comma_repeat': 2, 'newline_boundary': 5, 'period_stop': 4, 'the_continuation': 9} |
| glm4 | output_protocol | one_word | 24 | 0.3875 | 1.0 | 0.0 | {'answer_boundary': 6, 'be_continuation': 1, 'comma_repeat': 1, 'for_continuation': 1, 'newline_boundary': 1, 'period_stop': 3, 'the_continuation': 11} |
| glm4 | closure | pattern_matched | 24 | 0.3917 | 1.0 | 0.0 | {'answer_boundary': 5, 'comma_repeat': 2, 'for_continuation': 1, 'newline_boundary': 2, 'period_stop': 2, 'the_continuation': 12} |
| glm4 | output_protocol | short_answer | 24 | 0.3958 | 1.0 | 0.0 | {'be_continuation': 4, 'the_continuation': 20} |
| glm4 | closure | boundary_stable | 24 | 0.3958 | 1.0 | 0.0 | {'answer_boundary': 4, 'be_continuation': 1, 'comma_repeat': 2, 'for_continuation': 1, 'newline_boundary': 3, 'the_continuation': 13} |
| deepseek7b | state_drift | echo_drift | 24 | 0.3958 | 1.0 | 0.0 | {'be_continuation': 1, 'comma_repeat': 12, 'newline_boundary': 3, 'the_continuation': 8} |
| deepseek7b | closure | pattern_matched | 24 | 0.3958 | 1.0 | 0.0 | {'comma_repeat': 5, 'for_continuation': 2, 'period_stop': 6, 'the_continuation': 11} |
| deepseek7b | closure | done_state_stable | 24 | 0.3958 | 1.0 | 0.0 | {'answer_boundary': 1, 'comma_repeat': 6, 'for_continuation': 3, 'newline_boundary': 1, 'period_stop': 5, 'the_continuation': 8} |
| deepseek7b | closure | eos_pressure | 24 | 0.4146 | 1.0 | 0.0 | {'comma_repeat': 7, 'period_stop': 1, 'the_continuation': 16} |

## Caution

This phase is large-scale behavior/readout mapping. It intentionally treats negative results as atlas data and does not claim mechanism closure.
