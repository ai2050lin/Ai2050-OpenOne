# Phase241 Large-Scale Pattern Atlas Benchmark

case_count: 9
behavior_rows: 36
readout_rows: 36
negative_rows: 24
mean_score: 0.6306
semantic_match_rate: 0.7778
protocol_match_rate: 0.3333
negative_rate: 0.6667

## Negative Categories

- protocol_negative: 10
- semantic_failure: 8
- rollout_negative: 6

## Top Negative Modes

| model | family | mode | rows | score | negative | protocol | winners |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| glm4 | output_protocol | short_answer | 6 | 0.1417 | 1.0 | 0.0 | {'be_continuation': 1, 'the_continuation': 5} |
| deepseek7b | output_protocol | short_answer | 6 | 0.4667 | 1.0 | 0.0 | {'comma_repeat': 3, 'period_stop': 2, 'the_continuation': 1} |
| qwen3 | output_protocol | short_answer | 6 | 0.575 | 1.0 | 0.0 | {'comma_repeat': 1, 'period_stop': 4, 'the_continuation': 1} |
| qwen3 | content_knowledge | object_attribute | 6 | 0.8667 | 0.3333 | 0.6667 | {'answer_boundary': 3, 'comma_repeat': 1, 'period_stop': 1, 'the_continuation': 1} |
| glm4 | content_knowledge | object_attribute | 6 | 0.8667 | 0.3333 | 0.6667 | {'answer_boundary': 1, 'comma_repeat': 1, 'the_continuation': 4} |
| deepseek7b | content_knowledge | object_attribute | 6 | 0.8667 | 0.3333 | 0.6667 | {'comma_repeat': 1, 'for_continuation': 1, 'the_continuation': 4} |

## Caution

This phase is large-scale behavior/readout mapping. It intentionally treats negative results as atlas data and does not claim mechanism closure.
