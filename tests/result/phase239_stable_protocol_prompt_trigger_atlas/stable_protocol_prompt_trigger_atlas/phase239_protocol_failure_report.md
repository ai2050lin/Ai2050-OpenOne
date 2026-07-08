# Phase239 Stable Protocol Prompt Trigger Atlas

variant_rows: 264
mean_score: 0.6152
protocol_match_rate: 0.0038

## Model Summary

| model | rows | mean score | protocol match |
| --- | ---: | ---: | ---: |
| qwen3 | 88 | 0.6869 | 0.0 |
| glm4 | 88 | 0.7256 | 0.0114 |
| deepseek7b | 88 | 0.433 | 0.0 |

## Best Variants

| variant | rows | mean score | protocol match | over generation | score delta | winners |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| colon_removed | 24 | 0.725 | 0.0 | 0.9583 | 0.0125 | {'the_continuation': 12, 'be_continuation': 12} |
| full | 24 | 0.7125 | 0.0 | 0.9583 | 0.0 | {'the_continuation': 16, 'newline_boundary': 8} |
| short_answer_instruction | 24 | 0.6937 | 0.0 | 0.9167 | -0.0187 | {'the_continuation': 15, 'newline_boundary': 8, 'answer_boundary': 1} |
| one_word_strict | 24 | 0.6625 | 0.0 | 0.9583 | -0.05 | {'the_continuation': 13, 'answer_boundary': 9, 'newline_boundary': 2} |
| explain_instruction | 24 | 0.6437 | 0.0 | 0.9583 | -0.0687 | {'the_continuation': 12, 'answer_boundary': 8, 'newline_boundary': 4} |
| no_answer_anchor | 24 | 0.6188 | 0.0 | 0.7917 | -0.0938 | {'the_continuation': 16, 'newline_boundary': 8} |
| newline_removed | 24 | 0.575 | 0.0 | 0.7917 | -0.1375 | {'the_continuation': 16, 'answer_boundary': 7, 'newline_boundary': 1} |
| period_forced | 24 | 0.5625 | 0.0 | 0.75 | -0.15 | {'the_continuation': 14, 'newline_boundary': 7, 'period_stop': 2, 'answer_boundary': 1} |
| one_word_no_explain | 24 | 0.5479 | 0.0417 | 0.75 | -0.1646 | {'the_continuation': 12, 'newline_boundary': 8, 'answer_boundary': 3, 'be_continuation': 1} |
| target_seeded | 24 | 0.5312 | 0.0 | 0.7083 | -0.1812 | {'newline_boundary': 16, 'period_stop': 8} |
| strong_answer_anchor | 24 | 0.4937 | 0.0 | 0.7083 | -0.2188 | {'the_continuation': 12, 'newline_boundary': 7, 'answer_boundary': 3, 'for_continuation': 2} |
