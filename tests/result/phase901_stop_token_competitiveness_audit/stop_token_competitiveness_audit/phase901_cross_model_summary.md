# Phase 901 stop token competitiveness audit

## Overall

- models: qwen3, glm4, deepseek7b
- rows: 68
- answer_prefix_seen: 68
- stop_top10: 61
- stop_top50: 68
- stop_top100: 68
- eos_top100: 22
- period_top50: 68
- mean_stop_rank: 6.5588235294117645
- median_stop_rank: 6.0
- mean_eos_rank: 9460.14705882353
- median_eos_rank: 147.0
- mean_period_rank: 6.5588235294117645
- median_period_rank: 6.0
- mean_protocol_rank: 1.2794117647058822
- median_protocol_rank: 1.0
- next_top_tokens: {'\n': 34, '\n\n': 6, ' Kingdom': 1, ',': 20, '.': 7}

## Model summaries

| model | rows | stop top10 | stop top50 | stop top100 | eos top100 | period top50 | median stop rank | median eos rank | median protocol rank | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 18 | 18 | 18 | 18 | 0 | 18 | 3.0 | 29326.5 | 2.0 | stop_token_competitive_in_some_rows |
| glm4 | 17 | 12 | 17 | 17 | 12 | 17 | 7.0 | 36.0 | 1.0 | stop_token_competitive_in_some_rows |
| deepseek7b | 33 | 31 | 33 | 33 | 10 | 33 | 8.0 | 147.0 | 1.0 | stop_token_competitive_in_some_rows |
