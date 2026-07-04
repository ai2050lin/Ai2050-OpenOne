# Phase 906 EOS action boundary test

## Overall

- models: qwen3, glm4, deepseek7b
- after_period_eos_top1: 0
- after_period_eos_top10: 0
- after_period_eos_top50: 0
- baseline_eos_top1: 0
- baseline_eos_top10: 0
- baseline_eos_top50: 14
- eos_forced_generation_would_stop: 68
- eos_forced_strict_clean_answer_no_protocol: 68
- mask_protocol_eos_top1: 0
- mask_protocol_plus_period_eos_top1: 0
- period_after_generated_eos: 9
- period_forced_protocol_drift: 50
- period_forced_strict_clean_answer_no_protocol: 0
- period_forced_strict_protocol_drift: 68
- rows: 68

## Model Summaries

| model | rows | base eos top50 | after period eos top50 | after period eos top1 | period strict clean | eos forced clean | mask protocol+period eos top1 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 18 | 0 | 0 | 0 | 0 | 18 | 0 | eos_forced_clean_but_not_naturally_competitive |
| glm4 | 17 | 12 | 0 | 0 | 0 | 17 | 0 | eos_forced_clean_but_not_naturally_competitive |
| deepseek7b | 33 | 2 | 0 | 0 | 0 | 33 | 0 | eos_forced_clean_but_not_naturally_competitive |
