# Phase 201 stop/prose component atlas

## Post-answer metric summary
| model | relation | pair | protocol | rows | stop margin | prose margin | echo margin |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | color | en->en | plain | 18 | 4.222 | -4.222 | -12.226 |
| qwen3 | color | en->en | short_answer | 18 | 6.896 | -6.896 | -14.264 |
| qwen3 | color | en->en | stop_explicit | 18 | 8.694 | -8.694 | -16.500 |
| qwen3 | color | en->zh | plain | 14 | 10.013 | -10.013 | -7.777 |
| qwen3 | color | en->zh | short_answer | 14 | 7.205 | -7.205 | -13.987 |
| qwen3 | color | en->zh | stop_explicit | 14 | 8.500 | -8.500 | -15.482 |
| qwen3 | color | zh->en | plain | 14 | 4.134 | -4.134 | -11.125 |
| qwen3 | color | zh->en | short_answer | 14 | 7.205 | -7.205 | -14.009 |
| qwen3 | color | zh->en | stop_explicit | 14 | 8.509 | -8.509 | -15.482 |
| qwen3 | color | zh->zh | plain | 18 | 8.622 | -8.622 | -7.733 |
| qwen3 | color | zh->zh | short_answer | 18 | 6.896 | -6.896 | -14.281 |
| qwen3 | color | zh->zh | stop_explicit | 18 | 8.729 | -8.729 | -16.469 |
| glm4 | color | en->en | plain | 18 | 2.618 | -2.618 | -7.554 |
| glm4 | color | en->en | short_answer | 18 | 2.955 | -2.955 | -6.424 |
| glm4 | color | en->en | stop_explicit | 18 | 3.535 | -3.535 | -8.542 |
| glm4 | color | zh->en | plain | 14 | 0.917 | -0.917 | -6.637 |
| glm4 | color | zh->en | short_answer | 14 | 3.004 | -3.004 | -7.339 |
| glm4 | color | zh->en | stop_explicit | 14 | 3.580 | -3.580 | -8.652 |
| glm4 | color | zh->zh | plain | 18 | 6.039 | -6.039 | -6.183 |
| glm4 | color | zh->zh | short_answer | 18 | 2.955 | -2.955 | -6.424 |
| glm4 | color | zh->zh | stop_explicit | 18 | 3.535 | -3.535 | -8.542 |
| glm4 | function | en->en | plain | 9 | -9.240 | 9.240 | 5.145 |
| glm4 | function | en->en | short_answer | 9 | 1.389 | -1.389 | -5.898 |
| glm4 | function | en->en | stop_explicit | 9 | 1.764 | -1.764 | -6.493 |
| deepseek7b | function | en->en | plain | 12 | -2.729 | 2.729 | -7.180 |
| deepseek7b | function | en->en | short_answer | 12 | -0.833 | 0.833 | -7.417 |
| deepseek7b | function | en->en | stop_explicit | 12 | -0.844 | 0.844 | -7.307 |

## Top causal candidates
| model | type | relation | protocol | L | c | condition | stop Δ | prose Δ | echo Δ | score |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| glm4 | anti_prose_candidate | color | plain | 35 | 1018 | boost | 2.048 | -2.048 | -1.184 | 4.095 |
| glm4 | anti_prose_candidate | color | plain | 23 | 616 | ablate | 2.039 | -2.039 | -1.182 | 4.078 |
| glm4 | anti_prose_candidate | color | plain | 29 | 7118 | ablate | 2.034 | -2.034 | -1.186 | 4.069 |
| glm4 | anti_prose_candidate | color | plain | 23 | 616 | boost | 2.030 | -2.030 | -1.177 | 4.060 |
| glm4 | anti_prose_candidate | color | plain | 35 | 1018 | ablate | 2.030 | -2.030 | -1.182 | 4.060 |
| glm4 | anti_prose_candidate | color | plain | 29 | 7118 | boost | 2.025 | -2.025 | -1.177 | 4.051 |
| glm4 | stop_candidate | color | plain | 35 | 1018 | boost | 2.048 | -2.048 | -1.184 | 2.048 |
| glm4 | stop_candidate | color | plain | 23 | 616 | ablate | 2.039 | -2.039 | -1.182 | 2.039 |
| glm4 | stop_candidate | color | plain | 29 | 7118 | ablate | 2.034 | -2.034 | -1.186 | 2.034 |
| glm4 | stop_candidate | color | plain | 23 | 616 | boost | 2.030 | -2.030 | -1.177 | 2.030 |
| glm4 | stop_candidate | color | plain | 35 | 1018 | ablate | 2.030 | -2.030 | -1.182 | 2.030 |
| glm4 | stop_candidate | color | plain | 29 | 7118 | boost | 2.025 | -2.025 | -1.177 | 2.025 |
| glm4 | echo_suppress_candidate | color | short_answer | 35 | 755 | boost | 0.219 | -0.219 | -1.125 | 1.344 |
| glm4 | echo_suppress_candidate | color | short_answer | 35 | 755 | ablate | 0.214 | -0.214 | -1.125 | 1.339 |
| glm4 | echo_suppress_candidate | function | short_answer | 23 | 13014 | boost | 0.021 | -0.021 | -0.003 | 0.024 |
| glm4 | echo_suppress_candidate | color | stop_explicit | 23 | 3459 | ablate | 0.018 | -0.018 | -0.002 | 0.020 |
| glm4 | echo_suppress_candidate | color | stop_explicit | 23 | 3459 | boost | 0.022 | -0.022 | 0.004 | 0.018 |
| deepseek7b | echo_suppress_candidate | function | stop_explicit | 16 | 6402 | boost | 0.021 | -0.021 | 0.005 | 0.016 |
| glm4 | echo_suppress_candidate | function | short_answer | 23 | 13014 | ablate | 0.014 | -0.014 | 0.009 | 0.005 |
| deepseek7b | stop_candidate | function | short_answer | 12 | 1759 | boost | 0.000 | 0.000 | 0.047 | 0.000 |
| deepseek7b | anti_prose_candidate | function | short_answer | 12 | 1759 | boost | 0.000 | 0.000 | 0.047 | 0.000 |
| deepseek7b | echo_suppress_candidate | function | short_answer | 20 | 5477 | boost | -0.010 | 0.010 | 0.000 | 0.000 |
| deepseek7b | anti_prose_candidate | function | stop_explicit | 12 | 10417 | ablate | -0.010 | 0.010 | -0.010 | -0.010 |
| deepseek7b | anti_prose_candidate | function | stop_explicit | 12 | 10417 | boost | -0.010 | 0.010 | 0.016 | -0.010 |
| deepseek7b | anti_prose_candidate | function | short_answer | 24 | 16512 | ablate | -0.010 | 0.010 | -0.016 | -0.010 |
| deepseek7b | echo_suppress_candidate | function | stop_explicit | 16 | 18809 | ablate | -0.010 | 0.010 | 0.010 | -0.010 |
| deepseek7b | stop_candidate | function | stop_explicit | 12 | 10417 | ablate | -0.010 | 0.010 | -0.010 | -0.021 |
| deepseek7b | stop_candidate | function | stop_explicit | 12 | 10417 | boost | -0.010 | 0.010 | 0.016 | -0.021 |
| deepseek7b | stop_candidate | function | short_answer | 24 | 16512 | ablate | -0.010 | 0.010 | -0.016 | -0.021 |
| deepseek7b | echo_suppress_candidate | function | short_answer | 20 | 5477 | ablate | -0.021 | 0.021 | 0.026 | -0.026 |
| deepseek7b | echo_suppress_candidate | function | stop_explicit | 16 | 6402 | ablate | -0.010 | 0.010 | 0.026 | -0.026 |
| deepseek7b | anti_prose_candidate | function | short_answer | 12 | 1759 | ablate | -0.031 | 0.031 | 0.031 | -0.031 |
| deepseek7b | anti_prose_candidate | function | short_answer | 24 | 16512 | boost | -0.042 | 0.042 | 0.031 | -0.042 |
| deepseek7b | echo_suppress_candidate | function | stop_explicit | 16 | 18809 | boost | -0.021 | 0.021 | 0.042 | -0.042 |
| qwen3 | anti_prose_candidate | color | stop_explicit | 26 | 2192 | ablate | -0.054 | 0.054 | 0.143 | -0.054 |
| qwen3 | anti_prose_candidate | color | stop_explicit | 32 | 283 | ablate | -0.054 | 0.054 | 0.107 | -0.054 |
| qwen3 | anti_prose_candidate | color | stop_explicit | 32 | 283 | ablate | -0.062 | 0.062 | 0.107 | -0.062 |
| deepseek7b | stop_candidate | function | short_answer | 12 | 1759 | ablate | -0.031 | 0.031 | 0.031 | -0.062 |
| qwen3 | anti_prose_candidate | color | stop_explicit | 26 | 2192 | boost | -0.071 | 0.071 | 0.138 | -0.071 |
| qwen3 | anti_prose_candidate | color | stop_explicit | 32 | 283 | boost | -0.071 | 0.071 | 0.129 | -0.071 |
