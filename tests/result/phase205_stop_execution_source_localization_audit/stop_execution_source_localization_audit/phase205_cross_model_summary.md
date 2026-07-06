# Phase 205 stop-execution source localization audit

## Transition Summary
| model | mode | protocol | transition | rows | eos rank delta | stop margin delta | prose margin delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | post_answer | stop_explicit | after_period->after_continue1 | 36 | 54336.19 | 24.55 | -24.55 |
| qwen3 | post_answer | stop_explicit | before_period->after_period | 36 | -13767.58 | -27.01 | 27.01 |
| glm4 | post_answer | stop_explicit | after_period->after_continue1 | 36 | 9168.36 | -13.60 | 13.60 |
| glm4 | post_answer | stop_explicit | before_period->after_period | 36 | -1086.58 | 7.09 | -7.09 |
| deepseek7b | natural | stop_explicit | after_period->after_continue1 | 6 | 1785.00 | 3.69 | -3.69 |
| deepseek7b | natural | stop_explicit | before_period->after_period | 6 | -6614.83 | -14.25 | 14.25 |
| deepseek7b | post_answer | plain | after_period->after_continue1 | 8 | 15036.75 | 3.01 | -3.01 |
| deepseek7b | post_answer | plain | before_period->after_period | 8 | -3155.38 | -8.60 | 8.60 |
| deepseek7b | post_answer | short_answer | after_period->after_continue1 | 12 | 3942.17 | 1.42 | -1.42 |
| deepseek7b | post_answer | short_answer | before_period->after_period | 12 | -5185.83 | -13.16 | 13.16 |
| deepseek7b | post_answer | stop_explicit | after_period->after_continue1 | 10 | 873.20 | 4.72 | -4.72 |
| deepseek7b | post_answer | stop_explicit | before_period->after_period | 10 | -8682.20 | -14.32 | 14.32 |

## Top MLP Delta Channels
| model | transition | layer | rank | channel | mean abs delta | samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | after_period->after_continue1 | 12 | 1 | 22 | 1.0402 | 36 |
| qwen3 | after_period->after_continue1 | 12 | 2 | 1443 | 1.0138 | 36 |
| qwen3 | after_period->after_continue1 | 12 | 3 | 4390 | 1.0043 | 36 |
| qwen3 | after_period->after_continue1 | 18 | 1 | 5159 | 2.0349 | 36 |
| qwen3 | after_period->after_continue1 | 18 | 2 | 855 | 2.0227 | 36 |
| qwen3 | after_period->after_continue1 | 18 | 3 | 5457 | 1.9590 | 36 |
| qwen3 | after_period->after_continue1 | 23 | 1 | 1283 | 8.3225 | 36 |
| qwen3 | after_period->after_continue1 | 23 | 2 | 131 | 5.6542 | 36 |
| qwen3 | after_period->after_continue1 | 23 | 3 | 3076 | 4.0153 | 36 |
| qwen3 | after_period->after_continue1 | 28 | 1 | 205 | 9.5399 | 36 |
| qwen3 | after_period->after_continue1 | 28 | 2 | 1266 | 6.4377 | 36 |
| qwen3 | after_period->after_continue1 | 28 | 3 | 1313 | 5.8536 | 36 |
| qwen3 | after_period->after_continue1 | 33 | 1 | 1986 | 13.5997 | 36 |
| qwen3 | after_period->after_continue1 | 33 | 2 | 9 | 8.4180 | 36 |
| qwen3 | after_period->after_continue1 | 33 | 3 | 980 | 8.2207 | 36 |
| qwen3 | before_period->after_period | 12 | 1 | 3986 | 2.1756 | 36 |
| qwen3 | before_period->after_period | 12 | 2 | 8054 | 2.1499 | 36 |
| qwen3 | before_period->after_period | 12 | 3 | 7172 | 1.8029 | 36 |
| qwen3 | before_period->after_period | 18 | 1 | 2781 | 2.7322 | 36 |
| qwen3 | before_period->after_period | 18 | 2 | 625 | 1.9097 | 36 |
| qwen3 | before_period->after_period | 18 | 3 | 3693 | 1.8298 | 36 |
| qwen3 | before_period->after_period | 23 | 1 | 1283 | 9.0027 | 36 |
| qwen3 | before_period->after_period | 23 | 2 | 6123 | 6.2510 | 36 |
| qwen3 | before_period->after_period | 23 | 3 | 596 | 4.8503 | 36 |
| qwen3 | before_period->after_period | 28 | 1 | 27 | 8.0760 | 36 |
| qwen3 | before_period->after_period | 28 | 2 | 729 | 7.6361 | 36 |
| qwen3 | before_period->after_period | 28 | 3 | 816 | 7.5478 | 36 |
| qwen3 | before_period->after_period | 33 | 1 | 1986 | 12.8134 | 36 |
| qwen3 | before_period->after_period | 33 | 2 | 521 | 9.8952 | 36 |
| qwen3 | before_period->after_period | 33 | 3 | 73 | 9.3235 | 36 |
| glm4 | after_period->after_continue1 | 14 | 1 | 2167 | 0.3039 | 36 |
| glm4 | after_period->after_continue1 | 14 | 2 | 2036 | 0.2005 | 36 |
| glm4 | after_period->after_continue1 | 14 | 3 | 11695 | 0.1952 | 36 |
| glm4 | after_period->after_continue1 | 20 | 1 | 1865 | 1.1630 | 36 |
| glm4 | after_period->after_continue1 | 20 | 2 | 4684 | 0.9485 | 36 |
| glm4 | after_period->after_continue1 | 20 | 3 | 6466 | 0.6120 | 36 |
| glm4 | after_period->after_continue1 | 25 | 1 | 9938 | 3.7449 | 36 |
| glm4 | after_period->after_continue1 | 25 | 2 | 11266 | 2.9128 | 36 |
| glm4 | after_period->after_continue1 | 25 | 3 | 11326 | 2.0641 | 36 |
| glm4 | after_period->after_continue1 | 31 | 1 | 11903 | 5.3624 | 36 |
| glm4 | after_period->after_continue1 | 31 | 2 | 13362 | 4.2661 | 36 |
| glm4 | after_period->after_continue1 | 31 | 3 | 2853 | 3.7295 | 36 |
| glm4 | after_period->after_continue1 | 37 | 1 | 8035 | 10.8212 | 36 |
| glm4 | after_period->after_continue1 | 37 | 2 | 12922 | 5.7135 | 36 |
| glm4 | after_period->after_continue1 | 37 | 3 | 9954 | 5.4605 | 36 |
| glm4 | before_period->after_period | 14 | 1 | 8002 | 0.2010 | 36 |
| glm4 | before_period->after_period | 14 | 2 | 9605 | 0.1546 | 36 |
| glm4 | before_period->after_period | 14 | 3 | 3748 | 0.1472 | 36 |
| glm4 | before_period->after_period | 20 | 1 | 1792 | 0.5322 | 36 |
| glm4 | before_period->after_period | 20 | 2 | 4887 | 0.4097 | 36 |
| glm4 | before_period->after_period | 20 | 3 | 10599 | 0.3932 | 36 |
| glm4 | before_period->after_period | 25 | 1 | 9938 | 3.1820 | 36 |
| glm4 | before_period->after_period | 25 | 2 | 9522 | 1.7794 | 36 |
| glm4 | before_period->after_period | 25 | 3 | 11326 | 1.7763 | 36 |
| glm4 | before_period->after_period | 31 | 1 | 11316 | 3.0367 | 36 |
| glm4 | before_period->after_period | 31 | 2 | 5278 | 2.2435 | 36 |
| glm4 | before_period->after_period | 31 | 3 | 13362 | 2.2079 | 36 |
| glm4 | before_period->after_period | 37 | 1 | 9954 | 6.1724 | 36 |
| glm4 | before_period->after_period | 37 | 2 | 8881 | 5.9851 | 36 |
| glm4 | before_period->after_period | 37 | 3 | 8035 | 4.4740 | 36 |
| deepseek7b | after_period->after_continue1 | 9 | 1 | 271 | 6.5470 | 36 |
| deepseek7b | after_period->after_continue1 | 9 | 2 | 8763 | 4.9260 | 36 |
| deepseek7b | after_period->after_continue1 | 9 | 3 | 2337 | 3.8318 | 36 |
| deepseek7b | after_period->after_continue1 | 14 | 1 | 11019 | 2.6426 | 36 |
| deepseek7b | after_period->after_continue1 | 14 | 2 | 2431 | 2.2182 | 36 |
| deepseek7b | after_period->after_continue1 | 14 | 3 | 2429 | 2.0773 | 36 |
| deepseek7b | after_period->after_continue1 | 18 | 1 | 17901 | 4.4960 | 36 |
| deepseek7b | after_period->after_continue1 | 18 | 2 | 4662 | 3.5930 | 36 |
| deepseek7b | after_period->after_continue1 | 18 | 3 | 11010 | 3.5362 | 36 |
| deepseek7b | after_period->after_continue1 | 22 | 1 | 15320 | 25.5658 | 36 |
| deepseek7b | after_period->after_continue1 | 22 | 2 | 6112 | 10.6753 | 36 |
| deepseek7b | after_period->after_continue1 | 22 | 3 | 6264 | 10.6500 | 36 |
| deepseek7b | after_period->after_continue1 | 26 | 1 | 264 | 44.5643 | 36 |
| deepseek7b | after_period->after_continue1 | 26 | 2 | 9289 | 22.9893 | 36 |
| deepseek7b | after_period->after_continue1 | 26 | 3 | 16674 | 21.8225 | 36 |
| deepseek7b | before_period->after_period | 9 | 1 | 271 | 6.8619 | 36 |
| deepseek7b | before_period->after_period | 9 | 2 | 8763 | 5.0323 | 36 |
| deepseek7b | before_period->after_period | 9 | 3 | 9525 | 4.6768 | 36 |
| deepseek7b | before_period->after_period | 14 | 1 | 7805 | 2.4906 | 36 |
| deepseek7b | before_period->after_period | 14 | 2 | 6468 | 2.1284 | 36 |
| deepseek7b | before_period->after_period | 14 | 3 | 2429 | 2.1063 | 36 |
| deepseek7b | before_period->after_period | 18 | 1 | 17901 | 6.9396 | 36 |
| deepseek7b | before_period->after_period | 18 | 2 | 4662 | 3.6155 | 36 |
| deepseek7b | before_period->after_period | 18 | 3 | 13890 | 3.3123 | 36 |
| deepseek7b | before_period->after_period | 22 | 1 | 15320 | 24.9817 | 36 |
| deepseek7b | before_period->after_period | 22 | 2 | 6264 | 14.7585 | 36 |
| deepseek7b | before_period->after_period | 22 | 3 | 1969 | 11.2874 | 36 |
| deepseek7b | before_period->after_period | 26 | 1 | 264 | 44.5378 | 36 |
| deepseek7b | before_period->after_period | 26 | 2 | 9289 | 23.4050 | 36 |
| deepseek7b | before_period->after_period | 26 | 3 | 2378 | 20.6140 | 36 |
