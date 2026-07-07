# Phase 227 token trigger and gate/up decomposition

activation_rows: 10368
hidden_rows: 4752
readout_rows: 1188
channel_score_rows: 1728

## Component summary

| spec | group | variant | step | layer | component | K | axis | delta | success closer |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| qwen3_explain_l29_natural_trigger | drift | short_answer_instruction | 3 | 29 | up | 4 | 3.3289 | 29.4829 | 0 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 29 | up | 4 | -4.3436 | 21.8105 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 3 | 29 | up | 4 | -5.7025 | 20.4515 | 0 |
| qwen3_explain_l29_natural_trigger | success | no_instruction | 3 | 29 | up | 4 | 15.2370 | 18.4728 | 4 |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 29 | up | 4 | 7.8041 | 12.3730 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 3 | 29 | up | 4 | -37.3666 | -11.2126 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | up | 4 | 8.9051 | 9.8450 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_13_ short | 3 | 29 | up | 4 | -17.4864 | 8.6676 | 0 |
| qwen3_explain_l29_natural_trigger | drift | short_answer_instruction | 3 | 29 | up | 16 | 1.6833 | 8.2967 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_6_ with | 3 | 29 | up | 4 | -18.2470 | 7.9071 | 0 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 1 | 29 | up | 4 | -8.3237 | -7.3837 | 2 |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 29 | gate | 16 | 0.9129 | -5.8790 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ cherry | 3 | 29 | up | 4 | -8.9765 | -5.7407 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ cherry | 2 | 29 | gate | 64 | -5.5385 | -5.7172 | 0 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 29 | up | 16 | -0.9489 | 5.6645 | 4 |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 29 | up | 16 | 4.9092 | 5.6335 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_4_?\n | 3 | 29 | up | 4 | 2.3637 | 5.5995 | 4 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_12_ one | 3 | 29 | up | 4 | -20.7668 | 5.3873 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | up | 16 | 5.1006 | 5.3605 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ apple | 1 | 29 | up | 4 | -6.2288 | -5.2889 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ cardinal | 1 | 29 | up | 4 | -6.2288 | -5.2889 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 3 | 29 | up | 16 | -1.4565 | 5.1569 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ cherry | 1 | 29 | gate | 16 | 1.6496 | -5.1423 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ car | 1 | 29 | gate | 16 | 1.6496 | -5.1423 | 2 |
| qwen3_explain_l29_natural_trigger | success | short_answer_instruction | 3 | 29 | up | 4 | 1.6281 | 4.8639 | 4 |
| qwen3_explain_l29_natural_trigger | success | no_instruction | 3 | 29 | up | 16 | 5.0391 | 4.4667 | 4 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ car | 3 | 29 | up | 4 | 1.1687 | 4.4045 | 2 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_8_ answer | 1 | 29 | up | 4 | -5.2388 | -4.2989 | 2 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 1 | 29 | up | 4 | -8.8333 | -4.2644 | 0 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 1 | 29 | gate | 16 | 11.0250 | 4.2331 | 2 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_5_Answer | 3 | 29 | up | 4 | -30.2820 | -4.1279 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_9_ first | 3 | 29 | up | 4 | -22.0961 | 4.0580 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ cherry | 2 | 29 | up | 64 | -2.9384 | -3.9504 | 0 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 2 | 29 | gate | 64 | -0.4177 | 3.8449 | 0 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 1 | 29 | gate | 16 | 6.4716 | 3.7437 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_4_?\n | 3 | 29 | up | 4 | -22.5750 | 3.5790 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ car | 2 | 29 | product | 4 | 4.5485 | 3.5394 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ car | 2 | 29 | recomputed_product | 4 | 4.5397 | 3.5307 | 2 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_10_, | 3 | 29 | up | 4 | -22.9405 | 3.2136 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_11_ then | 1 | 29 | gate | 16 | 3.6617 | -3.1301 | 2 |
| qwen3_explain_l29_natural_trigger | success | because_removed | 3 | 29 | up | 4 | -0.1801 | 3.0557 | 4 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 3 | 29 | up | 16 | -9.6660 | -3.0526 | 0 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 2 | 29 | gate | 64 | 3.0826 | 2.9039 | 4 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_6_ with | 1 | 29 | gate | 16 | 3.9133 | -2.8786 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_2_ is | 1 | 29 | up | 4 | -7.4343 | -2.8654 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | gate | 4 | 1.5217 | 2.8172 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_10_, | 3 | 29 | up | 4 | -0.4203 | 2.8155 | 4 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_2_ is | 3 | 29 | up | 4 | -0.5055 | 2.7304 | 4 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_7_ the | 3 | 29 | up | 4 | -28.8081 | -2.6540 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ apple | 2 | 29 | gate | 64 | -1.6297 | 2.6329 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ cardinal | 2 | 29 | gate | 64 | -1.6297 | 2.6329 | 0 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 24 | up | 4 | 2.6075 | 2.6075 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_2_ is | 1 | 29 | up | 64 | -1.5060 | -2.6036 | 2 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_2_ is | 3 | 29 | up | 4 | -28.7487 | -2.5946 | 0 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_13_ short | 3 | 29 | up | 16 | -4.0700 | 2.5434 | 0 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ cherry | 1 | 29 | up | 64 | -1.3144 | -2.4119 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ car | 1 | 29 | up | 64 | -1.3144 | -2.4119 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_2_ is | 1 | 29 | gate | 16 | 9.1658 | 2.3739 | 2 |
| qwen3_explain_l29_natural_trigger | success | drop_tok_3_ car | 2 | 29 | up | 4 | 2.4571 | 2.3489 | 2 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_6_ with | 3 | 29 | up | 16 | -4.3307 | 2.2827 | 0 |

## Hidden summary

| spec | group | variant | step | layer | projection delta |
| --- | --- | --- | ---: | ---: | ---: |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 26 | 219.5280 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_3_ used | 1 | 26 | 155.8618 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 26 | 146.7012 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ cardinal | 1 | 33 | 126.9386 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 1 | 24 | -102.4149 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_6_Answer | 1 | 24 | -99.7267 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_6_Answer | 1 | 26 | -99.2561 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 24 | 97.1485 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ cardinal | 1 | 31 | 93.4874 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 33 | 88.0596 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 33 | 86.7255 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 26 | 80.3140 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 26 | 77.5213 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_3_ used | 1 | 24 | 76.1377 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_5_?\n | 1 | 26 | -72.0601 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_3_ used | 3 | 26 | 72.0182 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 31 | 71.8746 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ apple | 1 | 31 | 70.5697 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_4_ for | 1 | 26 | 70.1391 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ apple | 1 | 33 | 69.1631 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_5_?\n | 1 | 24 | -64.7225 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ cardinal | 1 | 29 | 63.7177 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_3_ used | 2 | 26 | 61.4164 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 3 | 33 | 60.7655 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 31 | 60.0024 |
| qwen3_explain_l29_natural_trigger | drift | short_answer_instruction | 3 | 33 | 59.6177 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_8_ the | 1 | 26 | -57.0512 |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 1 | 24 | -55.2653 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_5_?\n | 2 | 26 | -55.2052 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 2 | 26 | 54.9299 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 24 | 54.7182 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_6_Answer | 2 | 26 | -54.7090 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 3 | 26 | -53.9945 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 26 | -53.7528 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_9_ answer | 2 | 26 | 51.4965 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 1 | 24 | -50.9954 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 29 | 50.0816 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 27 | 50.0152 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 3 | 26 | 49.9923 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 20 | 49.2188 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 3 | 26 | 48.3859 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_6_ with | 1 | 33 | 47.6023 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ apple | 1 | 29 | 46.6088 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 24 | 45.7630 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | 44.5747 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 24 | -44.4414 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_7_ with | 1 | 26 | -42.7421 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 3 | 24 | -41.6807 |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_9_ answer | 1 | 26 | 41.5996 |
| qwen3_explain_l29_natural_trigger | drift | drop_tok_3_ apple | 2 | 33 | 40.9525 |

## Readout summary

| spec | group | variant | step | top changed | rank delta | prose delta | echo delta | top tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_6_Answer | 1 | 0 | -51866.0000 | -1.5938 | 1.0938 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | success | because_removed | 1 | 2 | -29119.0000 | 0.1562 | 1.5938 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | success | short_answer_instruction | 1 | 2 | 27510.5000 | 0.5625 | 1.8750 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 1 | 2 | 27064.0000 | 0.2188 | -2.2500 | {' ': 2, ' The': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_8_ the | 1 | 0 | -21934.0000 | -0.2188 | -0.8438 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 1 | 0 | 20699.5000 | -0.6562 | 0.2188 | {' Glass': 2, ' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_3_ used | 1 | 2 | 20138.5000 | -0.0625 | -1.0938 | {' The': 4} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_9_ answer | 1 | 0 | 17516.5000 | -0.2969 | -0.4531 | {' Glass': 2, ' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_instruction | 1 | 2 | 17258.5000 | 1.0625 | 2.1875 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_6_Answer | 1 | 2 | -16024.0000 | -1.0000 | 0.9375 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 0 | 16015.0000 | 2.0625 | -1.9062 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_12_ then | 1 | 2 | 15917.0000 | 0.6875 | 1.7812 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_5_?\n | 1 | 0 | -15276.0000 | -1.7500 | 0.8750 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 0 | 14899.0000 | 0.7500 | 0.5312 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 2 | -14631.0000 | -0.4688 | -0.4688 | {' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_13_ one | 1 | 2 | 12162.0000 | -0.2656 | -0.2969 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_4_ for | 1 | 0 | 11411.0000 | -0.5625 | -0.4062 | {' Glass': 2, ' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_10_ first | 1 | 2 | 10911.0000 | 0.2188 | -0.5625 | {' The': 2, ' [': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_7_ with | 1 | 0 | -10900.0000 | -1.4062 | -0.6562 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_14_ short | 1 | 4 | -10586.0000 | -0.3906 | 0.2344 | {' Cup': 2, ' [': 2} |
| deepseek7b_explain_l24_natural_trigger | success | because_removed | 2 | 0 | -7923.5000 | -1.5000 | -0.7500 | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | because_removed | 1 | 0 | -7688.0000 | 0.3750 | 1.0625 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 1 | 0 | -6857.0000 | -0.7500 | 0.8750 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | because_removed | 2 | 0 | 6678.0000 | -1.1562 | -1.7188 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_12_ then | 3 | 0 | -6237.5000 | -0.1562 | 1.0000 | {' is': 2, ' used': 2} |
| glm4_repeat_l30_natural_trigger | drift | no_answer_anchor | 1 | 4 | -6151.2500 | 1.7500 | -2.5000 | {' For': 4} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_5_?\n | 1 | 2 | 6109.0000 | -2.0469 | 0.7031 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_3_ used | 1 | 2 | 5971.0000 | -0.3438 | -0.0938 | {' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_3_ used | 2 | 0 | 5834.5000 | 0.2500 | -0.2812 | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_7_ with | 1 | 2 | 5767.5000 | -0.3281 | -0.2656 | {' Cup': 2, ' Glass': 2} |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 4 | -5467.0000 | 1.6406 | -2.3984 | {' For': 4} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_8_ the | 1 | 0 | 5069.5000 | 0.1719 | 0.2656 | {' Glass': 2, ' The': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 2 | 0 | 4927.0000 | -1.2188 | -1.8750 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 2 | 0 | -4855.0000 | 0.3125 | 0.5000 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_7_ with | 2 | 0 | -4690.0000 | -0.5000 | -0.0625 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_12_ then | 2 | 0 | -4670.5000 | 0.8750 | 0.8438 | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 0 | 4361.0000 | -0.7812 | 0.4062 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_3_ used | 2 | 0 | -4165.0000 | 1.2500 | 0.5625 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 2 | 3934.5000 | 1.3125 | -2.6094 | {' answer': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_6_Answer | 2 | 0 | -3763.0000 | -1.7500 | -0.2500 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_3_ used | 3 | 0 | 3068.5000 | -0.4688 | 0.5938 | {' is': 2, ' used': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_7_ with | 2 | 0 | 3056.5000 | 0.4062 | -0.4062 | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_4_ for | 2 | 0 | -3008.0000 | 0.5312 | 0.7188 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_9_ answer | 2 | 0 | -2967.0000 | -0.1250 | -0.2188 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 1 | 0 | -2909.0000 | 0.5312 | 1.8438 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_4_ for | 1 | 0 | 2769.0000 | -0.2188 | 0.2812 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_4_ for | 2 | 0 | 2556.5000 | 0.3750 | 0.5312 | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | drop_tok_14_ short | 2 | 0 | 2455.0000 | 0.2500 | -0.0312 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | drop_tok_5_?\n | 2 | 0 | 2186.0000 | -0.2188 | -0.7188 | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 3 | 0 | 2060.0000 | -1.6875 | 0.3438 | {' is': 2, ' used': 2} |
