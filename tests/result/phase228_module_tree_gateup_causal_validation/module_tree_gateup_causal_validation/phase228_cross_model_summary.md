# Phase 228 module-tree calibrated gate/up causal validation

patch_rows: 1530
channel_score_rows: 1152

## Module audit

| model | spec | layer | mlp type | attrs | shapes |
| --- | --- | ---: | --- | --- | --- |
| qwen3 | qwen3_explain_l29_gateup_causal | 29 | split_gate_up | {'dense_4h_to_h': False, 'dense_h_to_4h': False, 'down_proj': True, 'gate_proj': True, 'gate_up_proj': False, 'up_proj': True} | {'down_proj': [2560, 9728], 'gate_proj': [9728, 2560], 'up_proj': [9728, 2560]} |
| glm4 | glm4_repeat_l30_gateup_causal | 30 | merged_gate_up | {'dense_4h_to_h': False, 'dense_h_to_4h': False, 'down_proj': True, 'gate_proj': False, 'gate_up_proj': True, 'up_proj': False} | {'down_proj': [4096, 13696], 'gate_up_proj': [27392, 4096]} |
| deepseek7b | deepseek7b_explain_l24_gateup_causal | 24 | split_gate_up | {'dense_4h_to_h': False, 'dense_h_to_4h': False, 'down_proj': True, 'gate_proj': True, 'gate_up_proj': False, 'up_proj': True} | {'down_proj': [3584, 18944], 'gate_proj': [18944, 3584], 'up_proj': [18944, 3584]} |

## Product recompute calibration

| model | spec | step | layer | gate | up | product | down_out | rel error | cosine |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| glm4 | glm4_repeat_l30_gateup_causal | 1 | 30 | True | True | True | True | 0.001170 | 1.000000 |
| glm4 | glm4_repeat_l30_gateup_causal | 2 | 30 | True | True | True | True | 0.001268 | 0.999999 |
| glm4 | glm4_repeat_l30_gateup_causal | 1 | 30 | True | True | True | True | 0.001320 | 1.000000 |
| deepseek7b | deepseek7b_explain_l24_gateup_causal | 1 | 24 | True | True | True | True | 0.001392 | 0.999999 |
| qwen3 | qwen3_explain_l29_gateup_causal | 1 | 29 | True | True | True | True | 0.001606 | 0.999998 |
| glm4 | glm4_repeat_l30_gateup_causal | 2 | 30 | True | True | True | True | 0.001648 | 0.999998 |
| qwen3 | qwen3_explain_l29_gateup_causal | 2 | 29 | True | True | True | True | 0.001671 | 0.999998 |
| qwen3 | qwen3_explain_l29_gateup_causal | 2 | 29 | True | True | True | True | 0.001706 | 0.999998 |
| qwen3 | qwen3_explain_l29_gateup_causal | 1 | 29 | True | True | True | True | 0.001717 | 0.999998 |
| deepseek7b | deepseek7b_explain_l24_gateup_causal | 2 | 24 | True | True | True | True | 0.001812 | 0.999998 |
| deepseek7b | deepseek7b_explain_l24_gateup_causal | 2 | 24 | True | True | True | True | 0.002031 | 0.999998 |
| deepseek7b | deepseek7b_explain_l24_gateup_causal | 1 | 24 | True | True | True | True | 0.002512 | 0.999997 |

## Patch summary

| spec | group | component | scope | alpha | step | layer | rows | rank delta | logit delta | top changed |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b_explain_l24_gateup_causal | success | product | all | 1.0 | 2 | 24 | 3 | -30977.3333 | -3.5000 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | all | 1.0 | 2 | 24 | 3 | -30359.6667 | -3.4531 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top16 | 1.0 | 2 | 24 | 3 | -30359.6667 | -3.4531 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top64 | 1.0 | 2 | 24 | 3 | -30359.6667 | -3.4531 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | all | 1.0 | 2 | 24 | 3 | -20864.6667 | -3.5234 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | all | 1.0 | 2 | 24 | 3 | -19845.6667 | -4.2812 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | top64 | 1.0 | 2 | 24 | 3 | -18938.3333 | -2.8441 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | top16 | 1.0 | 2 | 24 | 3 | -15534.0000 | -2.0781 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | product | top64 | 1.0 | 2 | 24 | 2 | -12029.0000 | -1.6484 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | all | 0.5 | 2 | 24 | 3 | -11638.6667 | -1.7760 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | top64 | 1.0 | 2 | 24 | 2 | -11471.0000 | -1.5078 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | all | 0.5 | 2 | 24 | 3 | -11382.6667 | -1.7396 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top16 | 0.5 | 2 | 24 | 3 | -11382.6667 | -1.7396 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top64 | 0.5 | 2 | 24 | 3 | -11382.6667 | -1.7396 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | top16 | 1.0 | 2 | 24 | 3 | -10383.3333 | -1.7630 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | top64 | 1.0 | 2 | 24 | 3 | -10245.3333 | -2.1510 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | up | all | 1.0 | 2 | 24 | 2 | 9581.0000 | 0.9805 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | top64 | 1.0 | 2 | 24 | 3 | -9064.6667 | -2.8910 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate | top64 | 1.0 | 2 | 24 | 2 | -8803.0000 | -1.2109 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | top64 | 0.5 | 2 | 24 | 2 | -7154.0000 | -0.9922 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | all | 1.0 | 1 | 24 | 3 | -6953.6667 | -0.3646 | 1 |
| deepseek7b_explain_l24_gateup_causal | drift | product | top64 | 0.5 | 2 | 24 | 2 | -6600.0000 | -0.8203 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | all | 1.0 | 1 | 24 | 3 | -6418.6667 | -0.3698 | 1 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top16 | 1.0 | 1 | 24 | 3 | -6418.6667 | -0.3698 | 1 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top64 | 1.0 | 1 | 24 | 3 | -6418.6667 | -0.3698 | 1 |
| deepseek7b_explain_l24_gateup_causal | success | product | top64 | 1.0 | 1 | 24 | 3 | -6355.6667 | -0.4701 | 2 |
| deepseek7b_explain_l24_gateup_causal | success | product | top64 | 0.5 | 2 | 24 | 3 | -6086.6667 | -1.2760 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | top16 | 1.0 | 2 | 24 | 2 | -6077.0000 | -1.2891 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate | top64 | 0.5 | 2 | 24 | 2 | -5997.0000 | -0.8672 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | up | all | 0.5 | 2 | 24 | 2 | 5519.0000 | 0.4531 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | top16 | 0.5 | 2 | 24 | 3 | -5441.3333 | -0.9479 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate | top16 | 1.0 | 2 | 24 | 2 | -5274.0000 | -1.2109 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate | all | 1.0 | 2 | 24 | 2 | -5240.0000 | -0.6953 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | all | 1.0 | 1 | 24 | 3 | -4761.3333 | -0.3646 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | top16 | 0.5 | 2 | 24 | 2 | -4686.0000 | -0.8359 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | product | top16 | 1.0 | 2 | 24 | 2 | -4518.0000 | -1.2109 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | all | 0.5 | 1 | 24 | 3 | -4492.6667 | -0.2122 | 1 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | all | 0.5 | 2 | 24 | 3 | -4336.3333 | -1.1458 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | all | 0.5 | 1 | 24 | 3 | -4299.6667 | -0.2318 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top16 | 0.5 | 1 | 24 | 3 | -4299.6667 | -0.2318 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top64 | 0.5 | 1 | 24 | 3 | -4299.6667 | -0.2318 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | all | 0.5 | 1 | 24 | 3 | -4294.0000 | -0.2253 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | all | 0.25 | 2 | 24 | 3 | -4214.3333 | -0.8125 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top16 | 0.25 | 2 | 24 | 3 | -4214.3333 | -0.8125 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | down_out | top64 | 0.25 | 2 | 24 | 3 | -4214.3333 | -0.8125 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | all | 0.25 | 2 | 24 | 3 | -4138.6667 | -0.7604 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate | top16 | 0.5 | 2 | 24 | 2 | -3987.0000 | -0.7266 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | top16 | 1.0 | 1 | 24 | 3 | 3957.0000 | -0.0794 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | all | 0.25 | 1 | 24 | 3 | -3913.6667 | -0.1549 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | all | 1.0 | 2 | 24 | 2 | -3905.0000 | -0.8828 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | up | top16 | 1.0 | 2 | 24 | 2 | -3793.0000 | -0.3672 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | top64 | 0.5 | 1 | 24 | 3 | -3572.6667 | -0.2591 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | up | all | 0.25 | 2 | 24 | 2 | 3269.0000 | 0.2188 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | top64 | 0.25 | 2 | 24 | 2 | -3224.0000 | -0.4766 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | product | all | 0.5 | 2 | 24 | 2 | -3179.0000 | -0.6172 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | top16 | 1.0 | 2 | 24 | 3 | -3109.3333 | -1.2708 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | all | 0.5 | 1 | 24 | 3 | -3006.6667 | -0.1628 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | all | 0.25 | 1 | 24 | 3 | -2971.3333 | -0.1680 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | down_out | all | 0.5 | 2 | 24 | 2 | -2766.0000 | -0.6016 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | down_out | top16 | 0.5 | 2 | 24 | 2 | -2766.0000 | -0.6016 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | down_out | top64 | 0.5 | 2 | 24 | 2 | -2766.0000 | -0.6016 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate | all | 0.5 | 2 | 24 | 2 | -2728.0000 | -0.5234 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | top64 | 0.5 | 1 | 24 | 3 | -2681.3333 | -0.2122 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | product | top64 | 0.25 | 2 | 24 | 2 | -2680.0000 | -0.3516 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate_up_pair | top64 | 1.0 | 1 | 24 | 3 | -2676.3333 | -0.3776 | 2 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | all | 1.0 | 1 | 24 | 2 | 2509.0000 | 0.4062 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | top64 | 0.5 | 1 | 24 | 3 | -2464.0000 | -0.1289 | 0 |
| deepseek7b_explain_l24_gateup_causal | drift | gate_up_pair | top16 | 0.25 | 2 | 24 | 2 | -2339.0000 | -0.3984 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | product | top16 | 0.5 | 1 | 24 | 3 | -2292.6667 | -0.2409 | 0 |
| deepseek7b_explain_l24_gateup_causal | success | gate | all | 1.0 | 1 | 24 | 3 | -2271.3333 | -0.1341 | 0 |
