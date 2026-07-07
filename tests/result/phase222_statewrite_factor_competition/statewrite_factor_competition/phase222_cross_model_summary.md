# Phase 222 StateWrite signed channel competition split

spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 630
channel_score_rows: 3840
total_damage_match_loss: 6
total_repair_match_gain: 10

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_zero_L29_K16 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_zero_L29_K64 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_pos_zero_L30_K16 | 5 | 5 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_pos_zero_L30_K4 | 5 | 5 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_pos_zero_L30_K64 | 5 | 5 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_boost_L31_K16 | 5 | 5 | 0 | 1 | {'explain_answer': 5} | {'explain_answer': 1, 'other_or_wrong': 4} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_boost_L31_K64 | 5 | 5 | 0 | 1 | {'explain_answer': 5} | {'explain_answer': 1, 'other_or_wrong': 4} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_boost_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_boost_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_boost_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_boost_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_boost_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_boost_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_zero_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_zero_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_zero_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_zero_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_zero_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_neg_zero_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_boost_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_boost_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_boost_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_boost_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_zero_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_zero_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_zero_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_signed_channel_split | mlpchan_pos_zero_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_neg_boost_L31_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_neg_boost_L31_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_neg_boost_L31_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_neg_zero_L31_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_neg_zero_L31_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_neg_zero_L31_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_pos_boost_L31_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_pos_boost_L31_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_pos_boost_L31_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_pos_zero_L31_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_pos_zero_L31_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_signed_channel_split | mlpchan_pos_zero_L31_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_neg_boost_L28_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_neg_boost_L28_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_neg_boost_L28_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_neg_boost_L30_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_neg_boost_L30_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_signed_channel_split | mlpchan_neg_boost_L30_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |

## Top transitions

| spec | group | condition | from | to | rows |
| --- | --- | --- | --- | --- | ---: |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L29_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_boost_L29_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L29_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_boost_L29_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L29_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_boost_L29_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_zero_L29_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_boost_L29_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_zero_L29_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_boost_L29_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_zero_L29_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_boost_L29_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L31_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_boost_L31_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L31_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_boost_L31_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L31_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_pos_boost_L31_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_zero_L31_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_boost_L31_K4 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_zero_L31_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_boost_L31_K16 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_zero_L31_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | success_repro | mlpchan_neg_boost_L31_K64 | explain_answer | explain_answer | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_zero_L29_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_boost_L29_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_boost_L29_K16 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_boost_L29_K64 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_zero_L29_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_boost_L29_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_zero_L29_K16 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_boost_L29_K16 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_zero_L29_K64 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_boost_L29_K64 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_zero_L31_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_boost_L31_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_zero_L31_K16 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_pos_zero_L31_K64 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_zero_L31_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_boost_L31_K4 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_zero_L31_K16 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_boost_L31_K16 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_zero_L31_K64 | other_or_wrong | other_or_wrong | 5 |
| qwen3_explain_l29_l31_signed_channel_split | drift_repro | mlpchan_neg_boost_L31_K64 | other_or_wrong | other_or_wrong | 5 |
| qwen3_repeat_l31_signed_channel_split | success_repro | mlpchan_pos_zero_L31_K4 | repeat_answer | repeat_answer | 5 |

## Channel summaries

### qwen3_explain_l29_l31_signed_channel_split L29 neg
- step=1 rank=1 channel=1943 signed=-0.567456066608429 delta_z=7.744698524475098 dot=-0.07327026128768921
- step=4 rank=1 channel=1132 signed=-0.3202786445617676 delta_z=-3.884579658508301 dot=0.08244872838258743
- step=4 rank=2 channel=1943 signed=-0.2661280930042267 delta_z=7.844028949737549 dot=-0.03392747417092323
- step=3 rank=1 channel=543 signed=-0.21275269985198975 delta_z=4.894321918487549 dot=-0.04346929118037224
- step=4 rank=3 channel=1 signed=-0.17363591492176056 delta_z=-6.542782783508301 dot=0.026538541540503502
- step=1 rank=2 channel=543 signed=-0.1464025378227234 delta_z=2.7745535373687744 dot=-0.052766162902116776
- step=4 rank=4 channel=543 signed=-0.14494694769382477 delta_z=3.3693034648895264 dot=-0.0430198572576046
- step=4 rank=5 channel=23 signed=-0.13755667209625244 delta_z=-3.602585554122925 dot=0.038182761520147324
### qwen3_explain_l29_l31_signed_channel_split L29 pos
- step=3 rank=1 channel=6627 signed=3.7097275257110596 delta_z=-19.8707218170166 dot=-0.18669314682483673
- step=2 rank=1 channel=5880 signed=2.4407827854156494 delta_z=10.473272323608398 dot=0.2330487221479416
- step=3 rank=2 channel=1070 signed=2.080578565597534 delta_z=-11.31882381439209 dot=-0.18381579220294952
- step=1 rank=1 channel=1070 signed=1.7872856855392456 delta_z=-6.753348350524902 dot=-0.2646517753601074
- step=3 rank=3 channel=4057 signed=1.5858393907546997 delta_z=-9.561569213867188 dot=-0.1658555567264557
- step=4 rank=1 channel=199 signed=1.57142972946167 delta_z=8.85549259185791 dot=0.17745254933834076
- step=2 rank=2 channel=5347 signed=1.5075151920318604 delta_z=-9.7392578125 dot=-0.15478748083114624
- step=2 rank=3 channel=1070 signed=1.3311847448349 delta_z=-8.2559232711792 dot=-0.16123996675014496
### qwen3_explain_l29_l31_signed_channel_split L31 neg
- step=3 rank=1 channel=577 signed=-0.37264496088027954 delta_z=-7.086437702178955 dot=0.0525856539607048
- step=4 rank=1 channel=3298 signed=-0.3182692527770996 delta_z=-3.4899086952209473 dot=0.09119701385498047
- step=4 rank=2 channel=446 signed=-0.2728111147880554 delta_z=-1.8664201498031616 dot=0.14616811275482178
- step=3 rank=2 channel=3298 signed=-0.22247394919395447 delta_z=2.827194929122925 dot=-0.07869070023298264
- step=4 rank=3 channel=825 signed=-0.17303605377674103 delta_z=-5.145926475524902 dot=0.033625829964876175
- step=3 rank=3 channel=763 signed=-0.1724121868610382 delta_z=3.1678059101104736 dot=-0.054426372051239014
- step=2 rank=1 channel=6121 signed=-0.1346331685781479 delta_z=-1.841174840927124 dot=0.0731235146522522
- step=3 rank=4 channel=2983 signed=-0.12961344420909882 delta_z=-3.075730085372925 dot=0.04214070737361908
### qwen3_explain_l29_l31_signed_channel_split L31 pos
- step=2 rank=1 channel=580 signed=5.1657633781433105 delta_z=19.567848205566406 dot=0.26399239897727966
- step=3 rank=1 channel=1735 signed=2.9320759773254395 delta_z=16.059988021850586 dot=0.18257024884223938
- step=1 rank=1 channel=4800 signed=1.9085501432418823 delta_z=-9.118698120117188 dot=-0.20930072665214539
- step=1 rank=2 channel=9219 signed=1.8988991975784302 delta_z=10.383184432983398 dot=0.182882159948349
- step=3 rank=2 channel=8384 signed=1.8035664558410645 delta_z=-12.040108680725098 dot=-0.1497965306043625
- step=2 rank=2 channel=2779 signed=1.687170386314392 delta_z=-16.525564193725586 dot=-0.10209456831216812
- step=1 rank=3 channel=577 signed=1.3958849906921387 delta_z=-11.356398582458496 dot=-0.12291616946458817
- step=3 rank=3 channel=9219 signed=1.3236777782440186 delta_z=13.841703414916992 dot=0.09562969207763672
### qwen3_repeat_l31_signed_channel_split L31 neg
- step=1 rank=1 channel=577 signed=-0.9661674499511719 delta_z=-4.921289443969727 dot=0.19632405042648315
- step=3 rank=1 channel=577 signed=-0.5186589956283569 delta_z=-3.0445313453674316 dot=0.1703575849533081
- step=2 rank=1 channel=6869 signed=-0.33231112360954285 delta_z=-3.5286619663238525 dot=0.094174824655056
- step=3 rank=2 channel=2855 signed=-0.22834017872810364 delta_z=2.0134034156799316 dot=-0.11341004818677902
- step=3 rank=3 channel=8374 signed=-0.1467549353837967 delta_z=2.3885741233825684 dot=-0.061440397053956985
- step=1 rank=2 channel=1 signed=-0.13622531294822693 delta_z=-2.5375003814697266 dot=0.053684841841459274
- step=1 rank=3 channel=108 signed=-0.12714333832263947 delta_z=1.9353516101837158 dot=-0.06569521129131317
- step=1 rank=4 channel=26 signed=-0.1127719134092331 delta_z=1.1707885265350342 dot=-0.09632133692502975
### qwen3_repeat_l31_signed_channel_split L31 pos
- step=3 rank=1 channel=6567 signed=5.427565574645996 delta_z=16.722509384155273 dot=0.32456645369529724
- step=1 rank=1 channel=6567 signed=4.949769973754883 delta_z=17.39631462097168 dot=0.28452980518341064
- step=3 rank=2 channel=9407 signed=3.167288064956665 delta_z=15.954764366149902 dot=0.19851675629615784
- step=1 rank=2 channel=4350 signed=2.298314332962036 delta_z=-11.086706161499023 dot=-0.20730361342430115
- step=1 rank=3 channel=9407 signed=1.8780854940414429 delta_z=11.804327011108398 dot=0.1591014415025711
- step=3 rank=3 channel=3298 signed=1.6574785709381104 delta_z=-5.346875190734863 dot=-0.30999013781547546
- step=3 rank=4 channel=4350 signed=1.5481231212615967 delta_z=-9.54860782623291 dot=-0.16213077306747437
- step=1 rank=4 channel=9219 signed=1.5001933574676514 delta_z=11.274072647094727 dot=0.13306578993797302
### glm4_repeat_l28_l30_signed_channel_split L28 neg
- step=4 rank=1 channel=106 signed=-0.052590642124414444 delta_z=-0.40375974774360657 dot=0.13025231659412384
- step=3 rank=1 channel=7695 signed=-0.026130296289920807 delta_z=0.6272367238998413 dot=-0.041659384965896606
- step=3 rank=2 channel=13632 signed=-0.021022027358412743 delta_z=-0.21962890028953552 dot=0.09571612626314163
- step=4 rank=2 channel=9930 signed=-0.015651613473892212 delta_z=-0.6703227758407593 dot=0.023349367082118988
- step=1 rank=1 channel=7695 signed=-0.013624000363051891 delta_z=0.543505847454071 dot=-0.025066887959837914
- step=1 rank=2 channel=9534 signed=-0.010747569613158703 delta_z=-0.2957214415073395 dot=0.03634355962276459
- step=4 rank=3 channel=5384 signed=-0.01058032363653183 delta_z=0.5492818355560303 dot=-0.019262103363871574
- step=3 rank=3 channel=10074 signed=-0.009897108189761639 delta_z=0.3831543028354645 dot=-0.02583060786128044
### glm4_repeat_l28_l30_signed_channel_split L28 pos
- step=1 rank=1 channel=12792 signed=1.4816234111785889 delta_z=-3.2874863147735596 dot=-0.4506857991218567
- step=3 rank=1 channel=742 signed=0.5674709677696228 delta_z=3.770404100418091 dot=0.15050667524337769
- step=2 rank=1 channel=13262 signed=0.3287925720214844 delta_z=-2.141200304031372 dot=-0.1535552591085434
- step=2 rank=2 channel=742 signed=0.29201993346214294 delta_z=2.2307374477386475 dot=0.1309073567390442
- step=3 rank=2 channel=5867 signed=0.20316877961158752 delta_z=1.6885305643081665 dot=0.1203228309750557
- step=3 rank=3 channel=1260 signed=0.20230723917484283 delta_z=-1.3544433116912842 dot=-0.1493656039237976
- step=3 rank=4 channel=12792 signed=0.163046196103096 delta_z=-1.0841248035430908 dot=-0.15039430558681488
- step=3 rank=5 channel=6372 signed=0.15932083129882812 delta_z=1.3540802001953125 dot=0.11765981465578079
### glm4_repeat_l28_l30_signed_channel_split L30 neg
- step=2 rank=1 channel=6115 signed=-0.08598344773054123 delta_z=1.4216797351837158 dot=-0.060480181127786636
- step=1 rank=1 channel=10336 signed=-0.027119092643260956 delta_z=-0.5862289071083069 dot=0.04626024514436722
- step=1 rank=2 channel=11128 signed=-0.01671566255390644 delta_z=0.9163818359375 dot=-0.01824093610048294
- step=4 rank=1 channel=3577 signed=-0.01322717871516943 delta_z=0.5545898675918579 dot=-0.02385037951171398
- step=1 rank=3 channel=1335 signed=-0.012919855304062366 delta_z=0.609375 dot=-0.021201813593506813
- step=2 rank=2 channel=3402 signed=-0.01235159207135439 delta_z=-0.4559326171875 dot=0.027090827003121376
- step=2 rank=3 channel=2377 signed=-0.012003269977867603 delta_z=-0.3304198980331421 dot=0.03632732108235359
- step=1 rank=4 channel=10887 signed=-0.01063035149127245 delta_z=0.25386351346969604 dot=-0.04187427833676338
### glm4_repeat_l28_l30_signed_channel_split L30 pos
- step=1 rank=1 channel=7088 signed=1.2031055688858032 delta_z=3.48724365234375 dot=0.34500187635421753
- step=1 rank=2 channel=9374 signed=0.7667387127876282 delta_z=2.4660935401916504 dot=0.31091225147247314
- step=3 rank=1 channel=9374 signed=0.5124444961547852 delta_z=2.5059356689453125 dot=0.20449228584766388
- step=2 rank=1 channel=9892 signed=0.44883665442466736 delta_z=-2.973446846008301 dot=-0.15094827115535736
- step=3 rank=2 channel=7088 signed=0.43510037660598755 delta_z=2.08331298828125 dot=0.2088502198457718
- step=4 rank=1 channel=670 signed=0.41590723395347595 delta_z=2.8387253284454346 dot=0.146511971950531
- step=1 rank=3 channel=5760 signed=0.36317208409309387 delta_z=2.5959177017211914 dot=0.13990123569965363
- step=1 rank=4 channel=6118 signed=0.3356916308403015 delta_z=1.9002567529678345 dot=0.17665593326091766
