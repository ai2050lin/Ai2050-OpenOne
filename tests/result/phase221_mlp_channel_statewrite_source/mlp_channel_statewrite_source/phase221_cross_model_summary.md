# Phase 221 MLP channel StateWrite source

spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 264
channel_score_rows: 960
total_damage_match_loss: 6
total_repair_match_gain: 8

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L29_K16 | 4 | 4 | 0 | 3 | {'explain_answer': 4} | {'explain_answer': 3, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L29_K64 | 4 | 4 | 0 | 3 | {'explain_answer': 4} | {'explain_answer': 3, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L30_K16 | 4 | 4 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 2} | {'echo_then_answer': 2, 'next_task_or_format': 2} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L30_K4 | 4 | 4 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 2} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L30_K64 | 4 | 4 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 2} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L31_K16 | 4 | 4 | 0 | 1 | {'explain_answer': 4} | {'explain_answer': 1, 'other_or_wrong': 3} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L31_K64 | 4 | 4 | 0 | 1 | {'explain_answer': 4} | {'explain_answer': 1, 'other_or_wrong': 3} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L29_K16 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L29_K4 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L29_K64 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L31_K4 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L29_K4 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L31_K16 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L31_K4 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L31_K64 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_repeat_l31_mlp_channels | mlpchan_boost_L31_K16 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_mlp_channels | mlpchan_boost_L31_K4 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_mlp_channels | mlpchan_boost_L31_K64 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_mlp_channels | mlpchan_zero_L31_K16 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_mlp_channels | mlpchan_zero_L31_K4 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_mlp_channels | mlpchan_zero_L31_K64 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L28_K16 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L28_K4 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'echo_then_answer': 2, 'next_task_or_format': 2} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L28_K64 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L30_K16 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L30_K4 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L30_K64 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L28_K16 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L28_K4 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L28_K64 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |

## Channel summaries

### qwen3_explain_l29_l31_mlp_channels L29
- step=3 rank=1 channel=6627 score=3.7097275257110596 signed=3.7097275257110596
- step=2 rank=1 channel=5880 score=2.4407827854156494 signed=2.4407827854156494
- step=3 rank=2 channel=1070 score=2.080578565597534 signed=2.080578565597534
- step=1 rank=1 channel=1070 score=1.7872856855392456 signed=1.7872856855392456
- step=3 rank=3 channel=4057 score=1.5858393907546997 signed=1.5858393907546997
- step=2 rank=2 channel=5347 score=1.5075151920318604 signed=1.5075151920318604
- step=2 rank=3 channel=1070 score=1.3311847448349 signed=1.3311847448349
- step=3 rank=4 channel=8989 score=1.1823713779449463 signed=1.1823713779449463
### qwen3_explain_l29_l31_mlp_channels L31
- step=2 rank=1 channel=580 score=5.1657633781433105 signed=5.1657633781433105
- step=3 rank=1 channel=1735 score=2.9320759773254395 signed=2.9320759773254395
- step=1 rank=1 channel=4800 score=1.9085501432418823 signed=1.9085501432418823
- step=1 rank=2 channel=9219 score=1.8988991975784302 signed=1.8988991975784302
- step=3 rank=2 channel=8384 score=1.8035664558410645 signed=1.8035664558410645
- step=2 rank=2 channel=2779 score=1.687170386314392 signed=1.687170386314392
- step=1 rank=3 channel=577 score=1.3958849906921387 signed=1.3958849906921387
- step=3 rank=3 channel=9219 score=1.3236777782440186 signed=1.3236777782440186
### qwen3_repeat_l31_mlp_channels L31
- step=1 rank=1 channel=6567 score=6.372757911682129 signed=6.372757911682129
- step=3 rank=1 channel=6567 score=6.278130054473877 signed=6.278130054473877
- step=3 rank=2 channel=9407 score=4.402398109436035 signed=4.402398109436035
- step=1 rank=2 channel=9407 score=2.691056489944458 signed=2.691056489944458
- step=1 rank=3 channel=9219 score=2.3852298259735107 signed=2.3852298259735107
- step=1 rank=4 channel=4350 score=1.9816278219223022 signed=1.9816278219223022
- step=3 rank=3 channel=3298 score=1.9387078285217285 signed=1.9387078285217285
- step=2 rank=1 channel=6567 score=1.5403296947479248 signed=1.5403296947479248
### glm4_repeat_l28_l30_mlp_channels L28
- step=1 rank=1 channel=12792 score=1.1343353986740112 signed=1.1343353986740112
- step=2 rank=1 channel=13262 score=0.45227253437042236 signed=0.45227253437042236
- step=3 rank=1 channel=1260 score=0.30751144886016846 signed=0.30751144886016846
- step=3 rank=2 channel=742 score=0.28095343708992004 signed=0.28095343708992004
- step=3 rank=3 channel=5867 score=0.20887160301208496 signed=0.20887160301208496
- step=2 rank=2 channel=742 score=0.19712992012500763 signed=0.19712992012500763
- step=3 rank=4 channel=5370 score=0.15639552474021912 signed=0.15639552474021912
- step=1 rank=2 channel=7111 score=0.13683952391147614 signed=0.13683952391147614
### glm4_repeat_l28_l30_mlp_channels L30
- step=1 rank=1 channel=7088 score=1.8614083528518677 signed=1.8614083528518677
- step=3 rank=1 channel=7088 score=0.7079073786735535 signed=0.7079073786735535
- step=2 rank=1 channel=9892 score=0.6757321357727051 signed=0.6757321357727051
- step=2 rank=2 channel=7088 score=0.39279431104660034 signed=0.39279431104660034
- step=1 rank=2 channel=6118 score=0.38908255100250244 signed=0.38908255100250244
- step=1 rank=3 channel=9374 score=0.38095614314079285 signed=0.38095614314079285
- step=1 rank=4 channel=5760 score=0.32608088850975037 signed=0.32608088850975037
- step=3 rank=2 channel=9374 score=0.2751177251338959 signed=0.2751177251338959
