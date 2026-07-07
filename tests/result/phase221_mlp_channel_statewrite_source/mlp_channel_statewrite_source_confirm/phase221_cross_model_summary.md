# Phase 221 MLP channel StateWrite source

spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 330
channel_score_rows: 1920
total_damage_match_loss: 6
total_repair_match_gain: 10

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L29_K16 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L29_K64 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L30_K16 | 5 | 5 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L30_K4 | 5 | 5 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L30_K64 | 5 | 5 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L31_K16 | 5 | 5 | 0 | 1 | {'explain_answer': 5} | {'explain_answer': 1, 'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L31_K64 | 5 | 5 | 0 | 1 | {'explain_answer': 5} | {'explain_answer': 1, 'other_or_wrong': 4} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_boost_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_mlp_channels | mlpchan_zero_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_repeat_l31_mlp_channels | mlpchan_boost_L31_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_mlp_channels | mlpchan_boost_L31_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_mlp_channels | mlpchan_boost_L31_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_mlp_channels | mlpchan_zero_L31_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| qwen3_repeat_l31_mlp_channels | mlpchan_zero_L31_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_mlp_channels | mlpchan_zero_L31_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L28_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L28_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L28_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L30_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L30_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_boost_L30_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'list_answer': 1, 'next_task_or_format': 4} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L28_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L28_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l30_mlp_channels | mlpchan_zero_L28_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |

## Channel summaries

### qwen3_explain_l29_l31_mlp_channels L29
- step=3 rank=1 channel=6627 score=3.7097275257110596 signed=3.7097275257110596
- step=2 rank=1 channel=5880 score=2.4407827854156494 signed=2.4407827854156494
- step=3 rank=2 channel=1070 score=2.080578565597534 signed=2.080578565597534
- step=1 rank=1 channel=1070 score=1.7872856855392456 signed=1.7872856855392456
- step=3 rank=3 channel=4057 score=1.5858393907546997 signed=1.5858393907546997
- step=4 rank=1 channel=199 score=1.57142972946167 signed=1.57142972946167
- step=2 rank=2 channel=5347 score=1.5075151920318604 signed=1.5075151920318604
- step=2 rank=3 channel=1070 score=1.3311847448349 signed=1.3311847448349
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
- step=3 rank=1 channel=6567 score=5.427565574645996 signed=5.427565574645996
- step=1 rank=1 channel=6567 score=4.949769973754883 signed=4.949769973754883
- step=3 rank=2 channel=9407 score=3.167288064956665 signed=3.167288064956665
- step=1 rank=2 channel=4350 score=2.298314332962036 signed=2.298314332962036
- step=1 rank=3 channel=9407 score=1.8780854940414429 signed=1.8780854940414429
- step=3 rank=3 channel=3298 score=1.6574785709381104 signed=1.6574785709381104
- step=3 rank=4 channel=4350 score=1.5481231212615967 signed=1.5481231212615967
- step=1 rank=4 channel=9219 score=1.5001933574676514 signed=1.5001933574676514
### glm4_repeat_l28_l30_mlp_channels L28
- step=1 rank=1 channel=12792 score=1.4816234111785889 signed=1.4816234111785889
- step=3 rank=1 channel=742 score=0.5674709677696228 signed=0.5674709677696228
- step=2 rank=1 channel=13262 score=0.3287925720214844 signed=0.3287925720214844
- step=2 rank=2 channel=742 score=0.29201993346214294 signed=0.29201993346214294
- step=3 rank=2 channel=5867 score=0.20316877961158752 signed=0.20316877961158752
- step=3 rank=3 channel=1260 score=0.20230723917484283 signed=0.20230723917484283
- step=3 rank=4 channel=12792 score=0.163046196103096 signed=0.163046196103096
- step=3 rank=5 channel=6372 score=0.15932083129882812 signed=0.15932083129882812
### glm4_repeat_l28_l30_mlp_channels L30
- step=1 rank=1 channel=7088 score=1.2031055688858032 signed=1.2031055688858032
- step=1 rank=2 channel=9374 score=0.7667387127876282 signed=0.7667387127876282
- step=3 rank=1 channel=9374 score=0.5124444961547852 signed=0.5124444961547852
- step=2 rank=1 channel=9892 score=0.44883665442466736 signed=0.44883665442466736
- step=3 rank=2 channel=7088 score=0.43510037660598755 signed=0.43510037660598755
- step=4 rank=1 channel=670 score=0.41590723395347595 signed=0.41590723395347595
- step=1 rank=3 channel=5760 score=0.36317208409309387 signed=0.36317208409309387
- step=1 rank=4 channel=6118 score=0.3356916308403015 signed=0.3356916308403015
