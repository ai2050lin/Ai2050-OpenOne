# Phase 223 channel activation gate validation

spec_count: 3
filter_rows: 56
reproducible_success_rows: 17
reproducible_drift_rows: 20
rollout_rows: 560
channel_score_rows: 2304
activation_stat_rows: 288
total_damage_match_loss: 15
total_repair_match_gain: 12

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_success_L29_K64 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_zero_L29_K16 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_zero_L29_K64 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_pos_drift_L30_K4 | 5 | 5 | 4 | 0 | {'next_task_or_format': 3, 'other_or_wrong': 1, 'repeat_answer': 1} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_pos_drift_L30_K64 | 5 | 5 | 2 | 0 | {'next_task_or_format': 1, 'repeat_answer': 3, 'short_answer': 1} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_pos_zero_L30_K16 | 5 | 5 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l30_activation_gate | mlpchan_pos_zero_L30_K4 | 5 | 5 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l30_activation_gate | mlpchan_pos_zero_L30_K64 | 5 | 5 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_drift_L29_K16 | 5 | 5 | 1 | 0 | {'explain_answer': 4, 'other_or_wrong': 1} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_drift_L29_K64 | 5 | 5 | 1 | 0 | {'explain_answer': 4, 'other_or_wrong': 1} | {'other_or_wrong': 5} |
| glm4_repeat_l30_activation_gate | mlpchan_pos_drift_L30_K16 | 5 | 5 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 4} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_drift_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_drift_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_drift_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_drift_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_drift_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_drift_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_success_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_success_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_success_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_success_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_success_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_success_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_zero_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_zero_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_zero_L29_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_zero_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_zero_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_neg_zero_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_drift_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_drift_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_drift_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_drift_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_success_L29_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_success_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_success_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_success_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_success_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_zero_L29_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_zero_L31_K16 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_zero_L31_K4 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_activation_gate | mlpchan_pos_zero_L31_K64 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_drift_L30_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_drift_L30_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_drift_L30_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_success_L30_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_success_L30_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_success_L30_K64 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_zero_L30_K16 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_activation_gate | mlpchan_neg_zero_L30_K4 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |

## Top activation stats

- qwen3_explain_l29_l31_activation_gate L31 step=2 pos channel=580 rank=1 success_z=-1.0311105251312256 drift_z=-20.59895896911621 delta=19.567848443984985 signed=5.1657633781433105
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=6627 rank=1 success_z=-24.39676284790039 drift_z=-4.526041507720947 delta=-19.870721340179443 signed=3.7097275257110596
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=1735 rank=1 success_z=1.5053013563156128 drift_z=-14.5546875 delta=16.059988856315613 signed=2.9320759773254395
- qwen3_explain_l29_l31_activation_gate L29 step=2 pos channel=5880 rank=1 success_z=-5.568394184112549 drift_z=-16.04166603088379 delta=10.47327184677124 signed=2.4407827854156494
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=1070 rank=2 success_z=-9.735490798950195 drift_z=1.5833333730697632 delta=-11.318824172019958 signed=2.080578565597534
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=4800 rank=1 success_z=0.4411969780921936 drift_z=9.559895515441895 delta=-9.118698537349701 signed=1.9085501432418823
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=9219 rank=2 success_z=7.982142925262451 drift_z=-2.4010417461395264 delta=10.383184671401978 signed=1.8988991975784302
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=8384 rank=2 success_z=-14.352120399475098 drift_z=-2.31201171875 delta=-12.040108680725098 signed=1.8035664558410645
- qwen3_explain_l29_l31_activation_gate L29 step=1 pos channel=1070 rank=1 success_z=-5.854910850524902 drift_z=0.8984375 delta=-6.753348350524902 signed=1.7872856855392456
- qwen3_explain_l29_l31_activation_gate L31 step=2 pos channel=2779 rank=2 success_z=2.9744350910186768 drift_z=19.5 delta=-16.525564908981323 signed=1.687170386314392
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=4057 rank=3 success_z=-0.3171038031578064 drift_z=9.244465827941895 delta=-9.561569631099701 signed=1.5858393907546997
- qwen3_explain_l29_l31_activation_gate L29 step=4 pos channel=199 rank=1 success_z=0.3294503390789032 drift_z=-8.526041984558105 delta=8.855492323637009 signed=1.57142972946167
- qwen3_explain_l29_l31_activation_gate L29 step=2 pos channel=5347 rank=2 success_z=-9.7041015625 drift_z=0.03515625 delta=-9.7392578125 signed=1.5075151920318604
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=577 rank=3 success_z=-6.252232074737549 drift_z=5.104166507720947 delta=-11.356398582458496 signed=1.3958849906921387
- qwen3_explain_l29_l31_activation_gate L29 step=2 pos channel=1070 rank=3 success_z=-7.6569647789001465 drift_z=0.5989583134651184 delta=-8.255923092365265 signed=1.3311847448349
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=9219 rank=3 success_z=10.582589149475098 drift_z=-3.2591145038604736 delta=13.841703653335571 signed=1.3236777782440186
- glm4_repeat_l30_activation_gate L30 step=1 pos channel=7088 rank=1 success_z=2.8843750953674316 drift_z=-0.6028686761856079 delta=3.4872437715530396 signed=1.2031055688858032
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=8989 rank=4 success_z=0.3493390679359436 drift_z=8.841267585754395 delta=-8.491928517818451 signed=1.1823713779449463
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=4350 rank=4 success_z=-12.026785850524902 drift_z=-1.9752603769302368 delta=-10.051525473594666 signed=1.1280184984207153
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=3526 rank=5 success_z=0.9522879719734192 drift_z=9.229166984558105 delta=-8.276879012584686 signed=1.1092240810394287
- qwen3_explain_l29_l31_activation_gate L31 step=4 pos channel=6516 rank=1 success_z=2.226283550262451 drift_z=14.432291984558105 delta=-12.206008434295654 signed=1.0949585437774658
- qwen3_explain_l29_l31_activation_gate L29 step=4 pos channel=1070 rank=2 success_z=-6.674874305725098 drift_z=1.1875 delta=-7.862374305725098 signed=1.090144157409668
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=1132 rank=5 success_z=-0.0557338185608387 drift_z=-6.001302242279053 delta=5.945568423718214 signed=1.0566301345825195
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=1441 rank=6 success_z=-10.366768836975098 drift_z=-1.2666015625 delta=-9.100167274475098 signed=1.0429545640945435
- qwen3_explain_l29_l31_activation_gate L29 step=1 pos channel=6627 rank=2 success_z=-17.535715103149414 drift_z=-10.375 delta=-7.160715103149414 signed=1.0060380697250366
- qwen3_explain_l29_l31_activation_gate L31 step=4 pos channel=8384 rank=2 success_z=-11.033552169799805 drift_z=-0.8515625 delta=-10.181989669799805 signed=0.9703545570373535
- qwen3_explain_l29_l31_activation_gate L29 step=4 pos channel=1914 rank=3 success_z=0.4771205484867096 drift_z=-8.35546875 delta=8.83258929848671 signed=0.9654226303100586
- qwen3_explain_l29_l31_activation_gate L31 step=3 pos channel=9315 rank=7 success_z=-0.0725446417927742 drift_z=-9.546875 delta=9.474330358207226 signed=0.9257717132568359
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=8667 rank=4 success_z=0.0646449476480484 drift_z=6.244791507720947 delta=-6.180146560072899 signed=0.9136122465133667
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=8384 rank=5 success_z=-7.895089149475098 drift_z=-0.9733073115348816 delta=-6.921781837940216 signed=0.9119166135787964
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=2031 rank=6 success_z=-0.2622528076171875 drift_z=-7.620442867279053 delta=7.358190059661865 signed=0.9119136333465576
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=41 rank=7 success_z=0.8070591688156128 drift_z=-7.182291507720947 delta=7.98935067653656 signed=0.9058279395103455
- qwen3_explain_l29_l31_activation_gate L29 step=1 pos channel=8251 rank=3 success_z=-0.00244140625 drift_z=-6.148763179779053 delta=6.146321773529053 signed=0.8822207450866699
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=984 rank=8 success_z=0.4406389594078064 drift_z=-7.3125 delta=7.753138959407806 signed=0.8773189783096313
- qwen3_explain_l29_l31_activation_gate L31 step=4 pos channel=5437 rank=3 success_z=0.1166294664144516 drift_z=9.96875 delta=-9.852120533585548 signed=0.8621014952659607
- qwen3_explain_l29_l31_activation_gate L29 step=2 pos channel=1132 rank=4 success_z=-2.8773891925811768 drift_z=-7.604166507720947 delta=4.7267773151397705 signed=0.8573555946350098
- qwen3_explain_l29_l31_activation_gate L29 step=3 pos channel=8685 rank=9 success_z=8.769949913024902 drift_z=1.953125 delta=6.816824913024902 signed=0.8334767818450928
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=6783 rank=6 success_z=0.0546875 drift_z=6.3203125 delta=-6.265625 signed=0.8323106169700623
- qwen3_explain_l29_l31_activation_gate L31 step=1 pos channel=6121 rank=7 success_z=-7.0594305992126465 drift_z=0.9544270634651184 delta=-8.013857662677765 signed=0.8119567036628723
- qwen3_explain_l29_l31_activation_gate L29 step=1 pos channel=8989 rank=4 success_z=0.7710222601890564 drift_z=6.709635257720947 delta=-5.938612997531891 signed=0.8087530136108398
