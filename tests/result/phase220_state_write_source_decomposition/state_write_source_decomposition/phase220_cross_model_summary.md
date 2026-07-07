# Phase 220 StateWrite source decomposition

spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 520
source_alignment_rows: 48
total_damage_match_loss: 52
total_repair_match_gain: 12

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s1.0 | 4 | 4 | 2 | 4 | {'explain_answer': 2, 'other_or_wrong': 2} | {'explain_answer': 4} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s2.0 | 4 | 4 | 2 | 4 | {'explain_answer': 2, 'other_or_wrong': 2} | {'explain_answer': 4} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s1.0 | 4 | 4 | 4 | 0 | {'other_or_wrong': 2, 'short_answer': 2} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s1.5 | 4 | 4 | 4 | 0 | {'other_or_wrong': 2, 'short_answer': 2} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s2.0 | 4 | 4 | 4 | 0 | {'other_or_wrong': 2, 'short_answer': 2} | {'other_or_wrong': 4} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s1.0 | 4 | 4 | 4 | 0 | {'other_or_wrong': 3, 'short_answer': 1} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s1.5 | 4 | 4 | 4 | 0 | {'other_or_wrong': 4} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s2.0 | 4 | 4 | 4 | 0 | {'other_or_wrong': 4} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s1.0 | 4 | 4 | 4 | 0 | {'echo_then_answer': 2, 'other_or_wrong': 2} | {'list_answer': 1, 'other_or_wrong': 3} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s1.5 | 4 | 4 | 4 | 0 | {'other_or_wrong': 4} | {'other_or_wrong': 4} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s2.0 | 4 | 4 | 4 | 0 | {'other_or_wrong': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s0.5 | 4 | 4 | 0 | 3 | {'explain_answer': 4} | {'explain_answer': 3, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s1.5 | 4 | 4 | 2 | 1 | {'explain_answer': 2, 'other_or_wrong': 2} | {'echo_then_answer': 3, 'explain_answer': 1} |
| qwen3_explain_l29_l31_l33_source | mlp_sdm_add_L33 | 4 | 4 | 2 | 0 | {'echo_then_answer': 2, 'explain_answer': 2} | {'other_or_wrong': 4} |
| glm4_repeat_l28_l29_l30_source | mlp_proj_remove_L30 | 4 | 4 | 2 | 0 | {'next_task_or_format': 1, 'repeat_answer': 2, 'short_answer': 1} | {'next_task_or_format': 4} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s0.25 | 4 | 4 | 2 | 0 | {'next_task_or_format': 2, 'repeat_answer': 2} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s0.5 | 4 | 4 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 2} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s0.25 | 4 | 4 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 3} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s0.5 | 4 | 4 | 1 | 0 | {'other_or_wrong': 1, 'repeat_answer': 3} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_explain_l29_l31_l33_source | attn_proj_remove_L29 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | attn_proj_remove_L31 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | attn_proj_remove_L33 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | attn_sdm_add_L29 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | attn_sdm_add_L31 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | attn_sdm_add_L33 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | mlp_proj_remove_L29 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | mlp_proj_remove_L31 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | mlp_proj_remove_L33 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | mlp_sdm_add_L29 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | mlp_sdm_add_L31 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s0.25 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s0.25 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s0.5 | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_repeat_l31_l33_source | attn_proj_remove_L31 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | attn_proj_remove_L33 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | attn_sdm_add_L31 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_l33_source | attn_sdm_add_L33 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | mlp_proj_remove_L31 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_l33_source | mlp_proj_remove_L33 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | mlp_sdm_add_L31 | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'next_task_or_format': 4} |

## Top source alignments

| spec | layer | module | rows | cosine | abs cosine | norm ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| glm4_repeat_l28_l29_l30_source | 28 | mlp | 3 | 0.5280847549438477 | 0.5280847549438477 | 0.30728742605157383 |
| glm4_repeat_l28_l29_l30_source | 30 | mlp | 3 | 0.503204474846522 | 0.503204474846522 | 0.30693514969415736 |
| qwen3_repeat_l31_l33_source | 31 | mlp | 3 | 0.46829502781232196 | 0.46829502781232196 | 0.3835119518827954 |
| qwen3_explain_l29_l31_l33_source | 29 | mlp | 3 | 0.4625376562277476 | 0.4625376562277476 | 0.4253161881562961 |
| glm4_repeat_l28_l29_l30_source | 29 | mlp | 3 | 0.44503698746363324 | 0.44503698746363324 | 0.28039773664421586 |
| qwen3_explain_l29_l31_l33_source | 31 | mlp | 3 | 0.438957283894221 | 0.438957283894221 | 0.3875599832549749 |
| qwen3_explain_l29_l31_l33_source | 29 | attn | 3 | 0.4207950135072072 | 0.4207950135072072 | 0.27149516113196465 |
| qwen3_explain_l29_l31_l33_source | 33 | mlp | 3 | 0.39716293414433795 | 0.39716293414433795 | 0.43266032360759804 |
| qwen3_repeat_l31_l33_source | 33 | mlp | 3 | 0.31818339228630066 | 0.31818339228630066 | 0.3554370480360382 |
| glm4_repeat_l28_l29_l30_source | 30 | attn | 3 | 0.29045332471529645 | 0.29045332471529645 | 0.11368007984619771 |
| glm4_repeat_l28_l29_l30_source | 29 | attn | 3 | 0.2670993556578954 | 0.2670993556578954 | 0.1702571771522361 |
| qwen3_explain_l29_l31_l33_source | 31 | attn | 3 | 0.25413783888022107 | 0.25413783888022107 | 0.16348207036815524 |
| qwen3_repeat_l31_l33_source | 31 | attn | 3 | 0.23924875259399414 | 0.23924875259399414 | 0.14195884460355143 |
| qwen3_repeat_l31_l33_source | 33 | attn | 3 | 0.19944274922211966 | 0.19944274922211966 | 0.16036209417303876 |
| glm4_repeat_l28_l29_l30_source | 28 | attn | 3 | 0.17108083764712015 | 0.17108083764712015 | 0.1256130057259656 |
| qwen3_explain_l29_l31_l33_source | 33 | attn | 3 | 0.16029458741346994 | 0.16029458741346994 | 0.13063435584853691 |
