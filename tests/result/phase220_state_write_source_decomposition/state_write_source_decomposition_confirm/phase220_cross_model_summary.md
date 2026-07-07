# Phase 220 StateWrite source decomposition

spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 650
source_alignment_rows: 64
total_damage_match_loss: 64
total_repair_match_gain: 16

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s2.0 | 5 | 5 | 3 | 5 | {'explain_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} | {'explain_answer': 5} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s1.0 | 5 | 5 | 2 | 5 | {'explain_answer': 3, 'other_or_wrong': 2} | {'explain_answer': 5} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s1.0 | 5 | 5 | 5 | 1 | {'other_or_wrong': 3, 'short_answer': 2} | {'next_task_or_format': 2, 'other_or_wrong': 2, 'repeat_answer': 1} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s1.0 | 5 | 5 | 5 | 0 | {'other_or_wrong': 3, 'short_answer': 2} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s1.5 | 5 | 5 | 5 | 0 | {'other_or_wrong': 3, 'short_answer': 2} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s2.0 | 5 | 5 | 5 | 0 | {'other_or_wrong': 3, 'short_answer': 2} | {'other_or_wrong': 5} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s1.5 | 5 | 5 | 5 | 0 | {'other_or_wrong': 5} | {'other_or_wrong': 5} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s2.0 | 5 | 5 | 5 | 0 | {'other_or_wrong': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s1.0 | 5 | 5 | 5 | 0 | {'echo_then_answer': 3, 'other_or_wrong': 2} | {'next_task_or_format': 3, 'other_or_wrong': 2} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s1.5 | 5 | 5 | 5 | 0 | {'other_or_wrong': 5} | {'other_or_wrong': 5} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s2.0 | 5 | 5 | 5 | 0 | {'echo_then_answer': 2, 'other_or_wrong': 3} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s0.5 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s1.5 | 5 | 5 | 2 | 1 | {'explain_answer': 3, 'other_or_wrong': 2} | {'echo_then_answer': 4, 'explain_answer': 1} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s0.5 | 5 | 5 | 3 | 0 | {'next_task_or_format': 2, 'other_or_wrong': 1, 'repeat_answer': 2} | {'next_task_or_format': 1, 'other_or_wrong': 4} |
| qwen3_explain_l29_l31_l33_source | mlp_sdm_add_L33 | 5 | 5 | 2 | 0 | {'echo_then_answer': 2, 'explain_answer': 3} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | attn_sdm_add_L33 | 5 | 5 | 1 | 0 | {'echo_then_answer': 1, 'explain_answer': 4} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | mlp_sdm_add_L31 | 5 | 5 | 1 | 0 | {'echo_then_answer': 1, 'explain_answer': 4} | {'other_or_wrong': 5} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s0.25 | 5 | 5 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 4} | {'list_answer': 1, 'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | resid_sub_L31_s0.5 | 5 | 5 | 1 | 0 | {'other_or_wrong': 1, 'repeat_answer': 4} | {'list_answer': 1, 'next_task_or_format': 2, 'other_or_wrong': 2} |
| glm4_repeat_l28_l29_l30_source | mlp_proj_remove_L28 | 5 | 5 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 4} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| glm4_repeat_l28_l29_l30_source | mlp_proj_remove_L30 | 5 | 5 | 1 | 0 | {'repeat_answer': 4, 'short_answer': 1} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| glm4_repeat_l28_l29_l30_source | resid_sub_L28_s0.25 | 5 | 5 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 4} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_l31_l33_source | attn_proj_remove_L29 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | attn_proj_remove_L31 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | attn_proj_remove_L33 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | attn_sdm_add_L29 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | attn_sdm_add_L31 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | mlp_proj_remove_L29 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | mlp_proj_remove_L31 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | mlp_proj_remove_L33 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | mlp_sdm_add_L29 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | resid_add_L31_s0.25 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s0.25 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_l31_l33_source | resid_sub_L31_s0.5 | 5 | 5 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_repeat_l31_l33_source | attn_proj_remove_L31 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_l33_source | attn_proj_remove_L33 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'list_answer': 1, 'next_task_or_format': 4} |
| qwen3_repeat_l31_l33_source | attn_sdm_add_L31 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 2, 'other_or_wrong': 3} |
| qwen3_repeat_l31_l33_source | attn_sdm_add_L33 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_repeat_l31_l33_source | mlp_proj_remove_L31 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'list_answer': 1, 'next_task_or_format': 2, 'other_or_wrong': 2} |
| qwen3_repeat_l31_l33_source | mlp_proj_remove_L33 | 5 | 5 | 0 | 0 | {'repeat_answer': 5} | {'list_answer': 1, 'next_task_or_format': 4} |

## Top source alignments

| spec | layer | module | rows | cosine | abs cosine | norm ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| glm4_repeat_l28_l29_l30_source | 28 | mlp | 4 | 0.49653705954551697 | 0.49653705954551697 | 0.2893645566197019 |
| glm4_repeat_l28_l29_l30_source | 30 | mlp | 4 | 0.4678623601794243 | 0.4678623601794243 | 0.28372531555335306 |
| qwen3_explain_l29_l31_l33_source | 29 | mlp | 4 | 0.45187345147132874 | 0.45187345147132874 | 0.41523807814159375 |
| qwen3_repeat_l31_l33_source | 31 | mlp | 4 | 0.4297787547111511 | 0.4297787547111511 | 0.3425968594963294 |
| qwen3_explain_l29_l31_l33_source | 31 | mlp | 4 | 0.429529532790184 | 0.429529532790184 | 0.3761545365772766 |
| glm4_repeat_l28_l29_l30_source | 29 | mlp | 4 | 0.41477206349372864 | 0.41477206349372864 | 0.2632716402843352 |
| qwen3_explain_l29_l31_l33_source | 29 | attn | 4 | 0.4087904170155525 | 0.4087904170155525 | 0.2619488172211249 |
| qwen3_explain_l29_l31_l33_source | 33 | mlp | 4 | 0.3908563405275345 | 0.3908563405275345 | 0.42744247037795247 |
| qwen3_repeat_l31_l33_source | 33 | mlp | 4 | 0.3253673017024994 | 0.3253673017024994 | 0.34147564521078555 |
| glm4_repeat_l28_l29_l30_source | 30 | attn | 4 | 0.27509962767362595 | 0.27509962767362595 | 0.11125629784394653 |
| qwen3_explain_l29_l31_l33_source | 31 | attn | 4 | 0.25264747813344 | 0.25264747813344 | 0.16001966579021606 |
| glm4_repeat_l28_l29_l30_source | 29 | attn | 4 | 0.23783450573682785 | 0.23783450573682785 | 0.1614508342292083 |
| qwen3_repeat_l31_l33_source | 31 | attn | 4 | 0.23398784920573235 | 0.23398784920573235 | 0.1432110133496558 |
| qwen3_repeat_l31_l33_source | 33 | attn | 4 | 0.2188459448516369 | 0.2188459448516369 | 0.1817347402727953 |
| glm4_repeat_l28_l29_l30_source | 28 | attn | 4 | 0.20229433104395866 | 0.20229433104395866 | 0.1400709696599939 |
| qwen3_explain_l29_l31_l33_source | 33 | attn | 4 | 0.1618453823029995 | 0.1618453823029995 | 0.12848979702381808 |
