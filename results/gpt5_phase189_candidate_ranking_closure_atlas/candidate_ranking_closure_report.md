# Phase189 Candidate Ranking Closure Atlas Report

This report consumes existing Phase591-597 results. It does not rerun CUDA models.

## Evidence Counts

- evidence rows: 384

## Causal Levels

| level | count |
|---:|---:|
| 1 | 5 |
| 2 | 247 |
| 3 | 111 |
| 4 | 21 |

## Closure Statuses

| status | count |
|---|---:|
| patch_no_specific_effect | 136 |
| projection_or_weak_transition | 56 |
| transition_candidate | 49 |
| patch_inconclusive | 37 |
| strong_projection_not_causal | 28 |
| weak_projection_not_causal | 27 |
| common_activation_only | 22 |
| weak_component_margin_or_switch | 19 |
| low_projection | 5 |
| strong_mlp_transition | 3 |
| weak_component_margin_no_switch | 2 |

## Gap Types

| gap | count |
|---|---:|
| component_patch_no_specific_ranking | 136 |
| transition_not_component_closure | 108 |
| projection_not_causal | 55 |
| ranking_unresolved | 42 |
| candidate_common_only | 22 |
| weak_component_candidate | 21 |

## Best Level 4 Component Candidates

| phase | model | key | status | score | source |
|---:|---|---|---|---:|---|
| 593 | qwen3 | `prompt_last|L30|specific_norm_raw` | weak_component_margin_or_switch | 0.250 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L32|specific_norm_raw` | weak_component_margin_or_switch | 0.200 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L34|specific_norm_raw` | weak_component_margin_or_switch | 0.200 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L33|specific_norm_raw` | weak_component_margin_or_switch | 0.150 | atlas_guided_patch |
| 595 | qwen3 | `prompt_last|L33|mlp|specific_norm_raw` | weak_component_margin_or_switch | 0.125 | mlp_output_patch |
| 595 | qwen3 | `prompt_last|L32|mlp|specific_norm_raw` | weak_component_margin_or_switch | 0.125 | mlp_output_patch |
| 593 | qwen3 | `prompt_last|L32|specific_only` | weak_component_margin_or_switch | 0.100 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L34|raw` | weak_component_margin_or_switch | 0.100 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L33|specific_only` | weak_component_margin_or_switch | 0.100 | atlas_guided_patch |
| 595 | qwen3 | `prompt_last|L30|mlp|common_norm_raw` | weak_component_margin_or_switch | 0.075 | mlp_output_patch |
| 595 | qwen3 | `prompt_last|L34|mlp|specific_only` | weak_component_margin_or_switch | 0.075 | mlp_output_patch |
| 593 | qwen3 | `prompt_last|L34|common_norm_raw` | weak_component_margin_or_switch | 0.075 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L30|specific_only` | weak_component_margin_or_switch | 0.075 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L30|random_same_norm` | weak_component_margin_no_switch | 0.075 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L33|raw` | weak_component_margin_or_switch | 0.050 | atlas_guided_patch |
| 595 | qwen3 | `prompt_last|L33|mlp|raw` | weak_component_margin_or_switch | 0.050 | mlp_output_patch |
| 595 | qwen3 | `prompt_last|L33|mlp|common_norm_raw` | weak_component_margin_or_switch | 0.050 | mlp_output_patch |
| 593 | qwen3 | `query_category|L32|raw` | weak_component_margin_no_switch | 0.050 | atlas_guided_patch |
| 593 | qwen3 | `prompt_last|L32|common_only` | weak_component_margin_or_switch | 0.050 | atlas_guided_patch |
| 595 | qwen3 | `query_category|L32|mlp|common_norm_raw` | weak_component_margin_or_switch | 0.050 | mlp_output_patch |

## DS7B Critical Path

| phase | key | level | status | gap | metrics |
|---:|---|---:|---|---|---|
| 592 | `query_relation|L19|late_mid` | 2 | weak_projection_not_causal | projection_not_causal | `{"mean_correct_specific": 0.3135881416854404, "mean_old_top_wrong_specific": -0.1843544833716892, "mean_specific_margin": 0.4979426250571296, "positive_specific_rate": 0.6666666666` |
| 592 | `rule_value|L26|late` | 2 | weak_projection_not_causal | projection_not_causal | `{"mean_correct_specific": 0.7179522713025411, "mean_old_top_wrong_specific": -0.49218092929749263, "mean_specific_margin": 1.2101332006000338, "positive_specific_rate": 0.428571428` |
| 593 | `query_relation|L19|common_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.029733873970274414, "mean_specific_margin_gain": -0.029733873970274414, "mean_common_delta": -0.15005418770791343, "mean_co` |
| 593 | `query_relation|L19|common_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.010043446307203599, "mean_specific_margin_gain": -0.010043446307203599, "mean_common_delta": 0.007110547828709795, "mean_co` |
| 593 | `query_relation|L19|random_same_norm` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.037602904552061646, "mean_specific_margin_gain": -0.037602904552061646, "mean_common_delta": -0.23619643485428587, "mean_co` |
| 593 | `query_relation|L19|raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.03859521322218435, "mean_specific_margin_gain": -0.03859521322218435, "mean_common_delta": 0.10572433985015821, "mean_corre` |
| 593 | `query_relation|L19|specific_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.027180648852317107, "mean_specific_margin_gain": -0.027180648852317107, "mean_common_delta": -0.43662034806108013, "mean_co` |
| 593 | `query_relation|L19|specific_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.013234375233185432, "mean_specific_margin_gain": -0.013234375233185432, "mean_common_delta": -0.010849711737440279, "mean_c` |
| 593 | `rule_value|L26|common_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.01562700619078463, "mean_specific_margin_gain": -0.01562700619078463, "mean_common_delta": 0.008310958890000447, "mean_corr` |
| 593 | `rule_value|L26|common_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.011862882401882893, "mean_specific_margin_gain": -0.011862882401882893, "mean_common_delta": -0.0071852755638593365, "mean_` |
| 593 | `rule_value|L26|random_same_norm` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.0027990108577623254, "mean_specific_margin_gain": -0.0027990108577623254, "mean_common_delta": 0.0018185421303358105, "mean` |
| 593 | `rule_value|L26|raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.005859264882192725, "mean_specific_margin_gain": -0.005859264882192725, "mean_common_delta": 0.002075058507866093, "mean_co` |
| 593 | `rule_value|L26|specific_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.009084221008898956, "mean_specific_margin_gain": -0.009084221008898956, "mean_common_delta": 0.00455380862279396, "mean_cor` |
| 593 | `rule_value|L26|specific_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.008659094660764649, "mean_specific_margin_gain": -0.008659094660764649, "mean_common_delta": -0.00585917214907351, "mean_co` |
| 594 | `query_relation|L19|attn_update` | 2 | projection_or_weak_transition | transition_not_component_closure | `{"mean_correct_specific": -0.001594262286311104, "mean_old_top_wrong_specific": 0.10556402039669809, "mean_specific_margin": -0.1071582826830092, "positive_rate": 0.285714285714285` |
| 594 | `query_relation|L19|incoming` | 2 | projection_or_weak_transition | transition_not_component_closure | `{"mean_correct_specific": 0.12317643421036857, "mean_old_top_wrong_specific": 0.035125637338275, "mean_specific_margin": 0.08805079687209356, "positive_rate": 0.5238095238095238}` |
| 594 | `query_relation|L19|mlp_update` | 3 | strong_mlp_transition | transition_not_component_closure | `{"mean_correct_specific": 0.1951474485297998, "mean_old_top_wrong_specific": -0.32405381454598337, "mean_specific_margin": 0.5192012630757832, "positive_rate": 0.8571428571428571}` |
| 594 | `query_relation|L19|outgoing` | 3 | transition_candidate | transition_not_component_closure | `{"mean_correct_specific": 0.3135881416854404, "mean_old_top_wrong_specific": -0.1843544833716892, "mean_specific_margin": 0.4979426250571296, "positive_rate": 0.6666666666666666}` |
| 594 | `query_relation|L19|residual_update` | 3 | transition_candidate | transition_not_component_closure | `{"mean_correct_specific": 0.19041167855972335, "mean_old_top_wrong_specific": -0.2194803050231366, "mean_specific_margin": 0.40989198358285994, "positive_rate": 0.7619047619047619}` |
| 594 | `query_relation|L19|transition_gain` | 2 | projection_or_weak_transition | transition_not_component_closure | `{"positive_rate": 0.7619047619047619}` |
| 594 | `rule_value|L26|attn_update` | 2 | projection_or_weak_transition | transition_not_component_closure | `{"mean_correct_specific": -0.07823584015880312, "mean_old_top_wrong_specific": 0.5862197297669592, "mean_specific_margin": -0.6644555699257624, "positive_rate": 0.2857142857142857}` |
| 594 | `rule_value|L26|incoming` | 3 | transition_candidate | transition_not_component_closure | `{"mean_correct_specific": 0.468951553106308, "mean_old_top_wrong_specific": -0.14693957992962428, "mean_specific_margin": 0.6158911330359322, "positive_rate": 0.6190476190476191}` |
| 594 | `rule_value|L26|mlp_update` | 3 | strong_mlp_transition | transition_not_component_closure | `{"mean_correct_specific": 0.3085373796167828, "mean_old_top_wrong_specific": -0.897167485384714, "mean_specific_margin": 1.2057048650014968, "positive_rate": 0.7619047619047619}` |
| 594 | `rule_value|L26|outgoing` | 3 | transition_candidate | transition_not_component_closure | `{"mean_correct_specific": 0.7179522713025411, "mean_old_top_wrong_specific": -0.49218092929749263, "mean_specific_margin": 1.2101332006000338, "positive_rate": 0.42857142857142855}` |
| 594 | `rule_value|L26|residual_update` | 3 | transition_candidate | transition_not_component_closure | `{"mean_correct_specific": 0.24900052235240028, "mean_old_top_wrong_specific": -0.34524194257599966, "mean_specific_margin": 0.5942424649283999, "positive_rate": 0.47619047619047616` |
| 594 | `rule_value|L26|transition_gain` | 2 | projection_or_weak_transition | transition_not_component_closure | `{"positive_rate": 0.47619047619047616}` |
| 595 | `query_relation|L19|mlp|common_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.020065088434854432, "mean_specific_margin_gain": -0.020065088434854432, "mean_common_delta": 0.07102533613076611, "mean_cor` |
| 595 | `query_relation|L19|mlp|common_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.04551215414401321, "mean_specific_margin_gain": -0.04551215414401321, "mean_common_delta": 0.005711420577773381, "mean_corr` |
| 595 | `query_relation|L19|mlp|random_same_norm` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.06658284604505059, "mean_specific_margin_gain": -0.06658284604505059, "mean_common_delta": -0.15699463510342562, "mean_corr` |
| 595 | `query_relation|L19|mlp|raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.016166207391679996, "mean_specific_margin_gain": -0.016166207391679996, "mean_common_delta": 0.22885862312423774, "mean_cor` |
| 595 | `query_relation|L19|mlp|specific_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.026338816598235143, "mean_specific_margin_gain": -0.026338816598235143, "mean_common_delta": -0.09441556654582244, "mean_co` |
| 595 | `query_relation|L19|mlp|specific_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.03598229560468878, "mean_specific_margin_gain": -0.03598229560468878, "mean_common_delta": -0.018330513266846538, "mean_cor` |
| 595 | `rule_value|L26|mlp|common_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.012591222999617457, "mean_specific_margin_gain": -0.012591222999617457, "mean_common_delta": -0.00032928923887777186, "mean` |
| 595 | `rule_value|L26|mlp|common_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -6.273378884153706e-05, "mean_specific_margin_gain": -6.273378884153706e-05, "mean_common_delta": -0.0028549587836356033, "mea` |
| 595 | `rule_value|L26|mlp|random_same_norm` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.0058051427892808405, "mean_specific_margin_gain": -0.0058051427892808405, "mean_common_delta": 0.005389644246038404, "mean_` |
| 595 | `rule_value|L26|mlp|raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.008706996915861964, "mean_specific_margin_gain": -0.008706996915861964, "mean_common_delta": 0.0001766852164153187, "mean_c` |
| 595 | `rule_value|L26|mlp|specific_norm_raw` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.006623194586219532, "mean_specific_margin_gain": -0.006623194586219532, "mean_common_delta": -0.0016235046981212993, "mean_` |
| 595 | `rule_value|L26|mlp|specific_only` | 2 | patch_no_specific_effect | component_patch_no_specific_ranking | `{"switch": 0, "switch_rate": 0.0, "mean_margin_gain": -0.012213414967326182, "mean_specific_margin_gain": -0.012213414967326182, "mean_common_delta": -0.0030964925147903464, "mean_` |

## Main Conclusion

- Prompt-level repair can produce strong candidate-specific ranking, but static hidden/component patches mostly fail to transfer it.
- DS7B rule_value L26 and query_relation L19 remain the best mechanistic handles, but current evidence is still not Level 5 repair.
- The next useful experiment should isolate component/channel state conditions rather than adding another broad residual or MLP-output delta.

## Recommended Next Move

Run a small but targeted channel-state intervention on DS7B first, then only scale to Qwen3/GLM4 if DS7B shows a Level4 margin effect above controls.
