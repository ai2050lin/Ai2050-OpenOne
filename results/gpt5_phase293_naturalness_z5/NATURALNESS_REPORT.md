# Phase 293 Naturalness Report
## Inputs
- phase290_dir: `results/gpt5_phase290_contract_break_full`
- phase291_dir: `results/gpt5_phase291_block_contract_full`
- z_threshold: `5.0`
- drop_threshold: `0.5`
- both_min: `0.4`

## Event Summary
### deepseek7b / phase290
- norm_normal_functional_failure: 214
- off_manifold_no_drop: 116
### deepseek7b / phase291
- norm_normal_functional_failure: 467
- off_manifold_functional_failure: 11
- off_manifold_no_drop: 221
### glm4 / phase290
- norm_normal_functional_failure: 543
### glm4 / phase291
- norm_normal_functional_failure: 831
### qwen3 / phase290
- norm_normal_functional_failure: 221
- off_manifold_functional_failure: 1
- off_manifold_no_drop: 88
### qwen3 / phase291
- norm_normal_functional_failure: 474

## Top Functional Failures
- deepseek7b phase291 L20-L27 cross_battn_amlp contrast pair=logic_although_late drop=1.1641 off=1 z=1.36 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp get_passive pair=pass_get_invited drop=1.1063 off=1 z=1.00 labels=off_manifold_functional_failure
- qwen3 phase290 L0 cross_battn_amlp contrast pair=logic_although_cold drop=1.0115 off=1 z=1.39 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp syntactic_do_not pair=neg_expect drop=0.9965 off=1 z=3.37 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp complement_clause pair=rec_comp_think_leave drop=0.9878 off=1 z=2.31 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp scope_quantifier pair=neg_not_all drop=0.9845 off=1 z=3.34 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L23 cross_aattn_bmlp no_agent pair=pass_package_delivered drop=0.8893 off=1 z=1.84 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L23 cross_battn_amlp no_agent pair=pass_package_delivered drop=0.7421 off=1 z=2.62 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp never pair=neg_forgets_face drop=0.6760 off=1 z=1.16 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_aattn_bmlp no_agent pair=pass_window_broken drop=0.6285 off=1 z=1.93 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp no_agent pair=pass_car_repaired drop=0.5911 off=1 z=2.08 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_aattn_bmlp get_passive pair=pass_get_invited drop=0.5377 off=1 z=1.55 labels=off_manifold_functional_failure
- deepseek7b phase290 L25 cross_battn_amlp existential_no pair=neg_no_idea drop=2.5447 off=0 z=1.76 labels=norm_normal_functional_failure
- deepseek7b phase291 L20-L23 cross_battn_amlp existential_no pair=neg_no_idea drop=2.5334 off=0 z=1.02 labels=norm_normal_functional_failure
- deepseek7b phase291 L20-L23 cross_battn_amlp causal pair=logic_because_tired drop=2.3037 off=0 z=0.98 labels=norm_normal_functional_failure
- qwen3 phase291 L0-L8 cross_battn_amlp no_agent pair=pass_window_broken drop=2.0209 off=0 z=2.09 labels=norm_normal_functional_failure
- deepseek7b phase290 L22 cross_battn_amlp existential_no pair=neg_no_idea drop=2.0150 off=0 z=1.71 labels=norm_normal_functional_failure
- deepseek7b phase290 L20 cross_battn_amlp and_or pair=logic_and_or_milk drop=2.0075 off=0 z=1.53 labels=norm_normal_functional_failure
- deepseek7b phase290 L26 cross_battn_amlp conditional pair=logic_if_hungry drop=1.9250 off=0 z=2.36 labels=norm_normal_functional_failure
- deepseek7b phase290 L22 cross_battn_amlp never pair=neg_gives_up drop=1.9126 off=0 z=1.82 labels=norm_normal_functional_failure
- deepseek7b phase290 L25 cross_battn_amlp conditional pair=logic_if_hungry drop=1.8899 off=0 z=1.77 labels=norm_normal_functional_failure
- qwen3 phase291 L4-L8 cross_battn_amlp no_agent pair=pass_window_broken drop=1.8871 off=0 z=2.09 labels=norm_normal_functional_failure
- deepseek7b phase290 L22 cross_aattn_bmlp existential_no pair=neg_no_idea drop=1.8424 off=0 z=1.71 labels=norm_normal_functional_failure
- deepseek7b phase291 L20-L23 cross_battn_amlp never pair=neg_late drop=1.7323 off=0 z=1.66 labels=norm_normal_functional_failure
- qwen3 phase291 L0-L2 cross_battn_amlp no_agent pair=pass_window_broken drop=1.6846 off=0 z=2.03 labels=norm_normal_functional_failure
- deepseek7b phase290 L26 cross_battn_amlp conditional pair=logic_if_safe drop=1.6710 off=0 z=2.34 labels=norm_normal_functional_failure
- qwen3 phase291 L0-L8 cross_battn_amlp contrast pair=logic_although_expensive drop=1.6346 off=0 z=2.44 labels=norm_normal_functional_failure
- deepseek7b phase290 L26 cross_battn_amlp contrast pair=logic_although_tired drop=1.6096 off=0 z=0.82 labels=norm_normal_functional_failure
- qwen3 phase291 L4-L8 cross_battn_amlp lexical_not_adj pair=neg_clear drop=1.6060 off=0 z=1.53 labels=norm_normal_functional_failure
- qwen3 phase291 L0-L4 cross_battn_amlp no_agent pair=pass_window_broken drop=1.6034 off=0 z=2.03 labels=norm_normal_functional_failure

## Caution
- This is norm-based naturalness only; it is not PCA, kNN, or Mahalanobis manifold distance.
- A norm-normal functional failure is stronger than an off-manifold failure, but still not final proof of an internal contract.
