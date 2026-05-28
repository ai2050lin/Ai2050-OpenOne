# Phase 293 Naturalness Report
## Inputs
- phase290_dir: `results/gpt5_phase290_contract_break_full`
- phase291_dir: `results/gpt5_phase291_block_contract_full`
- z_threshold: `3.0`
- drop_threshold: `0.5`
- both_min: `0.4`

## Event Summary
### deepseek7b / phase290
- norm_normal_functional_failure: 211
- off_manifold_functional_failure: 3
- off_manifold_no_drop: 308
### deepseek7b / phase291
- norm_normal_functional_failure: 442
- off_manifold_functional_failure: 36
- off_manifold_no_drop: 501
### glm4 / phase290
- norm_normal_functional_failure: 537
- off_manifold_functional_failure: 6
- off_manifold_no_drop: 122
### glm4 / phase291
- norm_normal_functional_failure: 801
- off_manifold_functional_failure: 30
- off_manifold_no_drop: 334
### qwen3 / phase290
- norm_normal_functional_failure: 221
- off_manifold_functional_failure: 1
- off_manifold_no_drop: 146
### qwen3 / phase291
- norm_normal_functional_failure: 474
- off_manifold_no_drop: 114

## Top Functional Failures
- deepseek7b phase290 L26 cross_battn_amlp scope_quantifier pair=neg_not_all drop=1.5338 off=1 z=3.34 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp contrast pair=logic_although_late drop=1.1641 off=1 z=1.36 labels=off_manifold_functional_failure
- deepseek7b phase291 L26-L27 cross_battn_amlp complement_clause pair=rec_comp_know_safe drop=1.1291 off=1 z=3.33 labels=off_manifold_functional_failure
- deepseek7b phase291 L24-L27 cross_battn_amlp complement_clause pair=rec_comp_know_safe drop=1.1280 off=1 z=3.33 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp get_passive pair=pass_get_invited drop=1.1063 off=1 z=1.00 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp complement_clause pair=rec_comp_know_safe drop=1.0985 off=1 z=3.33 labels=off_manifold_functional_failure
- deepseek7b phase291 L24-L27 cross_battn_amlp scope_quantifier pair=neg_not_all drop=1.0772 off=1 z=3.34 labels=off_manifold_functional_failure
- deepseek7b phase291 L26-L27 cross_battn_amlp scope_quantifier pair=neg_not_all drop=1.0136 off=1 z=3.34 labels=off_manifold_functional_failure
- deepseek7b phase291 L24-L27 cross_battn_amlp syntactic_do_not pair=neg_understand drop=1.0130 off=1 z=3.36 labels=off_manifold_functional_failure
- qwen3 phase290 L0 cross_battn_amlp contrast pair=logic_although_cold drop=1.0115 off=1 z=1.39 labels=off_manifold_functional_failure
- deepseek7b phase291 L26-L27 cross_battn_amlp syntactic_do_not pair=neg_expect drop=1.0003 off=1 z=3.37 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp syntactic_do_not pair=neg_expect drop=0.9965 off=1 z=3.37 labels=off_manifold_functional_failure
- glm4 phase290 L0 cross_battn_amlp causal pair=logic_because_dark drop=0.9928 off=1 z=3.51 labels=off_manifold_functional_failure
- glm4 phase291 L0 cross_battn_amlp causal pair=logic_because_dark drop=0.9928 off=1 z=3.51 labels=off_manifold_functional_failure
- glm4 phase291 L0-L2 cross_battn_amlp causal pair=logic_because_dark drop=0.9926 off=1 z=3.51 labels=off_manifold_functional_failure
- deepseek7b phase291 L26-L27 cross_battn_amlp inference pair=logic_therefore_rain drop=0.9919 off=1 z=3.51 labels=off_manifold_functional_failure
- glm4 phase291 L0-L1 cross_battn_amlp causal pair=logic_because_dark drop=0.9917 off=1 z=3.51 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp complement_clause pair=rec_comp_think_leave drop=0.9878 off=1 z=2.31 labels=off_manifold_functional_failure
- glm4 phase291 L0-L4 cross_battn_amlp inference pair=logic_therefore_rain drop=0.9871 off=1 z=4.99 labels=off_manifold_functional_failure
- glm4 phase291 L0-L4 cross_battn_amlp inference pair=logic_therefore_study drop=0.9861 off=1 z=3.04 labels=off_manifold_functional_failure
- deepseek7b phase291 L20-L27 cross_battn_amlp scope_quantifier pair=neg_not_all drop=0.9845 off=1 z=3.34 labels=off_manifold_functional_failure
- glm4 phase291 L0-L8 cross_battn_amlp inference pair=logic_therefore_rain drop=0.9780 off=1 z=4.99 labels=off_manifold_functional_failure
- glm4 phase290 L4 cross_battn_amlp complement_clause pair=rec_comp_fear_lost drop=0.9767 off=1 z=3.15 labels=off_manifold_functional_failure
- deepseek7b phase291 L24-L27 cross_battn_amlp syntactic_do_not pair=neg_expect drop=0.9735 off=1 z=3.37 labels=off_manifold_functional_failure
- glm4 phase291 L4-L8 cross_battn_amlp complement_clause pair=rec_comp_expect_arrive drop=0.9686 off=1 z=3.13 labels=off_manifold_functional_failure
- glm4 phase291 L0-L4 cross_battn_amlp causal pair=logic_because_dark drop=0.9677 off=1 z=3.51 labels=off_manifold_functional_failure
- deepseek7b phase291 L26-L27 cross_battn_amlp syntactic_do_not pair=neg_understand drop=0.9502 off=1 z=3.36 labels=off_manifold_functional_failure
- glm4 phase291 L0-L8 cross_battn_amlp complement_clause pair=rec_comp_expect_arrive drop=0.9465 off=1 z=3.13 labels=off_manifold_functional_failure
- deepseek7b phase290 L27 cross_battn_amlp syntactic_do_not pair=neg_expect drop=0.9456 off=1 z=3.08 labels=off_manifold_functional_failure
- deepseek7b phase291 L27 cross_battn_amlp syntactic_do_not pair=neg_expect drop=0.9456 off=1 z=3.08 labels=off_manifold_functional_failure

## Caution
- This is norm-based naturalness only; it is not PCA, kNN, or Mahalanobis manifold distance.
- A norm-normal functional failure is stronger than an off-manifold failure, but still not final proof of an internal contract.
