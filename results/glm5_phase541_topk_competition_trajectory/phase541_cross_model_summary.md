# Phase541 Top-K Competition Trajectory Summary

## qwen3

top_k=20, core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [10, 12, 14], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], attn=sdpa

| source | condition | win | alpha | trajectory | class |
|---|---|---|---:|---|---|
| vehicle_furniture | residual_perp | extended | 6.0 | targetLogit +0.362, compLogit -0.957, targetRank -52.6, compRank +375.6, churn 0.15, topOtherΔ -0.101, topOtherRank +1.3 | specific_competitor_suppression |
| vehicle_furniture | residual_parallel | extended | 6.0 | targetLogit +0.909, compLogit -0.367, targetRank -111.0, compRank +143.3, churn 0.13, topOtherΔ -0.150, topOtherRank +1.2 | mixed_or_weak |
| vehicle_furniture | residual_full | extended | 6.0 | targetLogit +0.438, compLogit -0.961, targetRank -60.3, compRank +383.7, churn 0.15, topOtherΔ -0.086, topOtherRank +1.3 | specific_competitor_suppression |
| vehicle_tool | residual_perp | extended | 6.0 | targetLogit +0.414, compLogit -0.908, targetRank -66.5, compRank +327.5, churn 0.13, topOtherΔ -0.348, topOtherRank +1.2 | specific_competitor_suppression |
| vehicle_tool | residual_parallel | extended | 6.0 | targetLogit +0.763, compLogit -0.497, targetRank -106.1, compRank +178.1, churn 0.12, topOtherΔ -0.169, topOtherRank +1.1 | mixed_or_weak |
| vehicle_tool | residual_full | extended | 6.0 | targetLogit +0.492, compLogit -0.944, targetRank -76.5, compRank +351.6, churn 0.14, topOtherΔ -0.341, topOtherRank +1.2 | specific_competitor_suppression |
| vehicle_clothing | residual_perp | extended | 6.0 | targetLogit +0.365, compLogit -0.844, targetRank -41.5, compRank +455.1, churn 0.13, topOtherΔ +0.122, topOtherRank +0.9 | specific_competitor_suppression |
| vehicle_clothing | residual_parallel | extended | 6.0 | targetLogit +0.823, compLogit -0.530, targetRank -110.0, compRank +189.0, churn 0.15, topOtherΔ -0.280, topOtherRank +1.9 | specific_competitor_suppression |
| vehicle_clothing | residual_full | extended | 6.0 | targetLogit +0.419, compLogit -0.893, targetRank -50.0, compRank +474.6, churn 0.13, topOtherΔ +0.095, topOtherRank +0.9 | specific_competitor_suppression |

## glm4

top_k=20, core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [24, 26, 28], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], attn=sdpa

| source | condition | win | alpha | trajectory | class |
|---|---|---|---:|---|---|
| vehicle_furniture | residual_perp | extended | 6.0 | targetLogit +1.618, compLogit -0.316, targetRank -854.3, compRank +105.8, churn 0.35, topOtherΔ -0.156, topOtherRank +20.5 | mixed_or_weak |
| vehicle_furniture | residual_parallel | center | 6.0 | targetLogit +2.486, compLogit -2.143, targetRank -967.1, compRank +1331.9, churn 0.27, topOtherΔ -0.674, topOtherRank +7.6 | specific_competitor_suppression |
| vehicle_furniture | residual_full | extended | 6.0 | targetLogit +2.329, compLogit -0.713, targetRank -927.0, compRank +261.7, churn 0.35, topOtherΔ -0.198, topOtherRank +21.4 | specific_competitor_suppression |
| vehicle_tool | residual_perp | center | 6.0 | targetLogit +0.974, compLogit -1.213, targetRank -844.0, compRank +815.0, churn 0.30, topOtherΔ -0.297, topOtherRank +8.0 | specific_competitor_suppression |
| vehicle_tool | residual_parallel | extended | 6.0 | targetLogit +2.565, compLogit -4.798, targetRank -962.6, compRank +37699.4, churn 0.30, topOtherΔ -0.826, topOtherRank +20.7 | specific_competitor_suppression |
| vehicle_tool | residual_full | extended | 6.0 | targetLogit +1.529, compLogit -1.858, targetRank -923.4, compRank +2415.1, churn 0.34, topOtherΔ -0.480, topOtherRank +13.0 | specific_competitor_suppression |
| vehicle_clothing | residual_perp | extended | 6.0 | targetLogit +0.792, compLogit -1.522, targetRank -741.8, compRank +4924.8, churn 0.33, topOtherΔ -0.395, topOtherRank +15.9 | specific_competitor_suppression |
| vehicle_clothing | residual_parallel | extended | 6.0 | targetLogit +1.689, compLogit -4.576, targetRank -923.2, compRank +46853.2, churn 0.28, topOtherΔ -0.758, topOtherRank +13.9 | specific_competitor_suppression |
| vehicle_clothing | residual_full | extended | 6.0 | targetLogit +1.483, compLogit -2.013, targetRank -865.0, compRank +7792.3, churn 0.33, topOtherΔ -0.422, topOtherRank +17.0 | specific_competitor_suppression |

## deepseek7b

top_k=20, core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [16, 18, 20], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], attn=sdpa

| source | condition | win | alpha | trajectory | class |
|---|---|---|---:|---|---|
| vehicle_furniture | residual_perp | extended | 6.0 | targetLogit +0.136, compLogit -0.115, targetRank -2833.7, compRank -75.8, churn 0.08, topOtherΔ -0.102, topOtherRank +0.3 | mixed_or_weak |
| vehicle_furniture | residual_parallel | extended | 6.0 | targetLogit +0.625, compLogit -0.158, targetRank -8455.8, compRank +250.0, churn 0.05, topOtherΔ +0.102, topOtherRank +0.0 | mixed_or_weak |
| vehicle_furniture | residual_full | extended | 6.0 | targetLogit +0.115, compLogit -0.143, targetRank -2798.4, compRank -80.5, churn 0.08, topOtherΔ -0.141, topOtherRank +0.4 | mixed_or_weak |
| vehicle_tool | residual_perp | extended | 6.0 | targetLogit +0.094, compLogit -0.327, targetRank -2958.0, compRank +292.2, churn 0.07, topOtherΔ -0.157, topOtherRank +0.3 | mixed_or_weak |
| vehicle_tool | residual_parallel | extended | 6.0 | targetLogit +1.017, compLogit -0.402, targetRank -14203.4, compRank +1362.9, churn 0.07, topOtherΔ +0.094, topOtherRank +0.1 | mixed_or_weak |
| vehicle_tool | residual_full | extended | 6.0 | targetLogit +0.118, compLogit -0.350, targetRank -3378.6, compRank +329.6, churn 0.07, topOtherΔ -0.165, topOtherRank +0.2 | mixed_or_weak |
| vehicle_clothing | residual_perp | extended | 6.0 | targetLogit +0.197, compLogit -0.252, targetRank -4338.0, compRank +285.2, churn 0.07, topOtherΔ -0.079, topOtherRank +0.3 | mixed_or_weak |
| vehicle_clothing | residual_parallel | extended | 6.0 | targetLogit +0.602, compLogit -0.246, targetRank -9830.2, compRank +581.1, churn 0.04, topOtherΔ -0.028, topOtherRank -0.1 | mixed_or_weak |
| vehicle_clothing | residual_full | extended | 6.0 | targetLogit +0.208, compLogit -0.263, targetRank -4612.5, compRank +300.2, churn 0.07, topOtherΔ -0.094, topOtherRank +0.3 | mixed_or_weak |

## Residual Parallel Compact

| model | source | win | target logit | competitor logit | target rank Δ | competitor rank Δ | churn | top other Δ | class |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | vehicle_furniture | extended | +0.909 | -0.367 | -111.0 | +143.3 | 0.13 | -0.150 | mixed_or_weak |
| qwen3 | vehicle_tool | extended | +0.763 | -0.497 | -106.1 | +178.1 | 0.12 | -0.169 | mixed_or_weak |
| qwen3 | vehicle_clothing | extended | +0.823 | -0.530 | -110.0 | +189.0 | 0.15 | -0.280 | specific_competitor_suppression |
| glm4 | vehicle_furniture | center | +2.486 | -2.143 | -967.1 | +1331.9 | 0.27 | -0.674 | specific_competitor_suppression |
| glm4 | vehicle_tool | extended | +2.565 | -4.798 | -962.6 | +37699.4 | 0.30 | -0.826 | specific_competitor_suppression |
| glm4 | vehicle_clothing | extended | +1.689 | -4.576 | -923.2 | +46853.2 | 0.28 | -0.758 | specific_competitor_suppression |
| deepseek7b | vehicle_furniture | extended | +0.625 | -0.158 | -8455.8 | +250.0 | 0.05 | +0.102 | mixed_or_weak |
| deepseek7b | vehicle_tool | extended | +1.017 | -0.402 | -14203.4 | +1362.9 | 0.07 | +0.094 | mixed_or_weak |
| deepseek7b | vehicle_clothing | extended | +0.602 | -0.246 | -9830.2 | +581.1 | 0.04 | -0.028 | mixed_or_weak |

