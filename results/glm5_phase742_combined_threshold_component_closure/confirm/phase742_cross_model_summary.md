# Phase 742 Combined Threshold Component Closure (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: cumulative component donor-delta add measured by threshold fraction and target top1 rate.

| model | condition | components | fraction | joint add effect | target top1 rate | margin donor vs top |
|---|---|---|---:|---:|---:|---:|
| qwen3 | joint_base |  | 0.004 | 0.000 | 0.000 | -15.250 |
| qwen3 | joint_add_top1 | L34:attn_out | 0.167 | 0.164 | 0.000 | -11.541 |
| qwen3 | joint_add_top2 | L34:attn_out,L31:attn_out | 0.279 | 0.275 | 0.000 | -8.316 |
| qwen3 | joint_add_top3 | L34:attn_out,L31:attn_out,L33:mlp_out | 0.312 | 0.309 | 0.000 | -7.534 |
| glm4 | joint_base |  | 0.029 | 0.000 | 0.000 | -7.825 |
| glm4 | joint_add_top1 | L38:mlp_out | 0.700 | 0.671 | 0.200 | -1.678 |
| glm4 | joint_add_top2 | L38:mlp_out,L39:mlp_out | 0.877 | 0.848 | 0.300 | -1.116 |
| glm4 | joint_add_top3 | L38:mlp_out,L39:mlp_out,L37:mlp_out | 0.931 | 0.901 | 0.500 | -0.706 |
| deepseek7b | joint_base |  | 0.018 | 0.000 | 0.000 | -9.863 |
| deepseek7b | joint_add_top1 | L26:attn_out | 0.328 | 0.310 | 0.000 | -5.125 |
| deepseek7b | joint_add_top2 | L26:attn_out,L27:attn_out | 0.553 | 0.536 | 0.100 | -2.513 |
| deepseek7b | joint_add_top3 | L26:attn_out,L27:attn_out,L27:mlp_out | 0.587 | 0.569 | 0.050 | -2.356 |

## Strict Interpretation

- If joint_add_topK reaches fraction near or above 1, the validated components are close to sufficient for readout closure.
- If it remains below 1, the missing mechanism is probably competitor/format suppression or final readout geometry, not merely these visible components.
- Whole-component cumulative edits remain coarse and can be off-manifold.

Atlas graph: nodes=15 edges=12
