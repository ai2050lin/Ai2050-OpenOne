# Phase 742 Combined Threshold Component Closure (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: cumulative component donor-delta add measured by threshold fraction and target top1 rate.

| model | condition | components | fraction | joint add effect | target top1 rate | margin donor vs top |
|---|---|---|---:|---:|---:|---:|
| qwen3 | joint_base |  | -0.005 | 0.000 | 0.000 | -19.250 |
| qwen3 | joint_add_top1 | L34:attn_out | 0.029 | 0.035 | 0.000 | -17.812 |
| qwen3 | joint_add_top2 | L34:attn_out,L31:attn_out | 0.304 | 0.309 | 0.000 | -7.250 |
| qwen3 | joint_add_top3 | L34:attn_out,L31:attn_out,L33:mlp_out | 0.305 | 0.310 | 0.000 | -7.250 |
| glm4 | joint_base |  | 0.060 | 0.000 | 0.000 | -5.875 |
| glm4 | joint_add_top1 | L38:mlp_out | 1.104 | 1.044 | 1.000 | 0.000 |
| glm4 | joint_add_top2 | L38:mlp_out,L39:mlp_out | 1.435 | 1.375 | 1.000 | 0.000 |
| glm4 | joint_add_top3 | L38:mlp_out,L39:mlp_out,L37:mlp_out | 1.453 | 1.393 | 1.000 | 0.000 |
| deepseek7b | joint_base |  | 0.031 | 0.000 | 0.000 | -7.500 |
| deepseek7b | joint_add_top1 | L26:attn_out | 0.715 | 0.685 | 0.000 | -4.438 |
| deepseek7b | joint_add_top2 | L26:attn_out,L27:attn_out | 1.279 | 1.249 | 0.000 | -3.250 |
| deepseek7b | joint_add_top3 | L26:attn_out,L27:attn_out,L27:mlp_out | 1.205 | 1.174 | 0.000 | -2.500 |

## Strict Interpretation

- If joint_add_topK reaches fraction near or above 1, the validated components are close to sufficient for readout closure.
- If it remains below 1, the missing mechanism is probably competitor/format suppression or final readout geometry, not merely these visible components.
- Whole-component cumulative edits remain coarse and can be off-manifold.

Atlas graph: nodes=15 edges=12
