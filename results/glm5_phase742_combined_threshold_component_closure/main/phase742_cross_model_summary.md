# Phase 742 Combined Threshold Component Closure (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: cumulative component donor-delta add measured by threshold fraction and target top1 rate.

| model | condition | components | fraction | joint add effect | target top1 rate | margin donor vs top |
|---|---|---|---:|---:|---:|---:|
| qwen3 | joint_base |  | 0.003 | 0.000 | 0.000 | -15.995 |
| qwen3 | joint_add_top1 | L34:attn_out | 0.169 | 0.166 | 0.000 | -12.089 |
| qwen3 | joint_add_top2 | L34:attn_out,L31:attn_out | 0.317 | 0.314 | 0.000 | -7.536 |
| qwen3 | joint_add_top3 | L34:attn_out,L31:attn_out,L33:mlp_out | 0.332 | 0.329 | 0.000 | -7.120 |
| glm4 | joint_base |  | 0.028 | 0.000 | 0.000 | -8.479 |
| glm4 | joint_add_top1 | L38:mlp_out | 0.699 | 0.670 | 0.167 | -1.927 |
| glm4 | joint_add_top2 | L38:mlp_out,L39:mlp_out | 0.894 | 0.866 | 0.333 | -1.396 |
| glm4 | joint_add_top3 | L38:mlp_out,L39:mlp_out,L37:mlp_out | 0.942 | 0.914 | 0.500 | -1.089 |
| deepseek7b | joint_base |  | 0.013 | 0.000 | 0.000 | -10.261 |
| deepseek7b | joint_add_top1 | L26:attn_out | 0.304 | 0.291 | 0.000 | -5.833 |
| deepseek7b | joint_add_top2 | L26:attn_out,L27:attn_out | 0.561 | 0.548 | 0.083 | -2.719 |
| deepseek7b | joint_add_top3 | L26:attn_out,L27:attn_out,L27:mlp_out | 0.585 | 0.572 | 0.083 | -2.760 |

## Strict Interpretation

- If joint_add_topK reaches fraction near or above 1, the validated components are close to sufficient for readout closure.
- If it remains below 1, the missing mechanism is probably competitor/format suppression or final readout geometry, not merely these visible components.
- Whole-component cumulative edits remain coarse and can be off-manifold.

Atlas graph: nodes=15 edges=12
