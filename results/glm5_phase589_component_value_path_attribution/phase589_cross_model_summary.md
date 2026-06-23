# Phase 589 Component-Level Value Path Attribution Summary

Confirm setting: 24 value cases per model, prompt_last component-output patch at two late layers.

| model | target cases | best component | layer | switch | correct gain | top-wrong gain | margin gain |
|---|---:|---|---:|---:|---:|---:|---:|
| qwen3 | 2 | residual | L27 | 1/2 (50.0%) | 0.587 | 0.400 | 0.188 |
| glm4 | 1 | residual | L30 | 0/1 (0.0%) | -0.316 | -0.379 | 0.063 |
| deepseek7b | 9 | residual | L21 | 0/9 (0.0%) | 4.442 | 4.434 | 0.008 |

## DS7B Component Details

| component | layer | switch | correct gain | top-wrong gain | margin gain |
|---|---:|---:|---:|---:|---:|
| attn | L21 | 0/9 (0.0%) | 0.203 | 0.223 | -0.020 |
| attn | L26 | 0/9 (0.0%) | 1.835 | 1.863 | -0.027 |
| mlp | L21 | 0/9 (0.0%) | -0.234 | -0.193 | -0.041 |
| mlp | L26 | 0/9 (0.0%) | 0.060 | 0.094 | -0.034 |
| residual | L21 | 0/9 (0.0%) | 4.442 | 4.434 | 0.008 |
| residual | L26 | 0/9 (0.0%) | 6.205 | 6.254 | -0.049 |

## Objective Facts

- DS7B residual output carries the strongest value candidate co-activation: L26 correct +6.205, top-wrong +6.254, margin -0.049.
- DS7B attention output also co-activates candidates, especially L26: correct +1.835, top-wrong +1.863.
- DS7B MLP output does not improve margin and is weak/negative in this patch setup.
- No component produces winner switch or positive margin control on DS7B.
- Therefore candidate co-activation is visible at component level, but winner selection is still unresolved.
