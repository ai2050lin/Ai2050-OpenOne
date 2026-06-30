# Phase 799 Blocker-Field Causal Suppressor Localization (smoke)

- Status: `complete`
- Boundary: scores candidate fibers by target gain, identity-anchor improvement, baseline blocker suppression, and new-blocker penalty.
- This phase gives suppressor candidates, not final token closure.

## By Model

| model | rows | cases | target gain | blocker suppression | target-relative lift | new blocker rate | resolved rate | anchor gap | token gain | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | 1 | 2.938 | 1.093 | 4.031 | 0.065 | 0.838 | 4.312 | 0.000 | 1.854 |
| glm4 | 2 | 1 | -0.219 | 0.797 | 0.578 | 0.321 | 0.299 | 1.469 | 0.000 | 0.275 |
| deepseek7b | 4 | 1 | 3.094 | -1.164 | 1.930 | 0.094 | 0.653 | 2.641 | 0.000 | 0.639 |

## Top Suppressor Candidates

| model | component | selection | ladder | source group | rows | target gain | blocker suppression | target-relative lift | new rate | resolved rate | anchor gap | score |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `attn:L35` | `top` | `route_answer` | `all_pre_answer` | 1 | 2.938 | 1.093 | 4.031 | 0.065 | 0.838 | 4.312 | 1.854 |
| qwen3 | `attn:L35` | `top` | `kv_o_route` | `all_pre_answer` | 1 | 2.938 | 1.093 | 4.031 | 0.065 | 0.838 | 4.312 | 1.854 |
| qwen3 | `attn:L35` | `matched` | `route_answer` | `all_pre_answer` | 1 | 2.938 | 1.093 | 4.031 | 0.065 | 0.838 | 4.312 | 1.854 |
| qwen3 | `attn:L35` | `matched` | `kv_o_route` | `all_pre_answer` | 1 | 2.938 | 1.093 | 4.031 | 0.065 | 0.838 | 4.312 | 1.854 |
| glm4 | `route_only:L38` | `top` | `route_answer` | `all_pre_answer` | 1 | -0.219 | 0.797 | 0.578 | 0.321 | 0.299 | 1.469 | 0.275 |
| glm4 | `route_only:L38` | `matched` | `route_answer` | `all_pre_answer` | 1 | -0.219 | 0.797 | 0.578 | 0.321 | 0.299 | 1.469 | 0.275 |
| deepseek7b | `attn:L19` | `matched` | `kv_o_route` | `all_pre_answer` | 1 | 3.172 | -1.134 | 2.038 | 0.112 | 0.665 | 3.109 | 0.719 |
| deepseek7b | `attn:L19` | `top` | `route_answer` | `all_pre_answer` | 1 | 3.172 | -1.188 | 1.984 | 0.066 | 0.668 | 2.422 | 0.625 |
| deepseek7b | `attn:L19` | `matched` | `route_answer` | `all_pre_answer` | 1 | 3.172 | -1.188 | 1.984 | 0.066 | 0.668 | 2.422 | 0.625 |
| deepseek7b | `attn:L19` | `top` | `kv_o_route` | `all_pre_answer` | 1 | 2.859 | -1.147 | 1.712 | 0.130 | 0.612 | 2.609 | 0.585 |
