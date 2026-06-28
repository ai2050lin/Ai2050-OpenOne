# Phase 726 Category Channel Natural Generation Closure

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: greedy natural generation under single-channel ablation.

| model | channel | n | changed_rate | baseline_hit | ablated_hit | hit_drop |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | L24H29:119 | 22 | 0.000 | 0.955 | 0.955 | 0.000 |
| glm4 | L24H19:69 | 22 | 0.000 | 0.955 | 0.955 | 0.000 |
| deepseek7b | L20H17:25 | 22 | 0.045 | 0.500 | 0.545 | 0.000 |

## Strict Interpretation

- This tests natural greedy output, not stochastic decoding.
- A low hit-drop rate means the single channel affects likelihood more than final greedy category choice.
- Strong generation closure would require output category changes under ablation.
