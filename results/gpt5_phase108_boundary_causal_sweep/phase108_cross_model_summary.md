# Phase 108 Cross-Model Boundary Causal Sweep Summary

## Setup
| model | center layer | sweep layers | categories | scales | positions |
|---|---:|---|---:|---|---|
| qwen3 | L35 | [32, 33, 34, 35] | 6 | [0.25, 0.5, 1.0, 1.5] | ['answer_last', 'object_last', 'both'] |
| glm4 | L18 | [15, 16, 17, 18] | 6 | [0.25, 0.5, 1.0, 1.5] | ['answer_last', 'object_last', 'both'] |
| deepseek7b | L27 | [24, 25, 26, 27] | 6 | [0.25, 0.5, 1.0, 1.5] | ['answer_last', 'object_last', 'both'] |

## Strongest Boundary Effects With Same-Setting Controls
| category | qwen3 | glm4 | deepseek7b | objective reading |
|---|---|---|---|---|
| number | down L35 both s1.5 T-3.06 R0.02 N0.28; up 0.02; rel animal+0.37; specific_strong_target_down | down L15 answer_last s0.25 T-0.02 R-0.01 N-0.01; up 0.17; rel material+0.48; weak_or_control_like | down L27 both s1.5 T-4.75 R-0.02 N-1.51; up 0.08; rel clothing+0.46; specific_strong_target_down | specific strong target-down exists |
| time | down L35 both s1.5 T-1.35 R0.05 N0.69; up 0.05; rel animal+0.61; specific_strong_target_down | down L16 answer_last s1.5 T-0.24 R-0.01 N-0.42; up 0.05; rel material+0.32; weak_or_control_like | down L26 answer_last s1.5 T-0.05 R0.02 N-1.32; up 0.15; rel clothing+0.43; weak_or_control_like | specific strong target-down exists |
| container | down L32 answer_last s1.5 T-0.34 R-0.03 N-0.02; up 0.05; rel clothing+2.03; weak_or_control_like | down L15 both s1.5 T-0.05 R0.01 N0.07; up 0.07; rel event+0.25; weak_or_control_like | down L27 both s1.5 T-3.21 R-0.02 N0.07; up 0.02; rel clothing+0.09; specific_strong_target_down | specific strong target-down exists |
| clothing | down L33 answer_last s1.5 T-0.45 R-0.01 N-0.15; up 0.51; rel tool+1.08; target_down_boundary_gt_random | down L17 both s1.5 T-0.13 R-0.00 N0.13; up 0.00; rel property+0.16; weak_or_control_like | down L25 object_last s0.25 T-0.17 R-0.13 N-0.00; up 1.61; rel tool+2.17; weak_or_control_like | moderate boundary target-down |
| furniture | down L35 object_last s0.25 T-0.00 R0.00 N-0.00; up 2.10; rel clothing+2.22; weak_or_control_like | down L15 answer_last s1.0 T-0.08 R0.01 N0.00; up 0.14; rel material+0.22; weak_or_control_like | down L26 object_last s1.5 T-0.06 R0.07 N0.10; up 1.02; rel tool+1.48; weak_or_control_like | weak or control-like |
| plant | down L32 answer_last s1.5 T-0.37 R-0.06 N-0.20; up 0.04; rel color+0.25; weak_or_control_like | down L16 both s1.5 T-0.06 R-0.00 N0.02; up 0.01; rel shape+0.39; weak_or_control_like | down L25 object_last s0.25 T-0.10 R-0.03 N-0.09; up 1.31; rel animal+1.51; weak_or_control_like | weak or control-like |

## Objective Facts
- Qwen3 number becomes much stronger with both-position high-scale removal: L35 both scale1.5 target_delta=-3.06.
- Qwen3 time also strengthens with both-position high-scale removal: L35 both scale1.5 target_delta=-1.35, with animal release peaking at +0.61.
- DS7B number and container are strong target-down cases under both-position scale1.5 at L27: number=-4.75, container=-3.21.
- Clothing/furniture/plant remain mixed/opposed: their strongest boundary conditions often increase target while releasing tool/clothing/animal.
- GLM4 bf16 remains weak across the sweep; effects are finite but much smaller than Qwen3/DS7B.
- Position matters: the strongest target-down cases for number/container usually require both positions, while strongest releases often use answer_last or both.
- Layer matters: Qwen3 container/plant target-down appears earlier (L32) than boundary peak L35; a boundary-norm peak is not always the best causal layer.
