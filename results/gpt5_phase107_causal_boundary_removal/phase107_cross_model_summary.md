# Phase 107 Cross-Model Causal Boundary Removal Summary

## Global Setup
| model | boundary layer | train/category | test/category | prompts/category |
|---|---:|---:|---:|---:|
| qwen3 | L35 | 12 | 12 | 48 |
| glm4 | L18 | 12 | 12 | 48 |
| deepseek7b | L27 | 12 | 12 | 48 |

## Category Effects
| category | qwen3 | glm4 | deepseek7b | objective reading |
|---|---|---|---|---|
| fruit | TΔ=0.16 ctl=0.02; rel=sound+0.69; competitor_release_only | TΔ=0.08 ctl=0.01; rel=shape+0.39; weak_or_control_like | TΔ=0.94 ctl=-0.00; rel=time+1.48; competitor_release_only | release without target decrease |
| vehicle | TΔ=0.11 ctl=-0.00; rel=role+0.22; weak_or_control_like | TΔ=-0.01 ctl=-0.01; rel=place+0.53; competitor_release_only | TΔ=-0.04 ctl=0.13; rel=machine+0.48; weak_or_control_like | release without target decrease |
| clothing | TΔ=0.49 ctl=0.03; rel=tool+0.89; competitor_release_only | TΔ=-0.15 ctl=0.00; rel=property+0.29; weak_or_control_like | TΔ=1.05 ctl=0.35; rel=tool+1.58; competitor_release_only | release without target decrease |
| furniture | TΔ=1.37 ctl=0.04; rel=building+1.49; competitor_release_only | TΔ=0.01 ctl=0.00; rel=material+0.07; weak_or_control_like | TΔ=0.62 ctl=0.05; rel=tool+1.02; competitor_release_only | release without target decrease |
| plant | TΔ=0.04 ctl=-0.06; rel=color+0.15; weak_or_control_like | TΔ=-0.00 ctl=0.01; rel=material+0.27; weak_or_control_like | TΔ=1.04 ctl=-0.05; rel=animal+1.19; competitor_release_only | release without target decrease |
| body | TΔ=0.17 ctl=-0.05; rel=weather+0.73; competitor_release_only | TΔ=0.05 ctl=-0.01; rel=place+0.31; weak_or_control_like | TΔ=0.65 ctl=0.00; rel=container+1.00; competitor_release_only | release without target decrease |
| place | TΔ=0.05 ctl=-0.00; rel=shape+0.24; weak_or_control_like | TΔ=0.03 ctl=-0.00; rel=action+0.14; weak_or_control_like | TΔ=0.21 ctl=-0.01; rel=emotion+0.23; weak_or_control_like | weak or control-like |
| building | TΔ=0.36 ctl=-0.00; rel=shape+0.59; competitor_release_only | TΔ=-0.01 ctl=0.00; rel=action+0.08; weak_or_control_like | TΔ=0.22 ctl=0.04; rel=fruit+0.55; weak_or_control_like | release without target decrease |
| time | TΔ=-0.51 ctl=0.01; rel=animal+0.60; target_down_competitor_release | TΔ=-0.03 ctl=-0.00; rel=material+0.16; weak_or_control_like | TΔ=0.10 ctl=0.13; rel=clothing+0.23; weak_or_control_like | causal-like target decrease with release in at least one model |
| number | TΔ=-1.41 ctl=0.03; rel=animal+0.23; target_down_only | TΔ=0.05 ctl=0.01; rel=container+0.18; weak_or_control_like | TΔ=-2.58 ctl=-0.07; rel=none+0.00; target_down_only | target decrease without clean release |
| weather | TΔ=0.10 ctl=-0.00; rel=light+0.58; competitor_release_only | TΔ=-0.26 ctl=0.00; rel=shape+0.30; weak_or_control_like | TΔ=0.01 ctl=-0.13; rel=clothing+0.40; weak_or_control_like | release without target decrease |
| container | TΔ=0.03 ctl=-0.11; rel=fruit+1.12; competitor_release_only | TΔ=-0.01 ctl=0.01; rel=role+0.23; weak_or_control_like | TΔ=-2.28 ctl=0.03; rel=none+0.00; target_down_only | target decrease without clean release |

## Objective Facts
- Qwen3 L35 boundary removal produced clear target decreases for time (-0.51) and number (-1.41), while most concrete categories showed target increases or release-only behavior.
- Qwen3 clothing/furniture/body/container showed competitor releases beyond random control, but target DCF increased rather than decreased; these are not clean 'remove category boundary suppresses target' effects.
- GLM4 had to be rerun with PROBE_TORCH_DTYPE=bfloat16; fp16 logits produced NaN. In bf16, effects were small but finite.
- DS7B L27 boundary removal produced strong target decreases for number (-2.58) and container (-2.28), while several concrete categories showed target-up/opposed effects.
- Random same-norm controls were usually much smaller than boundary removal for release magnitude, so many release effects are direction-specific even when target decrease is absent.
- This phase confirms that atlas boundary vectors can affect final logits in real forward passes, but the sign is category/model-specific; a boundary vector is not always a simple positive support direction for its category.
