# Phase 139 Cross-model Restore/Swap Calibration Summary

## qwen3

Peak layer: L35; true last layer: L36; rank: 8; train/test: 8/16; restore_sites: ['input_answer', 'block_output']

| category | transfer | remove | best restore | best sample swap |
|---|---|---|---|---|
| number | R2=+0.28, cos=+0.99 | T-1.21 R+0.78 | input_answer s2.0 T+7.26 R+8.33 rec+7.02 | input_answer s1.5 T+4.14 R+6.69 swap=container SΔ+5.11 |
| container | R2=+0.30, cos=+0.99 | T-0.55 R+0.41 | input_answer s2.0 T+6.45 R+8.21 rec+12.77 | input_answer s1.5 T+4.85 R+5.91 swap=plant SΔ+2.93 |
| plant | R2=+0.60, cos=+0.99 | T+0.51 R+0.91 | input_answer s2.0 T+4.84 R+7.19 rec+8.52 | input_answer s1.5 T+3.47 R+6.91 swap=time SΔ+3.57 |
| time | R2=+0.35, cos=+0.99 | T-0.24 R+0.91 | input_answer s2.0 T+5.48 R+7.93 rec+23.40 | input_answer s1.5 T+5.91 R+8.22 swap=number SΔ+5.85 |

## glm4

Peak layer: L18; true last layer: L40; rank: 8; train/test: 8/16; restore_sites: ['input_answer', 'block_output']

| category | transfer | remove | best restore | best sample swap |
|---|---|---|---|---|
| number | R2=+0.42, cos=+0.93 | T-0.03 R+0.50 | block_output s0.25 T-0.29 R+0.20 rec-7.43 | input_answer s1.5 T-1.60 R+1.02 swap=container SΔ+0.28 |
| container | R2=+0.46, cos=+0.88 | T-0.14 R+0.34 | input_answer s2.0 T+0.32 R+1.21 rec+3.36 | input_answer s1.5 T+0.92 R+0.94 swap=plant SΔ+0.31 |
| plant | R2=+0.30, cos=+0.88 | T-0.14 R+0.32 | input_answer s2.0 T+0.40 R+1.18 rec+3.90 | block_output s0.5 T-0.05 R+0.23 swap=time SΔ-0.53 |
| time | R2=+0.53, cos=+0.93 | T+0.12 R+0.45 | block_output s0.25 T-0.19 R+0.22 rec-2.61 | block_output s0.5 T-0.54 R+0.04 swap=number SΔ-0.52 |

## deepseek7b

Peak layer: L27; true last layer: L28; rank: 8; train/test: 16/32; restore_sites: ['input_answer', 'block_output']

| category | transfer | remove | best restore | best sample swap |
|---|---|---|---|---|
| number | R2=+0.26, cos=+0.99 | T-1.27 R+0.00 | block_output s2.0 T+1.12 R+1.74 rec+1.88 | input_answer s1.5 T-1.71 R+0.34 swap=container SΔ+0.34 |
| container | R2=+0.41, cos=+0.98 | T-1.41 R+0.00 | input_answer s2.0 T+0.66 R+0.00 rec+1.47 | block_output s1.5 T-0.49 R+0.79 swap=plant SΔ-0.46 |
| plant | R2=+0.32, cos=+0.98 | T-1.59 R+0.00 | block_output s2.0 T-0.11 R+1.37 rec+0.93 | block_output s0.5 T-0.70 R+0.00 swap=time SΔ-1.69 |
| time | R2=+0.66, cos=+0.99 | T-1.44 R+0.17 | block_output s0.25 T-1.49 R+0.17 rec-0.04 | block_output s1.5 T+0.91 R+2.14 swap=number SΔ+1.19 |

