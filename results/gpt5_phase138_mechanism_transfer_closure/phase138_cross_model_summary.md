# Phase 138 Cross-model Mechanism Transfer Closure Summary

## qwen3

Peak layer: L35; true last layer: L36; rank: 8; train/test: 8/16

| category | transfer | remove | restore | recovery | swap |
|---|---|---|---|---|---|
| number | R2=+0.28, cos=+0.99 | T-1.21 R+0.78 A-589.79 | T-0.41 R+1.07 A-574.16 | +0.66 | T-1.03 R+0.54 A-583.77 swap=container SΔ+0.45 |
| container | R2=+0.30, cos=+0.99 | T-0.55 R+0.41 A-594.35 | T+0.90 R+0.97 A-582.49 | +2.64 | T+0.90 R+1.08 A-586.28 swap=plant SΔ-0.30 |
| plant | R2=+0.60, cos=+0.99 | T+0.51 R+0.91 A-570.89 | T-0.07 R+1.04 A-560.34 | -1.13 | T-0.43 R+1.64 A-563.37 swap=time SΔ-0.43 |
| time | R2=+0.35, cos=+0.99 | T-0.24 R+0.91 A-584.77 | T-0.40 R+1.72 A-572.52 | -0.62 | T-0.86 R+0.96 A-570.92 swap=clothing SΔ-0.55 |
| clothing | R2=+0.69, cos=+0.99 | T+0.65 R+0.71 A-583.71 | T-0.38 R+1.01 A-572.26 | -1.58 | T-0.48 R+1.44 A-579.83 swap=furniture SΔ-0.54 |
| furniture | R2=+0.61, cos=+0.99 | T-0.11 R+0.78 A-583.39 | T-0.87 R+1.32 A-573.72 | -6.84 | T+1.39 R+1.56 A-566.66 swap=number SΔ-0.11 |

## glm4

Peak layer: L18; true last layer: L40; rank: 8; train/test: 8/16

| category | transfer | remove | restore | recovery | swap |
|---|---|---|---|---|---|
| number | R2=+0.42, cos=+0.93 | T-0.03 R+0.50 A-38.58 | T-0.91 R+0.18 A+13.24 | -25.76 | T-1.11 R+0.25 A+0.18 swap=container SΔ-0.49 |
| container | R2=+0.46, cos=+0.88 | T-0.14 R+0.34 A-31.01 | T-0.30 R+0.17 A+14.24 | -1.25 | T+0.03 R+0.11 A-12.56 swap=plant SΔ+0.11 |
| plant | R2=+0.30, cos=+0.88 | T-0.14 R+0.32 A-33.10 | T-0.04 R+0.20 A+4.05 | +0.70 | T+0.10 R+0.33 A-1.21 swap=time SΔ-1.36 |
| time | R2=+0.53, cos=+0.93 | T+0.12 R+0.45 A-32.44 | T-1.02 R+0.28 A+15.93 | -9.43 | T-0.42 R+0.22 A+0.41 swap=clothing SΔ+0.07 |
| clothing | R2=+0.40, cos=+0.88 | T+0.08 R+0.22 A-28.95 | T-0.13 R+0.07 A+19.56 | -2.69 | T-0.77 R+0.00 A-11.40 swap=furniture SΔ-0.61 |
| furniture | R2=+0.28, cos=+0.84 | T+0.16 R+0.39 A-22.69 | T-0.93 R+0.01 A+23.95 | -6.89 | T-0.21 R+0.05 A+11.10 swap=number SΔ-0.99 |

## deepseek7b

Peak layer: L27; true last layer: L28; rank: 8; train/test: 16/32

| category | transfer | remove | restore | recovery | swap |
|---|---|---|---|---|---|
| number | R2=+0.26, cos=+0.99 | T-1.27 R+0.00 A-1087.12 | T+0.52 R+1.33 A-1042.55 | +1.41 | T-1.12 R+0.00 A-1047.88 swap=container SΔ-0.56 |
| container | R2=+0.41, cos=+0.98 | T-1.41 R+0.00 A-970.25 | T-0.52 R+0.00 A-919.17 | +0.63 | T-0.86 R+0.64 A-947.96 swap=plant SΔ-0.74 |
| plant | R2=+0.32, cos=+0.98 | T-1.59 R+0.00 A-910.91 | T-0.42 R+1.02 A-865.65 | +0.74 | T-0.37 R+0.00 A-861.08 swap=time SΔ-1.79 |
| time | R2=+0.66, cos=+0.99 | T-1.44 R+0.17 A-966.63 | T-1.71 R+0.00 A-922.32 | -0.19 | T-0.52 R+0.71 A-891.18 swap=clothing SΔ+0.71 |
| clothing | R2=+0.49, cos=+0.98 | T+0.42 R+0.43 A-937.90 | T+0.88 R+0.68 A-888.10 | +1.07 | T+0.75 R+0.64 A-894.33 swap=furniture SΔ+0.64 |
| furniture | R2=+0.41, cos=+0.98 | T-0.24 R+0.00 A-1005.00 | T+0.09 R+0.19 A-960.28 | +1.36 | T+1.64 R+1.51 A-959.56 swap=number SΔ+0.86 |

