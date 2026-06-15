# Phase 139 Restore/Swap Calibration: deepseek7b

Generated: 2026-06-15 06:27:26
Peak layer: L27; true last layer: L28; rank: 8; train/test: 16/32

| category | transfer | remove | best restore | best sample swap |
|---|---|---|---|---|
| number | R2=+0.26, cos=+0.99 | T-1.27 R+0.00 | block_output s2.0 T+1.12 R+1.74 rec+1.88 | input_answer s1.5 T-1.71 R+0.34 swap=container SΔ+0.34 |
| container | R2=+0.41, cos=+0.98 | T-1.41 R+0.00 | input_answer s2.0 T+0.66 R+0.00 rec+1.47 | block_output s1.5 T-0.49 R+0.79 swap=plant SΔ-0.46 |
| plant | R2=+0.32, cos=+0.98 | T-1.59 R+0.00 | block_output s2.0 T-0.11 R+1.37 rec+0.93 | block_output s0.5 T-0.70 R+0.00 swap=time SΔ-1.69 |
| time | R2=+0.66, cos=+0.99 | T-1.44 R+0.17 | block_output s0.25 T-1.49 R+0.17 rec-0.04 | block_output s1.5 T+0.91 R+2.14 swap=number SΔ+1.19 |
