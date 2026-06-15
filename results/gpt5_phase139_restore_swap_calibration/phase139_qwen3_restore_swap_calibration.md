# Phase 139 Restore/Swap Calibration: qwen3

Generated: 2026-06-15 06:25:18
Peak layer: L35; true last layer: L36; rank: 8; train/test: 8/16

| category | transfer | remove | best restore | best sample swap |
|---|---|---|---|---|
| number | R2=+0.28, cos=+0.99 | T-1.21 R+0.78 | input_answer s2.0 T+7.26 R+8.33 rec+7.02 | input_answer s1.5 T+4.14 R+6.69 swap=container SΔ+5.11 |
| container | R2=+0.30, cos=+0.99 | T-0.55 R+0.41 | input_answer s2.0 T+6.45 R+8.21 rec+12.77 | input_answer s1.5 T+4.85 R+5.91 swap=plant SΔ+2.93 |
| plant | R2=+0.60, cos=+0.99 | T+0.51 R+0.91 | input_answer s2.0 T+4.84 R+7.19 rec+8.52 | input_answer s1.5 T+3.47 R+6.91 swap=time SΔ+3.57 |
| time | R2=+0.35, cos=+0.99 | T-0.24 R+0.91 | input_answer s2.0 T+5.48 R+7.93 rec+23.40 | input_answer s1.5 T+5.91 R+8.22 swap=number SΔ+5.85 |
