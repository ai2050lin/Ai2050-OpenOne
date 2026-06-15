# Phase 139 Restore/Swap Calibration: qwen3

Generated: 2026-06-15 06:24:58
Peak layer: L35; true last layer: L36; rank: 4; train/test: 2/2

| category | transfer | remove | best restore | best sample swap |
|---|---|---|---|---|
| number | R2=+0.05, cos=+0.99 | T+0.30 R+1.01 | block_output s0.5 T-0.47 R+0.34 rec-2.56 | block_output s1.0 T-0.70 R+0.74 swap=container SΔ+0.29 |
| container | R2=+0.09, cos=+0.99 | T-0.66 R+0.90 | block_output s1.0 T-0.28 R+0.32 rec+0.57 | block_output s1.0 T-1.25 R+0.00 swap=number SΔ-1.95 |
