# Phase 129 Cross-model Position-corrected Gateway Audit Summary

## qwen3

Peak layer: L35; true last layer: L36

| category | audit | peak input | peak output | last input | last output | final norm input | final norm output |
|---|---|---|---|---|---|---|---|
| number | answer_in_pre=0, old_mismatch=0, mean_pre=3.2 | T-0.05 R+0.27 A-10.85 | T-0.07 R+0.15 A+0.00 | T-0.07 R+0.15 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |
| container | answer_in_pre=0, old_mismatch=0, mean_pre=3.2 | T-0.04 R+0.41 A-9.50 | T+0.07 R+0.22 A+0.00 | T+0.07 R+0.22 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |
| plant | answer_in_pre=0, old_mismatch=0, mean_pre=3.2 | T+0.27 R+0.45 A-10.46 | T+0.24 R+0.26 A+0.00 | T+0.24 R+0.26 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |

## glm4

Peak layer: L18; true last layer: L40

| category | audit | peak input | peak output | last input | last output | final norm input | final norm output |
|---|---|---|---|---|---|---|---|
| number | answer_in_pre=0, old_mismatch=32, mean_pre=3.2 | T-0.47 R+0.00 A-0.05 | T-0.61 R+0.00 A+0.00 | T-0.05 R+0.05 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |
| container | answer_in_pre=0, old_mismatch=62, mean_pre=3.2 | T-0.26 R+0.00 A-0.05 | T-0.32 R+0.00 A+0.00 | T+0.02 R+0.08 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |
| plant | answer_in_pre=0, old_mismatch=52, mean_pre=3.2 | T-0.17 R+0.00 A-0.07 | T-0.25 R+0.00 A+0.00 | T-0.15 R+0.08 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |

## deepseek7b

Peak layer: L27; true last layer: L28

| category | audit | peak input | peak output | last input | last output | final norm input | final norm output |
|---|---|---|---|---|---|---|---|
| number | answer_in_pre=0, old_mismatch=0, mean_pre=3.2 | T-0.96 R+0.00 A-8.96 | T-2.51 R+0.54 A+0.00 | T-2.51 R+0.54 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |
| container | answer_in_pre=0, old_mismatch=0, mean_pre=3.2 | T-0.98 R+0.00 A-7.96 | T-2.66 R+0.88 A+0.00 | T-2.66 R+0.88 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |
| plant | answer_in_pre=0, old_mismatch=0, mean_pre=3.2 | T-1.21 R+0.00 A-3.10 | T-2.42 R+1.56 A+0.00 | T-2.42 R+1.56 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 | T+0.00 R+0.00 A+0.00 |

