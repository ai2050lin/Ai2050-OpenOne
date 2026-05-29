# Phase 302 Passive Token Role Closure Summary

## qwen3
- complete: True
- bases/train/test: 16 / 8 / 4
- rows: 5184
- nonfinite_rows: 0
- probe_best:
  - agent_to_patient: L6 resid_out by_agent acc=0.750000 margin=5.128343
  - by_phrase: L0 resid_out last acc=1.000000 margin=9.339340
  - voice: L0 resid_out verb acc=1.000000 margin=2.951893
- best_by_variable:
  - by_phrase: last_only/last L5 resid_out progress=0.188462 kl=1.030767 delta=0.541394
  - role_swap: all_positions/object+subject L3 resid_out progress=0.057689 kl=1.094042 delta=0.205663
  - voice: all_positions/object+subject+verb L2 resid_in progress=0.267556 kl=1.272648 delta=0.590250

## glm4
- complete: True
- bases/train/test: 16 / 8 / 4
- rows: 5184
- nonfinite_rows: 0
- probe_best:
  - agent_to_patient: L4 resid_out by_agent acc=0.750000 margin=0.026202
  - by_phrase: L1 resid_out last acc=1.000000 margin=0.067451
  - voice: L1 resid_out verb acc=1.000000 margin=0.034583
- best_by_variable:
  - by_phrase: all_positions/last+subject+verb L4 resid_out progress=0.009157 kl=0.990765 delta=0.043229
  - role_swap: all_positions/object+subject L4 resid_out progress=0.047778 kl=0.948062 delta=0.119027
  - voice: subject_only/subject L8 resid_out progress=0.002460 kl=1.008206 delta=0.028401

## deepseek7b
- complete: True
- bases/train/test: 16 / 8 / 4
- rows: 4608
- nonfinite_rows: 0
- probe_best:
  - agent_to_patient: L20 mlp_out by_agent acc=0.750000 margin=59.389148
  - by_phrase: L20 resid_in last acc=1.000000 margin=7887.195618
  - voice: L20 resid_in verb acc=1.000000 margin=3432.862277
- best_by_variable:
  - by_phrase: last_only/last L24 resid_out progress=0.006703 kl=0.966365 delta=0.040732
  - role_swap: all_positions/object+subject L24 resid_out progress=0.093240 kl=0.951682 delta=0.146506
  - voice: all_positions/object+subject+verb L20 resid_in progress=0.184532 kl=1.185566 delta=0.312292
