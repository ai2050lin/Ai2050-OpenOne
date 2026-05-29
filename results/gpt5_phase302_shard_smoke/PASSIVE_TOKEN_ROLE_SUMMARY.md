# Phase 302 Passive Token Role Closure Summary

## qwen3
- complete: True
- bases/train/test: 4 / 2 / 1
- rows: 144
- nonfinite_rows: 0
- probe_best:
  - agent_to_patient: L0 resid_out by_agent acc=0.500000 margin=0.122736
  - by_phrase: L0 resid_out last acc=1.000000 margin=11.326123
  - voice: L0 resid_out verb acc=1.000000 margin=3.510676
- best_by_variable:
  - by_phrase: all_positions/last+subject+verb L0 resid_out progress=0.646771 kl=0.420359 delta=0.954916
  - role_swap: subject_only/subject L0 mlp_out progress=0.053008 kl=0.937296 delta=0.072409
  - voice: all_positions/object+subject+verb L0 resid_out progress=0.451506 kl=0.599473 delta=0.618040

## glm4
- missing

## deepseek7b
- missing
