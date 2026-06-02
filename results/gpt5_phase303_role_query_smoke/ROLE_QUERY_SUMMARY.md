# Phase 303 Role Query Closure Summary

## qwen3
- complete: True
- bases/train/test: 4 / 2 / 1
- baseline_rows: 8
- intervention_rows: 120
- nonfinite_rows: 0
- shards: 1
- baseline_summary:
  - agent/active_ab: acc=1.000000 margin=6.250000 n=1
  - agent/active_ba: acc=0.000000 margin=-0.875000 n=1
  - agent/passive_ab_by: acc=1.000000 margin=5.375000 n=1
  - agent/passive_ba_by: acc=0.000000 margin=-3.187500 n=1
  - patient/active_ab: acc=1.000000 margin=4.312500 n=1
  - patient/active_ba: acc=1.000000 margin=1.125000 n=1
  - patient/passive_ab_by: acc=1.000000 margin=5.000000 n=1
  - patient/passive_ba_by: acc=0.000000 margin=-1.125000 n=1
- best_by_query:
  - agent: by_agent_only L0 resid_out progress=4.771429 patched_margin=9.343750 flip=1.000000
  - patient: object_only L0 resid_in progress=1.655172 patched_margin=6.281250 flip=1.000000

## glm4
- missing

## deepseek7b
- missing
