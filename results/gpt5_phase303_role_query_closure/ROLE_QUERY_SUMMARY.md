# Phase 303 Role Query Closure Summary

## qwen3
- complete: True
- bases/train/test: 16 / 8 / 1
- baseline_rows: 8
- intervention_rows: 1080
- nonfinite_rows: 0
- shards: 1
- baseline_summary:
  - agent/active_ab: acc=1.000000 margin=6.187500 n=1
  - agent/active_ba: acc=1.000000 margin=3.750000 n=1
  - agent/passive_ab_by: acc=0.000000 margin=-1.125000 n=1
  - agent/passive_ba_by: acc=0.000000 margin=-0.250000 n=1
  - patient/active_ab: acc=1.000000 margin=8.187500 n=1
  - patient/active_ba: acc=1.000000 margin=5.500000 n=1
  - patient/passive_ab_by: acc=1.000000 margin=2.750000 n=1
  - patient/passive_ba_by: acc=1.000000 margin=0.875000 n=1
- best_by_query:
  - agent: subject_only L0 resid_in progress=4.104059 patched_margin=0.078125 flip=0.500000
  - patient: subject_only L2 resid_out progress=1.250118 patched_margin=0.031250 flip=0.500000

## glm4
- missing

## deepseek7b
- missing
