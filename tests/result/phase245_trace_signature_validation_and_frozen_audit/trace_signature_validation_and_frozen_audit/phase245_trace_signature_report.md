# Phase245 trace signature validation and frozen audit

## Core result

Phase245 reuses Phase244 trace rows to classify trace signatures, audit validate/frozen stability, and select causal-test candidates.
It does not run new model forwards and does not claim causal closure.

## Counts

- signature_rows: 300
- correlation_rows: 94
- validate_frozen_audit_rows: 79
- proxy_factor_projection_rows: 300
- causal_test_candidate_rows: 30

## Signature classes

```json
{
  "mixed_signature": 143,
  "high_component_low_readout": 71,
  "high_component_high_readout": 46,
  "low_component_high_readout": 36,
  "readout_boundary_weak_change": 4
}
```

## Correlation summary

- global_component_readout_corr_by_model: {"deepseek7b": -0.005691, "glm4": -0.071452, "qwen3": 0.117579}
- strongest_abs_corr: 0.999807

## Progress

```json
{
  "pattern_family_atlas": 0.73,
  "candidate_clustering": 0.42,
  "case_bank_calibration": 0.38,
  "high_value_trace_selection": 0.58,
  "first_internal_trace_batch": 0.36,
  "trace_signature_validation": 0.32,
  "gate_up_product_signature": 0.42,
  "residual_state_signature": 0.4,
  "readout_competition_trace": 0.61,
  "stepwise_rollout_trace": 0.21,
  "proxy_factor_decomposition": 0.16,
  "causal_closure": 0.1,
  "general_language_mechanism_confidence": 0.54
}
```
