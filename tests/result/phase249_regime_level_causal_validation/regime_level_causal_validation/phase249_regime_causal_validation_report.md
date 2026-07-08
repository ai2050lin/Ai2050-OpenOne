# Phase249 regime-level causal validation

Phase249 tests whether token-bank regime directions have causal readout or early-rollout effects.
It compares regime-level suppression with top-token suppression on the same Phase248 candidates.

## Counts

- candidates: 15
- validation_rows: 75
- regime_suppression_rows: 15
- target_vs_regime_comparison_rows: 15

## Mean effects

- mean_regime_suppression_margin_delta: 1.464583
- mean_regime_injection_margin_delta: -1.470833
- mean_top_token_replay_margin_delta: -2.23125
- regime_better_than_top_token_replay_count: 7

## Route counts

```json
{
  "continuation_regime_test": 9,
  "protocol_regime_test": 5,
  "reason_regime_test": 1
}
```

## Progress

```json
{
  "pattern_family_atlas": 0.77,
  "candidate_clustering": 0.42,
  "case_bank_calibration": 0.39,
  "high_value_trace_selection": 0.62,
  "first_internal_trace_batch": 0.38,
  "trace_signature_validation": 0.36,
  "focused_causal_validation": 0.23,
  "raw_delta_vector_archive": 0.25,
  "raw_vector_factor_decomposition": 0.22,
  "regime_field_direction_bank": 0.24,
  "regime_level_causal_validation": 0.18,
  "gate_up_product_signature": 0.45,
  "residual_state_signature": 0.43,
  "readout_competition_trace": 0.67,
  "stepwise_rollout_trace": 0.24,
  "causal_closure": 0.12,
  "general_language_mechanism_confidence": 0.58
}
```
