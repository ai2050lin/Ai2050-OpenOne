# Phase246 focused causal validation

Phase246 performs small-scale causal-signal tests over Phase245 candidates.
It is not closure validation.

## Counts

- candidates: 15
- validation_rows: 60
- raw_delta_vectors: 15
- necessity_signal_count: 3
- target_injection_gain_count: 14
- competitor_suppression_gain_count: 10

## Mean effects

- mean_ablation_margin_delta: 1.768229
- mean_target_injection_margin_delta: 8.764063
- mean_competitor_suppression_margin_delta: -0.696354

## Progress

```json
{
  "pattern_family_atlas": 0.74,
  "candidate_clustering": 0.42,
  "case_bank_calibration": 0.39,
  "high_value_trace_selection": 0.6,
  "first_internal_trace_batch": 0.38,
  "trace_signature_validation": 0.35,
  "focused_causal_validation": 0.2,
  "raw_delta_vector_archive": 0.18,
  "gate_up_product_signature": 0.44,
  "residual_state_signature": 0.41,
  "readout_competition_trace": 0.63,
  "stepwise_rollout_trace": 0.23,
  "proxy_factor_decomposition": 0.18,
  "causal_closure": 0.12,
  "general_language_mechanism_confidence": 0.55
}
```
