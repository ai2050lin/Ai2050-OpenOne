# Phase251 orthogonalized natural direction causal validation

Phase251 compares token-bank directions, natural contrast directions, and orthogonalized natural directions.
It is a direction-source validation stage, not closure.

## Counts

- candidates: 15
- validation_rows: 90
- high_confidence_rollout_candidates: 9

## Mean Suppression Effects

- tokenbank: 1.464583
- natural_raw: 1.220833
- natural_orth: -1.8375
- natural_orth_better_than_tokenbank_count: 2

## Best Suppression Sources

```json
{
  "natural_raw": 8,
  "tokenbank": 7
}
```

## Progress

```json
{
  "pattern_family_atlas": 0.79,
  "candidate_clustering": 0.43,
  "case_bank_calibration": 0.4,
  "high_value_trace_selection": 0.64,
  "first_internal_trace_batch": 0.38,
  "trace_signature_validation": 0.37,
  "focused_causal_validation": 0.24,
  "raw_delta_vector_archive": 0.26,
  "raw_vector_factor_decomposition": 0.24,
  "regime_field_direction_bank": 0.33,
  "natural_regime_direction_bank": 0.28,
  "regime_level_causal_validation": 0.23,
  "orthogonalized_direction_validation": 0.16,
  "residual_state_signature": 0.46,
  "readout_competition_trace": 0.69,
  "stepwise_rollout_trace": 0.26,
  "causal_closure": 0.12,
  "general_language_mechanism_confidence": 0.6
}
```
