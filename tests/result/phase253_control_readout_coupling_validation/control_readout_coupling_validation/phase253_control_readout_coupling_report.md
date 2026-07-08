# Phase253 control-to-readout coupling map validation

Phase253 tracks layerwise control/readout projections and 32-token rollout closure proxies.
It is not ModelClose validation.

## Counts

- candidates: 8
- control_readout_projection_rows: 240
- layerwise_coupling_rows: 48
- rollout_32token_rows: 1015
- closure_validation_candidate_rows: 15

## Mean Closure Proxy

```json
{
  "no_intervention": -3.43172,
  "tokenbank_suppression": -0.605981,
  "natural_raw_suppression": -1.59436,
  "combined_suppression": 0.374603
}
```

## Progress

```json
{
  "pattern_family_atlas": 0.81,
  "candidate_clustering": 0.43,
  "case_bank_calibration": 0.4,
  "high_value_trace_selection": 0.66,
  "trace_signature_validation": 0.38,
  "focused_causal_validation": 0.25,
  "raw_vector_factor_decomposition": 0.25,
  "regime_field_direction_bank": 0.35,
  "natural_regime_direction_bank": 0.3,
  "regime_level_causal_validation": 0.25,
  "shared_subspace_analysis": 0.2,
  "coupled_regime_field_analysis": 0.22,
  "control_readout_coupling": 0.18,
  "residual_state_signature": 0.49,
  "readout_competition_trace": 0.71,
  "stepwise_rollout_trace": 0.34,
  "causal_closure": 0.14,
  "general_language_mechanism_confidence": 0.62
}
```
