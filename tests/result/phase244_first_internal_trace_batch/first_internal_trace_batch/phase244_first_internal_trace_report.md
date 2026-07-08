# Phase244 first internal trace batch

## Core result

Phase244 converts the Phase243 high-value candidates into fixed-format internal trace rows.
This is trace evidence, not causal closure.

## Counts

- models: qwen3, glm4, deepseek7b
- component_trace_rows: 1500
- residual_trace_rows: 900
- readout_trace_rows: 300
- stepwise_rollout_rows: 420
- trace_selection_by_test: {"readout_competitor_trace": 120, "protocol_gate_product_residual_trace": 75, "stepwise_rollout_trace": 60, "rollout_closure_trace": 30, "cross_model_structure_comparison": 15}

## Aggregate signals

- mean_component_relative_delta: 0.479805
- mean_residual_relative_delta: 0.33815
- mean_readout_margin_delta_vs_full: 7.43396
- stable_winner_match_rate: 0.5533

## Pattern Atlas progress

```json
{
  "pattern_family_atlas": 0.72,
  "candidate_clustering": 0.4,
  "case_bank_calibration": 0.36,
  "high_value_trace_selection": 0.55,
  "first_internal_trace_batch": 0.3,
  "gate_up_product_signature": 0.38,
  "residual_state_signature": 0.37,
  "readout_competition_trace": 0.58,
  "stepwise_rollout_trace": 0.18,
  "causal_closure": 0.1,
  "general_language_mechanism_confidence": 0.53
}
```

## Next

Phase245 should validate whether the strongest trace signatures survive larger validation/frozen splits and then design focused causal probes.
