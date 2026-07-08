# Phase250 natural regime direction extraction

Phase250 replaces static token-bank directions with natural contrast directions from Phase241 variants.
It is direction extraction and projection calibration, not causal closure.

## Counts

- sample_rows: 270
- direction_rows: 150
- projection_rows: 225
- prediction_rows: 30

## Top Prediction Edges

```json
[
  {
    "model": "glm4",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_concise_answer",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": -0.786991
  },
  {
    "model": "glm4",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_target_seed",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": 0.782785
  },
  {
    "model": "glm4",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_answer_boundary",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": -0.694349
  },
  {
    "model": "glm4",
    "vector_name": "delta_residual",
    "contrast_id": "natural_target_seed",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": 0.656338
  },
  {
    "model": "glm4",
    "vector_name": "delta_residual",
    "contrast_id": "natural_answer_boundary",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": -0.634599
  },
  {
    "model": "glm4",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_protocol_short",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": -0.502829
  },
  {
    "model": "glm4",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_continuation_explain",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": 0.502829
  },
  {
    "model": "qwen3",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_target_seed",
    "rows": 10,
    "corr_projection_competitor_suppression_gain": -0.292339
  },
  {
    "model": "qwen3",
    "vector_name": "delta_down_out",
    "contrast_id": "natural_answer_boundary",
    "rows": 10,
    "corr_projection_competitor_suppression_gain": -0.287995
  },
  {
    "model": "glm4",
    "vector_name": "delta_residual",
    "contrast_id": "natural_concise_answer",
    "rows": 4,
    "corr_projection_competitor_suppression_gain": -0.240656
  }
]
```

## Progress

```json
{
  "pattern_family_atlas": 0.78,
  "candidate_clustering": 0.42,
  "case_bank_calibration": 0.4,
  "high_value_trace_selection": 0.63,
  "first_internal_trace_batch": 0.38,
  "trace_signature_validation": 0.36,
  "focused_causal_validation": 0.23,
  "raw_delta_vector_archive": 0.26,
  "raw_vector_factor_decomposition": 0.23,
  "regime_field_direction_bank": 0.3,
  "natural_regime_direction_bank": 0.2,
  "regime_level_causal_validation": 0.18,
  "gate_up_product_signature": 0.45,
  "residual_state_signature": 0.45,
  "readout_competition_trace": 0.68,
  "stepwise_rollout_trace": 0.24,
  "causal_closure": 0.12,
  "general_language_mechanism_confidence": 0.59
}
```
