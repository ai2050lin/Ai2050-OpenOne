# Phase252 shared subspace and coupled regime field analysis

Phase252 analyzes direction overlap, shared subspace structure, and rollout closure traces for high-confidence candidates.
It is not closure validation.

## Counts

- direction_cosine_rows: 315
- subspace_overlap_rows: 9
- rollout_closure_trace_rows: 384

## Closure Proxy Means

```json
{
  "no_intervention": -3.221354,
  "natural_raw_suppression": -1.554199,
  "tokenbank_suppression": 0.572266
}
```

## Top Direction Cosines

```json
[
  {
    "abs_cosine": 1.0,
    "cosine": -1.0,
    "cosine_id": "phase252:cosine:qwen3:natural_orth_natural_protocol_short:natural_raw_natural_continuation_explain",
    "created_at": "2026-07-08T09:23:54.105446+00:00",
    "direction_a": "natural_orth:natural_protocol_short",
    "direction_b": "natural_raw:natural_continuation_explain",
    "model": "qwen3",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": 1.0,
    "cosine_id": "phase252:cosine:qwen3:natural_orth_natural_protocol_short:natural_raw_natural_protocol_short",
    "created_at": "2026-07-08T09:23:54.105469+00:00",
    "direction_a": "natural_orth:natural_protocol_short",
    "direction_b": "natural_raw:natural_protocol_short",
    "model": "qwen3",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": -1.0,
    "cosine_id": "phase252:cosine:qwen3:natural_raw_natural_continuation_explain:natural_raw_natural_protocol_short",
    "created_at": "2026-07-08T09:23:54.106223+00:00",
    "direction_a": "natural_raw:natural_continuation_explain",
    "direction_b": "natural_raw:natural_protocol_short",
    "model": "qwen3",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": -1.0,
    "cosine_id": "phase252:cosine:glm4:natural_orth_natural_protocol_short:natural_raw_natural_continuation_explain",
    "created_at": "2026-07-08T09:24:04.954037+00:00",
    "direction_a": "natural_orth:natural_protocol_short",
    "direction_b": "natural_raw:natural_continuation_explain",
    "model": "glm4",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": 1.0,
    "cosine_id": "phase252:cosine:glm4:natural_orth_natural_protocol_short:natural_raw_natural_protocol_short",
    "created_at": "2026-07-08T09:24:04.954062+00:00",
    "direction_a": "natural_orth:natural_protocol_short",
    "direction_b": "natural_raw:natural_protocol_short",
    "model": "glm4",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": -1.0,
    "cosine_id": "phase252:cosine:glm4:natural_raw_natural_continuation_explain:natural_raw_natural_protocol_short",
    "created_at": "2026-07-08T09:24:04.954902+00:00",
    "direction_a": "natural_raw:natural_continuation_explain",
    "direction_b": "natural_raw:natural_protocol_short",
    "model": "glm4",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": -1.0,
    "cosine_id": "phase252:cosine:deepseek7b:natural_orth_natural_protocol_short:natural_raw_natural_continuation_explain",
    "created_at": "2026-07-08T09:24:15.147040+00:00",
    "direction_a": "natural_orth:natural_protocol_short",
    "direction_b": "natural_raw:natural_continuation_explain",
    "model": "deepseek7b",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  },
  {
    "abs_cosine": 1.0,
    "cosine": 1.0,
    "cosine_id": "phase252:cosine:deepseek7b:natural_orth_natural_protocol_short:natural_raw_natural_protocol_short",
    "created_at": "2026-07-08T09:24:15.147062+00:00",
    "direction_a": "natural_orth:natural_protocol_short",
    "direction_b": "natural_raw:natural_protocol_short",
    "model": "deepseek7b",
    "phase_id": "Phase252",
    "schema_version": "1.0.0"
  }
]
```

## Progress

```json
{
  "pattern_family_atlas": 0.8,
  "candidate_clustering": 0.43,
  "case_bank_calibration": 0.4,
  "high_value_trace_selection": 0.65,
  "trace_signature_validation": 0.37,
  "focused_causal_validation": 0.24,
  "raw_delta_vector_archive": 0.26,
  "raw_vector_factor_decomposition": 0.25,
  "regime_field_direction_bank": 0.34,
  "natural_regime_direction_bank": 0.29,
  "regime_level_causal_validation": 0.24,
  "orthogonalized_direction_validation": 0.17,
  "shared_subspace_analysis": 0.18,
  "coupled_regime_field_analysis": 0.16,
  "residual_state_signature": 0.47,
  "readout_competition_trace": 0.7,
  "stepwise_rollout_trace": 0.3,
  "causal_closure": 0.13,
  "general_language_mechanism_confidence": 0.61
}
```
