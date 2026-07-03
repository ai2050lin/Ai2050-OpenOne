# Phase 867 Clean Route Holdout Prediction (dry)

- Source: Phase 865 full-set route purity rows.
- Fixed rule: Phase 866 CleanMixedRoute, object_delta_threshold=0.25 unless configured.
- Boundary: holdout rule validation, not language closure.

## Cross-Model Summary

| model | status | candidates | domains | source-clean -> holdout-clean stats |
|---|---|---:|---|---|
| qwen3 | dry_run | 4 | `[]` | `{}` |
| glm4 | no_phase865_candidates | 0 | `[]` | `{}` |
| deepseek7b | dry_run | 8 | `[]` | `{}` |

## Holdout Effects

| model | domain | mode | source purity | source clean | holdout clean | clear gain/loss | ans delta | blocker red. | orig blocker delta | object delta | side effects |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
