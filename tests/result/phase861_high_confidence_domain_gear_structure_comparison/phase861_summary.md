# Phase 861 High-Confidence Domain Gear Structure Comparison

- Source: Phase 860 replicated evidence ladder.
- Boundary: offline structure comparison, not a new model intervention and not closure.

## Summary

- n_signatures: `3`
- models: `['deepseek7b', 'qwen3']`
- domains: `['animal', 'color', 'material']`
- depth_band_counts: `{'late': 3}`
- gear_count_counts: `{'2': 3}`
- candidate_role_counts: `{'negative_blocker': 3}`
- best_mode_counts: `{'flip': 3}`
- sign_ambiguous_count: `3`
- control_zero_count: `3`
- new_replication_supported_count: `2`
- avg_mean_norm_layer: `0.9372134038800706`
- avg_best_clear_gain: `6.666666666666667`
- avg_alternate_to_best_ratio: `0.6666666666666666`

## Signatures

| model | domain | level | gears | depth | role | mode | clear gain/loss | alt gain | control gain | split | prompt | new repl |
|---|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|
| deepseek7b | animal | 6 | `L27C16651+L24C3875` | late:0.944 | negative_blocker | flip | 10/0 | 6 | 0 | `[6.0, 2.0, 2.0]` | `[4.0, 2.0, 4.0]` | 2 |
| deepseek7b | color | 6 | `L27C15369+L26C8587` | late:0.981 | negative_blocker | flip | 5/0 | 5 | 0 | `[3.0, 2.0, 0.0]` | `[1.0, 2.0, 2.0]` | 0 |
| qwen3 | material | 6 | `L31C4800+L31C2257` | late:0.886 | negative_blocker | flip | 5/0 | 2 | 0 | `[1.0, 1.0, 3.0]` | `[2.0, 1.0, 2.0]` | 3 |

## Pairwise Fingerprint Comparison

| left | right | same depth | norm dist | same role | same mode | split cos | prompt cos | both control zero | both sign ambiguous |
|---|---|---|---:|---|---|---:|---:|---|---|
| deepseek7b:animal | deepseek7b:color | True | 0.037 | True | True | 0.920 | 0.889 | True | True |
| deepseek7b:animal | qwen3:material | True | 0.059 | True | True | 0.636 | 1.000 | True | True |
| deepseek7b:color | qwen3:material | True | 0.096 | True | True | 0.418 | 0.889 | True | True |

## Conservative Reading

- The strongest common fingerprint is late-layer, two-channel, negative-blocker, flip-mode gear sets with zero same-layer control gain.
- The same fingerprint appears in qwen3 material and DS7B animal/color, but this is structural similarity, not channel or semantic universality.
- Alternate zero mode remains effective, so sign calibration is still open.
- DS7B color lacks new-replication split gain in this round, so it is weaker than DS7B animal and qwen3 material on fresh-case replication.

