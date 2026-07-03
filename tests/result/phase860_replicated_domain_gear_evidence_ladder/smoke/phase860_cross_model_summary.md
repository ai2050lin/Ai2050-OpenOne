# Phase 860 Replicated Domain Gear Evidence Ladder (smoke)

- Source: Phase 859 clear-gain domain gears.
- Boundary: replicated domain evidence ladder, not cross-domain language closure.

## Cross-Model Summary

| model | rows | max level | L4+ domains | L5+ domains | L6 domains | sign ambiguous domains |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 24 | 5 | 2 | 1 | 0 | 1 |
| glm4 | 12 | 4 | 1 | 0 | 0 | 1 |
| deepseek7b | 24 | 5 | 2 | 1 | 0 | 2 |

## Evidence Ladder

| model | domain | level | label | best gears | best clear gain/loss | split hits | prompt hits | alternate clear | control clear | reasons |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `geometry` | 5 | `replicated_control_filtered_domain_gear` | `L29C1532` | 2/0 | 2 | 1 | 3 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,same_layer_control_clear_zero,alternate_mode_has_clear_gain_sign_ambiguous` |
| qwen3 | `material` | 4 | `replicated_domain_edge` | `L31C4800+L31C2257` | 1/0 | 1 | 1 | 0 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss` |
| glm4 | `color` | 4 | `replicated_domain_edge` | `L30C7088+L30C11128` | 1/0 | 1 | 1 | 1 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,alternate_mode_has_clear_gain_sign_ambiguous` |
| deepseek7b | `animal` | 5 | `replicated_control_filtered_domain_gear` | `L27C16651+L24C3875` | 2/0 | 1 | 1 | 2 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,same_layer_control_clear_zero,alternate_mode_has_clear_gain_sign_ambiguous` |
| deepseek7b | `color` | 4 | `replicated_domain_edge` | `L27C15369+L26C8587` | 1/0 | 1 | 1 | 1 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,alternate_mode_has_clear_gain_sign_ambiguous` |
