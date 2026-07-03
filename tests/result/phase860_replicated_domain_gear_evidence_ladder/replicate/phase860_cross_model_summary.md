# Phase 860 Replicated Domain Gear Evidence Ladder (replicate)

- Source: Phase 859 clear-gain domain gears.
- Boundary: replicated domain evidence ladder, not cross-domain language closure.

## Cross-Model Summary

| model | rows | max level | L4+ domains | L5+ domains | L6 domains | sign ambiguous domains |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 120 | 6 | 1 | 1 | 1 | 2 |
| glm4 | 60 | 4 | 1 | 0 | 0 | 1 |
| deepseek7b | 120 | 6 | 2 | 2 | 2 | 2 |

## Evidence Ladder

| model | domain | level | label | best gears | best clear gain/loss | split hits | prompt hits | alternate clear | control clear | reasons |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `geometry` | 3 | `domain_local_holdout_source` | `L29C1532` | 3/1 | 2 | 2 | 5 | 0 | `phase858_candidate,phase859_holdout_source,alternate_mode_has_clear_gain_sign_ambiguous` |
| qwen3 | `material` | 6 | `strong_domain_invariant_candidate` | `L31C4800+L31C2257` | 5/0 | 3 | 3 | 2 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,same_layer_control_clear_zero,multi_split_and_multi_prompt_support,broad_prompt_replication,alternate_mode_has_clear_gain_sign_ambiguous` |
| glm4 | `color` | 4 | `replicated_domain_edge` | `L30C7088+L30C11128` | 1/0 | 1 | 1 | 1 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,alternate_mode_has_clear_gain_sign_ambiguous` |
| deepseek7b | `animal` | 6 | `strong_domain_invariant_candidate` | `L27C16651+L24C3875` | 10/0 | 3 | 3 | 6 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,same_layer_control_clear_zero,multi_split_and_multi_prompt_support,broad_prompt_replication,alternate_mode_has_clear_gain_sign_ambiguous` |
| deepseek7b | `color` | 6 | `strong_domain_invariant_candidate` | `L27C15369+L26C8587` | 5/0 | 2 | 3 | 5 | 0 | `phase858_candidate,phase859_holdout_source,phase860_clear_gain_no_loss,same_layer_control_clear_zero,multi_split_and_multi_prompt_support,broad_prompt_replication,alternate_mode_has_clear_gain_sign_ambiguous` |
