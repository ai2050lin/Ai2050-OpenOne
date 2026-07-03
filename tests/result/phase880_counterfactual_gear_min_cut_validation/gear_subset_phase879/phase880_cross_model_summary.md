# Phase 880 Counterfactual Gear-Set Minimal Cut Validation (gear_subset_phase879)

- Boundary: gear-subset counterfactual validation; not token-level blocker minimal cut.
- Full next-token logits are recomputed for tested conditions.

## Models

| model | status | candidates | rows | full closed | gear-min candidates | subset closed |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | no_phase879_candidates | 0 | 0 | 0 | 0 | 0 |
| glm4 | no_phase879_candidates | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | complete | 12 | 12 | 12 | 1 | 11 |

## Overall

- Overall summary: `{'n': 12, 'transition_class_counts': {'clean_causal_transition': 4, 'nonclean_output_transition': 8}, 'route_counts': {'clean_causal_transition': 4, 'format_recovery': 3, 'semantic_pressure_transition': 3, 'protocol_pressure_transition': 2}, 'displacement_subtype_counts': {'top_membership_and_role_displacement': 5, 'rank_threshold_reclassification': 7}, 'minimality_class_counts': {'proper_subset_also_boundary_closed': 11, 'gear_set_boundary_minimal_candidate': 1}, 'full_boundary_closed': 12, 'full_answer_like': 12, 'full_output_transition': 12, 'gear_set_boundary_minimal_candidate': 1, 'gear_set_answer_minimal_candidate': 1, 'proper_subset_boundary_closed_total': 11, 'proper_subset_answer_like_total': 11, 'mean_full_original_blocker_delta': 0.02170138888888889}`

## DS7B Rows

| object | prompt | candidate | route | subtype | full closed | subset closed | minimality | base -> full |
|---|---|---|---|---|---:|---:|---|---|
| seal | format_pressure | `L27C16651+L24C3875:scale_up` | clean_causal_transition | top_membership_and_role_displacement | 1 | 1 | proper_subset_also_boundary_closed | `other -> strict_canonical` |
| bat | nonclean_direct | `L27C16651+L24C3875:flip` | clean_causal_transition | rank_threshold_reclassification | 1 | 1 | proper_subset_also_boundary_closed | `other -> strict_canonical` |
| sheep | echo_pressure | `L27C16651+L24C3875:flip` | format_recovery | top_membership_and_role_displacement | 1 | 1 | proper_subset_also_boundary_closed | `format_or_empty -> answer_alias` |
| sheep | echo_pressure | `L27C16651+L24C3875:half` | format_recovery | rank_threshold_reclassification | 1 | 1 | proper_subset_also_boundary_closed | `format_or_empty -> answer_alias` |
| sheep | echo_pressure | `L27C16651+L24C3875:zero` | format_recovery | rank_threshold_reclassification | 1 | 1 | proper_subset_also_boundary_closed | `format_or_empty -> answer_alias` |
| wolf | echo_pressure | `L27C16651+L24C3875:flip` | semantic_pressure_transition | top_membership_and_role_displacement | 1 | 1 | proper_subset_also_boundary_closed | `other -> answer_alias` |
| wolf | echo_pressure | `L27C16651+L24C3875:half` | semantic_pressure_transition | rank_threshold_reclassification | 1 | 1 | proper_subset_also_boundary_closed | `other -> answer_alias` |
| wolf | echo_pressure | `L27C16651+L24C3875:zero` | semantic_pressure_transition | rank_threshold_reclassification | 1 | 1 | proper_subset_also_boundary_closed | `other -> answer_alias` |
| wolf | format_pressure | `L27C16651+L24C3875:flip` | protocol_pressure_transition | top_membership_and_role_displacement | 1 | 1 | proper_subset_also_boundary_closed | `other -> strict_canonical` |
| wolf | format_pressure | `L27C16651+L24C3875:zero` | protocol_pressure_transition | top_membership_and_role_displacement | 1 | 1 | proper_subset_also_boundary_closed | `other -> strict_canonical` |
| navy | nonclean_direct | `L27C15369+L26C8587:flip` | clean_causal_transition | rank_threshold_reclassification | 1 | 1 | proper_subset_also_boundary_closed | `other -> strict_canonical` |
| navy | nonclean_direct | `L27C15369+L26C8587:zero` | clean_causal_transition | rank_threshold_reclassification | 1 | 0 | gear_set_boundary_minimal_candidate | `other -> strict_canonical` |
