# Phase 879 Blocker Min-Cut Proxy Audit

- Boundary: offline proxy audit from saved Phase878 top-k blocker displacement rows; no new model run.
- Goal: separate observed blocker-boundary cuts from true causal minimal cuts.

## Summary

- Rows: `12`
- Transition classes: `{'clean_causal_transition': 4, 'nonclean_output_transition': 8}`
- Routes: `{'clean_causal_transition': 4, 'format_recovery': 3, 'semantic_pressure_transition': 3, 'protocol_pressure_transition': 2}`
- Nonclean: `{'n': 8, 'transition_class_counts': {'nonclean_output_transition': 8}, 'route_counts': {'format_recovery': 3, 'semantic_pressure_transition': 3, 'protocol_pressure_transition': 2}, 'candidate_counts': {'L27C16651+L24C3875:flip': 3, 'L27C16651+L24C3875:half': 2, 'L27C16651+L24C3875:zero': 3}, 'objects': ['sheep', 'wolf'], 'prompts': ['echo_pressure', 'format_pressure'], 'observed_proxy_closed': 8, 'observed_proxy_status_counts': {'observed_blocker_boundary_closed': 8}, 'displacement_subtype_counts': {'top_membership_and_role_displacement': 4, 'rank_threshold_reclassification': 4}, 'rank_only_cut': 4, 'membership_cut': 4, 'role_cut': 4, 'cut_role_counts': {'other_blocker': 6, 'format_punct': 3, 'protocol_word': 2}, 'mean_proxy_cut_size': 1.375, 'mean_target_rank_improvement': 1.375, 'mean_target_logit_delta_raw': 1.90625, 'mean_original_blocker_delta': 0.05859375, 'mean_top_token_overlap_ratio': 0.9090909090909092}`
- Clean: `{'n': 4, 'transition_class_counts': {'clean_causal_transition': 4}, 'route_counts': {'clean_causal_transition': 4}, 'candidate_counts': {'L27C16651+L24C3875:scale_up': 1, 'L27C16651+L24C3875:flip': 1, 'L27C15369+L26C8587:flip': 1, 'L27C15369+L26C8587:zero': 1}, 'objects': ['bat', 'navy', 'seal'], 'prompts': ['format_pressure', 'nonclean_direct'], 'observed_proxy_closed': 4, 'observed_proxy_status_counts': {'observed_blocker_boundary_closed': 4}, 'displacement_subtype_counts': {'top_membership_and_role_displacement': 1, 'rank_threshold_reclassification': 3}, 'rank_only_cut': 3, 'membership_cut': 1, 'role_cut': 1, 'cut_role_counts': {'other_blocker': 4, 'format_punct': 5}, 'mean_proxy_cut_size': 2.25, 'mean_target_rank_improvement': 2.25, 'mean_target_logit_delta_raw': 1.21875, 'mean_original_blocker_delta': -0.05208333333333333, 'mean_top_token_overlap_ratio': 0.9545454545454546}`

## By Route

| route | n | observed closed | subtype counts | rank-only | membership | role | mean cut | mean rank improve | mean target logit delta | mean original blocker | cut roles |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean_causal_transition | 4 | 4 | {"rank_threshold_reclassification": 3, "top_membership_and_role_displacement": 1} | 3 | 1 | 1 | 2.250 | 2.250 | 1.219 | -0.0521 | {"format_punct": 5, "other_blocker": 4} |
| format_recovery | 3 | 3 | {"rank_threshold_reclassification": 2, "top_membership_and_role_displacement": 1} | 2 | 1 | 1 | 2.000 | 2.000 | 1.896 | 0.0104 | {"format_punct": 3, "other_blocker": 3} |
| protocol_pressure_transition | 2 | 2 | {"top_membership_and_role_displacement": 2} | 0 | 2 | 2 | 1.000 | 1.000 | 0.625 | 0.0625 | {"protocol_word": 2} |
| semantic_pressure_transition | 3 | 3 | {"rank_threshold_reclassification": 2, "top_membership_and_role_displacement": 1} | 2 | 1 | 1 | 1.000 | 1.000 | 2.771 | 0.1042 | {"other_blocker": 3} |

## By Object Prompt

| object prompt | n | routes | candidates | observed closed | subtype counts | cut roles |
|---|---:|---|---|---:|---|---|
| bat::nonclean_direct | 1 | {"clean_causal_transition": 1} | {"L27C16651+L24C3875:flip": 1} | 1 | {"rank_threshold_reclassification": 1} | {"format_punct": 1, "other_blocker": 1} |
| navy::nonclean_direct | 2 | {"clean_causal_transition": 2} | {"L27C15369+L26C8587:flip": 1, "L27C15369+L26C8587:zero": 1} | 2 | {"rank_threshold_reclassification": 2} | {"format_punct": 4, "other_blocker": 2} |
| seal::format_pressure | 1 | {"clean_causal_transition": 1} | {"L27C16651+L24C3875:scale_up": 1} | 1 | {"top_membership_and_role_displacement": 1} | {"other_blocker": 1} |
| sheep::echo_pressure | 3 | {"format_recovery": 3} | {"L27C16651+L24C3875:flip": 1, "L27C16651+L24C3875:half": 1, "L27C16651+L24C3875:zero": 1} | 3 | {"rank_threshold_reclassification": 2, "top_membership_and_role_displacement": 1} | {"format_punct": 3, "other_blocker": 3} |
| wolf::echo_pressure | 3 | {"semantic_pressure_transition": 3} | {"L27C16651+L24C3875:flip": 1, "L27C16651+L24C3875:half": 1, "L27C16651+L24C3875:zero": 1} | 3 | {"rank_threshold_reclassification": 2, "top_membership_and_role_displacement": 1} | {"other_blocker": 3} |
| wolf::format_pressure | 2 | {"protocol_pressure_transition": 2} | {"L27C16651+L24C3875:flip": 1, "L27C16651+L24C3875:zero": 1} | 2 | {"top_membership_and_role_displacement": 2} | {"protocol_word": 2} |

## Interpretation Boundary

- `observed_blocker_boundary_closed` means saved top-k blocker rows show all observed blockers removed and target rank moved to 1.
- It is not a true causal minimal cut, because each blocker token/edge was not counterfactually ablated and logits were not recomputed.
- `rank_threshold_reclassification` means top-k membership can stay stable while blocker labels disappear because target crosses the rank boundary.
