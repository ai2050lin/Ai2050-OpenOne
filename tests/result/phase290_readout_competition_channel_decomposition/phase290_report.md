# Phase290 Readout Competition Channel Decomposition

- source_signature_rows: 972
- readout_channel_rows: 972
- channel_family_model_matrix_rows: 86
- stop_continue_bottleneck_rows: 340
- readout_competition_audit_queue_rows: 144
- global_continue_winner_rate: 1.0
- global_mean_top_continue_vs_stop_margin: 8.155253
- continue_channel_family_counts: {"natural_language_continue": 502, "list_structure_continue": 334, "protocol_json_continue": 70, "protocol_format_continue": 66}
- readout_bottleneck_counts: {"continue_not_suppressed": 972, "stop_not_winner": 972, "protocol_or_structure_continue": 470, "gap_need_readout_competition": 376, "target_readout_weak": 186, "closure_continue_not_suppressed": 36, "closure_stop_not_winner": 36, "closure_rollout_not_stable": 35, "closure_semantic_not_done": 3}

This phase decomposes the stop/continue readout bottleneck into channel families and produces a case-level audit queue.
It does not claim closure and does not run new model interventions.
