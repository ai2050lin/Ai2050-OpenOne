# Phase304 Role Query Template Calibration Summary

## deepseek7b

rows=128 bases=2 reliable_templates=0 nonfinite=0

### Reliable Template Candidates

| query | template | pass | min_state_acc | min_state_margin | min_order_acc |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_performed_action | active_ba | 0.0000 | -3.4062 | 2 |
| agent | agent_responsible | active_ba | 0.0000 | -3.3438 | 2 |
| agent | agent_carried_out | active_ba | 0.0000 | -2.9688 | 2 |
| patient | patient_receiving | passive_ab_by | 0.0000 | -2.6875 | 2 |
| patient | patient_received | passive_ab_by | 0.0000 | -2.6562 | 2 |
| agent | agent_acted | active_ba | 0.0000 | -2.5000 | 2 |
| patient | patient_recipient | passive_ab_by | 0.0000 | -2.1250 | 2 |
| agent | agent_responsible | passive_ba_by | 0.0000 | -1.9375 | 2 |
| patient | patient_happened_to | active_ab | 0.0000 | -1.9375 | 2 |
| agent | agent_carried_out | passive_ba_by | 0.0000 | -1.8125 | 2 |
| agent | agent_actor | active_ba | 0.0000 | -1.7188 | 2 |
| agent | agent_acted | passive_ba_by | 0.0000 | -1.5938 | 2 |

## glm4

rows=128 bases=2 reliable_templates=12 nonfinite=0

### Reliable Template Candidates

| query | template | pass | min_state_acc | min_state_margin | min_order_acc |
|---|---|---:|---:|---:|---:|
| agent | agent_acted | True | 1.0000 | 4.3906 | 1.0000 |
| agent | agent_responsible | True | 1.0000 | 3.8750 | 1.0000 |
| agent | agent_performed_action | True | 1.0000 | 2.9062 | 1.0000 |
| agent | agent_doer | True | 1.0000 | 1.6250 | 1.0000 |
| agent | agent_done_by | True | 1.0000 | 1.0938 | 1.0000 |
| patient | patient_recipient | True | 1.0000 | 3.5000 | 1.0000 |
| patient | patient_target | True | 1.0000 | 3.2188 | 1.0000 |
| patient | patient_affected | True | 1.0000 | 2.7188 | 1.0000 |
| patient | patient_received | True | 1.0000 | 2.4062 | 1.0000 |
| patient | patient_acted_on | True | 1.0000 | 2.1250 | 1.0000 |
| patient | patient_receiving | True | 1.0000 | 2.0312 | 1.0000 |
| patient | patient_happened_to | True | 1.0000 | 0.3438 | 1.0000 |

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| patient | patient_action_affected | active_ba | 0.5000 | 0.0312 | 2 |
| agent | agent_actor | passive_ba_by | 0.5000 | 0.5000 | 2 |
| agent | agent_did_action | passive_ba_by | 0.5000 | 2.9062 | 2 |
| agent | agent_carried_out | passive_ba_by | 0.5000 | 3.0938 | 2 |
| patient | patient_happened_to | active_ba | 1.0000 | 0.3438 | 2 |
| agent | agent_done_by | passive_ba_by | 1.0000 | 1.0938 | 2 |
| agent | agent_done_by | passive_ab_by | 1.0000 | 1.1875 | 2 |
| patient | patient_action_affected | passive_ba_by | 1.0000 | 1.2812 | 2 |
| agent | agent_actor | passive_ab_by | 1.0000 | 1.5625 | 2 |
| agent | agent_doer | active_ba | 1.0000 | 1.6250 | 2 |
| patient | patient_happened_to | passive_ba_by | 1.0000 | 1.8125 | 2 |
| patient | patient_receiving | passive_ba_by | 1.0000 | 2.0312 | 2 |

## qwen3

rows=128 bases=2 reliable_templates=3 nonfinite=0

### Reliable Template Candidates

| query | template | pass | min_state_acc | min_state_margin | min_order_acc |
|---|---|---:|---:|---:|---:|
| agent | agent_done_by | True | 1.0000 | 1.5625 | 1.0000 |
| agent | agent_responsible | True | 1.0000 | 0.5625 | 1.0000 |
| agent | agent_acted | True | 1.0000 | 0.5000 | 1.0000 |

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_actor | passive_ba_by | 0.0000 | -0.9375 | 2 |
| patient | patient_affected | active_ba | 0.5000 | -1.8125 | 2 |
| patient | patient_action_affected | passive_ba_by | 0.5000 | -0.9375 | 2 |
| patient | patient_acted_on | passive_ba_by | 0.5000 | -0.5000 | 2 |
| patient | patient_happened_to | passive_ba_by | 0.5000 | -0.2500 | 2 |
| agent | agent_did_action | passive_ba_by | 0.5000 | 0.5000 | 2 |
| patient | patient_recipient | passive_ba_by | 0.5000 | 0.6250 | 2 |
| patient | patient_received | passive_ba_by | 0.5000 | 0.6875 | 2 |
| agent | agent_did_action | passive_ab_by | 0.5000 | 0.8125 | 2 |
| patient | patient_acted_on | passive_ab_by | 0.5000 | 0.8125 | 2 |
| agent | agent_doer | passive_ab_by | 0.5000 | 1.0000 | 2 |
| patient | patient_target | passive_ba_by | 0.5000 | 1.0625 | 2 |

