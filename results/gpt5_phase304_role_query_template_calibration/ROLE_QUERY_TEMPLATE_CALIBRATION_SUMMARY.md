# Phase304 Role Query Template Calibration Summary

## deepseek7b

rows=2048 bases=32 reliable_templates=0 nonfinite=0

### Reliable Template Candidates

| query | template | pass | min_state_acc | min_state_margin | min_order_acc |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_performed_action | active_ba | 0.1250 | -2.3574 | 32 |
| agent | agent_acted | active_ba | 0.1250 | -2.2285 | 32 |
| agent | agent_responsible | active_ba | 0.1562 | -2.2910 | 32 |
| agent | agent_doer | active_ba | 0.1562 | -1.9297 | 32 |
| agent | agent_did_action | active_ba | 0.1875 | -2.1973 | 32 |
| agent | agent_carried_out | active_ba | 0.1875 | -2.0566 | 32 |
| agent | agent_done_by | active_ba | 0.2188 | -1.6250 | 32 |
| agent | agent_actor | active_ba | 0.2500 | -1.4990 | 32 |
| patient | patient_acted_on | passive_ba_by | 0.2500 | -1.4238 | 32 |
| patient | patient_target | passive_ba_by | 0.2812 | -1.2969 | 32 |
| agent | agent_acted | passive_ba_by | 0.3125 | -0.6035 | 32 |
| patient | patient_received | passive_ab_by | 0.3438 | -0.9238 | 32 |

## glm4

rows=2048 bases=32 reliable_templates=0 nonfinite=0

### Reliable Template Candidates

| query | template | pass | min_state_acc | min_state_margin | min_order_acc |
|---|---|---:|---:|---:|---:|
| patient | patient_affected | False | 0.7812 | 1.4863 | 0.7812 |

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_did_action | passive_ba_by | 0.3125 | -0.0068 | 32 |
| agent | agent_acted | active_ba | 0.4062 | -0.3970 | 32 |
| agent | agent_responsible | active_ba | 0.5000 | -0.0957 | 32 |
| agent | agent_done_by | passive_ba_by | 0.5000 | 0.1406 | 32 |
| agent | agent_actor | passive_ba_by | 0.5000 | 0.1865 | 32 |
| agent | agent_actor | active_ba | 0.5312 | -0.0605 | 32 |
| agent | agent_performed_action | passive_ba_by | 0.5312 | 0.5107 | 32 |
| agent | agent_did_action | active_ba | 0.5625 | 0.7773 | 32 |
| agent | agent_done_by | active_ba | 0.5625 | 0.8105 | 32 |
| agent | agent_performed_action | active_ba | 0.5625 | 1.0371 | 32 |
| patient | patient_received | passive_ab_by | 0.5938 | 0.6523 | 32 |
| patient | patient_target | passive_ba_by | 0.5938 | 0.9609 | 32 |

## qwen3

rows=2048 bases=32 reliable_templates=0 nonfinite=0

### Reliable Template Candidates

| query | template | pass | min_state_acc | min_state_margin | min_order_acc |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_actor | passive_ba_by | 0.0625 | -2.6289 | 32 |
| agent | agent_did_action | passive_ba_by | 0.1250 | -2.3066 | 32 |
| agent | agent_carried_out | passive_ba_by | 0.1250 | -2.1074 | 32 |
| agent | agent_performed_action | passive_ba_by | 0.1562 | -2.5410 | 32 |
| agent | agent_actor | active_ba | 0.1562 | -2.2773 | 32 |
| agent | agent_doer | passive_ba_by | 0.1875 | -2.3125 | 32 |
| agent | agent_responsible | passive_ba_by | 0.1875 | -1.6094 | 32 |
| agent | agent_responsible | active_ba | 0.2812 | -1.7578 | 32 |
| agent | agent_acted | active_ba | 0.3125 | -1.8672 | 32 |
| agent | agent_acted | passive_ba_by | 0.3125 | -1.3594 | 32 |
| agent | agent_done_by | passive_ba_by | 0.3750 | -0.7402 | 32 |
| patient | patient_acted_on | passive_ab_by | 0.3750 | 0.0801 | 32 |

