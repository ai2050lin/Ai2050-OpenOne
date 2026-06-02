# Phase305 Role Query Option Calibration Summary

## deepseek7b

rows=128 bases=2 reliable_templates=0 nonfinite=0

### Template Candidates

| query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| patient | patient_choose_receiver | passive_ba_by | 0.5000 | -2.6875 | 4 |
| patient | patient_which_target | passive_ba_by | 0.5000 | -1.4219 | 4 |
| patient | patient_which_target | active_ba | 0.5000 | -1.2656 | 4 |
| patient | patient_who_received | passive_ba_by | 0.5000 | -1.0469 | 4 |
| patient | patient_who_received | active_ba | 0.5000 | -0.9375 | 4 |
| patient | patient_choose_receiver | active_ba | 0.5000 | -0.8281 | 4 |
| agent | agent_which_actor | passive_ab_by | 0.5000 | -0.8125 | 4 |
| agent | agent_choose_doer | passive_ab_by | 0.5000 | -0.6875 | 4 |
| patient | patient_who_affected | active_ba | 0.5000 | -0.3594 | 4 |
| patient | patient_who_affected | passive_ba_by | 0.5000 | -0.3281 | 4 |
| agent | agent_choose_doer | active_ab | 0.5000 | -0.0469 | 4 |
| agent | agent_who_performed | passive_ab_by | 0.5000 | 0.1094 | 4 |

## glm4

rows=128 bases=2 reliable_templates=1 nonfinite=0

### Template Candidates

| query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---:|---:|---:|---:|
| patient | patient_who_received | True | 1.0000 | 1.0000 | 1.3594 |
| patient | patient_which_target | False | 0.7500 | 0.7500 | 2.0781 |

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| patient | patient_choose_receiver | passive_ba_by | 0.2500 | -0.2500 | 4 |
| patient | patient_who_affected | active_ba | 0.5000 | 1.0469 | 4 |
| agent | agent_which_actor | active_ba | 0.5000 | 1.2500 | 4 |
| agent | agent_choose_doer | active_ba | 0.5000 | 1.2969 | 4 |
| agent | agent_which_actor | passive_ba_by | 0.5000 | 2.1406 | 4 |
| agent | agent_who_did | active_ba | 0.5000 | 4.8984 | 4 |
| agent | agent_who_performed | active_ba | 0.5000 | 5.6406 | 4 |
| patient | patient_choose_receiver | active_ba | 0.7500 | 0.8281 | 4 |
| agent | agent_choose_doer | passive_ab_by | 0.7500 | 0.9688 | 4 |
| agent | agent_choose_doer | active_ab | 0.7500 | 2.0000 | 4 |
| patient | patient_choose_receiver | passive_ab_by | 0.7500 | 2.0312 | 4 |
| patient | patient_which_target | active_ba | 0.7500 | 2.0781 | 4 |

## qwen3

rows=128 bases=2 reliable_templates=0 nonfinite=0

### Template Candidates

| query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_who_did | passive_ba_by | 0.0000 | -6.1562 | 4 |
| agent | agent_who_performed | passive_ba_by | 0.0000 | -5.1562 | 4 |
| agent | agent_choose_doer | passive_ba_by | 0.0000 | -2.8750 | 4 |
| agent | agent_which_actor | passive_ba_by | 0.0000 | -2.6250 | 4 |
| patient | patient_who_received | passive_ba_by | 0.2500 | -2.8125 | 4 |
| patient | patient_who_affected | passive_ba_by | 0.2500 | -1.2500 | 4 |
| patient | patient_choose_receiver | passive_ba_by | 0.2500 | -0.2812 | 4 |
| patient | patient_which_target | passive_ba_by | 0.5000 | -0.8125 | 4 |
| patient | patient_which_target | active_ba | 0.5000 | -0.3438 | 4 |
| agent | agent_which_actor | active_ba | 0.5000 | 1.3438 | 4 |
| agent | agent_choose_doer | active_ba | 0.5000 | 2.6875 | 4 |
| patient | patient_who_affected | active_ba | 0.5000 | 3.2812 | 4 |

