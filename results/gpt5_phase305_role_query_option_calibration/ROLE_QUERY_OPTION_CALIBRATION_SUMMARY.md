# Phase305 Role Query Option Calibration Summary

## deepseek7b

rows=2048 bases=32 reliable_templates=0 nonfinite=0

### Template Candidates

| query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| patient | patient_choose_receiver | passive_ba_by | 0.3906 | -0.6611 | 64 |
| patient | patient_who_received | passive_ba_by | 0.4062 | -0.8682 | 64 |
| patient | patient_who_received | active_ba | 0.4062 | -0.4102 | 64 |
| patient | patient_which_target | passive_ba_by | 0.4062 | -0.3730 | 64 |
| patient | patient_which_target | active_ba | 0.4375 | -0.3242 | 64 |
| patient | patient_which_target | passive_ab_by | 0.4531 | 0.1826 | 64 |
| patient | patient_who_affected | passive_ba_by | 0.4688 | -0.2832 | 64 |
| agent | agent_choose_doer | active_ba | 0.4688 | -0.0127 | 64 |
| patient | patient_choose_receiver | active_ba | 0.5000 | -0.1201 | 64 |
| patient | patient_who_affected | active_ba | 0.5000 | -0.0713 | 64 |
| agent | agent_which_actor | passive_ba_by | 0.5156 | 0.2637 | 64 |
| patient | patient_which_target | active_ab | 0.5312 | 0.4023 | 64 |

## glm4

rows=2048 bases=32 reliable_templates=0 nonfinite=0

### Template Candidates

| query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---:|---:|---:|---:|
| patient | patient_who_affected | False | 0.8594 | 0.8672 | 1.5508 |
| patient | patient_who_received | False | 0.7656 | 0.7422 | 1.3740 |

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| agent | agent_choose_doer | active_ba | 0.4688 | 0.0625 | 64 |
| agent | agent_choose_doer | passive_ba_by | 0.4844 | -0.1670 | 64 |
| agent | agent_who_performed | active_ba | 0.5469 | 2.0981 | 64 |
| agent | agent_who_performed | passive_ba_by | 0.5625 | 0.5771 | 64 |
| agent | agent_which_actor | passive_ba_by | 0.5625 | 0.7725 | 64 |
| agent | agent_who_did | passive_ba_by | 0.5938 | 0.5713 | 64 |
| patient | patient_choose_receiver | passive_ba_by | 0.6094 | 0.4092 | 64 |
| agent | agent_which_actor | active_ba | 0.6250 | 1.3545 | 64 |
| agent | agent_who_did | active_ba | 0.6406 | 2.1489 | 64 |
| patient | patient_choose_receiver | active_ba | 0.6562 | 0.7207 | 64 |
| patient | patient_which_target | active_ba | 0.6562 | 0.8105 | 64 |
| patient | patient_which_target | active_ab | 0.6719 | 1.0166 | 64 |

## qwen3

rows=2048 bases=32 reliable_templates=0 nonfinite=0

### Template Candidates

| query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---:|---:|---:|---:|

### Weakest State Rows

| query | template | state | acc | margin | n |
|---|---|---|---:|---:|---:|
| patient | patient_choose_receiver | passive_ba_by | 0.2188 | -0.8926 | 64 |
| agent | agent_who_did | passive_ba_by | 0.2500 | -2.2500 | 64 |
| agent | agent_who_performed | passive_ba_by | 0.2656 | -1.9297 | 64 |
| patient | patient_choose_receiver | passive_ab_by | 0.2812 | -1.2246 | 64 |
| agent | agent_which_actor | passive_ba_by | 0.3594 | -0.8818 | 64 |
| agent | agent_choose_doer | passive_ba_by | 0.3750 | -0.9648 | 64 |
| patient | patient_which_target | active_ba | 0.5469 | 0.5020 | 64 |
| patient | patient_which_target | passive_ba_by | 0.5625 | 0.4863 | 64 |
| agent | agent_which_actor | active_ba | 0.5625 | 0.9277 | 64 |
| agent | agent_choose_doer | active_ba | 0.5781 | 1.4346 | 64 |
| patient | patient_who_received | passive_ba_by | 0.5938 | 0.8457 | 64 |
| patient | patient_who_affected | active_ba | 0.6250 | 1.4766 | 64 |

