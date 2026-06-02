# Phase306 Symbolic Role Query Calibration Summary

## deepseek7b

rows=6144 bases=16 reliable_templates=0 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| entity_ab | entity | patient | json_patient | active_ab | 0.0000 | -5.1563 | 32 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -4.2578 | 32 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -4.0000 | 32 |
| nonce | letter | patient | role_table_patient | active_ab | 0.0000 | -4.0000 | 32 |
| entity_ab | letter | agent | role_table_agent | active_ba | 0.0000 | -3.9922 | 32 |
| nonce | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -3.9707 | 32 |
| nonce | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.8477 | 32 |
| ab | letter | patient | role_table_patient | active_ab | 0.0000 | -3.7383 | 32 |
| nonce | letter | agent | forced_agent | active_ba | 0.0000 | -3.6152 | 32 |
| ab | letter | patient | forced_patient | active_ab | 0.0000 | -3.4707 | 32 |
| nonce | letter | patient | forced_patient | active_ab | 0.0000 | -3.4580 | 32 |
| nonce | letter | patient | forced_patient | passive_ab_by | 0.0000 | -3.3428 | 32 |
| ab | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.3379 | 32 |
| entity_ab | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.1484 | 32 |
| ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -3.1406 | 32 |
| entity_ab | letter | agent | forced_agent | active_ba | 0.0000 | -3.1152 | 32 |

## glm4

rows=6144 bases=16 reliable_templates=3 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|
| ab | letter | patient | json_patient | False | 0.8125 | 0.9375 | 0.7812 |
| entity_ab | entity | agent | json_agent | True | 1.0000 | 1.0000 | 1.7891 |
| entity_ab | entity | agent | compact_agent | True | 0.9375 | 0.9844 | 1.3359 |
| entity_ab | entity | agent | role_table_agent | True | 0.9375 | 0.9844 | 1.2969 |
| entity_ab | entity | patient | forced_patient | False | 0.8438 | 0.8906 | 0.5313 |

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| nonce | entity | patient | json_patient | passive_ab_by | 0.0000 | -3.6602 | 32 |
| ab | entity | patient | role_table_patient | passive_ab_by | 0.0000 | -3.5801 | 32 |
| nonce | entity | agent | json_agent | passive_ba_by | 0.0000 | -3.4648 | 32 |
| nonce | entity | patient | role_table_patient | passive_ab_by | 0.0000 | -3.4473 | 32 |
| nonce | entity | agent | json_agent | active_ba | 0.0000 | -3.3604 | 32 |
| nonce | entity | patient | json_patient | active_ab | 0.0000 | -3.1895 | 32 |
| nonce | entity | patient | role_table_patient | active_ab | 0.0000 | -3.1797 | 32 |
| ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -3.1563 | 32 |
| nonce | entity | agent | compact_agent | passive_ba_by | 0.0000 | -3.1426 | 32 |
| nonce | entity | agent | role_table_agent | active_ba | 0.0000 | -2.9390 | 32 |
| nonce | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -2.7188 | 32 |
| ab | entity | agent | role_table_agent | active_ba | 0.0000 | -2.6953 | 32 |
| nonce | entity | agent | compact_agent | active_ba | 0.0000 | -2.5117 | 32 |
| ab | entity | agent | compact_agent | passive_ba_by | 0.0000 | -2.1816 | 32 |
| ab | entity | patient | role_table_patient | active_ab | 0.0000 | -2.0117 | 32 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -1.8672 | 32 |

## qwen3

rows=6144 bases=16 reliable_templates=0 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|
| entity_ab | entity | agent | forced_agent | False | 0.7500 | 0.9219 | 2.0781 |

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| entity_ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -11.9785 | 32 |
| ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -11.1660 | 32 |
| nonce | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -9.4073 | 32 |
| entity_ab | entity | patient | role_table_patient | active_ab | 0.0000 | -8.1016 | 32 |
| entity_ab | letter | agent | role_table_agent | active_ba | 0.0000 | -7.9629 | 32 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -7.4805 | 32 |
| nonce | entity | agent | forced_agent | passive_ba_by | 0.0000 | -6.0859 | 32 |
| nonce | entity | agent | forced_agent | active_ba | 0.0000 | -5.6680 | 32 |
| ab | entity | agent | role_table_agent | active_ba | 0.0000 | -5.5254 | 32 |
| entity_ab | letter | patient | role_table_patient | active_ab | 0.0000 | -5.5234 | 32 |
| ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -5.5054 | 32 |
| ab | letter | patient | role_table_patient | active_ab | 0.0000 | -5.4023 | 32 |
| entity_ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -5.1250 | 32 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -5.0918 | 32 |
| nonce | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -4.7480 | 32 |
| ab | letter | patient | json_patient | active_ab | 0.0000 | -4.4531 | 32 |

