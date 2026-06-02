# Phase306 Symbolic Role Query Calibration Summary

## deepseek7b

rows=768 bases=2 reliable_templates=0 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| entity_ab | entity | patient | json_patient | active_ab | 0.0000 | -4.9062 | 4 |
| nonce | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -4.1875 | 4 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -4.1406 | 4 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -4.0781 | 4 |
| nonce | letter | patient | role_table_patient | active_ab | 0.0000 | -4.0625 | 4 |
| entity_ab | letter | agent | role_table_agent | active_ba | 0.0000 | -4.0625 | 4 |
| entity_ab | entity | patient | compact_patient | active_ab | 0.0000 | -3.9375 | 4 |
| nonce | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.9062 | 4 |
| ab | letter | patient | forced_patient | active_ab | 0.0000 | -3.7188 | 4 |
| ab | letter | patient | role_table_patient | active_ab | 0.0000 | -3.5625 | 4 |
| ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -3.5313 | 4 |
| nonce | letter | patient | forced_patient | active_ab | 0.0000 | -3.5156 | 4 |
| nonce | letter | agent | forced_agent | active_ba | 0.0000 | -3.3906 | 4 |
| nonce | letter | patient | forced_patient | passive_ab_by | 0.0000 | -3.3906 | 4 |
| entity_ab | letter | agent | forced_agent | active_ba | 0.0000 | -3.2969 | 4 |
| entity_ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -3.2500 | 4 |

## glm4

rows=768 bases=2 reliable_templates=5 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|
| ab | letter | patient | json_patient | True | 1.0000 | 1.0000 | 0.9062 |
| entity_ab | entity | agent | json_agent | True | 1.0000 | 1.0000 | 2.1250 |
| entity_ab | entity | agent | compact_agent | True | 1.0000 | 1.0000 | 1.5000 |
| entity_ab | entity | agent | role_table_agent | True | 1.0000 | 1.0000 | 0.9375 |
| entity_ab | entity | agent | forced_agent | True | 1.0000 | 1.0000 | 0.6250 |
| entity_ab | entity | patient | forced_patient | False | 0.7500 | 0.8750 | 0.2188 |

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| ab | letter | agent | role_table_agent | active_ba | 0.0000 | -6.4063 | 4 |
| nonce | entity | patient | role_table_patient | passive_ab_by | 0.0000 | -3.6719 | 4 |
| nonce | entity | patient | json_patient | passive_ab_by | 0.0000 | -3.6094 | 4 |
| nonce | entity | patient | role_table_patient | active_ab | 0.0000 | -3.5000 | 4 |
| nonce | entity | agent | json_agent | passive_ba_by | 0.0000 | -3.4687 | 4 |
| nonce | entity | patient | json_patient | active_ab | 0.0000 | -3.3125 | 4 |
| nonce | entity | agent | compact_agent | passive_ba_by | 0.0000 | -3.1094 | 4 |
| ab | entity | patient | role_table_patient | passive_ab_by | 0.0000 | -3.0938 | 4 |
| ab | entity | agent | role_table_agent | active_ba | 0.0000 | -3.0312 | 4 |
| nonce | entity | agent | json_agent | active_ba | 0.0000 | -2.9062 | 4 |
| ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -2.8750 | 4 |
| ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -2.5625 | 4 |
| nonce | entity | agent | compact_agent | active_ba | 0.0000 | -2.4375 | 4 |
| nonce | entity | agent | role_table_agent | active_ba | 0.0000 | -2.4375 | 4 |
| nonce | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -2.3125 | 4 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -2.0938 | 4 |

## qwen3

rows=768 bases=2 reliable_templates=2 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|
| ab | letter | agent | json_agent | True | 1.0000 | 1.0000 | 2.0938 |
| entity_ab | entity | agent | forced_agent | True | 1.0000 | 1.0000 | 1.6851 |
| entity_ab | entity | patient | forced_patient | False | 0.7500 | 0.8750 | 0.0312 |

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| entity_ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -14.6719 | 4 |
| ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -11.2812 | 4 |
| entity_ab | entity | patient | role_table_patient | active_ab | 0.0000 | -8.9375 | 4 |
| entity_ab | letter | agent | role_table_agent | active_ba | 0.0000 | -8.8438 | 4 |
| nonce | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -8.8438 | 4 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -6.6875 | 4 |
| ab | entity | agent | role_table_agent | active_ba | 0.0000 | -6.3125 | 4 |
| nonce | entity | agent | forced_agent | passive_ba_by | 0.0000 | -6.1250 | 4 |
| entity_ab | letter | patient | role_table_patient | active_ab | 0.0000 | -6.1250 | 4 |
| nonce | entity | agent | forced_agent | active_ba | 0.0000 | -5.8125 | 4 |
| entity_ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -5.6875 | 4 |
| ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -5.5703 | 4 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -5.4688 | 4 |
| nonce | letter | agent | forced_agent | passive_ba_by | 0.0000 | -4.9687 | 4 |
| ab | letter | agent | role_table_agent | active_ba | 0.0000 | -4.8438 | 4 |
| entity_ab | entity | patient | json_patient | active_ab | 0.0000 | -4.8125 | 4 |

