# Phase306 Symbolic Role Query Calibration Summary

## deepseek7b

rows=12288 bases=32 reliable_templates=0 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| entity_ab | entity | patient | json_patient | active_ab | 0.0000 | -4.6563 | 64 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -4.2832 | 64 |
| nonce | letter | patient | role_table_patient | active_ab | 0.0000 | -4.0386 | 64 |
| entity_ab | letter | agent | role_table_agent | active_ba | 0.0000 | -3.9951 | 64 |
| nonce | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -3.9619 | 64 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -3.9346 | 64 |
| nonce | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.8848 | 64 |
| ab | letter | patient | role_table_patient | active_ab | 0.0000 | -3.8779 | 64 |
| nonce | letter | agent | forced_agent | active_ba | 0.0000 | -3.6523 | 64 |
| nonce | letter | patient | forced_patient | active_ab | 0.0000 | -3.4687 | 64 |
| ab | letter | patient | forced_patient | active_ab | 0.0000 | -3.3872 | 64 |
| entity_ab | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.3730 | 64 |
| nonce | letter | patient | forced_patient | passive_ab_by | 0.0000 | -3.3550 | 64 |
| ab | letter | agent | forced_agent | passive_ba_by | 0.0000 | -3.3369 | 64 |
| entity_ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -3.1885 | 64 |
| entity_ab | letter | patient | role_table_patient | active_ab | 0.0000 | -3.1543 | 64 |

## glm4

rows=12288 bases=32 reliable_templates=3 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|
| ab | letter | patient | json_patient | False | 0.8438 | 0.9219 | 0.7295 |
| entity_ab | entity | agent | json_agent | True | 0.9688 | 0.9922 | 1.7852 |
| entity_ab | entity | agent | role_table_agent | True | 0.9375 | 0.9609 | 1.2734 |
| entity_ab | entity | agent | compact_agent | True | 0.9062 | 0.9609 | 1.3047 |
| entity_ab | entity | patient | forced_patient | False | 0.7969 | 0.8516 | 0.5371 |

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| nonce | entity | patient | json_patient | passive_ab_by | 0.0000 | -3.5742 | 64 |
| ab | entity | patient | role_table_patient | passive_ab_by | 0.0000 | -3.5732 | 64 |
| nonce | entity | agent | json_agent | passive_ba_by | 0.0000 | -3.4541 | 64 |
| nonce | entity | agent | json_agent | active_ba | 0.0000 | -3.3345 | 64 |
| nonce | entity | patient | role_table_patient | passive_ab_by | 0.0000 | -3.3301 | 64 |
| nonce | entity | agent | role_table_agent | active_ba | 0.0000 | -3.3220 | 64 |
| nonce | entity | patient | json_patient | active_ab | 0.0000 | -3.2266 | 64 |
| nonce | entity | patient | role_table_patient | active_ab | 0.0000 | -3.1787 | 64 |
| nonce | entity | agent | compact_agent | passive_ba_by | 0.0000 | -3.0498 | 64 |
| ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -3.0430 | 64 |
| nonce | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -2.7979 | 64 |
| ab | entity | agent | role_table_agent | active_ba | 0.0000 | -2.7432 | 64 |
| nonce | entity | agent | compact_agent | active_ba | 0.0000 | -2.4795 | 64 |
| ab | entity | agent | compact_agent | passive_ba_by | 0.0000 | -2.2793 | 64 |
| ab | entity | patient | role_table_patient | active_ab | 0.0000 | -2.2480 | 64 |
| entity_ab | letter | patient | compact_patient | active_ab | 0.0000 | -1.8379 | 64 |

## qwen3

rows=12288 bases=32 reliable_templates=0 nonfinite=0

### Template Candidates

| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |
|---|---|---|---|---:|---:|---:|---:|
| entity_ab | entity | agent | forced_agent | False | 0.8281 | 0.9375 | 2.3359 |

### Weakest State Rows

| entity | answer | query | template | state | acc | margin | n |
|---|---|---|---|---|---:|---:|---:|
| entity_ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -11.9170 | 64 |
| ab | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -11.3018 | 64 |
| nonce | letter | agent | role_table_agent | passive_ba_by | 0.0000 | -9.4707 | 64 |
| entity_ab | letter | agent | role_table_agent | active_ba | 0.0000 | -8.0518 | 64 |
| nonce | letter | agent | role_table_agent | active_ba | 0.0000 | -7.7148 | 64 |
| entity_ab | entity | patient | role_table_patient | active_ab | 0.0000 | -7.6641 | 64 |
| nonce | entity | agent | forced_agent | passive_ba_by | 0.0000 | -6.1504 | 64 |
| ab | letter | patient | role_table_patient | active_ab | 0.0000 | -5.9004 | 64 |
| ab | entity | agent | role_table_agent | active_ba | 0.0000 | -5.6372 | 64 |
| nonce | entity | agent | forced_agent | active_ba | 0.0000 | -5.6230 | 64 |
| ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -5.4949 | 64 |
| entity_ab | letter | patient | role_table_patient | active_ab | 0.0000 | -5.2451 | 64 |
| entity_ab | letter | patient | role_table_patient | passive_ab_by | 0.0000 | -5.1182 | 64 |
| nonce | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -4.6865 | 64 |
| entity_ab | entity | agent | role_table_agent | passive_ba_by | 0.0000 | -4.5195 | 64 |
| nonce | entity | agent | role_table_agent | active_ba | 0.0000 | -4.3145 | 64 |

