# Phase 841 Signed Complementary Gear Role Validation (confirm)

- Source: Phase 840 strict pair rows and inferred role signs.
- Boundary: patch-mode perturbation evidence; not full natural ablation.

## Model Summary

| model | skipped | candidates | roles | rows | cases | pair-original target | target lost vs original | negative-role needed | positive-role needed | object_echo | format_echo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 2 | 3 | 256 | 4 | 14 | 10 | 10 | 0 | 66 | 64 |

## Role Signs

### deepseek7b

| component | role | mean signed sum | observations |
|---|---|---:|---:|
| `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `positive_carrier` | 1.2119 | 2 |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `positive_carrier` | 1.2119 | 2 |
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | `negative_suppressor_or_rewriter` | -3.1581 | 4 |

## Mode Summary

| model | mode | n | target | lost vs original | negative needed | positive needed | mean quality | mean delta quality | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip_negative` | 32 | 10 | 4 | 4 | 0 | -0.0573 | -0.2097 | `{"format_echo": 8, "object_echo": 14, "target_equivalent": 10}` |
| deepseek7b | `flip_positive` | 32 | 16 | 0 | 0 | 0 | 0.2526 | 0.1001 | `{"format_echo": 8, "object_echo": 4, "target_equivalent": 16, "unknown_other": 4}` |
| deepseek7b | `negative_only` | 32 | 16 | 0 | 0 | 0 | 0.2580 | 0.1056 | `{"format_echo": 8, "object_echo": 4, "target_equivalent": 16, "unknown_other": 4}` |
| deepseek7b | `pair_original` | 32 | 14 | 0 | 0 | 0 | 0.1524 | 0.0000 | `{"format_echo": 8, "object_echo": 6, "target_equivalent": 14, "unknown_other": 4}` |
| deepseek7b | `positive_only` | 32 | 8 | 6 | 6 | 0 | -0.1777 | -0.3302 | `{"format_echo": 8, "object_echo": 16, "target_equivalent": 8}` |
| deepseek7b | `zero_all` | 32 | 16 | 0 | 0 | 0 | 0.0463 | -0.1061 | `{"format_echo": 8, "object_echo": 8, "target_equivalent": 16}` |
| deepseek7b | `zero_negative` | 32 | 16 | 0 | 0 | 0 | 0.0798 | -0.0726 | `{"format_echo": 8, "object_echo": 8, "target_equivalent": 16}` |
| deepseek7b | `zero_positive` | 32 | 14 | 0 | 0 | 0 | 0.1290 | -0.0234 | `{"format_echo": 8, "object_echo": 6, "target_equivalent": 14, "unknown_other": 4}` |

## Top Mode Rows

| model | case | donor | mode | class | output | target | original target | lost | neg needed | pos needed | quality | delta quality |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5691 | -1.6421 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5691 | -1.6421 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5601 | -1.6331 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5601 | -1.6331 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.6147 | -1.5604 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.6147 | -1.5604 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5131 | -1.4588 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5131 | -1.4588 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5789 | -0.9787 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5789 | -0.9787 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_all` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.9541 | 1.7375 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_all` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.9541 | 1.7375 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_negative` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.9210 | 1.7044 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_negative` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.9210 | 1.7044 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `negative_only` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.8748 | 1.6583 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `negative_only` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.8748 | 1.6583 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_positive` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.8442 | 1.6277 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_positive` | `target_equivalent` | polygon | 1 | 0 | 0 | 0 | 0 | 0.8442 | 1.6277 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `flip_negative` | `target_equivalent` | Geometric shape | 1 | 1 | 0 | 0 | 0 | 1.5773 | 1.1775 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `flip_negative` | `target_equivalent` | Geometric shape | 1 | 1 | 0 | 0 | 0 | 1.5773 | 1.1775 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `zero_all` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8911 | -0.9069 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `zero_all` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8911 | -0.9069 |
| deepseek7b | `p816_oxygen_chemical_element` | `object_only` | `flip_positive` | `unknown_other` | Oxygen is a gas | 0 | 0 | 0 | 0 | 0 | 0.1281 | 0.9006 |
| deepseek7b | `p816_oxygen_chemical_element` | `object_only` | `flip_positive` | `unknown_other` | Oxygen is a gas | 0 | 0 | 0 | 0 | 0 | 0.1281 | 0.9006 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `flip_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8047 | -0.8204 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `flip_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8047 | -0.8204 |
| deepseek7b | `p816_oxygen_chemical_element` | `object_only` | `negative_only` | `unknown_other` | Oxygen is a gas | 0 | 0 | 0 | 0 | 0 | 0.0417 | 0.8142 |
| deepseek7b | `p816_oxygen_chemical_element` | `object_only` | `negative_only` | `unknown_other` | Oxygen is a gas | 0 | 0 | 0 | 0 | 0 | 0.0417 | 0.8142 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `zero_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.7984 | -0.8142 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `zero_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.7984 | -0.8142 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `zero_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.9097 | -0.7988 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `zero_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.9097 | -0.7988 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `flip_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.9082 | -0.7973 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `flip_negative` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.9082 | -0.7973 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `zero_all` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8911 | -0.7802 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `zero_all` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8911 | -0.7802 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `positive_only` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.7168 | -0.7325 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_question` | `positive_only` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.7168 | -0.7325 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `positive_only` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8344 | -0.7235 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `positive_only` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8344 | -0.7235 |
| deepseek7b | `p816_oxygen_chemical_element` | `object_only` | `zero_positive` | `unknown_other` | Oxygen is a gas | 0 | 0 | 0 | 0 | 0 | -0.0637 | 0.7088 |
| deepseek7b | `p816_oxygen_chemical_element` | `object_only` | `zero_positive` | `unknown_other` | Oxygen is a gas | 0 | 0 | 0 | 0 | 0 | -0.0637 | 0.7088 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `zero_positive` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.7899 | -0.6790 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `zero_positive` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.7899 | -0.6790 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `negative_only` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.6845 | -0.5736 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `negative_only` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.6845 | -0.5736 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9601 | 0.5603 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9601 | 0.5603 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `zero_all` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9541 | 0.5542 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `zero_all` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9541 | 0.5542 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `flip_positive` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.5840 | -0.4731 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `flip_positive` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.5840 | -0.4731 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_category` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3997 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_category` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3997 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_question` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3777 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_question` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3777 |
| deepseek7b | `p816_doctor_medical_worker` | `object_only` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3659 |
| deepseek7b | `p816_doctor_medical_worker` | `object_only` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3659 |
| deepseek7b | `p816_doctor_medical_worker` | `exact_choices` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3357 |
| deepseek7b | `p816_doctor_medical_worker` | `exact_choices` | `zero_all` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.8460 | -0.3357 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_category` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7670 | -0.3206 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_category` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7670 | -0.3206 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_question` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7786 | -0.3103 |
| deepseek7b | `p816_doctor_medical_worker` | `natural_question` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7786 | -0.3103 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_negative` | `object_echo` | Triangle | 0 | 0 | 0 | 0 | 0 | -0.5143 | 0.2691 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_negative` | `object_echo` | Triangle | 0 | 0 | 0 | 0 | 0 | -0.5143 | 0.2691 |
| deepseek7b | `p816_gold_precious_metal` | `exact_choices` | `flip_negative` | `target_equivalent` | Precious Metal | 1 | 1 | 0 | 0 | 0 | 0.8360 | -0.2606 |
| deepseek7b | `p816_gold_precious_metal` | `exact_choices` | `flip_negative` | `target_equivalent` | Precious Metal | 1 | 1 | 0 | 0 | 0 | 0.8360 | -0.2606 |
| deepseek7b | `p816_doctor_medical_worker` | `object_only` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7404 | -0.2603 |
| deepseek7b | `p816_doctor_medical_worker` | `object_only` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7404 | -0.2603 |
| deepseek7b | `p816_doctor_medical_worker` | `exact_choices` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7615 | -0.2512 |
| deepseek7b | `p816_doctor_medical_worker` | `exact_choices` | `zero_negative` | `format_echo` | ___________ | 0 | 0 | 0 | 0 | 0 | -0.7615 | -0.2512 |
| deepseek7b | `p816_doctor_medical_worker` | `exact_choices` | `flip_negative` | `format_echo` | _________ | 0 | 0 | 0 | 0 | 0 | -0.2808 | 0.2295 |
| deepseek7b | `p816_doctor_medical_worker` | `exact_choices` | `flip_negative` | `format_echo` | _________ | 0 | 0 | 0 | 0 | 0 | -0.2808 | 0.2295 |
| deepseek7b | `p816_oxygen_chemical_element` | `exact_choices` | `zero_all` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8911 | -0.2288 |
| deepseek7b | `p816_oxygen_chemical_element` | `exact_choices` | `zero_all` | `object_echo` | O2 | 0 | 0 | 0 | 0 | 0 | -0.8911 | -0.2288 |
| deepseek7b | `p816_doctor_medical_worker` | `object_only` | `flip_negative` | `format_echo` | _________ | 0 | 0 | 0 | 0 | 0 | -0.2706 | 0.2095 |
| deepseek7b | `p816_doctor_medical_worker` | `object_only` | `flip_negative` | `format_echo` | _________ | 0 | 0 | 0 | 0 | 0 | -0.2706 | 0.2095 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8873 | -0.1857 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8873 | -0.1857 |
