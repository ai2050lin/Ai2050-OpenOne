# Phase 842 Negative MLP Gear Channel Decomposition (confirm)

- Source: Phase 841 negative MLP role candidate.
- Boundary: channel-level patch decomposition; not natural ablation.

## Model Summary

| model | skipped | neg comps | rows | cases | full-original target | lost vs full | gained vs full | object_echo | format_echo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 1 | 268 | 1 | 4 | 73 | 0 | 72 | 0 |

## Mode Family Summary

| model | mode family | n | target | lost vs full | gained vs full | mean quality | mean delta quality | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip_one` | 64 | 59 | 5 | 0 | 0.6909 | -0.0934 | `{"object_echo": 4, "target_equivalent": 59, "unknown_other": 1}` |
| deepseek7b | `full` | 12 | 9 | 3 | 0 | 0.5433 | -0.2411 | `{"object_echo": 3, "target_equivalent": 9}` |
| deepseek7b | `leave_one_out` | 64 | 60 | 4 | 0 | 0.6925 | -0.0918 | `{"object_echo": 4, "target_equivalent": 60}` |
| deepseek7b | `single_original` | 64 | 4 | 60 | 0 | -0.5998 | -1.3842 | `{"object_echo": 60, "target_equivalent": 4}` |
| deepseek7b | `zero_one` | 64 | 63 | 1 | 0 | 0.7645 | -0.0198 | `{"object_echo": 1, "target_equivalent": 63}` |

## Channel Records

| model | local | channel id | single target | leave-one-out loss | flip-one loss | zero-one loss | mean delta quality | classes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | 8 | 7899 | 4 | 4 | 4 | 0 | -0.7426 | `{"object_echo": 7, "target_equivalent": 8, "unknown_other": 1}` |
| deepseek7b | 23 | 9305 | 0 | 0 | 1 | 0 | -0.4776 | `{"object_echo": 5, "target_equivalent": 11}` |
| deepseek7b | 3 | 14618 | 0 | 0 | 0 | 0 | -0.2281 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 14 | 3350 | 0 | 0 | 0 | 0 | -0.3603 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 12 | 1303 | 0 | 0 | 0 | 0 | -0.3661 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 9 | 16847 | 0 | 0 | 0 | 0 | -0.3672 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 15 | 2644 | 0 | 0 | 0 | 0 | -0.3675 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 0 | 1629 | 0 | 0 | 0 | 0 | -0.3680 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 22 | 13970 | 0 | 0 | 0 | 0 | -0.3687 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 10 | 16257 | 0 | 0 | 0 | 0 | -0.3687 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 21 | 17523 | 0 | 0 | 0 | 0 | -0.3689 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 13 | 6224 | 0 | 0 | 0 | 0 | -0.3710 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 1 | 2295 | 0 | 0 | 0 | 0 | -0.3719 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 11 | 1645 | 0 | 0 | 0 | 0 | -0.3754 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 2 | 12746 | 0 | 0 | 0 | 0 | -0.3758 | `{"object_echo": 4, "target_equivalent": 12}` |
| deepseek7b | 20 | 15305 | 0 | 0 | 0 | 1 | -0.4789 | `{"object_echo": 5, "target_equivalent": 11}` |

## Top Channel Rows

| model | case | donor | mode | local | channel id | class | output | target | full target | lost | gained | quality | delta |
|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_one_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.8440 | -1.7188 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_14` | 14 | 3350 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7180 | -1.7067 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_12` | 12 | 1303 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7152 | -1.7039 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_one_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.8272 | -1.7020 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7024 | -1.6911 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_21` | 21 | 17523 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7020 | -1.6907 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7014 | -1.6901 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_9` | 9 | 16847 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6973 | -1.6860 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7991 | -1.6859 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_10` | 10 | 16257 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6941 | -1.6828 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_11` | 11 | 1645 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6910 | -1.6797 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_22` | 22 | 13970 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6899 | -1.6786 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -1.2910 | -1.6780 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6889 | -1.6776 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6882 | -1.6769 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_15` | 15 | 2644 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6871 | -1.6758 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6866 | -1.6753 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `leave_one_out_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7993 | -1.6741 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7992 | -1.6740 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `leave_one_out_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -1.2848 | -1.6718 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_13` | 13 | 6224 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6797 | -1.6684 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7812 | -1.6679 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6748 | -1.6635 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_one_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7839 | -1.6587 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7581 | -1.6448 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `full_flip` | None | None | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6410 | -1.6297 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6390 | -1.6277 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_12` | 12 | 1303 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7200 | -1.6068 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6068 | -1.5955 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_22` | 22 | 13970 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7016 | -1.5883 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_14` | 14 | 3350 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6983 | -1.5851 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_21` | 21 | 17523 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6982 | -1.5849 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_10` | 10 | 16257 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6974 | -1.5841 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_21` | 21 | 17523 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7081 | -1.5829 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6956 | -1.5823 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6941 | -1.5808 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_9` | 9 | 16847 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6934 | -1.5802 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6927 | -1.5794 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_15` | 15 | 2644 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6920 | -1.5788 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_12` | 12 | 1303 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7034 | -1.5782 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_14` | 14 | 3350 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7033 | -1.5781 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6913 | -1.5780 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6995 | -1.5744 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6993 | -1.5742 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_10` | 10 | 16257 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6989 | -1.5738 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_15` | 15 | 2644 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6963 | -1.5711 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6841 | -1.5708 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6957 | -1.5706 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6938 | -1.5686 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_22` | 22 | 13970 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6928 | -1.5676 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6922 | -1.5670 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_13` | 13 | 6224 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6786 | -1.5654 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_9` | 9 | 16847 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6875 | -1.5623 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_11` | 11 | 1645 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6858 | -1.5606 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_13` | 13 | 6224 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6838 | -1.5587 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_11` | 11 | 1645 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6335 | -1.5202 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `full_flip` | None | None | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.5455 | -1.4323 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `full_flip` | None | None | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.5374 | -1.4122 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_14` | 14 | 3350 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7222 | -1.1092 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_12` | 12 | 1303 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7188 | -1.1058 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7113 | -1.0983 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_21` | 21 | 17523 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7097 | -1.0967 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_9` | 9 | 16847 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7048 | -1.0919 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7040 | -1.0911 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_10` | 10 | 16257 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7039 | -1.0909 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6938 | -1.0808 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_15` | 15 | 2644 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6832 | -1.0703 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_22` | 22 | 13970 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6821 | -1.0692 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_13` | 13 | 6224 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6730 | -1.0600 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6589 | -1.0460 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_11` | 11 | 1645 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6315 | -1.0186 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.5981 | -0.9851 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `flip_one_8` | 8 | 7899 | `unknown_other` | [Answer Here] | 0 | 1 | 1 | 0 | -0.5643 | -0.9513 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `full_flip` | None | None | `target_equivalent` | Geometric shape | 1 | 1 | 0 | 0 | 1.5533 | 1.1663 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `flip_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.5019 | 1.1149 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `single_original_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9517 | 0.5647 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `leave_one_out_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9343 | 0.5473 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `zero_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9223 | 0.5353 |
| deepseek7b | `p816_triangle_geometric_shape` | `exact_choices` | `full_zero` | None | None | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8881 | 0.5011 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `flip_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.1390 | 0.2641 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.1263 | 0.2396 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `single_original_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9856 | 0.1108 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `leave_one_out_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9826 | 0.1078 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `full_zero` | None | None | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8881 | -0.1006 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9855 | 0.0987 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9654 | 0.0906 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9730 | 0.0863 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9670 | 0.0802 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8253 | -0.0614 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9368 | 0.0500 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_one_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8305 | -0.0444 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0291 | 0.0404 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_23` | 23 | 9305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8517 | -0.0351 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0214 | 0.0327 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8543 | -0.0325 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9179 | 0.0311 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9577 | -0.0310 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_category` | `zero_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8449 | -0.0300 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9593 | -0.0294 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8575 | -0.0292 |
