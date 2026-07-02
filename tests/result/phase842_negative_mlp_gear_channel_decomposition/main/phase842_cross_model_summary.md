# Phase 842 Negative MLP Gear Channel Decomposition (main)

- Source: Phase 841 negative MLP role candidate.
- Boundary: channel-level patch decomposition; not natural ablation.

## Model Summary

| model | skipped | neg comps | rows | cases | full-original target | lost vs full | gained vs full | object_echo | format_echo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 1 | 134 | 1 | 2 | 36 | 0 | 36 | 0 |

## Mode Family Summary

| model | mode family | n | target | lost vs full | gained vs full | mean quality | mean delta quality | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip_one` | 32 | 30 | 2 | 0 | 0.8413 | -0.0965 | `{"object_echo": 2, "target_equivalent": 30}` |
| deepseek7b | `full` | 6 | 4 | 2 | 0 | 0.4109 | -0.5269 | `{"object_echo": 2, "target_equivalent": 4}` |
| deepseek7b | `leave_one_out` | 32 | 30 | 2 | 0 | 0.8367 | -0.1010 | `{"object_echo": 2, "target_equivalent": 30}` |
| deepseek7b | `single_original` | 32 | 2 | 30 | 0 | -0.5907 | -1.5284 | `{"object_echo": 30, "target_equivalent": 2}` |
| deepseek7b | `zero_one` | 32 | 32 | 0 | 0 | 0.9335 | -0.0043 | `{"target_equivalent": 32}` |

## Channel Records

| model | local | channel id | single target | leave-one-out loss | flip-one loss | zero-one loss | mean delta quality | classes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | 8 | 7899 | 2 | 2 | 2 | 0 | -0.8194 | `{"object_echo": 4, "target_equivalent": 4}` |
| deepseek7b | 3 | 14618 | 0 | 0 | 0 | 0 | -0.3698 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 14 | 3350 | 0 | 0 | 0 | 0 | -0.3961 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 12 | 1303 | 0 | 0 | 0 | 0 | -0.3994 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 0 | 1629 | 0 | 0 | 0 | 0 | -0.4049 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 9 | 16847 | 0 | 0 | 0 | 0 | -0.4051 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 21 | 17523 | 0 | 0 | 0 | 0 | -0.4061 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 22 | 13970 | 0 | 0 | 0 | 0 | -0.4082 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 10 | 16257 | 0 | 0 | 0 | 0 | -0.4086 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 15 | 2644 | 0 | 0 | 0 | 0 | -0.4099 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 20 | 15305 | 0 | 0 | 0 | 0 | -0.4105 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 13 | 6224 | 0 | 0 | 0 | 0 | -0.4129 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 23 | 9305 | 0 | 0 | 0 | 0 | -0.4165 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 11 | 1645 | 0 | 0 | 0 | 0 | -0.4167 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 2 | 12746 | 0 | 0 | 0 | 0 | -0.4174 | `{"object_echo": 2, "target_equivalent": 6}` |
| deepseek7b | 1 | 2295 | 0 | 0 | 0 | 0 | -0.4188 | `{"object_echo": 2, "target_equivalent": 6}` |

## Top Channel Rows

| model | case | donor | mode | local | channel id | class | output | target | full target | lost | gained | quality | delta |
|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_14` | 14 | 3350 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7180 | -1.7067 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_12` | 12 | 1303 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7152 | -1.7039 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7024 | -1.6911 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_21` | 21 | 17523 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7020 | -1.6907 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7014 | -1.6901 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_9` | 9 | 16847 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6973 | -1.6860 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7991 | -1.6859 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_10` | 10 | 16257 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6941 | -1.6828 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_11` | 11 | 1645 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6910 | -1.6797 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_22` | 22 | 13970 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6899 | -1.6786 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6889 | -1.6776 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6882 | -1.6769 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_15` | 15 | 2644 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6871 | -1.6758 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6866 | -1.6753 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_13` | 13 | 6224 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6797 | -1.6684 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7812 | -1.6679 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6748 | -1.6635 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7581 | -1.6448 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `full_flip` | None | None | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6410 | -1.6297 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6390 | -1.6277 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_12` | 12 | 1303 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7200 | -1.6068 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_8` | 8 | 7899 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6068 | -1.5955 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_22` | 22 | 13970 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.7016 | -1.5883 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_14` | 14 | 3350 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6983 | -1.5851 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_21` | 21 | 17523 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6982 | -1.5849 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_10` | 10 | 16257 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6974 | -1.5841 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6956 | -1.5823 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6941 | -1.5808 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_9` | 9 | 16847 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6934 | -1.5802 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_20` | 20 | 15305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6927 | -1.5794 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_15` | 15 | 2644 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6920 | -1.5788 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6913 | -1.5780 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_23` | 23 | 9305 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6841 | -1.5708 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_13` | 13 | 6224 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6786 | -1.5654 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_11` | 11 | 1645 | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.6335 | -1.5202 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `full_flip` | None | None | `object_echo` | Triangle | 0 | 1 | 1 | 0 | -0.5455 | -1.4323 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.1263 | 0.2396 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `full_zero` | None | None | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8881 | -0.1006 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9855 | 0.0987 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9730 | 0.0863 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9670 | 0.0802 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8253 | -0.0614 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9368 | 0.0500 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0291 | 0.0404 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_23` | 23 | 9305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8517 | -0.0351 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0214 | 0.0327 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8543 | -0.0325 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9179 | 0.0311 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9577 | -0.0310 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9593 | -0.0294 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8575 | -0.0292 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9608 | -0.0280 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_15` | 15 | 2644 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9609 | -0.0278 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8590 | -0.0277 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_13` | 13 | 6224 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8594 | -0.0273 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_2` | 2 | 12746 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8595 | -0.0273 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9639 | -0.0248 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_13` | 13 | 6224 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9641 | -0.0246 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0132 | 0.0245 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9664 | -0.0223 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_23` | 23 | 9305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8647 | -0.0220 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_2` | 2 | 12746 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9669 | -0.0218 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9700 | -0.0187 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_2` | 2 | 12746 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8686 | -0.0181 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8689 | -0.0178 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0065 | 0.0178 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_23` | 23 | 9305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9710 | -0.0177 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_20` | 20 | 15305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8692 | -0.0176 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9041 | 0.0173 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9037 | 0.0170 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_13` | 13 | 6224 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9721 | -0.0166 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_0` | 0 | 1629 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0031 | 0.0144 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8995 | 0.0127 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_3` | 3 | 14618 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9760 | -0.0127 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `single_original_8` | 8 | 7899 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9762 | -0.0125 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 1.0012 | 0.0125 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_15` | 15 | 2644 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9764 | -0.0123 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_22` | 22 | 13970 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9775 | -0.0112 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_15` | 15 | 2644 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9999 | 0.0112 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_20` | 20 | 15305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9776 | -0.0111 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_14` | 14 | 3350 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9998 | 0.0111 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_11` | 11 | 1645 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9778 | -0.0109 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_15` | 15 | 2644 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8964 | 0.0096 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9797 | -0.0090 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_22` | 22 | 13970 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8951 | 0.0084 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_one_21` | 21 | 17523 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9971 | 0.0084 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_21` | 21 | 17523 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9969 | 0.0082 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_20` | 20 | 15305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9809 | -0.0078 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_12` | 12 | 1303 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8793 | -0.0075 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_one_2` | 2 | 12746 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9815 | -0.0072 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9956 | 0.0069 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_13` | 13 | 6224 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8800 | -0.0068 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_23` | 23 | 9305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9819 | -0.0068 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_22` | 22 | 13970 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9823 | -0.0064 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_20` | 20 | 15305 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9950 | 0.0063 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_9` | 9 | 16847 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8930 | 0.0063 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_9` | 9 | 16847 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8928 | 0.0060 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_1` | 1 | 2295 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8808 | -0.0060 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `leave_one_out_9` | 9 | 16847 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.9945 | 0.0058 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_0` | 0 | 1629 | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0.8925 | 0.0057 |
