# Phase 842 Negative MLP Gear Channel Decomposition (smoke)

- Source: Phase 841 negative MLP role candidate.
- Boundary: channel-level patch decomposition; not natural ablation.

## Model Summary

| model | skipped | neg comps | rows | cases | full-original target | lost vs full | gained vs full | object_echo | format_echo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 1 | 19 | 1 | 0 | 0 | 0 | 19 | 0 |

## Mode Family Summary

| model | mode family | n | target | lost vs full | gained vs full | mean quality | mean delta quality | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip_one` | 4 | 0 | 0 | 0 | -0.7532 | 0.0405 | `{"object_echo": 4}` |
| deepseek7b | `full` | 3 | 0 | 0 | 0 | -0.7221 | 0.0717 | `{"object_echo": 3}` |
| deepseek7b | `leave_one_out` | 4 | 0 | 0 | 0 | -0.7691 | 0.0247 | `{"object_echo": 4}` |
| deepseek7b | `single_original` | 4 | 0 | 0 | 0 | -0.7237 | 0.0701 | `{"object_echo": 4}` |
| deepseek7b | `zero_one` | 4 | 0 | 0 | 0 | -0.7869 | 0.0069 | `{"object_echo": 4}` |

## Channel Records

| model | local | channel id | single target | leave-one-out loss | flip-one loss | zero-one loss | mean delta quality | classes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | 3 | 14618 | 0 | 0 | 0 | 0 | 0.0906 | `{"object_echo": 4}` |
| deepseek7b | 0 | 1629 | 0 | 0 | 0 | 0 | 0.0265 | `{"object_echo": 4}` |
| deepseek7b | 2 | 12746 | 0 | 0 | 0 | 0 | 0.0135 | `{"object_echo": 4}` |
| deepseek7b | 1 | 2295 | 0 | 0 | 0 | 0 | 0.0116 | `{"object_echo": 4}` |

## Top Channel Rows

| model | case | donor | mode | local | channel id | class | output | target | full target | lost | gained | quality | delta |
|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.6022 | 0.1916 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `full_flip` | None | None | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.6196 | 0.1742 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.6944 | 0.0994 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.6946 | 0.0992 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.6971 | 0.0967 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.6995 | 0.0943 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7120 | 0.0818 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `full_zero` | None | None | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7529 | 0.0409 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.8335 | -0.0397 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.8159 | -0.0221 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.8123 | -0.0186 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `single_original_3` | 3 | 14618 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.8037 | -0.0099 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.8017 | -0.0080 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_one_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7863 | 0.0075 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_one_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7966 | -0.0028 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_2` | 2 | 12746 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7956 | -0.0019 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_0` | 0 | 1629 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7920 | 0.0018 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `leave_one_out_1` | 1 | 2295 | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7941 | -0.0004 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `full_original` | None | None | `object_echo` | Triangle | 0 | 0 | 0 | 0 | -0.7938 | 0.0000 |
