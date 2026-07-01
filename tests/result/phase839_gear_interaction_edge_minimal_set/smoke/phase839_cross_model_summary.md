# Phase 839 Gear Interaction Edge and Minimal Set (smoke)

- Source: Phase 838 top gear components, tested on held-out cases.
- Boundary: patch-intervention interaction test; not natural mechanism proof.

## Model Summary

| model | rows | components | cases | target | object_echo | format_echo | degraded | positive interaction | minimal candidates | mean quality | mean echo risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 12 | 2 | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 0.4869 | 0.0000 |
| glm4 | 12 | 2 | 2 | 12 | 0 | 0 | 0 | 0 | 0 | 1.0198 | 0.0000 |
| deepseek7b | 12 | 2 | 2 | 9 | 3 | 0 | 0 | 0 | 0 | 0.4793 | 0.2500 |

## Combo Kind Summary

| model | combo kind | n | target | object_echo | format_echo | positive interaction | minimal | mean quality | mean echo risk | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `pair` | 4 | 2 | 0 | 0 | 0 | 0 | 0.4910 | 0.0000 | `{"broad_near_miss": 2, "target_equivalent": 2}` |
| qwen3 | `single` | 8 | 4 | 0 | 0 | 0 | 0 | 0.4849 | 0.0000 | `{"broad_near_miss": 4, "target_equivalent": 4}` |
| glm4 | `pair` | 4 | 4 | 0 | 0 | 0 | 0 | 1.0328 | 0.0000 | `{"target_equivalent": 4}` |
| glm4 | `single` | 8 | 8 | 0 | 0 | 0 | 0 | 1.0132 | 0.0000 | `{"target_equivalent": 8}` |
| deepseek7b | `pair` | 4 | 3 | 1 | 0 | 0 | 0 | 0.4249 | 0.2500 | `{"object_echo": 1, "target_equivalent": 3}` |
| deepseek7b | `single` | 8 | 6 | 2 | 0 | 0 | 0 | 0.5065 | 0.2500 | `{"object_echo": 2, "target_equivalent": 6}` |

## Top Interaction Rows

| model | case | donor | kind | combo | class | output | quality | gain | echo risk | echo gain | minimal |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
