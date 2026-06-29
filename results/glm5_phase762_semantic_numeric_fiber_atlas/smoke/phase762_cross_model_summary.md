# Phase 762 Semantic Numeric Fiber Atlas (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Test: convert objects into causal functional fingerprints over object-relation tasks, then test same-domain clustering and compare with first-token embedding baseline.

## Object Fiber Results

| model | tasks | objects | features | interface status | causal NN | embed NN | causal same | causal diff | causal sep | embed sep |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 6 | 6 | 744 | `semantic_numeric_interface_not_established` | 0.000 | 0.000 | 0.000 | -0.027 | 0.027 | -0.055 |
| glm4 | 6 | 6 | 768 | `semantic_numeric_interface_not_established` | 0.000 | 0.000 | 0.000 | -0.121 | 0.121 | -0.033 |
| deepseek7b | 6 | 6 | 792 | `semantic_numeric_interface_not_established` | 0.000 | 0.000 | 0.000 | -0.171 | 0.171 | -0.044 |

## Cross Model Object-Topology Correlation

| pair | common object pairs | centered similarity correlation |
|---|---:|---:|
| `qwen3__glm4` | 15 | -0.149 |
| `qwen3__deepseek7b` | 15 | -0.313 |
| `glm4__deepseek7b` | 15 | 0.649 |

## Nearest Neighbors

| model | object | nearest causal | causal same domain | nearest embedding | embedding same domain |
|---|---|---|---:|---|---:|
| qwen3 | `apple` | `cat` | 0 | `rose` | 0 |
| qwen3 | `cat` | `scissors` | 0 | `rose` | 0 |
| qwen3 | `justice` | `scissors` | 0 | `scissors` | 0 |
| qwen3 | `rose` | `scissors` | 0 | `apple` | 0 |
| qwen3 | `scissors` | `stone` | 0 | `justice` | 0 |
| qwen3 | `stone` | `scissors` | 0 | `apple` | 0 |
| glm4 | `apple` | `stone` | 0 | `stone` | 0 |
| glm4 | `cat` | `stone` | 0 | `rose` | 0 |
| glm4 | `justice` | `stone` | 0 | `rose` | 0 |
| glm4 | `rose` | `stone` | 0 | `stone` | 0 |
| glm4 | `scissors` | `cat` | 0 | `stone` | 0 |
| glm4 | `stone` | `rose` | 0 | `apple` | 0 |
| deepseek7b | `apple` | `justice` | 0 | `stone` | 0 |
| deepseek7b | `cat` | `justice` | 0 | `rose` | 0 |
| deepseek7b | `justice` | `apple` | 0 | `scissors` | 0 |
| deepseek7b | `rose` | `justice` | 0 | `apple` | 0 |
| deepseek7b | `scissors` | `rose` | 0 | `justice` | 0 |
| deepseek7b | `stone` | `justice` | 0 | `apple` | 0 |

## Strict Interpretation

- If causal NN domain accuracy exceeds the embedding baseline, this supports a first semantic-numeric interface signal.
- If the signal appears only in one model, it is model-local and not a universal semantic fiber.
- This phase stays at head/source level; it does not claim neuron-level or parameter-level localization.
