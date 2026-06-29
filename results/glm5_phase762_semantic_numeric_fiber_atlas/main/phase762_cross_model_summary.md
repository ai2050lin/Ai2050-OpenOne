# Phase 762 Semantic Numeric Fiber Atlas (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Test: convert objects into causal functional fingerprints over object-relation tasks, then test same-domain clustering and compare with first-token embedding baseline.

## Object Fiber Results

| model | tasks | objects | features | interface status | causal NN | embed NN | causal same | causal diff | causal sep | embed sep |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 54 | 18 | 2960 | `causal_fiber_domain_signal_above_embedding` | 0.944 | 0.667 | 0.765 | -0.137 | 0.902 | 0.092 |
| glm4 | 54 | 18 | 3000 | `causal_fiber_domain_signal_above_embedding` | 0.778 | 0.611 | 0.548 | -0.081 | 0.628 | 0.048 |
| deepseek7b | 54 | 18 | 3080 | `causal_fiber_domain_signal_above_embedding` | 0.833 | 0.611 | 0.814 | -0.174 | 0.988 | 0.043 |

## Cross Model Object-Topology Correlation

| pair | common object pairs | centered similarity correlation |
|---|---:|---:|
| `qwen3__glm4` | 153 | 0.515 |
| `qwen3__deepseek7b` | 153 | 0.652 |
| `glm4__deepseek7b` | 153 | 0.371 |

## Nearest Neighbors

| model | object | nearest causal | causal same domain | nearest embedding | embedding same domain |
|---|---|---|---:|---|---:|
| qwen3 | `apple` | `banana` | 1 | `banana` | 1 |
| qwen3 | `banana` | `apple` | 1 | `apple` | 1 |
| qwen3 | `bird` | `dog` | 1 | `dog` | 1 |
| qwen3 | `cat` | `dog` | 1 | `dog` | 1 |
| qwen3 | `chair` | `cup` | 1 | `knife` | 0 |
| qwen3 | `cup` | `chair` | 1 | `apple` | 0 |
| qwen3 | `dog` | `cat` | 1 | `cat` | 1 |
| qwen3 | `freedom` | `time` | 1 | `justice` | 1 |
| qwen3 | `hammer` | `knife` | 1 | `knife` | 1 |
| qwen3 | `justice` | `time` | 1 | `freedom` | 1 |
| qwen3 | `knife` | `hammer` | 1 | `scissors` | 1 |
| qwen3 | `oak` | `rose` | 1 | `apple` | 0 |
| qwen3 | `pear` | `banana` | 1 | `apple` | 1 |
| qwen3 | `rose` | `oak` | 1 | `apple` | 0 |
| qwen3 | `scissors` | `hammer` | 1 | `knife` | 1 |
| qwen3 | `stone` | `chair` | 1 | `apple` | 0 |
| qwen3 | `time` | `freedom` | 1 | `freedom` | 1 |
| qwen3 | `wheat` | `pear` | 0 | `banana` | 0 |
| glm4 | `apple` | `pear` | 1 | `banana` | 1 |
| glm4 | `banana` | `apple` | 1 | `apple` | 1 |
| glm4 | `bird` | `rose` | 0 | `dog` | 1 |
| glm4 | `cat` | `dog` | 1 | `dog` | 1 |
| glm4 | `chair` | `justice` | 0 | `knife` | 0 |
| glm4 | `cup` | `time` | 0 | `cat` | 0 |
| glm4 | `dog` | `cat` | 1 | `cat` | 1 |
| glm4 | `freedom` | `justice` | 1 | `justice` | 1 |
| glm4 | `hammer` | `knife` | 1 | `knife` | 1 |
| glm4 | `justice` | `time` | 1 | `freedom` | 1 |
| glm4 | `knife` | `hammer` | 1 | `hammer` | 1 |
| glm4 | `oak` | `wheat` | 1 | `stone` | 0 |
| glm4 | `pear` | `apple` | 1 | `banana` | 1 |
| glm4 | `rose` | `bird` | 0 | `stone` | 0 |
| glm4 | `scissors` | `knife` | 1 | `knife` | 1 |
| glm4 | `stone` | `chair` | 1 | `bird` | 0 |
| glm4 | `time` | `justice` | 1 | `chair` | 0 |
| glm4 | `wheat` | `oak` | 1 | `apple` | 0 |
| deepseek7b | `apple` | `pear` | 1 | `banana` | 1 |
| deepseek7b | `banana` | `apple` | 1 | `apple` | 1 |
| deepseek7b | `bird` | `cat` | 1 | `stone` | 0 |
| deepseek7b | `cat` | `bird` | 1 | `dog` | 1 |
| deepseek7b | `chair` | `scissors` | 0 | `cat` | 0 |
| deepseek7b | `cup` | `freedom` | 0 | `cat` | 0 |
| deepseek7b | `dog` | `cat` | 1 | `cat` | 1 |
| deepseek7b | `freedom` | `time` | 1 | `justice` | 1 |
| deepseek7b | `hammer` | `knife` | 1 | `chair` | 0 |
| deepseek7b | `justice` | `time` | 1 | `freedom` | 1 |
| deepseek7b | `knife` | `hammer` | 1 | `scissors` | 1 |
| deepseek7b | `oak` | `rose` | 1 | `wheat` | 1 |
| deepseek7b | `pear` | `apple` | 1 | `scissors` | 0 |
| deepseek7b | `rose` | `oak` | 1 | `oak` | 1 |
| deepseek7b | `scissors` | `knife` | 1 | `knife` | 1 |
| deepseek7b | `stone` | `scissors` | 0 | `bird` | 0 |
| deepseek7b | `time` | `freedom` | 1 | `stone` | 0 |
| deepseek7b | `wheat` | `oak` | 1 | `oak` | 1 |

## Strict Interpretation

- If causal NN domain accuracy exceeds the embedding baseline, this supports a first semantic-numeric interface signal.
- If the signal appears only in one model, it is model-local and not a universal semantic fiber.
- This phase stays at head/source level; it does not claim neuron-level or parameter-level localization.
