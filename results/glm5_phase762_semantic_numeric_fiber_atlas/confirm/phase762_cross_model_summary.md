# Phase 762 Semantic Numeric Fiber Atlas (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Test: convert objects into causal functional fingerprints over object-relation tasks, then test same-domain clustering and compare with first-token embedding baseline.

## Object Fiber Results

| model | tasks | objects | features | interface status | causal NN | embed NN | causal same | causal diff | causal sep | embed sep |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 108 | 18 | 2960 | `weak_causal_fiber_domain_signal` | 0.556 | 0.667 | 0.531 | -0.122 | 0.653 | 0.092 |
| glm4 | 108 | 18 | 3200 | `weak_causal_fiber_domain_signal` | 0.611 | 0.611 | 0.411 | -0.090 | 0.501 | 0.048 |
| deepseek7b | 108 | 18 | 3120 | `weak_causal_fiber_domain_signal` | 0.556 | 0.611 | 0.214 | -0.093 | 0.307 | 0.043 |

## Cross Model Object-Topology Correlation

| pair | common object pairs | centered similarity correlation |
|---|---:|---:|
| `qwen3__glm4` | 153 | 0.344 |
| `qwen3__deepseek7b` | 153 | 0.292 |
| `glm4__deepseek7b` | 153 | 0.287 |

## Nearest Neighbors

| model | object | nearest causal | causal same domain | nearest embedding | embedding same domain |
|---|---|---|---:|---|---:|
| qwen3 | `apple` | `banana` | 1 | `banana` | 1 |
| qwen3 | `banana` | `pear` | 1 | `apple` | 1 |
| qwen3 | `bird` | `cat` | 1 | `dog` | 1 |
| qwen3 | `cat` | `cup` | 0 | `dog` | 1 |
| qwen3 | `chair` | `dog` | 0 | `knife` | 0 |
| qwen3 | `cup` | `cat` | 0 | `apple` | 0 |
| qwen3 | `dog` | `stone` | 0 | `cat` | 1 |
| qwen3 | `freedom` | `time` | 1 | `justice` | 1 |
| qwen3 | `hammer` | `knife` | 1 | `knife` | 1 |
| qwen3 | `justice` | `freedom` | 1 | `freedom` | 1 |
| qwen3 | `knife` | `hammer` | 1 | `scissors` | 1 |
| qwen3 | `oak` | `cat` | 0 | `apple` | 0 |
| qwen3 | `pear` | `banana` | 1 | `apple` | 1 |
| qwen3 | `rose` | `justice` | 0 | `apple` | 0 |
| qwen3 | `scissors` | `knife` | 1 | `knife` | 1 |
| qwen3 | `stone` | `dog` | 0 | `apple` | 0 |
| qwen3 | `time` | `freedom` | 1 | `freedom` | 1 |
| qwen3 | `wheat` | `apple` | 0 | `banana` | 0 |
| glm4 | `apple` | `banana` | 1 | `banana` | 1 |
| glm4 | `banana` | `pear` | 1 | `apple` | 1 |
| glm4 | `bird` | `rose` | 0 | `dog` | 1 |
| glm4 | `cat` | `chair` | 0 | `dog` | 1 |
| glm4 | `chair` | `justice` | 0 | `knife` | 0 |
| glm4 | `cup` | `time` | 0 | `cat` | 0 |
| glm4 | `dog` | `time` | 0 | `cat` | 1 |
| glm4 | `freedom` | `justice` | 1 | `justice` | 1 |
| glm4 | `hammer` | `knife` | 1 | `knife` | 1 |
| glm4 | `justice` | `time` | 1 | `freedom` | 1 |
| glm4 | `knife` | `hammer` | 1 | `hammer` | 1 |
| glm4 | `oak` | `wheat` | 1 | `stone` | 0 |
| glm4 | `pear` | `banana` | 1 | `banana` | 1 |
| glm4 | `rose` | `justice` | 0 | `stone` | 0 |
| glm4 | `scissors` | `knife` | 1 | `knife` | 1 |
| glm4 | `stone` | `rose` | 0 | `bird` | 0 |
| glm4 | `time` | `justice` | 1 | `chair` | 0 |
| glm4 | `wheat` | `oak` | 1 | `apple` | 0 |
| deepseek7b | `apple` | `pear` | 1 | `banana` | 1 |
| deepseek7b | `banana` | `pear` | 1 | `apple` | 1 |
| deepseek7b | `bird` | `chair` | 0 | `stone` | 0 |
| deepseek7b | `cat` | `rose` | 0 | `dog` | 1 |
| deepseek7b | `chair` | `bird` | 0 | `cat` | 0 |
| deepseek7b | `cup` | `stone` | 1 | `cat` | 0 |
| deepseek7b | `dog` | `cat` | 1 | `cat` | 1 |
| deepseek7b | `freedom` | `time` | 1 | `justice` | 1 |
| deepseek7b | `hammer` | `justice` | 0 | `chair` | 0 |
| deepseek7b | `justice` | `hammer` | 0 | `freedom` | 1 |
| deepseek7b | `knife` | `hammer` | 1 | `scissors` | 1 |
| deepseek7b | `oak` | `rose` | 1 | `wheat` | 1 |
| deepseek7b | `pear` | `apple` | 1 | `scissors` | 0 |
| deepseek7b | `rose` | `cat` | 0 | `oak` | 1 |
| deepseek7b | `scissors` | `knife` | 1 | `knife` | 1 |
| deepseek7b | `stone` | `oak` | 0 | `bird` | 0 |
| deepseek7b | `time` | `freedom` | 1 | `stone` | 0 |
| deepseek7b | `wheat` | `stone` | 0 | `oak` | 1 |

## Strict Interpretation

- If causal NN domain accuracy exceeds the embedding baseline, this supports a first semantic-numeric interface signal.
- If the signal appears only in one model, it is model-local and not a universal semantic fiber.
- This phase stays at head/source level; it does not claim neuron-level or parameter-level localization.
