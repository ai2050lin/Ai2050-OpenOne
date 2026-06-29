# Phase 765 Commonsense Context and Object Identity Closure Test (main)

- Status: `complete`
- Test: no explicit fact profile; use commonsense prompts only.

## Base Answer Reliability

| model | top1 rate | mean target rank | mean contrast rank |
|---|---:|---:|---:|
| qwen3 | 0.806 | 1.324 | 51.741 |
| glm4 | 0.593 | 1.657 | 5.046 |
| deepseek7b | 0.185 | 5.833 | 66.102 |

## Commonsense Domain Separation

| model | context | cases | features | NN | same | diff | sep |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `commonsense_question` | 108 | 96 | 0.444 | 0.814 | 0.207 | 0.607 |
| qwen3 | `commonsense_statement` | 108 | 96 | 0.611 | 0.846 | 0.233 | 0.613 |
| glm4 | `commonsense_question` | 108 | 96 | 0.222 | 0.182 | -0.052 | 0.234 |
| glm4 | `commonsense_statement` | 108 | 96 | 0.222 | 0.092 | -0.070 | 0.162 |
| deepseek7b | `commonsense_question` | 108 | 96 | 0.778 | 0.272 | -0.098 | 0.370 |
| deepseek7b | `commonsense_statement` | 108 | 96 | 0.667 | 0.332 | -0.104 | 0.436 |

## Cross-Context Stability

| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `commonsense_question__commonsense_statement` | 0.894 | 0.803 | 0.212 | 0.091 | 0.591 |
| glm4 | `commonsense_question__commonsense_statement` | -0.277 | -0.403 | -0.436 | 0.126 | 0.033 |
| deepseek7b | `commonsense_question__commonsense_statement` | -0.119 | -0.188 | -0.322 | 0.069 | 0.134 |

## Strict Interpretation

- If target-top1 is low, the commonsense prompt does not form a reliable target state and causal fibers are hard to interpret.
- A natural semantic closure result requires positive domain separation and positive same-object stability across commonsense prompt formats.
