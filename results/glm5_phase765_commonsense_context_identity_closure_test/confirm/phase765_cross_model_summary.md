# Phase 765 Commonsense Context and Object Identity Closure Test (confirm)

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
| qwen3 | `commonsense_question` | 108 | 120 | 0.556 | 0.801 | 0.198 | 0.603 |
| qwen3 | `commonsense_statement` | 108 | 120 | 0.611 | 0.832 | 0.226 | 0.606 |
| glm4 | `commonsense_question` | 108 | 120 | 0.222 | 0.154 | -0.053 | 0.208 |
| glm4 | `commonsense_statement` | 108 | 120 | 0.278 | 0.087 | -0.070 | 0.156 |
| deepseek7b | `commonsense_question` | 108 | 120 | 0.667 | 0.258 | -0.097 | 0.355 |
| deepseek7b | `commonsense_statement` | 108 | 120 | 0.667 | 0.301 | -0.100 | 0.401 |

## Cross-Context Stability

| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `commonsense_question__commonsense_statement` | 0.877 | 0.789 | 0.203 | 0.088 | 0.585 |
| glm4 | `commonsense_question__commonsense_statement` | -0.261 | -0.385 | -0.422 | 0.124 | 0.037 |
| deepseek7b | `commonsense_question__commonsense_statement` | -0.115 | -0.186 | -0.312 | 0.071 | 0.126 |

## Strict Interpretation

- If target-top1 is low, the commonsense prompt does not form a reliable target state and causal fibers are hard to interpret.
- A natural semantic closure result requires positive domain separation and positive same-object stability across commonsense prompt formats.
