# Phase 765 Commonsense Context and Object Identity Closure Test (smoke)

- Status: `complete`
- Test: no explicit fact profile; use commonsense prompts only.

## Base Answer Reliability

| model | top1 rate | mean target rank | mean contrast rank |
|---|---:|---:|---:|
| qwen3 | 0.833 | 1.500 | 80.083 |
| glm4 | 0.500 | 1.667 | 5.000 |
| deepseek7b | 0.333 | 5.417 | 46.333 |

## Commonsense Domain Separation

| model | context | cases | features | NN | same | diff | sep |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `commonsense_question` | 12 | 96 | 0.000 | -0.106 | 0.126 | -0.232 |
| qwen3 | `commonsense_statement` | 12 | 96 | 0.000 | -0.633 | -0.057 | -0.576 |
| glm4 | `commonsense_question` | 12 | 96 | 0.000 | -0.621 | -0.070 | -0.550 |
| glm4 | `commonsense_statement` | 12 | 96 | 0.000 | -0.727 | -0.041 | -0.686 |
| deepseek7b | `commonsense_question` | 12 | 96 | 0.000 | -0.606 | -0.084 | -0.522 |
| deepseek7b | `commonsense_statement` | 12 | 96 | 0.000 | -0.649 | -0.085 | -0.564 |

## Cross-Context Stability

| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|

## Strict Interpretation

- If target-top1 is low, the commonsense prompt does not form a reliable target state and causal fibers are hard to interpret.
- A natural semantic closure result requires positive domain separation and positive same-object stability across commonsense prompt formats.
