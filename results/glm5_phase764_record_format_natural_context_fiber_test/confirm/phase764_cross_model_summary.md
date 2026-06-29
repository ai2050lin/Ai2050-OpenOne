# Phase 764 Record Format and Natural Context Fiber Test (confirm)

- Status: `complete`
- Test: compare causal fibers across key-value, sentence-line, and compact-sentence contexts.

## Context Domain Separation

| model | context | cases | features | NN | same | diff | sep |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `compact_sentence` | 162 | 144 | 0.611 | 0.863 | 0.182 | 0.681 |
| qwen3 | `key_value` | 162 | 144 | 0.722 | 0.833 | 0.146 | 0.687 |
| qwen3 | `sentence_lines` | 162 | 144 | 0.833 | 0.839 | 0.154 | 0.684 |
| glm4 | `compact_sentence` | 162 | 144 | 0.667 | 0.320 | -0.104 | 0.425 |
| glm4 | `key_value` | 162 | 144 | 0.500 | 0.243 | -0.087 | 0.331 |
| glm4 | `sentence_lines` | 162 | 144 | 0.556 | 0.354 | -0.099 | 0.453 |
| deepseek7b | `compact_sentence` | 162 | 144 | 0.667 | 0.401 | -0.114 | 0.515 |
| deepseek7b | `key_value` | 162 | 144 | 0.778 | 0.364 | -0.100 | 0.464 |
| deepseek7b | `sentence_lines` | 162 | 144 | 0.889 | 0.478 | -0.120 | 0.597 |

## Cross-Context Stability

| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `compact_sentence__key_value` | 0.874 | 0.798 | 0.137 | 0.076 | 0.661 |
| qwen3 | `compact_sentence__sentence_lines` | 0.777 | 0.718 | 0.087 | 0.059 | 0.631 |
| qwen3 | `key_value__sentence_lines` | 0.882 | 0.797 | 0.132 | 0.085 | 0.665 |
| glm4 | `compact_sentence__key_value` | 0.240 | 0.002 | -0.194 | 0.238 | 0.196 |
| glm4 | `compact_sentence__sentence_lines` | -0.066 | -0.176 | -0.301 | 0.111 | 0.125 |
| glm4 | `key_value__sentence_lines` | 0.081 | -0.110 | -0.299 | 0.191 | 0.189 |
| deepseek7b | `compact_sentence__key_value` | 0.165 | 0.045 | -0.201 | 0.120 | 0.246 |
| deepseek7b | `compact_sentence__sentence_lines` | 0.248 | 0.185 | -0.171 | 0.062 | 0.356 |
| deepseek7b | `key_value__sentence_lines` | 0.101 | -0.000 | -0.247 | 0.101 | 0.246 |

## Strict Interpretation

- Strong natural semantic fibers require positive domain separation inside each context and positive same-object stability across contexts.
- If key-value is strong but sentence/compact contexts weaken sharply, the previous signal is likely format-bound.
