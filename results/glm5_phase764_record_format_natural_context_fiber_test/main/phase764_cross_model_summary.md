# Phase 764 Record Format and Natural Context Fiber Test (main)

- Status: `complete`
- Test: compare causal fibers across key-value, sentence-line, and compact-sentence contexts.

## Context Domain Separation

| model | context | cases | features | NN | same | diff | sep |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `compact_sentence` | 162 | 96 | 0.556 | 0.885 | 0.194 | 0.691 |
| qwen3 | `key_value` | 162 | 96 | 0.722 | 0.848 | 0.153 | 0.695 |
| qwen3 | `sentence_lines` | 162 | 96 | 0.833 | 0.852 | 0.162 | 0.690 |
| glm4 | `compact_sentence` | 162 | 96 | 0.556 | 0.346 | -0.106 | 0.452 |
| glm4 | `key_value` | 162 | 96 | 0.500 | 0.283 | -0.093 | 0.376 |
| glm4 | `sentence_lines` | 162 | 96 | 0.556 | 0.370 | -0.099 | 0.469 |
| deepseek7b | `compact_sentence` | 162 | 96 | 0.500 | 0.353 | -0.103 | 0.457 |
| deepseek7b | `key_value` | 162 | 96 | 0.611 | 0.336 | -0.078 | 0.414 |
| deepseek7b | `sentence_lines` | 162 | 96 | 0.722 | 0.417 | -0.106 | 0.522 |

## Cross-Context Stability

| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `compact_sentence__key_value` | 0.891 | 0.816 | 0.147 | 0.074 | 0.670 |
| qwen3 | `compact_sentence__sentence_lines` | 0.791 | 0.732 | 0.094 | 0.060 | 0.637 |
| qwen3 | `key_value__sentence_lines` | 0.897 | 0.810 | 0.138 | 0.088 | 0.671 |
| glm4 | `compact_sentence__key_value` | 0.285 | 0.039 | -0.180 | 0.247 | 0.219 |
| glm4 | `compact_sentence__sentence_lines` | -0.085 | -0.204 | -0.329 | 0.119 | 0.126 |
| glm4 | `key_value__sentence_lines` | 0.072 | -0.114 | -0.323 | 0.186 | 0.209 |
| deepseek7b | `compact_sentence__key_value` | 0.088 | -0.044 | -0.186 | 0.132 | 0.141 |
| deepseek7b | `compact_sentence__sentence_lines` | 0.196 | 0.098 | -0.158 | 0.097 | 0.257 |
| deepseek7b | `key_value__sentence_lines` | 0.107 | -0.024 | -0.191 | 0.131 | 0.167 |

## Strict Interpretation

- Strong natural semantic fibers require positive domain separation inside each context and positive same-object stability across contexts.
- If key-value is strong but sentence/compact contexts weaken sharply, the previous signal is likely format-bound.
