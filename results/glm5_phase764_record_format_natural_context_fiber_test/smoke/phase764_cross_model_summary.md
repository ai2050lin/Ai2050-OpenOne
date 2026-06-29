# Phase 764 Record Format and Natural Context Fiber Test (smoke)

- Status: `complete`
- Test: compare causal fibers across key-value, sentence-line, and compact-sentence contexts.

## Context Domain Separation

| model | context | cases | features | NN | same | diff | sep |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `compact_sentence` | 9 | 64 | 0.000 | null | -0.015 | 0.015 |
| qwen3 | `key_value` | 9 | 64 | 0.000 | null | -0.334 | 0.334 |
| qwen3 | `sentence_lines` | 9 | 32 | 0.000 | null | -1.000 | 1.000 |
| glm4 | `compact_sentence` | 9 | 64 | 0.000 | null | -0.333 | 0.333 |
| glm4 | `key_value` | 9 | 64 | 0.000 | null | -0.379 | 0.379 |
| glm4 | `sentence_lines` | 9 | 32 | 0.000 | null | -1.000 | 1.000 |
| deepseek7b | `compact_sentence` | 9 | 64 | 0.000 | null | -0.265 | 0.265 |
| deepseek7b | `key_value` | 9 | 64 | 0.000 | null | -0.460 | 0.460 |
| deepseek7b | `sentence_lines` | 9 | 32 | 0.000 | null | -1.000 | 1.000 |

## Cross-Context Stability

| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|

## Strict Interpretation

- Strong natural semantic fibers require positive domain separation inside each context and positive same-object stability across contexts.
- If key-value is strong but sentence/compact contexts weaken sharply, the previous signal is likely format-bound.
