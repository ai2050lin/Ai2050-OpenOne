# Phase 671 Graph Atlas Counterfactual Tokenizer Validation

- generated: `2026-06-26 10:22:23`
- status: `pass`

| model | status | cases | pairs | invalid_cases | invalid_pairs | same_prefix_valid/total | max_prompt_tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | pass | 630 | 462 | 0 | 0 | 18/18 | 63 |
| glm4 | pass | 630 | 462 | 0 | 0 | 18/18 | 63 |
| deepseek7b | pass | 630 | 462 | 0 | 0 | 18/18 | 63 |

## Interpretation

- A pass means the prompt-level controls are tokenizer-safe for the current audit.
- Same-prefix continuation controls are strict: they must share the first expected token and diverge later.
- Writer topology and residual-boundary nodes still require later internal activation tests.
