# Phase 567: Step0 Logit-Field Source Decomposition Cross-Model Summary

Surgery: module-level donor injection at step 0 (layer/attn/mlp), then free generation

## free clean_non_object_rate

| model | baseline | r4_all | r4_attn | r4_mlp | r4_a+m | rand_all | rand_attn | rand_mlp |
|---|---|---|---|---|---|---|---|---|
| qwen3 |    0.64 |    0.48 |    0.66 |    0.68 |    0.56 |    0.56 |    0.64 |    0.55 |
| glm4 |    0.36 |    0.44 |    0.41 |    0.36 |    0.37 |    0.33 |    0.34 |    0.38 |
| deepseek7b |    0.15 |    0.22 |    0.14 |    0.17 |    0.19 |    0.15 |    0.16 |    0.15 |

## step0 target-competitor margin

| model | baseline | r4_all | r4_attn | r4_mlp | r4_a+m | rand_all | rand_attn | rand_mlp |
|---|---|---|---|---|---|---|---|---|
| qwen3 |   +0.56 |   +0.90 |   +0.72 |   +0.56 |   +0.77 |   +2.30 |   +0.55 |   +0.60 |
| glm4 |   -1.03 |   +0.40 |   -0.54 |   -0.78 |   -0.35 |   -0.01 |   -0.92 |   -1.52 |
| deepseek7b |   +0.96 |   -0.49 |   +1.33 |   +0.06 |   +0.27 |   -0.36 |   +1.10 |   +1.30 |

## step0 target rank

| model | baseline | r4_all | r4_attn | r4_mlp | r4_a+m | rand_all | rand_attn | rand_mlp |
|---|---|---|---|---|---|---|---|---|
| qwen3 |     374 |     195 |     334 |     316 |     259 |      74 |     383 |     344 |
| glm4 |     522 |    6537 |     405 |    1149 |     945 |   16714 |     550 |     945 |
| deepseek7b |     203 |     128 |     134 |     252 |     182 |   10718 |     147 |     155 |

## bfi_prefix2 clean_non_object_rate (baseline forced to intervention's first 2 tokens)

| model | baseline | r4_all | r4_attn | r4_mlp | r4_a+m | rand_all | rand_attn | rand_mlp |
|---|---|---|---|---|---|---|---|---|
| qwen3 |         |    0.46 |    0.58 |    0.56 |    0.58 |    0.58 |    0.60 |    0.54 |
| glm4 |         |    0.38 |    0.35 |    0.32 |    0.32 |    0.33 |    0.33 |    0.35 |
| deepseek7b |         |    0.23 |    0.22 |    0.23 |    0.22 |    0.12 |    0.27 |    0.29 |

## Semantic specificity (repeat4_all - random_all)

| model | free_clean | s0_margin | bfi_p2 |
|---|---|---|---|
| qwen3 |   -0.07 |   -1.40 |   -0.12 |
| glm4 |   +0.11 |   +0.41 |   +0.05 |
| deepseek7b |   +0.07 |   -0.14 |   +0.11 |

## Timing

| model | time (min) | test_n | seeds |
|---|---|---|---|
| qwen3 | 1.06 | 18 | 6 |
| glm4 | 19.63 | 24 | 8 |
| deepseek7b | 6.37 | 18 | 6 |

## Key Findings

### GLM4 (core model, test_n=24, 8 seeds):

| condition | free_clean | s0_margin | bfi_p2 |
|---|---|---|---|
| baseline | 0.36 | -1.03 | N/A |
| r4_all | 0.44 | +0.40 | 0.38 |
| r4_attn | 0.41 | -0.54 | 0.35 |
| r4_mlp | 0.36 | -0.78 | 0.32 |
| r4_a+m | 0.37 | -0.35 | 0.32 |
| rand_all | 0.33 | -0.01 | 0.33 |
| rand_attn | 0.34 | -0.92 | 0.33 |
| rand_mlp | 0.38 | -1.52 | 0.35 |

**Critical:** Only `repeat4_all` (full layer restore) produces step0 margin flip (+0.40).
Individual module injections (attn_only, mlp_only, attn_mlp) do NOT flip the margin.
This suggests the logit-field flip requires the complete residual state, not just module deltas.