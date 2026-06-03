# Phase56 Global Path + MLP Interaction Summary

## Relation Path Comparison

### qwen3

| relation | balance | net/gross | n |
|---|---:|---:|---:|
| binding | 1.0169 | 0.0279 | 45 |
| negation | 1.0532 | 0.0313 | 25 |
| antonym | 0.9984 | 0.0228 | 25 |
| role | 1.0160 | 0.0281 | 20 |
| tense | 0.9928 | 0.0168 | 15 |
| same_class | 1.0066 | 0.0331 | 25 |

binding_net_gross_rank_among_relations=4

## MLP Gate/Up/Interaction

qwen3: gate=0.2570, up=0.3157, interaction=0.4273, total_effect=0.5644
closure: ratio=1.1237, final=2.4064, mlp_sum=3.5695

### glm4

| relation | balance | net/gross | n |
|---|---:|---:|---:|
| binding | 0.9969 | 0.0200 | 36 |
| negation | 0.9910 | 0.0369 | 20 |
| antonym | 0.9947 | 0.0244 | 20 |
| role | 1.0028 | 0.0318 | 16 |
| tense | 0.9799 | 0.0170 | 12 |
| same_class | 0.9941 | 0.0317 | 20 |

binding_net_gross_rank_among_relations=5

## MLP Gate/Up/Interaction

glm4: gate=0.3034, up=0.3060, interaction=0.3906, total_effect=0.3340
closure: ratio=0.4228, final=2.9638, mlp_sum=1.2823

### deepseek7b

| relation | balance | net/gross | n |
|---|---:|---:|---:|
| binding | 0.9957 | 0.0214 | 32 |
| negation | 1.0133 | 0.0245 | 20 |
| antonym | 0.9921 | 0.0173 | 20 |
| role | 0.9929 | 0.0138 | 16 |
| tense | 0.9927 | 0.0155 | 12 |
| same_class | 0.9995 | 0.0198 | 20 |

binding_net_gross_rank_among_relations=2

## MLP Gate/Up/Interaction

deepseek7b: gate=0.2803, up=0.2635, interaction=0.4562, total_effect=1.0111
closure: ratio=-11.4383, final=1.2340, mlp_sum=4.2638
