# Phase57 Global Relation Path Matrix Summary

## qwen3

attn_implementations: `flash_attention_2,sdpa,eager`

| rank | relation | net/gross | balance | interaction | n |
|---:|---|---:|---:|---:|---:|
| 1 | temporal_order | 0.0451 | 0.9953 | 0.1758 | 180 |
| 2 | same_class | 0.0378 | 0.9947 | 0.3212 | 200 |
| 3 | role | 0.0315 | 1.0212 | 0.2745 | 200 |
| 4 | causal | 0.0293 | 1.0000 | 0.2388 | 200 |
| 5 | binding | 0.0278 | 1.0111 | 0.4260 | 200 |
| 6 | negation | 0.0278 | 1.0554 | 0.3364 | 200 |
| 7 | antonym | 0.0268 | 0.9990 | 0.3465 | 200 |
| 8 | contrast | 0.0249 | 1.0034 | 0.2002 | 200 |
| 9 | condition | 0.0245 | 1.0020 | 0.2036 | 200 |
| 10 | tense | 0.0238 | 0.9951 | 0.3255 | 160 |
| 11 | coreference | 0.0222 | 1.0018 | 0.3526 | 200 |
| 12 | quantifier | 0.0213 | 0.9871 | 0.3417 | 200 |
| 13 | comparison | 0.0195 | 0.9987 | 0.1899 | 200 |
| 14 | spatial | 0.0191 | 0.9871 | 0.2276 | 180 |

### Top Similarity Pairs

| relation_a | relation_b | cosine |
|---|---|---:|
| condition | contrast | 0.9945 |
| comparison | contrast | 0.9938 |
| spatial | contrast | 0.9895 |
| comparison | spatial | 0.9893 |
| comparison | condition | 0.9892 |
| spatial | condition | 0.9880 |
| antonym | role | 0.9816 |
| tense | coreference | 0.9756 |

### Bottom Similarity Pairs

| relation_a | relation_b | cosine |
|---|---|---:|
| negation | comparison | 0.5666 |
| negation | contrast | 0.5974 |
| negation | spatial | 0.6183 |
| negation | condition | 0.6225 |
| binding | comparison | 0.6382 |
| binding | contrast | 0.6663 |
| same_class | comparison | 0.6787 |
| binding | spatial | 0.6947 |

## glm4

attn_implementations: `flash_attention_2,sdpa,eager`

| rank | relation | net/gross | balance | interaction | n |
|---:|---|---:|---:|---:|---:|
| 1 | quantifier | 0.0442 | 0.9611 | 0.1602 | 160 |
| 2 | condition | 0.0392 | 1.0304 | 0.1195 | 160 |
| 3 | contrast | 0.0364 | 1.0082 | 0.1566 | 160 |
| 4 | negation | 0.0329 | 0.9879 | 0.3308 | 160 |
| 5 | same_class | 0.0316 | 0.9937 | 0.2783 | 160 |
| 6 | antonym | 0.0281 | 1.0080 | 0.3366 | 160 |
| 7 | role | 0.0266 | 0.9978 | 0.2811 | 160 |
| 8 | tense | 0.0265 | 0.9857 | 0.1667 | 128 |
| 9 | spatial | 0.0253 | 0.9982 | 0.1023 | 144 |
| 10 | coreference | 0.0243 | 1.0057 | 0.2348 | 160 |
| 11 | binding | 0.0228 | 0.9959 | 0.3757 | 160 |
| 12 | causal | 0.0225 | 0.9779 | 0.2462 | 160 |
| 13 | temporal_order | 0.0214 | 1.0162 | 0.1240 | 144 |
| 14 | comparison | 0.0201 | 1.0015 | 0.0824 | 160 |

### Top Similarity Pairs

| relation_a | relation_b | cosine |
|---|---|---:|
| tense | condition | 0.9962 |
| comparison | temporal_order | 0.9953 |
| coreference | causal | 0.9948 |
| tense | contrast | 0.9924 |
| tense | spatial | 0.9922 |
| spatial | contrast | 0.9921 |
| coreference | temporal_order | 0.9905 |
| role | tense | 0.9900 |

### Bottom Similarity Pairs

| relation_a | relation_b | cosine |
|---|---|---:|
| antonym | comparison | 0.8162 |
| negation | comparison | 0.8206 |
| same_class | comparison | 0.8342 |
| binding | comparison | 0.8468 |
| quantifier | comparison | 0.8506 |
| negation | temporal_order | 0.8516 |
| antonym | temporal_order | 0.8517 |
| same_class | temporal_order | 0.8676 |

## deepseek7b

attn_implementations: `eager`

| rank | relation | net/gross | balance | interaction | n |
|---:|---|---:|---:|---:|---:|
| 1 | temporal_order | 0.0260 | 0.9900 | 0.2246 | 144 |
| 2 | contrast | 0.0259 | 1.0302 | 0.1676 | 160 |
| 3 | same_class | 0.0243 | 0.9952 | 0.3267 | 160 |
| 4 | condition | 0.0243 | 0.9945 | 0.2685 | 160 |
| 5 | spatial | 0.0221 | 1.0061 | 0.2412 | 144 |
| 6 | quantifier | 0.0214 | 1.0057 | 0.3406 | 160 |
| 7 | antonym | 0.0203 | 1.0026 | 0.4522 | 160 |
| 8 | negation | 0.0203 | 1.0085 | 0.4657 | 160 |
| 9 | coreference | 0.0185 | 0.9843 | 0.3111 | 160 |
| 10 | role | 0.0184 | 0.9914 | 0.3298 | 160 |
| 11 | binding | 0.0179 | 0.9985 | 0.4586 | 160 |
| 12 | comparison | 0.0163 | 0.9930 | 0.1808 | 160 |
| 13 | tense | 0.0162 | 1.0024 | 0.2680 | 128 |
| 14 | causal | 0.0162 | 1.0088 | 0.2792 | 160 |

### Top Similarity Pairs

| relation_a | relation_b | cosine |
|---|---|---:|
| binding | antonym | 0.9894 |
| role | condition | 0.9783 |
| tense | condition | 0.9760 |
| tense | causal | 0.9758 |
| causal | contrast | 0.9710 |
| binding | negation | 0.9707 |
| comparison | contrast | 0.9704 |
| spatial | condition | 0.9667 |

### Bottom Similarity Pairs

| relation_a | relation_b | cosine |
|---|---|---:|
| negation | comparison | 0.6255 |
| negation | coreference | 0.7140 |
| binding | comparison | 0.7156 |
| negation | contrast | 0.7190 |
| antonym | comparison | 0.7198 |
| same_class | comparison | 0.7589 |
| coreference | temporal_order | 0.7647 |
| negation | causal | 0.7678 |
