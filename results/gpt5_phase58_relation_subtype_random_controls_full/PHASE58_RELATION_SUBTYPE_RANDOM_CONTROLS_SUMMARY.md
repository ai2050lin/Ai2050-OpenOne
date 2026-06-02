# Phase58 Relation Subtype Random Controls Summary

## qwen3

attn_implementations: `flash_attention_2,sdpa,eager`

### Ranked By Random Advantage

| rank | subtype | net/gross | random | advantage | interaction | n |
|---:|---|---:|---:|---:|---:|---:|
| 1 | temporal_order/before_after | 0.0446 | 0.0180 | 0.0265 | 0.1849 | 150 |
| 2 | same_class/category_peer | 0.0372 | 0.0214 | 0.0158 | 0.3365 | 150 |
| 3 | contrast/but_and | 0.0267 | 0.0181 | 0.0087 | 0.1826 | 150 |
| 4 | binding/color | 0.0294 | 0.0216 | 0.0078 | 0.3672 | 150 |
| 5 | negation/syntactic_not | 0.0276 | 0.0200 | 0.0076 | 0.3726 | 150 |
| 6 | role/active_swap | 0.0264 | 0.0193 | 0.0071 | 0.2500 | 150 |
| 7 | negation/quantifier_no | 0.0226 | 0.0182 | 0.0045 | 0.2814 | 150 |
| 8 | binding/taste | 0.0265 | 0.0221 | 0.0044 | 0.4537 | 150 |
| 9 | condition/if_unless | 0.0228 | 0.0187 | 0.0041 | 0.2164 | 150 |
| 10 | binding/temperature | 0.0248 | 0.0222 | 0.0026 | 0.4746 | 150 |
| 11 | binding/texture | 0.0232 | 0.0207 | 0.0025 | 0.3660 | 150 |
| 12 | quantifier/all_some | 0.0191 | 0.0184 | 0.0007 | 0.3037 | 150 |
| 13 | comparison/greater_less | 0.0152 | 0.0175 | -0.0022 | 0.2150 | 150 |

### Top Similarity Pairs

| subtype_a | subtype_b | cosine |
|---|---|---:|
| binding/color | binding/texture | 0.9992 |
| condition/if_unless | comparison/greater_less | 0.9982 |
| role/active_swap | comparison/greater_less | 0.9982 |
| binding/texture | same_class/category_peer | 0.9970 |
| condition/if_unless | contrast/but_and | 0.9966 |
| temporal_order/before_after | comparison/greater_less | 0.9963 |
| role/active_swap | condition/if_unless | 0.9962 |
| temporal_order/before_after | condition/if_unless | 0.9961 |

### Bottom Similarity Pairs

| subtype_a | subtype_b | cosine |
|---|---|---:|
| binding/temperature | contrast/but_and | 0.9571 |
| binding/temperature | temporal_order/before_after | 0.9612 |
| binding/taste | contrast/but_and | 0.9655 |
| binding/taste | temporal_order/before_after | 0.9669 |
| binding/temperature | comparison/greater_less | 0.9694 |
| binding/temperature | condition/if_unless | 0.9702 |
| binding/temperature | role/active_swap | 0.9736 |
| binding/taste | comparison/greater_less | 0.9741 |

## glm4

attn_implementations: `flash_attention_2,sdpa,eager`

### Ranked By Random Advantage

| rank | subtype | net/gross | random | advantage | interaction | n |
|---:|---|---:|---:|---:|---:|---:|
| 1 | negation/quantifier_no | 0.0525 | 0.0181 | 0.0344 | 0.2218 | 120 |
| 2 | quantifier/all_some | 0.0456 | 0.0180 | 0.0277 | 0.1761 | 120 |
| 3 | negation/syntactic_not | 0.0428 | 0.0194 | 0.0234 | 0.3321 | 120 |
| 4 | condition/if_unless | 0.0413 | 0.0195 | 0.0218 | 0.1638 | 120 |
| 5 | contrast/but_and | 0.0341 | 0.0191 | 0.0150 | 0.1921 | 120 |
| 6 | same_class/category_peer | 0.0310 | 0.0171 | 0.0139 | 0.3143 | 120 |
| 7 | binding/taste | 0.0282 | 0.0182 | 0.0100 | 0.3196 | 120 |
| 8 | role/active_swap | 0.0257 | 0.0176 | 0.0081 | 0.2668 | 120 |
| 9 | comparison/greater_less | 0.0221 | 0.0170 | 0.0051 | 0.1196 | 120 |
| 10 | binding/color | 0.0207 | 0.0186 | 0.0022 | 0.4133 | 120 |
| 11 | temporal_order/before_after | 0.0195 | 0.0179 | 0.0016 | 0.1854 | 120 |
| 12 | binding/texture | 0.0177 | 0.0177 | 0.0000 | 0.3793 | 120 |
| 13 | binding/temperature | 0.0180 | 0.0187 | -0.0006 | 0.3290 | 120 |

### Top Similarity Pairs

| subtype_a | subtype_b | cosine |
|---|---|---:|
| binding/taste | same_class/category_peer | 0.9988 |
| binding/temperature | binding/texture | 0.9982 |
| binding/color | binding/texture | 0.9979 |
| condition/if_unless | comparison/greater_less | 0.9979 |
| condition/if_unless | contrast/but_and | 0.9978 |
| role/active_swap | same_class/category_peer | 0.9977 |
| temporal_order/before_after | comparison/greater_less | 0.9975 |
| temporal_order/before_after | condition/if_unless | 0.9975 |

### Bottom Similarity Pairs

| subtype_a | subtype_b | cosine |
|---|---|---:|
| binding/color | comparison/greater_less | 0.9614 |
| binding/texture | comparison/greater_less | 0.9672 |
| binding/color | condition/if_unless | 0.9714 |
| binding/color | quantifier/all_some | 0.9715 |
| negation/syntactic_not | comparison/greater_less | 0.9749 |
| binding/texture | quantifier/all_some | 0.9750 |
| binding/color | temporal_order/before_after | 0.9755 |
| binding/texture | condition/if_unless | 0.9761 |

## deepseek7b

attn_implementations: `eager`

### Ranked By Random Advantage

| rank | subtype | net/gross | random | advantage | interaction | n |
|---:|---|---:|---:|---:|---:|---:|
| 1 | contrast/but_and | 0.0290 | 0.0148 | 0.0142 | 0.1906 | 120 |
| 2 | temporal_order/before_after | 0.0270 | 0.0140 | 0.0130 | 0.2368 | 120 |
| 3 | quantifier/all_some | 0.0269 | 0.0139 | 0.0129 | 0.3279 | 120 |
| 4 | condition/if_unless | 0.0251 | 0.0147 | 0.0104 | 0.2970 | 120 |
| 5 | negation/syntactic_not | 0.0240 | 0.0150 | 0.0090 | 0.4202 | 120 |
| 6 | binding/texture | 0.0225 | 0.0161 | 0.0065 | 0.3565 | 120 |
| 7 | same_class/category_peer | 0.0234 | 0.0170 | 0.0065 | 0.3044 | 120 |
| 8 | binding/color | 0.0178 | 0.0151 | 0.0027 | 0.4171 | 120 |
| 9 | negation/quantifier_no | 0.0163 | 0.0138 | 0.0025 | 0.3284 | 120 |
| 10 | binding/taste | 0.0167 | 0.0150 | 0.0016 | 0.6137 | 120 |
| 11 | comparison/greater_less | 0.0149 | 0.0136 | 0.0014 | 0.2059 | 120 |
| 12 | role/active_swap | 0.0163 | 0.0156 | 0.0007 | 0.3443 | 120 |
| 13 | binding/temperature | 0.0149 | 0.0154 | -0.0005 | 0.4124 | 120 |

### Top Similarity Pairs

| subtype_a | subtype_b | cosine |
|---|---|---:|
| binding/texture | role/active_swap | 0.9995 |
| binding/texture | quantifier/all_some | 0.9990 |
| binding/color | binding/temperature | 0.9990 |
| temporal_order/before_after | comparison/greater_less | 0.9989 |
| role/active_swap | quantifier/all_some | 0.9982 |
| binding/texture | negation/syntactic_not | 0.9981 |
| binding/color | negation/syntactic_not | 0.9981 |
| quantifier/all_some | condition/if_unless | 0.9980 |

### Bottom Similarity Pairs

| subtype_a | subtype_b | cosine |
|---|---|---:|
| binding/taste | contrast/but_and | 0.9269 |
| binding/taste | comparison/greater_less | 0.9386 |
| binding/taste | temporal_order/before_after | 0.9475 |
| binding/taste | condition/if_unless | 0.9633 |
| binding/taste | same_class/category_peer | 0.9652 |
| binding/taste | negation/quantifier_no | 0.9668 |
| binding/taste | quantifier/all_some | 0.9699 |
| binding/temperature | contrast/but_and | 0.9732 |
