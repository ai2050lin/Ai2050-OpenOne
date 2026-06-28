# Phase 721 Global Functional Head Atlas Data Expansion

## Status

- Models complete: `['qwen3', 'glm4', 'deepseek7b']`
- Status: `complete`
- Evidence type: observational answer-last attention mass, not causal patch.

## Top Source-Focus Heads By Model

### qwen3

| family | head | score | target_value | object | relation | instruction | top_tokens |
|---|---:|---:|---:|---:|---:|---:|---|
| simple_grammar_protocol_route | L28H0 | 1.1737 | 0.8258 | 0.6809 | 0.0176 | 0.0023 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L20H15 | 1.0790 | 0.7212 | 0.5960 | 0.0708 | 0.0051 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L23H4 | 0.9848 | 0.6763 | 0.5809 | 0.0182 | 0.0010 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| fruit_identity_reuse_difference | L26H26 | 0.9459 | 0.9305 | 0.0240 | 0.0070 | 0.0001 | {' animal': 6, ' fruit': 12, ' tool': 6} |
| translation_language_route | L28H0 | 0.9079 | 0.8971 | 0.0124 | 0.0000 | 0.0005 | {' rouge': 1, 'ana': 1, 'ane': 1, 'ano': 1, 'jo': 1, 'me': 1, '果': 1, '色': 2, '蕉': 1, '�': 2} |
| simple_grammar_protocol_route | L21H23 | 0.8925 | 0.5848 | 0.6017 | 0.0155 | 0.0032 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 2} |
| fruit_identity_reuse_difference | L28H0 | 0.8492 | 0.8366 | 0.0058 | 0.0250 | 0.0043 | {' animal': 6, ' fruit': 12, ' tool': 6} |
| translation_language_route | L24H29 | 0.8462 | 0.8298 | 0.0056 | 0.0000 | 0.0000 | {' rouge': 1, ' �': 2, 'ana': 1, 'ane': 1, 'ano': 1, 'jo': 1, 'me': 1, 'u': 1, '果': 1, '色': 2} |
| fruit_identity_reuse_difference | L24H29 | 0.8284 | 0.8062 | 0.0063 | 0.0383 | 0.0000 | {' animal': 6, ' fruit': 12, ' tool': 6} |
| simple_grammar_protocol_route | L24H29 | 0.8178 | 0.5329 | 0.4515 | 0.0835 | 0.0008 | {' apples': 1, ' boxes': 1, ' called': 1, ' dishes': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 2} |
| simple_grammar_protocol_route | L23H30 | 0.8084 | 0.4367 | 0.4073 | 0.2495 | 0.0066 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| fruit_identity_reuse_difference | L29H11 | 0.8026 | 0.8011 | 0.0032 | 0.0070 | 0.0066 | {' animal': 6, ' fruit': 12, ' tool': 6} |

### glm4

| family | head | score | target_value | object | relation | instruction | top_tokens |
|---|---:|---:|---:|---:|---:|---:|---|
| simple_grammar_protocol_route | L29H26 | 1.2467 | 0.8817 | 0.7213 | 0.0069 | 0.0002 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L23H10 | 1.2119 | 0.8579 | 0.7082 | 0.0044 | 0.0009 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L29H18 | 1.1837 | 0.8136 | 0.7327 | 0.0058 | 0.0005 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L23H13 | 1.1657 | 0.8226 | 0.6846 | 0.0106 | 0.0074 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L26H21 | 1.1633 | 0.8000 | 0.7140 | 0.0128 | 0.0015 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L24H24 | 1.1404 | 0.7534 | 0.7053 | 0.0348 | 0.0022 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L21H15 | 1.1249 | 0.7976 | 0.6486 | 0.0096 | 0.0011 | {' apples': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, '.\n': 2} |
| simple_grammar_protocol_route | L23H0 | 1.1172 | 0.7827 | 0.6557 | 0.0159 | 0.0012 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L29H28 | 1.0120 | 0.6497 | 0.7165 | 0.0092 | 0.0029 | {' apples': 1, ' called': 1, ' cars': 1, ' fast': 2, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 2} |
| simple_grammar_protocol_route | L32H28 | 1.0000 | 0.6542 | 0.5899 | 0.0786 | 0.0058 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L25H2 | 0.9951 | 0.7042 | 0.5782 | 0.0287 | 0.0215 | {' apples': 1, ' books': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 1} |
| simple_grammar_protocol_route | L32H13 | 0.9808 | 0.5858 | 0.7844 | 0.0106 | 0.0056 | {' apple': 1, ' book': 1, ' box': 1, ' call': 1, ' car': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' play': 1, ' walked': 1} |

### deepseek7b

| family | head | score | target_value | object | relation | instruction | top_tokens |
|---|---:|---:|---:|---:|---:|---:|---|
| simple_grammar_protocol_route | L22H1 | 0.9293 | 0.6546 | 0.5469 | 0.0129 | 0.0173 | {' apples': 1, ' called': 1, ' cars': 1, ' dishes': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 3} |
| simple_grammar_protocol_route | L21H25 | 0.8974 | 0.6091 | 0.5159 | 0.0799 | 0.0343 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L22H24 | 0.8706 | 0.6168 | 0.4901 | 0.0156 | 0.0132 | {' apples': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, '.\n': 1, 'Record': 3} |
| simple_grammar_protocol_route | L23H0 | 0.7274 | 0.4628 | 0.5170 | 0.0204 | 0.0315 | {' apples': 1, ' called': 1, ' cars': 1, ' dishes': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 7} |
| simple_grammar_protocol_route | L25H25 | 0.7215 | 0.5440 | 0.4791 | 0.0043 | 0.0906 | {' apples': 1, ' books': 1, ' boxes': 1, ' called': 1, ' cars': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1} |
| simple_grammar_protocol_route | L27H6 | 0.6665 | 0.4876 | 0.4002 | 0.0141 | 0.0518 | {' called': 1, ' faster': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' smaller': 1, ' walked': 1, '.\n': 1, 'Record': 6} |
| simple_grammar_protocol_route | L23H11 | 0.6412 | 0.4595 | 0.4062 | 0.0117 | 0.0475 | {' apples': 1, ' called': 1, ' cars': 1, ' faster': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, 'Record': 8} |
| simple_grammar_protocol_route | L19H17 | 0.5620 | 0.3591 | 0.3382 | 0.0475 | 0.0179 | {' brightest': 1, ' called': 1, ' easier': 1, ' faster': 1, ' fastest': 1, ' happier': 1, ' jumped': 1, ' looked': 1, ' played': 1, 'Record': 12} |
| simple_grammar_protocol_route | L24H17 | 0.5593 | 0.3960 | 0.3393 | 0.0383 | 0.0322 | {' brighter': 1, ' colder': 1, ' easier': 1, ' faster': 1, ' fastest': 1, ' happier': 1, ' jumped': 1, ' opened': 1, ' walked': 1, 'Record': 11} |
| simple_grammar_protocol_route | L16H1 | 0.5219 | 0.3397 | 0.2996 | 0.0400 | 0.0253 | {' brighter': 1, ' brightest': 1, ' colder': 1, ' easier': 1, ' faster': 1, ' fastest': 1, ' happier': 1, ' smaller': 1, ' smallest': 1, 'Record': 12} |
| color_value_reuse_difference | L23H0 | 0.5027 | 0.1340 | 0.2956 | 0.4517 | 0.0065 | {' coal': 1, ' color': 20, ' milk': 1, ' rose': 1, ' violet': 1} |
| simple_grammar_protocol_route | L19H15 | 0.4952 | 0.3461 | 0.3015 | 0.0278 | 0.0360 | {' called': 1, ' faster': 1, ' happier': 1, ' jumped': 1, ' looked': 1, ' opened': 1, ' played': 1, ' walked': 1, '.\n': 2, 'Record': 10} |

## Strict Interpretation

- This phase identifies candidate head routes for functional atlas expansion.
- It does not prove semantic identity, necessity, or sufficiency.
- The next causal phase should patch only the repeated top heads per function family.
