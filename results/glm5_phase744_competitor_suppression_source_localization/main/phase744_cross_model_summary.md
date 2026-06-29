# Phase 744 Competitor Suppression Source Localization (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: donor-recipient component delta add against the Phase743 current top competitor.

| model | component | in topK | n | margin delta | donor logit delta | competitor logit delta | donor top1 | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L32:attn_out | 0 | 12 | 2.797 | 2.536 | -0.260 | 0.167 | `{"boost_and_suppress_closure_candidate": 2, "boost_and_suppress_margin_candidate": 4, "boost_margin_candidate": 6}` |
| qwen3 | L33:attn_out | 0 | 12 | 4.542 | 2.948 | -1.594 | 0.000 | `{"boost_and_suppress_margin_candidate": 12}` |
| qwen3 | L30:attn_out | 0 | 12 | 1.974 | 1.349 | -0.625 | 0.000 | `{"boost_and_suppress_margin_candidate": 8, "boost_margin_candidate": 2, "weak_boost_candidate": 2}` |
| qwen3 | L35:attn_out | 0 | 12 | 1.818 | 2.609 | 0.792 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "boost_margin_candidate": 10}` |
| qwen3 | L28:mlp_out | 0 | 12 | 0.911 | 0.620 | -0.292 | 0.000 | `{"boost_and_suppress_margin_candidate": 5, "suppression_margin_candidate": 2, "weak_boost_candidate": 5}` |
| qwen3 | L31:mlp_out | 0 | 12 | 0.818 | 0.693 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 4, "boost_margin_candidate": 5, "suppression_margin_candidate": 1, "weak_boost_candidate": 2}` |
| qwen3 | L28:attn_out | 0 | 12 | 0.635 | 0.406 | -0.229 | 0.000 | `{"boost_and_suppress_margin_candidate": 8, "small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 2}` |
| qwen3 | L32:mlp_out | 0 | 12 | 0.599 | -0.047 | -0.646 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "harmful_or_competitor_support": 1, "small_or_no_effect": 1, "suppression_margin_candidate": 5, "weak_suppression_candidate": 3}` |
| glm4 | L34:attn_out | 0 | 6 | 2.708 | 2.615 | -0.094 | 0.667 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 3}` |
| glm4 | L35:attn_out | 0 | 6 | 0.927 | 0.729 | -0.198 | 0.333 | `{"boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 2, "small_or_no_effect": 1, "weak_suppression_candidate": 1}` |
| glm4 | L37:attn_out | 0 | 6 | 0.719 | 0.698 | -0.021 | 0.333 | `{"boost_closure_candidate": 2, "small_or_no_effect": 2, "weak_suppression_candidate": 2}` |
| glm4 | L35:mlp_out | 0 | 6 | 0.208 | 0.104 | -0.104 | 0.333 | `{"boost_closure_candidate": 2, "harmful_or_competitor_support": 1, "small_or_no_effect": 1, "weak_suppression_candidate": 2}` |
| glm4 | L36:attn_out | 0 | 6 | 0.500 | 0.458 | -0.042 | 0.167 | `{"boost_and_suppress_closure_candidate": 1, "boost_margin_candidate": 2, "small_or_no_effect": 1, "weak_suppression_candidate": 2}` |
| glm4 | L39:attn_out | 0 | 6 | 0.240 | 0.271 | 0.031 | 0.000 | `{"small_or_no_effect": 2, "weak_boost_candidate": 2, "weak_suppression_candidate": 2}` |
| glm4 | L34:mlp_out | 0 | 6 | 0.229 | 0.146 | -0.083 | 0.000 | `{"small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 4}` |
| glm4 | L38:attn_out | 0 | 6 | 0.135 | 0.219 | 0.083 | 0.000 | `{"small_or_no_effect": 2, "weak_boost_candidate": 4}` |
| deepseek7b | L23:attn_out | 0 | 11 | 1.994 | 1.597 | -0.398 | 0.182 | `{"boost_and_suppress_closure_candidate": 2, "boost_and_suppress_margin_candidate": 7, "boost_margin_candidate": 2}` |
| deepseek7b | L22:attn_out | 0 | 11 | 1.778 | 1.136 | -0.642 | 0.091 | `{"boost_and_suppress_margin_candidate": 5, "boost_closure_candidate": 1, "boost_margin_candidate": 3, "harmful_or_competitor_support": 1, "suppression_margin_candidate": 1}` |
| deepseek7b | L24:attn_out | 0 | 11 | 1.114 | 0.892 | -0.222 | 0.091 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 6, "boost_margin_candidate": 3, "weak_suppression_candidate": 1}` |
| deepseek7b | L24:mlp_out | 0 | 11 | 0.994 | 0.784 | -0.210 | 0.091 | `{"boost_and_suppress_margin_candidate": 7, "boost_closure_candidate": 1, "boost_margin_candidate": 1, "small_or_no_effect": 1, "suppression_margin_candidate": 1}` |
| deepseek7b | L26:mlp_out | 0 | 11 | 0.710 | 0.369 | -0.341 | 0.091 | `{"boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 1, "harmful_or_competitor_support": 2, "small_or_no_effect": 2, "suppression_margin_candidate": 1, "weak_boost_candidate": 2, "weak_suppression_candidate": 1}` |
| deepseek7b | L25:mlp_out | 0 | 11 | 0.676 | 0.273 | -0.403 | 0.091 | `{"boost_and_suppress_margin_candidate": 4, "boost_closure_candidate": 1, "small_or_no_effect": 1, "suppression_margin_candidate": 1, "weak_suppression_candidate": 4}` |
| deepseek7b | L25:attn_out | 0 | 11 | 0.710 | 0.403 | -0.307 | 0.000 | `{"boost_and_suppress_margin_candidate": 3, "harmful_or_competitor_support": 3, "small_or_no_effect": 2, "weak_boost_candidate": 1, "weak_suppression_candidate": 2}` |
| deepseek7b | L23:mlp_out | 0 | 11 | 0.307 | 0.125 | -0.182 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "harmful_or_competitor_support": 1, "suppression_margin_candidate": 1, "weak_boost_candidate": 3, "weak_suppression_candidate": 4}` |

## By Competitor Class

| model | class | component | n | margin delta | donor delta | competitor delta | donor top1 | roles |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | L32:attn_out | 12 | 2.797 | 2.536 | -0.260 | 0.167 | `{"boost_and_suppress_closure_candidate": 2, "boost_and_suppress_margin_candidate": 4, "boost_margin_candidate": 6}` |
| qwen3 | recipient_answer | L33:attn_out | 12 | 4.542 | 2.948 | -1.594 | 0.000 | `{"boost_and_suppress_margin_candidate": 12}` |
| qwen3 | recipient_answer | L30:attn_out | 12 | 1.974 | 1.349 | -0.625 | 0.000 | `{"boost_and_suppress_margin_candidate": 8, "boost_margin_candidate": 2, "weak_boost_candidate": 2}` |
| qwen3 | recipient_answer | L35:attn_out | 12 | 1.818 | 2.609 | 0.792 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "boost_margin_candidate": 10}` |
| qwen3 | recipient_answer | L28:mlp_out | 12 | 0.911 | 0.620 | -0.292 | 0.000 | `{"boost_and_suppress_margin_candidate": 5, "suppression_margin_candidate": 2, "weak_boost_candidate": 5}` |
| qwen3 | recipient_answer | L31:mlp_out | 12 | 0.818 | 0.693 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 4, "boost_margin_candidate": 5, "suppression_margin_candidate": 1, "weak_boost_candidate": 2}` |
| qwen3 | recipient_answer | L28:attn_out | 12 | 0.635 | 0.406 | -0.229 | 0.000 | `{"boost_and_suppress_margin_candidate": 8, "small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 2}` |
| qwen3 | recipient_answer | L32:mlp_out | 12 | 0.599 | -0.047 | -0.646 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "harmful_or_competitor_support": 1, "small_or_no_effect": 1, "suppression_margin_candidate": 5, "weak_suppression_candidate": 3}` |
| qwen3 | recipient_answer | L30:mlp_out | 12 | 0.344 | 0.438 | 0.094 | 0.000 | `{"boost_and_suppress_margin_candidate": 4, "harmful_or_competitor_support": 2, "small_or_no_effect": 2, "weak_boost_candidate": 3, "weak_suppression_candidate": 1}` |
| qwen3 | recipient_answer | L34:mlp_out | 12 | -2.104 | -1.479 | 0.625 | 0.000 | `{"harmful_or_competitor_support": 6, "small_or_no_effect": 1, "suppression_margin_candidate": 2, "weak_boost_candidate": 2, "weak_suppression_candidate": 1}` |
| qwen3 | recipient_answer | L35:mlp_out | 12 | -2.589 | 0.193 | 2.781 | 0.000 | `{"harmful_or_competitor_support": 10, "small_or_no_effect": 2}` |
| glm4 | other_vocab | L34:attn_out | 2 | 1.469 | 1.562 | 0.094 | 1.000 | `{"boost_closure_candidate": 2}` |
| glm4 | other_vocab | L35:attn_out | 2 | 0.812 | 0.781 | -0.031 | 1.000 | `{"boost_closure_candidate": 2}` |
| glm4 | other_vocab | L35:mlp_out | 2 | 0.531 | 0.594 | 0.062 | 1.000 | `{"boost_closure_candidate": 2}` |
| glm4 | other_vocab | L36:attn_out | 2 | 0.500 | 0.375 | -0.125 | 0.500 | `{"boost_and_suppress_closure_candidate": 1, "weak_suppression_candidate": 1}` |
| glm4 | other_vocab | L34:mlp_out | 2 | 0.375 | 0.188 | -0.188 | 0.000 | `{"weak_suppression_candidate": 2}` |
| glm4 | other_vocab | L39:attn_out | 2 | 0.250 | -0.031 | -0.281 | 0.000 | `{"weak_suppression_candidate": 2}` |
| glm4 | other_vocab | L38:attn_out | 2 | 0.125 | 0.250 | 0.125 | 0.000 | `{"weak_boost_candidate": 2}` |
| glm4 | other_vocab | L36:mlp_out | 2 | -1.438 | -1.906 | -0.469 | 0.000 | `{"harmful_or_competitor_support": 2}` |
| glm4 | other_vocab | L37:attn_out | 2 | 0.000 | 0.031 | 0.031 | 0.000 | `{"small_or_no_effect": 2}` |
| glm4 | echo_object_or_relation | L34:attn_out | 4 | 3.328 | 3.141 | -0.188 | 0.500 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 1}` |
| glm4 | echo_object_or_relation | L37:attn_out | 4 | 1.078 | 1.031 | -0.047 | 0.500 | `{"boost_closure_candidate": 2, "weak_suppression_candidate": 2}` |
| glm4 | echo_object_or_relation | L35:attn_out | 4 | 0.984 | 0.703 | -0.281 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "small_or_no_effect": 1, "weak_suppression_candidate": 1}` |
| glm4 | echo_object_or_relation | L36:mlp_out | 4 | 0.719 | 0.844 | 0.125 | 0.000 | `{"boost_margin_candidate": 2, "weak_suppression_candidate": 2}` |
| glm4 | echo_object_or_relation | L36:attn_out | 4 | 0.500 | 0.500 | 0.000 | 0.000 | `{"boost_margin_candidate": 2, "small_or_no_effect": 1, "weak_suppression_candidate": 1}` |
| glm4 | echo_object_or_relation | L39:attn_out | 4 | 0.234 | 0.422 | 0.188 | 0.000 | `{"small_or_no_effect": 2, "weak_boost_candidate": 2}` |
| glm4 | echo_object_or_relation | L34:mlp_out | 4 | 0.156 | 0.125 | -0.031 | 0.000 | `{"small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 2}` |
| deepseek7b | recipient_answer | L22:attn_out | 1 | 5.562 | 0.688 | -4.875 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L23:attn_out | 1 | 3.562 | 2.438 | -1.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L25:attn_out | 1 | 2.938 | 0.188 | -2.750 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L26:mlp_out | 1 | 2.688 | 2.062 | -0.625 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L24:attn_out | 1 | 1.500 | 0.500 | -1.000 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L24:mlp_out | 1 | 1.188 | 0.688 | -0.500 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L23:mlp_out | 1 | 0.750 | -0.125 | -0.875 | 0.000 | `{"suppression_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L25:mlp_out | 1 | 0.375 | -0.375 | -0.750 | 0.000 | `{"weak_suppression_candidate": 1}` |
| deepseek7b | recipient_answer | L22:mlp_out | 1 | -0.125 | -0.500 | -0.375 | 0.000 | `{"harmful_or_competitor_support": 1}` |
| deepseek7b | other_vocab | L26:mlp_out | 1 | 3.438 | 2.500 | -0.938 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | other_vocab | L23:attn_out | 1 | 3.000 | 2.188 | -0.812 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | other_vocab | L22:attn_out | 1 | 2.938 | 3.125 | 0.188 | 0.000 | `{"boost_margin_candidate": 1}` |
| deepseek7b | other_vocab | L25:attn_out | 1 | 2.875 | 2.250 | -0.625 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | other_vocab | L24:mlp_out | 1 | 2.375 | 1.688 | -0.688 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | other_vocab | L25:mlp_out | 1 | 1.938 | 0.188 | -1.750 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | other_vocab | L24:attn_out | 1 | 1.500 | 1.125 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |

## Strict Interpretation

- A positive margin delta means the component can improve donor-vs-current-competitor competition when transplanted.
- A negative competitor-logit delta is direct evidence of suppression; a positive donor-logit delta is boost-dominant rather than pure suppression.
- This phase is still whole-component level and does not yet identify head/channel/neuron mechanisms.

Atlas graph: nodes=30 edges=27
