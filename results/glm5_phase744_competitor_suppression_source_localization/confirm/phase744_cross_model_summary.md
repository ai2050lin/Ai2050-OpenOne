# Phase 744 Competitor Suppression Source Localization (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: donor-recipient component delta add against the Phase743 current top competitor.

| model | component | in topK | n | margin delta | donor logit delta | competitor logit delta | donor top1 | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L32:attn_out | 0 | 20 | 3.259 | 2.959 | -0.300 | 0.100 | `{"boost_and_suppress_closure_candidate": 2, "boost_and_suppress_margin_candidate": 11, "boost_margin_candidate": 7}` |
| qwen3 | L33:attn_out | 0 | 20 | 5.144 | 3.038 | -2.106 | 0.000 | `{"boost_and_suppress_margin_candidate": 20}` |
| qwen3 | L30:attn_out | 0 | 20 | 1.991 | 1.428 | -0.562 | 0.000 | `{"boost_and_suppress_margin_candidate": 13, "boost_margin_candidate": 3, "weak_boost_candidate": 2, "weak_suppression_candidate": 2}` |
| qwen3 | L35:attn_out | 0 | 20 | 1.644 | 2.406 | 0.762 | 0.000 | `{"boost_and_suppress_margin_candidate": 4, "boost_margin_candidate": 12, "small_or_no_effect": 1, "weak_boost_candidate": 3}` |
| qwen3 | L32:mlp_out | 0 | 20 | 0.975 | 0.175 | -0.800 | 0.000 | `{"boost_and_suppress_margin_candidate": 6, "harmful_or_competitor_support": 1, "small_or_no_effect": 1, "suppression_margin_candidate": 7, "weak_suppression_candidate": 5}` |
| qwen3 | L31:mlp_out | 0 | 20 | 0.928 | 0.728 | -0.200 | 0.000 | `{"boost_and_suppress_margin_candidate": 10, "boost_margin_candidate": 7, "suppression_margin_candidate": 1, "weak_boost_candidate": 2}` |
| qwen3 | L28:mlp_out | 0 | 20 | 0.784 | 0.591 | -0.194 | 0.000 | `{"boost_and_suppress_margin_candidate": 7, "boost_margin_candidate": 2, "small_or_no_effect": 1, "suppression_margin_candidate": 2, "weak_boost_candidate": 5, "weak_suppression_candidate": 3}` |
| qwen3 | L28:attn_out | 0 | 20 | 0.769 | 0.412 | -0.356 | 0.000 | `{"boost_and_suppress_margin_candidate": 15, "small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 3}` |
| glm4 | L34:attn_out | 0 | 10 | 2.306 | 2.131 | -0.175 | 0.800 | `{"boost_and_suppress_closure_candidate": 3, "boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 5}` |
| glm4 | L35:attn_out | 0 | 10 | 0.794 | 0.581 | -0.212 | 0.400 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 3, "small_or_no_effect": 1, "weak_suppression_candidate": 3}` |
| glm4 | L35:mlp_out | 0 | 10 | 0.231 | 0.188 | -0.044 | 0.400 | `{"boost_closure_candidate": 4, "harmful_or_competitor_support": 1, "small_or_no_effect": 3, "weak_suppression_candidate": 2}` |
| glm4 | L36:attn_out | 0 | 10 | 0.475 | 0.412 | -0.062 | 0.300 | `{"boost_and_suppress_closure_candidate": 3, "boost_margin_candidate": 2, "small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 3}` |
| glm4 | L37:attn_out | 0 | 10 | 0.444 | 0.438 | -0.006 | 0.200 | `{"boost_closure_candidate": 2, "small_or_no_effect": 6, "weak_suppression_candidate": 2}` |
| glm4 | L39:attn_out | 0 | 10 | 0.219 | 0.225 | 0.006 | 0.200 | `{"small_or_no_effect": 2, "suppression_closure_candidate": 2, "weak_boost_candidate": 4, "weak_suppression_candidate": 2}` |
| glm4 | L38:attn_out | 0 | 10 | 0.144 | 0.225 | 0.081 | 0.200 | `{"boost_closure_candidate": 2, "small_or_no_effect": 2, "weak_boost_candidate": 6}` |
| glm4 | L34:mlp_out | 0 | 10 | 0.131 | 0.037 | -0.094 | 0.200 | `{"harmful_or_competitor_support": 2, "small_or_no_effect": 1, "suppression_closure_candidate": 2, "weak_boost_candidate": 1, "weak_suppression_candidate": 4}` |
| deepseek7b | L23:attn_out | 0 | 19 | 1.977 | 1.671 | -0.306 | 0.421 | `{"boost_and_suppress_closure_candidate": 6, "boost_and_suppress_margin_candidate": 9, "boost_closure_candidate": 2, "boost_margin_candidate": 2}` |
| deepseek7b | L22:attn_out | 0 | 19 | 1.688 | 1.303 | -0.385 | 0.316 | `{"boost_and_suppress_closure_candidate": 2, "boost_and_suppress_margin_candidate": 6, "boost_closure_candidate": 4, "boost_margin_candidate": 4, "harmful_or_competitor_support": 1, "suppression_margin_candidate": 1, "weak_boost_candidate": 1}` |
| deepseek7b | L24:attn_out | 0 | 19 | 1.118 | 0.993 | -0.125 | 0.263 | `{"boost_and_suppress_closure_candidate": 3, "boost_and_suppress_margin_candidate": 7, "boost_closure_candidate": 2, "boost_margin_candidate": 6, "weak_suppression_candidate": 1}` |
| deepseek7b | L24:mlp_out | 0 | 19 | 0.868 | 0.753 | -0.115 | 0.105 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 9, "boost_closure_candidate": 1, "boost_margin_candidate": 2, "small_or_no_effect": 2, "suppression_margin_candidate": 1, "weak_boost_candidate": 2, "weak_suppression_candidate": 1}` |
| deepseek7b | L25:mlp_out | 0 | 19 | 0.658 | 0.401 | -0.257 | 0.105 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 8, "boost_closure_candidate": 1, "boost_margin_candidate": 1, "harmful_or_competitor_support": 1, "small_or_no_effect": 2, "suppression_margin_candidate": 1, "weak_suppression_candidate": 4}` |
| deepseek7b | L26:mlp_out | 0 | 19 | 0.609 | 0.372 | -0.237 | 0.105 | `{"boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 1, "boost_margin_candidate": 2, "harmful_or_competitor_support": 3, "small_or_no_effect": 3, "suppression_closure_candidate": 1, "suppression_margin_candidate": 2, "weak_boost_candidate": 3, "weak_suppression_candidate": 2}` |
| deepseek7b | L23:mlp_out | 0 | 19 | 0.375 | 0.171 | -0.204 | 0.105 | `{"boost_and_suppress_closure_candidate": 1, "boost_and_suppress_margin_candidate": 4, "boost_closure_candidate": 1, "harmful_or_competitor_support": 3, "suppression_margin_candidate": 1, "weak_boost_candidate": 3, "weak_suppression_candidate": 6}` |
| deepseek7b | L25:attn_out | 0 | 19 | 0.398 | 0.188 | -0.211 | 0.053 | `{"boost_and_suppress_margin_candidate": 3, "boost_margin_candidate": 1, "harmful_or_competitor_support": 6, "small_or_no_effect": 2, "suppression_closure_candidate": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 5}` |

## By Competitor Class

| model | class | component | n | margin delta | donor delta | competitor delta | donor top1 | roles |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | L32:attn_out | 20 | 3.259 | 2.959 | -0.300 | 0.100 | `{"boost_and_suppress_closure_candidate": 2, "boost_and_suppress_margin_candidate": 11, "boost_margin_candidate": 7}` |
| qwen3 | recipient_answer | L33:attn_out | 20 | 5.144 | 3.038 | -2.106 | 0.000 | `{"boost_and_suppress_margin_candidate": 20}` |
| qwen3 | recipient_answer | L30:attn_out | 20 | 1.991 | 1.428 | -0.562 | 0.000 | `{"boost_and_suppress_margin_candidate": 13, "boost_margin_candidate": 3, "weak_boost_candidate": 2, "weak_suppression_candidate": 2}` |
| qwen3 | recipient_answer | L35:attn_out | 20 | 1.644 | 2.406 | 0.762 | 0.000 | `{"boost_and_suppress_margin_candidate": 4, "boost_margin_candidate": 12, "small_or_no_effect": 1, "weak_boost_candidate": 3}` |
| qwen3 | recipient_answer | L32:mlp_out | 20 | 0.975 | 0.175 | -0.800 | 0.000 | `{"boost_and_suppress_margin_candidate": 6, "harmful_or_competitor_support": 1, "small_or_no_effect": 1, "suppression_margin_candidate": 7, "weak_suppression_candidate": 5}` |
| qwen3 | recipient_answer | L31:mlp_out | 20 | 0.928 | 0.728 | -0.200 | 0.000 | `{"boost_and_suppress_margin_candidate": 10, "boost_margin_candidate": 7, "suppression_margin_candidate": 1, "weak_boost_candidate": 2}` |
| qwen3 | recipient_answer | L28:mlp_out | 20 | 0.784 | 0.591 | -0.194 | 0.000 | `{"boost_and_suppress_margin_candidate": 7, "boost_margin_candidate": 2, "small_or_no_effect": 1, "suppression_margin_candidate": 2, "weak_boost_candidate": 5, "weak_suppression_candidate": 3}` |
| qwen3 | recipient_answer | L28:attn_out | 20 | 0.769 | 0.412 | -0.356 | 0.000 | `{"boost_and_suppress_margin_candidate": 15, "small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 3}` |
| qwen3 | recipient_answer | L30:mlp_out | 20 | 0.372 | 0.447 | 0.075 | 0.000 | `{"boost_and_suppress_margin_candidate": 6, "harmful_or_competitor_support": 4, "small_or_no_effect": 2, "weak_boost_candidate": 6, "weak_suppression_candidate": 2}` |
| qwen3 | recipient_answer | L34:mlp_out | 20 | -1.956 | -1.062 | 0.894 | 0.000 | `{"harmful_or_competitor_support": 12, "small_or_no_effect": 1, "suppression_margin_candidate": 2, "weak_boost_candidate": 4, "weak_suppression_candidate": 1}` |
| qwen3 | recipient_answer | L35:mlp_out | 20 | -3.297 | -0.409 | 2.888 | 0.000 | `{"harmful_or_competitor_support": 18, "small_or_no_effect": 2}` |
| glm4 | other_vocab | L34:attn_out | 4 | 1.438 | 1.484 | 0.047 | 1.000 | `{"boost_closure_candidate": 4}` |
| glm4 | other_vocab | L35:attn_out | 4 | 0.781 | 0.719 | -0.062 | 1.000 | `{"boost_and_suppress_closure_candidate": 1, "boost_closure_candidate": 3}` |
| glm4 | other_vocab | L35:mlp_out | 4 | 0.500 | 0.578 | 0.078 | 1.000 | `{"boost_closure_candidate": 4}` |
| glm4 | other_vocab | L36:attn_out | 4 | 0.547 | 0.406 | -0.141 | 0.750 | `{"boost_and_suppress_closure_candidate": 3, "weak_suppression_candidate": 1}` |
| glm4 | other_vocab | L34:mlp_out | 4 | 0.312 | 0.109 | -0.203 | 0.500 | `{"suppression_closure_candidate": 2, "weak_suppression_candidate": 2}` |
| glm4 | other_vocab | L39:attn_out | 4 | 0.250 | -0.031 | -0.281 | 0.500 | `{"suppression_closure_candidate": 2, "weak_suppression_candidate": 2}` |
| glm4 | other_vocab | L38:attn_out | 4 | 0.156 | 0.234 | 0.078 | 0.500 | `{"boost_closure_candidate": 2, "weak_boost_candidate": 2}` |
| glm4 | other_vocab | L37:attn_out | 4 | 0.016 | 0.047 | 0.031 | 0.000 | `{"small_or_no_effect": 4}` |
| glm4 | other_vocab | L36:mlp_out | 4 | -1.375 | -1.891 | -0.516 | 0.000 | `{"harmful_or_competitor_support": 4}` |
| glm4 | echo_object_or_relation | L34:attn_out | 6 | 2.885 | 2.562 | -0.323 | 0.667 | `{"boost_and_suppress_closure_candidate": 3, "boost_and_suppress_margin_candidate": 2, "boost_closure_candidate": 1}` |
| glm4 | echo_object_or_relation | L37:attn_out | 6 | 0.729 | 0.698 | -0.031 | 0.333 | `{"boost_closure_candidate": 2, "small_or_no_effect": 2, "weak_suppression_candidate": 2}` |
| glm4 | echo_object_or_relation | L36:mlp_out | 6 | 0.698 | 0.802 | 0.104 | 0.333 | `{"boost_closure_candidate": 2, "boost_margin_candidate": 2, "weak_suppression_candidate": 2}` |
| glm4 | echo_object_or_relation | L35:attn_out | 6 | 0.802 | 0.490 | -0.312 | 0.000 | `{"boost_and_suppress_margin_candidate": 2, "small_or_no_effect": 1, "weak_suppression_candidate": 3}` |
| glm4 | echo_object_or_relation | L36:attn_out | 6 | 0.427 | 0.417 | -0.010 | 0.000 | `{"boost_margin_candidate": 2, "small_or_no_effect": 1, "weak_boost_candidate": 1, "weak_suppression_candidate": 2}` |
| glm4 | echo_object_or_relation | L39:attn_out | 6 | 0.198 | 0.396 | 0.198 | 0.000 | `{"small_or_no_effect": 2, "weak_boost_candidate": 4}` |
| glm4 | echo_object_or_relation | L38:attn_out | 6 | 0.135 | 0.219 | 0.083 | 0.000 | `{"small_or_no_effect": 2, "weak_boost_candidate": 4}` |
| deepseek7b | recipient_answer | L22:attn_out | 1 | 5.562 | 0.688 | -4.875 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L23:attn_out | 1 | 3.562 | 2.438 | -1.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L25:attn_out | 1 | 2.938 | 0.188 | -2.750 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L26:mlp_out | 1 | 2.688 | 2.062 | -0.625 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L24:attn_out | 1 | 1.500 | 0.500 | -1.000 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L24:mlp_out | 1 | 1.188 | 0.688 | -0.500 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L23:mlp_out | 1 | 0.750 | -0.125 | -0.875 | 0.000 | `{"suppression_margin_candidate": 1}` |
| deepseek7b | recipient_answer | L25:mlp_out | 1 | 0.375 | -0.375 | -0.750 | 0.000 | `{"weak_suppression_candidate": 1}` |
| deepseek7b | recipient_answer | L22:mlp_out | 1 | -0.125 | -0.500 | -0.375 | 0.000 | `{"harmful_or_competitor_support": 1}` |
| deepseek7b | punctuation_or_stop | L24:mlp_out | 1 | 2.125 | 2.125 | 0.000 | 0.000 | `{"boost_margin_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L22:attn_out | 1 | 1.875 | 2.000 | 0.125 | 0.000 | `{"boost_margin_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L23:attn_out | 1 | 1.688 | 1.312 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L23:mlp_out | 1 | 1.688 | 0.438 | -1.250 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L25:mlp_out | 1 | 1.500 | 1.000 | -0.500 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L24:attn_out | 1 | 1.312 | 1.312 | 0.000 | 0.000 | `{"boost_margin_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L22:mlp_out | 1 | 1.000 | 0.625 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |

## Strict Interpretation

- A positive margin delta means the component can improve donor-vs-current-competitor competition when transplanted.
- A negative competitor-logit delta is direct evidence of suppression; a positive donor-logit delta is boost-dominant rather than pure suppression.
- This phase is still whole-component level and does not yet identify head/channel/neuron mechanisms.

Atlas graph: nodes=30 edges=27
