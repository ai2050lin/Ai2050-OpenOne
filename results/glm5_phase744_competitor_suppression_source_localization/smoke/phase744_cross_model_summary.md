# Phase 744 Competitor Suppression Source Localization (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: donor-recipient component delta add against the Phase743 current top competitor.

| model | component | in topK | n | margin delta | donor logit delta | competitor logit delta | donor top1 | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L30:attn_out | 0 | 1 | 1.750 | 1.500 | -0.250 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | L28:mlp_out | 0 | 1 | 1.250 | 0.875 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | L32:attn_out | 0 | 1 | 1.000 | 2.375 | 1.375 | 0.000 | `{"boost_margin_candidate": 1}` |
| qwen3 | L30:mlp_out | 0 | 1 | 0.750 | 0.375 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | L28:attn_out | 0 | 1 | 0.625 | 0.500 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | L31:mlp_out | 0 | 1 | 0.375 | 0.625 | 0.250 | 0.000 | `{"weak_boost_candidate": 1}` |
| deepseek7b | L23:attn_out | 0 | 1 | 2.625 | 2.500 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | L24:mlp_out | 0 | 1 | 1.375 | 1.250 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | L24:attn_out | 0 | 1 | 0.875 | 1.000 | 0.125 | 0.000 | `{"boost_margin_candidate": 1}` |
| deepseek7b | L22:mlp_out | 0 | 1 | 0.125 | 0.250 | 0.125 | 0.000 | `{"weak_boost_candidate": 1}` |
| deepseek7b | L23:mlp_out | 0 | 1 | 0.125 | 0.375 | 0.250 | 0.000 | `{"weak_boost_candidate": 1}` |
| deepseek7b | L22:attn_out | 0 | 1 | -0.188 | 0.062 | 0.250 | 0.000 | `{"harmful_or_competitor_support": 1}` |

## By Competitor Class

| model | class | component | n | margin delta | donor delta | competitor delta | donor top1 | roles |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | L30:attn_out | 1 | 1.750 | 1.500 | -0.250 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | recipient_answer | L28:mlp_out | 1 | 1.250 | 0.875 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | recipient_answer | L32:attn_out | 1 | 1.000 | 2.375 | 1.375 | 0.000 | `{"boost_margin_candidate": 1}` |
| qwen3 | recipient_answer | L30:mlp_out | 1 | 0.750 | 0.375 | -0.375 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | recipient_answer | L28:attn_out | 1 | 0.625 | 0.500 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| qwen3 | recipient_answer | L31:mlp_out | 1 | 0.375 | 0.625 | 0.250 | 0.000 | `{"weak_boost_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L23:attn_out | 1 | 2.625 | 2.500 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L24:mlp_out | 1 | 1.375 | 1.250 | -0.125 | 0.000 | `{"boost_and_suppress_margin_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L24:attn_out | 1 | 0.875 | 1.000 | 0.125 | 0.000 | `{"boost_margin_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L22:mlp_out | 1 | 0.125 | 0.250 | 0.125 | 0.000 | `{"weak_boost_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L23:mlp_out | 1 | 0.125 | 0.375 | 0.250 | 0.000 | `{"weak_boost_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L22:attn_out | 1 | -0.188 | 0.062 | 0.250 | 0.000 | `{"harmful_or_competitor_support": 1}` |

## Strict Interpretation

- A positive margin delta means the component can improve donor-vs-current-competitor competition when transplanted.
- A negative competitor-logit delta is direct evidence of suppression; a positive donor-logit delta is boost-dominant rather than pure suppression.
- This phase is still whole-component level and does not yet identify head/channel/neuron mechanisms.

Atlas graph: nodes=18 edges=15
