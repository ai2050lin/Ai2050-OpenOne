# Phase337 Protocol Qualification

- Registered/executed: 108/108
- Capable model-interface cells: 7/9
- Passing interfaces: raw_completion, answer_aligned_chat
- Preferred next-stage interface: answer_aligned_chat
- Protocol qualification gate: True
- Mechanism causal claims: 0
- Single-unit causal claims: 0

## Cell Results

- deepseek7b / answer_aligned_chat: capable 12/12, answer reached 12/12, answer-head correct 12/12, semantic anywhere 12/12, phrase valid 12/12, gate True
- deepseek7b / native_chat: capable 0/12, answer reached 1/12, answer-head correct 0/12, semantic anywhere 12/12, phrase valid 12/12, gate False
- deepseek7b / raw_completion: capable 11/12, answer reached 12/12, answer-head correct 11/12, semantic anywhere 12/12, phrase valid 12/12, gate True
- glm4 / answer_aligned_chat: capable 12/12, answer reached 12/12, answer-head correct 12/12, semantic anywhere 12/12, phrase valid 12/12, gate True
- glm4 / native_chat: capable 12/12, answer reached 12/12, answer-head correct 12/12, semantic anywhere 12/12, phrase valid 12/12, gate True
- glm4 / raw_completion: capable 12/12, answer reached 12/12, answer-head correct 12/12, semantic anywhere 12/12, phrase valid 12/12, gate True
- qwen3 / answer_aligned_chat: capable 12/12, answer reached 12/12, answer-head correct 12/12, semantic anywhere 12/12, phrase valid 12/12, gate True
- qwen3 / native_chat: capable 2/12, answer reached 2/12, answer-head correct 2/12, semantic anywhere 12/12, phrase valid 12/12, gate False
- qwen3 / raw_completion: capable 12/12, answer reached 12/12, answer-head correct 12/12, semantic anywhere 12/12, phrase valid 12/12, gate True

This phase measures protocol eligibility only. It does not capture activations, intervene on the model, or establish a language mechanism.
