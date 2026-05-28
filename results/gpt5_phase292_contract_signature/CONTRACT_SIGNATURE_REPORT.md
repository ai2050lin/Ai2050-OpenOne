# Phase 292 Contract Signature Report
## Inputs
- Phase 290 dir: `/home/rankrank/Documents/OpenOne/Ai2050-OpenOne/results/gpt5_phase290_contract_break_full`
- Phase 291 dir: `/home/rankrank/Documents/OpenOne/Ai2050-OpenOne/results/gpt5_phase291_block_contract_full`
## Model Summaries
### qwen3
- subtypes: 19
- avg p290 both best: 0.8506
- avg p291 both best: 0.8971
- avg p291 cross max drop: 0.7375
- top reuse pairs:
  - complement_clause / syntactic_do_not: 0.9732
  - complement_clause / relative_clause: 0.9721
  - get_passive / pp_chain: 0.9704
  - relative_clause / syntactic_do_not: 0.9607
  - pp_chain / relative_clause: 0.9519
### glm4
- subtypes: 19
- avg p290 both best: 0.9550
- avg p291 both best: 0.9963
- avg p291 cross max drop: 0.9648
- top reuse pairs:
  - complement_clause / possessive_chain: 0.9986
  - conditional / existential_no: 0.9958
  - complement_clause / never: 0.9939
  - never / possessive_chain: 0.9933
  - and_or / inference: 0.9893
### deepseek7b
- subtypes: 19
- avg p290 both best: 0.6624
- avg p291 both best: 0.9131
- avg p291 cross max drop: 0.7937
- top reuse pairs:
  - possessive_chain / relative_clause: 0.9992
  - get_passive / possessive_chain: 0.9988
  - never / syntactic_do_not: 0.9985
  - relative_clause / scope_quantifier: 0.9984
  - possessive_chain / scope_quantifier: 0.9983
## Cross Model Same Subtype Similarity
- glm4 vs deepseek7b: mean=0.8445, n=19
- qwen3 vs deepseek7b: mean=0.8495, n=19
- qwen3 vs glm4: mean=0.8264, n=19
