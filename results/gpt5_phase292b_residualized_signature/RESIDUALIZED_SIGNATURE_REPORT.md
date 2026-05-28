# Phase 292b Residualized Signature Report
## Inputs
- input: `results/gpt5_phase292_contract_signature/contract_signatures.json`
- vector_kind: `canonical`

## Model Summary
### qwen3
- subtypes: 19, pairs: 171
- raw similarity: mean=0.7789, min=0.3541, max=0.9732
- model-centered similarity: mean=-0.0523, min=-0.9219, max=0.8795
- category-centered similarity: mean=-0.0535, min=-0.8586, max=0.9236
- diagnostic labels: stable=11, model_shape=2, category_shape=5, differentiation=21
- top residual-stable candidates:
  - complement_clause / syntactic_do_not: raw=0.9732, model_resid=0.8614, category_resid=0.6784
  - complement_clause / relative_clause: raw=0.9721, model_resid=0.7949, category_resid=0.3559
  - get_passive / pp_chain: raw=0.9704, model_resid=0.7676, category_resid=0.7962
  - relative_clause / syntactic_do_not: raw=0.9607, model_resid=0.6346, category_resid=0.3115
  - causal / contrast: raw=0.9507, model_resid=0.8795, category_resid=0.6785
- high-raw but residual-weak candidates:
  - dative_passive / possessive_chain: label=category_shape_candidate, raw=0.9482, model_resid=0.5706, category_resid=-0.1889
  - lexical_not_adj / pp_chain: label=category_shape_candidate, raw=0.9404, model_resid=0.2382, category_resid=-0.2319
  - get_passive / lexical_not_adj: label=category_shape_candidate, raw=0.9385, model_resid=0.3498, category_resid=-0.3191
  - lexical_not_adj / never: label=category_shape_candidate, raw=0.9034, model_resid=0.3412, category_resid=-0.1214
  - complement_clause / pp_chain: label=category_shape_candidate, raw=0.9024, model_resid=0.4194, category_resid=-0.4181
### glm4
- subtypes: 19, pairs: 171
- raw similarity: mean=0.8623, min=0.6715, max=0.9986
- model-centered similarity: mean=-0.0476, min=-0.9630, max=0.9905
- category-centered similarity: mean=-0.0494, min=-0.9599, max=0.9726
- diagnostic labels: stable=21, model_shape=8, category_shape=18, differentiation=0
- top residual-stable candidates:
  - complement_clause / possessive_chain: raw=0.9986, model_resid=0.9905, category_resid=0.9726
  - conditional / existential_no: raw=0.9958, model_resid=0.9673, category_resid=0.7816
  - complement_clause / never: raw=0.9939, model_resid=0.9571, category_resid=0.7936
  - never / possessive_chain: raw=0.9933, model_resid=0.9539, category_resid=0.7973
  - and_or / inference: raw=0.9893, model_resid=0.9277, category_resid=0.5166
- high-raw but residual-weak candidates:
  - and_or / get_passive: label=category_shape_candidate, raw=0.9707, model_resid=0.8167, category_resid=0.0659
  - causal / get_passive: label=category_shape_candidate, raw=0.9626, model_resid=0.8128, category_resid=0.0612
  - dative_passive / morphological_neg: label=category_shape_candidate, raw=0.9602, model_resid=0.6439, category_resid=-0.1039
  - inference / no_agent: label=category_shape_candidate, raw=0.9523, model_resid=0.7280, category_resid=-0.0000
  - by_phrase / dative_passive: label=category_shape_candidate, raw=0.9427, model_resid=0.6205, category_resid=-0.1721
### deepseek7b
- subtypes: 19, pairs: 171
- raw similarity: mean=0.9556, min=0.7980, max=0.9992
- model-centered similarity: mean=-0.0168, min=-0.7865, max=0.9474
- category-centered similarity: mean=-0.0002, min=-0.8773, max=0.9662
- diagnostic labels: stable=21, model_shape=108, category_shape=7, differentiation=0
- top residual-stable candidates:
  - possessive_chain / relative_clause: raw=0.9992, model_resid=0.9447, category_resid=0.9662
  - get_passive / possessive_chain: raw=0.9988, model_resid=0.9197, category_resid=0.6551
  - never / syntactic_do_not: raw=0.9985, model_resid=0.9052, category_resid=0.8377
  - by_phrase / dative_passive: raw=0.9983, model_resid=0.9474, category_resid=0.9165
  - get_passive / relative_clause: raw=0.9981, model_resid=0.8734, category_resid=0.6329
- high-raw but residual-weak candidates:
  - and_or / existential_no: label=category_shape_candidate, raw=0.9941, model_resid=0.7787, category_resid=0.0074
  - contrast / morphological_neg: label=category_shape_candidate, raw=0.9705, model_resid=0.2303, category_resid=-0.2948
  - causal / morphological_neg: label=category_shape_candidate, raw=0.9701, model_resid=0.2283, category_resid=-0.2891
  - and_or / morphological_neg: label=category_shape_candidate, raw=0.9694, model_resid=0.2035, category_resid=-0.3097
  - conditional / no_agent: label=category_shape_candidate, raw=0.9547, model_resid=0.2050, category_resid=-0.2915

## Cross Model Same Subtype Means
- glm4 vs deepseek7b / category_centered: mean=0.1949, n=19
- glm4 vs deepseek7b / group_normalized: mean=0.7476, n=19
- glm4 vs deepseek7b / model_centered: mean=0.1767, n=19
- glm4 vs deepseek7b / raw: mean=0.8445, n=19
- glm4 vs deepseek7b / zscore_model: mean=-0.0501, n=19
- qwen3 vs deepseek7b / category_centered: mean=0.0729, n=19
- qwen3 vs deepseek7b / group_normalized: mean=0.7721, n=19
- qwen3 vs deepseek7b / model_centered: mean=0.1890, n=19
- qwen3 vs deepseek7b / raw: mean=0.8495, n=19
- qwen3 vs deepseek7b / zscore_model: mean=-0.0792, n=19
- qwen3 vs glm4 / category_centered: mean=0.1177, n=19
- qwen3 vs glm4 / group_normalized: mean=0.8518, n=19
- qwen3 vs glm4 / model_centered: mean=0.3531, n=19
- qwen3 vs glm4 / raw: mean=0.8264, n=19
- qwen3 vs glm4 / zscore_model: mean=0.3774, n=19

## Caution
- Residualized cosine values can be negative; they measure deviation-pattern similarity, not absolute functional strength.
- Diagnostic labels are screening labels, not proof of true reuse or differentiation.
