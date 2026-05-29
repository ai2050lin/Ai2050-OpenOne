# Phase 299 GSSC Residualized Dynamic Similarity

## qwen3
- raw_mean: 0.865291
- model_centered_mean: -0.013480
- category_centered_mean: -0.015242
- group_centered_mean: -0.013480
- label_counts: {'model_curve_artifact_candidate': 91, 'ordinary': 659, 'residual_stable_reuse_candidate': 85, 'category_curve_artifact_candidate': 108, 'stable_differentiation_candidate': 47}
- stable reuse candidates:
  - contrast / no_agent: raw=0.9897, model=0.9083, category=0.8689, group=0.9083, label=residual_stable_reuse_candidate
  - causal / contrast: raw=0.9899, model=0.9017, category=0.8500, group=0.9017, label=residual_stable_reuse_candidate
  - contrast / inference: raw=0.9885, model=0.8887, category=0.8266, group=0.8887, label=residual_stable_reuse_candidate
  - causal / inference: raw=0.9862, model=0.8579, category=0.7744, group=0.8579, label=residual_stable_reuse_candidate
  - causal / no_agent: raw=0.9833, model=0.8456, category=0.7540, group=0.8456, label=residual_stable_reuse_candidate
- stable differentiation candidates:
  - concise / syntactic_do_not: raw=0.3775, model=-0.3702, category=-0.1236, group=-0.3702, label=stable_differentiation_candidate
  - by_phrase / concise: raw=0.4109, model=-0.4806, category=-0.1506, group=-0.4806, label=stable_differentiation_candidate
  - center_embedding / concise: raw=0.4166, model=-0.2784, category=-0.3283, group=-0.2784, label=stable_differentiation_candidate
  - concise / lexical_not_adj: raw=0.4309, model=-0.4422, category=-0.1981, group=-0.4422, label=stable_differentiation_candidate
  - concise / nested_condition: raw=0.4408, model=-0.1780, category=-0.1290, group=-0.1780, label=stable_differentiation_candidate

## glm4
- raw_mean: 0.922709
- model_centered_mean: -0.014066
- category_centered_mean: -0.015589
- group_centered_mean: -0.014066
- label_counts: {'ordinary': 356, 'residual_stable_reuse_candidate': 178, 'model_curve_artifact_candidate': 336, 'category_curve_artifact_candidate': 120}
- stable reuse candidates:
  - casual / concise: raw=0.9944, model=0.9722, category=0.9302, group=0.9722, label=residual_stable_reuse_candidate
  - long_passive / nested_passive: raw=0.9963, model=0.8977, category=0.9399, group=0.8977, label=residual_stable_reuse_candidate
  - conditional / contrast: raw=0.9964, model=0.9472, category=0.8906, group=0.9472, label=residual_stable_reuse_candidate
  - concise / they_coref: raw=0.9957, model=0.9794, category=0.8888, group=0.9794, label=residual_stable_reuse_candidate
  - causal / inference: raw=0.9952, model=0.9343, category=0.8681, group=0.9343, label=residual_stable_reuse_candidate
- stable differentiation candidates:
  - it_coref / scope_quantifier: raw=0.7516, model=-0.6468, category=-0.2928, group=-0.6468, label=ordinary
  - it_coref / lexical_not_adj: raw=0.7572, model=-0.6765, category=-0.2697, group=-0.6765, label=ordinary
  - it_coref / never: raw=0.7574, model=-0.6223, category=-0.3174, group=-0.6223, label=ordinary
  - it_coref / relative_clause: raw=0.7575, model=-0.6782, category=-0.2424, group=-0.6782, label=ordinary
  - scope_quantifier / they_coref: raw=0.7610, model=-0.6895, category=-0.4123, group=-0.6895, label=ordinary

## deepseek7b
- raw_mean: 0.851783
- model_centered_mean: -0.020525
- category_centered_mean: -0.020343
- group_centered_mean: -0.020525
- label_counts: {'ordinary': 800, 'residual_stable_reuse_candidate': 65, 'category_curve_artifact_candidate': 87, 'stable_differentiation_candidate': 28, 'model_curve_artifact_candidate': 10}
- stable reuse candidates:
  - long_passive / nested_condition: raw=0.9649, model=0.8367, category=0.8118, group=0.8367, label=residual_stable_reuse_candidate
  - nested_contrast / nested_passive: raw=0.9709, model=0.8543, category=0.8032, group=0.8543, label=residual_stable_reuse_candidate
  - center_embedding / nested_contrast: raw=0.9690, model=0.8447, category=0.7766, group=0.8447, label=residual_stable_reuse_candidate
  - center_embedding / nested_condition: raw=0.9738, model=0.8806, category=0.7711, group=0.8806, label=residual_stable_reuse_candidate
  - nested_condition / nested_contrast: raw=0.9455, model=0.7572, category=0.7702, group=0.7572, label=residual_stable_reuse_candidate
- stable differentiation candidates:
  - complement_clause / they_coref: raw=0.5700, model=-0.6381, category=-0.3789, group=-0.6381, label=stable_differentiation_candidate
  - complement_clause / concise: raw=0.6104, model=-0.3662, category=-0.2878, group=-0.3662, label=stable_differentiation_candidate
  - complement_clause / en_zh_word: raw=0.6366, model=-0.5023, category=-0.1275, group=-0.5023, label=stable_differentiation_candidate
  - complement_clause / en_zh_phrase: raw=0.6372, model=-0.5400, category=-0.0633, group=-0.5400, label=stable_differentiation_candidate
  - casual / causal: raw=0.6460, model=-0.4031, category=-0.2965, group=-0.4031, label=stable_differentiation_candidate

