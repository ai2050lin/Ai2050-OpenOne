# Phase 295 Global Functional Contract Mapping Report
## Inputs
- phase290_dir: `results/gpt5_phase296_expanded_function_pilot`
- phase291_dir: `results/gpt5_phase297_expanded_block_pilot`
- phase293_dir: `results/gpt5_phase293_naturalness`
- phase294_dir: `results/gpt5_phase298_expanded_dynamic_normal`

## qwen3
- meta: {'phase290_rows': 3780, 'phase291_rows': 6300, 'phase293_event_rows': 784, 'phase294_rows': 43920, 'subtypes': 45}
- features: min=346, max=368
- combined similarity: mean=0.8653, min=0.3775, max=0.9974
- same category mean=0.8929, cross category mean=0.8618
- top reuse candidates:
  - future_will / perfect: combined=0.9974, layer=0.9967, block=0.9973, dynamic=0.9944, natural=0.0000
  - future_will / progressive: combined=0.9961, layer=0.9933, block=0.9949, dynamic=0.9941, natural=0.0000
  - perfect / progressive: combined=0.9948, layer=0.9949, block=0.9940, dynamic=0.9894, natural=0.0000
  - and_or / morphological_neg: combined=0.9926, layer=0.9872, block=0.9806, dynamic=0.9929, natural=1.0000
  - causal / contrast: combined=0.9899, layer=0.9699, block=0.9930, dynamic=0.9973, natural=0.9950
- bottom differentiation candidates:
  - concise / syntactic_do_not: combined=0.3775, layer=0.4585, block=0.6265, dynamic=0.6545, natural=0.0000
  - by_phrase / concise: combined=0.4109, layer=0.6077, block=0.5873, dynamic=0.6825, natural=0.0000
  - center_embedding / concise: combined=0.4166, layer=0.5364, block=0.6526, dynamic=0.6431, natural=0.0000
  - concise / lexical_not_adj: combined=0.4309, layer=0.6483, block=0.6684, dynamic=0.6747, natural=0.0000
  - concise / nested_condition: combined=0.4408, layer=0.6554, block=0.7229, dynamic=0.6641, natural=0.0000

## glm4
- meta: {'phase290_rows': 3780, 'phase291_rows': 7560, 'phase293_event_rows': 1468, 'phase294_rows': 43920, 'subtypes': 45}
- features: min=366, max=388
- combined similarity: mean=0.9227, min=0.7516, max=0.9990
- same category mean=0.9477, cross category mean=0.9196
- top reuse candidates:
  - future_will / perfect: combined=0.9990, layer=0.9966, block=0.9988, dynamic=0.9986, natural=0.0000
  - perfect / progressive: combined=0.9989, layer=0.9972, block=0.9973, dynamic=0.9991, natural=0.0000
  - future_will / progressive: combined=0.9988, layer=0.9992, block=0.9959, dynamic=0.9986, natural=0.0000
  - conditional / contrast: combined=0.9964, layer=0.9967, block=0.9966, dynamic=0.9970, natural=0.9962
  - long_passive / nested_passive: combined=0.9963, layer=0.9886, block=0.9959, dynamic=0.9941, natural=0.0000
- bottom differentiation candidates:
  - it_coref / scope_quantifier: combined=0.7516, layer=0.7019, block=0.8023, dynamic=0.8576, natural=0.0000
  - it_coref / lexical_not_adj: combined=0.7572, layer=0.6913, block=0.8173, dynamic=0.8264, natural=0.0000
  - it_coref / never: combined=0.7574, layer=0.6427, block=0.8149, dynamic=0.8681, natural=0.0000
  - it_coref / relative_clause: combined=0.7575, layer=0.7059, block=0.8340, dynamic=0.8655, natural=0.0000
  - scope_quantifier / they_coref: combined=0.7610, layer=0.7624, block=0.8438, dynamic=0.8840, natural=0.0000

## deepseek7b
- meta: {'phase290_rows': 3780, 'phase291_rows': 6300, 'phase293_event_rows': 1078, 'phase294_rows': 39040, 'subtypes': 45}
- features: min=334, max=356
- combined similarity: mean=0.8518, min=0.5700, max=0.9819
- same category mean=0.8763, cross category mean=0.8487
- top reuse candidates:
  - future_will / perfect: combined=0.9819, layer=0.9934, block=0.9843, dynamic=0.9441, natural=0.0000
  - by_phrase / get_passive: combined=0.9814, layer=0.9766, block=0.9491, dynamic=0.9912, natural=0.9799
  - formal / negated_condition: combined=0.9809, layer=0.9746, block=0.9724, dynamic=0.9662, natural=0.0000
  - and_or / morphological_neg: combined=0.9775, layer=0.9625, block=0.9650, dynamic=0.9835, natural=0.9914
  - by_phrase / relative_clause: combined=0.9741, layer=0.9700, block=0.9466, dynamic=0.9712, natural=0.9849
- bottom differentiation candidates:
  - complement_clause / they_coref: combined=0.5700, layer=0.7173, block=0.7679, dynamic=0.7618, natural=0.0000
  - complement_clause / concise: combined=0.6104, layer=0.5206, block=0.8913, dynamic=0.7340, natural=0.0000
  - complement_clause / en_zh_word: combined=0.6366, layer=0.8714, block=0.9066, dynamic=0.8652, natural=0.0000
  - complement_clause / en_zh_phrase: combined=0.6372, layer=0.8161, block=0.8194, dynamic=0.7382, natural=0.0000
  - casual / causal: combined=0.6460, layer=0.7150, block=0.3309, dynamic=0.7314, natural=0.0000
