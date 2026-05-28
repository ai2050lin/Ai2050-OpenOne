# Phase 295 Global Functional Contract Mapping Report
## Inputs
- phase290_dir: `results/gpt5_phase296_expanded_function_pilot`
- phase291_dir: `results/gpt5_phase297_expanded_block_pilot`
- phase293_dir: `results/gpt5_phase293_naturalness`
- phase294_dir: `results/gpt5_phase294b_dynamic_recompute_full`

## qwen3
- meta: {'phase290_rows': 3780, 'phase291_rows': 6300, 'phase293_event_rows': 784, 'phase294_rows': 13680, 'subtypes': 45}
- features: min=210, max=368
- combined similarity: mean=0.8022, min=0.2757, max=0.9983
- same category mean=0.8550, cross category mean=0.7955
- top reuse candidates:
  - future_will / perfect: combined=0.9983, layer=0.9967, block=0.9973, dynamic=0.0000, natural=0.0000
  - future_will / progressive: combined=0.9965, layer=0.9933, block=0.9949, dynamic=0.0000, natural=0.0000
  - perfect / progressive: combined=0.9963, layer=0.9949, block=0.9940, dynamic=0.0000, natural=0.0000
  - and_or / morphological_neg: combined=0.9931, layer=0.9872, block=0.9806, dynamic=0.9952, natural=1.0000
  - never / scope_quantifier: combined=0.9915, layer=0.9880, block=0.9903, dynamic=0.9803, natural=0.9990
- bottom differentiation candidates:
  - concise / syntactic_do_not: combined=0.2757, layer=0.4585, block=0.6265, dynamic=0.0000, natural=0.0000
  - by_phrase / concise: combined=0.3053, layer=0.6077, block=0.5873, dynamic=0.0000, natural=0.0000
  - concise / lexical_not_adj: combined=0.3266, layer=0.6483, block=0.6684, dynamic=0.0000, natural=0.0000
  - concise / never: combined=0.3347, layer=0.6959, block=0.6129, dynamic=0.0000, natural=0.0000
  - concise / scope_quantifier: combined=0.3464, layer=0.6965, block=0.6553, dynamic=0.0000, natural=0.0000

## glm4
- meta: {'phase290_rows': 3780, 'phase291_rows': 7560, 'phase293_event_rows': 1468, 'phase294_rows': 13680, 'subtypes': 45}
- features: min=230, max=388
- combined similarity: mean=0.8518, min=0.6245, max=0.9989
- same category mean=0.9017, cross category mean=0.8454
- top reuse candidates:
  - future_will / perfect: combined=0.9989, layer=0.9966, block=0.9988, dynamic=0.0000, natural=0.0000
  - future_will / progressive: combined=0.9986, layer=0.9992, block=0.9959, dynamic=0.0000, natural=0.0000
  - perfect / progressive: combined=0.9986, layer=0.9972, block=0.9973, dynamic=0.0000, natural=0.0000
  - double_relative / perfect: combined=0.9968, layer=0.9958, block=0.9919, dynamic=0.0000, natural=0.0000
  - double_relative / future_will: combined=0.9967, layer=0.9927, block=0.9939, dynamic=0.0000, natural=0.0000
- bottom differentiation candidates:
  - it_coref / never: combined=0.6245, layer=0.6427, block=0.8149, dynamic=0.0000, natural=0.0000
  - it_coref / scope_quantifier: combined=0.6259, layer=0.7019, block=0.8023, dynamic=0.0000, natural=0.0000
  - it_coref / relative_clause: combined=0.6327, layer=0.7059, block=0.8340, dynamic=0.0000, natural=0.0000
  - casual / never: combined=0.6344, layer=0.6394, block=0.8010, dynamic=0.0000, natural=0.0000
  - casual / scope_quantifier: combined=0.6347, layer=0.6923, block=0.7893, dynamic=0.0000, natural=0.0000

## deepseek7b
- meta: {'phase290_rows': 3780, 'phase291_rows': 6300, 'phase293_event_rows': 1078, 'phase294_rows': 12160, 'subtypes': 45}
- features: min=210, max=356
- combined similarity: mean=0.7962, min=0.5235, max=0.9917
- same category mean=0.8412, cross category mean=0.7906
- top reuse candidates:
  - future_will / perfect: combined=0.9917, layer=0.9934, block=0.9843, dynamic=0.0000, natural=0.0000
  - center_embedding / nested_condition: combined=0.9882, layer=0.9886, block=0.9842, dynamic=0.0000, natural=0.0000
  - nested_contrast / nested_passive: combined=0.9834, layer=0.9747, block=0.9749, dynamic=0.0000, natural=0.0000
  - formal / negated_condition: combined=0.9832, layer=0.9746, block=0.9724, dynamic=0.0000, natural=0.0000
  - and_or / morphological_neg: combined=0.9821, layer=0.9625, block=0.9650, dynamic=0.9875, natural=0.9914
- bottom differentiation candidates:
  - complement_clause / they_coref: combined=0.5235, layer=0.7173, block=0.7679, dynamic=0.0000, natural=0.0000
  - casual / causal: combined=0.5236, layer=0.7150, block=0.3309, dynamic=0.0000, natural=0.0000
  - casual / no_agent: combined=0.5468, layer=0.7202, block=0.4718, dynamic=0.0000, natural=0.0000
  - causal / complement_clause: combined=0.5563, layer=0.7387, block=0.8322, dynamic=0.5789, natural=0.7500
  - complement_clause / no_agent: combined=0.5637, layer=0.7591, block=0.7951, dynamic=0.5793, natural=0.9377
