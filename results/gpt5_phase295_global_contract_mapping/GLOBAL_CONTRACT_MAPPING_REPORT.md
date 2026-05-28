# Phase 295 Global Functional Contract Mapping Report
## Inputs
- phase290_dir: `results/gpt5_phase290_contract_break_full`
- phase291_dir: `results/gpt5_phase291_block_contract_full`
- phase293_dir: `results/gpt5_phase293_naturalness`
- phase294_dir: `results/gpt5_phase294b_dynamic_recompute_full`

## qwen3
- meta: {'phase290_rows': 28512, 'phase291_rows': 15840, 'phase293_event_rows': 784, 'phase294_rows': 13680, 'subtypes': 19}
- features: min=504, max=504
- combined similarity: mean=0.9505, min=0.8731, max=0.9953
- same category mean=0.9693, cross category mean=0.9453
- top reuse candidates:
  - causal / contrast: combined=0.9953, layer=0.9930, block=0.9937, dynamic=0.9899, natural=0.9950
  - contrast / no_agent: combined=0.9950, layer=0.9839, block=0.9908, dynamic=0.9906, natural=0.9993
  - contrast / inference: combined=0.9949, layer=0.9870, block=0.9896, dynamic=0.9904, natural=0.9960
  - causal / inference: combined=0.9944, layer=0.9876, block=0.9932, dynamic=0.9825, natural=0.9998
  - causal / no_agent: combined=0.9938, layer=0.9851, block=0.9922, dynamic=0.9933, natural=0.9914
- bottom differentiation candidates:
  - morphological_neg / syntactic_do_not: combined=0.8731, layer=0.8298, block=0.8875, dynamic=0.8490, natural=0.9417
  - get_passive / morphological_neg: combined=0.8763, layer=0.8393, block=0.9133, dynamic=0.8552, natural=0.9035
  - and_or / syntactic_do_not: combined=0.8794, layer=0.8568, block=0.8947, dynamic=0.8607, natural=0.9415
  - and_or / get_passive: combined=0.8825, layer=0.8636, block=0.9160, dynamic=0.8641, natural=0.9034
  - causal / syntactic_do_not: combined=0.8843, layer=0.8569, block=0.9133, dynamic=0.8627, natural=0.9807

## glm4
- meta: {'phase290_rows': 31680, 'phase291_rows': 19008, 'phase293_event_rows': 1468, 'phase294_rows': 13680, 'subtypes': 19}
- features: min=544, max=544
- combined similarity: mean=0.9675, min=0.9112, max=0.9980
- same category mean=0.9812, cross category mean=0.9638
- top reuse candidates:
  - by_phrase / dative_passive: combined=0.9980, layer=0.9955, block=0.9965, dynamic=0.9956, natural=0.9998
  - causal / inference: combined=0.9979, layer=0.9950, block=0.9980, dynamic=0.9933, natural=0.9979
  - dative_passive / get_passive: combined=0.9971, layer=0.9940, block=0.9949, dynamic=0.9923, natural=0.9997
  - by_phrase / get_passive: combined=0.9969, layer=0.9946, block=0.9943, dynamic=0.9939, natural=0.9989
  - conditional / contrast: combined=0.9962, layer=0.9929, block=0.9970, dynamic=0.9959, natural=0.9962
- bottom differentiation candidates:
  - and_or / syntactic_do_not: combined=0.9112, layer=0.8359, block=0.9700, dynamic=0.9094, natural=0.9705
  - and_or / lexical_not_adj: combined=0.9159, layer=0.8292, block=0.9574, dynamic=0.9096, natural=0.9667
  - causal / syntactic_do_not: combined=0.9188, layer=0.8271, block=0.9737, dynamic=0.9019, natural=0.9368
  - and_or / scope_quantifier: combined=0.9201, layer=0.8782, block=0.9541, dynamic=0.9111, natural=0.9711
  - and_or / relative_clause: combined=0.9215, layer=0.8671, block=0.9725, dynamic=0.9397, natural=0.9545

## deepseek7b
- meta: {'phase290_rows': 25344, 'phase291_rows': 15840, 'phase293_event_rows': 1078, 'phase294_rows': 12160, 'subtypes': 19}
- features: min=472, max=472
- combined similarity: mean=0.8913, min=0.6143, max=0.9900
- same category mean=0.9148, cross category mean=0.8848
- top reuse candidates:
  - by_phrase / get_passive: combined=0.9900, layer=0.9865, block=0.9869, dynamic=0.9832, natural=0.9799
  - by_phrase / possessive_chain: combined=0.9832, layer=0.9882, block=0.9825, dynamic=0.9406, natural=0.9874
  - possessive_chain / relative_clause: combined=0.9832, layer=0.9687, block=0.9678, dynamic=0.9719, natural=0.9820
  - by_phrase / relative_clause: combined=0.9814, layer=0.9649, block=0.9806, dynamic=0.9514, natural=0.9849
  - get_passive / possessive_chain: combined=0.9814, layer=0.9955, block=0.9910, dynamic=0.9579, natural=0.9386
- bottom differentiation candidates:
  - complement_clause / no_agent: combined=0.6143, layer=0.8062, block=0.8647, dynamic=0.5793, natural=0.9377
  - complement_clause / morphological_neg: combined=0.6205, layer=0.8404, block=0.6968, dynamic=0.5597, natural=0.7675
  - complement_clause / existential_no: combined=0.6440, layer=0.8052, block=0.8682, dynamic=0.6314, natural=0.8150
  - causal / complement_clause: combined=0.6492, layer=0.8080, block=0.9219, dynamic=0.5789, natural=0.7500
  - and_or / complement_clause: combined=0.6543, layer=0.8496, block=0.9314, dynamic=0.5758, natural=0.7650
