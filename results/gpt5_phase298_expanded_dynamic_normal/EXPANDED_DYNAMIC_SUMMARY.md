# Phase 298 Expanded Dynamic Recompute Summary

## qwen3
- complete: True
- pairs: 244
- rows: 43920
- nonfinite_rows: 0
- target_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8]
- best_patch_counts: {'resid_out': 25, 'attn_out': 4, 'resid_in': 10, 'mlp_out': 6}
- best_layer_counts: {0: 21, 1: 1, 2: 1, 3: 3, 4: 1, 6: 7, 7: 5, 8: 6}
- best_by_patch_type:
  - attn_out: layer=0, progress=0.333718
  - mlp_out: layer=0, progress=0.386442
  - resid_in: layer=4, progress=0.488311
  - resid_out: layer=3, progress=0.488309
- top_best_subtypes:
  - en_zh_phrase (translation): resid_out L0 progress=1.377287
  - en_fr_phrase (translation): resid_out L1 progress=1.203791
  - he_coref (coreference): mlp_out L0 progress=1.180285
  - en_fr_word (translation): resid_out L7 progress=1.052615
  - causal (logical): attn_out L0 progress=1.034798
- bottom_best_subtypes:
  - it_coref (coreference): resid_out L7 progress=0.396908
  - double_relative (recursive): resid_out L4 progress=0.473765
  - deep_complement (recursive): resid_out L8 progress=0.549443
  - nested_condition (logical): mlp_out L3 progress=0.615510
  - perfect (tense): attn_out L0 progress=0.623038

## glm4
- complete: True
- pairs: 244
- rows: 43920
- nonfinite_rows: 0
- target_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8]
- best_patch_counts: {'resid_in': 14, 'resid_out': 29, 'mlp_out': 2}
- best_layer_counts: {0: 19, 1: 4, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2, 8: 14}
- best_by_patch_type:
  - attn_out: layer=1, progress=0.081681
  - mlp_out: layer=0, progress=0.490105
  - resid_in: layer=2, progress=0.590795
  - resid_out: layer=1, progress=0.590795
- top_best_subtypes:
  - dative_passive (passive): resid_out L1 progress=1.122602
  - get_passive (passive): resid_in L0 progress=1.062892
  - long_passive (passive): resid_out L1 progress=1.047683
  - by_phrase (passive): resid_in L0 progress=1.039192
  - complement_clause (recursive): resid_out L6 progress=1.027453
- bottom_best_subtypes:
  - nested_condition (logical): resid_out L8 progress=0.866974
  - double_relative (recursive): resid_out L8 progress=0.869599
  - pp_chain (recursive): resid_out L5 progress=0.874349
  - it_coref (coreference): resid_out L8 progress=0.877729
  - possessive_chain (recursive): resid_out L8 progress=0.894064

## deepseek7b
- complete: True
- pairs: 244
- rows: 39040
- nonfinite_rows: 0
- target_layers: [20, 21, 22, 23, 24, 25, 26, 27]
- best_patch_counts: {'resid_in': 9, 'resid_out': 32, 'attn_out': 3, 'mlp_out': 1}
- best_layer_counts: {20: 9, 22: 1, 23: 1, 24: 1, 26: 1, 27: 32}
- best_by_patch_type:
  - attn_out: layer=27, progress=0.288754
  - mlp_out: layer=27, progress=0.250668
  - resid_in: layer=21, progress=0.278892
  - resid_out: layer=27, progress=0.483323
- top_best_subtypes:
  - concise (style): resid_out L24 progress=1.109055
  - existential_no (negation): resid_out L23 progress=1.097377
  - deep_complement (recursive): attn_out L27 progress=1.083397
  - deictic_switch (coreference): mlp_out L26 progress=1.056671
  - pp_chain (recursive): attn_out L27 progress=1.048624
- bottom_best_subtypes:
  - double_relative (recursive): resid_out L27 progress=0.999981
  - nested_contrast (logical): resid_out L27 progress=0.999984
  - long_passive (passive): resid_out L27 progress=0.999985
  - he_coref (coreference): resid_in L20 progress=0.999985
  - en_zh_word (translation): resid_out L27 progress=0.999986

