# Phase 301 Passive Factor Closure Summary

## qwen3
- complete: True
- bases/train/test: 24 / 12 / 12
- rows: 13608
- nonfinite_rows: 0
- probe_best:
  - by_phrase: L0 resid_in acc=1.000000 margin=0.100888
  - role_swap: L7 mlp_out acc=0.944444 margin=0.620264
  - voice: L0 resid_in acc=1.000000 margin=0.054826
- best_by_variable_direction:
  - by_phrase:forward: L1 mlp_out progress=0.480497 kl=2.771863 delta=0.972437
  - by_phrase:reverse: L2 resid_out progress=0.522776 kl=1.089960 delta=0.941896
  - role_swap:forward: L0 resid_in progress=0.017493 kl=0.983960 delta=0.088298
  - role_swap:reverse: L1 resid_out progress=0.027947 kl=0.983883 delta=0.126692
  - voice:forward: L0 resid_in progress=0.346220 kl=1.296394 delta=0.934865
  - voice:reverse: L2 resid_out progress=0.428750 kl=0.755255 delta=0.761295
- variable_direction_curve:
  - by_phrase forward: progress=0.136004 kl=1.188874
  - by_phrase reverse: progress=0.146432 kl=1.111486
  - role_swap forward: progress=0.006346 kl=1.004198
  - role_swap reverse: progress=0.012143 kl=1.003458
  - voice forward: progress=0.183000 kl=1.036725
  - voice reverse: progress=0.087780 kl=0.981955

## glm4
- complete: True
- bases/train/test: 24 / 12 / 12
- rows: 13608
- nonfinite_rows: 0
- probe_best:
  - by_phrase: L0 resid_in acc=1.000000 margin=0.000759
  - role_swap: L5 mlp_out acc=0.972222 margin=0.000808
  - voice: L0 resid_in acc=1.000000 margin=0.000386
- best_by_variable_direction:
  - by_phrase:forward: L0 resid_in progress=0.269420 kl=0.916097 delta=0.470889
  - by_phrase:reverse: L7 resid_out progress=0.020095 kl=0.990468 delta=0.144877
  - role_swap:forward: L0 resid_out progress=0.017619 kl=0.963600 delta=0.095322
  - role_swap:reverse: L8 resid_out progress=0.030030 kl=1.000330 delta=0.172919
  - voice:forward: L0 resid_in progress=0.472474 kl=0.598427 delta=0.890664
  - voice:reverse: L8 resid_out progress=0.015532 kl=0.998028 delta=0.120513
- variable_direction_curve:
  - by_phrase forward: progress=0.108485 kl=0.933662
  - by_phrase reverse: progress=-0.005009 kl=0.991968
  - role_swap forward: progress=0.010789 kl=1.001808
  - role_swap reverse: progress=0.011120 kl=0.999127
  - voice forward: progress=0.271165 kl=0.760783
  - voice reverse: progress=-0.001104 kl=0.997715

## deepseek7b
- complete: True
- bases/train/test: 24 / 12 / 12
- rows: 12096
- nonfinite_rows: 0
- probe_best:
  - by_phrase: L21 mlp_out acc=0.958333 margin=207.724309
  - role_swap: L21 mlp_out acc=0.847222 margin=12.688938
  - voice: L20 mlp_out acc=1.000000 margin=123.004636
- best_by_variable_direction:
  - by_phrase:forward: L26 resid_out progress=0.620331 kl=1.080685 delta=0.859595
  - by_phrase:reverse: L26 resid_out progress=0.482621 kl=1.392317 delta=0.618689
  - role_swap:forward: L24 resid_in progress=0.018090 kl=0.981939 delta=0.094300
  - role_swap:reverse: L27 resid_out progress=0.015873 kl=0.985071 delta=0.127823
  - voice:forward: L20 resid_in progress=0.056489 kl=1.055646 delta=0.175124
  - voice:reverse: L27 resid_out progress=0.118703 kl=0.798672 delta=0.203346
- variable_direction_curve:
  - by_phrase forward: progress=0.220030 kl=1.243953
  - by_phrase reverse: progress=0.003117 kl=1.198228
  - role_swap forward: progress=0.008298 kl=0.980360
  - role_swap reverse: progress=-0.001965 kl=1.157389
  - voice forward: progress=0.017968 kl=1.029938
  - voice reverse: progress=-0.027237 kl=1.647137

