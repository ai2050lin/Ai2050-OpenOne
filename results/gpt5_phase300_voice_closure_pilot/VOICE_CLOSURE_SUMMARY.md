# Phase 300 Voice Closure Pilot Summary

## qwen3
- complete: True
- pairs/train/test: 92 / 46 / 46
- rows: 9936
- nonfinite_rows: 0
- probe_mean_accuracy: 0.884461
- best_probe: L0 resid_in acc=1.000000 margin=0.038125
- best_by_direction:
  - active_to_passive: L1 mlp_out progress=0.338931 kl=1.440244 delta=0.940350
  - passive_to_active: L1 resid_out progress=0.259063 kl=0.730867 delta=0.656921
- subtype_curve:
  - get_passive active_to_passive: progress=0.149125 kl=1.017736
  - get_passive passive_to_active: progress=0.060890 kl=0.943842
  - long_passive active_to_passive: progress=0.204198 kl=1.174004
  - long_passive passive_to_active: progress=0.051512 kl=0.924667

## glm4
- complete: True
- pairs/train/test: 92 / 46 / 46
- rows: 9936
- nonfinite_rows: 0
- probe_mean_accuracy: 0.894525
- best_probe: L0 resid_in acc=1.000000 margin=0.000290
- best_by_direction:
  - active_to_passive: L0 resid_in progress=0.519231 kl=0.568755 delta=0.895013
  - passive_to_active: L8 resid_out progress=0.008440 kl=0.986994 delta=0.109372
- subtype_curve:
  - get_passive active_to_passive: progress=0.207318 kl=0.845267
  - get_passive passive_to_active: progress=0.002790 kl=0.986730
  - long_passive active_to_passive: progress=0.196754 kl=0.870809
  - long_passive passive_to_active: progress=-0.008807 kl=0.994658

## deepseek7b
- complete: True
- pairs/train/test: 92 / 46 / 46
- rows: 8832
- nonfinite_rows: 0
- probe_mean_accuracy: 0.612772
- best_probe: L21 mlp_out acc=1.000000 margin=98.304702
- best_by_direction:
  - active_to_passive: L22 mlp_out progress=0.060151 kl=0.944483 delta=0.141649
  - passive_to_active: L26 mlp_out progress=0.245558 kl=0.838117 delta=0.431108
- subtype_curve:
  - get_passive active_to_passive: progress=0.030126 kl=0.996648
  - get_passive passive_to_active: progress=-0.138458 kl=1.251563
  - long_passive active_to_passive: progress=0.020518 kl=1.020142
  - long_passive passive_to_active: progress=0.055611 kl=1.091733

