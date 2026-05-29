# Phase 300 Voice Closure Pilot Summary

## qwen3
- complete: True
- pairs/train/test: 92 / 46 / 46
- rows: 9936
- nonfinite_rows: 0
- probe_mean_accuracy: 0.923108
- best_probe: L0 resid_in acc=1.000000 margin=0.039576
- best_by_direction:
  - active_to_passive: L1 mlp_out progress=0.306731 kl=1.197655 delta=0.905774
  - passive_to_active: L2 resid_out progress=0.305248 kl=0.705311 delta=0.664987
- subtype_curve:
  - by_phrase active_to_passive: progress=0.166095 kl=0.977055
  - by_phrase passive_to_active: progress=0.045212 kl=0.980464
  - dative_passive active_to_passive: progress=0.112386 kl=0.859541
  - dative_passive passive_to_active: progress=0.056769 kl=0.977553
  - get_passive active_to_passive: progress=0.186283 kl=1.060281
  - get_passive passive_to_active: progress=0.051683 kl=0.954104
  - long_passive active_to_passive: progress=0.185860 kl=1.216805
  - long_passive passive_to_active: progress=0.041687 kl=0.918104

## glm4
- complete: True
- pairs/train/test: 92 / 46 / 46
- rows: 9936
- nonfinite_rows: 0
- probe_mean_accuracy: 0.932770
- best_probe: L0 resid_in acc=1.000000 margin=0.000298
- best_by_direction:
  - active_to_passive: L0 resid_in progress=0.487534 kl=0.597967 delta=0.863517
  - passive_to_active: L8 resid_out progress=0.012508 kl=1.005237 delta=0.105789
- subtype_curve:
  - by_phrase active_to_passive: progress=0.213462 kl=0.833887
  - by_phrase passive_to_active: progress=0.001275 kl=0.999672
  - dative_passive active_to_passive: progress=0.198536 kl=0.859857
  - dative_passive passive_to_active: progress=0.000378 kl=1.004233
  - get_passive active_to_passive: progress=0.213752 kl=0.863313
  - get_passive passive_to_active: progress=0.002016 kl=0.990902
  - long_passive active_to_passive: progress=0.186301 kl=0.874394
  - long_passive passive_to_active: progress=-0.008697 kl=0.998774

## deepseek7b
- complete: True
- pairs/train/test: 92 / 46 / 46
- rows: 8832
- nonfinite_rows: 0
- probe_mean_accuracy: 0.640399
- best_probe: L21 mlp_out acc=1.000000 margin=108.308620
- best_by_direction:
  - active_to_passive: L27 resid_out progress=0.053636 kl=0.854817 delta=0.283468
  - passive_to_active: L27 resid_out progress=0.184869 kl=0.765157 delta=0.390689
- subtype_curve:
  - by_phrase active_to_passive: progress=0.008890 kl=0.981858
  - by_phrase passive_to_active: progress=0.002664 kl=1.060324
  - dative_passive active_to_passive: progress=0.015331 kl=0.972741
  - dative_passive passive_to_active: progress=0.050772 kl=1.026681
  - get_passive active_to_passive: progress=0.026669 kl=1.006689
  - get_passive passive_to_active: progress=-0.000450 kl=0.977828
  - long_passive active_to_passive: progress=0.012186 kl=0.996154
  - long_passive passive_to_active: progress=0.095901 kl=1.049935

