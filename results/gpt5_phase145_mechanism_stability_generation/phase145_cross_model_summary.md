# Phase 145 Cross-model Mechanism Stability Summary

## qwen3

families=['long', 'short', 'neutral']; splits=['front_back', 'back_front']; train/test=12/12

| category | path | kind | n | clean_rate | mean_rec | mean_release | category_argmax |
|---|---|---|---|---|---|---|---|
| container | clean_input_a | clean | 6 | 0.17 | +4.42 | +2.89 | 0.00 |
| container | clean_input_b | clean | 6 | 0.00 | +20.09 | +5.16 | 0.00 |
| container | dirty | dirty | 6 | 0.00 | +nan | +4.21 | 0.00 |
| number | clean_a | clean | 6 | 0.00 | -4.07 | +0.00 | 0.00 |
| number | clean_b | clean | 6 | 0.00 | -4.46 | +0.00 | 0.00 |
| number | dirty | dirty | 6 | 0.00 | +13.06 | +8.91 | 0.00 |
| plant | clean_attn | clean | 6 | 0.00 | -11.64 | +0.00 | 0.00 |
| plant | clean_input | clean | 6 | 0.00 | -10.06 | +2.72 | 0.00 |
| plant | dirty | dirty | 6 | 0.00 | +4.00 | +6.30 | 0.00 |
| time | clean_mlp | clean | 6 | 0.00 | -12.15 | +0.00 | 0.00 |
| time | dirty | dirty | 6 | 0.00 | -5.64 | +1.03 | 0.00 |
| time | weak_last | clean | 6 | 0.00 | +34.47 | +8.93 | 0.00 |

## glm4

families=['long', 'short', 'neutral']; splits=['front_back', 'back_front']; train/test=12/12

| category | path | kind | n | clean_rate | mean_rec | mean_release | category_argmax |
|---|---|---|---|---|---|---|---|
| container | clean_input_a | clean | 6 | 0.00 | +27.62 | +1.15 | 0.00 |
| container | clean_input_b | clean | 6 | 0.00 | +30.14 | +1.28 | 0.00 |
| container | dirty | dirty | 6 | 0.00 | +41.77 | +2.05 | 0.00 |
| number | clean_a | clean | 6 | 0.00 | -1.21 | +0.34 | 0.00 |
| number | clean_b | clean | 6 | 0.00 | -1.41 | +0.38 | 0.00 |
| number | dirty | dirty | 6 | 0.00 | -6.63 | +1.28 | 0.00 |
| plant | clean_attn | clean | 6 | 0.17 | -0.56 | +0.60 | 0.00 |
| plant | clean_input | clean | 6 | 0.00 | +1.83 | +1.04 | 0.00 |
| plant | dirty | dirty | 6 | 0.00 | +2.08 | +1.55 | 0.00 |
| time | clean_mlp | clean | 6 | 0.00 | -0.52 | +1.08 | 0.00 |
| time | dirty | dirty | 6 | 0.00 | -1.63 | +1.99 | 0.00 |
| time | weak_last | clean | 6 | 0.00 | -4.11 | +0.65 | 0.00 |

## deepseek7b

families=['long', 'short', 'neutral']; splits=['front_back', 'back_front']; train/test=12/12

| category | path | kind | n | clean_rate | mean_rec | mean_release | category_argmax |
|---|---|---|---|---|---|---|---|
| container | clean_input_a | clean | 6 | 0.33 | +1.11 | +0.69 | 0.00 |
| container | clean_input_b | clean | 6 | 0.17 | +1.41 | +1.15 | 0.00 |
| container | dirty | dirty | 6 | 0.00 | +1.05 | +1.11 | 0.04 |
| number | clean_a | clean | 6 | 0.17 | +0.32 | +0.00 | 0.00 |
| number | clean_b | clean | 6 | 0.17 | +0.37 | +0.03 | 0.00 |
| number | dirty | dirty | 6 | 0.17 | +1.06 | +1.53 | 0.00 |
| plant | clean_attn | clean | 6 | 0.17 | +0.41 | +0.04 | 0.01 |
| plant | clean_input | clean | 6 | 0.50 | +0.63 | +0.18 | 0.00 |
| plant | dirty | dirty | 6 | 0.00 | +1.17 | +1.81 | 0.00 |
| time | clean_mlp | clean | 6 | 0.00 | +0.16 | +0.36 | 0.02 |
| time | dirty | dirty | 6 | 0.00 | +0.66 | +0.97 | 0.05 |
| time | weak_last | clean | 6 | 0.00 | +0.18 | +0.38 | 0.14 |

