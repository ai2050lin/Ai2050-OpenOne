# Phase510 Surface-format Stepwise Probe Summary

## qwen3

L=36, categories=fruit,action,emotion, train=20, test=10, templates=3, steps=3

| category | support axis | release axis | condition | hit Δ | s1 cat-comp Δ | s2 cat-comp Δ | s3 cat-comp Δ | s1 cat-punct Δ | s1 cat-generic Δ | s1 top category Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fruit | L27 combo1 -0.596 | L27 svd2 +1.224 | remove_support | -0.100 | -0.562 | -0.281 | -0.344 | -0.660 | +0.047 | -0.067 |
| fruit | L27 combo1 -0.596 | L27 svd2 +1.224 | add_support | +0.133 | +0.562 | +0.312 | +0.562 | +0.711 | +0.000 | +0.000 |
| fruit | L27 combo1 -0.596 | L27 svd2 +1.224 | remove_release | +0.133 | +1.469 | +0.938 | +1.344 | +0.992 | +1.766 | +0.100 |
| fruit | L27 combo1 -0.596 | L27 svd2 +1.224 | add_release | -0.200 | -1.219 | -0.719 | -1.000 | -1.017 | -1.406 | -0.067 |
| action | L18 combo2 -0.089 | L33 svd3 +0.723 | remove_support | +0.000 | -0.141 | -0.102 | -0.031 | +0.016 | -0.047 | +0.000 |
| action | L18 combo2 -0.089 | L33 svd3 +0.723 | add_support | +0.000 | +0.078 | +0.094 | +0.086 | +0.016 | +0.039 | +0.000 |
| action | L18 combo2 -0.089 | L33 svd3 +0.723 | remove_release | +0.000 | +0.742 | +0.973 | +0.672 | -1.375 | -0.922 | +0.000 |
| action | L18 combo2 -0.089 | L33 svd3 +0.723 | add_release | +0.000 | -0.078 | -0.203 | -0.078 | +0.375 | +0.688 | +0.000 |
| emotion | L27 svd1 -0.970 | L33 combo1 +0.410 | remove_support | -0.067 | -1.227 | -0.772 | -1.351 | -3.047 | -2.141 | +0.000 |
| emotion | L27 svd1 -0.970 | L33 combo1 +0.410 | add_support | +0.033 | +1.242 | +0.778 | +1.649 | +2.066 | +1.464 | +0.000 |
| emotion | L27 svd1 -0.970 | L33 combo1 +0.410 | remove_release | +0.033 | -0.246 | +0.431 | -1.210 | -2.422 | +0.861 | +0.000 |
| emotion | L27 svd1 -0.970 | L33 combo1 +0.410 | add_release | +0.000 | -0.018 | -0.257 | -0.163 | +1.445 | -1.266 | +0.000 |

## glm4

L=40, categories=emotion,color,fruit, train=20, test=10, templates=3, steps=3

| category | support axis | release axis | condition | hit Δ | s1 cat-comp Δ | s2 cat-comp Δ | s3 cat-comp Δ | s1 cat-punct Δ | s1 cat-generic Δ | s1 top category Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| emotion | L37 svd1 -0.468 | L37 combo0 +1.502 | remove_support | -0.067 | -0.109 | +0.225 | -0.992 | -0.969 | -2.805 | +0.000 |
| emotion | L37 svd1 -0.468 | L37 combo0 +1.502 | add_support | +0.000 | +0.055 | -0.176 | +0.555 | +0.844 | +1.748 | +0.000 |
| emotion | L37 svd1 -0.468 | L37 combo0 +1.502 | remove_release | +0.067 | +1.379 | +2.135 | +1.664 | +0.125 | +2.336 | +0.000 |
| emotion | L37 svd1 -0.468 | L37 combo0 +1.502 | add_release | -0.133 | -1.250 | -2.014 | -2.031 | -0.844 | -2.742 | +0.000 |
| color | L37 svd3 -0.579 | L37 svd0 +0.360 | remove_support | -0.033 | -1.115 | -0.475 | -0.539 | +0.141 | +1.226 | -0.033 |
| color | L37 svd3 -0.579 | L37 svd0 +0.360 | add_support | -0.067 | +0.447 | +0.588 | +0.895 | -1.031 | -2.181 | -0.033 |
| color | L37 svd3 -0.579 | L37 svd0 +0.360 | remove_release | -0.033 | +0.350 | +0.320 | +0.672 | -0.469 | +0.233 | -0.033 |
| color | L37 svd3 -0.579 | L37 svd0 +0.360 | add_release | +0.033 | -0.101 | -0.182 | -0.195 | +0.484 | -0.224 | +0.033 |
| fruit | L30 svd0 -0.775 | L37 combo0 +0.550 | remove_support | -0.067 | -0.484 | -1.406 | -0.859 | -1.188 | -0.566 | +0.000 |
| fruit | L30 svd0 -0.775 | L37 combo0 +0.550 | add_support | +0.067 | +0.961 | +0.719 | +1.078 | +1.375 | +0.766 | +0.000 |
| fruit | L30 svd0 -0.775 | L37 combo0 +0.550 | remove_release | +0.000 | +0.039 | -0.031 | -0.281 | +0.297 | -0.765 | +0.000 |
| fruit | L30 svd0 -0.775 | L37 combo0 +0.550 | add_release | +0.033 | +0.125 | -0.281 | +0.406 | -0.234 | +0.617 | +0.000 |

## deepseek7b

L=28, categories=action,fruit,color, train=20, test=10, templates=3, steps=3

| category | support axis | release axis | condition | hit Δ | s1 cat-comp Δ | s2 cat-comp Δ | s3 cat-comp Δ | s1 cat-punct Δ | s1 cat-generic Δ | s1 top category Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| action | L25 svd2 -0.316 | L25 svd3 +0.258 | remove_support | +0.000 | -0.334 | -0.234 | -0.145 | +1.062 | +0.000 | +0.000 |
| action | L25 svd2 -0.316 | L25 svd3 +0.258 | add_support | +0.000 | +0.135 | +0.000 | +0.338 | -1.125 | -0.016 | +0.000 |
| action | L25 svd2 -0.316 | L25 svd3 +0.258 | remove_release | +0.000 | +0.256 | +0.414 | +0.756 | -0.094 | -0.359 | +0.000 |
| action | L25 svd2 -0.316 | L25 svd3 +0.258 | add_release | +0.000 | -0.451 | -0.453 | -0.172 | -0.188 | -0.047 | +0.000 |
| fruit | L21 svd1 -0.352 | L25 svd2 +0.430 | remove_support | -0.033 | -0.203 | -0.222 | -0.716 | -0.383 | -0.141 | +0.000 |
| fruit | L21 svd1 -0.352 | L25 svd2 +0.430 | add_support | +0.100 | +0.266 | +0.426 | +0.204 | +0.281 | +0.055 | +0.000 |
| fruit | L21 svd1 -0.352 | L25 svd2 +0.430 | remove_release | +0.033 | +0.266 | +0.410 | +0.561 | +0.906 | +0.469 | +0.000 |
| fruit | L21 svd1 -0.352 | L25 svd2 +0.430 | add_release | +0.000 | -0.320 | -0.319 | -0.954 | -0.961 | -0.594 | +0.000 |
| color | L25 svd2 -0.146 | L25 svd1 +0.172 | remove_support | +0.000 | -0.312 | -0.188 | -0.273 | -0.406 | -0.297 | +0.000 |
| color | L25 svd2 -0.146 | L25 svd1 +0.172 | add_support | +0.000 | +0.250 | +0.133 | +0.078 | +0.484 | +0.336 | +0.000 |
| color | L25 svd2 -0.146 | L25 svd1 +0.172 | remove_release | +0.000 | +0.172 | -0.031 | -0.094 | +0.094 | +0.883 | +0.000 |
| color | L25 svd2 -0.146 | L25 svd1 +0.172 | add_release | +0.000 | -0.070 | +0.062 | +0.195 | +0.312 | -0.656 | +0.000 |

## Cross-model Compact

| model | condition | hit Δ | s1 cat-comp Δ | s2 cat-comp Δ | s3 cat-comp Δ | s1 cat-punct Δ | s1 cat-generic Δ | s1 top category Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | remove_support | -0.0556 | -0.6432 | -0.3851 | -0.5752 | -1.2305 | -0.7135 | -0.0222 |
| qwen3 | add_support | +0.0556 | +0.6276 | +0.3949 | +0.7660 | +0.9310 | +0.5010 | +0.0000 |
| qwen3 | remove_release | +0.0556 | +0.6549 | +0.7803 | +0.2686 | -0.9349 | +0.5684 | +0.0333 |
| qwen3 | add_release | -0.0667 | -0.4382 | -0.3929 | -0.4137 | +0.2679 | -0.6615 | -0.0222 |
| glm4 | remove_support | -0.0556 | -0.5697 | -0.5521 | -0.7969 | -0.6719 | -0.7152 | -0.0111 |
| glm4 | add_support | -0.0000 | +0.4876 | +0.3770 | +0.8424 | +0.3958 | +0.1110 | -0.0111 |
| glm4 | remove_release | +0.0111 | +0.5892 | +0.8077 | +0.6849 | -0.0156 | +0.6012 | -0.0111 |
| glm4 | add_release | -0.0222 | -0.4085 | -0.8255 | -0.6068 | -0.1979 | -0.7829 | +0.0111 |
| deepseek7b | remove_support | -0.0111 | -0.2832 | -0.2145 | -0.3780 | +0.0911 | -0.1458 | +0.0000 |
| deepseek7b | add_support | +0.0333 | +0.2168 | +0.1862 | +0.2065 | -0.1198 | +0.1250 | +0.0000 |
| deepseek7b | remove_release | +0.0111 | +0.2311 | +0.2643 | +0.4078 | +0.3021 | +0.3307 | +0.0000 |
| deepseek7b | add_release | +0.0000 | -0.2806 | -0.2365 | -0.3103 | -0.2786 | -0.4323 | +0.0000 |

