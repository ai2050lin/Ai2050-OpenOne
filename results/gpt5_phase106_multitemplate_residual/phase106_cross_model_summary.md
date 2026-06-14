# Phase 106 Cross-Model Multi-template Residual Summary

## Global Objective Results
| model | position | basis | top1 layer/count | best mean margin | best boundary |
|---|---|---|---:|---:|---:|
| qwen3 | answer_last | raw | L36 21/32 | L32 0.718 | L35 155.255 |
| qwen3 | answer_last | template_residual | L0 32/32 | L33 7.587 | L35 155.255 |
| qwen3 | object_last | raw | L0 18/32 | L0 -0.000 | L35 119.261 |
| qwen3 | object_last | template_residual | L13 22/32 | L32 0.946 | L35 119.261 |
| glm4 | answer_last | raw | L40 25/32 | L0 -0.001 | L18 2.644 |
| glm4 | answer_last | template_residual | L19 32/32 | L0 -0.000 | L18 2.644 |
| glm4 | object_last | raw | L24 24/32 | L0 -0.000 | L19 70.176 |
| glm4 | object_last | template_residual | L20 32/32 | L0 -0.000 | L19 70.176 |
| deepseek7b | answer_last | raw | L28 9/32 | L0 -0.017 | L27 263.246 |
| deepseek7b | answer_last | template_residual | L0 32/32 | L27 4.723 | L27 263.246 |
| deepseek7b | object_last | raw | L4 5/32 | L0 -0.009 | L27 213.556 |
| deepseek7b | object_last | template_residual | L28 15/32 | L0 -0.007 | L27 213.556 |

## Answer Slot: Raw vs Template Residual Category Margins
| category | qwen3 raw/resid | glm4 raw/resid | deepseek7b raw/resid | objective reading |
|---|---:|---:|---:|---|
| fruit | 12.57/14.07 L32->L32 | -0.00/-0.00 L20->L19 | -0.03/9.18 L0->L27 | residual improves: deepseek7b+resid |
| animal | 7.95/11.36 L32->L32 | -0.00/0.00 L24->L19 | 0.54/4.20 L28->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| tool | 5.33/5.34 L34->L35 | -0.00/0.01 L24->L19 | -0.02/2.29 L0->L27 | mostly stable or model-specific |
| vehicle | 10.10/12.08 L33->L33 | -0.00/0.00 L24->L19 | 0.00/9.19 L0->L27 | residual improves: deepseek7b+resid |
| clothing | 3.41/14.79 L33->L35 | -0.00/-0.00 L24->L19 | -0.02/10.21 L0->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| furniture | 0.62/7.67 L36->L34 | -0.00/-0.00 L19->L19 | -0.03/2.23 L0->L27 | residual improves: qwen3+resid |
| food | 14.77/16.22 L33->L34 | -0.00/-0.00 L24->L19 | 0.68/1.93 L28->L28 | mostly stable or model-specific |
| plant | 15.02/11.99 L34->L32 | -0.00/0.00 L24->L19 | 1.01/11.07 L28->L27 | residual improves: deepseek7b+resid |
| body | 0.43/4.87 L36->L34 | 0.50/-0.00 L40->L19 | 0.44/8.80 L28->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| place | 0.49/7.84 L36->L34 | -0.00/-0.00 L0->L19 | -0.02/6.93 L0->L26 | residual improves: qwen3+resid,deepseek7b+resid |
| building | 11.10/10.46 L32->L35 | 0.02/0.01 L24->L19 | -0.03/7.21 L0->L27 | residual improves: deepseek7b+resid |
| material | 5.76/1.36 L35->L34 | -0.00/0.00 L24->L19 | -0.02/1.81 L0->L27 | mostly stable or model-specific |
| color | 11.00/13.42 L32->L33 | -0.00/0.00 L24->L19 | 0.20/9.14 L28->L27 | residual improves: deepseek7b+resid |
| emotion | 2.56/6.67 L36->L33 | -0.00/0.00 L24->L19 | 1.22/1.39 L28->L23 | residual improves: qwen3+resid |
| role | -0.07/0.08 L0->L10 | -0.00/0.00 L24->L19 | -0.02/1.96 L0->L23 | mostly stable or model-specific |
| profession | 22.52/14.17 L35->L32 | -0.00/0.01 L24->L19 | 43.89/8.57 L27->L27 | mostly stable or model-specific |
| abstract | -0.05/0.00 L0->L0 | -0.00/-0.00 L20->L19 | -0.01/0.06 L0->L6 | still weak across models |
| action | -0.08/4.39 L0->L35 | -0.00/0.01 L0->L19 | -0.01/1.74 L0->L26 | residual improves: qwen3+resid |
| event | 4.61/6.09 L35->L34 | -0.00/-0.00 L0->L19 | -0.01/4.24 L0->L27 | residual improves: deepseek7b+resid |
| time | -0.07/8.93 L0->L35 | -0.00/-0.00 L20->L19 | -0.02/5.77 L0->L26 | residual improves: qwen3+resid,deepseek7b+resid |
| number | -0.07/7.80 L0->L35 | -0.00/-0.00 L0->L19 | -0.01/9.63 L0->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| shape | 7.38/13.34 L34->L35 | -0.00/-0.00 L20->L19 | -0.02/8.51 L0->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| sound | 25.38/14.57 L33->L33 | 0.08/-0.00 L24->L19 | 0.37/7.05 L2->L26 | residual improves: deepseek7b+resid |
| light | 4.98/12.25 L32->L35 | 0.01/0.00 L24->L19 | -0.02/5.77 L0->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| weather | 7.71/18.62 L30->L32 | -0.00/-0.00 L24->L19 | 0.56/14.40 L28->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| container | 0.16/7.64 L36->L34 | -0.00/-0.00 L24->L19 | -0.03/5.32 L0->L26 | residual improves: qwen3+resid,deepseek7b+resid |
| instrument | 3.07/6.26 L31->L33 | 0.01/0.00 L24->L19 | 0.13/1.26 L28->L27 | residual improves: qwen3+resid |
| machine | 1.17/5.37 L36->L35 | -0.00/-0.00 L24->L19 | -0.01/6.28 L0->L27 | residual improves: qwen3+resid,deepseek7b+resid |
| communication | 0.52/6.50 L36->L35 | 0.00/0.00 L0->L19 | -0.01/3.36 L0->L25 | residual improves: qwen3+resid,deepseek7b+resid |
| relation | -0.10/0.29 L0->L31 | -0.00/-0.00 L0->L19 | -0.00/0.27 L0->L25 | still weak across models |
| property | -0.08/5.26 L0->L35 | -0.00/-0.00 L0->L19 | -0.01/2.57 L0->L27 | residual improves: qwen3+resid |
| substance | 0.57/3.38 L36->L35 | -0.00/-0.00 L24->L19 | -0.02/2.07 L0->L26 | mostly stable or model-specific |

## Object Position Survival
| category | qwen3 object resid | glm4 object resid | deepseek7b object resid |
|---|---:|---:|---:|
| fruit | 1.86 L33 rank1 | -0.00 L20 rank1 | 1.47 L25 rank1 |
| animal | 0.03 L0 rank1 | -0.00 L20 rank1 | -0.01 L0 rank9 |
| tool | 2.39 L35 rank1 | 0.00 L20 rank1 | 0.11 L22 rank1 |
| vehicle | 3.72 L32 rank1 | -0.00 L20 rank1 | 0.34 L27 rank1 |
| clothing | 0.17 L11 rank1 | -0.00 L20 rank1 | -0.00 L0 rank9 |
| furniture | 1.66 L32 rank1 | -0.00 L20 rank1 | -0.01 L0 rank2 |
| food | 0.81 L26 rank1 | -0.00 L20 rank1 | 0.28 L28 rank1 |
| plant | 3.05 L32 rank1 | -0.00 L20 rank1 | 4.19 L27 rank1 |
| body | 0.02 L10 rank1 | -0.00 L20 rank1 | 1.00 L23 rank1 |
| place | 1.86 L35 rank1 | -0.00 L20 rank1 | 2.59 L26 rank1 |
| building | 1.72 L34 rank1 | -0.00 L20 rank1 | 1.24 L21 rank1 |
| material | 1.52 L35 rank1 | -0.00 L20 rank1 | -0.01 L0 rank2 |
| color | 3.25 L35 rank1 | -0.00 L20 rank1 | 0.22 L28 rank1 |
| emotion | -0.02 L0 rank4 | 0.00 L20 rank1 | 0.08 L5 rank1 |
| role | 0.17 L15 rank1 | -0.00 L20 rank1 | 1.86 L27 rank1 |
| profession | 3.10 L34 rank1 | -0.00 L20 rank1 | 1.61 L23 rank1 |
| abstract | -0.00 L0 rank2 | -0.00 L20 rank1 | -0.00 L0 rank2 |
| action | -0.02 L0 rank7 | -0.00 L20 rank1 | 0.23 L21 rank1 |
| event | 0.02 L0 rank1 | -0.00 L20 rank1 | -0.00 L0 rank4 |
| time | 0.39 L36 rank1 | -0.00 L20 rank1 | 1.59 L27 rank1 |
| number | 0.37 L14 rank1 | -0.00 L20 rank1 | 1.21 L21 rank1 |
| shape | 3.68 L35 rank1 | -0.00 L20 rank1 | 0.84 L21 rank1 |
| sound | 2.05 L32 rank1 | -0.00 L20 rank1 | 0.36 L26 rank1 |
| light | 5.13 L33 rank1 | 0.00 L20 rank1 | 4.13 L27 rank1 |
| weather | 7.85 L31 rank1 | -0.00 L20 rank1 | 2.86 L27 rank1 |
| container | 4.18 L35 rank1 | -0.00 L20 rank1 | 4.35 L27 rank1 |
| instrument | 1.87 L30 rank1 | -0.00 L20 rank1 | 0.28 L25 rank1 |
| machine | 1.53 L30 rank1 | -0.00 L20 rank1 | -0.00 L0 rank2 |
| communication | 0.64 L28 rank1 | -0.00 L20 rank1 | 0.47 L25 rank1 |
| relation | 3.51 L35 rank1 | -0.00 L20 rank1 | 1.97 L27 rank1 |
| property | 2.38 L35 rank1 | -0.00 L20 rank1 | 0.91 L25 rank1 |
| substance | 0.36 L23 rank1 | -0.00 L20 rank1 | -0.01 L0 rank4 |

## Direct Corrections To Phase105
- Phase105 single-template Qwen3 conclusion is mostly retained at answer_last, but several weak categories become strong after template residual subtraction: clothing, furniture, body, place, action, time, number, container, communication, property.
- Qwen3 object_last has much weaker margins than answer_last, but many categories survive after template residual subtraction; category information exists before the answer slot, yet is amplified at the answer slot.
- GLM4 remains near-zero margin after template residual subtraction at both positions; this points to readout-token/model-format calibration, not just template common-vector contamination.
- DS7B answer_last changes strongly after template residual subtraction: best mean margin rises to 4.723 at L27, supporting that Phase105 understated DS7B because common template/format components masked category directions.
- Boundary norm peaks are unchanged by subtracting same-template mean because the same vector is subtracted from every category in a template; boundary-layer conclusions remain stable.
- Top1 counts after residual subtraction can be inflated when margins are tiny; margin magnitude is the stricter objective signal.
