# Phase 105 Cross-Model Global Category Atlas Summary

## Global Layer Distribution
| model | layers | best top1 layer | top1 count | best mean margin layer | best mean boundary layer |
|---|---:|---:|---:|---:|---:|
| qwen3 | 36 | L36 | 23/32 | L36 (0.68) | L35 (161.17) |
| glm4 | 40 | L40 | 22/32 | L0 (-0.00) | L19 (2.48) |
| deepseek7b | 28 | L28 | 8/32 | L0 (-0.02) | L27 (238.80) |

## Category Relative Map
| category | qwen3 | glm4 | deepseek7b | stable reading |
|---|---|---|---|---|
| fruit | M32/B35 margin=12.54 rank=1 sharp_readout_cohesive | M20/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.03 rank=2 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| animal | M32/B35 margin=11.58 rank=1 sharp_readout_cohesive | M24/B20 margin=-0.00 rank=1 diffuse_or_contextual | M28/B27 margin=0.91 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| tool | M35/B35 margin=6.88 rank=1 readout_clear | M24/B20 margin=0.01 rank=1 diffuse_or_contextual | M0/B27 margin=-0.01 rank=3 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| vehicle | M33/B35 margin=12.25 rank=1 sharp_readout_cohesive | M40/B20 margin=1.05 rank=1 diffuse_or_contextual | M0/B27 margin=0.00 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| clothing | M33/B35 margin=2.07 rank=1 cohesive_boundary_unclear_readout | M37/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=4 cohesive_boundary_unclear_readout | model-specific or diffuse |
| furniture | M36/B35 margin=1.07 rank=1 cohesive_boundary_unclear_readout | M20/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.04 rank=2 cohesive_boundary_unclear_readout | model-specific or diffuse |
| food | M33/B35 margin=16.44 rank=1 sharp_readout_cohesive | M37/B20 margin=0.03 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=2 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| plant | M34/B35 margin=15.91 rank=1 sharp_readout_cohesive | M37/B20 margin=-0.00 rank=1 diffuse_or_contextual | M28/B27 margin=0.75 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| body | M36/B35 margin=0.84 rank=1 cohesive_boundary_unclear_readout | M40/B20 margin=1.08 rank=1 diffuse_or_contextual | M28/B27 margin=0.46 rank=1 cohesive_boundary_unclear_readout | rank-stable but margin weak |
| place | M34/B35 margin=2.96 rank=1 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=5 competitive_broad | M0/B27 margin=-0.02 rank=9 cohesive_boundary_unclear_readout | model-specific or diffuse |
| building | M35/B35 margin=14.42 rank=1 sharp_readout_cohesive | M24/B20 margin=0.02 rank=1 diffuse_or_contextual | M9/B27 margin=0.26 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| material | M35/B35 margin=5.87 rank=1 readout_clear | M33/B20 margin=0.29 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=4 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| color | M32/B35 margin=6.23 rank=1 readout_clear | M24/B20 margin=-0.00 rank=1 diffuse_or_contextual | M14/B27 margin=1.01 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| emotion | M36/B35 margin=1.83 rank=1 cohesive_boundary_unclear_readout | M40/B20 margin=2.64 rank=1 diffuse_or_contextual | M28/B27 margin=0.46 rank=1 cohesive_boundary_unclear_readout | rank-stable but margin weak |
| role | M0/B35 margin=-0.09 rank=3 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=2 competitive_broad | M28/B27 margin=0.04 rank=1 cohesive_boundary_unclear_readout | model-specific or diffuse |
| profession | M35/B35 margin=22.24 rank=1 sharp_readout_cohesive | M37/B20 margin=0.21 rank=1 diffuse_or_contextual | M27/B27 margin=26.42 rank=1 sharp_readout_cohesive | Qwen3 readable; cross-model weak/variant |
| abstract | M0/B35 margin=-0.06 rank=4 cohesive_boundary_unclear_readout | M20/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.01 rank=10 cohesive_boundary_unclear_readout | model-specific or diffuse |
| action | M0/B35 margin=-0.09 rank=7 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=6 competitive_broad | M0/B27 margin=-0.00 rank=2 cohesive_boundary_unclear_readout | model-specific or diffuse |
| event | M34/B35 margin=2.48 rank=1 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=3 competitive_broad | M0/B27 margin=-0.02 rank=5 cohesive_boundary_unclear_readout | model-specific or diffuse |
| time | M0/B35 margin=-0.09 rank=2 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=3 competitive_broad | M0/B27 margin=-0.03 rank=4 cohesive_boundary_unclear_readout | model-specific or diffuse |
| number | M0/B35 margin=-0.09 rank=2 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=8 competitive_broad | M0/B27 margin=-0.02 rank=6 cohesive_boundary_unclear_readout | model-specific or diffuse |
| shape | M34/B35 margin=8.63 rank=1 sharp_readout_cohesive | M20/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.03 rank=2 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| sound | M33/B35 margin=23.19 rank=1 sharp_readout_cohesive | M37/B20 margin=0.98 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=2 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| light | M34/B35 margin=3.01 rank=1 cohesive_boundary_unclear_readout | M37/B20 margin=0.02 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=8 cohesive_boundary_unclear_readout | model-specific or diffuse |
| weather | M30/B35 margin=6.71 rank=1 readout_clear | M37/B20 margin=0.00 rank=1 diffuse_or_contextual | M28/B27 margin=0.17 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| container | M36/B35 margin=0.88 rank=1 cohesive_boundary_unclear_readout | M37/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.04 rank=2 cohesive_boundary_unclear_readout | model-specific or diffuse |
| instrument | M32/B35 margin=6.21 rank=1 readout_clear | M24/B20 margin=-0.00 rank=1 diffuse_or_contextual | M28/B27 margin=0.38 rank=1 cohesive_boundary_unclear_readout | Qwen3 readable; cross-model weak/variant |
| machine | M36/B35 margin=0.69 rank=1 cohesive_boundary_unclear_readout | M40/B20 margin=1.11 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=2 cohesive_boundary_unclear_readout | model-specific or diffuse |
| communication | M36/B35 margin=0.60 rank=1 cohesive_boundary_unclear_readout | M0/B20 margin=0.00 rank=1 competitive_broad | M8/B27 margin=0.08 rank=1 cohesive_boundary_unclear_readout | rank-stable but margin weak |
| relation | M0/B35 margin=-0.13 rank=3 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=6 competitive_broad | M0/B27 margin=-0.00 rank=3 cohesive_boundary_unclear_readout | model-specific or diffuse |
| property | M36/B35 margin=0.38 rank=1 cohesive_boundary_unclear_readout | M0/B20 margin=-0.00 rank=2 competitive_broad | M12/B27 margin=1.48 rank=1 cohesive_boundary_unclear_readout | model-specific or diffuse |
| substance | M36/B35 margin=1.03 rank=1 cohesive_boundary_unclear_readout | M37/B20 margin=-0.00 rank=1 diffuse_or_contextual | M0/B27 margin=-0.02 rank=10 cohesive_boundary_unclear_readout | model-specific or diffuse |

## Qwen3 Strong Readout Types
- sound: margin=23.19, marginL=L33, boundaryL=L35, neighbors=action, communication, light
- profession: margin=22.24, marginL=L35, boundaryL=L35, neighbors=role, relation, action
- food: margin=16.44, marginL=L33, boundaryL=L35, neighbors=substance, material, container
- plant: margin=15.91, marginL=L34, boundaryL=L35, neighbors=color, fruit, relation
- building: margin=14.42, marginL=L35, boundaryL=L35, neighbors=place, container, action
- fruit: margin=12.54, marginL=L32, boundaryL=L35, neighbors=plant, color, food
- vehicle: margin=12.25, marginL=L33, boundaryL=L35, neighbors=machine, container, building
- animal: margin=11.58, marginL=L32, boundaryL=L35, neighbors=relation, plant, role
- shape: margin=8.63, marginL=L34, boundaryL=L35, neighbors=property, number, light

## Diffuse Or Weak Readout Types
- qwen3: role, abstract, action, time, number, relation
- glm4: place, role, action, event, time, number, relation, property
- deepseek7b: fruit, tool, clothing, furniture, food, place, material, abstract, action, event, time, number, shape, sound, light, container, machine, relation, substance

## Main Interpretation
- Qwen3 shows the clearest late-layer category readout: concrete object classes and sensory/event classes often peak at L32-L36.
- GLM4 shows very small DCF margins in this readout basis; rank can be correct while amplitude remains weak, so GLM4 needs a better calibrated readout or stronger templates.
- DS7B shows strong late boundary norms and cohesion but weak category-label margins; its internal category structure is present but not cleanly decoded by this DCF word basis.
- Across models, boundary norm usually peaks very late, while margin peaks can be category-specific; this supports a layer-development view rather than a single universal category layer.
- Local boundary removal is only logit-lens evidence; any high-value release edge must be followed by downstream causal patching before being treated as mechanism.
