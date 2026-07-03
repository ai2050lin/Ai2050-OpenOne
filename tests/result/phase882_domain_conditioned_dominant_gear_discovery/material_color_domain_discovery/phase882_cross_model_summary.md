# Phase 882 Domain-Conditioned Dominant Gear Discovery (material_color_domain_discovery)

- Boundary: atlas construction and candidate discovery; not closure.
- Discovery: class-vs-object readout-coupled MLP activation.
- Controls: same-layer random controls and cross-domain evaluation.

## Models

| model | status | discovered | rows | closure from open | answer gain | domain-specific | cross-domain |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | complete | 2 | 576 | 0 | 0 | 0 | 0 |
| glm4 | complete | 2 | 576 | 0 | 0 | 0 | 0 |
| deepseek7b | complete | 2 | 576 | 4 | 4 | 0 | 4 |

## Overall

- Summary: `{'n': 1728, 'models': {'qwen3': 576, 'glm4': 576, 'deepseek7b': 576}, 'discovery_domains': {'material': 864, 'color': 864}, 'eval_domains': {'animal': 576, 'material': 576, 'color': 576}, 'control_types': {'discovered': 864, 'same_layer_random': 864}, 'closure_from_open': 4, 'answer_gain': 4, 'domain_specific_closure': 0, 'cross_domain_closure': 4, 'intervened_boundary_closed': 353, 'mean_blocker_reduction': -0.2991898148148148, 'mean_rank_improvement': -0.2991898148148148, 'mean_class_logit_delta': -0.10095666956018519}`

## Candidate Evidence Labels

| model | candidate | label | n | same-domain closure | cross-domain closure | answer gain | control |
|---|---|---|---:|---:|---:|---:|---|
| deepseek7b | `discovered|color|L27C1851:zero` | side_effect_or_non_specific | 72 | 0 | 2 | 2 | discovered |
| deepseek7b | `discovered|material|L27C1851:zero` | side_effect_or_non_specific | 72 | 0 | 2 | 2 | discovered |
| qwen3 | `discovered|color|L31C3157:flip` | no_repair | 72 | 0 | 0 | 0 | discovered |
| qwen3 | `discovered|color|L31C3157:zero` | no_repair | 72 | 0 | 0 | 0 | discovered |
| qwen3 | `discovered|material|L31C3101:flip` | weak_modulator | 72 | 0 | 0 | 0 | discovered |
| qwen3 | `discovered|material|L31C3101:zero` | weak_modulator | 72 | 0 | 0 | 0 | discovered |
| qwen3 | `same_layer_random|color|L31C7265:flip` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| qwen3 | `same_layer_random|color|L31C7265:zero` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| qwen3 | `same_layer_random|material|L31C3986:flip` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| qwen3 | `same_layer_random|material|L31C3986:zero` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| glm4 | `discovered|color|L31C6437:flip` | weak_modulator | 72 | 0 | 0 | 0 | discovered |
| glm4 | `discovered|color|L31C6437:zero` | weak_modulator | 72 | 0 | 0 | 0 | discovered |
| glm4 | `discovered|material|L28C6334:flip` | no_repair | 72 | 0 | 0 | 0 | discovered |
| glm4 | `discovered|material|L28C6334:zero` | no_repair | 72 | 0 | 0 | 0 | discovered |
| glm4 | `same_layer_random|color|L31C12142:flip` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| glm4 | `same_layer_random|color|L31C12142:zero` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| glm4 | `same_layer_random|material|L28C8380:flip` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| glm4 | `same_layer_random|material|L28C8380:zero` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| deepseek7b | `discovered|color|L27C1851:flip` | no_repair | 72 | 0 | 0 | 0 | discovered |
| deepseek7b | `discovered|material|L27C1851:flip` | no_repair | 72 | 0 | 0 | 0 | discovered |
| deepseek7b | `same_layer_random|color|L27C14480:flip` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| deepseek7b | `same_layer_random|color|L27C14480:zero` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| deepseek7b | `same_layer_random|material|L27C12125:flip` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
| deepseek7b | `same_layer_random|material|L27C12125:zero` | no_repair | 72 | 0 | 0 | 0 | same_layer_random |
