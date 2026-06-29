# Phase 733 Prompt-Type Skeleton Source Localization (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: prompt-type formation scan + site replacement.

| model | attn | earliest layer_out | top transfer | delta | changed | hit_gain | hit_loss |
|---|---|---|---|---:|---:|---:|---:|
| qwen3 | sdpa | hidden_28 | commonsense<-explicit|L35_layer_input | 4.543 | 0.000 | 0.000 | 0.000 |
| glm4 | sdpa | hidden_39 | commonsense<-explicit|L39_layer_input | 2.318 | 0.000 | 0.000 | 0.000 |
| deepseek7b | sdpa | hidden_24 | commonsense<-explicit|L27_layer_input | 4.443 | 1.000 | 0.500 | 0.500 |

## Strict Interpretation

- Earliest layer is based on 35% of the model/kind maximum commonsense-vs-explicit effect.
- Replacement validates causal influence at a site, but still may introduce distribution shift.
- This phase localizes source candidates; it does not prove neuron-level writers.

Atlas graph: nodes=30 edges=27
