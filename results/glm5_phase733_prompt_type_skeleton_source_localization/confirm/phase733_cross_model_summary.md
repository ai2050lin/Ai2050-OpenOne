# Phase 733 Prompt-Type Skeleton Source Localization (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: prompt-type formation scan + site replacement.

| model | attn | earliest layer_out | top transfer | delta | changed | hit_gain | hit_loss |
|---|---|---|---|---:|---:|---:|---:|
| qwen3 | sdpa | hidden_28 | commonsense<-explicit|hidden_36 | 3.112 | 0.056 | 0.056 | 0.000 |
| glm4 | sdpa | hidden_39 | commonsense<-explicit|hidden_40 | 2.341 | 0.167 | 0.056 | 0.000 |
| deepseek7b | sdpa | hidden_24 | commonsense<-explicit|hidden_28 | 5.898 | 0.889 | 0.611 | 0.111 |

## Strict Interpretation

- Earliest layer is based on 35% of the model/kind maximum commonsense-vs-explicit effect.
- Replacement validates causal influence at a site, but still may introduce distribution shift.
- This phase localizes source candidates; it does not prove neuron-level writers.

Atlas graph: nodes=45 edges=42
