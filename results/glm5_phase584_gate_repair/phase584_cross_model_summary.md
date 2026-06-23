# Phase 584 Gate Repair Cross-Model Summary

Confirm setting: n_tables=15, two-hop samples=80 per model, polarity samples=60 per model.

| model | direct | gold-cat | rel-emphasis | rel-filter | polarity base | polarity rule+fmt | neg base | neg rule+fmt | wrong-cat->wrong-val | wrong-cat->correct-val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 73.8% | 93.8% | 95.0% | 100.0% | 90.0% | 100.0% | 80.0% | 100.0% | 61.3% | 48.8% |
| glm4 | 70.0% | 78.8% | 96.2% | 92.5% | 50.0% | 100.0% | 0.0% | 100.0% | 62.5% | 41.2% |
| deepseek7b | 53.8% | 60.0% | 92.5% | 97.5% | 50.0% | 98.3% | 0.0% | 96.7% | 41.2% | 51.2% |

Key objective facts:

- Relation-filter repair is the strongest value-retrieval repair: qwen3 100.0%, glm4 92.5%, deepseek7b 97.5%.
- Rule+format polarity repair nearly closes the negative-answer gate: qwen3 100.0%, glm4 100.0%, deepseek7b 96.7% on negatives.
- System instruction alone is weak for polarity repair: glm4 negative 10.0%, deepseek7b negative 0.0%.
- Wrong-category forcing often changes value choice, but not cleanly enough to prove a single mandatory O-C-V path.
- No-CRV accuracy remains near 26-28%, close to chance among four value tokens, so values are not simply memorized without relation rules.
