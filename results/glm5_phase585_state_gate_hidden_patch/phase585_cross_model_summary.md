# Phase 585 State Gate Hidden Patch Summary

Confirm setting: value samples=32, polarity negative samples=30, four probe layers per model, alpha=1.0.

## Best Target-Layer Results

| model | gate | best layer | base | repair prompt | hidden patch | random control | target patch | target random | target n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | value_relation_filter | L9 | 87.5% | 100.0% | 90.6% | 87.5% | 25.0% | 0.0% | 4 |
| qwen3 | polarity_format | L34 | 90.0% | 100.0% | 100.0% | 83.3% | 100.0% | 0.0% | 3 |
| glm4 | value_relation_filter | L38 | 90.6% | 100.0% | 90.6% | 90.6% | 0.0% | 0.0% | 3 |
| glm4 | polarity_format | L30 | 76.7% | 100.0% | 100.0% | 80.0% | 100.0% | 14.3% | 7 |
| deepseek7b | value_relation_filter | L7 | 62.5% | 100.0% | 62.5% | 62.5% | 0.0% | 0.0% | 12 |
| deepseek7b | polarity_format | L21 | 53.3% | 100.0% | 96.7% | 53.3% | 92.9% | 7.1% | 14 |

## Key Objective Facts

- Polarity-format hidden patch is strong in middle/late layers for all three models.
- Qwen3 polarity target repair: L27/L34 repaired 3/3 targets while random repaired 0/3.
- GLM4 polarity target repair: L20/L30/L38 repaired 7/7 targets; random repaired 4/7, 1/7, 3/7.
- DS7B polarity target repair: L21 repaired 13/14 and L26 repaired 14/14; random repaired 1/14 and 5/14.
- Value relation-filter prompt repair is strong, but answer_last hidden delta does not transfer on GLM4/DS7B and only weakly transfers on Qwen3.
- Therefore Phase585 supports hidden causal repair for polarity-format gate, but not yet for relation-filter/value gate.
