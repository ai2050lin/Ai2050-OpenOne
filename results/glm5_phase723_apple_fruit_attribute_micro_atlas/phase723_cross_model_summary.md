# Phase 723 Apple-Fruit-Attribute Reuse-Difference Micro-Atlas

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: teacher-forced answer phrase likelihood under local zero head ablation.
- Interpretation: positive necessity means zeroing the head reduced answer phrase likelihood.

## Most Harmful Candidate Heads

### qwen3

| head | mean_logprob_delta | first_rank_delta | top1_drop | apple_need | fruit_need | nonfruit_need | fruit-nonfruit | apple-fruit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L24H29 | -0.0611 | 0.08 | 0.035 | 0.1146 | 0.0557 | 0.1195 | -0.0638 | 0.0589 |
| L26H26 | -0.0223 | 0.02 | 0.009 | 0.0147 | 0.0042 | 0.0706 | -0.0663 | 0.0105 |
| L28H0 | -0.0095 | 0.02 | 0.018 | 0.0305 | 0.0149 | 0.0396 | -0.0247 | 0.0157 |

### glm4

| head | mean_logprob_delta | first_rank_delta | top1_drop | apple_need | fruit_need | nonfruit_need | fruit-nonfruit | apple-fruit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L29H28 | -0.0059 | 0.01 | 0.000 | 0.0168 | 0.0104 | 0.0085 | 0.0018 | 0.0065 |
| L24H19 | -0.0040 | 0.02 | 0.000 | 0.0081 | 0.0021 | 0.0037 | -0.0016 | 0.0060 |
| L29H18 | 0.0059 | 0.01 | 0.000 | -0.0059 | -0.0062 | -0.0059 | -0.0002 | 0.0003 |

### deepseek7b

| head | mean_logprob_delta | first_rank_delta | top1_drop | apple_need | fruit_need | nonfruit_need | fruit-nonfruit | apple-fruit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L20H17 | -0.3161 | 4.11 | 0.061 | 0.2269 | 0.2819 | 0.2440 | 0.0379 | -0.0550 |
| L27H23 | -0.1885 | 2.16 | 0.044 | 0.0840 | 0.3778 | 0.0437 | 0.3341 | -0.2938 |
| L23H0 | -0.1253 | 11.56 | 0.061 | 0.1541 | 0.1867 | 0.0317 | 0.1549 | -0.0326 |

## Strict Interpretation

- This is a micro-world causal screen, not a global neuron atlas.
- Strong shared fruit necessity suggests a reusable category route.
- Strong apple-minus-fruit suggests object-specific differential routing.
- Weak qwen3/GLM4 effects may mean redundancy or different implementation, not absence of coding.
