# Phase339 Frozen-Block Cross-Task Boundary

- Registered cases: 1458
- Phrase rows: 8748
- Rollout rows: 1620
- Eligible model-task denominators: 15/27
- Full task gates: 3/27

## Model scope

- qwen3: prior=False, eligible=7/9, relation=1/4, source=0/2, cross=1/3, scope=phase338_candidate_not_qualified, shrink_gate=False
- glm4: prior=True, eligible=0/9, relation=0/4, source=0/2, cross=0/3, scope=cross_task_denominator_ineligible, shrink_gate=False
- deepseek7b: prior=False, eligible=8/9, relation=1/4, source=0/2, cross=0/3, scope=phase338_candidate_not_qualified, shrink_gate=False

## Boundaries

- Wrong-depth and wrong-position blocks are null-location controls.
- Same-block permutation is recorded as structure sensitivity, not a null control.
- Qwen3 and DeepSeek7B remain descriptive because their Phase338 gates failed.
- Non-finite phrase rows: 53; any affected split is ineligible.
- GLM4 has no fully eligible task denominator in this matrix, so its scope remains unresolved.
- No layer, channel, or neuron shrinking was performed.
- No behavior mechanism or intelligent theory was closed.
