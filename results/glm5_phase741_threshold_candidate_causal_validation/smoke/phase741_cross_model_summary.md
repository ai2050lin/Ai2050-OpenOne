# Phase 741 Threshold Candidate Causal Validation (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: component-output transplant and erasure measured by final readout threshold fraction.

| model | component | joint add effect | joint erase effect | donor erase effect | role |
|---|---|---:|---:|---:|---|
| qwen3 | L34:attn_out | 0.035 | 0.001 | -0.115 | weak_boost_candidate |
| glm4 | L38:mlp_out | 1.044 | -0.053 | -0.891 | causal_boost_candidate |
| deepseek7b | L26:attn_out | 0.685 | -0.003 | -0.981 | causal_boost_candidate |

## Strict Interpretation

- Positive joint add effect means the donor-recipient component delta can push the final readout direction.
- Negative joint erase or donor erase effect means the component is necessary at this coarse output granularity.
- Whole-component edits are stronger than neuron-level proof and can be off-manifold.

Atlas graph: nodes=9 edges=12
