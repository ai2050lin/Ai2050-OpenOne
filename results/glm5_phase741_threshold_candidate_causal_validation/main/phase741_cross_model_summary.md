# Phase 741 Threshold Candidate Causal Validation (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: component-output transplant and erasure measured by final readout threshold fraction.

| model | component | joint add effect | joint erase effect | donor erase effect | role |
|---|---|---:|---:|---:|---|
| qwen3 | L31:attn_out | 0.161 | 0.001 | -0.140 | causal_boost_candidate |
| qwen3 | L33:mlp_out | 0.011 | 0.001 | -0.023 | weak_boost_candidate |
| qwen3 | L34:attn_out | 0.166 | 0.002 | -0.196 | causal_boost_candidate |
| glm4 | L37:mlp_out | 0.046 | -0.000 | -0.043 | weak_boost_candidate |
| glm4 | L38:mlp_out | 0.670 | -0.027 | -0.681 | causal_boost_candidate |
| glm4 | L39:mlp_out | 0.225 | -0.006 | -0.156 | causal_boost_candidate |
| deepseek7b | L26:attn_out | 0.291 | -0.002 | -0.304 | causal_boost_candidate |
| deepseek7b | L27:attn_out | 0.238 | -0.002 | -0.225 | causal_boost_candidate |
| deepseek7b | L27:mlp_out | 0.028 | 0.002 | -0.017 | weak_boost_candidate |

## Strict Interpretation

- Positive joint add effect means the donor-recipient component delta can push the final readout direction.
- Negative joint erase or donor erase effect means the component is necessary at this coarse output granularity.
- Whole-component edits are stronger than neuron-level proof and can be off-manifold.

Atlas graph: nodes=15 edges=30
