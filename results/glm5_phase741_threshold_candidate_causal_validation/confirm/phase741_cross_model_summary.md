# Phase 741 Threshold Candidate Causal Validation (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: component-output transplant and erasure measured by final readout threshold fraction.

| model | component | joint add effect | joint erase effect | donor erase effect | role |
|---|---|---:|---:|---:|---|
| qwen3 | L31:attn_out | 0.124 | 0.001 | -0.131 | causal_boost_candidate |
| qwen3 | L33:mlp_out | 0.019 | 0.001 | -0.038 | weak_boost_candidate |
| qwen3 | L34:attn_out | 0.164 | 0.001 | -0.230 | causal_boost_candidate |
| glm4 | L37:mlp_out | 0.053 | -0.000 | -0.039 | causal_boost_candidate |
| glm4 | L38:mlp_out | 0.671 | -0.028 | -0.678 | causal_boost_candidate |
| glm4 | L39:mlp_out | 0.204 | -0.005 | -0.135 | causal_boost_candidate |
| deepseek7b | L26:attn_out | 0.310 | -0.005 | -0.320 | causal_boost_candidate |
| deepseek7b | L27:attn_out | 0.211 | -0.002 | -0.204 | causal_boost_candidate |
| deepseek7b | L27:mlp_out | 0.034 | -0.001 | -0.031 | weak_boost_candidate |

## Strict Interpretation

- Positive joint add effect means the donor-recipient component delta can push the final readout direction.
- Negative joint erase or donor erase effect means the component is necessary at this coarse output granularity.
- Whole-component edits are stronger than neuron-level proof and can be off-manifold.

Atlas graph: nodes=15 edges=30
