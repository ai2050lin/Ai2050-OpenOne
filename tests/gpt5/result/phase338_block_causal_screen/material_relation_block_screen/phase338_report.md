# Phase338 Staged Coarse-Block Causal Screen

- Registered cases: 216
- Discovery block summaries: 81
- Discovery gate blocks: 52
- Calibration gate blocks: 9
- Full model block gates: 1/3
- Cross-model block gate: False
- Minimal causal set entry gate: False
- Behavior mechanism closure: 0/72
- Single-unit causal closure: 0/72

## Model Results

- qwen3: block=residual_increment__early__source, calibration=True, heldout=False, private=True, full=False
- glm4: block=mlp_output__early__source, calibration=True, heldout=True, private=True, full=True
- deepseek7b: block=residual_increment__early__source, calibration=True, heldout=False, private=True, full=False

Matched attribute binding was not used as a null control because it is a real relation-binding mechanism.
Mean replacement and recursive neuron splitting remain closed unless the cross-model block gate passes.
