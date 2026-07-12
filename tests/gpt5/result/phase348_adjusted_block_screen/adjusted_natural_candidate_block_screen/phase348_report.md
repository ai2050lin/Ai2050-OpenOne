# Phase348 Adjusted Natural-Candidate Block Screen

- Screen rows: 4590
- Candidate-model gates: 1/6
- Heldout revealed: True (selective DS7B candidate only)
- Initial decision: run heldout for the sole passing candidate

- qwen3 / contiguous_multi_token_answer: discovery=False, calibration=False, full=False
- qwen3 / no_morphology_control: discovery=False, calibration=False, full=False
- glm4 / contiguous_multi_token_answer: discovery=False, calibration=False, full=False
- glm4 / no_morphology_control: discovery=False, calibration=False, full=False
- deepseek7b / contiguous_multi_token_answer: discovery=False, calibration=False, full=False
- deepseek7b / no_morphology_control: discovery=True, calibration=True, full=True

No neuron search, sufficiency test, mediation test, or mechanism closure was executed.

## Heldout Reveal

- heldout: phrase=True, behavior=False, full=False
- private_heldout: phrase=False, behavior=False, full=False
- Full heldout/private gate: False
- Stop decision: heldout_failed_stop_before_mcue
