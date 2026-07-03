# Phase 863 Dominant/Auxiliary Channel Role Split (main)

- Source: Phase 862 main single-channel effects.
- Boundary: offline role split, not a new model intervention and not closure.

## Domain Results

| model | domain | full gain/loss | dominant gear | dominant role | dominant gain/loss | gain share | interpretation |
|---|---|---:|---|---|---:|---:|---|
| qwen3 | material | 9/0 | `L31C2257` | `dominant_answer_and_blocker_channel` | 8/0 | 0.889 | `dominant_channel_with_auxiliary_support` |
| deepseek7b | animal | 21/3 | `L27C16651` | `dominant_answer_and_blocker_channel` | 20/2 | 0.952 | `dominant_channel_with_auxiliary_support` |
| deepseek7b | color | 12/2 | `L27C15369` | `dominant_answer_lift_channel` | 10/2 | 0.833 | `dominant_channel_with_auxiliary_support` |

## Channel Details

| model | domain | gear | subset | role | clear gain/loss | blocker modes | answer modes | mean blocker reduction | mean answer delta |
|---|---|---|---|---|---:|---|---|---:|---:|
| qwen3 | material | `L31C2257` | `single1` | `dominant_answer_and_blocker_channel` | 8/0 | `['flip', 'half', 'zero']` | `['flip', 'half', 'zero']` | 0.5667 | 0.1833 |
| qwen3 | material | `L31C4800` | `single0` | `auxiliary_mixed_channel` | 2/0 | `['flip', 'zero']` | `['flip', 'zero']` | 0.3500 | 0.0479 |
| deepseek7b | animal | `L27C16651` | `single0` | `dominant_answer_and_blocker_channel` | 20/2 | `['flip', 'half', 'zero']` | `['flip', 'half', 'zero']` | 0.1000 | 0.9729 |
| deepseek7b | animal | `L24C3875` | `single1` | `auxiliary_mixed_channel` | 1/1 | `['flip']` | `['flip']` | 0.0833 | 0.0115 |
| deepseek7b | color | `L27C15369` | `single0` | `dominant_answer_lift_channel` | 10/2 | `['flip']` | `['flip', 'half', 'zero']` | -0.0167 | 0.8021 |
| deepseek7b | color | `L26C8587` | `single1` | `auxiliary_answer_lift_channel` | 3/1 | `[]` | `['flip', 'zero']` | 0.1000 | 0.1094 |
