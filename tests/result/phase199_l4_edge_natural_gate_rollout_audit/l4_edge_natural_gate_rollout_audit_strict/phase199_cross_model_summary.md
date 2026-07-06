# Phase 199 L4 edge natural-gate and rollout audit

| model | edge | sign | base stable | ablate stable | boost stable | boost gain | ablate loss | act gap stable-unstable |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | deepseek7b|function|en->en|h14|c3033|mixed_side_effect_channel | mixed_side_effect_channel | 0 | 0 | 0 | 0 | 0 | None |
| deepseek7b | deepseek7b|function|en->en|h14|c6030|suppressor_or_blocker_channel | suppressor_or_blocker_channel | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | glm4|color|en->en|h30|c1165|support_channel | support_channel | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | glm4|color|en->en|h30|c5532|suppressor_or_blocker_channel | suppressor_or_blocker_channel | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | glm4|function|en->en|h30|c1165|mixed_side_effect_channel | mixed_side_effect_channel | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | glm4|function|en->en|h30|c5532|suppressor_or_blocker_channel | suppressor_or_blocker_channel | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | glm4|function|zh->en|h30|c1165|mixed_side_effect_channel | mixed_side_effect_channel | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | glm4|function|zh->en|h30|c5532|suppressor_or_blocker_channel | suppressor_or_blocker_channel | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|color|en->en|h36|c16|mixed_side_effect_channel | mixed_side_effect_channel | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|color|en->en|h36|c249|support_channel | support_channel | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|color|en->en|h36|c2509|near_zero_or_correlational | near_zero_or_correlational | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|function|en->en|h27|c3|support_channel | support_channel | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|function|zh->en|h27|c3|mixed_side_effect_channel | mixed_side_effect_channel | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|function|zh->zh|h27|c2|support_channel | support_channel | 0 | 0 | 0 | 0 | 0 | None |
| qwen3 | qwen3|function|zh->zh|h27|c58|support_channel | support_channel | 0 | 0 | 0 | 0 | 0 | None |
