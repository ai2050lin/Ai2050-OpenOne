# Phase 800 Target Booster vs True Suppressor Disentanglement

- Source phase: `799`
- Source root: `tests/result/phase799_blocker_field_causal_suppressor_localization`
- Model runs: none; this is offline analysis over Phase 799 rows.
- Boundary: separates target boost, true blocker suppression, and threshold shift behavior.

## Cross-Round Model Summary

| round | model | rows | target gain | blocker suppression | count reduction | resolved | new rate | anchor gap | label hint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| smoke | qwen3 | 4 | 2.938 | 1.093 | 0.827 | 0.838 | 0.065 | 4.312 | true_suppressor_like |
| smoke | glm4 | 2 | -0.219 | 0.797 | -0.032 | 0.299 | 0.321 | 1.469 | unstable_new_blocker |
| smoke | deepseek7b | 4 | 3.094 | -1.164 | 0.617 | 0.653 | 0.094 | 2.641 | target_booster_or_threshold_shift |
| main | qwen3 | 48 | 3.153 | 0.820 | 0.822 | 0.812 | 0.032 | 2.314 | true_suppressor_like |
| main | glm4 | 30 | 1.573 | 0.497 | 0.663 | 0.712 | 0.053 | 0.227 | true_suppressor_like |
| main | deepseek7b | 48 | 3.271 | -0.580 | 0.388 | 0.686 | 0.134 | 2.028 | target_booster_or_threshold_shift |
| confirm | qwen3 | 192 | 2.890 | 0.662 | 0.818 | 0.728 | 0.041 | 2.439 | true_suppressor_like |
| confirm | glm4 | 120 | 1.593 | 0.444 | 0.766 | 0.684 | 0.045 | 0.147 | true_suppressor_like |
| confirm | deepseek7b | 192 | 2.970 | -0.522 | 0.429 | 0.664 | 0.123 | 2.097 | target_booster_or_threshold_shift |

## Confirm Label Counts

```json
{
  "deepseek7b": {
    "target_booster_or_threshold_shift": 40
  },
  "glm4": {
    "true_suppressor_like": 16,
    "unstable_new_blocker": 4
  },
  "qwen3": {
    "true_suppressor_like": 30,
    "weak_or_mixed": 2
  }
}
```

## Top True Suppressor-Like Candidates

| model | component | source group | ladder | target gain | blocker suppression | resolved | new rate | true score |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `attn:L35` | `instruction` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `instruction` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `route_answer` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L35` | `all_pre_answer` | `kv_o_route` | 2.729 | 0.889 | 0.740 | 0.060 | 1.686 |
| qwen3 | `attn:L34` | `instruction` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `all_pre_answer` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `instruction` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `all_pre_answer` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `instruction` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `all_pre_answer` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `instruction` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |
| qwen3 | `attn:L34` | `all_pre_answer` | `route_answer` | 2.974 | 0.494 | 0.717 | 0.022 | 1.029 |

## Top Threshold-Shift Candidates

| model | component | source group | ladder | target gain | blocker suppression | resolved | new rate | threshold score |
|---|---|---|---|---:|---:|---:|---:|---:|
| deepseek7b | `attn:L27` | `instruction` | `route_answer` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `instruction` | `kv_o_route` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `route_answer` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `kv_o_route` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `instruction` | `route_answer` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `instruction` | `kv_o_route` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `route_answer` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `kv_o_route` | 4.432 | -0.699 | 0.689 | 0.043 | 2.228 |
| deepseek7b | `attn:L27` | `instruction` | `route_answer` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `instruction` | `kv_o_route` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `route_answer` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `kv_o_route` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `instruction` | `route_answer` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `instruction` | `kv_o_route` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `route_answer` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L27` | `all_pre_answer` | `kv_o_route` | 3.142 | -0.666 | 0.593 | 0.039 | 1.289 |
| deepseek7b | `attn:L26` | `instruction` | `route_answer` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `instruction` | `kv_o_route` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `all_pre_answer` | `route_answer` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `all_pre_answer` | `kv_o_route` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `instruction` | `route_answer` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `instruction` | `kv_o_route` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `all_pre_answer` | `route_answer` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
| deepseek7b | `attn:L26` | `all_pre_answer` | `kv_o_route` | 3.152 | -0.561 | 0.703 | 0.018 | 1.267 |
