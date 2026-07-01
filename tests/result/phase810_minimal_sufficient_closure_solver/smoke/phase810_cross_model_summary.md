# Phase 810 Minimal Sufficient Closure Solver (smoke)

- Status: `complete`
- Boundary: small combination solver over unit candidates; not global language closure.

## Best Rows

| model | stage | case | size | identity | above | bias | margin | net | resolved | emerged | closure | objective | label | items |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 21.000 | 10.500 | -10.500 | -5.000 | 5.000 | 0.000 | 0 | 26.500 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 21.000 | 10.500 | -10.500 | -5.000 | 5.000 | 0.000 | 0 | 26.750 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935 + attention_head:attn:L35:u26` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 21.000 | 10.500 | -10.500 | -5.000 | 5.000 | 0.000 | 0 | 26.750 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935 + mlp_channel:mlp:L35:u1147` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 21.000 | 10.500 | -10.500 | -5.000 | 5.000 | 0.000 | 0 | 26.750 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935 + mlp_channel:mlp:L34:u1028` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 22.000 | 10.500 | -10.500 | -4.000 | 4.000 | 0.000 | 0 | 27.750 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935 + mlp_channel:mlp:L35:u991` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 22.000 | 11.125 | -11.125 | -4.000 | 4.000 | 0.000 | 0 | 28.062 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935 + identity_anchor:beta0.5` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 22.000 | 11.875 | -11.875 | -4.000 | 5.000 | 1.000 | 0 | 28.637 | `combo_reducer_no_closure` | `mlp_channel:mlp:L35:u935 + identity_anchor:beta1` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.562 | `combo_mixed_or_neutral` | `attention_head:attn:L35:u26` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.562 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L35:u1147` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.562 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L35:u991` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.562 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L34:u1028` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.562 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L34:u3372` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.562 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L33:u219` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.812 | `combo_mixed_or_neutral` | `attention_head:attn:L35:u26 + mlp_channel:mlp:L35:u1147` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.812 | `combo_mixed_or_neutral` | `attention_head:attn:L35:u26 + mlp_channel:mlp:L35:u991` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.812 | `combo_mixed_or_neutral` | `attention_head:attn:L35:u26 + mlp_channel:mlp:L34:u1028` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.812 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L35:u1147 + mlp_channel:mlp:L34:u1028` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 26.000 | 10.625 | -10.625 | 0.000 | 0.000 | 0.000 | 0 | 31.812 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L35:u991 + mlp_channel:mlp:L34:u1028` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 27.000 | 10.562 | -10.562 | 1.000 | 0.000 | 1.000 | 0 | 32.731 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L33:u3304` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 27.000 | 10.625 | -10.625 | 1.000 | 0.000 | 1.000 | 0 | 32.763 | `combo_new_blocker_or_deformer` | `attention_head:attn:L35:u27` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 27.000 | 10.625 | -10.625 | 1.000 | 0.000 | 1.000 | 0 | 32.763 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L33:u1166` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 0 | 27.000 | 10.625 | -10.625 | 1.000 | 0.000 | 1.000 | 0 | 33.013 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L35:u1147 + mlp_channel:mlp:L35:u991` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 1 | 27.000 | 11.312 | -11.312 | 1.000 | 0.000 | 1.000 | 0 | 33.106 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 27.000 | 11.312 | -11.312 | 1.000 | 0.000 | 1.000 | 0 | 33.356 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5 + attention_head:attn:L35:u26` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 27.000 | 11.312 | -11.312 | 1.000 | 0.000 | 1.000 | 0 | 33.356 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5 + mlp_channel:mlp:L35:u1147` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 27.000 | 11.312 | -11.312 | 1.000 | 0.000 | 1.000 | 0 | 33.356 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5 + mlp_channel:mlp:L35:u991` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 27.000 | 11.312 | -11.312 | 1.000 | 0.000 | 1.000 | 0 | 33.356 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5 + mlp_channel:mlp:L34:u1028` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 0 | 28.000 | 10.688 | -10.688 | 2.000 | 0.000 | 2.000 | 0 | 33.994 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L34:u265` |
| qwen3 | single_prefilter | p765_0041_commonsense_question_plant:oak:grows_on_tree | 1 | 1 | 28.000 | 11.875 | -11.875 | 2.000 | 0.000 | 2.000 | 0 | 34.587 | `combo_new_blocker_or_deformer` | `identity_anchor:beta1` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 28.000 | 11.875 | -11.875 | 2.000 | 0.000 | 2.000 | 0 | 34.837 | `combo_new_blocker_or_deformer` | `attention_head:attn:L35:u26 + identity_anchor:beta1` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 28.000 | 11.875 | -11.875 | 2.000 | 0.000 | 2.000 | 0 | 34.837 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L35:u1147 + identity_anchor:beta1` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 28.000 | 11.875 | -11.875 | 2.000 | 0.000 | 2.000 | 0 | 34.837 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L35:u991 + identity_anchor:beta1` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 28.000 | 11.875 | -11.875 | 2.000 | 0.000 | 2.000 | 0 | 34.837 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L34:u1028 + identity_anchor:beta1` |
| qwen3 | combo_search | p765_0041_commonsense_question_plant:oak:grows_on_tree | 2 | 1 | 28.000 | 12.875 | -12.875 | 2.000 | 0.000 | 2.000 | 0 | 35.337 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5 + identity_anchor:beta1` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 1 | 95.000 | 5.906 | -5.906 | -2.000 | 2.000 | 0.000 | 0 | 98.203 | `combo_reducer_no_closure` | `identity_anchor:beta0.5` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 96.000 | 5.531 | -5.531 | -1.000 | 1.000 | 0.000 | 0 | 99.016 | `combo_reducer_no_closure` | `mlp_channel:mlp:L38:u5084` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 96.000 | 5.531 | -5.531 | -1.000 | 1.000 | 0.000 | 0 | 99.016 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u11792` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.031 | `combo_reducer_no_closure` | `mlp_channel:mlp:L34:u1917` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.031 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u1012` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.031 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u12358` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.531 | -5.531 | -1.000 | 1.000 | 0.000 | 0 | 99.266 | `combo_reducer_no_closure` | `mlp_channel:mlp:L38:u5084 + mlp_channel:mlp:L27:u11792` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.531 | -5.531 | -1.000 | 1.000 | 0.000 | 0 | 99.266 | `combo_reducer_no_closure` | `mlp_channel:mlp:L38:u5084 + mlp_channel:mlp:L27:u12358` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L38:u5084 + mlp_channel:mlp:L34:u1917` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L38:u5084 + mlp_channel:mlp:L27:u1012` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u11792 + mlp_channel:mlp:L34:u1917` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u11792 + mlp_channel:mlp:L27:u12358` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L34:u1917 + mlp_channel:mlp:L27:u1012` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L34:u1917 + mlp_channel:mlp:L27:u12358` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 96.000 | 5.562 | -5.562 | -1.000 | 1.000 | 0.000 | 0 | 99.281 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u1012 + mlp_channel:mlp:L27:u12358` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 96.000 | 5.938 | -5.938 | -1.000 | 1.000 | 0.000 | 0 | 99.469 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L38:u5084` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 96.000 | 5.938 | -5.938 | -1.000 | 1.000 | 0.000 | 0 | 99.469 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L27:u11792` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 96.000 | 5.938 | -5.938 | -1.000 | 1.000 | 0.000 | 0 | 99.469 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L34:u1917` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 96.000 | 5.938 | -5.938 | -1.000 | 1.000 | 0.000 | 0 | 99.469 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L27:u1012` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 96.000 | 5.938 | -5.938 | -1.000 | 1.000 | 0.000 | 0 | 99.469 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L27:u12358` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 97.000 | 5.562 | -5.562 | 0.000 | 0.000 | 0.000 | 0 | 100.031 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L38:u4526` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 97.000 | 5.562 | -5.562 | 0.000 | 0.000 | 0.000 | 0 | 100.031 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L38:u12913` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 97.000 | 5.562 | -5.562 | 0.000 | 0.000 | 0.000 | 0 | 100.031 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L39:u7043` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 97.000 | 5.562 | -5.562 | 0.000 | 0.000 | 0.000 | 0 | 100.031 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L34:u8761` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 97.000 | 5.562 | -5.562 | 0.000 | 0.000 | 0.000 | 0 | 100.031 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L34:u7327` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 0 | 97.000 | 5.562 | -5.562 | 0.000 | 0.000 | 0.000 | 0 | 100.281 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L27:u11792 + mlp_channel:mlp:L27:u1012` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 1 | 98.000 | 6.312 | -6.312 | 1.000 | 1.000 | 2.000 | 0 | 101.806 | `combo_new_blocker_or_deformer` | `identity_anchor:beta1` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 98.000 | 6.312 | -6.312 | 1.000 | 1.000 | 2.000 | 0 | 102.056 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L38:u5084 + identity_anchor:beta1` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 98.000 | 6.312 | -6.312 | 1.000 | 1.000 | 2.000 | 0 | 102.056 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L27:u11792 + identity_anchor:beta1` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 98.000 | 6.312 | -6.312 | 1.000 | 1.000 | 2.000 | 0 | 102.056 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L34:u1917 + identity_anchor:beta1` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 98.000 | 6.312 | -6.312 | 1.000 | 1.000 | 2.000 | 0 | 102.056 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L27:u12358 + identity_anchor:beta1` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 98.000 | 6.688 | -6.688 | 1.000 | 1.000 | 2.000 | 0 | 102.244 | `combo_new_blocker_or_deformer` | `identity_anchor:beta0.5 + identity_anchor:beta1` |
| glm4 | combo_search | p765_0051_commonsense_question_plant:wheat:edible | 2 | 1 | 102.000 | 6.344 | -6.344 | 5.000 | 0.000 | 5.000 | 0 | 106.672 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L27:u1012 + identity_anchor:beta1` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 103.000 | 5.625 | -5.625 | 6.000 | 1.000 | 7.000 | 0 | 107.463 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L39:u4744` |
| glm4 | single_prefilter | p765_0051_commonsense_question_plant:wheat:edible | 1 | 0 | 103.000 | 5.625 | -5.625 | 6.000 | 1.000 | 7.000 | 0 | 107.463 | `combo_new_blocker_or_deformer` | `mlp_channel:mlp:L39:u7968` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 341.000 | 8.562 | -8.562 | -24.000 | 24.000 | 0.000 | 0 | 345.781 | `combo_reducer_no_closure` | `identity_anchor:beta1 + identity_anchor:beta0.5` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 341.000 | 9.062 | -9.062 | -24.000 | 24.000 | 0.000 | 0 | 346.031 | `combo_reducer_no_closure` | `identity_anchor:beta1 + mlp_channel:mlp:L26:u9394` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 342.000 | 9.062 | -9.062 | -23.000 | 23.000 | 0.000 | 0 | 347.031 | `combo_reducer_no_closure` | `identity_anchor:beta1 + mlp_channel:mlp:L27:u2295` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 1 | 344.000 | 9.062 | -9.062 | -21.000 | 22.000 | 1.000 | 0 | 348.981 | `combo_reducer_no_closure` | `identity_anchor:beta1` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 346.000 | 9.531 | -9.531 | -19.000 | 19.000 | 0.000 | 0 | 351.266 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L27:u2295` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 347.000 | 9.094 | -9.094 | -18.000 | 18.000 | 0.000 | 0 | 352.047 | `combo_reducer_no_closure` | `identity_anchor:beta1 + mlp_channel:mlp:L24:u1787` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 347.000 | 9.531 | -9.531 | -18.000 | 18.000 | 0.000 | 0 | 352.266 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L26:u9394` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 349.000 | 9.531 | -9.531 | -16.000 | 16.000 | 0.000 | 0 | 354.266 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + mlp_channel:mlp:L24:u1787` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 1 | 350.000 | 9.594 | -9.594 | -15.000 | 15.000 | 0.000 | 0 | 355.047 | `combo_reducer_no_closure` | `identity_anchor:beta0.5` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 0 | 351.000 | 10.000 | -10.000 | -14.000 | 15.000 | 1.000 | 0 | 356.700 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u2295 + attention_head:attn:L19:u1` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 0 | 351.000 | 9.938 | -9.938 | -14.000 | 17.000 | 3.000 | 0 | 357.069 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u2295 + mlp_channel:mlp:L26:u9394` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 352.000 | 9.500 | -9.500 | -13.000 | 16.000 | 3.000 | 0 | 357.850 | `combo_reducer_no_closure` | `identity_anchor:beta0.5 + attention_head:attn:L19:u1` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 355.000 | 9.938 | -9.938 | -10.000 | 13.000 | 3.000 | 0 | 360.819 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u2295` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 0 | 358.000 | 9.969 | -9.969 | -7.000 | 10.000 | 3.000 | 0 | 364.084 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u2295 + mlp_channel:mlp:L24:u1787` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 360.000 | 9.969 | -9.969 | -5.000 | 5.000 | 0.000 | 0 | 365.234 | `combo_reducer_no_closure` | `mlp_channel:mlp:L24:u1787` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 360.000 | 10.094 | -10.094 | -5.000 | 6.000 | 1.000 | 0 | 365.497 | `combo_reducer_no_closure` | `attention_head:attn:L19:u1` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 0 | 360.000 | 10.094 | -10.094 | -5.000 | 6.000 | 1.000 | 0 | 365.747 | `combo_reducer_no_closure` | `attention_head:attn:L19:u1 + mlp_channel:mlp:L26:u9394` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 361.000 | 10.031 | -10.031 | -4.000 | 5.000 | 1.000 | 0 | 366.466 | `combo_reducer_no_closure` | `mlp_channel:mlp:L26:u9394` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 361.000 | 10.031 | -10.031 | -4.000 | 5.000 | 1.000 | 0 | 366.466 | `combo_reducer_no_closure` | `mlp_channel:mlp:L24:u4514` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 0 | 361.000 | 10.094 | -10.094 | -4.000 | 5.000 | 1.000 | 0 | 366.747 | `combo_reducer_no_closure` | `mlp_channel:mlp:L24:u1787 + attention_head:attn:L19:u1` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 0 | 362.000 | 10.031 | -10.031 | -3.000 | 4.000 | 1.000 | 0 | 367.716 | `combo_reducer_no_closure` | `mlp_channel:mlp:L24:u1787 + mlp_channel:mlp:L26:u9394` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 363.000 | 10.031 | -10.031 | -2.000 | 2.000 | 0.000 | 0 | 368.266 | `combo_reducer_no_closure` | `attention_head:attn:L19:u13` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 363.000 | 10.031 | -10.031 | -2.000 | 3.000 | 1.000 | 0 | 368.466 | `combo_reducer_no_closure` | `mlp_channel:mlp:L24:u15099` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 364.000 | 10.031 | -10.031 | -1.000 | 1.000 | 0.000 | 0 | 369.266 | `combo_reducer_no_closure` | `mlp_channel:mlp:L27:u12909` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 364.000 | 10.031 | -10.031 | -1.000 | 1.000 | 0.000 | 0 | 369.266 | `combo_reducer_no_closure` | `mlp_channel:mlp:L26:u16013` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 364.000 | 10.031 | -10.031 | -1.000 | 1.000 | 0.000 | 0 | 369.266 | `combo_reducer_no_closure` | `mlp_channel:mlp:L26:u1219` |
| deepseek7b | single_prefilter | p765_0075_commonsense_question_tool:hammer:edible | 1 | 0 | 365.000 | 10.031 | -10.031 | 0.000 | 0.000 | 0.000 | 0 | 370.266 | `combo_mixed_or_neutral` | `mlp_channel:mlp:L27:u16230` |
| deepseek7b | combo_search | p765_0075_commonsense_question_tool:hammer:edible | 2 | 1 | 365.000 | 9.031 | -9.031 | 0.000 | 4.000 | 4.000 | 0 | 370.816 | `combo_mixed_or_neutral` | `identity_anchor:beta1 + attention_head:attn:L19:u1` |

## By Label

| model | labels | token closures | valid rows |
|---|---|---:|---:|
| qwen3 | `{"combo_mixed_or_neutral": 11, "combo_new_blocker_or_deformer": 16, "combo_reducer_no_closure": 7}` | 0 | 34 |
| glm4 | `{"combo_mixed_or_neutral": 6, "combo_new_blocker_or_deformer": 9, "combo_reducer_no_closure": 20}` | 0 | 35 |
| deepseek7b | `{"combo_mixed_or_neutral": 2, "combo_reducer_no_closure": 26}` | 0 | 28 |

## By Combo Size

| model | stage | size | rows | cases | above | bias | net | closure rate | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | combo_search | 2 | 21 | 1 | 25.429 | 11.161 | -0.571 | 0.000 | `{"combo_mixed_or_neutral": 5, "combo_new_blocker_or_deformer": 10, "combo_reducer_no_closure": 6}` |
| qwen3 | single_prefilter | 1 | 13 | 1 | 26.231 | 10.764 | 0.231 | 0.000 | `{"combo_mixed_or_neutral": 6, "combo_new_blocker_or_deformer": 6, "combo_reducer_no_closure": 1}` |
| glm4 | combo_search | 2 | 21 | 1 | 96.810 | 5.882 | -0.190 | 0.000 | `{"combo_mixed_or_neutral": 1, "combo_new_blocker_or_deformer": 6, "combo_reducer_no_closure": 14}` |
| glm4 | single_prefilter | 1 | 14 | 1 | 97.429 | 5.645 | 0.429 | 0.000 | `{"combo_mixed_or_neutral": 5, "combo_new_blocker_or_deformer": 3, "combo_reducer_no_closure": 6}` |
| deepseek7b | combo_search | 2 | 15 | 1 | 351.533 | 9.535 | -13.467 | 0.000 | `{"combo_mixed_or_neutral": 1, "combo_reducer_no_closure": 14}` |
| deepseek7b | single_prefilter | 1 | 13 | 1 | 359.538 | 9.916 | -5.462 | 0.000 | `{"combo_mixed_or_neutral": 1, "combo_reducer_no_closure": 12}` |
