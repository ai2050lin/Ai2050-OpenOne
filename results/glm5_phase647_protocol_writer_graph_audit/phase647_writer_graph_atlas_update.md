# Phase 647 Writer Graph Atlas Update

## qwen3

- baselines: original exact=19/26, newline=0/26; inline exact=0/26, newline=15/26
### sufficiency

- 1. `to_original_interval_L18_19_attn_out_restore` exact=21/26, newline=0/26, rank=1.12
- 2. `to_original_L19_mlp_out_restore` exact=20/26, newline=0/26, rank=1.04
- 3. `to_original_L20_attn_out_restore` exact=20/26, newline=0/26, rank=1.08
- 4. `to_original_interval_L17_20_attn_out_restore` exact=20/26, newline=0/26, rank=1.15
- 5. `to_original_interval_L17_20_mlp_out_restore` exact=17/26, newline=0/26, rank=1.27
- 6. `to_original_L19_attn_out_restore` exact=17/26, newline=2/26, rank=1.31
- 7. `to_original_interval_L18_19_mlp_out_restore` exact=16/26, newline=2/26, rank=1.31
- 8. `to_original_L18_attn_out_restore` exact=15/26, newline=0/26, rank=1.19
- 9. `to_original_L20_mlp_out_restore` exact=11/26, newline=4/26, rank=1.73
- 10. `to_original_L17_mlp_out_restore` exact=9/26, newline=0/26, rank=1.65
- 11. `to_original_L18_mlp_out_restore` exact=8/26, newline=14/26, rank=2.65
- 12. `to_original_L17_attn_out_restore` exact=5/26, newline=14/26, rank=3.12

### necessity

- 1. `remove_from_inline_interval_L18_19_mlp_out_restore` exact=0/26, newline=26/26, rank=3.73
- 2. `remove_from_inline_L19_mlp_out_restore` exact=0/26, newline=26/26, rank=5.12
- 3. `remove_from_inline_L17_mlp_out_restore` exact=0/26, newline=21/26, rank=4.96
- 4. `remove_from_inline_L20_attn_out_restore` exact=0/26, newline=14/26, rank=5.27
- 5. `remove_from_inline_L20_mlp_out_restore` exact=1/26, newline=18/26, rank=4.58
- 6. `remove_from_inline_L18_mlp_out_restore` exact=1/26, newline=11/26, rank=4.15
- 7. `remove_from_inline_L18_attn_out_restore` exact=2/26, newline=16/26, rank=4.54
- 8. `remove_from_inline_L19_attn_out_restore` exact=3/26, newline=19/26, rank=4.81
- 9. `remove_from_inline_L17_attn_out_restore` exact=3/26, newline=10/26, rank=3.50
- 10. `remove_from_inline_interval_L17_20_attn_out_restore` exact=7/26, newline=7/26, rank=3.58
- 11. `remove_from_inline_interval_L18_19_layer_out_restore` exact=7/26, newline=0/26, rank=2.65
- 12. `remove_from_inline_L18_layer_out_restore` exact=8/26, newline=2/26, rank=2.58

## glm4

- baselines: original exact=29/36, newline=0/36; inline exact=27/36, newline=0/36
### sufficiency

- 1. `to_original_interval_L18_19_attn_out_restore` exact=30/36, newline=0/36, rank=1.11
- 2. `to_original_L19_attn_out_restore` exact=29/36, newline=0/36, rank=1.31
- 3. `to_original_L18_attn_out_restore` exact=29/36, newline=0/36, rank=1.42
- 4. `to_original_L20_attn_out_restore` exact=28/36, newline=0/36, rank=1.42
- 5. `to_original_L18_mlp_out_restore` exact=28/36, newline=0/36, rank=1.47
- 6. `to_original_L19_mlp_out_restore` exact=28/36, newline=0/36, rank=1.50
- 7. `to_original_L17_layer_input_restore` exact=28/36, newline=0/36, rank=1.64
- 8. `to_original_L19_layer_out_restore` exact=28/36, newline=0/36, rank=1.75
- 9. `to_original_L20_layer_input_restore` exact=28/36, newline=0/36, rank=1.75
- 10. `to_original_L20_layer_out_restore` exact=28/36, newline=0/36, rank=1.78
- 11. `to_original_L18_layer_out_restore` exact=28/36, newline=0/36, rank=1.94
- 12. `to_original_L19_layer_input_restore` exact=28/36, newline=0/36, rank=1.94

### necessity

- 1. `remove_from_inline_interval_L17_20_mlp_out_restore` exact=10/36, newline=0/36, rank=2.28
- 2. `remove_from_inline_interval_L17_20_attn_out_restore` exact=17/36, newline=0/36, rank=4.61
- 3. `remove_from_inline_L19_attn_out_restore` exact=25/36, newline=0/36, rank=1.75
- 4. `remove_from_inline_L17_attn_out_restore` exact=28/36, newline=0/36, rank=1.53
- 5. `remove_from_inline_interval_L18_19_mlp_out_restore` exact=29/36, newline=0/36, rank=1.33
- 6. `remove_from_inline_L20_attn_out_restore` exact=29/36, newline=0/36, rank=1.36
- 7. `remove_from_inline_L20_mlp_out_restore` exact=29/36, newline=0/36, rank=1.39
- 8. `remove_from_inline_L18_mlp_out_restore` exact=29/36, newline=0/36, rank=1.47
- 9. `remove_from_inline_interval_L18_19_attn_out_restore` exact=30/36, newline=0/36, rank=1.31
- 10. `remove_from_inline_interval_L17_20_layer_out_restore` exact=31/36, newline=0/36, rank=1.11
- 11. `remove_from_inline_L20_layer_out_restore` exact=31/36, newline=0/36, rank=1.11
- 12. `remove_from_inline_interval_L18_19_layer_out_restore` exact=31/36, newline=0/36, rank=1.17

## deepseek7b

- baselines: original exact=12/48, newline=34/48; inline exact=45/48, newline=0/48
### sufficiency

- 1. `to_original_L17_layer_input_restore` exact=46/48, newline=0/48, rank=1.02
- 2. `to_original_L18_layer_out_restore` exact=46/48, newline=0/48, rank=1.02
- 3. `to_original_L19_layer_input_restore` exact=46/48, newline=0/48, rank=1.02
- 4. `to_original_L17_layer_out_restore` exact=46/48, newline=0/48, rank=1.04
- 5. `to_original_L18_layer_input_restore` exact=46/48, newline=0/48, rank=1.04
- 6. `to_original_L20_layer_out_restore` exact=45/48, newline=0/48, rank=1.04
- 7. `to_original_interval_L18_19_layer_out_restore` exact=45/48, newline=0/48, rank=1.06
- 8. `to_original_L19_layer_out_restore` exact=45/48, newline=0/48, rank=1.06
- 9. `to_original_L20_layer_input_restore` exact=45/48, newline=0/48, rank=1.06
- 10. `to_original_interval_L17_20_layer_out_restore` exact=43/48, newline=0/48, rank=1.04
- 11. `to_original_interval_L17_20_mlp_out_restore` exact=33/48, newline=4/48, rank=1.31
- 12. `to_original_L18_mlp_out_restore` exact=23/48, newline=23/48, rank=3.15

### necessity

- 1. `remove_from_inline_interval_L17_20_attn_out_restore` exact=8/48, newline=0/48, rank=2.42
- 2. `remove_from_inline_interval_L17_20_layer_out_restore` exact=12/48, newline=35/48, rank=4.77
- 3. `remove_from_inline_L20_layer_out_restore` exact=12/48, newline=35/48, rank=4.77
- 4. `remove_from_inline_interval_L18_19_layer_out_restore` exact=14/48, newline=30/48, rank=3.35
- 5. `remove_from_inline_L19_layer_out_restore` exact=14/48, newline=30/48, rank=3.35
- 6. `remove_from_inline_L20_layer_input_restore` exact=14/48, newline=30/48, rank=3.35
- 7. `remove_from_inline_L18_layer_out_restore` exact=14/48, newline=22/48, rank=2.77
- 8. `remove_from_inline_L19_layer_input_restore` exact=14/48, newline=22/48, rank=2.77
- 9. `remove_from_inline_L17_layer_out_restore` exact=15/48, newline=18/48, rank=2.52
- 10. `remove_from_inline_L18_layer_input_restore` exact=15/48, newline=18/48, rank=2.52
- 11. `remove_from_inline_L17_layer_input_restore` exact=15/48, newline=15/48, rank=2.38
- 12. `remove_from_inline_interval_L17_20_mlp_out_restore` exact=22/48, newline=22/48, rank=1.81

