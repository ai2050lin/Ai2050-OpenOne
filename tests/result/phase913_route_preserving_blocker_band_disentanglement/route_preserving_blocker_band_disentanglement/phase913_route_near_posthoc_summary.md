# Phase 913 route-near posthoc boundary summary

## qwen3

| subset | rows | unique cases | top5 | top10 | top50 | margin>=0 | weak | strong | weak families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all | 2322 | 11 | 0 | 0 | 0 | 0 | 76 | 22 | {'l0_attention_span': 76} |
| route_not_top50 | 2322 | 11 | 0 | 0 | 0 | 0 | 76 | 22 | {'l0_attention_span': 76} |

### Top weak examples

- p856_023_material_plastic plastic L0_attention_span_prompt_first8_scale_0.5: route_rank=1396 patched_rank=969 band16_delta=-0.31640625 eos_delta=0.6875 blocker= 

-> 

 margin=-10.375
- p885_041_material_rubber rubber L0_attention_span_prompt_first8_scale_0.5: route_rank=1624 patched_rank=1062 band16_delta=-0.62890625 eos_delta=0.5 blocker= 

-> 

 margin=-11.0625
- p856_004_geometry_circle circle L0_attention_span_prompt_first8_scale_0.25: route_rank=1886 patched_rank=1075 band16_delta=-0.42578125 eos_delta=1.0 blocker= 

-> The margin=-10.625
- p885_046_animal_cow cow L0_attention_span_prompt_first8_scale_0.5: route_rank=1384 patched_rank=1084 band16_delta=-0.3828125 eos_delta=0.5 blocker= The-> The margin=-10.5625
- p856_023_material_plastic plastic L0_attention_span_prompt_first8_scale_0.75: route_rank=1396 patched_rank=1174 band16_delta=-0.25 eos_delta=0.25 blocker= 

-> The margin=-10.6875
- p885_046_animal_cow cow L0_attention_span_prompt_all_scale_0.5: route_rank=1384 patched_rank=1208 band16_delta=-0.453125 eos_delta=0.25 blocker= The-> The margin=-10.4375
- p885_046_animal_cow cow L0_attention_span_prompt_first8_scale_0.75: route_rank=1384 patched_rank=1216 band16_delta=-0.35546875 eos_delta=0.125 blocker= The-> The margin=-10.8125
- p885_041_material_rubber rubber L0_attention_span_prompt_first8_scale_0.75: route_rank=1624 patched_rank=1348 band16_delta=-0.48046875 eos_delta=0.125 blocker= 

-> The margin=-11.4375
- p856_005_geometry_polygon polygon L0_attention_span_prompt_first8_scale_0.25: route_rank=2060 patched_rank=1353 band16_delta=-0.80859375 eos_delta=0.75 blocker= 

-> The margin=-10.75
- p856_004_geometry_circle circle L0_attention_span_prompt_all_scale_0.5: route_rank=1886 patched_rank=1489 band16_delta=-0.5546875 eos_delta=0.3125 blocker= 

-> 

 margin=-11.4375

## glm4

| subset | rows | unique cases | top5 | top10 | top50 | margin>=0 | weak | strong | weak families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all | 2193 | 8 | 2 | 466 | 1933 | 0 | 41 | 20 | {'l0_attention_span': 23, 'l0_attention_head': 2, 'l4_mlp_channel_group': 16} |
| route_top50 | 1935 | 8 | 0 | 462 | 1917 | 0 | 17 | 0 | {'l4_mlp_channel_group': 16, 'l0_attention_span': 1} |
| route_top10 | 516 | 1 | 0 | 404 | 512 | 0 | 0 | 0 | {} |
| route_not_top50 | 258 | 1 | 2 | 4 | 16 | 0 | 24 | 20 | {'l0_attention_span': 22, 'l0_attention_head': 2} |

### Top weak examples

- p885_049_animal_insect insect L4_mlp_channels_top_abs_64_scale_0.25: route_rank=12 patched_rank=7 band16_delta=-0.271484375 eos_delta=0.25 blocker=a->a margin=-1.59375
- p856_022_material_iron iron L4_mlp_channels_top_abs_64_scale_0.25: route_rank=15 patched_rank=9 band16_delta=-0.32421875 eos_delta=0.1875 blocker=a->a margin=-1.6875
- p885_048_animal_lizard lizard L4_mlp_channels_top_abs_64_scale_0.25: route_rank=17 patched_rank=11 band16_delta=-0.27734375 eos_delta=0.1875 blocker=a->a margin=-1.96875
- p856_023_material_plastic plastic L4_mlp_channels_band32_support_64_scale_0.25: route_rank=12 patched_rank=11 band16_delta=-0.255859375 eos_delta=0.0 blocker=a->a margin=-1.5625
- p885_048_animal_lizard lizard L4_mlp_channels_band16_support_32_scale_0.25: route_rank=17 patched_rank=12 band16_delta=-0.3046875 eos_delta=0.03125 blocker=a->a margin=-2.125
- p856_009_animal_fish fish L4_mlp_channels_top_abs_64_scale_0.25: route_rank=19 patched_rank=12 band16_delta=-0.27734375 eos_delta=0.25 blocker=a->a margin=-1.875
- p856_009_animal_fish fish L4_mlp_channels_top_abs_64_scale_0.25: route_rank=19 patched_rank=12 band16_delta=-0.27734375 eos_delta=0.25 blocker=a->a margin=-1.875
- p856_022_material_iron iron L0_attention_span_prompt_last8_scale_0.25: route_rank=15 patched_rank=12 band16_delta=-0.265625 eos_delta=0.03125 blocker=a->a margin=-1.84375
- p885_050_animal_whale whale L4_mlp_channels_top_abs_64_scale_0.25: route_rank=18 patched_rank=13 band16_delta=-0.35546875 eos_delta=0.0 blocker=a->a margin=-2.53125
- p885_048_animal_lizard lizard L4_mlp_channels_band32_support_64_scale_0.25: route_rank=17 patched_rank=13 band16_delta=-0.3359375 eos_delta=0.0 blocker=a->a margin=-2.15625

## deepseek7b

| subset | rows | unique cases | top5 | top10 | top50 | margin>=0 | weak | strong | weak families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all | 3861 | 12 | 31 | 37 | 49 | 0 | 124 | 101 | {'l0_attention_span': 118, 'l0_attention_head': 6} |
| route_not_top50 | 3861 | 12 | 31 | 37 | 49 | 0 | 124 | 101 | {'l0_attention_span': 118, 'l0_attention_head': 6} |

### Top weak examples

- p856_007_animal_cat cat L0_attention_span_prompt_last8_scale_0.25: route_rank=697 patched_rank=3 band16_delta=-5.3154296875 eos_delta=4.59375 blocker= Sub->
 margin=-1.375
- p856_007_animal_cat cat L0_attention_span_last8_before_period_scale_0.25: route_rank=697 patched_rank=3 band16_delta=-5.22265625 eos_delta=5.15625 blocker= Sub->
 margin=-0.5
- p856_009_animal_fish fish L0_attention_span_prompt_all_scale_0.25: route_rank=13727 patched_rank=4 band16_delta=-13.602313995361328 eos_delta=4.40625 blocker=</think>->
 margin=-3.125
- p885_047_animal_shark shark L0_attention_span_prompt_all_scale_0.25: route_rank=13361 patched_rank=4 band16_delta=-13.4130859375 eos_delta=5.375 blocker=The->
 margin=-3.6875
- p885_047_animal_shark shark L0_attention_span_prompt_all_scale_0.25: route_rank=13361 patched_rank=4 band16_delta=-13.4130859375 eos_delta=5.375 blocker=The->
 margin=-3.6875
- p856_001_geometry_triangle triangle L0_attention_span_prompt_all_scale_0.25: route_rank=24118 patched_rank=4 band16_delta=-13.111328125 eos_delta=5.765625 blocker=</think>->
 margin=-1.875
- p856_001_geometry_triangle triangle L0_attention_span_prompt_all_scale_0.25: route_rank=24118 patched_rank=4 band16_delta=-13.111328125 eos_delta=5.765625 blocker=</think>->
 margin=-1.875
- p856_002_geometry_square square L0_attention_span_prompt_all_scale_0.25: route_rank=30361 patched_rank=4 band16_delta=-13.076171875 eos_delta=6.0625 blocker=Category->
 margin=-1.8125
- p856_002_geometry_square square L0_attention_span_prompt_all_scale_0.25: route_rank=30361 patched_rank=4 band16_delta=-13.076171875 eos_delta=6.0625 blocker=Category->
 margin=-1.8125
- p856_002_geometry_square square L0_attention_span_prompt_all_scale_0.25: route_rank=30361 patched_rank=4 band16_delta=-13.076171875 eos_delta=6.0625 blocker=Category->
 margin=-1.8125
