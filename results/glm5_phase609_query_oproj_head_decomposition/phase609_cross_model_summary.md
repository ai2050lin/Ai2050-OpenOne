# Phase609 Cross-Model Summary

Query / o_proj-input / head-slot decomposition.

## qwen3

cases=96, rows=7, target_cases_seen=7, layers=[29], heads={'29': 32}, time_min=1.03

### Best Patches

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L29|o_input_delta` | L29 | o_input_delta |  | 6/7 | 2.055 | 1.173 | -0.882 |
| `L29|head_delta|H11` | L29 | head_delta | H11 | 5/7 | 1.894 | 1.149 | -0.745 |
| `L29|q_delta` | L29 | q_delta |  | 3/7 | 1.393 | 0.812 | -0.581 |
| `L29|head_delta|H23` | L29 | head_delta | H23 | 2/7 | 0.018 | -0.002 | -0.019 |
| `L29|head_delta|H6` | L29 | head_delta | H6 | 1/7 | 0.340 | 0.271 | -0.068 |
| `L29|head_delta|H14` | L29 | head_delta | H14 | 1/7 | 0.054 | 0.035 | -0.018 |
| `L29|head_delta|H5` | L29 | head_delta | H5 | 1/7 | 0.053 | 0.028 | -0.025 |
| `L29|head_delta|H2` | L29 | head_delta | H2 | 1/7 | 0.036 | 0.020 | -0.016 |
| `L29|head_delta|H10` | L29 | head_delta | H10 | 1/7 | 0.036 | 0.017 | -0.018 |
| `L29|head_delta|H30` | L29 | head_delta | H30 | 1/7 | 0.036 | 0.013 | -0.023 |
| `L29|head_random|H29` | L29 | head_random | H29 | 1/7 | 0.028 | 0.020 | -0.009 |
| `L29|head_delta|H21` | L29 | head_delta | H21 | 1/7 | 0.018 | 0.004 | -0.014 |
| `L29|head_delta|H1` | L29 | head_delta | H1 | 1/7 | 0.018 | 0.007 | -0.011 |
| `L29|head_delta|H29` | L29 | head_delta | H29 | 1/7 | 0.018 | 0.001 | -0.017 |
| `L29|head_delta|H7` | L29 | head_delta | H7 | 1/7 | 0.018 | 0.008 | -0.010 |
| `L29|head_delta|H4` | L29 | head_delta | H4 | 1/7 | 0.018 | -0.034 | -0.052 |
| `L29|head_random|H12` | L29 | head_random | H12 | 1/7 | 0.018 | 0.005 | -0.012 |
| `L29|head_random|H1` | L29 | head_random | H1 | 1/7 | 0.017 | 0.008 | -0.009 |
| `L29|head_random|H14` | L29 | head_random | H14 | 1/7 | 0.001 | -0.013 | -0.015 |
| `L29|head_delta|H24` | L29 | head_delta | H24 | 1/7 | 0.000 | -0.009 | -0.009 |
| `L29|head_delta|H20` | L29 | head_delta | H20 | 1/7 | -0.000 | -0.006 | -0.006 |
| `L29|head_delta|H13` | L29 | head_delta | H13 | 1/7 | -0.018 | -0.019 | -0.001 |
| `L29|q_random` | L29 | q_random |  | 1/7 | -0.065 | -0.171 | -0.106 |
| `L29|head_random|H16` | L29 | head_random | H16 | 0/7 | 0.057 | 0.047 | -0.010 |
| `L29|head_random|H17` | L29 | head_random | H17 | 0/7 | 0.046 | 0.044 | -0.003 |
| `L29|head_random|H19` | L29 | head_random | H19 | 0/7 | 0.040 | 0.029 | -0.011 |
| `L29|head_random|H11` | L29 | head_random | H11 | 0/7 | 0.037 | 0.016 | -0.021 |
| `L29|head_random|H5` | L29 | head_random | H5 | 0/7 | 0.037 | 0.034 | -0.003 |
| `L29|head_random|H8` | L29 | head_random | H8 | 0/7 | 0.032 | 0.022 | -0.010 |
| `L29|head_random|H18` | L29 | head_random | H18 | 0/7 | 0.031 | 0.025 | -0.006 |
| `L29|head_random|H21` | L29 | head_random | H21 | 0/7 | 0.030 | 0.013 | -0.016 |
| `L29|head_random|H23` | L29 | head_random | H23 | 0/7 | 0.020 | 0.010 | -0.010 |
| `L29|head_random|H20` | L29 | head_random | H20 | 0/7 | 0.020 | 0.015 | -0.005 |
| `L29|head_random|H24` | L29 | head_random | H24 | 0/7 | 0.019 | 0.008 | -0.011 |
| `L29|head_random|H6` | L29 | head_random | H6 | 0/7 | 0.018 | 0.013 | -0.005 |
| `L29|head_delta|H31` | L29 | head_delta | H31 | 0/7 | 0.018 | 0.006 | -0.012 |
| `L29|head_delta|H17` | L29 | head_delta | H17 | 0/7 | 0.018 | 0.013 | -0.005 |
| `L29|head_delta|H3` | L29 | head_delta | H3 | 0/7 | 0.018 | 0.015 | -0.003 |
| `L29|head_delta|H25` | L29 | head_delta | H25 | 0/7 | 0.018 | 0.016 | -0.002 |
| `L29|head_delta|H18` | L29 | head_delta | H18 | 0/7 | 0.018 | 0.009 | -0.009 |
| `L29|head_random|H26` | L29 | head_random | H26 | 0/7 | 0.015 | 0.004 | -0.011 |
| `L29|head_random|H15` | L29 | head_random | H15 | 0/7 | 0.015 | 0.001 | -0.014 |
| `L29|head_random|H30` | L29 | head_random | H30 | 0/7 | 0.002 | -0.007 | -0.009 |
| `L29|head_delta|H12` | L29 | head_delta | H12 | 0/7 | 0.000 | 0.001 | 0.001 |
| `L29|head_delta|H27` | L29 | head_delta | H27 | 0/7 | 0.000 | 0.000 | 0.000 |
| `L29|head_delta|H16` | L29 | head_delta | H16 | 0/7 | 0.000 | -0.008 | -0.008 |
| `L29|head_delta|H26` | L29 | head_delta | H26 | 0/7 | -0.000 | -0.007 | -0.007 |
| `L29|head_delta|H22` | L29 | head_delta | H22 | 0/7 | -0.000 | -0.005 | -0.005 |

### Core Modes

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L29|q_delta` | L29 | q_delta |  | 3/7 | 1.393 | 0.812 | -0.581 |
| `L29|q_random` | L29 | q_random |  | 1/7 | -0.065 | -0.171 | -0.106 |
| `L29|o_input_delta` | L29 | o_input_delta |  | 6/7 | 2.055 | 1.173 | -0.882 |
| `L29|o_input_random` | L29 | o_input_random |  | 0/7 | -0.020 | -0.057 | -0.037 |

### Head Delta Ranking

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L29|head_delta|H11` | L29 | head_delta | H11 | 5/7 | 1.894 | 1.149 | -0.745 |
| `L29|head_delta|H23` | L29 | head_delta | H23 | 2/7 | 0.018 | -0.002 | -0.019 |
| `L29|head_delta|H6` | L29 | head_delta | H6 | 1/7 | 0.340 | 0.271 | -0.068 |
| `L29|head_delta|H14` | L29 | head_delta | H14 | 1/7 | 0.054 | 0.035 | -0.018 |
| `L29|head_delta|H5` | L29 | head_delta | H5 | 1/7 | 0.053 | 0.028 | -0.025 |
| `L29|head_delta|H2` | L29 | head_delta | H2 | 1/7 | 0.036 | 0.020 | -0.016 |
| `L29|head_delta|H10` | L29 | head_delta | H10 | 1/7 | 0.036 | 0.017 | -0.018 |
| `L29|head_delta|H30` | L29 | head_delta | H30 | 1/7 | 0.036 | 0.013 | -0.023 |
| `L29|head_delta|H21` | L29 | head_delta | H21 | 1/7 | 0.018 | 0.004 | -0.014 |
| `L29|head_delta|H1` | L29 | head_delta | H1 | 1/7 | 0.018 | 0.007 | -0.011 |
| `L29|head_delta|H29` | L29 | head_delta | H29 | 1/7 | 0.018 | 0.001 | -0.017 |
| `L29|head_delta|H7` | L29 | head_delta | H7 | 1/7 | 0.018 | 0.008 | -0.010 |
| `L29|head_delta|H4` | L29 | head_delta | H4 | 1/7 | 0.018 | -0.034 | -0.052 |
| `L29|head_delta|H24` | L29 | head_delta | H24 | 1/7 | 0.000 | -0.009 | -0.009 |
| `L29|head_delta|H20` | L29 | head_delta | H20 | 1/7 | -0.000 | -0.006 | -0.006 |
| `L29|head_delta|H13` | L29 | head_delta | H13 | 1/7 | -0.018 | -0.019 | -0.001 |
| `L29|head_delta|H31` | L29 | head_delta | H31 | 0/7 | 0.018 | 0.006 | -0.012 |
| `L29|head_delta|H17` | L29 | head_delta | H17 | 0/7 | 0.018 | 0.013 | -0.005 |
| `L29|head_delta|H3` | L29 | head_delta | H3 | 0/7 | 0.018 | 0.015 | -0.003 |
| `L29|head_delta|H25` | L29 | head_delta | H25 | 0/7 | 0.018 | 0.016 | -0.002 |
| `L29|head_delta|H18` | L29 | head_delta | H18 | 0/7 | 0.018 | 0.009 | -0.009 |
| `L29|head_delta|H12` | L29 | head_delta | H12 | 0/7 | 0.000 | 0.001 | 0.001 |
| `L29|head_delta|H27` | L29 | head_delta | H27 | 0/7 | 0.000 | 0.000 | 0.000 |
| `L29|head_delta|H16` | L29 | head_delta | H16 | 0/7 | 0.000 | -0.008 | -0.008 |
| `L29|head_delta|H26` | L29 | head_delta | H26 | 0/7 | -0.000 | -0.007 | -0.007 |
| `L29|head_delta|H22` | L29 | head_delta | H22 | 0/7 | -0.000 | -0.005 | -0.005 |
| `L29|head_delta|H8` | L29 | head_delta | H8 | 0/7 | -0.018 | -0.017 | 0.001 |
| `L29|head_delta|H15` | L29 | head_delta | H15 | 0/7 | -0.018 | -0.012 | 0.006 |
| `L29|head_delta|H28` | L29 | head_delta | H28 | 0/7 | -0.018 | -0.020 | -0.002 |
| `L29|head_delta|H19` | L29 | head_delta | H19 | 0/7 | -0.036 | -0.033 | 0.002 |
| `L29|head_delta|H0` | L29 | head_delta | H0 | 0/7 | -0.054 | -0.046 | 0.007 |
| `L29|head_delta|H9` | L29 | head_delta | H9 | 0/7 | -0.179 | -0.145 | 0.033 |

## glm4

cases=96, rows=13, target_cases_seen=13, layers=[34], heads={'34': 32}, time_min=2.60

### Best Patches

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L34|o_input_delta` | L34 | o_input_delta |  | 3/13 | 0.173 | 0.090 | -0.083 |
| `L34|head_delta|H12` | L34 | head_delta | H12 | 1/13 | 0.125 | 0.066 | -0.059 |
| `L34|head_delta|H8` | L34 | head_delta | H8 | 1/13 | 0.067 | 0.035 | -0.033 |
| `L34|head_delta|H4` | L34 | head_delta | H4 | 1/13 | 0.063 | 0.033 | -0.029 |
| `L34|head_delta|H28` | L34 | head_delta | H28 | 1/13 | 0.029 | 0.020 | -0.009 |
| `L34|head_delta|H6` | L34 | head_delta | H6 | 1/13 | 0.024 | 0.015 | -0.009 |
| `L34|head_delta|H7` | L34 | head_delta | H7 | 1/13 | 0.024 | 0.015 | -0.009 |
| `L34|head_delta|H17` | L34 | head_delta | H17 | 1/13 | 0.014 | 0.009 | -0.005 |
| `L34|head_delta|H14` | L34 | head_delta | H14 | 1/13 | 0.010 | 0.009 | -0.001 |
| `L34|head_delta|H20` | L34 | head_delta | H20 | 1/13 | 0.005 | 0.001 | -0.003 |
| `L34|q_delta` | L34 | q_delta |  | 1/13 | -0.048 | -0.090 | -0.042 |
| `L34|q_random` | L34 | q_random |  | 0/13 | 0.050 | 0.003 | -0.047 |
| `L34|head_delta|H15` | L34 | head_delta | H15 | 0/13 | 0.038 | 0.023 | -0.015 |
| `L34|head_random|H3` | L34 | head_random | H3 | 0/13 | 0.020 | 0.011 | -0.009 |
| `L34|head_random|H6` | L34 | head_random | H6 | 0/13 | 0.019 | 0.016 | -0.003 |
| `L34|head_delta|H13` | L34 | head_delta | H13 | 0/13 | 0.019 | 0.019 | -0.000 |
| `L34|head_random|H24` | L34 | head_random | H24 | 0/13 | 0.015 | 0.014 | -0.001 |
| `L34|head_delta|H5` | L34 | head_delta | H5 | 0/13 | 0.014 | 0.013 | -0.002 |
| `L34|head_delta|H24` | L34 | head_delta | H24 | 0/13 | 0.014 | 0.009 | -0.006 |
| `L34|head_random|H4` | L34 | head_random | H4 | 0/13 | 0.014 | 0.002 | -0.013 |
| `L34|head_random|H23` | L34 | head_random | H23 | 0/13 | 0.012 | 0.015 | 0.003 |
| `L34|head_random|H2` | L34 | head_random | H2 | 0/13 | 0.012 | 0.004 | -0.008 |
| `L34|head_random|H7` | L34 | head_random | H7 | 0/13 | 0.010 | 0.008 | -0.002 |
| `L34|head_delta|H19` | L34 | head_delta | H19 | 0/13 | 0.010 | 0.007 | -0.002 |
| `L34|head_random|H13` | L34 | head_random | H13 | 0/13 | 0.008 | 0.008 | -0.000 |
| `L34|head_random|H18` | L34 | head_random | H18 | 0/13 | 0.006 | 0.005 | -0.002 |
| `L34|head_random|H27` | L34 | head_random | H27 | 0/13 | 0.006 | 0.006 | -0.000 |
| `L34|head_random|H0` | L34 | head_random | H0 | 0/13 | 0.006 | 0.003 | -0.003 |
| `L34|head_random|H8` | L34 | head_random | H8 | 0/13 | 0.006 | 0.005 | -0.001 |
| `L34|head_random|H16` | L34 | head_random | H16 | 0/13 | 0.006 | 0.011 | 0.006 |
| `L34|o_input_random` | L34 | o_input_random |  | 0/13 | 0.005 | 0.016 | 0.011 |
| `L34|head_delta|H11` | L34 | head_delta | H11 | 0/13 | 0.005 | 0.001 | -0.004 |
| `L34|head_random|H17` | L34 | head_random | H17 | 0/13 | 0.004 | 0.003 | -0.001 |
| `L34|head_random|H15` | L34 | head_random | H15 | 0/13 | 0.003 | 0.002 | -0.002 |
| `L34|head_random|H5` | L34 | head_random | H5 | 0/13 | 0.003 | 0.001 | -0.002 |
| `L34|head_random|H26` | L34 | head_random | H26 | 0/13 | 0.002 | 0.004 | 0.002 |
| `L34|head_delta|H21` | L34 | head_delta | H21 | 0/13 | 0.000 | -0.002 | -0.002 |
| `L34|head_delta|H16` | L34 | head_delta | H16 | 0/13 | 0.000 | 0.000 | 0.000 |
| `L34|head_delta|H30` | L34 | head_delta | H30 | 0/13 | 0.000 | 0.002 | 0.002 |
| `L34|head_random|H11` | L34 | head_random | H11 | 0/13 | -0.001 | 0.000 | 0.001 |
| `L34|head_random|H19` | L34 | head_random | H19 | 0/13 | -0.001 | 0.010 | 0.011 |
| `L34|head_random|H1` | L34 | head_random | H1 | 0/13 | -0.004 | -0.005 | -0.001 |
| `L34|head_random|H31` | L34 | head_random | H31 | 0/13 | -0.004 | -0.001 | 0.003 |
| `L34|head_random|H29` | L34 | head_random | H29 | 0/13 | -0.005 | -0.008 | -0.004 |
| `L34|head_delta|H0` | L34 | head_delta | H0 | 0/13 | -0.005 | -0.003 | 0.002 |
| `L34|head_delta|H18` | L34 | head_delta | H18 | 0/13 | -0.005 | -0.002 | 0.003 |
| `L34|head_delta|H22` | L34 | head_delta | H22 | 0/13 | -0.005 | -0.001 | 0.004 |
| `L34|head_delta|H31` | L34 | head_delta | H31 | 0/13 | -0.005 | -0.003 | 0.002 |

### Core Modes

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L34|q_delta` | L34 | q_delta |  | 1/13 | -0.048 | -0.090 | -0.042 |
| `L34|q_random` | L34 | q_random |  | 0/13 | 0.050 | 0.003 | -0.047 |
| `L34|o_input_delta` | L34 | o_input_delta |  | 3/13 | 0.173 | 0.090 | -0.083 |
| `L34|o_input_random` | L34 | o_input_random |  | 0/13 | 0.005 | 0.016 | 0.011 |

### Head Delta Ranking

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L34|head_delta|H12` | L34 | head_delta | H12 | 1/13 | 0.125 | 0.066 | -0.059 |
| `L34|head_delta|H8` | L34 | head_delta | H8 | 1/13 | 0.067 | 0.035 | -0.033 |
| `L34|head_delta|H4` | L34 | head_delta | H4 | 1/13 | 0.063 | 0.033 | -0.029 |
| `L34|head_delta|H28` | L34 | head_delta | H28 | 1/13 | 0.029 | 0.020 | -0.009 |
| `L34|head_delta|H6` | L34 | head_delta | H6 | 1/13 | 0.024 | 0.015 | -0.009 |
| `L34|head_delta|H7` | L34 | head_delta | H7 | 1/13 | 0.024 | 0.015 | -0.009 |
| `L34|head_delta|H17` | L34 | head_delta | H17 | 1/13 | 0.014 | 0.009 | -0.005 |
| `L34|head_delta|H14` | L34 | head_delta | H14 | 1/13 | 0.010 | 0.009 | -0.001 |
| `L34|head_delta|H20` | L34 | head_delta | H20 | 1/13 | 0.005 | 0.001 | -0.003 |
| `L34|head_delta|H15` | L34 | head_delta | H15 | 0/13 | 0.038 | 0.023 | -0.015 |
| `L34|head_delta|H13` | L34 | head_delta | H13 | 0/13 | 0.019 | 0.019 | -0.000 |
| `L34|head_delta|H5` | L34 | head_delta | H5 | 0/13 | 0.014 | 0.013 | -0.002 |
| `L34|head_delta|H24` | L34 | head_delta | H24 | 0/13 | 0.014 | 0.009 | -0.006 |
| `L34|head_delta|H19` | L34 | head_delta | H19 | 0/13 | 0.010 | 0.007 | -0.002 |
| `L34|head_delta|H11` | L34 | head_delta | H11 | 0/13 | 0.005 | 0.001 | -0.004 |
| `L34|head_delta|H21` | L34 | head_delta | H21 | 0/13 | 0.000 | -0.002 | -0.002 |
| `L34|head_delta|H16` | L34 | head_delta | H16 | 0/13 | 0.000 | 0.000 | 0.000 |
| `L34|head_delta|H30` | L34 | head_delta | H30 | 0/13 | 0.000 | 0.002 | 0.002 |
| `L34|head_delta|H0` | L34 | head_delta | H0 | 0/13 | -0.005 | -0.003 | 0.002 |
| `L34|head_delta|H18` | L34 | head_delta | H18 | 0/13 | -0.005 | -0.002 | 0.003 |
| `L34|head_delta|H22` | L34 | head_delta | H22 | 0/13 | -0.005 | -0.001 | 0.004 |
| `L34|head_delta|H31` | L34 | head_delta | H31 | 0/13 | -0.005 | -0.003 | 0.002 |
| `L34|head_delta|H10` | L34 | head_delta | H10 | 0/13 | -0.010 | -0.007 | 0.002 |
| `L34|head_delta|H29` | L34 | head_delta | H29 | 0/13 | -0.010 | -0.004 | 0.005 |
| `L34|head_delta|H27` | L34 | head_delta | H27 | 0/13 | -0.010 | -0.005 | 0.005 |
| `L34|head_delta|H1` | L34 | head_delta | H1 | 0/13 | -0.014 | -0.006 | 0.008 |
| `L34|head_delta|H9` | L34 | head_delta | H9 | 0/13 | -0.014 | -0.008 | 0.007 |
| `L34|head_delta|H23` | L34 | head_delta | H23 | 0/13 | -0.014 | -0.009 | 0.006 |
| `L34|head_delta|H25` | L34 | head_delta | H25 | 0/13 | -0.014 | -0.006 | 0.008 |
| `L34|head_delta|H3` | L34 | head_delta | H3 | 0/13 | -0.019 | -0.013 | 0.006 |
| `L34|head_delta|H26` | L34 | head_delta | H26 | 0/13 | -0.024 | -0.013 | 0.011 |
| `L34|head_delta|H2` | L34 | head_delta | H2 | 0/13 | -0.192 | -0.129 | 0.063 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, layers=[22], heads={'22': 28}, time_min=4.79

### Best Patches

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L22|o_input_delta` | L22 | o_input_delta |  | 33/37 | 3.428 | 1.585 | -1.843 |
| `L22|head_delta|H3` | L22 | head_delta | H3 | 16/37 | 1.516 | 0.796 | -0.721 |
| `L22|head_delta|H1` | L22 | head_delta | H1 | 8/37 | 0.759 | 0.540 | -0.219 |
| `L22|head_delta|H7` | L22 | head_delta | H7 | 5/37 | 0.547 | 0.436 | -0.111 |
| `L22|head_delta|H24` | L22 | head_delta | H24 | 4/37 | 0.229 | 0.189 | -0.039 |
| `L22|head_delta|H25` | L22 | head_delta | H25 | 3/37 | 0.067 | 0.077 | 0.010 |
| `L22|head_delta|H13` | L22 | head_delta | H13 | 2/37 | 0.294 | 0.232 | -0.062 |
| `L22|q_delta` | L22 | q_delta |  | 2/37 | 0.121 | -0.084 | -0.206 |
| `L22|head_random|H7` | L22 | head_random | H7 | 2/37 | -0.003 | 0.002 | 0.005 |
| `L22|head_delta|H2` | L22 | head_delta | H2 | 1/37 | 0.076 | 0.031 | -0.046 |
| `L22|head_delta|H23` | L22 | head_delta | H23 | 1/37 | 0.046 | 0.027 | -0.019 |
| `L22|head_random|H23` | L22 | head_random | H23 | 1/37 | 0.000 | -0.014 | -0.014 |
| `L22|head_random|H16` | L22 | head_random | H16 | 1/37 | -0.010 | -0.017 | -0.008 |
| `L22|head_random|H4` | L22 | head_random | H4 | 1/37 | -0.010 | -0.015 | -0.005 |
| `L22|head_delta|H16` | L22 | head_delta | H16 | 1/37 | -0.019 | -0.019 | -0.001 |
| `L22|head_random|H11` | L22 | head_random | H11 | 1/37 | -0.021 | -0.020 | 0.001 |
| `L22|head_random|H15` | L22 | head_random | H15 | 1/37 | -0.021 | -0.020 | 0.001 |
| `L22|head_random|H1` | L22 | head_random | H1 | 1/37 | -0.021 | -0.013 | 0.009 |
| `L22|head_random|H21` | L22 | head_random | H21 | 1/37 | -0.022 | -0.023 | -0.001 |
| `L22|head_random|H24` | L22 | head_random | H24 | 1/37 | -0.028 | -0.033 | -0.005 |
| `L22|q_random` | L22 | q_random |  | 1/37 | -0.118 | -0.195 | -0.076 |
| `L22|o_input_random` | L22 | o_input_random |  | 0/37 | 0.024 | -0.020 | -0.044 |
| `L22|head_delta|H4` | L22 | head_delta | H4 | 0/37 | 0.014 | 0.010 | -0.003 |
| `L22|head_delta|H20` | L22 | head_delta | H20 | 0/37 | 0.007 | 0.006 | -0.001 |
| `L22|head_random|H10` | L22 | head_random | H10 | 0/37 | 0.005 | -0.001 | -0.006 |
| `L22|head_delta|H10` | L22 | head_delta | H10 | 0/37 | 0.004 | 0.003 | -0.001 |
| `L22|head_delta|H22` | L22 | head_delta | H22 | 0/37 | 0.002 | 0.003 | 0.000 |
| `L22|head_random|H5` | L22 | head_random | H5 | 0/37 | 0.002 | -0.003 | -0.005 |
| `L22|head_delta|H6` | L22 | head_delta | H6 | 0/37 | -0.000 | -0.009 | -0.008 |
| `L22|head_delta|H8` | L22 | head_delta | H8 | 0/37 | -0.005 | -0.010 | -0.005 |
| `L22|head_random|H2` | L22 | head_random | H2 | 0/37 | -0.008 | -0.007 | 0.001 |
| `L22|head_random|H12` | L22 | head_random | H12 | 0/37 | -0.008 | -0.002 | 0.006 |
| `L22|head_random|H22` | L22 | head_random | H22 | 0/37 | -0.008 | -0.006 | 0.003 |
| `L22|head_delta|H26` | L22 | head_delta | H26 | 0/37 | -0.010 | -0.009 | 0.000 |
| `L22|head_random|H14` | L22 | head_random | H14 | 0/37 | -0.010 | -0.012 | -0.001 |
| `L22|head_random|H20` | L22 | head_random | H20 | 0/37 | -0.011 | -0.009 | 0.001 |
| `L22|head_delta|H15` | L22 | head_delta | H15 | 0/37 | -0.011 | -0.009 | 0.002 |
| `L22|head_delta|H14` | L22 | head_delta | H14 | 0/37 | -0.013 | -0.012 | 0.000 |
| `L22|head_delta|H19` | L22 | head_delta | H19 | 0/37 | -0.014 | -0.014 | -0.000 |
| `L22|head_random|H6` | L22 | head_random | H6 | 0/37 | -0.014 | -0.008 | 0.006 |
| `L22|head_random|H26` | L22 | head_random | H26 | 0/37 | -0.014 | -0.026 | -0.012 |
| `L22|head_delta|H18` | L22 | head_delta | H18 | 0/37 | -0.015 | -0.013 | 0.001 |
| `L22|head_random|H9` | L22 | head_random | H9 | 0/37 | -0.017 | -0.021 | -0.004 |
| `L22|head_random|H19` | L22 | head_random | H19 | 0/37 | -0.020 | -0.018 | 0.001 |
| `L22|head_delta|H11` | L22 | head_delta | H11 | 0/37 | -0.022 | -0.014 | 0.008 |
| `L22|head_random|H0` | L22 | head_random | H0 | 0/37 | -0.026 | -0.022 | 0.004 |
| `L22|head_random|H17` | L22 | head_random | H17 | 0/37 | -0.027 | -0.014 | 0.013 |
| `L22|head_random|H27` | L22 | head_random | H27 | 0/37 | -0.027 | -0.025 | 0.002 |

### Core Modes

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L22|q_delta` | L22 | q_delta |  | 2/37 | 0.121 | -0.084 | -0.206 |
| `L22|q_random` | L22 | q_random |  | 1/37 | -0.118 | -0.195 | -0.076 |
| `L22|o_input_delta` | L22 | o_input_delta |  | 33/37 | 3.428 | 1.585 | -1.843 |
| `L22|o_input_random` | L22 | o_input_random |  | 0/37 | 0.024 | -0.020 | -0.044 |

### Head Delta Ranking

| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L22|head_delta|H3` | L22 | head_delta | H3 | 16/37 | 1.516 | 0.796 | -0.721 |
| `L22|head_delta|H1` | L22 | head_delta | H1 | 8/37 | 0.759 | 0.540 | -0.219 |
| `L22|head_delta|H7` | L22 | head_delta | H7 | 5/37 | 0.547 | 0.436 | -0.111 |
| `L22|head_delta|H24` | L22 | head_delta | H24 | 4/37 | 0.229 | 0.189 | -0.039 |
| `L22|head_delta|H25` | L22 | head_delta | H25 | 3/37 | 0.067 | 0.077 | 0.010 |
| `L22|head_delta|H13` | L22 | head_delta | H13 | 2/37 | 0.294 | 0.232 | -0.062 |
| `L22|head_delta|H2` | L22 | head_delta | H2 | 1/37 | 0.076 | 0.031 | -0.046 |
| `L22|head_delta|H23` | L22 | head_delta | H23 | 1/37 | 0.046 | 0.027 | -0.019 |
| `L22|head_delta|H16` | L22 | head_delta | H16 | 1/37 | -0.019 | -0.019 | -0.001 |
| `L22|head_delta|H4` | L22 | head_delta | H4 | 0/37 | 0.014 | 0.010 | -0.003 |
| `L22|head_delta|H20` | L22 | head_delta | H20 | 0/37 | 0.007 | 0.006 | -0.001 |
| `L22|head_delta|H10` | L22 | head_delta | H10 | 0/37 | 0.004 | 0.003 | -0.001 |
| `L22|head_delta|H22` | L22 | head_delta | H22 | 0/37 | 0.002 | 0.003 | 0.000 |
| `L22|head_delta|H6` | L22 | head_delta | H6 | 0/37 | -0.000 | -0.009 | -0.008 |
| `L22|head_delta|H8` | L22 | head_delta | H8 | 0/37 | -0.005 | -0.010 | -0.005 |
| `L22|head_delta|H26` | L22 | head_delta | H26 | 0/37 | -0.010 | -0.009 | 0.000 |
| `L22|head_delta|H15` | L22 | head_delta | H15 | 0/37 | -0.011 | -0.009 | 0.002 |
| `L22|head_delta|H14` | L22 | head_delta | H14 | 0/37 | -0.013 | -0.012 | 0.000 |
| `L22|head_delta|H19` | L22 | head_delta | H19 | 0/37 | -0.014 | -0.014 | -0.000 |
| `L22|head_delta|H18` | L22 | head_delta | H18 | 0/37 | -0.015 | -0.013 | 0.001 |
| `L22|head_delta|H11` | L22 | head_delta | H11 | 0/37 | -0.022 | -0.014 | 0.008 |
| `L22|head_delta|H17` | L22 | head_delta | H17 | 0/37 | -0.027 | -0.024 | 0.003 |
| `L22|head_delta|H9` | L22 | head_delta | H9 | 0/37 | -0.033 | -0.038 | -0.005 |
| `L22|head_delta|H5` | L22 | head_delta | H5 | 0/37 | -0.038 | -0.002 | 0.036 |
| `L22|head_delta|H27` | L22 | head_delta | H27 | 0/37 | -0.040 | -0.008 | 0.033 |
| `L22|head_delta|H0` | L22 | head_delta | H0 | 0/37 | -0.060 | -0.041 | 0.019 |
| `L22|head_delta|H21` | L22 | head_delta | H21 | 0/37 | -0.067 | -0.053 | 0.014 |
| `L22|head_delta|H12` | L22 | head_delta | H12 | 0/37 | -0.161 | -0.134 | 0.027 |

