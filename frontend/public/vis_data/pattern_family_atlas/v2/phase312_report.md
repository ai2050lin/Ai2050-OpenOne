# Phase312 Matched Path Feature Analysis

## Independent Evidence

- independent_model_cases: 360
- layer_component_rows: 37440
- heldout_prediction_rows: 72
- heldout_family_accuracy: 0.847222
- heldout_mechanism_accuracy: 0.736111

## Adjusted Reuse By Family

- content_knowledge: 0.011873
- reasoning_constraint: 0.100799
- syntax_structure: 0.016024

## Model / Family / Position / Component

- deepseek7b / content_knowledge / last / attention: adjusted=0.020682, peak_depth=0.72963, persistence=0.350893
- deepseek7b / content_knowledge / last / mlp: adjusted=-0.00017, peak_depth=0.52037, persistence=0.480357
- deepseek7b / content_knowledge / last / residual: adjusted=0.00196, peak_depth=0.622222, persistence=0.0
- deepseek7b / content_knowledge / query / attention: adjusted=0.04587, peak_depth=0.451852, persistence=0.342857
- deepseek7b / content_knowledge / query / mlp: adjusted=0.051178, peak_depth=0.551852, persistence=0.442857
- deepseek7b / content_knowledge / query / residual: adjusted=-0.005778, peak_depth=0.607407, persistence=0.0
- deepseek7b / content_knowledge / source / attention: adjusted=-0.029666, peak_depth=0.212037, persistence=0.275893
- deepseek7b / content_knowledge / source / mlp: adjusted=0.000328, peak_depth=0.397222, persistence=0.426786
- deepseek7b / content_knowledge / source / residual: adjusted=0.032402, peak_depth=0.378704, persistence=0.0
- deepseek7b / reasoning_constraint / last / attention: adjusted=0.011855, peak_depth=0.0, persistence=0.30625
- deepseek7b / reasoning_constraint / last / mlp: adjusted=0.036878, peak_depth=0.487963, persistence=0.605357
- deepseek7b / reasoning_constraint / last / residual: adjusted=0.012176, peak_depth=0.551852, persistence=0.0
- deepseek7b / reasoning_constraint / query / attention: adjusted=0.035218, peak_depth=0.010185, persistence=0.320536
- deepseek7b / reasoning_constraint / query / mlp: adjusted=0.039325, peak_depth=0.544444, persistence=0.484821
- deepseek7b / reasoning_constraint / query / residual: adjusted=-0.047075, peak_depth=0.505555, persistence=0.0
- deepseek7b / reasoning_constraint / source / attention: adjusted=0.413809, peak_depth=0.096296, persistence=0.330357
- deepseek7b / reasoning_constraint / source / mlp: adjusted=0.288544, peak_depth=0.312963, persistence=0.532143
- deepseek7b / reasoning_constraint / source / residual: adjusted=0.274359, peak_depth=0.197222, persistence=0.0
- deepseek7b / syntax_structure / last / attention: adjusted=0.078904, peak_depth=0.476852, persistence=0.344643
- deepseek7b / syntax_structure / last / mlp: adjusted=0.019537, peak_depth=0.55463, persistence=0.459821
- deepseek7b / syntax_structure / last / residual: adjusted=0.003802, peak_depth=0.671296, persistence=0.0
- deepseek7b / syntax_structure / query / attention: adjusted=-0.059111, peak_depth=0.160185, persistence=0.379464
- deepseek7b / syntax_structure / query / mlp: adjusted=-0.009621, peak_depth=0.294444, persistence=0.436607
- deepseek7b / syntax_structure / query / residual: adjusted=-0.006902, peak_depth=0.550926, persistence=0.0
- deepseek7b / syntax_structure / source / attention: adjusted=0.040231, peak_depth=0.246296, persistence=0.355357
- deepseek7b / syntax_structure / source / mlp: adjusted=0.055698, peak_depth=0.433333, persistence=0.449107
- deepseek7b / syntax_structure / source / residual: adjusted=-0.027325, peak_depth=0.500926, persistence=0.0
- glm4 / content_knowledge / last / attention: adjusted=0.021648, peak_depth=0.551282, persistence=0.261875
- glm4 / content_knowledge / last / mlp: adjusted=-0.008223, peak_depth=0.762821, persistence=0.3875
- glm4 / content_knowledge / last / residual: adjusted=-0.036583, peak_depth=0.759616, persistence=0.0
- glm4 / content_knowledge / query / attention: adjusted=0.087504, peak_depth=0.444872, persistence=0.384375
- glm4 / content_knowledge / query / mlp: adjusted=-0.012731, peak_depth=0.351282, persistence=0.46125
- glm4 / content_knowledge / query / residual: adjusted=-0.014075, peak_depth=0.59359, persistence=0.0
- glm4 / content_knowledge / source / attention: adjusted=-0.020067, peak_depth=0.224359, persistence=0.33
- glm4 / content_knowledge / source / mlp: adjusted=-0.004833, peak_depth=0.258333, persistence=0.4375
- glm4 / content_knowledge / source / residual: adjusted=-0.003938, peak_depth=0.48782, persistence=0.0
- glm4 / reasoning_constraint / last / attention: adjusted=0.013508, peak_depth=0.504487, persistence=0.52125
- glm4 / reasoning_constraint / last / mlp: adjusted=0.012608, peak_depth=0.746795, persistence=0.373125
- glm4 / reasoning_constraint / last / residual: adjusted=0.032458, peak_depth=0.791026, persistence=0.0
- glm4 / reasoning_constraint / query / attention: adjusted=0.096428, peak_depth=0.341666, persistence=0.36375
- glm4 / reasoning_constraint / query / mlp: adjusted=0.040523, peak_depth=0.171154, persistence=0.378125
- glm4 / reasoning_constraint / query / residual: adjusted=0.027484, peak_depth=0.58141, persistence=0.0
- glm4 / reasoning_constraint / source / attention: adjusted=0.212152, peak_depth=0.08782, persistence=0.295
- glm4 / reasoning_constraint / source / mlp: adjusted=0.233871, peak_depth=0.339102, persistence=0.4575
- glm4 / reasoning_constraint / source / residual: adjusted=0.174787, peak_depth=0.511538, persistence=0.0
- glm4 / syntax_structure / last / attention: adjusted=-0.035867, peak_depth=0.292308, persistence=0.278125
- glm4 / syntax_structure / last / mlp: adjusted=-0.006793, peak_depth=0.464744, persistence=0.431875
- glm4 / syntax_structure / last / residual: adjusted=-0.003909, peak_depth=0.700641, persistence=0.0
- glm4 / syntax_structure / query / attention: adjusted=-0.001113, peak_depth=0.283333, persistence=0.260625
- glm4 / syntax_structure / query / mlp: adjusted=-0.016515, peak_depth=0.351923, persistence=0.406875
- glm4 / syntax_structure / query / residual: adjusted=0.005396, peak_depth=0.461538, persistence=0.0
- glm4 / syntax_structure / source / attention: adjusted=-0.020679, peak_depth=0.135897, persistence=0.3025
- glm4 / syntax_structure / source / mlp: adjusted=0.029706, peak_depth=0.414744, persistence=0.38375
- glm4 / syntax_structure / source / residual: adjusted=0.032138, peak_depth=0.552564, persistence=0.0
- qwen3 / content_knowledge / last / attention: adjusted=-7.2e-05, peak_depth=0.373571, persistence=0.135417
- qwen3 / content_knowledge / last / mlp: adjusted=-0.022952, peak_depth=0.862143, persistence=0.375694
- qwen3 / content_knowledge / last / residual: adjusted=-0.015962, peak_depth=0.722857, persistence=0.0
- qwen3 / content_knowledge / query / attention: adjusted=-0.000307, peak_depth=0.23, persistence=0.233333
- qwen3 / content_knowledge / query / mlp: adjusted=0.080652, peak_depth=0.338572, persistence=0.3875
- qwen3 / content_knowledge / query / residual: adjusted=0.018123, peak_depth=0.547857, persistence=0.0
- qwen3 / content_knowledge / source / attention: adjusted=0.088123, peak_depth=0.057857, persistence=0.146528
- qwen3 / content_knowledge / source / mlp: adjusted=0.036436, peak_depth=0.245714, persistence=0.377778
- qwen3 / content_knowledge / source / residual: adjusted=0.01103, peak_depth=0.323571, persistence=0.0
- qwen3 / reasoning_constraint / last / attention: adjusted=0.006136, peak_depth=0.0, persistence=0.059723
- qwen3 / reasoning_constraint / last / mlp: adjusted=0.037474, peak_depth=0.772857, persistence=0.486806
- qwen3 / reasoning_constraint / last / residual: adjusted=0.02453, peak_depth=0.736429, persistence=0.0
- qwen3 / reasoning_constraint / query / attention: adjusted=0.014864, peak_depth=0.012143, persistence=0.220833
- qwen3 / reasoning_constraint / query / mlp: adjusted=0.028365, peak_depth=0.291429, persistence=0.4
- qwen3 / reasoning_constraint / query / residual: adjusted=0.033609, peak_depth=0.359286, persistence=0.0
- qwen3 / reasoning_constraint / source / attention: adjusted=0.154479, peak_depth=0.0, persistence=0.046528
- qwen3 / reasoning_constraint / source / mlp: adjusted=0.252215, peak_depth=0.141429, persistence=0.38125
- qwen3 / reasoning_constraint / source / residual: adjusted=0.261009, peak_depth=0.732143, persistence=0.0
- qwen3 / syntax_structure / last / attention: adjusted=0.020151, peak_depth=0.17, persistence=0.106945
- qwen3 / syntax_structure / last / mlp: adjusted=-0.003857, peak_depth=0.424286, persistence=0.49375
- qwen3 / syntax_structure / last / residual: adjusted=-0.026516, peak_depth=0.537857, persistence=0.0
- qwen3 / syntax_structure / query / attention: adjusted=0.058894, peak_depth=0.081429, persistence=0.295139
- qwen3 / syntax_structure / query / mlp: adjusted=0.034007, peak_depth=0.164286, persistence=0.406944
- qwen3 / syntax_structure / query / residual: adjusted=-0.003401, peak_depth=0.320714, persistence=0.0
- qwen3 / syntax_structure / source / attention: adjusted=0.117329, peak_depth=0.037143, persistence=0.125
- qwen3 / syntax_structure / source / mlp: adjusted=0.15724, peak_depth=0.175714, persistence=0.322917
- qwen3 / syntax_structure / source / residual: adjusted=0.001221, peak_depth=0.332857, persistence=0.0

## Caution

Adjusted reuse subtracts a same-family, same-item-index mechanism control. It is still observational and is not a causal subspace proof.
Heldout prediction uses frozen item_index=4 cases and simple cosine prototypes; lexical/template leakage remains a possible baseline.
