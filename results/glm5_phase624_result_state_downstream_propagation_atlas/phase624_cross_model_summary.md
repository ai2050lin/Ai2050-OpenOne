# Phase 624 Cross-Model Summary

Result-state downstream propagation atlas.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- patch layers: [20, 21, 22]
- downstream layers: [22, 23, 24, 25, 26, 27]

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 75/82 | 2.890 | 1.537 | -1.352 |
| result_random_norm | 2/82 | -0.092 | -0.094 | -0.002 |
| selection_both | 63/82 | 1.907 | 1.192 | -0.715 |
| selection_both_plus_result | 75/82 | 2.892 | 1.540 | -1.352 |

### Top Result-Only Propagation Nodes

| layer | component | repair_proj | repair_cos | repair_norm | seed_proj | seed_cos |
|---:|---|---:|---:|---:|---:|---:|
| L22 | layer_out | 0.848 | 0.921 | 0.921 | 1.000 | 1.000 |
| L23 | layer_input | 0.848 | 0.921 | 0.921 | 1.000 | 1.000 |
| L23 | mlp_out | 0.834 | 0.918 | 0.909 | -0.009 | -0.021 |
| L23 | layer_out | 0.817 | 0.920 | 0.888 | 1.012 | 0.901 |
| L24 | layer_input | 0.817 | 0.920 | 0.888 | 1.012 | 0.901 |
| L25 | mlp_out | 0.800 | 0.913 | 0.877 | 0.027 | 0.052 |
| L24 | mlp_out | 0.800 | 0.906 | 0.883 | 0.013 | 0.029 |
| L27 | mlp_out | 0.797 | 0.938 | 0.850 | 0.016 | 0.009 |
| L24 | layer_out | 0.796 | 0.915 | 0.869 | 1.033 | 0.824 |
| L25 | layer_input | 0.796 | 0.915 | 0.869 | 1.033 | 0.824 |
| L25 | layer_out | 0.785 | 0.914 | 0.860 | 1.054 | 0.752 |
| L26 | layer_input | 0.785 | 0.914 | 0.860 | 1.054 | 0.752 |
| L27 | layer_out | 0.783 | 0.919 | 0.853 | 1.047 | 0.413 |
| L26 | layer_out | 0.773 | 0.906 | 0.853 | 1.085 | 0.635 |
| L27 | layer_input | 0.773 | 0.906 | 0.853 | 1.085 | 0.635 |
| L26 | mlp_out | 0.772 | 0.905 | 0.853 | 0.015 | 0.019 |

### Top All Propagation Nodes

| mode | layer | component | repair_proj | repair_cos | seed_proj |
|---|---:|---|---:|---:|---:|
| result_only | L22 | layer_out | 0.848 | 0.921 | 1.000 |
| result_only | L23 | layer_input | 0.848 | 0.921 | 1.000 |
| selection_both_plus_result | L22 | layer_out | 0.848 | 0.921 | 1.000 |
| selection_both_plus_result | L23 | layer_input | 0.848 | 0.921 | 1.000 |
| result_only | L23 | mlp_out | 0.834 | 0.918 | -0.009 |
| selection_both_plus_result | L23 | mlp_out | 0.834 | 0.918 | -0.009 |
| result_only | L23 | layer_out | 0.817 | 0.920 | 1.012 |
| result_only | L24 | layer_input | 0.817 | 0.920 | 1.012 |
| selection_both_plus_result | L23 | layer_out | 0.817 | 0.920 | 1.012 |
| selection_both_plus_result | L24 | layer_input | 0.817 | 0.920 | 1.012 |
| result_only | L25 | mlp_out | 0.800 | 0.913 | 0.027 |
| selection_both_plus_result | L25 | mlp_out | 0.800 | 0.913 | 0.027 |
| result_only | L24 | mlp_out | 0.800 | 0.906 | 0.013 |
| selection_both_plus_result | L24 | mlp_out | 0.800 | 0.906 | 0.013 |
| result_only | L27 | mlp_out | 0.797 | 0.938 | 0.016 |
| selection_both_plus_result | L27 | mlp_out | 0.797 | 0.938 | 0.016 |
| result_only | L24 | layer_out | 0.796 | 0.915 | 1.033 |
| result_only | L25 | layer_input | 0.796 | 0.915 | 1.033 |
| selection_both_plus_result | L24 | layer_out | 0.796 | 0.915 | 1.033 |
| selection_both_plus_result | L25 | layer_input | 0.796 | 0.915 | 1.033 |

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- patch layers: [31, 32, 34]
- downstream layers: [34, 35, 36, 37, 38, 39]

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 29/31 | 2.131 | 0.974 | -1.157 |
| result_random_norm | 3/31 | -0.069 | -0.066 | 0.003 |
| selection_both | 7/31 | 0.347 | 0.192 | -0.155 |
| selection_both_plus_result | 29/31 | 2.131 | 0.974 | -1.157 |

### Top Result-Only Propagation Nodes

| layer | component | repair_proj | repair_cos | repair_norm | seed_proj | seed_cos |
|---:|---|---:|---:|---:|---:|---:|
| L34 | layer_out | 0.939 | 0.969 | 0.969 | 1.000 | 1.000 |
| L35 | layer_input | 0.939 | 0.969 | 0.969 | 1.000 | 1.000 |
| L38 | layer_out | 0.925 | 0.967 | 0.956 | 1.310 | 0.711 |
| L39 | layer_input | 0.925 | 0.967 | 0.956 | 1.310 | 0.711 |
| L36 | layer_out | 0.922 | 0.963 | 0.957 | 1.000 | 0.905 |
| L37 | layer_input | 0.922 | 0.963 | 0.957 | 1.000 | 0.905 |
| L35 | layer_out | 0.919 | 0.961 | 0.957 | 1.001 | 0.945 |
| L36 | layer_input | 0.919 | 0.961 | 0.957 | 1.001 | 0.945 |
| L39 | layer_out | 0.917 | 0.968 | 0.947 | 1.272 | 0.546 |
| L37 | layer_out | 0.916 | 0.961 | 0.952 | 1.027 | 0.851 |
| L38 | layer_input | 0.916 | 0.961 | 0.952 | 1.027 | 0.851 |
| L39 | mlp_out | 0.915 | 0.974 | 0.938 | -0.037 | -0.029 |
| L35 | mlp_out | 0.912 | 0.950 | 0.961 | 0.021 | 0.071 |
| L38 | mlp_out | 0.911 | 0.966 | 0.942 | 0.270 | 0.260 |
| L37 | mlp_out | 0.896 | 0.954 | 0.940 | 0.024 | 0.059 |
| L36 | mlp_out | 0.887 | 0.932 | 0.951 | 0.006 | 0.021 |

### Top All Propagation Nodes

| mode | layer | component | repair_proj | repair_cos | seed_proj |
|---|---:|---|---:|---:|---:|
| result_only | L34 | layer_out | 0.939 | 0.969 | 1.000 |
| result_only | L35 | layer_input | 0.939 | 0.969 | 1.000 |
| selection_both_plus_result | L34 | layer_out | 0.939 | 0.969 | 1.000 |
| selection_both_plus_result | L35 | layer_input | 0.939 | 0.969 | 1.000 |
| result_only | L38 | layer_out | 0.925 | 0.967 | 1.310 |
| result_only | L39 | layer_input | 0.925 | 0.967 | 1.310 |
| selection_both_plus_result | L38 | layer_out | 0.925 | 0.967 | 1.310 |
| selection_both_plus_result | L39 | layer_input | 0.925 | 0.967 | 1.310 |
| result_only | L36 | layer_out | 0.922 | 0.963 | 1.000 |
| result_only | L37 | layer_input | 0.922 | 0.963 | 1.000 |
| selection_both_plus_result | L36 | layer_out | 0.922 | 0.963 | 1.000 |
| selection_both_plus_result | L37 | layer_input | 0.922 | 0.963 | 1.000 |
| result_only | L35 | layer_out | 0.919 | 0.961 | 1.001 |
| result_only | L36 | layer_input | 0.919 | 0.961 | 1.001 |
| selection_both_plus_result | L35 | layer_out | 0.919 | 0.961 | 1.001 |
| selection_both_plus_result | L36 | layer_input | 0.919 | 0.961 | 1.001 |
| result_only | L39 | layer_out | 0.917 | 0.968 | 1.272 |
| selection_both_plus_result | L39 | layer_out | 0.917 | 0.968 | 1.272 |
| result_only | L37 | layer_out | 0.916 | 0.961 | 1.027 |
| result_only | L38 | layer_input | 0.916 | 0.961 | 1.027 |

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- patch layers: [26, 27, 29]
- downstream layers: [29, 30, 31, 32, 33, 34, 35]

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 15/17 | 4.407 | 1.814 | -2.593 |
| result_random_norm | 2/17 | 0.088 | 0.060 | -0.029 |
| selection_both | 9/17 | 1.384 | 0.850 | -0.534 |
| selection_both_plus_result | 15/17 | 4.407 | 1.814 | -2.593 |

### Top Result-Only Propagation Nodes

| layer | component | repair_proj | repair_cos | repair_norm | seed_proj | seed_cos |
|---:|---|---:|---:|---:|---:|---:|
| L31 | mlp_out | 0.886 | 0.925 | 0.959 | 0.031 | 0.062 |
| L29 | layer_out | 0.865 | 0.930 | 0.930 | 1.000 | 1.000 |
| L30 | layer_input | 0.865 | 0.930 | 0.930 | 1.000 | 1.000 |
| L30 | mlp_out | 0.856 | 0.844 | 1.018 | 0.002 | 0.005 |
| L31 | layer_out | 0.837 | 0.905 | 0.924 | 1.059 | 0.801 |
| L32 | layer_input | 0.837 | 0.905 | 0.924 | 1.059 | 0.801 |
| L30 | layer_out | 0.824 | 0.906 | 0.910 | 1.043 | 0.901 |
| L31 | layer_input | 0.824 | 0.906 | 0.910 | 1.043 | 0.901 |
| L32 | mlp_out | 0.815 | 0.909 | 0.897 | 0.019 | 0.034 |
| L32 | layer_out | 0.803 | 0.900 | 0.893 | 1.079 | 0.724 |
| L33 | layer_input | 0.803 | 0.900 | 0.893 | 1.079 | 0.724 |
| L33 | layer_out | 0.800 | 0.894 | 0.895 | 1.111 | 0.654 |
| L34 | layer_input | 0.800 | 0.894 | 0.895 | 1.111 | 0.654 |
| L34 | mlp_out | 0.775 | 0.922 | 0.842 | 0.004 | 0.005 |
| L34 | layer_out | 0.771 | 0.893 | 0.863 | 1.111 | 0.571 |
| L35 | layer_input | 0.771 | 0.893 | 0.863 | 1.111 | 0.571 |

### Top All Propagation Nodes

| mode | layer | component | repair_proj | repair_cos | seed_proj |
|---|---:|---|---:|---:|---:|
| result_only | L31 | mlp_out | 0.886 | 0.925 | 0.031 |
| selection_both_plus_result | L31 | mlp_out | 0.886 | 0.925 | 0.031 |
| result_only | L29 | layer_out | 0.865 | 0.930 | 1.000 |
| result_only | L30 | layer_input | 0.865 | 0.930 | 1.000 |
| selection_both_plus_result | L29 | layer_out | 0.865 | 0.930 | 1.000 |
| selection_both_plus_result | L30 | layer_input | 0.865 | 0.930 | 1.000 |
| result_only | L30 | mlp_out | 0.856 | 0.844 | 0.002 |
| selection_both_plus_result | L30 | mlp_out | 0.856 | 0.844 | 0.002 |
| result_only | L31 | layer_out | 0.837 | 0.905 | 1.059 |
| result_only | L32 | layer_input | 0.837 | 0.905 | 1.059 |
| selection_both_plus_result | L31 | layer_out | 0.837 | 0.905 | 1.059 |
| selection_both_plus_result | L32 | layer_input | 0.837 | 0.905 | 1.059 |
| result_only | L30 | layer_out | 0.824 | 0.906 | 1.043 |
| result_only | L31 | layer_input | 0.824 | 0.906 | 1.043 |
| selection_both_plus_result | L30 | layer_out | 0.824 | 0.906 | 1.043 |
| selection_both_plus_result | L31 | layer_input | 0.824 | 0.906 | 1.043 |
| result_only | L32 | mlp_out | 0.815 | 0.909 | 0.019 |
| selection_both_plus_result | L32 | mlp_out | 0.815 | 0.909 | 0.019 |
| result_only | L32 | layer_out | 0.803 | 0.900 | 1.079 |
| result_only | L33 | layer_input | 0.803 | 0.900 | 1.079 |
