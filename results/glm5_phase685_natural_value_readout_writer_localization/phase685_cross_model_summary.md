# Phase 685 Natural Value-Readout Writer Localization

- generated: `2026-06-26 12:32:19`

| model | paired_cases | short_rank | terse_rank | rank_delta | top_site | top_delta | top_pos_rate |
|---|---:|---:|---:|---:|---|---:|---:|
| deepseek7b | 72 | 167.69 | 1.00 | 166.69 | L27_layer_out | 34.718 | 0.958 |
| glm4 | 5 | 2.00 | 1.00 | 1.00 | L38_layer_out | 3.443 | 1.000 |
| qwen3 | 3 | 2.00 | 1.00 | 1.00 | L34_layer_out | 8.932 | 1.000 |

## Top Positive Sites

### deepseek7b

| site | component | mean_delta | positive_rate | short_proj | terse_proj |
|---|---|---:|---:|---:|---:|
| L27 | layer_out | 34.718 | 0.958 | 25.644 | 60.362 |
| L26 | layer_out | 24.362 | 0.944 | 37.825 | 62.187 |
| L26 | attn_out | 12.838 | 1.000 | 10.975 | 23.813 |
| L24 | layer_out | 7.913 | 0.944 | 15.703 | 23.616 |
| L25 | layer_out | 7.285 | 0.903 | 24.173 | 31.458 |
| L23 | attn_out | 6.249 | 1.000 | 3.888 | 10.136 |
| L23 | layer_out | 5.511 | 0.931 | 15.867 | 21.378 |
| L27 | mlp_out | 5.461 | 0.583 | -37.868 | -32.407 |
| L27 | attn_out | 4.899 | 0.819 | 25.680 | 30.579 |
| L26 | mlp_out | 4.240 | 0.764 | 2.668 | 6.908 |
| L24 | mlp_out | 1.890 | 0.944 | 0.119 | 2.010 |
| L22 | attn_out | 1.849 | 0.972 | 0.179 | 2.028 |

### glm4

| site | component | mean_delta | positive_rate | short_proj | terse_proj |
|---|---|---:|---:|---:|---:|
| L38 | layer_out | 3.443 | 1.000 | -2.781 | 0.663 |
| L39 | layer_out | 3.222 | 1.000 | -3.271 | -0.049 |
| L34 | layer_out | 2.229 | 1.000 | 2.640 | 4.870 |
| L35 | layer_out | 2.166 | 1.000 | 1.685 | 3.851 |
| L36 | layer_out | 2.058 | 1.000 | 1.270 | 3.327 |
| L33 | layer_out | 2.004 | 1.000 | 2.386 | 4.391 |
| L38 | mlp_out | 1.769 | 0.800 | -1.512 | 0.257 |
| L32 | layer_out | 1.753 | 1.000 | 2.212 | 3.965 |
| L37 | layer_out | 1.580 | 1.000 | -0.208 | 1.372 |
| L31 | layer_out | 1.058 | 1.000 | 0.839 | 1.897 |
| L30 | layer_out | 0.755 | 1.000 | 0.100 | 0.854 |
| L29 | layer_out | 0.574 | 1.000 | 0.242 | 0.815 |

### qwen3

| site | component | mean_delta | positive_rate | short_proj | terse_proj |
|---|---|---:|---:|---:|---:|
| L34 | layer_out | 8.932 | 1.000 | 27.262 | 36.193 |
| L33 | layer_out | 7.808 | 1.000 | 14.167 | 21.975 |
| L35 | layer_out | 7.373 | 1.000 | 7.531 | 14.904 |
| L32 | layer_out | 7.256 | 1.000 | 20.795 | 28.051 |
| L31 | layer_out | 5.718 | 1.000 | 9.120 | 14.838 |
| L30 | layer_out | 3.069 | 1.000 | 6.893 | 9.962 |
| L29 | layer_out | 2.734 | 1.000 | 2.600 | 5.334 |
| L31 | attn_out | 1.907 | 1.000 | 1.341 | 3.248 |
| L34 | attn_out | 1.303 | 1.000 | 17.584 | 18.887 |
| L32 | attn_out | 1.279 | 1.000 | 10.814 | 12.093 |
| L29 | attn_out | 1.049 | 1.000 | 2.726 | 3.775 |
| L28 | layer_out | 1.047 | 1.000 | 0.205 | 1.253 |

