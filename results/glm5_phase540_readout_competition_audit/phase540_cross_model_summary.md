# Phase540 Readout Competition Audit Summary

## qwen3

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [10, 12, 14], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], attn=sdpa

| source | condition | best win | alpha | token deltas | class |
|---|---|---|---:|---|---|
| vehicle_furniture | residual_perp | extended | 6.0 | margin +0.311, target +0.720, comp +0.409, supp -0.409, cluster -0.207, off +0.464, shortcut +0.256 | target_push_shortcut |
| vehicle_furniture | residual_parallel | extended | 6.0 | margin +1.115, target +1.071, comp -0.045, supp +0.045, cluster +0.049, off +0.359, shortcut +0.711 | target_push_shortcut |
| vehicle_furniture | residual_full | extended | 6.0 | margin +0.421, target +0.799, comp +0.379, supp -0.379, cluster -0.226, off +0.468, shortcut +0.331 | target_push_shortcut |
| vehicle_tool | residual_perp | extended | 6.0 | margin +1.258, target +0.730, comp -0.527, supp +0.527, cluster +0.181, off +0.648, shortcut +0.082 | global_readout_spill |
| vehicle_tool | residual_parallel | extended | 6.0 | margin +1.635, target +1.032, comp -0.603, supp +0.603, cluster +0.137, off +0.418, shortcut +0.428 | mixed |
| vehicle_tool | residual_full | extended | 6.0 | margin +1.376, target +0.803, comp -0.573, supp +0.573, cluster +0.180, off +0.663, shortcut +0.140 | global_readout_spill |
| vehicle_clothing | residual_perp | extended | 6.0 | margin +1.312, target +0.717, comp -0.595, supp +0.595, cluster +0.449, off +0.552, shortcut +0.122 | global_readout_spill |
| vehicle_clothing | residual_parallel | extended | 6.0 | margin +1.630, target +1.040, comp -0.591, supp +0.591, cluster +0.092, off +0.259, shortcut +0.449 | mixed |
| vehicle_clothing | residual_full | extended | 6.0 | margin +1.455, target +0.792, comp -0.663, supp +0.663, cluster +0.399, off +0.543, shortcut +0.130 | mixed |

## glm4

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [24, 26, 28], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], attn=sdpa

| source | condition | best win | alpha | token deltas | class |
|---|---|---|---:|---|---|
| vehicle_furniture | residual_perp | extended | 6.0 | margin +1.363, target +2.377, comp +1.015, supp -1.015, cluster -0.031, off +1.451, shortcut +0.927 | target_push_shortcut |
| vehicle_furniture | residual_parallel | extended | 6.0 | margin +5.900, target +4.551, comp -1.349, supp +1.349, cluster +1.299, off +1.326, shortcut +3.202 | mixed |
| vehicle_furniture | residual_full | extended | 6.0 | margin +2.412, target +3.122, comp +0.710, supp -0.710, cluster +0.013, off +1.331, shortcut +1.792 | target_push_shortcut |
| vehicle_tool | residual_perp | extended | 6.0 | margin +3.453, target +1.575, comp -1.878, supp +1.878, cluster +0.499, off +1.322, shortcut -0.303 | global_readout_spill |
| vehicle_tool | residual_parallel | extended | 6.0 | margin +12.866, target +5.155, comp -7.711, supp +7.711, cluster +0.656, off +1.464, shortcut -2.555 | mixed |
| vehicle_tool | residual_full | extended | 6.0 | margin +5.164, target +2.373, comp -2.791, supp +2.791, cluster +0.464, off +1.291, shortcut -0.418 | mixed |
| vehicle_clothing | residual_perp | extended | 6.0 | margin +3.957, target +1.430, comp -2.527, supp +2.527, cluster +0.501, off +1.276, shortcut -1.097 | global_readout_spill |
| vehicle_clothing | residual_parallel | extended | 6.0 | margin +10.040, target +4.443, comp -5.598, supp +5.598, cluster +0.744, off +0.887, shortcut -1.155 | mixed |
| vehicle_clothing | residual_full | extended | 6.0 | margin +5.652, target +2.183, comp -3.469, supp +3.469, cluster +0.517, off +1.218, shortcut -1.286 | mixed |

## deepseek7b

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [16, 18, 20], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], attn=sdpa

| source | condition | best win | alpha | token deltas | class |
|---|---|---|---:|---|---|
| vehicle_furniture | residual_perp | extended | 6.0 | margin +0.132, target +0.209, comp +0.077, supp -0.077, cluster -0.053, off +0.083, shortcut +0.126 | mixed |
| vehicle_furniture | residual_parallel | extended | 6.0 | margin +0.687, target +0.688, comp +0.002, supp -0.002, cluster +0.197, off +0.236, shortcut +0.452 | target_push_shortcut |
| vehicle_furniture | residual_full | extended | 6.0 | margin +0.121, target +0.187, comp +0.066, supp -0.066, cluster -0.080, off +0.065, shortcut +0.122 | mixed |
| vehicle_tool | residual_perp | extended | 6.0 | margin +0.337, target +0.159, comp -0.178, supp +0.178, cluster +0.037, off +0.038, shortcut -0.019 | mixed |
| vehicle_tool | residual_parallel | extended | 6.0 | margin +1.589, target +1.113, comp -0.476, supp +0.476, cluster +0.291, off +0.343, shortcut +0.637 | mixed |
| vehicle_tool | residual_full | extended | 6.0 | margin +0.375, target +0.192, comp -0.183, supp +0.183, cluster +0.037, off +0.039, shortcut +0.009 | mixed |
| vehicle_clothing | residual_perp | extended | 6.0 | margin +0.344, target +0.259, comp -0.085, supp +0.085, cluster +0.078, off +0.086, shortcut +0.173 | mixed |
| vehicle_clothing | residual_parallel | extended | 6.0 | margin +0.839, target +0.638, comp -0.201, supp +0.201, cluster +0.079, off +0.103, shortcut +0.438 | mixed |
| vehicle_clothing | residual_full | extended | 6.0 | margin +0.386, target +0.275, comp -0.110, supp +0.110, cluster +0.077, off +0.083, shortcut +0.165 | mixed |

## Residual Parallel Compact

| model | source | win | margin | target | competitor | suppression | cluster_other | off_cluster | shortcut | class |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | vehicle_furniture | extended | +1.115 | +1.071 | -0.045 | +0.045 | +0.049 | +0.359 | +0.711 | target_push_shortcut |
| qwen3 | vehicle_tool | extended | +1.635 | +1.032 | -0.603 | +0.603 | +0.137 | +0.418 | +0.428 | mixed |
| qwen3 | vehicle_clothing | extended | +1.630 | +1.040 | -0.591 | +0.591 | +0.092 | +0.259 | +0.449 | mixed |
| glm4 | vehicle_furniture | extended | +5.900 | +4.551 | -1.349 | +1.349 | +1.299 | +1.326 | +3.202 | mixed |
| glm4 | vehicle_tool | extended | +12.866 | +5.155 | -7.711 | +7.711 | +0.656 | +1.464 | -2.555 | mixed |
| glm4 | vehicle_clothing | extended | +10.040 | +4.443 | -5.598 | +5.598 | +0.744 | +0.887 | -1.155 | mixed |
| deepseek7b | vehicle_furniture | extended | +0.687 | +0.688 | +0.002 | -0.002 | +0.197 | +0.236 | +0.452 | target_push_shortcut |
| deepseek7b | vehicle_tool | extended | +1.589 | +1.113 | -0.476 | +0.476 | +0.291 | +0.343 | +0.637 | mixed |
| deepseek7b | vehicle_clothing | extended | +0.839 | +0.638 | -0.201 | +0.201 | +0.079 | +0.103 | +0.438 | mixed |

