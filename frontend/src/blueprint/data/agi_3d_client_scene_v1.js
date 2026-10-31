const agi3dClientScene = {
  version: 'agi_3d_client_scene_v1',
  axes: {
    x: '对象或任务分区',
    y: '层级深度',
    z: '运行机制角色',
  },
  layers: {
    shared_carrier_3d: {
      label: '共享承载层',
      score: 0.4379344914760285,
      sourceStage: 'Stage341 / Stage375',
      sourceScript:
        'tests/codex/stage341_shared_carrier_3d_layer.py + tests/codex/stage375_shared_carrier_subchain_extractor.py',
      sourceOutput:
        'tests/codex_temp/stage341_shared_carrier_3d_layer_20260324 + tests/codex_temp/stage375_shared_carrier_subchain_extractor_20260325',
      runtimeProfiles: [
        {
          id: 'shared-qwen',
          label: 'Qwen 运行链',
          color: '#3b82f6',
          nodes: [
            { id: 'sq-1', label: '共享入口 L5', layerIndex: 5, position: [-14, 1.6, 10] },
            { id: 'sq-2', label: '共享桥接 L14', layerIndex: 14, position: [-8, 4.6, 10] },
          ],
        },
        {
          id: 'shared-deepseek7',
          label: 'DeepSeek 7B 运行链',
          color: '#8b5cf6',
          nodes: [
            { id: 'sd7-1', label: '共享入口 L5', layerIndex: 5, position: [0, 1.6, 10] },
            { id: 'sd7-2', label: '共享桥接 L23', layerIndex: 23, position: [4, 7.6, 10] },
          ],
        },
        {
          id: 'shared-deepseek14',
          label: 'DeepSeek 14B 运行链',
          color: '#22c55e',
          nodes: [
            { id: 'sd14-1', label: '共享入口 L5', layerIndex: 5, position: [10, 1.6, 10] },
            { id: 'sd14-2', label: '任务桥接 L27', layerIndex: 27, position: [14, 7.6, 10] },
          ],
        },
      ],
      clusters: [
        {
          id: 'fruit_shared_cluster',
          label: '水果共享簇',
          color: '#38bdf8',
          stability: 0.3090850969959451,
          memberCount: 3,
          position: [-12, 1.8, -4],
          size: 0.62,
          members: [138, 306, 660],
        },
        {
          id: 'cross_domain_cluster',
          label: '跨类共享簇',
          color: '#4ecdc4',
          stability: 0.28511778177478575,
          memberCount: 5,
          position: [-4, 2.6, 0],
          size: 0.7,
          members: [469, 364, 215, 530, 273],
        },
        {
          id: 'object_cluster',
          label: '器物共享簇',
          color: '#f59e0b',
          stability: 0.2772529521699285,
          memberCount: 1,
          position: [4, 4.6, 3],
          size: 0.52,
          members: [521],
        },
        {
          id: 'mixed_cluster',
          label: '混合共享簇',
          color: '#fb7185',
          stability: 0.26749978645910133,
          memberCount: 1,
          position: [12, 7.0, 6],
          size: 0.48,
          members: [477],
        },
      ],
      taskBridges: [
        {
          id: 'translation_bridge',
          label: '翻译共享桥',
          color: '#22c55e',
          strength: 0.3555210327936543,
          from: [-4, 2.6, 0],
          to: [-10, 6.2, 8],
        },
        {
          id: 'refactor_bridge',
          label: '重构共享桥',
          color: '#a855f7',
          strength: 0.542985042815821,
          from: [-4, 2.6, 0],
          to: [10, 6.8, 8],
        },
      ],
      rawPoints: [
        { id: 'sc-bundle-position', label: 'position_rows', value: 31, position: [-13.8, 1.1, -6.2], color: '#7dd3fc', kind: '原始行束' },
        { id: 'sc-bundle-coverage', label: 'coverage_rows', value: 12, position: [-12.8, 1.4, -7.0], color: '#7dd3fc', kind: '原始行束' },
        { id: 'sc-bundle-distribution', label: 'distribution_rows', value: 10, position: [-11.7, 1.7, -7.3], color: '#7dd3fc', kind: '原始行束' },
        { id: 'sc-label-cross', label: '跨类共享触发', value: 20, position: [-4.8, 2.2, -2.0], color: '#4ecdc4', kind: '原始标签' },
        { id: 'sc-label-fruit', label: '水果内部差分', value: 13, position: [-3.1, 2.5, -1.4], color: '#4ecdc4', kind: '原始标签' },
        { id: 'sc-field-family', label: 'family_hit_count', value: 53, position: [-5.8, 3.3, 1.5], color: '#22c55e', kind: '原始字段' },
        { id: 'sc-field-dim', label: 'dim_index', value: 48, position: [-4.2, 3.6, 1.9], color: '#22c55e', kind: '原始字段' },
        { id: 'sc-field-base', label: 'base_load', value: 44, position: [-2.8, 3.9, 1.6], color: '#22c55e', kind: '原始字段' },
        { id: 'sc-subchain-1', label: '共享子链-位置', value: 31, position: [-9.5, 6.0, 5.8], color: '#38bdf8', kind: '子链候选' },
        { id: 'sc-subchain-2', label: '共享子链-覆盖', value: 12, position: [-8.2, 6.3, 6.3], color: '#38bdf8', kind: '子链候选' },
      ],
      rawLinks: [
        { id: 'sc-link-1', from: [-13.8, 1.1, -6.2], to: [-12, 1.8, -4], color: '#7dd3fc' },
        { id: 'sc-link-2', from: [-4.8, 2.2, -2.0], to: [-4, 2.6, 0], color: '#4ecdc4' },
        { id: 'sc-link-3', from: [-5.8, 3.3, 1.5], to: [-4, 2.6, 0], color: '#22c55e' },
      ],
    },
    bias_deflection_3d: {
      label: '偏置偏转层',
      score: 0.6394210230966288,
      sourceStage: 'Stage342 / Stage376 / Stage378',
      sourceScript:
        'tests/codex/stage342_bias_deflection_3d_layer.py + tests/codex/stage376_bias_deflection_subchain_extractor.py + tests/codex/stage378_task_bias_dedicated_extractor.py',
      sourceOutput:
        'tests/codex_temp/stage342_bias_deflection_3d_layer_20260324 + tests/codex_temp/stage376_bias_deflection_subchain_extractor_20260325 + tests/codex_temp/stage378_task_bias_dedicated_extractor_20260325',
      runtimeProfiles: [
        {
          id: 'bias-qwen',
          label: 'Qwen 约束偏转链',
          color: '#3b82f6',
          nodes: [
            { id: 'bq-1', label: '约束入口 L5', layerIndex: 5, position: [-14, 1.6, 12] },
            { id: 'bq-2', label: '约束偏转 L23', layerIndex: 23, position: [-9, 7.6, 12] },
          ],
        },
        {
          id: 'bias-deepseek7',
          label: 'DeepSeek 7B 操作偏转链',
          color: '#8b5cf6',
          nodes: [
            { id: 'bd7-1', label: '操作入口 L5', layerIndex: 5, position: [0, 1.6, 12] },
            { id: 'bd7-2', label: '操作偏转 L23', layerIndex: 23, position: [4, 7.6, 12] },
          ],
        },
        {
          id: 'bias-deepseek14',
          label: 'DeepSeek 14B 任务偏转链',
          color: '#22c55e',
          nodes: [
            { id: 'bd14-1', label: '任务入口 L5', layerIndex: 5, position: [10, 1.6, 12] },
            { id: 'bd14-2', label: '任务偏转 L27', layerIndex: 27, position: [14, 7.6, 12] },
          ],
        },
      ],
      directions: [
        {
          id: 'object_deflection',
          label: '对象偏转',
          color: '#f97316',
          selectivity: 0.6952735190665997,
          memberCount: 4,
          from: [-10, 1.0, -6],
          to: [-4, 4.6, -1],
        },
        {
          id: 'intra_class_deflection',
          label: '类内竞争偏转',
          color: '#ef4444',
          selectivity: 0.7279994743855448,
          memberCount: 3,
          from: [0, 2.6, -6],
          to: [0, 5.4, 0],
        },
        {
          id: 'task_deflection',
          label: '任务偏转',
          color: '#22c55e',
          selectivity: 0.6848760575761188,
          memberCount: 2,
          from: [10, 5.0, -6],
          to: [5, 7.4, 2],
        },
      ],
      modelBias: [
        { id: 'qwen_bias', label: 'Qwen 约束偏转', color: '#3b82f6', thickness: 0.7100637927651405, position: [-9, 7.8, 8] },
        { id: 'deepseek_bias', label: 'DeepSeek 操作偏转', color: '#8b5cf6', thickness: 0.7779081965175767, position: [9, 8.2, 8] },
      ],
      rawPoints: [
        { id: 'bd-label-fruit', label: '水果内部差分', value: 17, position: [-6.3, 3.1, -2.2], color: '#f97316', kind: '原始标签' },
        { id: 'bd-label-animal', label: '动物内部差分', value: 13, position: [-5.2, 3.4, -1.2], color: '#f97316', kind: '原始标签' },
        { id: 'bd-label-tool', label: '工具与器物差分', value: 12, position: [-4.0, 3.7, -0.4], color: '#f97316', kind: '原始标签' },
        { id: 'bd-label-brand', label: '品牌与跨类触发', value: 8, position: [0.8, 4.5, -0.8], color: '#ef4444', kind: '原始标签' },
        { id: 'bd-field-dim', label: 'dim_index', value: 58, position: [1.8, 5.9, 1.8], color: '#f97316', kind: '原始字段' },
        { id: 'bd-field-delta', label: 'mean_delta_load', value: 54, position: [0.4, 6.2, 1.3], color: '#f97316', kind: '原始字段' },
        { id: 'bd-field-base', label: 'base_load', value: 50, position: [-1.0, 6.5, 1.1], color: '#f97316', kind: '原始字段' },
        { id: 'bd-task-translation', label: 'translation', value: 2, position: [6.2, 6.2, 1.2], color: '#22c55e', kind: '任务标签' },
        { id: 'bd-task-refactor', label: '翻译_vs_重构', value: 2, position: [7.2, 6.5, 1.8], color: '#22c55e', kind: '任务标签' },
        { id: 'bd-task-operation', label: 'operation', value: 2, position: [8.2, 6.8, 2.3], color: '#22c55e', kind: '任务标签' },
      ],
      rawLinks: [
        { id: 'bd-link-1', from: [-6.3, 3.1, -2.2], to: [-4, 4.6, -1], color: '#f97316' },
        { id: 'bd-link-2', from: [0.8, 4.5, -0.8], to: [0, 5.4, 0], color: '#ef4444' },
        { id: 'bd-link-3', from: [6.2, 6.2, 1.2], to: [5, 7.4, 2], color: '#22c55e' },
      ],
    },
    layerwise_amplification_3d: {
      label: '逐层放大层',
      score: 0.39212748480455417,
      sourceStage: 'Stage343 / Stage377',
      sourceScript:
        'tests/codex/stage343_layerwise_amplification_3d_layer.py + tests/codex/stage377_amplification_subchain_extractor.py',
      sourceOutput:
        'tests/codex_temp/stage343_layerwise_amplification_3d_layer_20260324 + tests/codex_temp/stage377_amplification_subchain_extractor_20260325',
      runtimeProfiles: [
        {
          id: 'amp-qwen',
          label: 'Qwen 放大链',
          color: '#3b82f6',
          nodes: [
            { id: 'aq-1', label: '第一次放大 L5', layerIndex: 5, position: [-14, 1.6, 14] },
            { id: 'aq-2', label: '主放大 L23', layerIndex: 23, position: [-10, 4.6, 14] },
            { id: 'aq-3', label: '持续放大 L27', layerIndex: 27, position: [-6, 7.6, 14] },
          ],
        },
        {
          id: 'amp-deepseek7',
          label: 'DeepSeek 7B 放大链',
          color: '#8b5cf6',
          nodes: [
            { id: 'ad7-1', label: '第一次放大 L5', layerIndex: 5, position: [-1, 1.6, 14] },
            { id: 'ad7-2', label: '主放大 L23', layerIndex: 23, position: [3, 4.6, 14] },
            { id: 'ad7-3', label: '持续放大 L27', layerIndex: 27, position: [7, 7.6, 14] },
          ],
        },
        {
          id: 'amp-deepseek14',
          label: 'DeepSeek 14B 放大链',
          color: '#22c55e',
          nodes: [
            { id: 'ad14-1', label: '第一次放大 L5', layerIndex: 5, position: [10, 1.6, 14] },
            { id: 'ad14-2', label: '主放大 L23', layerIndex: 23, position: [13, 4.6, 14] },
            { id: 'ad14-3', label: '持续放大 L27', layerIndex: 27, position: [16, 7.6, 14] },
          ],
        },
      ],
      bands: [
        { id: 'early_band', label: '早层第一次放大', color: '#38bdf8', strength: 0.19215232735155346, gain: 0.13066358259905636, position: [-10, 1.6, -10], length: 7 },
        { id: 'mid_band', label: '中层主放大', color: '#f59e0b', strength: 0.22459534682221427, gain: 0.1527248358391057, position: [0, 4.6, 0], length: 9 },
        { id: 'late_band', label: '后层持续放大', color: '#ec4899', strength: 0.32667894294807864, gain: 0.2221416812046935, position: [10, 7.6, 10], length: 11 },
      ],
      rawPoints: [
        { id: 'amp-label-coupling', label: 'residual_coupling', value: 6, position: [-10.6, 2.2, -11.4], color: '#38bdf8', kind: '原始标签' },
        { id: 'amp-label-relay', label: 'relay_strength', value: 6, position: [-8.9, 2.5, -10.8], color: '#38bdf8', kind: '原始标签' },
        { id: 'amp-label-net', label: '残余耦合抑制后净增益', value: 4, position: [0.4, 5.0, -0.8], color: '#f59e0b', kind: '原始标签' },
        { id: 'amp-label-ratio', label: '独立放大核 / 接力整体比值', value: 4, position: [2.2, 5.3, 0.8], color: '#f59e0b', kind: '原始标签' },
        { id: 'amp-field-strength', label: 'strength', value: 66, position: [10.3, 8.0, 8.8], color: '#ec4899', kind: '原始字段' },
        { id: 'amp-field-ind', label: 'independent_gain', value: 9, position: [11.8, 8.3, 9.4], color: '#ec4899', kind: '原始字段' },
        { id: 'amp-anchor-early', label: '第一次放大主核候选', value: 2, position: [-6.6, 2.8, -8.4], color: '#7dd3fc', kind: '锚点' },
        { id: 'amp-anchor-mid', label: '中层主放大主核候选', value: 2, position: [4.2, 5.8, 2.4], color: '#fbbf24', kind: '锚点' },
        { id: 'amp-anchor-late', label: '后层持续放大主核候选', value: 2, position: [16.2, 8.8, 12.2], color: '#f472b6', kind: '锚点' },
      ],
      rawLinks: [
        { id: 'amp-link-early', from: [-6.6, 2.8, -8.4], to: [-3, 1.6, -10], color: '#7dd3fc' },
        { id: 'amp-link-mid', from: [4.2, 5.8, 2.4], to: [4.5, 4.6, 0], color: '#fbbf24' },
        { id: 'amp-link-late', from: [16.2, 8.8, 12.2], to: [15.5, 7.6, 10], color: '#f472b6' },
      ],
    },
    multispace_operator_3d: {
      label: '多空间角色层',
      score: 0.4057423151296106,
      sourceStage: 'Stage344 / Stage257 / Stage258 / Stage260',
      sourceScript:
        'tests/codex/stage344_multispace_operator_3d_layer.py + tests/codex/stage257_object_attribute_position_operation_role_map.py + tests/codex/stage258_task_semantic_to_processing_route_bridge.py + tests/codex/stage260_full_semantic_role_total_map.py',
      sourceOutput: 'tests/codex_temp/stage344_multispace_operator_3d_layer_20260324 等',
      runtimeProfiles: [
        {
          id: 'ms-qwen',
          label: 'Qwen 多空间运行链',
          color: '#3b82f6',
          nodes: [
            { id: 'msq-1', label: '对象角色 L5', layerIndex: 5, position: [-14, 1.6, 16] },
            { id: 'msq-2', label: '任务角色 L23', layerIndex: 23, position: [-10, 4.6, 16] },
            { id: 'msq-3', label: '传播角色 L27', layerIndex: 27, position: [-6, 7.6, 16] },
          ],
        },
        {
          id: 'ms-deepseek14',
          label: 'DeepSeek 14B 多空间运行链',
          color: '#22c55e',
          nodes: [
            { id: 'msd-1', label: '对象角色 L5', layerIndex: 5, position: [10, 1.6, 16] },
            { id: 'msd-2', label: '任务角色 L23', layerIndex: 23, position: [13, 4.6, 16] },
            { id: 'msd-3', label: '传播角色 L27', layerIndex: 27, position: [16, 7.6, 16] },
          ],
        },
      ],
      roleNodes: [
        { id: 'object_space_node', label: '对象空间', color: '#38bdf8', alignment: 0.13492383658885956, position: [-10, 2.4, -8] },
        { id: 'task_space_node', label: '任务空间', color: '#22c55e', alignment: 0.4992530378047377, position: [0, 5.4, 0] },
        { id: 'relay_space_node', label: '传播空间', color: '#f97316', alignment: 0.3577044799144627, position: [10, 7.6, 8] },
      ],
      operatorParts: [
        { id: 'local_operator', label: '局部运算元', color: '#8b5cf6', value: 0.5437379815663305, position: [-12, 8.4, 8] },
        { id: 'multispace_mapping', label: '多空间映射', color: '#06b6d4', value: 0.17415108316661917, position: [-4, 8.7, 10] },
        { id: 'fuzzy_sparse_joint', label: '模糊承载与稀疏偏转', color: '#10b981', value: 0.6559466945381602, position: [4, 9.0, 10] },
        { id: 'relay_part', label: '层间接力', color: '#f59e0b', value: 0.35859767831932726, position: [12, 9.3, 8] },
      ],
      rawPoints: [
        { id: 'ms-role-object', label: 'object', value: 0.03492383658885956, position: [-12.8, 1.9, -9.4], color: '#38bdf8', kind: '原始角色' },
        { id: 'ms-role-attribute', label: 'attribute', value: 0.032115254551172256, position: [-11.1, 2.2, -9.9], color: '#38bdf8', kind: '原始角色' },
        { id: 'ms-role-position', label: 'position', value: 0.039707958698272705, position: [-9.1, 2.5, -9.5], color: '#38bdf8', kind: '原始角色' },
        { id: 'ms-role-operation', label: 'operation', value: 0.04177581146359444, position: [-7.3, 2.8, -8.8], color: '#38bdf8', kind: '原始角色' },
        { id: 'ms-task-translation', label: 'translation', value: 93.19064331054688, position: [-1.8, 5.7, -1.4], color: '#22c55e', kind: '任务行' },
        { id: 'ms-task-image', label: 'image_edit', value: 131.33920288085938, position: [0.2, 6.1, -0.8], color: '#22c55e', kind: '任务行' },
        { id: 'ms-task-refactor', label: 'refactor', value: 213.40591430664062, position: [2.8, 6.5, 0.4], color: '#22c55e', kind: '任务行' },
        { id: 'ms-total-operation', label: 'operation 总角色', value: 0.547635509322087, position: [11.4, 8.1, 6.4], color: '#f59e0b', kind: '总角色' },
        { id: 'ms-total-constraint', label: 'constraint 总角色', value: 0.40361887613932296, position: [9.8, 8.4, 6.9], color: '#f59e0b', kind: '总角色' },
      ],
      rawLinks: [
        { id: 'ms-link-object', from: [-12.8, 1.9, -9.4], to: [-10, 2.4, -8], color: '#38bdf8' },
        { id: 'ms-link-task', from: [2.8, 6.5, 0.4], to: [0, 5.4, 0], color: '#22c55e' },
        { id: 'ms-link-constraint', from: [9.8, 8.4, 6.9], to: [10, 7.6, 8], color: '#f59e0b' },
      ],
    },
    cross_model_compare_3d: {
      label: '跨模型对照层',
      score: 0.5076367744402507,
      sourceStage: 'Stage345 / Stage379',
      sourceScript:
        'tests/codex/stage345_five_layer_3d_client_manifest.py + tests/codex/stage379_cross_model_common_segment_extractor.py',
      sourceOutput:
        'tests/codex_temp/stage345_five_layer_3d_client_manifest_20260324 + tests/codex_temp/stage379_cross_model_common_segment_extractor_20260325',
      runtimeProfiles: [
        {
          id: 'cm-qwen',
          label: 'Qwen 真实层级链',
          color: '#3b82f6',
          nodes: [
            { id: 'cmq-1', label: '入口 L5', layerIndex: 5, position: [-14, 1.6, 18] },
            { id: 'cmq-2', label: '重写 L23', layerIndex: 23, position: [-10, 4.6, 18] },
            { id: 'cmq-3', label: '读出 L27', layerIndex: 27, position: [-6, 7.6, 18] },
          ],
        },
        {
          id: 'cm-deepseek14',
          label: 'DeepSeek 14B 真实层级链',
          color: '#22c55e',
          nodes: [
            { id: 'cmd-1', label: '入口 L5', layerIndex: 5, position: [10, 1.6, 18] },
            { id: 'cmd-2', label: '重写 L23', layerIndex: 23, position: [13, 4.6, 18] },
            { id: 'cmd-3', label: '读出 L27', layerIndex: 27, position: [16, 7.6, 18] },
          ],
        },
      ],
      models: [
        {
          id: 'qwen',
          label: 'Qwen',
          color: '#3b82f6',
          position: [-8, 0, 0],
          metrics: [
            { key: '共享承载', value: 0.4728 },
            { key: '偏置偏转', value: 0.5219 },
            { key: '逐层放大', value: 0.3198 },
            { key: '共同主核', value: 0.5076 },
          ],
        },
        {
          id: 'deepseek',
          label: 'DeepSeek',
          color: '#8b5cf6',
          position: [8, 0, 0],
          metrics: [
            { key: '共享承载', value: 0.4728 },
            { key: '偏置偏转', value: 0.5488 },
            { key: '逐层放大', value: 0.3921 },
            { key: '共同主核', value: 0.4081 },
          ],
        },
      ],
      rawPoints: [
        { id: 'cm-common-core', label: 'common_core_strength', value: 3, position: [0, 7.8, -1.6], color: '#eab308', kind: '共同原段' },
        { id: 'cm-family', label: 'family', value: 2.6015625, position: [-3.2, 6.2, 1.4], color: '#60a5fa', kind: '共同原段' },
        { id: 'cm-tr-ref', label: '翻译_vs_重构', value: 1.3093815346558888, position: [0.2, 6.6, 1.7], color: '#34d399', kind: '共同原段' },
        { id: 'cm-img-tr', label: '图像编辑_vs_翻译', value: 1.243113140265147, position: [3.6, 6.1, 1.5], color: '#34d399', kind: '共同原段' },
        { id: 'cm-hook', label: 'parameter_hook_score', value: 1.1023156996816397, position: [6.1, 5.6, 1.1], color: '#a78bfa', kind: '共同原段' },
        { id: 'cm-repair', label: 'repair_fidelity_score', value: 1.0, position: [-5.8, 5.1, -0.6], color: '#f472b6', kind: '共同原段' },
        { id: 'cm-natural', label: 'natural_fidelity_score', value: 0.6666666666666666, position: [5.5, 4.8, -0.8], color: '#f472b6', kind: '共同原段' },
      ],
      rawLinks: [
        { id: 'cm-link-left', from: [0, 7.8, -1.6], to: [-8, 4.0, 0], color: '#94a3b8' },
        { id: 'cm-link-right', from: [0, 7.8, -1.6], to: [8, 4.0, 0], color: '#94a3b8' },
      ],
    },
  },
};

export default agi3dClientScene;
