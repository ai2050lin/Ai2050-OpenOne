export const PERSISTED_PUZZLE_RECORDS_V1 = [
  {
    "id": "puzzle_foundation_static_anchor_band_v1",
    "title": "基础静态编码层中的稳定锚点带",
    "priority": "P1",
    "status": "active",
    "puzzleType": "foundation_structure",
    "puzzleTypeLabel": "基础结构拼图",
    "priorityAxis": "parameter_coupling",
    "priorityAxisLabel": "参数耦合裂缝",
    "objectFamily": "fruit_shared_family",
    "layerKey": "static_encoding",
    "layerLabel": "静态编码层",
    "layerRange": [
      4,
      11
    ],
    "mappedVariables": [
      "a",
      "r",
      "b"
    ],
    "observation": "静态编码层出现可重复的锚点带，但部分维度在对象切换时发生弱漂移。",
    "nextAction": "补充跨对象族稳定性对比。",
    "reproducibility": "high",
    "confidence": 0.71,
    "sourceStage": "stage440"
  },
  {
    "id": "puzzle_brain_grounding_route_anchor_gap_v1",
    "title": "动态路由层到局部锚点的脑编码落地缺口",
    "priority": "P0",
    "status": "active",
    "puzzleType": "brain_grounding",
    "puzzleTypeLabel": "脑编码落地拼图",
    "priorityAxis": "brain_grounding",
    "priorityAxisLabel": "脑编码落地",
    "objectFamily": "language_route_object_family",
    "layerKey": "dynamic_route",
    "layerLabel": "动态路径层",
    "layerRange": [
      10,
      16
    ],
    "mappedVariables": [
      "g",
      "q",
      "a",
      "r"
    ],
    "observation": "链路强度保持存在，但局部锚点解释在中段层发生跳变，说明落地映射仍不稳。",
    "nextAction": "建立跨层锚点漂移对照样本。",
    "reproducibility": "partial",
    "confidence": 0.61,
    "sourceStage": "stage440"
  },
  {
    "id": "puzzle_evidence_isolation_backfeed_pressure_v1",
    "title": "证据隔离薄弱导致的回灌压力拼图",
    "priority": "P0",
    "status": "active",
    "puzzleType": "evidence_isolation",
    "puzzleTypeLabel": "证据隔离拼图",
    "priorityAxis": "evidence_isolation",
    "priorityAxisLabel": "证据隔离",
    "objectFamily": "cross_plane_evidence_family",
    "layerKey": "advanced_analysis",
    "layerLabel": "高级分析层",
    "layerRange": [
      18,
      26
    ],
    "mappedVariables": [
      "f",
      "h",
      "m",
      "c"
    ],
    "observation": "多个观测面的解释结论看起来一致，但独立证据链拆开后仍存在回灌依赖。",
    "nextAction": "为最弱条款增加单独来源追踪。",
    "reproducibility": "medium",
    "confidence": 0.58,
    "sourceStage": "stage440"
  },
  {
    "id": "puzzle_novelty_generalization_repair_bridge_v1",
    "title": "新颖泛化裂缝到修复候选的桥接拼图",
    "priority": "P0",
    "status": "active",
    "puzzleType": "repair_candidate",
    "puzzleTypeLabel": "修复候选拼图",
    "priorityAxis": "novelty_generalization",
    "priorityAxisLabel": "新颖泛化修复",
    "objectFamily": "novelty_failure_family",
    "layerKey": "result_recovery",
    "layerLabel": "结果回收层",
    "layerRange": [
      20,
      27
    ],
    "mappedVariables": [
      "p",
      "h",
      "m",
      "c"
    ],
    "observation": "当前修复候选能改善最坏裂缝，但在更高抽象压力下仍可能掉线。",
    "nextAction": "补充修复副作用记录并纳入裂缝地图。",
    "reproducibility": "partial",
    "confidence": 0.64,
    "sourceStage": "stage440"
  }
];

export const PERSISTED_PUZZLE_SUMMARY_V1 = {
  "totalCount": 4,
  "priorityAxisCounts": [
    {
      "id": "brain_grounding",
      "label": "脑编码落地",
      "count": 1,
      "priority": "P0"
    },
    {
      "id": "evidence_isolation",
      "label": "证据隔离",
      "count": 1,
      "priority": "P0"
    },
    {
      "id": "novelty_generalization",
      "label": "新颖泛化修复",
      "count": 1,
      "priority": "P0"
    },
    {
      "id": "parameter_coupling",
      "label": "参数耦合裂缝",
      "count": 1,
      "priority": "P1"
    }
  ],
  "topPriorityIds": [
    "puzzle_brain_grounding_route_anchor_gap_v1",
    "puzzle_evidence_isolation_backfeed_pressure_v1",
    "puzzle_novelty_generalization_repair_bridge_v1"
  ]
};
