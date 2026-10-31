export const PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1 = [
  {
    "slot_id": "repair_slot_novelty_vs_evidence_v1",
    "label": "新颖泛化修复 vs 证据隔离回灌",
    "repair_puzzle_id": "puzzle_novelty_generalization_repair_bridge_v1",
    "baseline_puzzle_id": "puzzle_evidence_isolation_backfeed_pressure_v1",
    "sample_id": "counterexample_blueprint_novelty_backfeed_v1",
    "sample_label": "高抽象新颖样本下的回灌裂缝",
    "status": "partial",
    "replay_readiness": 0.46,
    "anchor_variable": "m",
    "shared_variable_candidates": [
      "h",
      "m",
      "c"
    ],
    "shared_subcircuit_hint": "route -> unifiedDecode",
    "validation_goal": "确认修复候选是否真的压低回灌依赖，而不是只转移到更晚层。",
    "required_assets": [
      "before_trace",
      "bridge_trace",
      "after_trace",
      "counterexample_prompt"
    ],
    "missing_assets": [
      "before_trace",
      "after_trace"
    ],
    "phase_slots": [
      {
        "phase": "before",
        "label": "修复前裂缝轨迹",
        "status": "planned",
        "expected_change": "h/m/c 共享变量在高级分析层出现回灌聚集。"
      },
      {
        "phase": "bridge",
        "label": "共享候选链",
        "status": "partial",
        "expected_change": "route -> unifiedDecode 候选链应承担主要修复差异。"
      },
      {
        "phase": "after",
        "label": "修复后轨迹",
        "status": "planned",
        "expected_change": "回灌桥链减少，shared anchor 稳定度上升。"
      }
    ],
    "source_stage": "stage452"
  },
  {
    "slot_id": "repair_slot_novelty_vs_brain_grounding_v1",
    "label": "新颖泛化修复 vs 脑编码落地缺口",
    "repair_puzzle_id": "puzzle_novelty_generalization_repair_bridge_v1",
    "baseline_puzzle_id": "puzzle_brain_grounding_route_anchor_gap_v1",
    "sample_id": "counterexample_blueprint_anchor_drop_v1",
    "sample_label": "局部锚点跳变导致的修复失真样本",
    "status": "planned",
    "replay_readiness": 0.28,
    "anchor_variable": "q",
    "shared_variable_candidates": [
      "a"
    ],
    "shared_subcircuit_hint": "query -> route",
    "validation_goal": "确认修复后是否改善局部锚点跳变，而不是只在结果层补偿。",
    "required_assets": [
      "before_trace",
      "bridge_trace",
      "after_trace",
      "anchor_alignment_labels"
    ],
    "missing_assets": [
      "before_trace",
      "bridge_trace",
      "after_trace",
      "anchor_alignment_labels"
    ],
    "phase_slots": [
      {
        "phase": "before",
        "label": "锚点掉线前轨迹",
        "status": "planned",
        "expected_change": "a/q 变量在中段层出现跳变。"
      },
      {
        "phase": "bridge",
        "label": "共享候选链",
        "status": "planned",
        "expected_change": "query -> route 候选链承担主要对齐压力。"
      },
      {
        "phase": "after",
        "label": "修复后锚点轨迹",
        "status": "planned",
        "expected_change": "局部锚点漂移减弱，但需检查是否引入新桥链。"
      }
    ],
    "source_stage": "stage452"
  },
  {
    "slot_id": "repair_slot_novelty_vs_parameter_coupling_v1",
    "label": "新颖泛化修复 vs 参数耦合裂缝",
    "repair_puzzle_id": "puzzle_novelty_generalization_repair_bridge_v1",
    "baseline_puzzle_id": "puzzle_foundation_static_anchor_band_v1",
    "sample_id": "counterexample_blueprint_static_drift_v1",
    "sample_label": "对象切换下的静态漂移样本",
    "status": "partial",
    "replay_readiness": 0.39,
    "anchor_variable": "c",
    "shared_variable_candidates": [],
    "shared_subcircuit_hint": "fruitGeneral -> query",
    "validation_goal": "确认修复增益是否以牺牲静态共享基底为代价。",
    "required_assets": [
      "before_trace",
      "after_trace",
      "family_switch_labels"
    ],
    "missing_assets": [
      "after_trace"
    ],
    "phase_slots": [
      {
        "phase": "before",
        "label": "静态漂移轨迹",
        "status": "partial",
        "expected_change": "fruitGeneral/query 链在对象切换时轻微漂移。"
      },
      {
        "phase": "bridge",
        "label": "共享候选链",
        "status": "planned",
        "expected_change": "需要确认是否真的存在共享最小链。"
      },
      {
        "phase": "after",
        "label": "修复后静态轨迹",
        "status": "planned",
        "expected_change": "结果层修复不应拉坏静态共享基底。"
      }
    ],
    "source_stage": "stage452"
  }
];

export const PERSISTED_REPAIR_REPLAY_SLOT_META_V1 = {
  "slotSetId": "repair_replay_sample_slots_v1",
  "label": "修复前后回放样本槽位",
  "description": "为修复前后对照和共享子回路候选预留真实样本回放槽位，后续可直接挂接样本路径、前后轨迹和验证结果。",
  "statusLegend": [
    {
      "id": "planned",
      "label": "已规划",
      "meaning": "槽位结构已定义，但真实样本还未挂接。"
    },
    {
      "id": "partial",
      "label": "部分接入",
      "meaning": "已经有样本标签或局部证据，但还缺少完整前后轨迹。"
    },
    {
      "id": "ready",
      "label": "可回放",
      "meaning": "修复前、共享候选链、修复后三段都具备可展示数据。"
    }
  ]
};
