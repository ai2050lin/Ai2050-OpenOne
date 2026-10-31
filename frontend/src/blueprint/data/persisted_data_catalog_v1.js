export const PERSISTED_DATA_CATALOG_V1 = [
  {
    "id": "raw_events_v1",
    "label": "原始事件数据",
    "layer": "原始抓取层",
    "relativePath": "research/gpt5/data/raw/layer_parameter_state_events_v1.jsonl",
    "recordCount": 15,
    "description": "参数级原始事件，记录层号、神经元、维度、参数位、激活值与来源阶段。",
    "fields": [
      "event_id",
      "layer_index",
      "neuron_index",
      "dim_index",
      "parameter_ids",
      "activation_value",
      "source_stage"
    ]
  },
  {
    "id": "research_rows_v1",
    "label": "研究结构行",
    "layer": "研究结构层",
    "relativePath": "research/gpt5/data/rows/layer_parameter_state_rows_v1.jsonl",
    "recordCount": 15,
    "description": "把原始事件整理成研究结构行，保留类别、字段签名和来源链。",
    "fields": [
      "row_id",
      "category",
      "field_signature",
      "layer_key",
      "source_stage",
      "source_event_id"
    ]
  },
  {
    "id": "analysis_v1",
    "label": "基础分析结果",
    "layer": "分析结果层",
    "relativePath": "research/gpt5/data/analysis/layer_parameter_state_analysis_v1.json",
    "recordCount": 15,
    "description": "汇总参数节点数、链路数、剖面数和主要分类。",
    "fields": [
      "profile_count",
      "node_count",
      "chain_count",
      "layers",
      "categories"
    ]
  },
  {
    "id": "visualization_overlay_v1",
    "label": "可视化叠加数据",
    "layer": "可视化资产层",
    "relativePath": "research/gpt5/data/visualization/layer_parameter_state_overlay_v1.json",
    "recordCount": 15,
    "description": "前端当前使用的参数态节点、链路和层级叠加信息。",
    "fields": [
      "label",
      "color",
      "nodes",
      "chains",
      "sourceDataPath"
    ]
  },
  {
    "id": "audit_v1",
    "label": "审计溯源数据",
    "layer": "审计与溯源层",
    "relativePath": "research/gpt5/data/audit/layer_parameter_state_audit_v1.json",
    "recordCount": 0,
    "description": "记录脚本、生成时间、输出文件和源文件映射。",
    "fields": [
      "script_path",
      "generated_at",
      "outputs",
      "source_files"
    ]
  },
  {
    "id": "puzzle_warehouse_manifest_v1",
    "label": "拼图仓总览",
    "layer": "拼图仓总览层",
    "relativePath": "research/gpt5/data/puzzle_warehouse/puzzle_warehouse_manifest_v1.json",
    "recordCount": 6,
    "description": "定义拼图仓主入口、默认优先轴、默认视图和拼图类型。",
    "fields": [
      "warehouse_id",
      "records_path",
      "default_priority_axes",
      "default_views",
      "puzzle_types"
    ]
  },
  {
    "id": "puzzle_records_v1",
    "label": "拼图记录数据",
    "layer": "拼图记录层",
    "relativePath": "research/gpt5/data/puzzle_warehouse/puzzle_records_v1.jsonl",
    "recordCount": 4,
    "description": "基础拼图仓的核心记录，保存裂缝、反例、修复候选和下一步动作。",
    "fields": [
      "puzzle_id",
      "puzzle_type",
      "priority_axis",
      "layer_scope",
      "mapped_variables",
      "next_action"
    ]
  },
  {
    "id": "puzzle_record_template_v1",
    "label": "拼图记录模板",
    "layer": "拼图模板层",
    "relativePath": "research/gpt5/data/puzzle_warehouse/puzzle_record_template_v1.json",
    "recordCount": 18,
    "description": "规定每条拼图记录必须包含的字段和示例写法。",
    "fields": [
      "required_fields",
      "field_notes",
      "example_record"
    ]
  },
  {
    "id": "puzzle_focus_axes_v1",
    "label": "优先轴配置",
    "layer": "优先轴配置层",
    "relativePath": "research/gpt5/data/puzzle_warehouse/puzzle_focus_axes_v1.json",
    "recordCount": 4,
    "description": "固定当前最值得持续推进的研究主线和目标问题。",
    "fields": [
      "axes",
      "priority",
      "reason",
      "target_questions"
    ]
  },
  {
    "id": "puzzle_view_registry_v1",
    "label": "拼图视图注册表",
    "layer": "拼图可视化层",
    "relativePath": "research/gpt5/data/puzzle_warehouse/puzzle_view_registry_v1.json",
    "recordCount": 5,
    "description": "定义基础总览图、对象对比图、运行回放图、裂缝地图和证据关系图。",
    "fields": [
      "views",
      "purpose",
      "default_layers",
      "default_priority_axes"
    ]
  },
  {
    "id": "repair_replay_sample_slots_v1",
    "label": "修复回放槽位",
    "layer": "样本回放槽位层",
    "relativePath": "research/gpt5/data/puzzle_warehouse/repair_replay_sample_slots_v1.json",
    "recordCount": 3,
    "description": "为修复前后对照预留真实样本回放槽位、缺失资产和验证目标。",
    "fields": [
      "slot_id",
      "repair_puzzle_id",
      "baseline_puzzle_id",
      "sample_id",
      "phase_slots",
      "missing_assets"
    ]
  }
];
