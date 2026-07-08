# Atlas Graph v1 数据格式说明

`atlas_graph_v1` 是给 3D 机制图谱客户端使用的测试结果格式。它的目标不是保存所有原始实验数据，而是把测试结果压成可持续积累、可查询、可渲染的机制图谱。

## 测试脚本输出契约

新的 Phase 测试脚本不能只输出原始 `json/jsonl/log`，必须在脚本结束时直接生成客户端可加载的图谱结果文件：

```text
tests/result/<phase_slug>/<run_name>/phaseXXX_cross_model_summary.json
tests/result/<phase_slug>/<run_name>/phaseXXX_atlas_graph.json
```

如果该 Phase 需要进入 3D 客户端，还必须同步生成或复制一份到：

```text
frontend/public/vis_data/atlas/<phase_slug>_<run_name>_phaseXXX_atlas_graph.json
```

其中：

- `phaseXXX_atlas_graph.json` 必须符合本文档的 `atlas_graph_v1`。
- `phaseXXX_cross_model_summary.json` 保存实验汇总，供右侧详情面板和审计面板读取。
- 原始 `jsonl` 可以保留，但客户端不应依赖原始 `jsonl` 才能显示主图谱。
- 测试脚本必须在同一次运行中完成测试结果聚合和图谱生成，不再要求后续手工运行单独转换脚本。

推荐脚本流程：

```text
1. run experiment
2. collect raw rows
3. build summary
4. build atlas_graph_v1
5. write phaseXXX_cross_model_summary.json
6. write phaseXXX_atlas_graph.json
7. optional: mirror atlas graph to frontend/public/vis_data/atlas
```

最小 Python 输出骨架：

```python
from pathlib import Path
import json

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def build_atlas_graph(phase, title, nodes, edges, source_files):
    return {
        "schema_version": "atlas_graph_v1",
        "title": title,
        "model_info": {
            "model": "cross_model",
            "models": ["qwen3", "glm4", "deepseek7b"],
            "phase": phase,
            "evidence_type": "mechanism atlas"
        },
        "layout": {
            "x": "mechanism stage / component offset",
            "y": "layer / generation step",
            "z": "model lane"
        },
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "metrics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_phase": phase
        },
        "source_files": source_files
    }
```

写文件示例：

```python
out_dir = Path("tests/result/phase215_prompt_attention_route_atlas/main")
atlas = build_atlas_graph(
    phase=215,
    title="Phase 215 Prompt Attention Route Atlas",
    nodes=nodes,
    edges=edges,
    source_files=[
        "phase215_cross_model_summary.json",
        "phase215_*_route_delta_rows.jsonl"
    ],
)

write_json(out_dir / "phase215_cross_model_summary.json", summary)
write_json(out_dir / "phase215_atlas_graph.json", atlas)
write_json(
    "frontend/public/vis_data/atlas/gpt5_phase215_prompt_attention_route_atlas_main_phase215_atlas_graph.json",
    atlas,
)
```

## 顶层结构

```json
{
  "schema_version": "atlas_graph_v1",
  "title": "Phase 727 Category/Fruit Route Cluster Intervention Atlas",
  "model_info": {
    "model": "cross_model",
    "models": ["qwen3", "glm4", "deepseek7b"],
    "phase": 727,
    "timestamp": "2026-06-28 17:18:09",
    "evidence_type": "cluster-level likelihood and greedy generation intervention"
  },
  "layout": {
    "x": "component offset + head/channel index",
    "y": "layer index",
    "z": "model lane"
  },
  "graph": {
    "nodes": [],
    "edges": []
  },
  "metrics": {
    "node_count": 0,
    "edge_count": 0,
    "source_phase": 727
  },
  "source_files": []
}
```

## 节点格式

每个节点代表一个可研究对象：模型、阶段、任务、层、head、channel、cluster、concept、failure 等。

```json
{
  "id": "deepseek7b:channel:L20H17C25",
  "type": "channel",
  "label": "L20H17C25",
  "model": "deepseek7b",
  "layer": 20,
  "head": 17,
  "channel": 25,
  "role": "category_channel_candidate",
  "evidence_level": "likelihood_only",
  "mean_logprob_delta": -0.2613,
  "changed_rate_vs_baseline": 0.0,
  "hit_drop_rate_vs_baseline": 0.0
}
```

### 必填字段

- `id`：全局唯一 ID，推荐格式为 `model:type:detail`。
- `type`：节点类型。
- `model`：模型名。跨模型总节点可用 `cross_model`。

### 推荐字段

- `label`：显示名。
- `layer`：层号。
- `head`：注意力头编号。
- `channel`：通道编号。
- `channels`：通道簇，例如 `[24, 25, 30, 23]`。
- `role`：机制角色。
- `evidence_level`：证据等级。
- `score`：节点主分数。
- `mean_logprob_delta`：目标答案似然变化。
- `changed_rate_vs_baseline`：自然生成变化率。
- `hit_drop_rate_vs_baseline`：命中下降率。
- `position`：可选显式坐标 `[x, y, z]`。如果没有，客户端会自动按 layer/head/channel/model 推断。
- `phase`：来源 Phase。
- `stage`：阶段名称，例如 `answer_boundary`、`stop_execution`、`pattern_competition`、`trigger_path`、`route_path`。
- `pattern_id`：语言模式，例如 `answer_short`、`answer_explain`、`answer_list`、`answer_repeat`、`answer_target_seeded`。
- `step`：生成步或轨迹步。
- `anchor`：轨迹锚点，例如 `prompt_last`、`gen_after_step_1`、`gen_after_step_6`。
- `formula`：该节点对应的核心公式，供详情面板显示。
- `summary`：一句话结论。
- `next_action`：下一步验证任务。

## 边格式

边表示机制关系、因果证据或失败边界。

```json
{
  "source": "deepseek7b:cluster:category:L20H17",
  "target": "deepseek7b:intervention:category_cluster",
  "relation": "supports_likelihood",
  "weight": 0.2613,
  "phase": 727,
  "evidence": "intervention"
}
```

### 必填字段

- `source`：源节点 ID。
- `target`：目标节点 ID。
- `relation`：关系类型。

### 推荐字段

- `weight`：关系强度。客户端会用它控制边粗细。
- `phase`：来源阶段。
- `evidence`：证据来源，例如 `intervention`、`ablation`、`generation`。
- `formula`：该关系对应的计算公式。
- `result`：关系的实验结果，例如 `repair_gain=2`、`delta=0.6162`。
- `interpretation`：关系解释，例如 `route_candidate_not_causal_proof`。

## 推荐节点类型

- `model`：模型。
- `phase`：实验阶段。
- `task`：任务。
- `concept`：概念。
- `layer`：层。
- `head`：注意力头。
- `channel`：单通道。
- `cluster`：通道簇或 head 簇。
- `intervention`：干预结果。
- `failure`：失败边界或硬伤。
- `pattern`：输出模式，例如短答、解释、列表、复读。
- `trigger_token`：Prompt 触发词或回答槽。
- `route_head`：注意力路由头候选。
- `switchpoint`：模式成功/漂移切换点候选。
- `boundary`：答案边界、停止边界、协议边界。
- `formula`：公式或理论节点。

## 推荐关系类型

- `contains`：包含关系。
- `tested_by`：被某个实验测试。
- `supports_likelihood`：影响目标答案似然。
- `changes_generation`：改变自然生成。
- `weak_generation_effect`：似然有变化但生成闭合不足。
- `negative_effect`：负向作用。
- `shared_by`：被多个概念复用。
- `differs_from`：差异机制。
- `upstream_of`：上游传播。
- `washed_by`：被后续层冲洗。
- `candidate_of`：候选机制。
- `supports_boundary`：支持答案或类别边界。
- `tests_stop_execution`：测试停止执行。
- `pattern_drift`：目标模式漂移到其他模式。
- `route_candidate`：注意力路由候选。
- `trigger_to_state`：触发词状态连接到生成状态。
- `switchpoint_candidate`：切换点候选。
- `patch_repair`：patch 后修复漂移。
- `patch_damage`：patch 后破坏成功轨迹。
- `disproves_single_point`：否定单点因果解释。

## 证据等级

推荐使用以下等级，不要把弱证据写成强证据：

- `candidate`：相关候选。
- `likelihood_only`：只影响 teacher-forced likelihood（教师强制似然）。
- `generation_changed`：自然生成发生变化。
- `causal_closure`：同时满足似然、生成和对照闭合。
- `weak_or_null`：弱效应或无效。
- `boundary`：失败边界或机制瓶颈。
- `correlation_only`：只有相关性图谱，尚无干预。
- `weak_causal`：有弱方向性因果信号，但样本量或质量不足。
- `negative_result`：明确负结果。
- `implementation_risk`：实现或解析风险，例如 EOS/pad 混淆。

## 3D 坐标约定

默认坐标含义：

```text
x = component offset + head/channel index
y = layer index
z = model lane
```

如果测试脚本给出 `position`，客户端优先使用显式坐标。否则客户端会根据 `layer/head/channel/model` 自动定位。

对 Phase195-215 这类模式路径图谱，推荐显式坐标：

```text
x = 机制路径阶段
    0 answer boundary
    1 channel/component
    2 rollout/stop
    3 EOS/decode
    4 pattern competition
    5 switchpoint
    6 trigger path
    7 route path

y = layer 或 generation step
z = model lane
```

这样客户端可以把测试结果显示成：

```text
Prompt Trigger -> Attention Route -> State Trajectory -> Pattern Competition -> Readout Boundary -> Stop/Closure
```

## 后续测试脚本输出要求

每个新的 Phase 如果产生可视化图谱，必须同时输出：

```text
tests/result/<phase_slug>/<run_name>/phaseXXX_cross_model_summary.json
tests/result/<phase_slug>/<run_name>/phaseXXX_atlas_graph.json
```

其中 `phaseXXX_atlas_graph.json` 必须符合本文件的 `atlas_graph_v1` 格式。这样客户端可以直接加载，不需要为每个 Phase 单独写解析代码。

## 校验要求

测试脚本写出 `phaseXXX_atlas_graph.json` 前必须做最小校验：

```text
1. schema_version == "atlas_graph_v1"
2. graph.nodes 是数组
3. graph.edges 是数组
4. metrics.node_count == len(graph.nodes)
5. metrics.edge_count == len(graph.edges)
6. node.id 全局唯一
7. edge.source 和 edge.target 都能在 node.id 中找到
8. 每个非 model / phase 节点必须有 evidence_level
9. 弱证据不能写成 causal_closure
```

推荐校验函数：

```python
def validate_atlas_graph(atlas):
    assert atlas.get("schema_version") == "atlas_graph_v1"
    nodes = atlas.get("graph", {}).get("nodes", [])
    edges = atlas.get("graph", {}).get("edges", [])
    ids = [node["id"] for node in nodes]
    assert len(ids) == len(set(ids)), "duplicate node id"
    id_set = set(ids)
    for edge in edges:
        assert edge["source"] in id_set, f"missing source: {edge['source']}"
        assert edge["target"] in id_set, f"missing target: {edge['target']}"
    atlas.setdefault("metrics", {})
    atlas["metrics"]["node_count"] = len(nodes)
    atlas["metrics"]["edge_count"] = len(edges)
    return atlas
```

## Phase 727 样例

当前样例文件：

```text
results/glm5_phase727_category_fruit_cluster_intervention/phase727_atlas_graph.json
```

生成命令：

```bash
python tests/gpt5/build_phase727_atlas_graph.py
```
