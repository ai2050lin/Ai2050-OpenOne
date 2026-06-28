# Atlas Graph v1 数据格式说明

`atlas_graph_v1` 是给 3D 机制图谱客户端使用的测试结果格式。它的目标不是保存所有原始实验数据，而是把测试结果压成可持续积累、可查询、可渲染的机制图谱。

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

## 证据等级

推荐使用以下等级，不要把弱证据写成强证据：

- `candidate`：相关候选。
- `likelihood_only`：只影响 teacher-forced likelihood（教师强制似然）。
- `generation_changed`：自然生成发生变化。
- `causal_closure`：同时满足似然、生成和对照闭合。
- `weak_or_null`：弱效应或无效。
- `boundary`：失败边界或机制瓶颈。

## 3D 坐标约定

默认坐标含义：

```text
x = component offset + head/channel index
y = layer index
z = model lane
```

如果测试脚本给出 `position`，客户端优先使用显式坐标。否则客户端会根据 `layer/head/channel/model` 自动定位。

## 后续测试脚本输出要求

每个新的 Phase 如果产生可视化图谱，建议同时输出：

```text
results/<phase_dir>/phaseXXX_cross_model_summary.json
results/<phase_dir>/phaseXXX_atlas_graph.json
```

其中 `phaseXXX_atlas_graph.json` 必须符合本文件的 `atlas_graph_v1` 格式。这样客户端可以直接加载，不需要为每个 Phase 单独写解析代码。

## Phase 727 样例

当前样例文件：

```text
results/glm5_phase727_category_fruit_cluster_intervention/phase727_atlas_graph.json
```

生成命令：

```bash
python tests/gpt5/build_phase727_atlas_graph.py
```
