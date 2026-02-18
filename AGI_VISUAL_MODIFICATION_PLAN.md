# AGI 可视化修改方案（结构分析 / 路线验证 / 进度总结）

## 1. 方案目标

本方案围绕三个核心目标：

1. 分析深度神经网络结构  
通过统一分析入口与 3D 视图，持续观察激活、拓扑、流管、回路与稳定性信号。

2. 测试不同 AGI 路线可行性并记录结果  
对同一任务并行运行多条路线，统一产出指标、结论、证据，并保存为固定 JSON。

3. 总结 AGI 研究进度  
按阶段、路线、实验结果自动汇总进度，形成时间线与阶段评估。

---

## 2. 总体架构

采用“算法路线解耦 + 可视化统一协议”架构：

1. 运行层（Runtime）
- 统一 `run` 协议（`route + analysis_type + params + events + summary`）
- 每次运行自动归档时间线 JSON

2. 服务层（API）
- `/api/v1/runs`：创建/查询运行
- `/api/v1/runs/{id}/events`：读取事件流
- `/api/v1/experiments/timeline`：按路线返回测试时间线

3. 可视化层（Frontend）
- 结构分析工作台（DNN）
- 路线测试时间线（Route Timeline）
- 研究进度看板（Progress Dashboard）

---

## 3. 固定 JSON 记录规范

文件路径（固定）：

- `tempdata/agi_route_test_timeline.json`

顶层结构：

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-02-18T10:55:33Z",
  "routes": {
    "fiber_bundle": {
      "route": "fiber_bundle",
      "latest_test_id": "run_xxx",
      "stats": {
        "total_runs": 12,
        "completed_runs": 10,
        "failed_runs": 2,
        "avg_score": 0.71,
        "latest_timestamp": "2026-02-18T10:55:33Z"
      },
      "tests": [
        {
          "test_id": "run_xxx",
          "run_id": "run_xxx",
          "timestamp": "2026-02-18T10:55:33Z",
          "status": "completed",
          "analysis_type": "tda_snapshot",
          "params": {},
          "metrics": [],
          "conclusion": {},
          "artifacts": [],
          "evaluation": {
            "score": 0.74,
            "grade": "B",
            "feasibility": "high",
            "summary": "..."
          }
        }
      ]
    }
  }
}
```

要求：

1. 每次测试必须可追溯（`run_id`、`timestamp`、`analysis_type`）
2. 必须有可比较评估字段（`score/grade/feasibility`）
3. 必须包含证据链（`metrics + conclusion + artifacts`）

---

## 4. 可视化改造方案

### 4.1 结构分析工作区（目标1）

改造内容：

1. 所有分析入口统一走 runtime-first（失败再 fallback）
2. 每个分析结果统一显示结论卡：`目标/思路/3D原理/算法说明/指标范围/结论`
3. 3D 区支持“结果来源标记”（runtime-v1 / legacy）

重点视图：

1. Topology（拓扑层结构）
2. Flow Tubes（语义演化轨迹）
3. TDA（持久同调）
4. Debias / Global Topology（路线验证相关分析）

### 4.2 路线可行性时间线（目标2）

改造内容：

1. 新增路线时间线面板（按 route 分组）
2. 每条 route 展示：
- 总运行数、成功数、失败数、平均评分
- 每次测试状态、时间、分析类型、结论摘要、评估等级
3. 支持按 route 筛选与按时间倒序查看

核心页面：

1. `AGIProgressDashboard` 增加“路线测试时间线”
2. 单次记录卡片显示“可行性评估”

### 4.3 研究进度看板（目标3）

改造内容：

1. 阶段路线图（Phase）
2. 最新实验指标（accuracy / stability / score）
3. 研究日志摘要（里程碑证据）
4. 自动读取路线时间线统计结果，展示阶段推进状态

---

## 5. 实施清单（文件级）

后端：

1. `server/runtime/experiment_store.py`
- 固定 JSON 持久化
- 评估打分与路线聚合

2. `server/runtime/run_service.py`
- run 完成/失败自动写入时间线
- 提供 `get_experiment_timeline`

3. `server/api_v1_runs.py`
- 新增 `GET /api/v1/experiments/timeline`

前端：

1. `frontend/src/config/api.js`
- 增加 `runtime.experimentTimeline(...)`

2. `frontend/src/components/evaluation/RouteTimelineBoard.jsx`
- 路线时间线可视化组件

3. `frontend/src/AGIProgressDashboard.jsx`
- 打开界面自动加载 `/agi/progress` 与 `/experiments/timeline`
- 展示路线时间线 + 阶段 + 指标 + 日志

---

## 6. 分阶段推进

### Phase A（已完成）

1. runtime 协议接入主要结构分析入口
2. 固定 JSON 自动归档
3. 路线时间线接口
4. 时间线前端展示

### Phase B（下一步）

1. 路线筛选器（route / analysis_type / 时间范围）
2. 单路线导出报告 JSON
3. A/B 对比面板（同任务多路线并排）

### Phase C（增强）

1. 自动周报（阶段变化 + 新增证据）
2. 失败模式聚类（失败原因分布图）
3. 路线可行性趋势预测（score trend）

---

## 7. 验收标准（Definition of Done）

1. 每次测试自动写入固定 JSON，且字段完整  
2. 打开进度界面可看到“每条路线时间线 + 每次结果 + 评估”  
3. 同一路线历史可比较（有统一 score/grade）  
4. 结构分析、路线验证、进度总结三类页面可联通  
5. 构建通过，接口可稳定返回

---

## 8. 输出成果

完成后，项目具备以下能力：

1. 结构分析能力：可持续观察 DNN 内部结构与动力学变化  
2. 路线验证能力：可对比不同 AGI 路线并沉淀证据  
3. 进度治理能力：可按时间线和阶段自动总结研究进展

