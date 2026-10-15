# CLIENT_SERVER_ALIGNMENT_v1 — G 线研究流程 ↔ 可视化客户端/服务器对齐整理

版本：v1（2026-10-14，Phase 3237 收官后的工程整理臂，非测量账本轮）
范围：`frontend/`（React+Vite+three.js 图谱透镜）、`server/`（FastAPI :5001）、`shared/dist_protocol/`（v3 bundle 协议）、`deploy/`（中心-节点分布式）、G 线研究产物链。
原则：**派生视图零改动封存产物**（registry/gap/ledger immutable / append-only 不受影响）；服务器端只做发现口径修复与只读派生；客户端数据面统一走派生档。

---

## 1. 组件地图与数据流（现状 + 本轮修复后）

```
G 线测量臂（tests/glm5/phaseN_*.py，五段式）
  └─ 产物: registry v1_N / gap v1_N / npz（append-only 版本链，存各 phase 产物目录）
       ├─ research/gpt5/atlas/atlas_led.json ← closeout governance entry（n=389 post d37ca65f）
       │
       ├─[派生] server/derive_atlas_tree.py（glob 最新 registry + ledger）
       │    └─ atlas_tree_v1/v2.json → GET /api/ai-rnd/atlas/tree → frontend rdc_fusion 透镜
       │
       ├─[API] server/ai_rnd_service.py
       │    ├─ GET /atlas/registry（修复 A1：自动发现全版本链，默认最新 v1.32）
       │    ├─ GET /atlas/feature/{fid}（特征卡 + anchors sha 钻取）
       │    ├─ GET /atlas/ledger（账本分页）
       │    ├─ GET /atlas/gaps（修复 A2：gap_ledger 自动发现，GAP-5 final_verdict 可见）
       │    └─ GET /queue/gline（五段式队列，closed=ledger G 线尾部）
       │
       └─[分布式] shared/dist_protocol v3（grammar 模式族 + bundle 三件套）
            ├─ deploy/distributed_service.py（中心：注册 S2/调度 S3/回传 S4/聚合）
            ├─ deploy/node_agent.py（终端：register→claim→执行→分块上传→complete）
            └─ shared/dist_protocol/examples/TM-ISA-DEMO-v1/（修复 A3：3236 演示臂 bundle 化样例）
```

## 2. 版本链对齐表（整理时点实测）

| 消费点 | 整理前 | 整理后 | 动作 |
|---|---|---|---|
| `derive_atlas_tree.py latest_registry()` | glob 自动（活） | registry **v32** / ledger **n=389**（tree sha `0a62ca19`/`0d15645a`） | 已重跑派生刷新数据面 |
| `ai_rnd_service /atlas/registry` | 硬编码候选只到 **v1.8**（客户端默认拿到 7 个版本前旧图谱） | **A1**：glob 自动发现 32 版全链，默认 v1.32，versions 列表完整 | `server/ai_rnd_service.py` |
| gap_ledger 接入 | server **0 命中**（GAP 完全不可见） | **A2**：`_discover_gap_versions()` + `GET /atlas/gaps`（含 GAP-5 `final_verdict` UPSTREAM_ABSENT） | 同上 |
| 分布式任务 | TM-01..06 legacy + grammar 模式族（KN-ISA 等四族），与 G 线五段式**无映射文档** | **A3**：TM-ISA-DEMO-v1 bundle 样例（3236 五步流程逐字导出）+ §3 映射表 | `shared/dist_protocol/examples/` |

## 3. G 线五段式 ↔ dist_protocol v3 bundle 映射（分布式研发的协议桥）

| G 线五段式（单机臂） | dist_protocol v3（分布式） | 说明 |
|---|---|---|
| DESIGN 冻结（freeze sha8，观测前） | `contract.json.design_sha`（corpus+runner 规范化哈希，预注册） | 同为"观测前冻结设计"；design_sha 不符中心拒收 |
| SMOKE（P{N}_SMOKE=1，必看数字） | runner `--smoke` 模式 + 中心 smoke 标记（不进合并） | canary.v1 补充：模型/精度漂移检测 |
| FULL（正式测量） | node_agent claim → 真实模式执行 → 分块上传 | 充分统计量 suffstats.v1 保证跨节点可合并且逐位等价 |
| 独立复核（verify，FAIL 只修探针） | 中心 sha256 内容寻址 + audits 表 + canary 一致性 | 复核语义一致：封存产物不可改 |
| closeout 五写（ledger/MEMO/daily/MEMORY/self-check） | 聚合 AGG → registry/gap 落账（仍在单机上执行，回传物只入 results） | **governance 写权留在中心研究臂**，终端只产出测量 |

任务清单侧：语言图谱任务清单 = registry 的 FTR 体系 + `atlas_tree_v2` 的模板×机制关联；下发任务 = 3236 式模版矩阵 bundle 化（TM-ISA-DEMO-v1 为参考实现：F-DEMO-ISA-001 × 5 模版 × 160 行，G1–G4 门逐字预注册进 contract）。

## 4. 已修复（本轮，全部只读派生/新增文件，封存产物零改动）

- **A1** `server/ai_rnd_service.py`：`_discover_atlas_versions()`（glob.glob 递归，与 derive_atlas_tree 逐字同口径，容忍 phase2746/field_store 坏链接）+ `_atlas_registry_path` 默认最新 + `_atlas_version_list` 完整 32 版；legacy 硬编码候选降级为 fallback。
- **A2** 同文件：`_discover_gap_versions()` + 新端点 `GET /api/ai-rnd/atlas/gaps`（gap v1.11 全量 5 GAP + final_verdict）。
- **A3** `shared/dist_protocol/examples/TM-ISA-DEMO-v1/{corpus,runner.py,contract}.json`：corpus 160 行从 3236 DESIGN 逐字导出；design_sha= `2ecf0e9e…`（`_canonical_design_sha` 同口径复算）；runner smoke 自跑通过（无模型时 synthetic 自检，结果永不进合并）；contract 预注册 G1–G4 + 来源锚（freeze b1fa889f / res 8b79ed5e / registry v1.32 / gap v1.11 / ledger #389）。
- **A4** 数据面刷新：`atlas_tree_v1.json`（sha `0a62ca19`）与 `atlas_tree_v2.json`（sha `0d15645a`）现派生自 registry v32 + ledger n=389；旧档备份 `*.bak_pre3238align`。

## 5. 遗留差距（后续分布式研发路线，按优先级）

1. **前端 Gap 面板**：`/atlas/gaps` 端点已就绪，frontend 尚无消费组件（UI 线 v1.10 冻结；建议 UI v1.7 工单：Gap 矩阵 + GAP-5 final_verdict 展示）。
2. **atlas_graph_v1 契约缺口**：G 线 3231–3237 臂未产出 `phaseXXX_atlas_graph.json`（frontend/ATLAS_GRAPH_FORMAT.md 契约）；建议下个测量臂（3238）起按契约同步产出 + 镜像 `frontend/public/vis_data/atlas/`。
3. **真分布式采集**：TM-ISA-DEMO runner 的 real 模式已含 margin 采集核心；F4/F5（封存方向投影、逐层曲线）需把 3236 的 hidden-state 捕获拆成分片协议（suffstats 扩展或 per-shard npz）。
4. **AI2050-RDC-Archive 坏链接**：`phase2746/field_store` 悬空（pathlib.glob 会炸，glob.glob 容忍）；建议运维清理或重挂。
5. **多终端 G2 拼图校准**：拼图纪律（先族内收敛再跨族校准）在分布式场景 = 跨终端结果合并前的端口校准（3172 的 29% 教训），需在聚合层加校准门。

## 6. 验收（本轮自检）

`tests/gpt5_temp/phase3238align_check.py`：A1/A2 发现链、tree sha、bundle design_sha 复算、smoke summary 契约字段、封存产物只读断言。全 PASS 方视为整理完成。
