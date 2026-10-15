# OpenOne / AI2050

OpenOne 是一个面向语言机制、神经网络原理与通用智能的逆向工程研究项目，核心问题：

> 深度神经网络如何把知识网络、逻辑推理和语法规则融合为稳定的语言能力？它的底层编码、动态运行和形成机制是什么？

项目由三部分组成：**研究方法与纪律**（五段式冻结、证据门链、失败账本）、**分布式测试平台**（生成式模板 → 分片采集 → 合并分析 → 语言谱图）、**可视化客户端**（图谱透镜与实验响应）。研究进展以研究备忘录为准，本 README 不维护进度（见[当前状态](#当前状态)）。

---

## 快速启动

**最简方式**：双击仓库根目录的 `start_client.bat`（关闭窗口即停止客户端）。

```powershell
# 前端（默认 http://localhost:5173；5001 是后端 API，勿混淆）
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_visualization.ps1
.\scripts\start_visualization.ps1 -Install      # 首次安装依赖
.\scripts\start_visualization.ps1 -Port 5174    # 换端口

# 后端
.\.venv\Scripts\python.exe .\server\install_deps.py    # 首次
.\.venv\Scripts\python.exe -m server.server            # 默认 http://localhost:5001
$env:AI2050_SKIP_MODEL_LOAD = '1'; .\.venv\Scripts\python.exe -m server.server   # 仅验证 API，不加载模型
```

分布式中心与 GPU 节点的启动方式见 `deploy/README_CENTOS.md` 与 `design/platform_spec_v1.md`。

## 研究主路线

```text
外部语言模式族图谱（知识 / 推理 / 语法的类型化超图）
→ Embedding / HiddenState 完整场（全坐标保存，Top-K 只用于显示）
→ 语言操作 ↔ 内部条件算子的关联机制（行为资格 → 仪器 → 重复 → 选择性 → 充分性 → 必要性 → 负控制 → 未见预测 → 自然生成 → 跨模型）
→ 网络可塑性与形成规律 → 可预测、可干预的智能理论 → AGI 工程
```

三条工程主线：

1. **外部图谱**：语言模式（知识、任务、语法、序列、输出）组织成可执行对象，自动生成 discovery / confirmation / lockbox 数据；
2. **内部图谱**：对每个行为合格的语言操作，保存全部 token、全部层、全部坐标的 HiddenState 场及层更新、条件差分；
3. **关联机制**：寻找可预测、可干预的"外部语言操作 → 内部算子"对应，依次通过证据门链，不做"苹果神经元"式枚举。

理论框架、路线分析、大脑视角类比与常见误区的完整论述见[旧版 README 归档](research/README_v1_full_archive.md)与[智能理论总整理](research/IntelligentTheory.md)。

## 平台组成

| 目录 | 内容 |
|---|---|
| `frontend/` | 可视化客户端（React + Vite + three.js；总览 / 空间 / 研发 / 图谱 / 路线五个一级界面） |
| `server/` | FastAPI 后端：模型接口、研究运行、中心路由自动挂载（:5001） |
| `shared/dist_protocol/` | 分布式测试协议 v3：生成式模板（`grammars/`）、充分统计量、金丝雀、合并、证据门链、runner |
| `shared/dist_protocol/analysis.py` | 规律分析（analysis.v1）：η²、置换检验、句式不变性、可加性、3D 坐标 |
| `shared/dist_protocol/atlas.py` | 语言谱图（atlas.v1）：跨族稳定特征、成边、共享编码轴、概念假设 |
| `deploy/` | 中心节点服务与节点 Agent（本地保留，不进公开仓库） |
| `design/` | 平台设计方案（`client_v6_pipeline_ia.md`、`platform_spec_v1.md`） |
| `tests/dist_protocol/` | 协议与中心测试（unittest；改动协议后必须全绿） |
| `research/` | 研究记录：GLM5 / GPT5 备忘录、智能理论、主分析、旧版 README 归档 |
| `tests/glm5*/`、`tests/gpt5*/` | 研究测试脚本与结果（正式 / 临时 / result） |
| `models/` | 本地模型入口 |

## 客户端

| 界面 | 回答的问题 |
|---|---|
| 总览 | 闭环推进到哪、题库状态、待我决策什么 |
| 空间 | 模型内部对条件怎样响应（3D 响应空间、规律分析、语言谱图、研究包下载） |
| 研发 | 下一批测什么（自动测试、题库、模板工作室、试跑冻结、分发审计） |
| 图谱 | 语言—响应关联与证据来源 |
| 路线 | 每个假设卡在哪一道证据门 |

URL 即状态（`?lens=home|spatial|process|atlas|progress`），`⌘K` 跳转。账号体系（accounts.v1）：第一个账号为管理员，写操作需管理员登录或管理令牌。界面细节见 [design/client_v6_pipeline_ia.md](design/client_v6_pipeline_ia.md)。

## 分布式测试闭环

```text
生成式模板（句式框架 × 填充词表 × 因子水平）
→ 按填充确定性切分 discovery / confirm / lockbox
→ 试跑行为门 → 冻结（design_sha 固定语料、runner 与预注册阈值）
→ 分片派发 GPU 节点 → 回传充分统计量 + 行为 + 金丝雀
→ 中心按「模型快照 × 精度」合并 → η² / 规律分析 / 留出预测
→ 证据门链判定（等级在 Research OS 人工登记）
```

关键设计：可合并统计量（n、Σx、Σx²）、金丝雀漂移检测、lockbox 防泄漏、逐样本向量（items.npz）、植入找回与负对照测试。一键跑完一个模式族：`python scripts/run_language_family.py --family KN-ISA --center http://127.0.0.1:5001 ...`，或用客户端「研发 · 自动测试」。**完整规范见 [design/platform_spec_v1.md](design/platform_spec_v1.md) 与 AGENTS.md 第 6 节**（改动协议后运行 `python -m unittest discover -s tests/dist_protocol`）。

## 当前状态

- **进度不写进 README**——权威记录是研究备忘录，按时间追加：
  - [GLM5 研究备忘录](research/glm5/docs/AGI_GLM5_MEMO.md)（行为基线与完整场线）
  - [GPT5 研究备忘录](research/gpt5/docs/AGI_GPT5_MEMO.md)（G 线现役：registry / gap 账本 / 客户端对齐）
- 快照锚（2026-10-15）：测量账本 n=389、registry v1.32、gap 账本 v1.11（GAP-5 定谳 UPSTREAM_ABSENT）；分布式协议 v3 与客户端 v6 已实现并通过离线测试，内置模板族尚未在真实模型上全量运行。
- 平台与研究的对接清单见 [research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md](research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md)。
- 研究方法的完整纪律（五段式、E 级证据、失败账本、门不后移）见 AGENTS.md。

## 纪律与工件

核心纪律（完整版见 AGENTS.md）：先行为后内部，先仪器后机制；discovery / confirmation / lockbox 严格分离；对象、阈值、分区预先冻结，看到结果后不得修改；正结果、反证、失败全部记账；AI 意见不属于实验复现。机制边闭合门：`E = I ∧ S ∧ N ∧ C ∧ P ∧ R`（仪器身份 / 充分性 / 必要性 / 特异性 / 预注册预测 / 独立重复）。

标准研究工件：`manifest.json`、`samples.jsonl`、`metrics.json`、`events.jsonl`、`causal_graph.json`、`report.md`（六件套；Schema 见 [shared/schemas/mechanism_case.schema.json](shared/schemas/mechanism_case.schema.json)）。分布式 v3 节点产物：`summary.json`、`suffstats.npz`、`behavior.json`、`canary.json`。只有 `kind=real`、design_sha 一致、金丝雀未漂移且审计未驳回的结果进入合并；合并统计不自动成为 Evidence。

## 本地模型测试规则

正式测试使用本地 CUDA 模型，依次加载、运行、释放，禁止同时占用 GPU：Qwen3-4B → Qwen3-14B → GLM4 → DeepSeek7B。重要结果须扩大数据量并在独立材料、lockbox 或其他模型上确认；行为门未通过的模型不得进入相同的内部机制比较。

## 构建与检查

```powershell
cd frontend && npm run lint && npm run build && cd ..
.\.venv\Scripts\python.exe -m unittest discover -s tests/dist_protocol -v
```

## 研究入口

- [GLM5 研究备忘录](research/glm5/docs/AGI_GLM5_MEMO.md) ｜ [GPT5 研究备忘录](research/gpt5/docs/AGI_GPT5_MEMO.md)
- [客户端与研究操作系统方案](ai2050_research_os/README.md) ｜ [当前智能理论](research/IntelligentTheory.md) ｜ [主分析文档](research/MainAnalysis/)
- [平台规范 v1](design/platform_spec_v1.md) ｜ [客户端 v6 信息架构](design/client_v6_pipeline_ia.md) ｜ [客户端/服务器对齐](research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md)
- [旧版完整 README（v1 归档，含理论框架与 Phase 2578 进展全文）](research/README_v1_full_archive.md)
