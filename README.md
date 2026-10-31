# OpenOne / AI2050

OpenOne 是一个面向语言机制、神经网络原理与通用智能的逆向工程研究项目，核心问题：

> 深度神经网络如何把知识网络、逻辑推理和语法规则融合为稳定的语言能力？
> 它的底层编码、动态运行和形成机制是什么？

项目由三部分组成：

- **研究方法与纪律** —— 找特征的三引擎闭环、五段式冻结、证据门链、失败账本；
- **分布式研发平台** —— 生成式模板 → 试跑行为门 → 分片全场采集 → 合并分析 → 语言谱图，后续扩展字典训练、特征标注、归因与干预任务；
- **可视化客户端** —— 在同一份真实数据上做研发控制、3D 观察学习和图谱查询。

研究进展以研究备忘录为准，本 README 不维护进度（见[当前状态](#当前状态)）。

## 开放平台（v1.0）

AI2050 是一个开源的分布式研究平台。中心服务器保存图谱数据与研究进度，任何人都可以登录浏览、学习，也可以用自己的 GPU 跑测试分片并上传。

| 我想… | 去哪里 |
| --- | --- |
| 看进展、学方法、浏览图谱与神经元卡 | 平台网站：首页 / 学习 / 图谱 |
| 用自己的 GPU 参与 | 网站「参与」页下载一键包；说明见 [CONTRIBUTING.md](CONTRIBUTING.md) |
| 部署自己的中心 | [deploy/README_DEPLOY.md](deploy/README_DEPLOY.md)（`scripts/build_center_release.py` 生成部署包） |
| 参与研发 | [AGENTS.md](AGENTS.md)（研究纪律与协议）· `tests/dist_protocol`（协议测试） |

第一版开放两类任务：

- **模式族测试**：冻结的生成式模板，节点逐分片采集，中心合并统计。
- **神经元普查分片**：一段自然语料 × 一组层的 MLP 神经元统计。中心逐项校验，并抽样派给另一个账号复算。

两类任务的任务包都按 sha256 冻结，参与者无需修改任何东西。

**许可**：代码采用 Apache-2.0（[LICENSE](LICENSE)），平台公开的研究数据采用 CC BY 4.0（[NOTICE](NOTICE)）。

---

## 快速启动

最简方式：双击仓库根目录的 `start_client.bat`（关闭窗口即停止客户端）。
在 VSCode 里打开本工程即会**自动确保客户端在跑**（终端任务 `AI2050: Frontend`，
幂等：已在运行就复用，3 秒内打印地址）。

```powershell
# ── 前端 ── 默认 http://localhost:5173（5001 是后端 API，勿混淆）
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_visualization.ps1
.\scripts\start_visualization.ps1 -Install          # 首次安装依赖
.\scripts\start_visualization.ps1 -Port 5174        # 更换端口
.\scripts\start_visualization.ps1 -Stop             # 停止并确认端口释放
.\scripts\start_visualization.ps1 -Status           # 查端口 / 守护进程 / 日志
.\scripts\start_visualization.ps1 -Foreground       # 前台守护（Ctrl+C 停止）
```

前端启动由 `scripts/dev_server.mjs` 托管：幂等复用、僵死端口回收、等到 HTTP 200
才打印地址、vite 异常退出自动退避重启；全部输出带时间戳写入 `.run/devserver.log`。
**客户端打不开时先看 `-Status`，再看该日志**——不再需要靠重开碰运气。

```powershell
# ── 后端 ── 默认 http://localhost:5001
.\.venv\Scripts\python.exe .\server\install_deps.py      # 首次安装依赖
.\.venv\Scripts\python.exe -m server.server              # 启动后端
$env:AI2050_SKIP_MODEL_LOAD = '1'                        # 仅验证 API 时设置
.\.venv\Scripts\python.exe -m server.server              # 不加载本地模型启动
```

分布式中心与 GPU 节点的启动方式：见 `deploy/README_CENTOS.md` 与 `design/platform_spec_v1.md`。

## 研究主路线

```text
外部语言模式族图谱（知识 / 推理 / 语法的类型化超图）
  ↓ Embedding / HiddenState 完整场（全坐标保存，Top-K 只用于显示）
  ↓ 语言操作 ↔ 内部条件算子的关联机制（依次通过证据门链）
  ↓ 网络可塑性与形成规律
  ↓ 可预测、可干预的智能理论 → AGI 工程
```

证据门链（依序判定，不可跳步）：

```text
行为资格 → 仪器 → 重复 → 选择性 → 充分性
        → 必要性 → 负控制 → 未见预测 → 自然生成 → 跨模型
```

三条工程主线：

1. **外部图谱** —— 语言模式（知识、任务、语法、序列、输出）组织成可执行对象，自动生成 discovery / confirmation / lockbox 数据；
2. **内部图谱** —— 对每个行为合格的语言操作，保存全部 token、全部层、全部坐标的 HiddenState 场及层更新、条件差分；以特征卡为记录单位；
3. **关联机制** —— 寻找可预测、可干预的「外部语言操作 → 内部算子」对应，依次通过证据门链，不做「苹果神经元」式枚举。

### 找特征：三引擎与闭环

"先找特征 → 再完成图谱 → 最后破解"中，找特征的方式决定效率。只用实验者标注的因子求方向，只能验证已经想到的变量，所以拆成三个互补引擎：

| 引擎 | 做什么 | 产出 |
| --- | --- | --- |
| A 自底向上：字典学习 | 在自然语料激活上训练或借用 SAE / transcoder，得到不依赖标注的稀疏特征 | 内部图谱的节点（特征卡） |
| B 自顶向下：受控对照 | 用最小对模板度量特征对语言变量的选择性，并在未见词表、未见句式上复现 | 外部模式 → 内部特征的边 |
| C 因果归因 | 归因图、attribution patching、预注册 steering / 消融 | 特征 → 特征 → logit 的计算边 |

```text
A 提出候选特征 → B 检验它表达哪个语言变量、能否推广 → C 检验它是否参与预测
      ↑                                                            ↓
      └──── 失败与反常样本 → 新的预注册文法 → 分布式运行 ←────────────┘
```

关键原则：

- **读出位置的因果原则**：decoder 模型中某位置只看得到前缀，只能在区分信息全部出现之后读出；对照句式要控制句首前缀的有无与长度。
- **采一次，多次分析**：模板族默认采集全部层（含嵌入层）与多个读出位置，节点本地保留原生全场，方法改进时重算而不重跑。
- **先小后大**：方法先在小模型或参考模型（有公开字典与归因工具的开源模型）上迭代，再上主研究模型。
- **可度量的"破解"**：用谱图预测保真度衡量，即只用图谱在留出句子上预测特征激活与下一 token 分布，与真实模型比较，并与词汇、位置等简单基线对照；不用覆盖率或特征数量充当进度。

理论框架、路线分析、大脑视角类比与常见误区的完整论述：见[旧版 README 归档](research/README_v1_full_archive.md)与[智能理论总整理](research/IntelligentTheory.md)。

## 平台组成

| 目录                               | 内容                                                               |
| ---------------------------------- | ------------------------------------------------------------------ |
| `frontend/`                        | 可视化客户端（React + Vite + three.js，五个一级界面）              |
| `server/`                          | FastAPI 后端：模型接口、研究运行、中心路由自动挂载（:5001）        |
| `shared/dist_protocol/`            | 分布式测试协议 v3：生成式模板、充分统计量、金丝雀、合并、证据门链  |
| `shared/dist_protocol/grammar.py`  | 文法校验与生成：因果可见性检查、读出位置、确定性切分、试跑抽样     |
| `shared/dist_protocol/runner_v3.py`| 节点 runner：全层多读出采集、逐层 logit lens、模型快照哈希、本地原生全场 |
| `shared/dist_protocol/grammars/`   | 内置模板族文法（template_grammar.v1）                              |
| `shared/dist_protocol/analysis.py` | 规律分析（analysis.v1）：η²、置换检验、句式不变性、可加性、3D 坐标 |
| `shared/dist_protocol/atlas.py`    | 语言谱图（atlas.v1）：跨族稳定特征、成边、共享编码轴、概念假设     |
| `deploy/`                          | 中心节点服务与节点 Agent（本地保留，不进公开仓库）                 |
| `design/`                          | 平台设计方案（client_v6_pipeline_ia.md、platform_spec_v1.md）      |
| `tests/dist_protocol/`             | 协议与中心测试（unittest，改动协议后必须全绿）                     |
| `research/`                        | 研究记录：GLM5 / GPT5 备忘录、智能理论、主分析、README 归档        |
| `tests/glm5*/` `tests/gpt5*/`      | 研究测试脚本与结果（正式 / 临时 / result）                         |
| `models/`                          | 本地模型入口                                                       |

## 客户端

| 界面 | 回答的问题                                                 |
| ---- | ---------------------------------------------------------- |
| 总览 | 闭环推进到哪、题库状态、待我决策什么                       |
| 空间 | 模型内部对条件怎样响应（3D 响应空间、规律分析、语言谱图）  |
| 研发 | 下一批测什么（神经元测试、模板流水线、题库、普查作业、自动测试） |
| 图谱 | 语言—响应关联与证据来源                                    |
| 路线 | 每个假设卡在哪一道证据门                                   |

3D 与图谱用于观察学习和研究，原则是每个视图都对应一个可回查原始数值的查询：精细比较用二维，三维用于全局概览，布局距离不代表模型几何。规划中的核心视图：

| 视图 | 含义 | 回答的问题 |
| ---- | ---- | ---------- |
| 响应空间 | 点 = 样本，轴 = 白化 LDA / PCA（标注层与读出位置） | 因子是否把样本分开、是否跨句式稳定 |
| 层流轨迹 | 同一样本跨全部层的轨迹，可叠加 logit lens | 响应如何跨层接续、何时可读出答案 |
| 特征谱图 | 节点 = 特征卡，边按类型与证据等级着色 | 特征之间的联系、族间共享结构 |
| 归因图 | token → 各层特征 → logit 的分层图 | 一次预测经过哪些计算边 |

观察学习入口（设计中）："输入一句话"后显示被激活的特征、图谱预测与模型真实输出的对比；反常句子可一键生成文法草稿，进入预注册流程。

- 没有真实数据时显示空态，不以示意图冒充真实 HiddenState 或计算连接；各来源计数分开呈现，不拼成转化率。
- URL 即状态：`?lens=home|spatial|process|atlas|progress`；`⌘K` 快速跳转。
- 账号体系（accounts.v1）：第一个注册账号为管理员；写操作需管理员登录或管理令牌。
- 界面细节：见 [design/client_v6_pipeline_ia.md](design/client_v6_pipeline_ia.md)。

## 分布式测试闭环

| 步骤           | 内容                                                |
| -------------- | --------------------------------------------------- |
| ① 生成式模板   | 句式框架 × 填充词表 × 因子水平；声明读出位置（target / cue / final / probe_answer）与采集层 |
| ② 静态检查     | 因果可见性（因子在读出位置是否按构造不可见）、标签词泄漏、句首前缀不对称告警 |
| ③ 确定性切分   | 按填充词分 discovery / confirm / lockbox，句式泛化用 lockbox_frame，防泄漏 |
| ④ 试跑与冻结   | 试跑只从 discovery 抽样看行为门；design_sha 固定语料、runner 与预注册阈值 |
| ⑤ 分片派发     | GPU 节点全层采集，回传充分统计量、逐样本向量、逐层 logit lens、行为结果与金丝雀；原生全场留在节点本地 |
| ⑥ 中心合并     | 按「模型快照 × 精度」合并；规律分析（analysis.v1）与语言谱图（atlas.v1） |
| ⑦ 证据门链判定 | 等级在 Research OS 人工登记，合并统计不自动晋级     |

关键设计：

- 可合并统计量：节点回传 n、Σx、Σx²，跨节点直接相加，与单机全量计算一致；
- 质量控制：金丝雀漂移检测、lockbox 防泄漏、植入找回与负对照测试；
- 模型身份：权重与 tokenizer 文件的 sha256（model_snapshot），并对账 hs[-1] 是否已过最终归一化；
- 逐样本向量：items.npz（超限时用随机投影，投影维数与矩阵只由语料决定，各分片一致）；节点本地按层保存 float16 原生全场与 manifest（`AI2050_NATIVE_DIR`，缺省 `~/.ai2050/native_fields`）；
- 行为门评分：probe 声明可接受答案集与对照集，按完整答案对数概率强制选择；没有对照集时看下一 token 是否为任一可接受答案的首 token。旧口径（首 token top-1）继续记录。

一键跑完一个模式族：`python scripts/run_language_family.py --family KN-ISA --center http://127.0.0.1:5001 ...`，或在客户端「研发 · 自动测试」操作。完整规范见 [design/platform_spec_v1.md](design/platform_spec_v1.md) 与 AGENTS.md 第 6 节。

## 当前状态

- **进度不写进 README** —— 权威记录是研究备忘录，按时间追加：
  - [GLM5 研究备忘录](research/glm5/docs/AGI_GLM5_MEMO.md)（行为基线与完整场线）；
  - [GPT5 研究备忘录](research/gpt5/docs/AGI_GPT5_MEMO.md)（G 线现役：registry / gap 账本 / 客户端对齐）。
- 快照锚（2026-10-15）：测量账本 n=389、registry v1.32、gap 账本 v1.11（GAP-5 定谳 UPSTREAM_ABSENT）。
- 平台现状：分布式协议 v3 与客户端 v6 已实现并通过离线测试；内置模板族已开始在 Qwen3-4B 上真实试跑，结果以研究记录为准。
- 对接清单：平台与研究的对接见 [CLIENT_SERVER_ALIGNMENT_v1.md](research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md)。

## 纪律与工件

核心纪律（完整版见 AGENTS.md）：

- 先行为后内部，先仪器后机制；
- discovery / confirmation / lockbox 严格分离；
- 对象、阈值、分区预先冻结，看到结果后不得修改；
- 正结果、反证、失败全部记账；AI 意见不属于实验复现。

机制边闭合门：`E = I ∧ S ∧ N ∧ C ∧ P ∧ R`
（仪器身份 / 充分性 / 必要性 / 特异性 / 预注册预测 / 独立重复）。

标准研究工件：

- 六件套：`manifest.json`、`samples.jsonl`、`metrics.json`、`events.jsonl`、`causal_graph.json`、`report.md`（Schema 见 [shared/schemas/mechanism_case.schema.json](shared/schemas/mechanism_case.schema.json)）；
- v3 节点产物：`summary.json`、`suffstats.npz`、`behavior.json`、`canary.json`；
- 合并条件：`kind=real`、design_sha 一致、金丝雀未漂移、审计未驳回。

## 本地模型测试规则

1. 顺序加载：Qwen3-4B → Qwen3-14B → GLM4 → DeepSeek7B；依次运行、释放，禁止同时占用 GPU；
2. 重要结果须扩大数据量，并在独立材料、lockbox 或其他模型上再次确认；
3. 行为门未通过的模型，不得进入相同的内部机制比较；
4. 方法先在小模型或参考模型上迭代，结论与主研究模型分开记录，坐标与方向只在同一模型内比较。

## 改进路线（计划，非已完成进度）

| 阶段 | 内容 | 进入下一阶段的依据 |
| ---- | ---- | ------------------ |
| P0 修正与首次真实闭环 | 文法因果检查、全层多读出采集、模型快照、试跑只用 discovery（已实现）；四个内置模式族在 Qwen3-4B 上跑通 试跑 → 采集 → 分析 → 谱图 | 真实数据上至少一族得到可复现的候选观测，或明确方法在哪里失效 |
| P1 自底向上引擎 | 接入公开字典或在小模型上自训 SAE；特征卡与特征库；模板选择性挂到字典特征上 | 字典特征在未见词表上复现模板选择性 |
| P2 因果引擎 | 归因与预注册干预任务类型；边的证据分级；开始计算谱图预测保真度 | 有一批经干预检验的边，保真度有基线可比 |
| P3 规模化 | 分布式字典训练与标注；跨模型特征匹配；扩展模式族与组合深度 | 跨模型、跨族规律可比 |

## 构建与检查

```powershell
cd frontend ; npm run lint ; npm run build ; cd ..
.\.venv\Scripts\python.exe -m unittest discover -s tests/dist_protocol -v
```

## 研究入口

- **研究备忘录**：[GLM5](research/glm5/docs/AGI_GLM5_MEMO.md) ｜ [GPT5](research/gpt5/docs/AGI_GPT5_MEMO.md)
- **理论与分析**：[智能理论](research/IntelligentTheory.md) ｜ [主分析文档](research/MainAnalysis/) ｜ [研究 OS 方案](ai2050_research_os/README.md)
- **平台设计**：[平台规范 v1](design/platform_spec_v1.md) ｜ [客户端 v6 信息架构](design/client_v6_pipeline_ia.md) ｜ [客户端/服务器对齐](research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md)
- **历史归档**：[旧版完整 README（v1，含理论框架与 Phase 2578 进展全文）](research/README_v1_full_archive.md)
