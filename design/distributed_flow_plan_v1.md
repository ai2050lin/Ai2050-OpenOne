# 客户端优化方案 v1 —— 分布式研发全流程贯通（M6 提案）

日期：2026-10-08 ｜ 状态：**P0+P1+P2+P3 已落地（2026-10-08）；M6 全部分期完成**
落地注记（P0）：平台进度 tab 已新增 ①新人教程四步卡（TUTORIAL_STEPS 数据层驱动，命令可一键复制，占位符说明内嵌）②双注册表卡（语言模板 LANG_TEMPLATES 可点击展开示例序列/焦点特征/状态注记；分析技术 ANALYSES 展开输入契约/口径版本；下方研究包组合 RESEARCH_KITS 6 条 DEMO 占位）③统计卡 +「可用研究包」项。纯前端零服务端改动；vite build 7.44s、dev 200。
落地注记（P1）：服务端 S7 端点上线——`GET /api/kits`（模板×分析技术枚举，availability 由结果库按 input 契约谓词自动判定：rsa-rdm=manifest 含 npz、eta2=summary 含 eta2_by_factor、cka=双结果 P2 前恒 pending）+ `GET /api/kits/{kit_id}/bundle`（bundle 文件 corpus/runner/contract + 结果清单≤50 + 服务端生成 README；下载计入各结果引用计数；负路径 400/404/409）。node_agent.py 新增 `download` 子命令（落地 bundle+manifest+README，本地重算 design_sha 复核冻结，不一致即 SystemExit）。前端研究包组合接 /api/kits 转 LIVE（离线回退 DEMO 叙事），available 行点击/按钮下载 bundle JSON；教程第 4 步命令更新为 `download --kit {tm}__{analysis}`。验证：端点测试 14/14（tests/deepseek_temp/_kits_test.py，临时数据目录全场景含 design_sha 复核与负路径）、py_compile 通过、CLI help 正常、vite build 7.83s。**中心节点 :5010 需重启加载 S7**（与 NEWS_FALLBACK_TTL 一并生效）。
落地注记（P2）：调度队列只读视图贯通——①服务端 S8 `GET /api/tasks`（公开只读：活跃租约 50 条含 node_name/gpu/model/剩余租约、done/failed 最近 30 条、全队列 stats、按模板分桶 by_template；claim 时自动清移过期租约；无 token 泄漏，claim 本身仍需 X-Node-Token）②node_agent.py 新增 `status`（凭据 → GET /api/nodes/me，本机节点信息+最近任务）与 `queue`（GET /api/tasks 全队列快照，无需凭据）两个子命令——注意 `--server` 为顶层参数须置于子命令前③流程透镜任务队列加第三 tab「调度」（S8 LIVE/DEMO_SCHED 回退；顶部 chip 三 tab 各自显示；页脚注记概念消歧：调度队列=节点租约实时状态 / 研究线=phase 依赖序 / 分布式=模板结果库）④当前机器进度 tab 本机节点横幅下新增「调度队列快照」状态条（GET /api/tasks 轮询 30s，离线显式声明 demo 叙事不代表本机实时；指路终端 `node_agent.py status/queue`——凭据不出终端，浏览器不读取 node_credentials.json）⑤空间透镜模板选择器加「研究包 ↓」虚线 chip（onGo 守卫）→ 跳平台进度研究包组合。验证：S8 端点测试 19/19（tests/deepseek_temp/_tasks_test.py：空队列→register→claim→活跃可见且无 token 泄漏→heartbeat→complete→recent→过期自动清移）、py_compile 通过、CLI 五子命令注册确认、vite build 7.56s、五个改动模块 dev 200。**中心节点 :5010 须重启加载 S8**（与 S7/NEWS_FALLBACK_TTL 一并）。
落地注记（P3）：①**一键本地复现**——研究包内嵌 `reproduce.py`（S7 bundle_version 升 S7-v1，服务端 REPRODUCE_SCRIPT 常量随包下发）：design_sha 冻结校验 → runner 复现（--model-path 走真实口径）→ 输出对账（`--server <center>:5010` 联网逐文件 sha256 比对 results_manifest 每份结果；无 server 时诚实降级为本地产出列举）→ reproduce_report.txt；README 复现步骤重写（一键复现置首）。②**AGG-v1**——新端点 `GET /api/agg/{tm_id}/v1`（_agg_v1_payload 助手与 kit bundle 共用）：桶内（模型×kind×seed，seed NULL 归一 0）η² 按因子跨节点/seed 汇总（n/mean/min/max/std），离散度强制展示、跨桶平均仍禁止（13.1）；AGG-v0 端点不回归；kit bundle 顶层 agg_v1 键随包下发。③node_agent.py 两处 bundle 落盘 write_text→**write_bytes**（修复 Windows CRLF 翻译破坏 runner 字节→design_sha 校验必挂的保真隐患）。④前端教程第 4 步命令升级为 `download --kit … && python reproduce.py`。验证：P3 测试 14/14（tests/deepseek_temp/_repro_test.py：AGG-v1 桶内 η² mean/std 正确+纪律 warning+AGG-v0 不回归+S7-v1 三要素+**bundle 落盘真跑 reproduce.py 全链路**）；回归 P1 14/14、P2 19/19；py_compile 通过；vite build 7.76s。**中心节点 :5010 须重启加载 S7-v1/AGG-v1/S8/NEWS_FALLBACK_TTL**。
遗留（M6 外）：E2E「首屏零 console error」浏览器级兜底断言需 playwright（本机未装、global 安装被纪律禁止）——暂以 vite build + dev 模块 200 + 数据层 node import 冒烟代替，待浏览器工具可用后补。
上游：`ui_decoupled_plan_v1/v2.md`（M2/M3 已落地）、`client_analysis_tech_plan_v1.md`（M4，P0 已落地）、`client_template_tech_plan_v1.md`（M5，P0 已落地）

## 0 目标句式（用户原话 → 三个可验收能力）

> 整理大量的语言模板，以及透镜等分析技术，在路线-平台进度中显示出来（需要包括新人教程说明）。然后各个分布式终端领取任务，完成后上传到服务器，任何人都可以下载服务器上的某个语言模板+分析技术数据，在本地观察和研究。

| # | 能力 | 验收 |
|---|------|------|
| C1 | **目录可见**：语言模板 × 分析技术两张注册表在平台进度 tab 可浏览 | 打开「路线 → 平台进度」即可看到全部模板/技术、各自口径与状态 |
| C2 | **新人教程**：从零参与到产出上传的完整路径说明 | 新人只看平台进度 tab 就知道：接入 → 领任务 → 上传 → 下载研究 四步怎么做 |
| C3 | **下载可研**：任意人可按「语言模板 + 分析技术」下载研究包，本地复现观察 | 一条命令（或一次点击）得到 corpus + runner 口径 + 结果清单，本地可跑 |

## 1 现状盘点（逐界面 × 逐端点，代码核实）

### 1.1 服务端：闭环已存在，缺「出口」

| 环节 | 端点（deploy/distributed_service.py） | 状态 |
|---|---|---|
| 节点接入 | `POST /api/nodes/register` | ✅ |
| 领取任务 | `POST /api/tasks/claim`（租约 6h）+ heartbeat/complete | ✅ |
| 上传结果 | `POST /api/results/upload/init\|chunk\|finish`（sha256 内容寻址，design_sha 不符拒收） | ✅ |
| 结果浏览 | `GET /api/results` · `GET /api/results/{sha}` | ✅ |
| 聚合 | `GET /api/agg/{tm_id}`（AGG-v0 分桶） | ✅ |
| **模板包下载** | **缺失** —— /api/templates 只回元数据，corpus+runner 无法整体取走 | ❌ |
| **任务队列只读视图** | **缺失** —— claim 是终端私有动作，公众看不到队列全貌 | ❌ |
| **研究包（模板+技术）** | **缺失** —— 「语言模板 × 分析技术」的组合没有一等表示 | ❌ |

### 1.2 前端逐界面

| 界面 | 现状 | 与目标的差距 |
|---|---|---|
| **路线 → 平台进度** | 统计卡 / 节点目录 / 模板测试矩阵（TM 编号）/ 聚合管线 / 协议卡 / AggLiveCard / 发布时间线 | ① 只有 TM 矩阵，**没有** LANG_TEMPLATES（4 语言模板）与 ANALYSES（3 分析技术）两张注册表的展示位；② **无新人教程**——接入方式只在一行小字里提了 deploy/README_CENTOS.md；③ 节点目录只有心跳，看不到「队列里有多少任务待领」 |
| **路线 → 当前机器进度** | 本机实验时间线（NODE-A3F7 demo 叙事） | 不显示本机 agent 的真实领取/上传状态（若本机跑了 node_agent） |
| **流程透镜** | 任务队列双 tab（研发线 LIVE / 分布式线读 phase_queue_v1.json） | 分布式线读的是**研究队列文件**，不是**调度队列**（claim 租约状态）；两个「队列」概念并存易混淆 |
| **分析技术（数据）** | 注册表驱动：按技术/按结果双模式，rsa-rdm 客户端现算 | 无**下载入口**——看到了结果却带不走 |
| **空间透镜** | LANG_TEMPLATES 选择器（M5） | 语言模板在这里是「观景模式」，与 C3 的「研究包下载」没有互相指路 |

### 1.3 数据层资产（已备好，只缺展示位）

- `LANG_TEMPLATES`（4 语言模板，is_a measured + 3 demo）——M5
- `ANALYSES`（3 分析技术，input 契约 + metric_version + evidence_level）——M4
- `TEMPLATES`（TM 测试矩阵，来自 /api/templates LIVE）——S2
- `PLATFORM_STATS / NODES / AGG_STEPS / AGG_DIMS`——M1

## 2 方案

### 2.1 主张：平台进度 = 分布式研发的「总目录 + 总教程 + 总出口」

平台进度 tab 从「状态看板」升格为**流程门户**，新增三个区块（数据全进数据层，遵守解耦红线）：

```
┌ 统计卡（已有，增 2 项：待领任务 / 研究包下载数）─────────┐
├ 语言模板注册表（LANG_TEMPLATES）│ 分析技术注册表（ANALYSES）┤  ← C1
├ 新人教程四步卡（接入→领任务→上传→下载，每步带可复制命令）──┤  ← C2
├ 节点目录 + 调度队列只读（任务池/租约状态，P2 接新端点）────┤
├ 模板测试矩阵 + 聚合管线 + AggLiveCard（已有）────────────┤
└ 发布时间线（已有）───────────────────────────────────┘
```

**新人教程四步卡**（纯前端，DEMO/LIVE 双模）：
1. **接入节点**：`python node_agent.py register --server http://<center>:5010 --gpu <gpu> --model <model>`
2. **领取任务**：`python node_agent.py run`（claim → 租约 6h → heartbeat 自动续）
3. **完成上传**：agent 自动 sha256 分块上传，design_sha 不符拒收（预注册冻结纪律）
4. **下载研究**：`python node_agent.py download --template TM-04 --analysis rsa-rdm --out ./kit`（P1 新端点）

### 2.2 「研究包」= 语言模板 × 分析技术 的一等组合（协议第七件）

C3 的核心抽象。注册表新增：

```json
// distributedData.js → 服务端 S7 对应物
RESEARCH_KITS: [{
  kit_id: 'is_a__rsa-rdm',
  lang_tpl: 'is_a',            // ← LANG_TEMPLATES
  analysis: 'rsa-rdm',         // ← ANALYSES
  input_contract: 'result.cos',
  bundle: '/api/kits/is_a__rsa-rdm/bundle',   // tar.gz：corpus.json + runner 口径 + 结果清单
  status: 'available' | 'pending_results',
}]
```

服务端新增两个只读端点（S7 研究包）：
- `GET /api/kits` —— 枚举「语言模板 × 分析技术」组合与可用性（结果库里有满足 input 契约的结果 → available）
- `GET /api/kits/{kit_id}/bundle` —— 打包下载：模板 corpus.json + runner（含 metric_version）+ 满足契约的结果 sha 清单 + README（本地复现步骤）

配套：`GET /api/tasks?public=1` 只读队列视图（任务池大小 / 各任务租约状态 / 最近完成），喂给平台进度的调度队列卡。**红线**：公开端点不暴露节点 token 与任务私有载荷。

### 2.3 逐界面改动清单

| 界面 | 改动 | 期 |
|---|---|---|
| **平台进度** | +双注册表卡（语言模板/分析技术，点击展开口径与端点）；+新人教程四步卡；统计卡 +2 项；+调度队列只读卡（P2）；每个 TM 矩阵单元加「下载研究包」下拉（P1） | P0/P1/P2 |
| **当前机器进度** | 若本机 `~/.ai2050/creds` 存在 → 顶部加「本机 agent 状态」条（最近 claim/上传，读新只读端点） | P2 |
| **流程透镜** | 分布式队列 tab 更名「研究队列」，另起「调度队列」tab 接 /api/tasks?public=1；消除双「队列」歧义 | P2 |
| **分析技术** | 模板筛选器每卡加「下载研究包」按钮（P1 有端点后转 LIVE）；DEMO 期显示 curl 命令 | P1 |
| **空间透镜** | 模板选择器旁加「获取该模板研究包 →」跳转平台进度对应 kit | P2 |
| **行业进展** | 时间轴 NOW 节点与流程无关，不动 | — |

### 2.4 node_agent.py 扩展（P1）

`download` 子命令：拉 bundle → 校验 sha256 → 落地 `kit/{lang_tpl}/{analysis}/`（corpus/runner/results_manifest.json/README.md）。与既有 register/run 同风格（argparse + http_json）。

## 3 纪律延续

- 协议键白名单渲染；JSX 无实验字面量（教程命令属于**接入文档**不是实验数据，进数据层常量）。
- LIVE/DEMO 双模：中心节点离线时教程卡照常显示（命令本身是静态文档），注册表回退 demo 数据并标注。
- metric_version 强制展示（研究包 README 必带），防跨量纲比较——13.1 红线延伸到下载物。
- 内容寻址不变：bundle 清单里每个结果带 sha256，本地复现前可校验。

## 4 分期

| 期 | 内容 | 依赖 |
|---|---|---|
| **P0** 纯前端 | 平台进度双注册表卡 + 新人教程四步卡（命令可复制）+ 统计卡扩充；数据层 RESEARCH_KITS 占位（DEMO） | 无 |
| **P1** 服务端 | GET /api/kits · /api/kits/{id}/bundle（S7）+ node_agent download 子命令；分析技术/矩阵下载入口转 LIVE | 服务重启 |
| **P2** 流程贯通 | GET /api/tasks?public=1 + 平台进度调度队列卡 + 流程透镜队列拆分 + 空间透镜指路 + 本机 agent 状态条 | P1 |
| **P3** | 研究包内含一键本地复现脚本（对齐 runner 预注册口径）；跨节点 η² 聚合 v1 纳入 bundle | P2 |

## 5 验收句式

**新人只看平台进度 tab 就能接入并产出；任何人在任何机器上一条命令取走「语言模板 × 分析技术」研究包并本地复现；注册表（模板/技术/研究包）之外无隐藏流程。**
