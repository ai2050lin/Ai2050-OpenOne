# 客户端优化方案 v1 —— 从「透镜为核心」到「技术可插拔」（M4 提案）

日期：2026-10-08 ｜ 状态：**P0 已落地**（2026-10-08，P1-lite 形态）
上游：`design/ui_decoupled_plan_v1.md`（M2）、`design/ui_decoupled_plan_v2.md`（M3 均已落地）

> P0 落地注记（2026-10-08）：实施与提案的差异——
> ① P1 的服务端 `POST /api/analyses/{id}/run` 暂缓：`/api/results/{sha}` 已尽力提取
>   `means.npz` 的 cos 矩阵，rsa-rdm 改为**客户端现算**（RDM = 1 − cos），零服务端改动即达「对真实结果运行 RSA」；
>   内容寻址缓存与服务端现算留到 P1 正式形态（大 keys top-K 截断时再上服务端）。
> ② 已落地：`distributedData.js` 新增 `ANALYSES` 注册表（rsa-rdm / eta2-decompose / cka-linear 三条目，
>   input 契约 + metric_version + evidence_level）与 `DEMO_RDM` 演示输出；
>   `LensData.jsx` 新增 TechPanel（技术 chips 按契约过滤可用性、缺输入/未开放显式置灰并给原因、
>   rsa-rdm 现算 RDM→`CellCosGrid` 新增 mode="rdm" 渲染 + 最分离/最相似对摘要、
>   eta2-decompose 条形视图、CKA 登记 disabled 待 P2）；`RdcFusionWorkspace.jsx` 对象卡新增
>   「技术足迹」行（schema 扩展 `tech_footprint[]`，纯渲染器，LIVE 服务端未供即不显示），
>   `demoData.js` DEMO_OBJECT 播种两行足迹。
> ③ 验证：vite build 通过；dev 按需编译五模块 200。浏览器级截图验证与 e2e 断言待补。

## 0 核心思路：透镜降级为骨架，技术升格为一等公民

现状：客户端以四透镜（空间 / 过程 / 脉络 / 数据）为核心组织。透镜本质是**固定的浏览视角**——
数据透镜只会按协议键渲染结果卡，空间透镜只有四种固定模式。分析视角是写死在组件里的：
想用 RSA 看、换 CKA 看、再叠一层 η² 分解，每种都要改代码。

新思路：**对象（object）是主角，分析技术（analysis technique）是可插拔的镜头**。
用户选定一个对象/结果/模板后，应能从「技术清单」中选择任意可用技术施加于它，
输出统一走协议键渲染。透镜保留为工作流骨架（研发流程、行业进展、空间导航），
不再独占「分析能力」的定义权。

一句话判据：**透镜回答「去哪里看」，技术回答「用什么看」——后者应当可注册、可枚举、可替换。**

## 1 现状核实（代码事实）

| 现状 | 位置 | 与新思路的差距 |
|---|---|---|
| 结果渲染固定：summary_digest 协议键白名单平铺 | `LensData.jsx` + `protocolComponents.jsx` | 技术视角写死（η²/指纹/前缀 Jaccard 有啥显啥），无法选择或叠加 |
| RSA 已有领域叙事（时间轴 + 对比注脚），但无「对数据运行 RSA」的入口 | `LensProgress.jsx` + `distributedData.js`（RSA_ERAS/RSA_NOTE） | 叙事与能力脱节：看得见 RSA 的历史，用不了 RSA |
| cos 矩阵已随结果上传（`means.npz` 含 keys/means/eta2_factor_dim/cos） | `deploy/dist_templates_seed.py`、runner v2 协议 | RDM/RSA 的**原材料已在库里**，但没有消费端点 |
| 空间透镜四模式固定（点云/热图等 demo） | `LensSpatial.jsx` + `CloudMode/NeuronMode/StackMode/ParamMode` | 模式=组件，无注册机制，加一种模式要动主组件 |
| 对象卡已协议化（object_registry_v1 + `GET /api/object/{fid}`） | M3-P2 落地 | 对象卡只登记「被谁看过（透镜）」，不登记「被哪些技术分析过、结论如何」 |
| 技术口径有登记惯例（`metric_dict` v2→v4、E_read/E_ar 口径） | research/atlas 层 | 前端完全不可见，用户无法知道某结果的量纲与适用条件 |

## 2 设计：分析技术注册表（协议第六件）

### 2.1 技术登记 schema

```json
{
  "id": "rsa-rdm",
  "name": "RSA / RDM 相似性结构",
  "input":  ["result.means.npz:cos", "result.keys"],
  "output": {"type": "rdm-matrix", "protocol_keys": ["rdm_cells", "rsa_summary"]},
  "metric_version": "metric_dict-v4",
  "evidence_level": "observed",
  "source": "builtin",
  "status": "active"
}
```

要点：
- **input 契约**决定可用性：选中对象的 `means.npz` 没有 `cos` 键，RSA 技术自动置灰并说明缺什么——
  可用性由契约匹配得出，不硬编码。
- **output 契约**对齐现有协议键白名单渲染：新技术的输出仍是「协议键 → 协议组件」，
  不为单个技术写专用渲染分支。
- `metric_version` 强制登记口径：延续 metric_dict 纪律，**不同量纲并排只读、禁止跨量纲平均**（research_os 13.1）。

### 2.2 服务端

- `GET /api/analyses`：返回注册表（内置技术 + 未来用户注册技术）。
- `POST /api/analyses/{id}/run`：对指定 result sha 现场计算（v1 只实现 rsa-rdm：
  读 `means.npz` 的 cos 矩阵 → 上三角 RDM → Spearman 摘要 + 热图数据），
  结果**内容寻址缓存**（输入 sha + 技术 id + 参数 → 输出 sha），可复现、可登记。
- 计算结果写回结果附注（`analysis_runs`），对象卡由此显示「技术足迹」。

### 2.3 前端落点（全部复用现有惯例）

1. **数据透镜 · 技术选择器**：ResultDetail 顶部加技术下拉（按 input 契约过滤可用项）；
   选技术 → `run` → 输出协议键渲染。无 LIVE 时 DEMO 徽标，红线与 LensData 一致。
2. **对象卡 · 技术足迹**：evidence 区新增「已用技术」行（RSA·CKA·η²…，各带 evidence_level 徽标），
   点击跳到对应输出——对象路由闭环的横向扩展。
3. **空间透镜 · 模式注册化**：现有四模式改为注册表驱动（mode = 特殊的 view 技术），
   新增模式（如 RDM 热图、轨迹叠加）零改主组件。
4. **行业进展 · 叙事↔能力打通**：RSA 时间轴论文行加「对该技术运行」按钮（有 LIVE 结果时直达技术选择器），
   消除「看得见用不上」的脱节。
5. **多技术并排**：同一对象选择多种技术 → 输出并排卡片，卡片间标注量纲桥
   （沿用 Q04 的 rel 桥思想：量纲不同只可并排读，需显式给桥）。

### 2.4 不动什么（纪律延续）

- 协议键白名单渲染、LIVE/DEMO 双模 + 徽标、URL `?lens=&obj=&tech=` 即状态——全部沿用。
- **JSX 无实验字面量红线扩展到技术层**：技术名、参数、口径说明只能进数据层
  （服务端注册表 / `demoData.js` 回退）。
- 四透镜骨架保留为流程件；本方案不改 LensProcess/LensProgress 的结构。

## 3 分期与验收

| 期 | 内容 | 依赖 | 验收断言 |
|---|---|---|---|
| **P0**（纯前端壳） | 数据透镜加技术选择器（DEMO 数据演示 rsa-rdm/cka/eta2 三个条目）；对象卡加技术足迹 DEMO 行 | 无 | 切换技术，输出区变化且徽标正确；JSX 无技术字面量 |
| **P1**（首个真技术） | `GET /api/analyses` + `POST /api/analyses/rsa-rdm/run`（从 cos 现算 RDM）+ 热图渲染 | results 库有 means.npz | 对 TM-04 真实结果现算 RDM，输出 sha 可复现；无 cos 键的对象该技术置灰并提示缺什么 |
| **P2**（插槽化） | ResultDetail 技术输出插槽 + 多技术并排 + 对象卡技术足迹 LIVE | P1 | 换 result，同一技术输出跟随；对象卡足迹随 analysis_runs 联动 |
| **P3**（开放注册） | 用户上传自定义分析算子（脚本 bundle，内容寻址登记，复用 runner 模式） | P2 | 上传一个自定义技术 → 契约校验 → 出现在注册表并可运行 |

验收总则（沿用 M3 句式）：**换任意技术，渲染器不变输出变；换任意对象，同一技术输出跟随；注册表之外无隐藏技术。**

## 4 硬伤与风险（先行声明）

1. **量纲纪律风险**：多技术并排最容易诱发跨量纲比较——`metric_version` 强制展示 +
   并排卡显式标注「不可平均，仅并排读」，P1 起就要做，不能拖到 P2。
2. **现算成本**：RSA/CKA 在大 keys（万级特征）上 O(n²)，v1 限 top-K（如 top-64 维/cell）并登记截断口径。
3. **注册表膨胀**：技术多了之后选择器需要分组/搜索——⌘K 已有分组先例，P2 复用。
4. **叙事与能力边界**：行业进展时间轴的 RSA 论文是领域史，客户端能力是工程实现，二者只做链接不做互证。

## 5 与既有路线的关系

- 行业进展 tab 的双时间轴（机制可解释性 / RSA）不改动——它们是本方案「技术叙事层」的样板，
  rsa-rdm 是第一个从叙事走进注册表的技术。
- v1 遗留的「AI 模板生成器」「空间特征目录」不受影响：模板生成器产物自动进队列（M3 已接），
  特征目录条目自动进对象卡；本方案给它们再加一层「可用技术」的横切视角。
