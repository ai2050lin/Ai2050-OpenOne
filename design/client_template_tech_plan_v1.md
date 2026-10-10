# 客户端优化方案 v1 —— 「is-a」语言模板可选化 × 数据透镜升格为分析技术视图（M5 提案）

日期：2026-10-08 ｜ 状态：**P0 已落地（P1–P3 待排期）**

> **落地注记（P0，2026-10-08）**：① `LANG_TEMPLATES` 注册表已进数据层（4 模板：is_a measured / attr·syntax·part_of demo，各带 demo 包 tokens/peaks/attn_lookback/focus/cloud/act_lines + metrics 覆盖）；② `NeuronMode`/`CloudMode` 重写为模板驱动（props.tpl，组件零模板字面量，缓存键 layer×tpl.id，t 步换模板钳制）；③ 空间透镜顶栏模板选择器（neuron/cloud 模式显示，demo 模板虚线徽标）；④ 「数据透镜」→「分析技术」：`LENS_NAME`/rail/Home 卡/融合主张/对象卡足迹/LensProcess 跨镜文案全同步；⑤ LensData 主语反转：`按技术`（默认，视图级技术导航 + 行级契约预过滤 + 展开自动运行）/`按结果`（原视图）双模式，TechPanel 支持受控 techId + autoRun + hideChips，运行内核抽取为 runTech()（rsa-rdm LIVE 无 cos 时不再误回 DEMO，显式提示缺契约）；⑥ CSS 追加 .fw-tpl-sel/.fw-tech-seg/.fw-tech-nav。验证：vite build 7.68s 通过、7 模块 dev 200、组件 is-a 字面量清零（LensProgress 研究史实 2 处按豁免条款保留）。URL `tpl`/`tech` 参数留待 P2。

上游：`design/ui_decoupled_plan_v1.md`（M2）、`design/ui_decoupled_plan_v2.md`（M3）、`design/client_analysis_tech_plan_v1.md`（M4，P0 已落地）
同构主张：M4 把「分析技术」从渲染器里解耦出来；本方案把两件剩余的硬编码升格为可选注册表——

1. **「is-a」是一个语言模板（LT），不是平台常量**：客户端的演示叙事（水果族、`X 是一种 Y`、F#3734）全部绑定 is-a 一种关系。语言模板应可注册、可枚举、可切换（is-a / attr / syntax / part-of / …），渲染器不变、叙事跟随。
2. **「数据透镜」应升格为「分析技术」视图**：M4-P0 的技术注册表（ANALYSIS_TECHS）目前只是数据透镜里的附属面板，视图主语仍是「结果」。应反转为主语是「技术」——选技术 → 列出满足其输入契约的对象 → 渲染输出。

---

## 1 现状盘点（硬编码清单）

### 1.1 is-a 硬编码分布（rdc_fusion 内 grep 实测）

| 文件 | 处数 | 内容 | 可选化方式 |
|---|---|---|---|
| `demoData.js` | 3+ | `ACT_LINES`（水果 token 激活序列）、`CMDK_GROUPS`（F#3734/F#0821 is-a 条目）、`DEMO_OBJECT.label`、`DEMO_ARTIFACTS`（`arms=("is_a","attr",...)`）、`DEMO_TERMINAL` | 全部迁入 LT registry 的 `is_a` 模板包 |
| `NeuronMode.jsx` | 6 | token 时间轴 demo 序列、候选特征徽标「is-a 上位」、叙述注释（is-a 绑定故事线） | 接收 `template` props，demo 序列从 registry 取 |
| `CloudMode.jsx` | 4 | 点云族 `fam:'is-a'`、图例「水果族 · is-a」、E_ar 叠加按钮 alert 文案 | 点云族/图例/箭头按 registry 渲染 |
| `LensProgress.jsx` | 1 | 步进器「P4–P7 权重绑定定 is-a」 | **豁免**：研究史实记录，非叙事模板（见 §5） |
| `distributedData.js` | 3 | `PLATFORM_STATS.task`、`NODES[0].tm`、`TEMPLATES` TM-07「is-a 关系族（四臂消融）」（TM-08 attr 已存在） | TM 即模板的分布式采集实例，与 registry 合并口径 |
| `LensData.jsx` / `LensProcess.jsx` / 协议组件 | 0 | 已协议驱动 ✓ | 无需改 |

### 1.2 数据透镜现状

- tab 名「数据透镜」（`RdcFusionWorkspace.jsx` `LENS_NAME.data`），视图标题「数据透镜 · 结果浏览器」，主语=结果，技术选择是行内附属面板（M4-P0 `TechPanel`）。
- 已有资产可直接复用：`ANALYSIS_TECHS` 注册表（rsa-rdm 可运行 / eta2 可运行 / cka 置灰）、输入契约可用性判定、`CellCosGrid mode=rdm`、对象卡技术足迹。
- 跨透镜入口文案：「到数据透镜看结果」（LensProcess）、「点击数据透镜可再次运行」（对象卡足迹）。

---

## 2 设计 A：语言模板注册表（LT registry）

**数据层**（`distributedData.js` 新增第六·二件）：

```js
export const LANG_TEMPLATES = [
  { id:'is_a', name:'is-a 上位/下位', pattern:'{X} 是一种 {Y}',
    demo:{ act_lines:ACT_LINES_isA, cloud_fams:[...], object:{...F#3734...} },
    metrics:{ E_read:0.331615, metric_version:'v4' } },
  { id:'attr', name:'属性关系', pattern:'{X} 是 {ADJ} 的',
    demo:{ act_lines:..., cloud_fams:..., object:{...} },
    metrics:{ E_read:null } },            // 未测量 → 显式置灰，禁复用 is-a 数字
  { id:'syntax', name:'句法角色', pattern:'{X} {V} {Y}（语序消融）', demo:{...} },
  { id:'part_of', name:'部分-整体', pattern:'{X} 的 {Y}', demo:{...} },  // DEMO 占位
];
```

**组件落点**：
- 空间透镜顶栏加模板选择器（chips，复用 `fw-tech-chip` 样式惯例）；`template` 状态提升到 `LensSpatial`，下传 `NeuronMode`/`CloudMode`；`StackMode`/`ParamMode` 与模板无关不动。
- 点云族、token 序列、对象卡、CMDK 对象条目全部改为按当前模板渲染；默认 `is_a`，切换后叙事整体跟随。
- `TEMPLATES`（TM-07/TM-08）与 registry 对齐：TM = 某模板的分布式采集任务实例，`tm.template_id` 关联。
- CMDK/对象注册表条目带 `template_id`，过滤随选择器走。

**红线延续**：demo 内容仍只允许存在于数据层文件；组件内 **禁止出现 `is-a` 字面量**（M3「JSX 无实验字面量」红线的模板版）。切换模板 = 换数据种子，四个透镜渲染器零改动。

---

## 3 设计 B：数据透镜 → 分析技术视图

**更名与主语反转**：
- tab「数据透镜」→「分析技术」（`LENS_NAME.data='分析技术'`），视图标题「分析技术 · 结果浏览器」。
- 现有 `TechPanel` 升格为**视图级导航**：顶部技术 chips（注册表驱动，可用性由输入契约自动判定）→ 选中某技术 → 下方只列满足其 input 契约的结果 → 展开即渲染该技术输出（rsa-rdm 热图 / η² 条形）。
- 保留「按结果浏览」为第二模式（seg 切换：**按技术 / 按结果**）——结果浏览器完整保留，只是不再默认。
- 跨透镜文案同步：「到数据透镜看结果」→「到分析技术运行」；对象卡足迹「点击数据透镜可再次运行」→「点击分析技术可再次运行」。

**URL 即状态扩展**：`#data` → 增加 `tech` 参数（如 `tech=rsa-rdm`），M2 红线「URL 即状态」自然延伸。

---

## 4 分期

| 期 | 内容 | 依赖 |
|---|---|---|
| **P0** 纯前端 | LT registry + 空间透镜模板选择器 + is-a demo 数据整体迁入 registry；tab 更名 + TechPanel 升格视图导航 + 跨透镜文案同步 | 无（复用 M4-P0 资产） |
| **P1** 服务端 | `GET /api/templates`（从 metric_dict / TM 协议派生已采集模板及指标覆盖）、按模板过滤结果 | metric_dict v4 |
| **P2** 状态与 e2e | URL `tech` 参数；`dist_e2e.py` 增断言：切模板/切技术后渲染器 DOM 结构不变 | P0 |
| **P3** 用户自定义 | 自定义模板（prompt pattern 输入 → 生成采集任务草案，进 DEMO_QUEUE 流程） | P1 |

**验收句式**：换语言模板，四个透镜叙事跟随、渲染器不变；换分析技术，同一结果对象输出跟随；注册表（LANG_TEMPLATES + ANALYSIS_TECHS）之外无隐藏叙事。

---

## 5 红线与豁免

1. **LensProgress 步进器豁免**：「P4–P7 权重绑定定 is-a」是研究史实（MEMO Phase 记录），不是可选叙事——保留 is-a 字样，不计入硬编码红线。
2. **指标不得跨模板复用**：attr/syntax 模板未测量的指标显式显示「未测量」，禁止把 is-a 的 E_read/ share 数字带过去（量纲/来源纪律的模板版）。
3. **demo 内容只进数据层**：`demoData.js` + `distributedData.js` 两处；registry 的 demo 包也在数据层。
4. **tab 更名须同步 e2e 选择器**（`tests/platform/dist_e2e.py` 中「数据透镜」文案断言）。

## 6 风险

- **切换器状态与 URL 失步**：模板选择进入 URL（`tpl=is_a`），否则刷新丢状态（M2 红线）。
- **CloudMode 点云族只有 is-a 实测形状**：attr/syntax 族先以 DEMO 占位（同族生成逻辑、标注 DEMO），接入 Q20 扩 target 族结果后转 LIVE——与「LIVE/DEMO 双模」红线一致。
- **NeuronMode 叙事注释大量绑定 is-a 故事线**：注释随 demo 包迁移，组件注释只写接入点，不写模板内容。
