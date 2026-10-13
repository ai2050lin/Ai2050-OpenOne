# 语言图谱「三域树」研究方案 v1（atlas_tree_v1）

> 2026-10-11 · 基于 registry v1.28（26 特征）+ atlas_ledger n=374（Phase 3222 收官态）+ 用户三域提案。
> 原则：本方案是现有资产的**派生视图 + 扩展协议**，不推翻 TMA ledger / registry 任何字节。

## 0. 提案与判定

用户提案：语言图谱 = 知识网络 / 分析推理 / 语法系统 三部分 → 每部分多个语言族 → 细分到 token → 每个都有对应 LLM 编码机制。

**判定：方向可行，四处必须改造。**

### 0.1 可行的部分

1. **三域顶层与现有 K/R/S 族高度兼容**：26 特征全部可无损归位（§2 映射表），无一悬空。
2. **族→token 细分链已有两套现成资产**：受控模板 TM-xx（客户端「模板与聚合」卡）+ token 级证据样本结构（LANG_TEMPLATES 的 tokens/peaks/attn_lookback/focus），叶层不是从零建。
3. **「对应编码机制」与主轴三段方法论一致**：机制归因字段（主轴位置、执行者、E 级）已在 FTR-09/10/11 等特征中成熟。

### 0.2 三个结构性硬伤（不改会塌）

| # | 硬伤 | 证据 | 改造 |
|---|------|------|------|
| H1 | **三分法不完备，交叉是常态不是例外** | 26 特征中 4 个是交叉族（SxK/RxS/RxK：FTR-05/06/07/08）；FTR-14 的 T_C 共享分量「关系无关」本身跨域 | 三域只做**展示层主视图**；数据层交叉存为一等边 `cross_link`，树上可复现但唯一存储（信息唯一家） |
| H2 | **「每部分对应一个编码机制」是多对多错觉** | attention 再分配（FTR-10）同时服务知识读出与上下文门控；C_steer=0（FTR-16）、RoPE 纯相对（FTR-13）是全域命题 | 机制独立为**横切层 MEC-xx**，特征↔机制用多对多 `encoded_by` 边；每域不做专属机制承诺，改为「该域已确证机制覆盖表」 |
| H3 | **token 不能当图谱节点粒度** | FTR-12 前缀塌缩二元门、FTR-13 位置平移不变、FTR-14 变换代数：token 级响应是位置/上下文的函数，提为节点会爆炸且证据不稳定 | 叶子 = **锚定证据样本 SAM-xx**（受控句 + token 级读数），挂在特征 anchors 下实例化；图谱节点粒度保持为特征；样本不独立评级，继承所引特征 E 级 |

### 0.3 两个工程级改进

- **E1（兼容性）**：新结构必须是 registry v1.28 + ledger n=374 的**派生层**（生成器产出 `atlas_tree_v1.json`，附 derived_from 双 sha8），否则 374 条测量与 19 条失败记录的证据链断裂，违反 immutable/append-only 纪律。
- **E2（空域风险）**：AR 域现状仅 1 独立特征（FTR-03）+ 2 交叉，GR 域更薄——三域视图会立刻暴露证据密度失衡。这是缺口驱动选题的优点，但必须**配域覆盖度指标并写进门控纪律**：禁止「为填树降 E 级门槛」。

## 1. 数据结构（atlas_tree_v1 schema）

四层树 + 两类横切边：

```
L1 域  domain      KN 知识网络 / AR 分析推理 / GR 语法系统
L2 族  family      每域 2–4 族（即 Pattern Family Atlas v2 的 9 族落位）
L3 特征 feature    FTR-xx（E 级、anchors、counter_evidence、model_scope —— registry v1.28 字段原样引用）
L4 样本 sample     SAM-xx（token 级锚定证据，受控句 + peaks/lookback/focus 读数）

横切 M 机制 mechanism  MEC-xx（主轴三段位置 + 执行者 + E 级）
横切 X 边  cross_link  特征↔域间语义边（SxK/RxS/RxK 的家）
```

```jsonc
{
  "schema": "atlas_tree_v1",
  "derived_from": { "registry": "v1.28@<sha8>", "ledger": "n=374@<sha8>" },
  "domains": [{
    "id": "KN", "name": "知识网络",
    "desc": "实体/类/属性/关系的表征、读出与泛化边界",
    "families": [
      { "id": "KN.readout",   "name": "读出几何", "features": ["FTR-01","FTR-02","FTR-17","FTR-18"] },
      { "id": "KN.class_axis","name": "类轴方向", "features": ["FTR-04"] },
      { "id": "KN.oov_limit", "name": "谱外边界", "features": ["FTR-20","FTR-21","FTR-22","FTR-25","FTR-26"] }
    ]
  },
  {
    "id": "AR", "name": "分析推理",
    "desc": "逻辑/真值/推理方向的表征与可分性",
    "families": [
      { "id": "AR.logic", "name": "逻辑方向", "features": ["FTR-03"] }
    ]
  },
  {
    "id": "GR", "name": "语法系统",
    "desc": "句式/位置/上下文的编码几何",
    "families": [
      { "id": "GR.subspace", "name": "属性·语法子空间", "features": ["FTR-06"] },
      { "id": "GR.context",  "name": "上下文与位置",   "features": ["FTR-12","FTR-13","FTR-14","FTR-23"] }
    ]
  }],
  "cross_links": [
    { "edge": "X1", "type": "SxK", "features": ["FTR-05"], "domains": ["GR","KN"] },
    { "edge": "X2", "type": "RxS", "features": ["FTR-07"], "domains": ["AR","GR"] },
    { "edge": "X3", "type": "RxK", "features": ["FTR-08"], "domains": ["AR","KN"] }
  ],
  "mechanisms": [
    { "id": "MEC-01", "name": "attention 再分配执行消耗", "axis": "mid-layer", "features": ["FTR-09","FTR-10","FTR-11"], "evidence_level": "E3_causal_scoped" },
    { "id": "MEC-02", "name": "上下文二元门控",           "features": ["FTR-12"], "evidence_level": "E2_predictive" },
    { "id": "MEC-03", "name": "RoPE 纯相对编码（输出级）", "features": ["FTR-13"], "evidence_level": "E2_predictive" },
    { "id": "MEC-04", "name": "残差流全流分布式读出",      "features": ["FTR-01"], "evidence_level": "E2_predictive" },
    { "id": "MEC-05", "name": "读出-控制解耦（C_steer=0）","features": ["FTR-16"], "evidence_level": "E3_causal_scoped" },
    { "id": "MEC-06", "name": "k*/读出双轨判定",          "features": ["FTR-19"], "evidence_level": "E2_predictive" },
    { "id": "MEC-07", "name": "局部薄邻域等价边界",        "features": ["FTR-15"], "evidence_level": "E2_predictive" }
  ],
  "evidence_samples": [{
    "id": "SAM-001", "feature": "FTR-12", "template": "TM-xx",
    "sentence": "<受控句>", "tokens": ["..."],
    "readings": { "peaks": {"mlp":3,"res":4,"attn":3}, "attn_lookback": {"from":2,"to":0} },
    "evidence_note": "样本不独立评级，E 级继承 FTR-12"
  }],
  "coverage": {
    "KN": { "E2_plus": 9, "E1": 1 }, "AR": { "E2_plus": 2, "E1": 0 }, "GR": { "E2_plus": 5, "E1": 2 }
  }
}
```

**纪律继承**：coverage 只由派生器从 registry 算出，人手禁写；cross_link/mechanisms 变更走预注册臂；SAM-xx 只能引用已入册 FTR 的封存 npz 证据。

## 2. 26 特征归位表（映射的唯一事实源）

| 域 | 族 | 特征 | E 级 |
|----|----|------|------|
| KN | KN.readout | FTR-01（content 份额 57.6%）、FTR-02（K2 门）、FTR-17（E_read）、FTR-18（E_ar） | E2×4 |
| KN | KN.class_axis | FTR-04（类词方向配方） | E1 |
| KN | KN.oov_limit | FTR-20/21/22（谱外崩塌 2.54×、量化、端口校准）、FTR-25/26（共享方向 u*，E2_held_out） | E2×5 |
| AR | AR.logic | FTR-03（真/假 LOEO AUC 0.856+） | E2 |
| GR | GR.subspace | FTR-06（属性/语法词子空间可分） | E2 |
| GR | GR.context | FTR-12（前缀二元门）、FTR-13（RoPE 纯相对）、FTR-14（T_C 几何）、FTR-23（关系特异残差，E2_held_out）、FTR-24（跨关系偏移低秩） | E2×3+E1×2 |
| 交叉边 X | — | FTR-05（SxK）、FTR-07（RxS）、FTR-08（RxK） | E2×3 |
| 机制层 MEC | — | FTR-09/10/11（消耗链 E3）、FTR-15（商边界）、FTR-16（C_steer=0 E3）、FTR-19（双轨门） | E3×3+E2×3 |

## 3. 表现形式

### 3.1 客户端（研发平台 · 图谱透镜）

- 「图谱」新增第四视图**「域树」**：三列树（域 → 族 → 特征卡，复用现有特征卡/证据链组件），特征卡内嵌 **token 样本抽屉**（SAM-xx 展开：受控句上高亮 peaks/lookback）。
- 顶部**域覆盖度条**：每域 E2+ 密度（KN 9 / AR 2 / GR 5），失衡即缺口提示。
- **机制横切过滤器**：选 MEC-xx，全树高亮 encoded_by 特征（体现多对多，回击 H2）。
- 数据源：新端点 `GET /api/ai-rnd/atlas/tree`（server 从 registry v1.28 + ledger 派生；前端零实验字面量，数值全部来自派生 JSON）。顺带解决「客户端只有 v1.5–v1.8」的版本滞后挂账。

### 3.2 研究侧渲染

- atlas HTML 增域树分区（data-k 键扩展为域树语义路径，延续逐字段校验纪律）。

### 3.3 研究流程（五段式不变，选题改为缺口驱动）

1. 派生器每次 registry 升级后重算 atlas_tree_v1.json + coverage；
2. **域 × 族的 E0/E1 空格 → 候选臂池**：AR/GR 的空白族按「与已闭合机制的贴近度」排序预注册立项/升级臂；
3. cross_link 上的新特征优先（交叉处证据密度最低、信息量最高——FTR-08 全表唯一 mixed 带即例证）；
4. GAP 账本挂到域树节点（gap 归属域/族），关闭即覆盖度更新。

## 4. 落地路线

| 阶段 | 内容 | GPU |
|------|------|-----|
| A | `atlas_tree_v1.json` 派生器 + §2 归位表评审（26 特征逐条核对） | 零 |
| B | server `/api/ai-rnd/atlas/tree` 端点 + 客户端「域树」视图（含覆盖度条、机制过滤、样本抽屉） | 零 |
| C | SAM-xx 批量锚定（从封存 npz 提取 token 级样本） | 零（读封存） |
| D | 覆盖度缺口 → AR/GR 立项臂预注册 | 部分 GPU |

## 5. 风险与红线

- 三域命名不得暗示「机制专属」（H2）——文档与 UI 文案用「该域已确证机制」表述。
- token 样本永不升格为独立特征（H3）——新特征仍走 FTR 立项 + E 级门。
- 派生层与权威层冲突时，以 registry/ledger 为准；派生 sha 不匹配即拒绝渲染（LIVE/DEMO 纪律延伸）。
