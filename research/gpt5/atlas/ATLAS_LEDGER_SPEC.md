# TMA Atlas Ledger 数据格式规范 v2

> Token-Mechanism Atlas 双图谱积累格式。目标：把语言族谱（词坐标）、LLM 响应图谱
> （机制轴）与关联机制证据沉淀为**可累积、可校验、可合并**的单一事实源。
> 原则继承自 LPF v5.3 纪律：产物 immutable、SHA256 登记、判据与 null 随测量绑定。
>
> **v2 变更（phase 2882，向后兼容 v1）**：新增 §10 四象限表（通道分化）、
> §11 跨族迁移矩阵、§12 阴性结果登记、§13 model namespace（多模型复制预留）
> 与 migration_history（格式迁移日志）。v1 文件由 v2 loader 原样加载
> （缺省节视为空）。

## 1. 文件布局

```
research/gpt5/atlas/
  ATLAS_LEDGER_SPEC.md     # 本规范（格式版本随主版本号递增）
  atlas_ledger.json        # 唯一主索引（append-only，人工/AI 编辑）
  blocks/                  # （可选）跨 Phase 汇总的派生 npz，一律带 sha256
```

原始测量产物**不移动**：ledger 只存指向
`tests/glm5/result/rdc_query_construction_20260913/phaseNNNN/<name>/` 的
相对引用 + SHA256 前缀，保证单一大文件不复制、不失效。

## 2. 顶层结构

```json
{
  "format": "tma-atlas-ledger",
  "version": 2,
  "model": "qwen3-4b",
  "protocol_family": "LPF v5.3",
  "axes":        [],   // §3 语言族谱：词族轴
  "blocks":      [],   // §4 响应图谱：坐标块
  "headsets":    [],   // §5 头集合（机制组件清单）
  "measurements":[],   // §6 判据记录（含 null 与判决）
  "growth_curve":[],   // §7 机制基增长率曲线
  "linkage":     [],   // §8 关联机制证据
  "quadrants":   [],   // §10 v2 通道分化四象限表
  "transfer":    {},   // §11 v2 跨族迁移矩阵
  "negatives":   [],   // §12 v2 阴性结果登记
  "model_namespace": {},  // §13 v2 多模型复制预留
  "migration_history": [] // §13 v2 格式迁移日志
}
```

## 3. axes —— 语言族谱（词坐标系）

| 字段 | 类型 | 说明 |
|---|---|---|
| axis_id | str | 如 `class`, `attr:size`, `syntax:pos`, `task:translate` |
| kind | str | `taxonomic` / `attributional` / `syntactic` / `task` |
| words | [str] | 词位清单（single_tok 过滤后） |
| labels | [int] | 与 words 对齐的轴内标签 |
| source | {phase, path, sha256_8} | 词表产物的 execution/result 引用 |
| status | str | `active` / `quarantined`（如 2858 G2 未过的 E200 线） |

**规则**：词表变动（扩容/清洗）不覆盖旧 axis，新增 `axis_id@rev2` 并在
`supersedes` 字段指向旧版（2857 subset_incompatibility 教训）。

## 4. blocks —— 响应图谱（机制坐标块）

| 字段 | 类型 | 说明 |
|---|---|---|
| block_id | str | 如 `B2_causal`, `B3_mlp`, `B_attr_causal` |
| axis_id | str | 所属词族轴 |
| kind | str | `unembed_proj` / `causal_spectrum` / `mlp_response` / `ov_static` / `eta2` / `share` |
| shape | [int] | [n_words, n_dims] |
| heads | str/null | 若为头子集，指向 headsets.set_id |
| path | str | npz 相对路径 |
| key | str | npz 内键名 |
| sha256_8 | str | npz 文件 SHA256 前 8 位 |
| phase | int | 产生 Phase |
| notes | str | 已知缺陷注记（如"B1 含构造性循环成分"） |

**规则**：同 kind 重测（新词表/新协议）→ 新 block_id 追加，禁止覆盖。

## 5. headsets —— 机制组件清单

```json
{ "set_id": "class43", "definition": "F-test p<0.01 per-word eta2, 80 words",
  "n": 43, "path": ".../growth_v2.npz", "key": "sig_mask",
  "sha256_8": "...", "phase": 2868 }
```

**规则**：每个集合必须带**定义判据**（含阈值与方向），无判据的"top-N"不收录
（2859 排行榜禁令）。

## 6. measurements —— 判据记录

```json
{ "meas_id": "M2869_H1", "type": "fusion_gain",
  "inputs": ["B3_mlp", "B2s_class43"],
  "stats": {"alpha": 0.25, "acc": 0.8875, "null_p95": 0.175},
  "verdict": "fusion_gain=true",
  "prereg": "max-alpha-null corrected",
  "source": {"phase": 2869, "path": ".../fusion_curve/result.json",
             "sha256_8": "..."},
  "errata": [] }
```

**规则**：`verdict` 原样抄录 result.json；后续 Phase 推翻判决时不删除，追加
`errata: [{"by_phase": N, "note": "..."}]`（如 2860 对 2853/2855 的勘误链）。

## 7. growth_curve —— 机制基增长率曲线（总判据）

```json
{ "point_id": "G1_axis1_class", "axis_id": "class",
  "components": 43, "acc": 0.425, "block": "B2s_class43",
  "shared_with_prev": null, "new": 43, "phase": 2868 }
```

**判读**：每新增一个 axis/block，记 components 与 new。曲线次线性 → 组合编码
（"有限参数→无限能力"的机制级解释）；线性 → 功能独立电路。当前实测点见
`atlas_ledger.json`。

## 8. linkage —— 关联机制（两图谱之间的边）

```json
{ "link_id": "L1_l13h30_mlp_indirect",
  "from": {"headset": "top64_causal", "head": "L13H30"},
  "to": {"block": "B3_mlp", "kind": "mlp_response"},
  "evidence": "G_mlp/G_attn=4.7; e_ff=-0.23 (~30x OV static 0.0075)",
  "phase": 2865, "status": "confirmed" }
```

## 9. 校验与合并

- `rdc_atlas_ledger.py`：`load()`（校验 schema + SHA 重算）/
  `add_block()` / `add_measurement()` / `growth_table()`。
- SHA 不匹配 → 该条目标记 `stale: true` 并拒绝参与统计。
- 多机/多会话合并 = JSON 深合并，冲突以 phase 大者为准并自动登记 errata。

## 10. quadrants —— 通道分化四象限表（v2）

每个词族轴一行，登记"drop 谱组织 × mlp 载体"两通道观测。**每格必须引用
`measurements` 中已存在的 meas_id**（可导出性检查 V2 的对象），禁止无引用断言。

```json
{ "family": "class",
  "drop_spectrum": {"organized": true, "frontedge_heads": 43,
    "sources": ["M2859_stability"]},
  "mlp_carrier": {"acc": 0.875, "sources": ["M2867_word_coords_v1"]},
  "quadrant": "organized+strong_mlp" }
```

**规则**：新轴族入表须同时具备两通道测量（或显式 `not_measured`）；判决变化
追加新行（family@rev），不覆盖旧行。

## 11. transfer —— 跨族迁移矩阵（v2）

三族联合谱（2881）导出的方向可迁移性快照：

```json
{ "source": {"meas_id": "M2881_joint_coords", "phase": 2881},
  "matrix_rows=fa_dirs_cols=fb_words": {"class": {"class": 0.9625, ...}},
  "family_centroid_cos": [[1.0, -0.6475, -0.3283], ...],
  "interpretation": "all cells > 0.76; family centroids mutually negative" }
```

**规则**：descriptive 快照，无门线判决；重测追加 `transfer@rev2`，不覆盖。

## 12. negatives —— 阴性结果登记（v2）

真阴性/隔离结果专位，防止"阴性结果蒸发"（2878 教训制度化）：

```json
{ "neg_id": "N1_translation_axis_absent",
  "kind": "real_negative",        // real_negative / quarantine
  "claim": "no unified cross-lingual axis direction in tied unembed",
  "evidence": "VG2b pair-pair cos -0.0007/-0.0029/0.0011",
  "sources": ["M2878_syntax_trans_vocab"],
  "status": "settled",            // settled（已定案）/ reopenable（可复议）
  "reopen_condition": null }      // quarantine 须写明复议条件
```

**规则**：`quarantine` 条目必须带 `reopen_condition`（如 tense VG2b 0.1902
差 0.01 → "扩池至 ≥8 对后重测"）；阴性升阳性时保留原条目、追加 errata。

## 13. model_namespace / migration_history（v2）

多模型复制战役（DS7B/GLM4/Gemma4）预留：

```json
{ "model_namespace": {
    "primary": "qwen3-4b",
    "protocol_family": "LPF v5.3",
    "pending_replication": ["ds7b", "glm4", "gemma4"],
    "rule": "跨模型条目复制时 model 字段为必填；同 model_id 内 id 唯一，
             跨 model 允许同 id 并存于 ledgers/<model>/ 子文件" } }

{ "migration_history": [
    { "phase": 2882, "from_version": 1, "to_version": 2,
      "backup": "atlas_ledger_v1_backup.json", "backup_sha256_8": "..." } ]}
```

**规则**：格式迁移必须先落 v1 备份并登记其 SHA256-8；主文件仅升 version 与
追加节，既有条目**逐字节不动**（V1 保真检查的对象）。
