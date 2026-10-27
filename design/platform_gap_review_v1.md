# LLM 逆向分析平台 · 顶级对标审计 v1

- 审计日期：2026-10-27
- 审计对象：`/workbench`（研究工作台三透镜）、`server/`、`deploy/`、`research/gpt5/atlas` 账本、census/EXPL 数据面
- 方法声明：**本文每个数字都来自磁盘文件或端点实测**，附文件路径；标注「量级参考」的才是行业对照，不是实测。
- 结论先行：**这不是世界级"平台"，这是一个有顶级实验室纪律的垂直"机制图册"。** 差距不在严谨度（严谨度可能领先业界），而在四个平台属性：单位规模、因果深度、电路分辨率、第三方可验证性。

---

## 0 一句话判定

平台的科研纪律（预注册、冻结哈希、失败入账、held-out 门、证据分级）已经达到甚至超过多数公开可解释性团队；但**产品表面只暴露了已测资产的 0.007%**，且逆向分析的最终交付物（电路、干预闭环）在界面里完全缺席。

一句话：**资产是实验室级的，产品是报告级的。**

---

## 1 资产盘点（磁盘实测）

| 资产 | 实测规模 | 出处 | 是否出现在产品 |
|---|---|---|---|
| 自然语料普查 qwen3-4b | 19,042,014 token；1,801.5 tok/s；峰值 11.31 GiB；11,969.2 s；**36 层 × 9,728 单元 = 350,208 个 MLP 单元**，每单元含 32 条强激活样本（token 位置 + token id）、32 条弱激活、65 桶 × 4 象限直方图、readout 极性 | `D:\ai2050_native\_census\...\L*.npz`（36 个文件，1123.6 MB） | **否**。空间视图只有 29 个残差坐标点 |
| 解释检验 EXPL-v1（盲测代理：只给普查片段写的解释，让另一个看不到普查的代理写强激活句 + 近似反例） | 200 单元（36 层各 5 + 20 层各多 1），预注册 + sha256 | `research/interventions/EXPL-v1.{prereg,units,results}.json` | **否** |
| 特征注册表 | 26 条：E2_predictive 17 / E2_held_out 1 / E2_behavioral 2 / E1 3 / **E3 3**；3 主线模型 | `atlas_registry_v1_39.json` | 是（完整） |
| 失败账本 | **21 条**（F1–F21），含反例与否定范围 | 同上 `failures` | 部分（散在文本里，无数轴入口） |
| 测量账本 | **438 条** measurement，哈希链 | `atlas_ledger.json` | 否（无时间线 UI） |
| 语言树 | 3 模块 / 24 链接 / 3 跨模块链接 / **7 机制节点** | `atlas_tree_v2.json` | 是（文本级 + 3D 簇） |
| 注意力头静态分析 | qwen3-4b **1152 头** W_OV 相对强度；top 写入头聚层 23–30；两子空间共享中继头族；判定 **BLOCKED** | MEMO Phase 3280 | **否**（产品里只有 legacy `/head_details`） |
| 干预脚本 | patch / steer / ablate / selective-intervention 共 6+ 个 | `scripts/*intervention*.py` | **否**（界面无发起入口） |
| 任务级行为评测 | 100+ prompt-target 任务、多 token 打分、paired sign test + bootstrap CI | `scripts/task_level_causal_eval.py` | **否**（未进账本流水线） |

**资产侧最该被看见的一句话**：`350,208` 个单元已经带完整 profile 躺在盘上，而产品只展示了 `26` 条人工写下的结论。

---

## 2 产品表面（实测）

- 三个一级透镜：`route`（进展，2 子页签）、`spatial`（空间，5 视图：神经元 / 语言图谱 / 响应点云 / 谱图关系 / 本地包）、`process`（研发，7 工位，但远端客户端只剩「题库」1 个）。
- Atlas 相关端点共 9 个：`/atlas/registry`、`/atlas/feature/{fid}`、`/atlas/ledger`、`/atlas/gaps`、`/atlas/tree`、`/atlas/neuron3d/{model}`、`/atlas/neuron/{model}/{layer}/slice`、`/atlas/neuron/{model}/{layer}/{dim}`、`/atlas/fidelity`。
- 后端总量约 530 个端点、220 个 `scripts/*.py`、`App.jsx` 5204 行（已归档 `/legacy`）、10 个 `rdc-*` 专项页、8 条研究线目录。
- FastAPI 自述仍是 `title="TransformerLens 3D API"`（`server/server.py:246`）——对外身份没跟上平台现状。
- 界面自认缺口（原文，非我推断）：`RouteLens.jsx:78`「尚未覆盖：注意力头的逐头分析、跨层计算边（归因图）、多数单元的因果检验」；`LangPage.jsx:231`「路线 N 尚未实现」。

---

## 3 漏斗：资产 → 产品

```
350,208  已测 MLP 单元（36 层 × 9,728，每单元 top-32 激活样本）
     ↓  仅 0.057%
    200  做过盲测解释检验的单元（EXPL-v1）
     ↓  13%
     26  进入注册表的特征（人工撰写 statement + 锚点）
     ↓  11.5%
      3  达到 E3 因果分级
      0  有消融验证的电路
```

每一层损耗都不是"数据不够"，而是**没有把上一层的产物接进下一层的流水线**。这是本审计的核心判断。

---

## 4 八个问题（证据 → 后果 → 改法）

### 问题 1：产品只暴露 0.0074% 的已测单位，平台被做成了报告

- 证据：350,208 个单元有 profile（`L*.npz`，每层 `top_val/top_idx/top_v/bot_*` + `hist(9728,65,4)`）；产品只有 29 个 3D 点 + 26 条特征。
- 后果：访客无法"探索"，只能"阅读"。Neuronpedia 类平台的第一性体验是"任意单元可点开看它的激活句"，这里做不到——尽管数据就在盘上。
- 改法（零 GPU）：
  1. 新端点 `GET /api/v1/atlas/units/{model}/{layer}?offset&limit&sort=strong_rate`，直接从 npz 分页读 9,728 行（字段：unit、strong_rate、pos_rate、amax、象限份额、top 样本句 + 命中位置）。
  2. 新视图 `spatial.view=units`：层 × 单元网格（复用 `NeuronSpace` 的网格与 `NeuronUnitPanel`），点开单元 = 32 条真实激活句 + 65 桶直方图 + readout 极性。
  3. EXPL-v1 的 200 单元解释检验结果做成"可解释性"列（通过 / 未通过 / 未测），给出**平台自己的 auto-interp 分数**。

### 问题 2：因果深度不足，且账本自己否证了"读出几何=控制"

- 证据：26 条里 E2 家族 20 条（76.9%），E3 仅 3 条（FTR-10 MLP 无贡献 / FTR-11 无单点头 / FTR-16 读出≠控制）。FTR-16 记 `C_steer=0`，Wilson 上界 1.01%；F20 记 `GA_flip=0`、`cos(dR,Δw)=+0.0997/+0.0625`。
- 后果：非 E3 的 23 条（88.5%）目前是"能预测"，不是"能控制"。对外若被当作机制引用，F20 会当场反驳；这正是 Phase 3280 转向"消费方归因"的原因。
- 改法：
  1. **E3 变成注册门槛**：无干预锚（ablate / patch / steer 三选二）不得标 E3，现有 E2 条目保留但加 `causal_status: "untested" | "failed" | "passed"`。
  2. 对 17 条 E2 做**批量行为敲除筛查**（一次前向 × 每特征一个子空间置零，测任务级 log-prob 变化），用便宜的门决定谁值得升 E3。
  3. 特征卡固定显示三行：`因果状态 / 行为提升(CI) / 为什么不是 E3`。

### 问题 3：电路层为空——逆向分析的核心交付物缺席

- 证据：`RouteLens.jsx:78` 自认无逐头、无跨层边；3280 已算出 1152 头的 W_OV 与中继头族，但判定 BLOCKED 且未进产品；产品里的头部分析只有 legacy `HeadAnalysisPanel`（走 `/head_details`），与 26 条特征无关联。
- 后果：能说"attention 再分配执行了消耗"，不能说"头 (34,15)/(35,0) 承重、边 X→Y 必要、去掉它 KL 掉多少"。而后者才是"逆向分析"四个字的交付物。
- 改法：
  1. 定义边 schema：`{src: {layer, head|mlp, unit?}, sink: {...}, metric: "wov_rel"|"path_patch_delta", value, ci, model, evidence}`。
  2. `scripts/run_path_patch.py`：对 3 条 E3 特征做 path patching（干净/污染对 → 只恢复某条边 → 测恢复比例），输出边权 + CI。
  3. 归因图视图：节点=层×头/MLP，边宽=贡献，**必须带"消融 top-k 边后 KL 掉多少"的验证行**，否则不给图。

### 问题 4：行为接地仍是模板级（2190 条探针撑起全部结论）

- 证据：26 条特征的探针集是 73 实体 × 10 类别 × 3 模板 = 2190；`scripts/task_level_causal_eval.py`（100+ 任务、多 token、bootstrap CI）存在但不在账本流水线内；普查 `strong_rate_median` 从 L0 的 1.77e-4 到 L34 的 1.37e-1，横跨三个数量级。
- 后果：held-out 可能是 held-out **模板**，不是 held-out **分布**。F20 正是这类"模板结构不过渡到行为"的直接警报。另外 19M token 对 1e-6 量级的稀有单元只有约 19 次触发，稀有单元统计不稳。
- 改法：
  1. 行为电池 v2：任务级评测升为强制项，每条特征必须报 `behavior_lift`（含 95% CI、n、配对检验）。
  2. 自然语料 split：普查语料切 train/held-out，禁止用同一批句子的激活讲故事。
  3. 稀有单元标注 `n_activations`，<100 次的单元在 UI 上直接标"统计不足"。

### 问题 5：模型矩阵窄，且跨模型结论受 tokenizer 混淆

- 证据：`models_mainline = [qwen3-4b, qwen3-14b, glm4-9b]`，`models_pending = [ds7b, gemma4]`——2 个模型同族。APPX-2 记录 tokenizer-face 分歧：syntax 轴存活 `3/1/0/1/4`，"从不池化"。
- 后果：`fpmin=0.996`、`S 族 14/14 可分离` 这类"跨模型守恒"结论，外部会质疑只有 3 个模型、2 个共享 tokenizer；尺度只有 4b→14b 一跳。
- 改法：
  1. 加 **Pythia-410m / 1.4b / 2.8b**（同 tokenizer 尺度梯）→ 干净的形成曲线；加 **Llama-3.2-1B**、**Gemma-2-2b**（Gemma Scope 有公开 SAE，可做单位对照）。
  2. 先修 **tokenizer 对齐层**（byte 级投影），再允许写跨模型结论；不对齐的模型只做族内比较。

### 问题 6：可移植性不足——严谨性外部不可见

- 证据：FastAPI 标题仍是 `TransformerLens 3D API`；9 个 atlas 端点全是内部形态（`/{model}/{layer}/{dim}`），无版本前缀、无 SDK、无 verify CLI；数据分发靠浏览器打开 zip（local pack）；账本哈希链只有内部能读。
- 后果：第三方无法在 30 分钟内复现任何一条 FTR；也无法引用。**一个不可复现的账本，外部只能当博客看。**
- 改法：
  1. `/api/v1/atlas/*` 版本化 + OpenAPI tag + ETag + 速率限制；保留旧路径一年。
  2. `pip install ai2050-atlas`：`atlas get FTR-01` / `atlas verify FTR-01 --model qwen3-4b`（重算锚点值，diff 即非零退出）。
  3. 每个 registry 版本打 Zenodo DOI（v1.39 → v1.40 起），feature 卡带可引用 ID。

### 问题 7：统计区间缺失，多重比较未控

- 证据：现役 G 线无 bootstrap（仅 legacy 脚本有），26 条特征的 `values` 全是点估计；Wilson 只在 FTR-16 用过一次；26 特征 × 3 模型 × 多指标没有 FDR。
- 后果：`C=57.6% / S=19.4% / R=15.9%` 这类份额没有区间，读者无法判断稳定性；批量筛查时假阳性会随条目数增长。
- 改法：registry schema v2 增加 `{n, ci_low, ci_high, test, p, p_adj}`；分层 BH-FDR；UI 所有份额带误差条。

### 问题 8：产品无实验闭环（"研究工作台"目前只是查看器）

- 证据：界面能看不能做——无"从特征卡发起干预"，无参数扫描，无结果回写；3239 的 DAS 交换臂已预注册未实现；`LensSpatial` 仍有演示种子；`ProgressPage.jsx` 已是死文件。
- 后果：研究者必须离开 UI 去写脚本，跑完再把结论贴回账本 → 界面天然滞后于研究，且无法承接外部贡献者。
- 改法：干预 DSL（`swap(subspace, L, src, dst)` / `ablate(head)` / `steer(dir, α)`）+ 复用现有 dispatch/node runner 执行 + 结果自动落账本（带 freeze/sha8）；特征卡加"发起实验"。

---

## 5 必须补的六项能力（按 ROI 排序）

| # | 能力 | 为什么必须先做 | 依赖 | 增量成本 |
|---|---|---|---|---|
| 1 | **单元浏览器**（Unit Browser） | 数据已在盘，唯一缺的是接口与视图；一次把 350k 资产变成可探索对象 | census npz + EXPL-v1 | 零 GPU，约 1–2 周 |
| 2 | **干预运行时**（Intervention DSL + runner） | 把 E2 升 E3 的唯一通道；也是外部贡献者的入口 | dispatch/node 已有 | 轻 GPU |
| 3 | **电路图**（边 schema + path patching + 验证） | "逆向分析"的交付物；补上自认的最大空缺 | 1 条 E3 特征即可试点 | 轻 GPU |
| 4 | **行为电池**（任务级 + 自然分布 + CI） | 决定结论是真机制还是模板拟合 | 已有 `task_level_causal_eval.py` | 轻 GPU |
| 5 | **可复现包 + 公共 API** | 决定账本能否被引用、平台能否被信任 | 无 | 零 GPU |
| 6 | **SAE / 转码器对照臂** | 不是替换单位，是**检验你自己的论点**（多义是观察者产物） | 4b 2–3 层即可 | 中 GPU |

第 6 项单独强调：现役 G 线的注册单位里**没有任何 SAE/转码器**（仓库里 SAE 代码都在历史 AGI 线与 gemini 线，未接入账本）。而平台的立场恰恰是"多义性是观察者问题"——这个立场只有把"原生神经元 / SAE 特征 / 子空间轴"在同一批 2190 探针上对齐，才算被证明。**不做对照臂，论点永远只是立场。**

---

## 6 路线（含验收标准）

### P0 · 让平台"可探索、可引用"（2–3 周，零 GPU）

1. 单元浏览器：`/api/v1/atlas/units/{model}/{layer}` + `spatial.view=units`。
   - 验收：任取 5 个随机单元，能在 2 次点击内看到 top-32 激活句与直方图；首屏 < 800 ms（分页 200 行）。
2. registry schema v2：`ci / n / test / p_adj / unit_ref / behavior / causal_status`。
   - 验收：26 条全部有字段（缺值显式 `null` + `needs_backfill`，不许编造）；`--check` 幂等。
3. `/api/v1/atlas/*` + OpenAPI 文档页 + `atlas verify` CLI。
   - 验收：在干净环境 `pip install` 后，一条命令复现 FTR-01 的 `C/S/R` 份额与 `fpmin`，误差 < 1e-6。
4. 特征卡加"证据阶梯 + 为什么不是 E3 + 因果状态"。
   - 验收：26 条里 23 条非 E3 的卡片都能说明缺哪一门。

### P1 · 让平台"能干预、有电路"（4–8 周，轻 GPU）

5. 行为电池 v2 接入账本流水线（任务级 + 自然分布 + CI）。
6. 逐头表：1152 头 × {W_OV 相对强度, 子空间投影, 消融 ΔKL, top 激活句}，替换 legacy 头面板。
7. 对 3 条 E3 特征做 path patching，产边表 + 归因图 + 消融验证行。
8. 干预 DSL + runner，结果自动落账本。
   - 验收：从界面发起一次 steer，账本新增 1 条 measurement（含 freeze/sha8/CI），无需人工改文件。

### P2 · 让平台"能被别人用"（2–3 月，中/重 GPU）

9. SAE/转码器对照臂（4b，2–3 层）+ 三方对齐（原生神经元 ↔ SAE 特征 ↔ 子空间轴）。
10. 模型矩阵扩到 ≥8 模型 / ≥3 族 / ≥3 尺度点；tokenizer 对齐层先行。
11. 协作：特征评审态（proposed / reviewed / verified）、评论、外部模型沙盒、每版本 DOI。
    - 验收：一个外部研究者不用读代码，30 分钟内提交一条带干预锚的新特征。

---

## 7 顶级判据对照

| 维度 | 现在（实测） | 顶级阈值 | 差距性质 |
|---|---|---|---|
| 可浏览单元 | 26（背后 350,208 有 profile） | ≥10^4 可浏览 | **接口问题，非数据问题** |
| E3 占比 | 3/26 = 11.5% | ≥50% | 流水线缺口 |
| 已验证电路 | 0 | ≥3 条（消融验证） | 能力缺口 |
| 每特征行为提升（带 CI） | 0 | 100% | 统计 + 流程缺口 |
| 第三方 30 分钟复现 | 不可 | 1 条命令 | 打包缺口 |
| 模型 / 族 / 尺度点 | 3 / 2 / 1 跳（4b→14b） | ≥8 / ≥3 / ≥3 | 算力 + 工程缺口 |
| 失败入账率 | 21 失败 / 47 条目 = 44.7% | ≥30%（业界罕见） | **领先项** |
| 预注册 + 冻结哈希 | phase 编号至 3280、438 条测量、哈希链 | 少数团队做到 | **领先项** |

最后两行是本平台真正的护城河：**别人做不出这份"知道自己错在哪"的账本**。世界级平台不是把这两行丢掉去追规模，而是让它们变成别人能验证的公共资产。

---

## 8 四件不要做的事

1. **不要为了规模而规模。** 追百万级自动标注特征，会让 F 账本失去含义——你换来的是一个特征动物园，丢掉的是"每条结论都有反例与否定范围"。
2. **不要把 SAE 当成"正确单位"。** 你的论点是相反的；SAE 只能当**对照臂**，不能当替换。
3. **不要把模板通过率包装成机制。** F20 已经证明这条路会被自己的数据打回。
4. **不要让 legacy 继续长。** 约 530 端点 / 220 脚本 / 5204 行 App.jsx 的历史包袱，会把"平台"稀释成"资料堆"。冻结 `/legacy`，`rdc-*` 专项页逐条并轨后删行。

---

## 9 附录：核查命令（任何人可复跑）

```bash
# 注册表规模与证据分布
python -c "import json;d=json.load(open('tests/glm5/result/rdc_query_construction_20260913/phase3279/g5h42_appx1_landing/atlas_registry_v1_39.json'));print(len(d['features']),len(d['failures']),d['models_mainline'],d['models_pending'])"

# 普查规模
python -c "import json;d=json.load(open('research/neurons/census-qwen3-4b-2ca5f0a756d36c8a-20M.json'));print(d['n_tokens_counted'],d['gpu_peak_gib'],d['seconds_total'])"

# 单层单元 profile 字段
python -c "import numpy as np;z=np.load(r'D:\ai2050_native\_census\6be1034f0b55a6df-bf16\2ca5f0a756d36c8a-20000000\L16.npz');print(z.files);print(z['hist'].shape)"

# 失败账本条目
grep -c '"id": "F' tests/glm5/result/rdc_query_construction_20260913/phase3279/g5h42_appx1_landing/atlas_registry_v1_39.json
```
