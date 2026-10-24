# 神经元 3D 空间方案 · 点击神经元看特征（neuron3d v1）

> 2026-10-18。目标：在客户端 3D 空间中按"神经元"组织场景，点击任意神经元即查看其特征证据链——
> 方法论对齐 Cammarata《Neural Networks Are Already Clean》(2026-10-15) 的分析方式：
> 极点分析、激活样例、上下游一致性（"nothing reads" 判据）、steering 验证、弱/强激活分离、盲最大化槽位。
> 与既有方案的关系：`atlas_space3d_scheme_v1`（特征中心，26 FTR 主角点）管"结构总览"；
> 本方案（神经元中心，全坐标点云）管"下钻与逐点证据"。二者经 `?atlasSel=` 共用选中态互跳。

---

## 0. 方法论 → 界面映射表（文章技巧如何落成面板）

| 文章分析技巧 | 3D 空间落点 | 数据来源（现状） |
|---|---|---|
| 极点分析（正/负极分开解释） | 详情卡·极点页签：正极/负极分列 top-k 激活样例与 unembedding 读出 | 封存 npz（全层原生精度）+ 封存特征 |
| 激活样例挖掘 + 表面噪声解码 | 详情卡·样例页签：top/bottom 激活 token 上下文，附解码注记槽 | 同上 |
| 上游-下游一致性（"nothing reads" 判据） | 详情卡·电路页签：上游构成边 + 下游读出边，双列表对照；"下游无读出"显式标记 DECOUPLED | GAP-4 机制链产物（3188–3192） |
| Steering 验证 | 详情卡·干预页签：预留 steering/消融结果槽（只读展示已入册结果，不在界面内执行） | E3 干预线（3225–3232） |
| 弱/强激活分离检验 | 详情卡·统计页签：强弱激活份额图（删最弱 75% / 最强 7% 的对照口径） | 封存 npz 派生 |
| 盲最大化测试 | 详情卡·合成检验槽位（v2 预留）：由解释生成合成例句的激活验证结果 | 未实施，留插槽 |
| 递归泛化（分支生长） | v2 预留：checkpoint 生长轨迹动画（当前无 checkpoint 采集，不承诺） | 无，标注"未采集" |
| 跨模型对比审计 | 同一 (layer, dim) 三模型并排 + 垂直连线（复用 space3d 模型层交互） | registry model_scope + 三模型 npz |
| 证据分级（可读 ≠ 被使用） | 颜色双模式：E 级着色 / 标签来源着色（"读出标签"灰、"因果入册"绿）——强制可视化"能解释"与"控得住"的分离（F20 纪律） | registry evidence_level + F 账本 |

**红线**：界面展示的一切"特征标签"必须回链 registry/ledger 条目（phase + sha8）；
无 registry 依据的神经元一律灰色"未入册"，界面不得生成自然语言标签——标签生产走三引擎闭环 + 三门准入，不由前端编造。

## 1. 场景定义：坐标即结构

```
X = d_model 坐标索引（qwen3-4b: 0–2559，原生坐标顺序，不重排）
Y = 层号（hs[0] 嵌入层 → 末层；等距楼层，同 llm-neuron-atlas 的"层=楼层"隐喻）
Z = 坐标索引的次轴（X/Z 合围成层平面网格，每层 2560 点）
```

- 每层一个"楼层"：qwen3-4b 全场 ≈ 36 层 × 2560 ≈ 92k 点，单 InstancedMesh 无压力（3d_visualization_paradigms M1 口径 ≤50 万点）；
- 点颜色 = 当前样本激活值（原生精度，双视图：原始值 / 行归一化，切换按钮，不默认归一化）；
- 点大小 = 静态：参数范数分位（默认关闭，避免与激活通道双编码冲突）；
- 26 FTR 主角点 = 白心高亮环 + 到其载体的连线（与特征中心视角同一 sel）；
- 已知机制载体（CARRIER 池化维度、k*=3 锚点）= 常显标记层（可关）。

**诚实标注**：X/Z 是原生坐标索引，非语义几何；界面固定注记"位置=坐标索引，非模型真实几何"。

## 2. 场景三层结构（沿用 space3d 分层纪律）

```
L3 语境层：层楼板 + 层号标签 + 参考轴 + FTR 主角环
L2 关系层：选中神经元的上游/下游归因边（只显示选中者的边，避免全场边爆炸）
L1 几何层：全坐标点云（激活着色）+ 未入册灰底
```

## 3. 交互设计

### 3.1 相机与导航
- 复用 CloudMode 相机（拖拽旋转/滚轮缩放/惯性）；相机预设：全景俯瞰 / 单层聚焦 / 沿层穿越（连续 zoom：全场 → 单层 → 单坐标带，多尺度连续缩放范式）；
- 搜索框：直接输入 `L22.D758` 式坐标或 FTR-xx，相机飞至目标（借鉴 llm-neuron-atlas）。

### 3.2 点击神经元 → 详情卡（核心交互）
右侧滑出详情卡，五页签（v1 实装前三页签，后两页签按数据到位解锁）：

| 页签 | 内容 | 数据管线 |
|---|---|---|
| ① 概览 | (model, layer, dim) 身份、激活分布直方图（当前样本+封存样本池）、是否入册、E 级 | /atlas/neuron 查询 + npz 派生 |
| ② 极点 | 正极 top-10 激活（样例上下文 + unembedding 读出）；负极同列；正负分离，不合并解释 | 封存 npz + 封存特征 |
| ③ 电路 | 上游构成（top-k |weight| 入边）+ 下游读出（哪些维度/logit 读它）；双列表 + "下游无读出"徽标 | registry 电路条目 / 归因产物 |
| ④ 干预 | 已入册 steering/消融结果只读展示（A1 行为带、三门结论）；无则显示"未做因果检验" | E3 线产物 |
| ⑤ 跨模型 | 三模型同 (layer,dim) 并排激活曲线 + 是否同特征锚 | 三模型 npz |

### 3.3 与既有界面贯通
- `?atlasSel=` 三处一体：特征中心 3D / 本神经元 3D / 特征详情页共用选中态；
- 特征中心视角点击 FTR → 飞至其载体神经元并自动打开详情卡①；
- 神经元详情卡点"所属特征" → 跳特征中心视角。

## 4. 数据协议：neuron3d.v1（派生器新增段）

```json
"neuron3d": {
  "version": 1,
  "model": "qwen3-4b",
  "grid": {"n_layers": 36, "d_model": 2560, "origin": "native_coord_index"},
  "points": {
    "registered": [{"layer": 22, "dim": 758, "role": "carrier", "ftr": ["FTR-23"],
                     "e_level": "E2_behavioral", "circuit_up": 3, "circuit_down": 0,
                     "downstream_read": false}],
    "note": "registered 为白名单；其余点由前端从 npz 切片渲染，不进 JSON"
  },
  "edges": {"carrier_links": [["FTR-23", 22, 758]]},
  "labels": {"policy": "registry_only", "forbidden": "frontend_generated"}
}
```

派生纪律（延续 space3d 红线）：
1. registered 白名单、边表、极点摘要全部由派生器从 registry/npz 计算，`--check` 幂等，人手禁写；
2. npz 切片走服务端按层取块接口（M2 预演：单样本全层直载可 M1）；
3. 前端只认 `(model, layer, dim, activation)` 统一 schema（experiment-agnostic），换模型换加载器不换界面。

## 5. 服务端增量

| 端点 | 内容 | 量级 |
|---|---|---|
| `GET /atlas/neuron3d/{model}` | neuron3d.v1 派生段（白名单+边表） | 小，派生器一次 |
| `GET /atlas/neuron/{model}/{layer}/{dim}` | 详情卡数据：激活统计、极点摘要、电路边、入册链接 | 中，需 npz 查询函数复用 native_*_parameter_query 模式 |
| `GET /atlas/neuron/{model}/{layer}/slice?sample=` | 单层激活切片（前端渲染主数据） | 中，npz mmap 直读 |

## 6. 落地拆步

| 步 | 内容 | 量级 | 前置 |
|---|---|---|---|
| A | 派生器 neuron3d 段（白名单：26 FTR 载体 + CARRIER 锚 + k*=3 模型锚；边表） | 零 GPU | registry v1.32 已备 |
| B | `NeuronSpace.jsx`：全坐标点云场景 + 相机 + 层楼板（复用 CloudMode 相机与 space3d 图层板） | 中 | A |
| C | 详情卡①②③（概览/极点/电路）+ `/atlas/neuron` 查询端点 | 中 | B + npz 查询函数 |
| D | sel 贯通（特征中心 ↔ 神经元中心互跳）+ 跨模型页签④ | 小 | C |
| E | 干预页签⑤（E3 结果只读）+ 合成检验槽位 + checkpoint 轨迹插槽 | 小 | 各自研究线产出 |

建议本轮 A+B+C；D/E 挂账。UI 冻结红线：本方案为新增视图，不触碰 UI 线 v1.10 既有组件语义。

## 7. 边界与风险

- **标签纪律是生命线**：文章最大的反面教训是"缺口的解释"（给每个神经元硬编故事）。本方案前端零标签生成，全部 registry_only；未入册点只显示数字证据，不显示文字解释。
- **规模**：qwen3-14b 单样本全层 ≈ 40×5120 ≈ 20 万点，M1 可载；多样本对比必须走 M2 分块（zarr/neuroglancer precomputed），v1 不承诺。
- **归因边数据现状**：GAP-4 机制链已有电路级结论，但"每神经元上下游 top-k"需要一次离线烘焙（复用 llm-neuron-atlas 的 `down_proj @ gate_proj` 有效传递烘焙思路，纯 CPU 可行）；烘焙脚本归 tests/gpt5，产物哈希入 ledger。
- **与 Neuronpedia 的关系**：Neuronpedia（本地已部署）是 SAE 特征视角；本方案是原生坐标视角，经"特征↔坐标"映射表互补，不重复造轮子。
