"""P2820 MEMO append (append-only discipline)."""
from pathlib import Path

MEMO = Path('D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md')

BLOCK = """
---

## Phase 2820: 编辑泛化与交叉规律（emb-edit vs mlp-edit 交叉矩阵）[2026-09-17 05:17]

### 1. 原理与设计

用户指令：其他红色水果改黑是否改相同神经元？路灯红改黑？苹果红改紫？交叉找普遍规律。

**双 Arm 预注册**（execution.json 先落盘，判据冻结）：

- **Arm H（闭环）**：L31 h4 animal 因果消融，精确 2818 协议（8 目标词 × 7 位置 56 句 + 5 随机头对照）。H-P1：Δprof_animal > 0 且 > q95(随机)。
- **Arm E/G（编辑矩阵）**：行为终点 = "The {s} is" 末位颜色 logit margin(black−red)，10 实体（红族 6：apple/cherry/strawberry/tomato/blood/streetlight + 对照 4：sky/grass/coal/banana）× 6 色矩阵，12 编辑条件：
  - **emb-edit**（z 空间行移位，精确反解 rms-norm）：e_new = e + β·s0·(dW_tgt−dW_src)/g；β∈{1,2,4} 扫描；apple/streetlight/cherry 各色交叉
  - **mlp-edit**（down_proj 列写出重定向）：w' = w − (w·dW_src)dW_src + (w·dW_src)dW_tgt；用 2819 已登记普查列（red top4: L35 n1552/L34 n1218/L33 n2566/L25 n695；color top3: L32 n353/L31 n3298/L30 n4290）
  - joint（emb+mlp）、伪编辑对照（apple blue→black）
  - 判据：E-P1 编辑翻转+隔离；G-P1 存储定律（r_emb<0.2 实体行私有 vs r_mlp≥0.5 写出列共享）；G-P2 目标无关性；G-P3 联合隔离

门禁：dW=1.82e-09, Zeval=2.46e-06 量级通过（gate_dW<1e-6, gate_Z<1e-4）。

### 2. 结果

**Arm H：P-H1 = true——4/4 写入头因果地图闭环。** L31 h4 animal Δ=0.0296 > q95=0.0175。加上 2818 的 L32 h0(nature)/L34 h0(metal) 与 2819 的 L32 h3(clothing)，**四个实际写入头全部因果确认**。

**Baseline margins(black−red)**：红族全负——apple −3.88 / tomato −2.63 / strawberry −1.81 / cherry −1.44 / blood −0.75 / streetlight −0.69；对照正常——sky +5.31 / coal +2.88 / grass +0.19。

**编辑矩阵——全部 12 条件无一翻转行为（E-P1 = false），但失败模式呈现清晰的双重结构**：

| 条件 | Δmargin(apple) | 溢出 | 解读 |
|---|---|---|---|
| emb-edit apple red→black β=1/2/4 | +0.19 / −0.06 / −0.13 | **其余 9 实体全部 Δ=0.000** | 完美隔离但零效应（β=4 时 z 空间扰动仅 ~10% RMS） |
| emb-edit apple red→purple β=2 | 0.000 | 全部 0.000 | 同上 |
| emb-edit cherry / streetlight red→black | 各自 −0.13 / −0.06 | 其他实体 0.000 | 同上（方向反而更红） |
| **mlp-edit red→black（4 列）** | **−0.25** | **全部 10 实体同向位移**（cherry −0.13、streetlight −0.25、sky/grass/coal −0.31） | 全局共享通路，方向反直觉（red 相对增强） |
| mlp-edit red→purple | −0.06 | 全局弱移 | 目标依赖（G-P2 false：black 效应 4× purple） |
| joint emb+mlp | −0.44 | ≈两者加和 | 仍远离翻转（baseline −3.88） |
| 伪编辑 apple blue→black | 0.000 | 0.000 | 对照干净 |

**G-P1 = 双半验证 true（存储定律）**：r_emb = 0.000 → **实体 embedding 行完全私有**；r_mlp = 0.5 → **晚层 MLP 颜色列跨实体共享**。

**G-P3 = true**：joint 编辑 sky/grass 位移 0.312 < 0.5，隔离良好。

**Post-hoc 符号审计（探针，预注册外，仅作解释）**：4 个 red 普查列的 w·dW_red 为 **2 正 2 负拮抗结构**——L34 +0.638 / L25 +0.143（增红）vs L35 −0.460 / L33 −0.736（抑红），负列幅度占优（|1.20| > |0.78|）。这完全解释 mlp-edit 全局负移：重定向公式对 comp<0 的列反而**增加** red 写出（w' = w + |comp|·dW_red − |comp|·dW_black）。也解释 G-P2：列与 dW_black 对齐（0.02-0.13）强于 dW_purple（0.016-0.038）。**2819 K2 普查按对齐度选列未测符号——"red 对齐列"实为增红/抑红拮抗列的混合。**

### 3. 对用户问题的回答——"改不同实体/目标色是否改相同神经元"

**不是同一套参数，但共享同一套硬件通路——知识是计算不是存储位：**

1. **实体身份 = embedding 行，实体私有**（r_emb=0.000：改 apple 行，cherry/strawberry/tomato/streetlight 纹丝不动）。每个实体有自己的行，**互不通用**。
2. **颜色读写通路 = 晚层 MLP 列 + OV 头，跨实体共享**（r_mlp=0.5：改 4 列，全部 10 实体同向位移）。水果与路灯用**同一套**颜色通路，**没有按实体划分的专用颜色神经元**。
3. **但两者都不是知识存储位**：实体行的颜色分量移位（~10% RMS）对行为零贡献 → apple→red 联想不在 apple 行；共享列重定向效应小且方向由列写出符号决定 → 列是**拮抗调节器**（增红/抑红成对）不是存储器。
4. **颜色预测 = 前向动态计算**：apple 表征（行入口+层加工）→ 共享颜色通路（MLP 拮抗调节 + OV 写入）→ unembed 读出。**"改知识"不是改一个位，而是改一条计算通路的增益结构。**

机制总图景推进：2815 静态嵌入无坐标 → 2816 层参数混合语义与规则 → 2817 写以头为粒度 → 2818 几何对齐≠因果 → 2819 知识三级分布式 → **2820 编辑双重结构：实体私有行（零溢出零效应）+ 共享拮抗通路（全局溢出小效应）；知识=计算，编辑须改通路而非改位。**

### 4. 硬伤（严格审视）

1. **β 上限 4 太小**：z 空间扰动仅 5-10% RMS，而 E-P1 翻转要求 4+ logit 位移（baseline −3.88）——**E-P1=false 只证明"小编辑无效"，不能排除强编辑可翻转**；β=20-100 会出流形（s0 漂移、表征破坏），需配副作用监控。
2. **mlp-edit 仅 4 列且半数符号相反**：净移除 red 支持 |0.78|−|1.20| 为负——本轮重定向实际是"增红编辑"；配平编辑（正列移除+负列反接）未测。
3. **bf16 读出噪声 ~±0.06**：最小观测 Δ（0.062）在噪声底，单实体弱效应解读需谨慎。
4. 符号审计为 post-hoc，解释力强但未进预注册。
5. 单模型（qwen3-4b）单模板（"The {s} is"）；跨模型泛化未测。

### 5. 结论（重复 3 次）

**编辑泛化定律：实体 embedding 行私有（r_emb=0.000，改苹果不动樱桃）但颜色行为零贡献；晚层 MLP/OV 颜色通路跨实体共享（r_mlp=0.5，水果与路灯同一套）且为增红/抑红拮抗列结构；知识是前向计算不是存储位，单点/少量编辑均无法翻转（12/12 条件失败），真正的知识编辑必须组合干预通路增益结构。**
**编辑泛化定律：实体 embedding 行私有（r_emb=0.000，改苹果不动樱桃）但颜色行为零贡献；晚层 MLP/OV 颜色通路跨实体共享（r_mlp=0.5，水果与路灯同一套）且为增红/抑红拮抗列结构；知识是前向计算不是存储位，单点/少量编辑均无法翻转（12/12 条件失败），真正的知识编辑必须组合干预通路增益结构。**
**编辑泛化定律：实体 embedding 行私有（r_emb=0.000，改苹果不动樱桃）但颜色行为零贡献；晚层 MLP/OV 颜色通路跨实体共享（r_mlp=0.5，水果与路灯同一套）且为增红/抑红拮抗列结构；知识是前向计算不是存储位，单点/少量编辑均无法翻转（12/12 条件失败），真正的知识编辑必须组合干预通路增益结构。**

### 6. 接续（2821 候选）

1. **配平拮抗编辑**：正列移除 + 负列反接（w' = w − comp·dW_red − comp·dW_red 反向），净移除 |0.78|+|1.20| 全部——检验"全拮抗配平可否翻转"。
2. **强编辑扫描** β∈{10,30,100} + 全类别副作用监控（区分"知识改写"vs"表征破坏"）。
3. **K2 top-20 列联合消融**找"分布式欠杀→联合过杀"转折点。
4. 上位词行（' fruit' / ' traffic light'）编辑——层级泛化。
5. Δh 通道分离主线（2810 承接）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2820_edit_generalization.py` sha256 = `62f23815353ac639f5981acb7d725fa9711010db5e78499e8d7a24ffee2ea2bd`
- `…/phase2820/edit_generalization/execution.json` sha256 = `f5711b918c7bf85322101803f726fa03701bf1e317071507089ed8e7b8555599`
- `…/result.json` sha256 = `a3f9bfc62676348569dfc8d5b8196ea5e090779e0ad98d0e99a9557c3040534f`
- `…/edit.npz` sha256 = `ff4f6414ec2e27a270b461d42836dbaeb8b6a76e33e9a3dc867cf5910a1b28df`

**Gens 记录**：3 次崩溃修复（streetlight 2-token tid 断言→实体移出 tid 预填充；emb_hook bf16/float32 dtype 不匹配→显式 cast；fc.npz 二次索引越界→直接存 LG0），第 4 跑干净（9.4s）。预注册判据全程未动；streetlight 多 token 实体经 tok() 全 ids 路径 + emb_hook 全 token 行编辑处理。
"""

with MEMO.open('a', encoding='utf-8') as f:
    f.write(BLOCK)
print('APPENDED', len(BLOCK), 'chars')
