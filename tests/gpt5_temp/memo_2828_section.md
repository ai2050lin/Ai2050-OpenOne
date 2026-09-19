## Phase 2828 = Phase Beta: 知识链 key 方向链式传递 [2026-09-17 08:06]

### 1. 原理与设计

用户三阶段计划第二步。核心问题："苹果→水果→食物"多跳链中 key 方向如何逐跳传递？

范式：双句链 prompt（token 全对齐，seq_len=12）：`The <E1> is a fruit. The fruit is a food.`，位置 e1=1 / fruit1=4 / fruit2=7 / food=10。8 变体：真链起点 apple/cherry/lemon/banana，断链起点 rock/steel/hammer/ocean。链方向 = 2806 类方向 dW_fruit、dW_food（gate 复现通过：dW=1.82e-9）。

谱：c[L, v, pos, h, 2]（36×8×4×32×2）。臂：
- E1 链式编辑：rank-1 消掉 pos=1 处 c_fruit top-5 头的 fruit 分量（key=apple 行），重测 pos=7 的 c_food（跟随测试）
- E3 修复注入：断链行 pos=1 残差流注入 β·dW_fruit（β∈{2,5,10}），测 c_food 恢复

预注册：B1 跳1真实（真链 c_fruit(e1) > null q95 且 > 断链）；B2 第二跳跟随（真链 c_food(fruit2) > 断链）；B3 编辑跟随（E1 使真链 c_food(fruit2) 降 ≥20%）；B4 修复（存在 β 使断链恢复至真链 50%）。

### 2. 结果（Gen4 成功，61.7s；Gen1 缺 d_model 提取、Gen2/3 einsum 下标与位置索引错误）

**判决：B1=true, B2=true, B3=false, B4=true（3/4）**。

- B1：真链 c_fruit(e1)=0.131 vs 断链 0.053（2.5 倍），超 null q95=0.079 —— 第一跳"实体→类别"写出真实且实体依赖。
- B2：真链 c_food(fruit2)=0.124 vs 断链 0.106（1.17 倍）—— 通过但余量小。
- **B3=false（核心负结果）**：E1 消掉跳 1 的 fruit 写出后，c_food(fruit2) 仅降 **0.2%**（0.124→0.123）——第二跳**不因果依赖**第一跳的 rank-1 写出分量。
- B4：注入 β=5 即把断链 c_food 推到 0.124 = 真链水平；但基线差距本来就小（0.106→0.124），修复幅度有限。
- 辅助谱：c_fruit(fruit2) 真链 0.289 vs 断链 0.185（1.56 倍）——第二句 fruit 的自我写出受上文语义一致性影响最强。

### 3. 分析

1. **E1 头全部在 L0**（h27 c=2.51、h15/23/0/19）——c_fruit 跨层求和的 top-5 全是 embedding 层"词法头"。这些头做低层信息搬运，不承载语义跳；编辑它们不影响下游属预期。
2. **谱级链存在（B1/B2）但因果链弱（B3）**：真链与断链的谱差异是"全句语义一致性"的弥散效应，不是 key 方向逐跳传递。**相关 ≠ 因果**在本范式被干净分离——B2 的 1.17 倍相关差距在 rank-1 消除后几乎不动（0.2%）。
3. 结论：**该句对范式下 LLM 不做链式 key 传递——两句并行独立处理**。"苹果→水果→食物"的知识链在简单陈述句中不表现为逐跳 key 通路；rank-1 类方向谱可测的因果传递不存在。
4. B4 "修复"实为弱效应：注入只覆盖 B2 的小差距，不构成强链证据。

### 4. 硬伤

1. 范式可能不适配：陈述句对是弱链；真正的多跳推理（问答链 "apple is a fruit. What is a fruit?"）或需 chain-of-thought 才激活传递。B3 的否证限于本范式。
2. E1 只消 top-5 头的 rank-1 分量；传递若走高阶/非线性通道则不可见。
3. 链方向用类别均值方向（dW_fruit/food），若传递走实例级方向（apple 的 fruit ≠ lemon 的 fruit）则被平均掉。
4. B4 判据阈值（真链 50%）因 B2 差距小而过于宽松。

### 5. 结论（BETA 交付）

**知识链 key 传递机制的判决：谱级相关存在、因果传递不存在（本范式）**。LLM 处理 "apple is a fruit. The fruit is a food." 时两句近似独立，各跳写出由当前 token + 局部上下文决定，上文实体通过语义一致性微调（1.17 倍）而非逐跳 key 通路。这本身是对"LLM 如何组合知识"的重要刻画：**知识链的连贯性来自训练出的句内语义计算，不需要运行时的链式 key 传递**——与 Alpha 的"实体条件化写头"图景一致（每句独立调用写头硬件）。链式传递若存在，需更强范式（问答/推理链）激活，留待后续检验。

### 6. 接续（Phase Gamma，2829）

跨模型验证：14B 上重复 2824-2825 实测写出谱与三件套编辑，检验"写头复用×实体 key"结构跨规模存在性。前置：探针本机是否有 Qwen3-14B 权重；若无则请用户提供或降级为 8B/其他可用模型。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2828_beta_chain.py sha256 = cceefc90826c9356d4c1486eeef84dfed1bff1daf5e8390955af8b2d0e340a47
- …/phase2828/beta_chain/execution.json sha256 = cd31b975f5b2936dacd48c4a4ea4cfe273deb0ee3ced452bcc82893ac2722e72
- …/result.json sha256 = 9bf8862e88dc9cfd693fdefa8ae1f0a8e77839139f7c856df6e5aa60bbf5159f
- …/chain_spec.npz sha256 = dc375fe3894efc6af82b5d29808fbe58074d13f3caceb2255c6fdd55e5fa8630

**Gens 记录**：4 次运行（①缺 d_model 提取 NameError；②einsum 'bkh' 下标错位；③grp_stat 用绝对 token 位置越界，改 PIDX 压缩索引；④干净 61.7s）。预注册判据全程未动；B3 负结果如实入账。
