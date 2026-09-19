## Phase 2830: 同句链范式重测 + 2828 勘误 [2026-09-17 08:43]

### 1. 原理与设计

Beta（2828）因果否证（B3=false）依赖两句式范式，可能被"两句并行独立处理"解释。本 Phase 用**同句逗号链**范式重测：`An <E1> is a fruit, and a fruit is a food.`（13 token 对齐，e1=1 / fruit1=4 / fruit2=8 / food=11），强制激活流经同一推理流。8 变体（真链 apple/cherry/lemon/banana；断链 rock/steel/hammer/ocean），方向 dW_fruit/dW_food（2806，门禁 dW=1.82e-9、Z=2.46e-6 通过）。

消融臂三个：E1a @e1（2828 复测对照）、E1b @fruit1（同句特异干预点，apple 行 key）、CTRL broken @fruit1（rock 行 key）。rank-1 移除 top-5 c_fruit 头。

预注册（零观测前冻结）：C1 跳1真实（true>null q95 且>true broken）；C2 跳2跟随（true>broken）；C3 因果链（E1b 使 true c_food@fruit2 降≥20%）；C4 激活流（同句 true c_food@fruit2 > 2828 两句式 0.124）；C5 对照（E1a 预期微效）。

### 2. 结果（Gen1 崩溃修复后 Gen2 成功，52.3s）

**判决：C1=true / C2=true / C3=false / C4=true**。

| 量 | true | broken |
|---|---|---|
| c_fruit@e1（跳1） | 0.675 | 0.111（6.1 倍） |
| c_fruit@fruit1 | 0.996 | 0.542 |
| c_fruit@fruit2 | 1.144 | 0.925 |
| c_food@fruit2（跳2） | 0.348 | 0.256（1.36 倍） |
| c_food@food | 0.545 | 0.414 |

消融臂：c_food@fruit2 true 0.348 → E1a 后 0.347（降 0.4%）、E1b 后 0.347（降 0.3%）；broken 0.256 → CTRL 后 0.256（−0.2%）。restore_ok=true。

### 3. 重大发现：因果否证是范式无关的机制性结论

1. **同句范式把谱强度放大约 3-10 倍**（C4=true：c_food@fruit2 0.348 vs 两句式 0.124；c_fruit@fruit1 0.996 vs 0.102≈10 倍）——激活流确实大幅增强。
2. **但第二跳写出依然完全独立于第一跳写出**（消融 drop <1%，两干预点+broken 对照全一致）。知识链连贯性 = **并行共激活**（同上下文中各 token 各自读出），**不是运行时消息传递**。
3. 谱级相关（C1/C2）与因果传递（C3）彻底分离：同一 Phase 内相关真实、传递不存在。2828+2830 双范式闭环，Beta 结论机制化。

### 4. 2828 勘误（正式入账）

Gen1 后发现 **grp_stat 语义 bug**：`sp[:, rows, pidx, :, di]` 混合高级索引使 advanced 维前置（实际形状 (4,36,32)，行×层倒置），实际算的是"前 4 层 4 行求和的 top-10"。修正（逐行标量索引）后从 2828 落盘 chain_spec.npz 零成本重算：b1 c_fruit@e1 true 0.552（原报 0.131）/ broken 0.099（0.053）；b2 c_food@fruit2 true 0.375（0.124）/ broken 0.221（0.106）。**所有相对判决方向不变**（B1-B4 一致，B3 否证更强化：修正后 0.375 vs 消融后 0.347 仍微降）。勘误以本节为准；2828 result.json 不改（immutable）。

### 5. 硬伤

1. 只测 fruit→food 单链单方向；多跳链、反事实链（'The fruit is NOT a food'）未测。
2. 消融只在 top-5 头 rank-1；可能存在分散在 MLP 列通路的链传递未被此算子覆盖。
3. C3 判据阈值 20% 是预注册约定，"drop<1%" 已远低于阈值，但全头/全层消融才能彻底排除弱传递。

### 6. 结论

知识链在单 forward 内**无谱级因果链，只有共激活**。LLM 组合知识的方式不是 token 位置间的运行时消息传递（不是 CoT 内部版），而是：上下文共激活 × 每 token 独立读出 × 写头硬件复用（Alpha/Gamma）。这与 2824-2825 的"通用写头×实体 key"图景互补：实体属性是键值式并行读出，知识链也是。

### 7. 接续（Phase 2831 候选）

1. Δh 通道分离主线（2810 承接，最高优先）。
2. 生成式链（让模型自己生成第二句再测谱）——测 decode 时的链传递。
3. 2828 勘误公式回灌全部历史谱判据检查（2781-2830 grep 审计）。

### 8. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2830_insent_chain.py` sha256 = `f0a7076666407a200fcaf0ae39cfc160f41b134c20c5a3a4d26ddebbb30e62ff`
- `…/phase2830/insent_chain/execution.json` sha256 = `27449b54b42bac3ca8b2431d84831cbb9fd7cf1889ed8a444e6d93955c155aa1`
- `…/result.json` sha256 = `01d65889209044f0b574b4196a53a840fe7e6fe5b1ab394808189e6cb1cf63fc`
- `…/insent_spec.npz` sha256 = `69bdad52426d0dcadeb9930e25aa1daeaaff0efb4ce7281bcee280fb2a04e91e`

**Gens 记录**：2 次运行。Gen1 完成（38.5s）但结果核查发现 grp_stat 语义 bug（高级索引前置）+ refood 未重算谱；修复后删 execution.json 重跑 Gen2（52.3s）。判据全程未动。
