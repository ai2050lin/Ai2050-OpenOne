
---

## Phase 2854：g_jac 特殊性对照——2853 伪影坐实与机制终版修订（2026-09-18）

### 原理
2853 遗留两项对照义务：①cdir 增益特殊性（缺随机方向 null）；②工作点移动假说（g_jac 测于 base 工作点）。本 Phase 每词每层一次批量 mlp 调用（mlp 逐元素，rows=[cdir, r1..r16]）同时测 17 个方向，base 与 clamp 双工作点，null 分布 = 80 词×16 方向。

预注册（冻结，execution.json `6c3ec81b`）：Z1 = cdir_specific iff ≥6/10 层（L26-35）z=(g_cdir−median(null))/(1.4826·MAD(null)) ≥3；Z2 = operating_point_shift iff ≥6/10 层 g′/g <0.5；判决 saturated_cdir_lattice（Z1∧Z2）/ active_cdir_lattice（Z1∧¬Z2）/ anisotropic_background。

### 执行事故与勘误（本 Phase 最大价值）
1. **Gen1 重大测量 bug（连带 2853 判决作废）**：mlp_batch 直接调用 `model.model.layers[l].mlp`——但 Qwen3MLP 不含内部 LN，真实前向是 mlp(**post_attention_layernorm**(h))。缺失 LN2 导致基线错配：hook 基线 = mlp(LN2(h_b))，扰动臂 = mlp_raw(x+εd)。固定差向量 Δ0=mlp_raw(x)−mlp(LN2(h_b)) 造成系统偏移——完全解释 2853/2854Gen1 的观测结构（g_rand 均值≈0 但 MAD 高达几十；g_cdir 5-51 落入伪影噪声带）。**2853 的 T1=active_amplification（本征 5-51× 放大器）正式作废**——"饱和制动的透射放大晶格"命名作废。D1 一致性 r=1.0 恰是两处同 bug 的复现，不是验证。
2. **Gen2 ratio 病态（第五次负分母防零教训）**：`np.maximum(gcb,1e-9)` 在 gcb<0 时返回 1e-9 → ratio ±4e8 病态、Z2 误判 true。修复为 abs 阈值直接除。
3. **操作事故：phase2852 产物误删**（清理脚本路径复制错），立即重跑恢复——**result.json/npz SHA 与原登记逐位一致**（9b04cac3/e8767687，确定性复现验证通过），仅 execution.json 时间戳更新（`3642fb61`）。纪律有效性的实证：immutable+SHA 登记使误删可完全恢复。
4. **Gen3 正式**（脚本 `302f68f7`，43s，max_resid 0.0039）。

### 正式判决
| 判据 | 值 | 结果 |
|---|---|---|
| Z1 cdir 特殊性 | z 谱 [−0.54,−1.19,−0.31,−1.02,−0.14,−0.42,0.86,1.61,1.55,−0.13]，中位 **−0.23** | **false** |
| Z2 工作点移动 | ratio = [1.00,0.99,1.03,0.94,1.04,0.84,1.02,1.01,0.96,1.19] | **false** |
| **final_verdict** | | **anisotropic_background** |

修复后真实增益谱 g_cdir_base（LN2+MLP 复合路径，cdir·J·cdir）：L26-31 = −0.4~−0.8（微负），L32-34 = 1.7/4.5/7.1，L35 = −1.8。与 2853 g_emp（0.2-1.7）量级相容。

### 机制终版（2846→2854 修订）
1. **无 cdir 特异性放大器**：深层 MLP 的 cdir 增益落在随机方向 null 分布内（z<2）；L32-34 的 4.5-7.1 与 null 背景同量级（J 谱整体抬升，非 cdir 专属）。
2. **无工作点移动/饱和制动**：clamp 态与 base 态切线增益比 0.84-1.19 ≈ 1——2853 的"饱和制动"解释作废。
3. 终版图景：**被动透传带（passive transmission band）**——LN2+MLP 复合路径对 cdir 的有效增益 ~±1（g_emp 0.2-1.7，切线谱 −0.8~+1.8 主体），位移 cdir 分量逐层积累 = din 链式传播 × 透传增益；"cdirchg L30-35 陡增"的真源 = **L29-31 增益 >1 窗口**（g_emp 1.68/1.28/0.90）的积分形状，非放大器阵列。
4. 分布式画像定稿：无魔法头（2846）、无魔法链（2828/2830/2834）、无魔法消费者（2850）、无魔法重组器（2851）、无正源写入层（2852）、**无 cdir 特异放大器（2854）**——损伤的语义后果由"早层发起 + 全层被动线性透传 + 中段 >1 增益窗口"承载，方向结构完全由 cdir 本身（unembed 类间方向）的几何携带。

### 教训制度化
- **LN 边界**：任何模块级 Jacobian/扰动测量必须复现真实前向的前置 LN 链（Qwen3：mlp 前 post_attention_layernorm；attn 前 input_layernorm）。
- **负分母防零（第五次）**：signed 分母一律 abs 阈值 + 显式分支，禁用 max(x,ε)。
- **清理脚本路径核对**：删除类操作执行前必须核对目标路径（本次误删 2852 产物，靠 SHA 登记完全恢复）。

### 文件
- 脚本 `tests/glm5/phase2854_gain_specificity.py` sha `302f68f7`（16,157 B）
- 产物 `phase2854/gain_specificity/`：execution.json `6c3ec81b`、result.json `7347efad`、specificity.npz `b0f5ac17`
- 2852 恢复：execution.json `3642fb61`（result `9b04cac3`/npz `e8767687` 与原登记一致）

### 接续（2855 候选）
1. **L29-31 增益窗口解剖**：g_emp>1 的层内 ln2 前后分解（LN2 贡献 vs SwiGLU 贡献）+ 逐神经元 top-k 贡献（>1 窗口的分子来源）。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。
