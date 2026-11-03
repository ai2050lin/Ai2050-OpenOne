# PAT Q9-A 预注册 v1：风格/语域编码的组件定位（冻结于 2026-11-01 18:50）

## 0. 动机（owner 命题的可检验切口）

owner 命题：词嵌入已清、MLP 已被 Goodfire（SAE）思路破解、注意力是各语言模式族的底层实体，
下一步把三者拼成整体编码机制，入口是**风格**（中文 / 英文 / 论文 / 代码）怎么编码、数学结构是什么。

本项目审计边界（G5-R01）：atom_registry n=0，**"MLP 已被 SAE 破解"在本项目内部尚未发生**，
是待建项不是已有事实。因此本轮不训练 SAE，而是先回答一个更便宜的定位问题：
**同一命题内容在 6 种语域变体下，风格信息写进残差流的责任组件是 attention 还是 MLP，
以及有多少在 hs[0]（token 身份）就已存在。** 这个定位直接决定后续 SAE 该往哪个组件上挂
（Goodfire 思路的本项目化入口），也给"注意力=模式族底层实体"命题第一块组件级证据。

## 1. 设计

### 1.1 材料（语域 = 帧模板，内容词共享）

60 个事实五元组 `(zh_s, zh_p, en_s, en_p)`（脚本内冻结清单），6 个条件：

| 条件 | 模板 |
|---|---|
| zh_colloq | `你知道吧，{zh_s}很{zh_p}。` |
| zh_acad | `已有研究表明，{zh_s}呈现{zh_p}的特性。` |
| zh_code | `# {zh_s}: {zh_p}=True  # 已验证` |
| en_colloq | `You know, {en_s} is really {en_p}.` |
| en_acad | `Prior research indicates that {en_s} is highly {en_p}.` |
| en_code | `# {en_s}: {en_p}=True  # verified` |

切分：事实索引偶数 → discovery（n=30），奇数 → confirm（n=30）。逐条件同数。
确定性切分，无随机。（R-MET-30 / 近重复纪律：同一事实的 6 个变体**永不跨划分**，划分按事实。）

### 1.2 采集

- 读出位置 A = 每条序列最后一个非 padding token（含句末标点，Q7-A 末位残差流口径）；
  对照位置 B = A−1（标点前一格，排除"标点聚合"解释）。
- 每层采集三个向量（float16，全 36×(L+1) 层，"采一次多次分析"）：
  - `resid_in[l]`：第 l 层 decoder block 的输入（= HF `hidden_states[l]`；l=0 即 hs[0] 嵌入层输出）
  - `attn_out[l]`：`layer.self_attn` 模块输出（写入残差流的注意力增量）
  - `mlp_out[l]`：`layer.mlp` 模块输出（写入残差流的 MLP 增量）
- 恒等式：`resid_in[l+1] = resid_in[l] + attn_out[l] + mlp_out[l]`，跑后逐样本逐层断言（容差 1e-3 相对）。

### 1.3 模型

Qwen3-4B（bf16）先跑，GLM-4-9B（8-bit bitsandbytes，embed/lm_head 跳量化，R-MTH-26 检查）
后跑，串行。两模型结论分别记录，不共享坐标。

## 2. 分析（全部 discovery 拟合 → confirm 评估；主口径 = 位置 A）

- **P1 hs[0] 基线**：`resid_in[0]` 上 3-way（colloq/acad/code）线性探针（LogisticRegression C=1），
  每语言单独 + 合并。这衡量纯 token 身份携带的风格量。
- **P2 层剖面**：同一探针在 `resid_in[l]`，l=0..L−1。报告 confirm 3-way 准确率逐层曲线。
- **P3 组件归因（主判）**：风格轴 `v_c(l)` 由 discovery 均值差定义（各语言内）：
  `v_zh = mean(zh_colloq) − mean(zh_acad)`，`v_en` 同理，`v_code = mean(code) − mean(colloq,acad)`
  （均以该层 discovery 全 6 条件总均值为中心后计算）。逐样本投影：
  - 增量投影 `proj(attn_out[l])`、`proj(mlp_out[l])`
  - 残差投影变化 `Δr(l) = proj(resid_in[l+1]) − proj(resid_in[l])`
  - 报告逐层：Δr 中位数、attn 份额 = median(proj_attn)/（median(proj_attn)+median(proj_mlp)）
    （符号分别处理）、以及 |proj_attn| vs |proj_mlp| 的配对置换 p。
- **P4 跨语言迁移**：zh(colloq vs acad) discovery 训练 → en confirm 测试（反向同）。
  迁移差 = 语言内 confirm − 跨语言 confirm。
- **P5 四基线（R-MET-30）**：字符 2-gram 词袋、token 长度、主导单坐标（discovery 上与风格标签
  |点二列| 最大的单一残差坐标）、L1 范数。全部 discovery→confirm。
- **P6 零分布**：训练标签置换 200 次的 confirm 准确率分布，仅在 l∈{0, L/2, L−1} 三个层与
  合并探针上计算。阈值按零分布倍数写（Q8-A 教训），不写绝对差。

## 3. 预注册判据

- **V1 风格是否只是 token 身份**：若 P2 的 l=0 confirm 3-way ≥ 0.90 且全层最大增量 ≤ 0.03
  且增量 < 3× 零分布 SD ⇒ `STYLE_IS_TOKEN_IDENTITY`；
  若存在某层段 confirm − l=0 ≥ 0.05 且 ≥ 3× 零分布 SD ⇒ `STYLE_HAS_COMPUTED_INCREMENT`。
- **V2 载体组件**：按层带（0–L/4 / L/4–L/2 / L/2–3L/4 / 3L/4–L）报告
  |proj(attn)| : |proj(mlp)| 的中位比与置换 p。判 `ATT_CARRIES` / `MLP_CARRIES` / `MIXED`
  （阈值：比值超 2× 且 p<0.01 才单判，否则 MIXED）。
- **V3 风格的抽象度**：跨语言迁移差 ≤ 0.05 ⇒ `STYLE_ABSTRACT`（语域可跨语言迁移）；
  ≥ 0.15 ⇒ `STYLE_LANGUAGE_BOUND`；之间 `PARTIAL`。
- **V4 hs[0] 与计算增量之比**：报告 `increment_ratio = (max_l acc − acc(l=0)) / max_l acc`。

## 4. 证据等级

无干预 ⇒ **E1 剖面**。按 D10：E1 剖面可独立叫规律的前提是
分布形状 + 置换零分布 + confirm 划分三件齐；单点显著不得叫规律。
任何"注意力/MLP 承载风格"的表述限于"写入增量"层面，**不**等于计算必经（Q2-C1/Q8-D 教训）。

## 5. 风险与已知坑

- 语域差 = 帧词差，词袋基线很可能接近天花板 ⇒ 结论重心在**残差流内部分解**（P3），
  不在探针绝对值。
- 末位标点聚合（Q8-D 材料闸教训）⇒ 位置 B 对照强制。
- 探针在 90 样本 × 2560/4096 维上可能过拟合 ⇒ C=1 强正则 + confirm 划分 + 置换零分布三保险。
- GLM 8-bit 噪声：断言阈值放宽到相对 2e-3，并在结果里记录量化口径。
