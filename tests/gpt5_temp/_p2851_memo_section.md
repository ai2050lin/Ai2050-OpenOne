
---

## Phase 2851：L30-35 几何重组层解剖——透射放大假说否证（2026-09-18）

### 原理
2850 发现形成器（L13 h30）钳制损伤引发全局状态几何扰动，cdir 分量在 L30-35 陡增（sain 口径 0.04→0.63），且无任何单头与形成器写出方向对齐——析出非离散交接。本 Phase 检验**透射放大假说（transmission amplification）**：放大器（2846 census 前缘，如 L34 h15 直写份额 0.47）读取自身位移态、经 cdir 对齐 OV 通道把位移几何"复印放大"进 cdir 空间。

三臂设计（80 词 × 5 前向，matched protocol，clamp L13h30 至 base awc）：
- **臂 A** 位移旋转剖面：δ(l) = hs_c[l] − hs_b2[l]（residual hs 口径，pos1），报 a_l = δ·cdir/‖δ‖ 与 ‖δ(l)‖ 全 36 层；sain 口径并列作敏感性。
- **臂 B** 逐层增量归因：attn/mlp 残差增量投影 cdir（L14-35）；L30-35 内逐头 Δwrite = a_c·(V_c·OV) − a_b·(V_b·OV) 投影 cdir（绝对口径）→ top 贡献头。
- **臂 C** 透射增益：gain_h = (Δwrite_h·cdir)/(δ(l−1)·cdir)。

预注册（冻结于任何观测前，execution.json `345690ea`）：K1 = transmission_amplification iff ≥60% 的 L14→35 cdirchg 增量来自 L30-35 attn 写出 AND top-5 贡献头 ≥3 属 census 前缘；K2 = top-3 gain 描述性；判决 transmission_lattice iff K1 否则 diffuse_unresolved。

### 执行事故（4 次运行，均观测前修复，正式判决取 Gen4）
1. Gen1 NameError hs_b2：**Edit 报成功未落盘**（沙箱视图缺陷复发）——Grep 复核后重编辑落盘。
2. Gen2 ValueError matmul 0-dim：`hs_c[l][1]` 多套一层索引——forward_run 已做 pos 切片返回 (NL,dim)，`[1]` 取成标量。修为 `hs_c[l]`。
3. Gen3 分母保护 bug：`max(x, 1e-30)` 在 x<0 时返回 1e-30（防零保护反噬）→ frac_late = −6.49e28、gains ±1e28 病态值。**2841 分母病态教训第四次变体：负分母 + max 防零**。修为直接除（|分母|>1e-6 才除，否则 NaN→null）。教训应升级纪律：**防零保护只允许用于已知非负分母；signed 分母一律显式 abs 阈值判断**。
4. Gen4 正式（脚本 `5c16d430`，37.8s，max_resid = 0.004 合格）。

### 结果
| 指标 | 值 | 判据 |
|---|---|---|
| K1 第一条 frac_late（signed/abs 敏感性） | **0.2033 / 0.5053** | ≥0.6 → **fail** |
| K1 第二条 n_front_in_top5 | **0/5** | ≥3 → **fail** |
| K1 整体 | **false** | → **diffuse_unresolved** |
| K2 透射增益 top-3 | L35h22 +0.060 / L35h20 −0.033 / L35h26 +0.030 | 衰减非放大 |

关键数字：
- top5 贡献头（L30-35 Δwrite·cdir）：L35h22（−0.094）、L35h20（+0.052）、L35h26（−0.047）、L32h9（+0.045）、L31h1（+0.045）——**与 census 前缘零重叠**。
- **深层反直觉发现：L30-35 attn 增量 cdir 投影为负（signed Σ −0.065），MLP 更负（−0.59）**；同期残差位移 cdir 分量 0.373→1.761（+1.39）。cdir 析出**不是**深层组件写入——深层组件反而在负向拉。
- 位移范数 ‖δ‖：L14 = 2.47 → L35 = 16.88（全尺度增长，L30-35 加速）；signed 对齐 a_l 自 L24 转负、深层达 −0.10（位移与 cdir 微弱负对齐且加深）。
- cdir 绝对分量 L24 起缓升（0.104→0.373→1.761），但相对份额下降——cdirchg 陡增是"绝对量上升+整体旋转"复合，非份额析出。
- 口径敏感性：sain 口径 cdir 剖面与 residual 口径形状一致（L30-35 陡增复现，0.05→0.62 vs 0.10→1.76），结论不依赖口径。

### 结论
1. **透射放大假说否证**：cdir 在深层的析出既非 census 前缘头写出（0/5），亦非任何 L30-35 组件写入（attn/mlp 增量均为负贡献），透射增益仅 0.03-0.06（衰减量级）。
2. cdirchg L30-35 陡增的本质 = **全局几何重组的表征**：早中层写入的位移经残差流身份携带 + 深层组件的负向重组，位移矢量整体旋转中 cdir 分量绝对值上升。与 2850"全局状态几何扰动"合流。
3. 分布式画像补完最后一块：无魔法头（2846）、无魔法链（2828/2830/2834）、无魔法消费者（2850）、**无魔法重组器（2851）**——深层重组是全组件分布式效应，正源必在早中层写入+携带路径。

### 硬伤
- **索引错位一层**：hidden_states[l] = layer l−1 输出，profile[35] 实为 L34 输出后，L35 写出仅影响未计入的 hs[36]；late 窗口 [30:] 含 L35 写出、漏 L29。对 fail 判决无影响（两口径均 <<0.6，且深层增量方向与判据要求的正贡献相反），但层归属表述需按此约定。
- signed 跨层抵消 vs abs 口径差异大（0.20 vs 0.51），判据在两种口径下均 fail——稳健。
- **析出正源未定位**：深层组件是负源；谁把位移往 cdir 推（早中层哪一层/哪类组件、LN 是否参与旋转）仍开放。

### 文件
- 脚本 `tests/glm5/phase2851_emergence_anatomy.py` sha `5c16d430`（17,210 B）
- 产物 `tests/glm5/result/rdc_query_construction_20260913/phase2851/emergence_anatomy/`：execution.json `345690ea`、result.json `08b6c6f4`、emergence.npz `0769bbe8`

### 接续（2852 候选）
1. **析出正源定位**：L14-29 窗口内逐层 attn/mlp 增量 cdir 剖面（正源应该在 L24-29 的 0.104→0.373 段）+ LN 重组效应检验（LN 是否把弥散位移分量旋转进 cdir 方向）。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。
