## Phase 2833-2835: Δh 通道分离、生成式链与历史谱公式审计 [2026-09-17 09:28]

用户指令：确认 ①Δh 通道分离主线（2810 承接）②生成式链 decode 谱传递 ③历史谱判据公式审计（2781-2830）三项是否完成，未完成则完成。三项均未做过，本节一次补齐。

### Phase 2833: Δh 通道分离（2810 接续 a+d）

**原理**：2810 发现裸 Δ = h_ctx − h_iso 被 attention-sink/位置通用通道支配（"the" 位移最大，P-K2/P-K4 双否证），语义通道只占次要成分。接续方案 (a)：Δ_specific(cond) = Δ_cond − 0.5(Δ_func + Δ_null)（func/null 同位置对照估计通用通道，逐层分离）；方案 (d) 探索性：末层分离 Δ 分解到 32 头输出方向定位搬运头。

**判决：S1=S2=S3=S4=true，channel_separated_substantive**（Gen4 成功 15.1s；三次崩溃：hd 误算 2560//32、hook 键错位 'x' 槽未转存、头分解 einsum 'dkh,kh->d' 把头维求掉应为 '->dk'）。

| 判据 | 读数 | 结论 |
|---|---|---|
| S1 选择性 | cls_spec same 0.0743 vs diff 0.0294（2.5 倍） | 分离后语义通道显现 |
| S2 分离增益 | 分离 0.0743 > 裸 0.0683 | 分离口径更纯净 |
| S3 剂量-反应恢复 | mag same 51.19 > diff 41.39 | 2810 P-K2 反向在分离后转正 |
| S4 层依赖 | cls_spec 层极差 0.110，峰值 L33 | 语义调制集中中晚层 |

头分解（探索性）：L35 h22（0.0143）、h23（0.0098）、h20（−0.0087 拮抗）、h26——与 2824 apple→red 主写头簇（h22/h23/h26）**重合**：通道分离后的语义调制载体就是属性写头硬件。

**结论**：2810 悬案关闭——档案调制的语义通道真实存在，但必须在 func/null 通道分离口径下测量；语义通道载体 = 已知的属性写头簇。LPF v5.3 档案-上下文接口定性：**通用上下文通道（大幅）+ 语义调制通道（小幅但方向特异）双通道结构**。

### Phase 2834: 生成式链 decode 谱传递

**原理**：2828/2830 在 teacher forcing 下发现知识链谱相关但无因果传递。本 Phase 让模型自己贪心解码第二跳（The apple is a fruit. → 8 token），检验 decode 阶段链谱传递（true=apple vs broken=rock 双臂）。

**判决：G1=false / G2=false / G3=true，not_confirmed**（一次成功 12.2s）。

- true 臂生成 " The apple is a fruit that is commonly"——**归纳式复读实体**，未生成 food；
- broken 臂生成 " The fruit is a tree. The tree"——**分类学续写**（rock→fruit→tree），未生成 food；
- c_food 比值 true/broken = 0.874（broken 反而更高）；c_fruit true 0.377 > broken 0.357（G3=true，首句实体激活持续）。

**结论**：生成式解码下链传递第三次否证——模型续写走归纳/分类学局部模式，不主动完成知识链第二跳。知识链连贯性完全是训练时习得的表征结构（共激活），不是运行时传播机制。与 2828/2830 构成三范式收敛证据（teacher forcing 句对 / 同句 / 自由生成）。

### Phase 2835: 历史谱判据公式审计（2781-2834）

**原理**：2830 发现 2828 grp_stat 数值异常并记录"高级索引前置 bug"勘误，但勘误时的机制解释未实证。本 Phase：①静态扫描全部谱脚本索引模式；②numpy 混合索引形状判定矩阵实证；③在落盘谱上数值重算对质。

**判决：audit_clean_with_2828_erratum**（一次成功 0.1s）。

1. **numpy 规则实证**（关键新知识）：`(slice, list, slice)` 布局列表索引**原地保留**（(36,4,32)）；`(slice, list, int, :, int)` 布局**pushfront**（(4,36,32)）——标量 int 在列表后且元组以标量结尾时触发前置。此前"所有混合布局都前置"的假设是错的。
2. **2825 复核 = 无 bug**：`spec[:, ctrl_rows, :].mean(axis=1)` 属原地布局，c_ctrl = 逐层控制行均值，语义正确。B_diff 选头有效。
3. **2828 勘误确认成立**：其 grp_stat 属 pushfront 布局，原值（b1 0.131/0.053）是"逐层 top10（4 行求和）再层均"统计；修正值（0.552/0.099）才是"逐实体层求和 top10 均值"。**两种口径下 true>broken 判决均成立**（orig_order_ok=fixed_order_ok=true），B1-B4 相对判决稳定。
4. **其余全部安全**：2824（纯标量）、2826（纯标量/slice）、2827（标量 e/x）、2830 Gen2（逐行标量）、2831/2832（标量 pos）、2833/2834（复核 einsum 与标量索引）。无其他波及。

### 产物登记（immutable + SHA256）

- 2833：execution 70065439… / result 5da6972f… / delta_specific 94074ffd… / script 0a1e861c…
- 2834：execution 9e8de6c9… / result bfe9ad7c… / gen_spec 0f320378… / script ca3d9979…
- 2835：execution 4468d7dd… / result 68f8f2cd… / audit_report a8bde353… / script 414a9680…

**Gens 记录**：2833 四跑（三崩溃修复后成功 15.1s）；2834 一次成功（12.2s）；2835 一次成功（0.1s）。预注册判据全程未动。

### 接续（2836 候选）

1. 双通道定量模型：通用通道 vs 语义通道的逐层增益分解（对接 2810 数据集全层曲线）。
2. 归纳头与属性写头的关系：2834 true 臂归纳式复读是否经由 L35 属性写头（写谱检验）。
3. 2830 已修复公式的全量谱判据规范入纪律（混合索引布局禁令）。
