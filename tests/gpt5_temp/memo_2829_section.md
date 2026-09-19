## Phase 2829 = Phase Gamma: 跨模型 14B 验证 [2026-09-17 08:25]

### 1. 原理与设计

用户三阶段计划第三步：4B 的"写头复用×实体key"结构在 14B 上是否存在？

本机确认 Qwen3-14B（40 层×40 头=1600 头，hidden 5120，tie_word_embeddings=False）。GPU 16GB < 28GB 权重 → device_map auto 混合卸载（GPU 12GiB / CPU 21GiB 配额）。

**关键管线发现**：14B 的 embed_tokens 行与 lm_head 行**近正交**（实测 cos≈0.005，untied 且训练后独立）——4B 的 z(w)=rmsnorm(E)·g 反解在 14B 上不代表读出方向。修正：隐藏空间读出方向 u_w = g⊙Wu_row(w)（logit_w = <rmsnorm(h)·g, Wu_row>）。gate 改为方向判别力自检（d_red 对 red 的 margin > 对全部 11 个他色，min margin = 2.060）。

重复 2824/2825 核心：48 实体捕获、9 方向实测谱（1600 头）、臂 A_raw（apple top-20 plain）/ C_orth（差分头+正交 key）。**编辑算子改为 hook 等效实现**（混合卸载使部分 o_proj 权重在 meta device 不可读写；rank-1 权重修改 W += outer(d, key/kn2) 数学等效于 o_proj 前向 hook：output += (input·key/kn2)·d，hook 移除即恢复）。c_{l,h} 从已测 spec9 取。

预注册：G1 谱显著；G2 三件套效力（C_orth d_apple ≥ 1.0）；G3 正交隔离（spill < 0.7×A_raw）；G4 末层写头（argmax 在最后层）；G5 束头存在（≥6/9 方向正 且 apple/sky > 5）。

### 2. 结果（Gen5 成功，81.1s；Gen1 跨模型 gate 维度不匹配 → Gen2 z 反解 cos≈0.005 失效 → Gen3/4 加载段错误与 meta tensor → hook 等效编辑解决）

**判决：G1=true, G2=true, G3=true, G4=false, G5=true（4/5）**。

| 量 | 4B | 14B |
|---|---|---|
| 主写头 | L35 h0（c=1.301，末层） | **L32 h11（c=1.560，80% 深度）** |
| 末层最大头 | L35 h0 = argmax | L39 h1（c=0.912）≠ argmax |
| A_raw d_apple | +2.688 | +2.125 |
| C_orth d_apple | +2.312 | +1.188 |
| spill A_raw→C_orth | 1.106→0.218（5.1 倍） | **0.735→0.084（8.75 倍）** |
| 束头 | L29 h27，apple/sky=27.9 | **h5，6/9 方向正，apple/sky=37.1** |
| raw/diff 头重叠 | 13/20 | 14/20 |

### 3. 分析

1. **结构跨规模确认（G1/G2/G3/G5）**：14B 同样存在实测写头谱、"差分谱+正交 key"三件套有效、正交隔离（且更强：8.75 倍 vs 5 倍）、实体条件化属性束写头。"写头复用×实体key"不是 4B 特例，是 Qwen3 家族的架构级组织原则。
2. **G4 层位漂移**：14B 主写头在 L32（80% 深度）而非末层——"末层聚集"不是普适常数，但"晚层（≥75% 深度）主写头+中晚层宽带"结构保持。14B top-10 头全部在 L28-39。
3. **hook 等效编辑**是混合卸载模型的正确干预算子：数学等效、零权重接触、移除即恢复，为受限显存下的大模型编辑提供了通用方案。
4. 14B untied embedding 近正交 lm_head 是重要管线事实：**跨模型迁移方向构造时必须以 Wu 行（读出空间）而非 embedding 行构造方向**。

### 4. 硬伤

1. C_orth 效力保留率下降（14B 1.188/2.125=56% vs 4B 2.312/2.688=86%）——正交化的效力-隔离权衡随规模变化，未细查。
2. c_{l,h} 用 spec9 近似（plain key 值×keep_ratio），非正交 key 的精确 W@k'（meta 不可读）。
3. 未做 14B 几何普查对照与 cols 臂（预算控制）；跨模型对应是功能级（束头/主写头），非逐头映射。
4. B4（Beta）已示注入修复弱，14B 修复未重复。

### 5. 结论（GAMMA 交付：三阶段计划完成）

**跨模型对应图**：4B L35 h0 ↔ 14B L32 h11（主写头，晚层、最强 c）；4B L29 h27 ↔ 14B h5（实体条件化属性束写头，apple/sky 28→37）。**"通用写头硬件（复用）× 实体条件化 key（私有）"在 4B 与 14B 同构成立**，且正交 key 隔离在 14B 更纯（spill 8.75 倍压缩）。参数经济性机制随规模稳定：写头数量与层位有漂移（L35/36→L32/40），组织原则不变。

### 6. 三阶段大计划总结

- **Alpha（2827）**：9 域×48 实体×1152 头谱矩阵；三件套 9/9 域全域有效；分工图修正（无只能走列的域，head/col 比 1.8-12.25）；束头确认。
- **Beta（2828）**：知识链 key 传递判决——谱级相关存在、**因果传递不存在**（B3 消跳 1 写出后 c_food 仅降 0.2%）；陈述句对范式下两句并行独立处理。
- **Gamma（2829）**：14B 跨规模确认"写头复用×实体key"结构同构；束头/主写头功能对应建立；hook 等效编辑算子产出。

### 7. 接续候选

1. 正交化效力-隔离权衡的规模效应细查（keep_ratio 分布 vs 层深）。
2. 14B 束头 h5 的方向谱全展开与编辑（对应 2825 全流程）。
3. 问答式链范式重测 Beta（激活式多跳）。
4. Δh 通道分离主线（2810 承接）。

### 8. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2829_gamma_14b.py sha256 = 699be012b7eb4ef3862bdb8935162d06553547399bf16151de9c5139f80f4430
- …/phase2829/gamma_14b/execution.json sha256 = 756e8a27ea506763c476ea991b213fdcaf9a47f95e1cb76ef48daeb94f775c3b
- …/result.json sha256 = 45cf4f93c6f9f94423d4d80cc495f948202b5cd9bc531e1b05e478e1c55a79da
- …/spec9_14b.npz sha256 = 06ee842258314eb6ac24e3d50b50979983386f132db7616d91fd8fcd92d8528f

**Gens 记录**：5 次运行（①4B gate 2560 维不匹配；②z 反解 cos≈0.005 失效——14B embed⊥lm_head，改 u_w=g⊙Wu_row；③加载段错误 RAM 峰值；④meta tensor 不可写；⑤hook 等效编辑干净通过 81.1s）。预注册 G1-G5 全程未动；G4 负结果如实入账（层位漂移）。
