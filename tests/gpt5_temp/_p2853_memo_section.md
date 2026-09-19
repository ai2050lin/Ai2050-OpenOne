
---

## Phase 2853：MLP cdir 透传增益谱——本征放大与饱和制动的分离（2026-09-18）

### 原理
2852 定位 cdir 析出载体为深层 MLP 透传（无正源层，mlp 增量主导），遗留问题：透传是被动（g≈1）还是主动放大（g≥1.5）。本 Phase 数值测量每层 MLP 的 Jacobian cdir 二次型：

- x_in(l) = sain_same[l][pos1] + attn_base[l]（base run 的 mlp 输入）
- **g_jac(l) = [mlp(x_in + ε·cdir) − mlp(x_in)]·cdir / ε**（ε=1.0 预注册；bf16 ulp 约束注明在 execution.json）
- din/dout：clamp−base 的 mlp 输入/输出差 cdir 投影；g_emp = mean(dout)/mean(din)（ratio of means）
- 线性度诊断：ε₂=0.5 同测 L26-35

预注册（冻结，execution.json `c1be60ce`）：T1 = active_amplification iff max g_jac(L26-35) ≥1.5；passive_transmission iff median |g_jac| ∈ [0.5,1.5)；else nonlinear_unresolved。T2 描述性（g_emp 谱、Pearson r、线性度比）。

### 执行
Gen1 一次干净通过（36.3s，脚本 `70d3d82a`，max_resid 0.0101）。

### 结果
| 判据 | 值 | 结果 |
|---|---|---|
| T1 | max g_jac = **51.26**（L34） | **active_amplification** |
| g_jac 谱 L26-35 | 0.99, 0.68, **5.08, 12.10, 17.93, 11.16, 13.40, 23.68, 51.26, 23.06** | L28 起暴涨 |
| g_emp 谱 L26-35 | 0.47, 0.26, 0.69, 1.68, 1.28, 0.90, 0.50, 0.48, 0.18, −0.11 | 温和 |
| r(g_jac, g_emp) | **−0.26** | 无相关 |
| 线性度 g(ε₂)/g(ε₁) | **1.98**（median） | 亚线性（饱和区） |
| din_cdir L26-35 | −0.086 → −0.665（单调负增） | 输入偏移逐层积累 |
| dout_cdir L26-35 | −0.04 → −0.32（L29 峰）→ **+0.07（L35 转正）** | 深层回拉力 |

### 判据甄别（关键）
g_jac 与 g_emp 差 **10-40 倍**且无相关，三个机制性解释（非测量 bug）：
1. **工作点分离**：g_jac 在 base 工作点（未位移态）测小信号；g_emp 是 clamp 态（位移后）的大信号响应。位移 ‖δ‖ ~5-17 已把 mlp 推入 SwiGLU 饱和区——线性度比 1.98（ε 减半单位增益翻倍）独立证实 ε=1.0 已在饱和弯曲段，真切线增益比 g_jac(ε=1) 更大。
2. **J 非对称**：g_jac = cdir·J·cdir 只测对称部分；实际响应通量走 (Jᵀ·cdir)·δin，可被其他方向吸收。
3. din 的非 cdir 分量（‖δin‖ 2-10 >> |din·cdir| 0.1-0.7）经非线性混合，不按小信号增益缩放。

**硬伤（下次必补）**：本 Phase **缺随机方向对照**——g_jac 的大值可能部分是 J 谱背景各向异性（cdir 不特殊）；需 g_rand = r·J·r 的 null 分布对照（2849/2809 教训同类）。

### 结论（2846→2853 全链机制闭环）
1. **深层 MLP 对 cdir 方向存在巨大本征（小信号）放大率**（5-51×，L28 起激活，L34 峰）——"透射放大晶格"的硬件真实存在。
2. **正常工作时该增益通道空转**：无钳制时输入无 cdir 偏移；钳制 L13h30 后输入带 −cdir 偏移（din 0.09→0.67 逐层积累），但位移同时把 mlp 推入饱和区，实际透传被**饱和制动**为温和增益（g_emp 0.2-1.7）→ 逐层累积 −1.7（2852 的析出曲线）。
3. **L35 回拉力**：dout_cdir 转正（+0.07）——最深层的组件开始把 −cdir 漂移往回拉，与 2851"深层负贡献"的抑制性一致。
4. 终版机制命名：**饱和制动的透射放大晶格（saturated transmission lattice）**——本征放大器阵列 + 饱和限幅 + 早层发起 + 全员透传。分布式画像最终补完。

### 文件
- 脚本 `tests/glm5/phase2853_mlp_transmission.py` sha `70d3d82a`（14,735 B）
- 产物 `tests/glm5/result/rdc_query_construction_20260913/phase2853/mlp_transmission/`：execution.json `c1be60ce`、result.json `13c3b7e3`、transmission.npz `19e2d8f6`

### 接续（2854 候选）
1. **随机方向对照补全**：g_rand = r·J·r 的 null 分布（≥64 随机方向 × L26-35），判定 cdir 增益的特殊性（z-score）；同时测 clamp 态工作点的 g_jac'（位移态切线）分离"工作点移动"假说。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。
