## Phase 2825: 差分谱、正交化 key 与真翻转 [2026-09-17 07:01]

### 1. 原理与设计

2824 遗留 P4 失败：top-20 头 rank-1 编辑溢出（banana +2.188），机制定位为"条件化在输入端（key），写出端共享"。本 Phase 攻击写出端共享，两路机制 + 真翻转冲击：

1. **全 48 实体实测谱**：捕获全部 48 实体的 o_proj 输入（每层 48×2 行），产出 spec[L,e,h]（36×48×32）。与 2824 四实体谱交叉验证（apple/sky/grass max diff = 0.000，可复现性完美）。
2. **差分选头（B_diff）**：diff = c_apple − mean(c_sky, c_grass, c_coal, c_banana)，取 top-20（c_apple>0）。与 raw 选头重叠仅 13/20。diff 榜首：L29 h27（0.94）、L35 h22（0.919）、L26 h20（0.627）、L34 h28（0.599）。
3. **正交化 key（C_orth）**：同 B_diff 头组，rank-1 的 key 做 Gram-Schmidt：k' = k_apple − Σ_c (k·k̂_c)k̂_c（c ∈ 4 控制实体）。rank-1 响应 = (x·k')·d —— 对 apple 响应保留（keep_ratio 0.19-0.93，均值 0.65），对控制实体响应被正交化消除。
4. **翻转冲击（FLIP）**：cols20（2822 V1 列组）+ 头组（B_diff 或 C_orth）+ emb-edit（apple token 行 e += β·(dWb−dWr)/g，token 私有 → 结构上零溢出到其他实体行），β ∈ {1,2,4}。

预注册判据（零观测前冻结于 execution.json）：
- P1 差分选头隔离：|d_sky(B_diff)| < |d_sky(A_raw)| 且 d_apple(B_diff) ≥ 0.5·d_apple(A_raw)
- P2 正交 key 隔离：|d_sky(C_orth)| < |d_sky(A_raw)| 且 d_apple(C_orth) ≥ 0.5·d_apple(A_raw)
- P3 banana 溢出修复：min(|d_banana(B_diff)|, |d_banana(C_orth)|) < 1.0
- P4 真翻转：存在（cols+B_diff 或 cols+C_orth）× β ∈ {1,2,4} 使 margin_black_minus_red(apple) > 0

门禁：gate_dW = 1.82e-9 < 1e-6，gate_Z = 2.46e-6 < 1e-4（通过）。

### 2. 结果（Gen2 成功，20.4s）

**判决：P1=true, P2=true, P3=true, P4=true（4/4 全过）**。

| 臂 | d_apple | d_banana | d_sky | d_coal | selectivity | spill_mean(47 实体) |
|---|---|---|---|---|---|---|
| A_raw（2824 复现） | +2.688 | +2.188 | +0.688 | +1.062 | 2.43 | 1.106 |
| B_diff 差分选头 | +2.062 | +1.625 | +0.312 | +0.562 | 3.58 | 0.576 |
| **C_orth 正交 key** | **+2.312** | **+0.125** | **−0.438** | **−0.375** | **10.61** | **0.218** |

**真翻转达成（P4）**：cols20 + B_diff 头组 + emb β=4.0 → margin(apple) = **+4.125 > 0**（基线 −5.562，总位移 9.7）。apple 的 black−red 读数从强红翻为强黑，而 banana margin 移动 = 0.000、cherry 仅 −0.625。**首例实体条件化的颜色改写**：只把苹果改成黑色，香蕉/樱桃不动。

### 3. 分析

1. **正交化 key 是实体条件化的正确算子**：C_orth 把 spill 均值从 1.106 压到 0.218（5 倍），且效力保留 86%（2.312/2.688）。sky/grass/coal/blood 的 Δ 变为小幅负值（−0.1~−0.6）——正交化把"对其他实体的 red 写出"变成了轻微 anti-red，即 rank-1 响应几乎严格限制在 apple key 方向上。
2. **emb-edit 是零结构溢出通道**：apple token 行修改对其他 47 实体行结构上不可见（token id 不同），flip 臂中其余实体的残余移动全部来自头/列部件。
3. **β 阈值效应**：emb β=1/2 时 apple 仅 −1.25/−1.0，β=4 突跳 +4.125——logit 竞争（red vs black softmax 竞争）的非线性门。属性读出不是注入的线性函数；"改颜色"需要越过类别竞争阈值。
4. **2822 问题最终答案升级**：大小能改（2822 E4 翻转）、颜色能改且**可实体条件化**（本 Phase）、持久化 = 写回 safetensors。
5. 全 48 实体 Δ 矩阵（result.json full_delta_matrix）：C_orth 臂 47 个非 apple 实体中 44 个 |Δ|<0.32，最大 lemon +0.812（同为水果，语义近邻溢出——符合"语义距离决定溢出"的预测）。

### 4. 硬伤

1. flip 臂 β=4 是手工扫描值，未做 β 细扫定位阈值；翻转稳定性（多模板、多句式）未验证。
2. C_orth 只对 4 控制实体正交化；对 47 实体全体正交需更大的 key 子空间投影（lemon 溢出即证据）。
3. emb-edit 依赖 tie_word_embeddings（apple 行同时是 lm_head 行），对其他位置预测 apple token 有副作用，本测试未观测但理论存在。
4. 只测 red→black 单向。

### 5. 结论

**具体机制闭环完成**：实体条件化写 = 差分谱选头（输入端 key 私有）× 正交化 key（写出端去共享）× token 私有 emb 注入（零结构溢出）。三件套叠加实现首例"只改苹果颜色"的真翻转（margin −5.56→+4.13，banana Δ=0.000）。参数经济性机制的工程含义确认：写头硬件全局复用、每实体仅需一个条件化 key 方向——约 20 头 × rank-1 ≈ 20×2560 参数即可承载一个实体的一个属性通道。

### 6. 接续（Phase 2826 候选）

1. **8 域谱矩阵**：一次捕获 × 8 个域方向（color/size/weight/speed/taste/...）→ 1152×8 头级功能图谱，检验"写头硬件复用"跨域成立。
2. **阈值机制**：β 细扫 + 中间层读出，定位类别竞争门的位置（logit-lens 轨迹）。
3. 多模板/多句式翻转稳定性验证。
4. Δh 通道分离主线（2810 承接）不变。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2825_diff_spectrum_flip.py sha256 = eff20e39b7beb7bd62465979a8ddad19302e3feee69315a84917daae34c51fa5
- …/phase2825/diff_spectrum_flip/execution.json sha256 = 4a4ddaee695c845104b8299fe7a4eabaa58c7b786279b745840dc83ac77817d5
- …/result.json sha256 = c4a47b2be3fa7767a12c115abeff08f6bb57a269ecb87fa81f7d39d70c99a92a
- …/spec48.npz sha256 = b18f04e3801bdb383ab63ff661d53e9c4af4c6794ea116a720ffe762fa40cf5d

**Gens 记录**：2 次运行（①einsum 下标错误——store[L].reshape(96,32,128) 把 batch×位置维混展，K3 降为二维；改为先按位置维 mean 再 reshape；②干净 20.4s）。预注册判据全程未动。
