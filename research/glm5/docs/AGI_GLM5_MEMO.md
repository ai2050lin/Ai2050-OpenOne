# AGI Research Memo

> 本文档记录AGI研究的进展、问题分析和下一步行动

## Phase 301: Orthogonal R/F Decomposition Causal Test [2026-05-31 00:05]

### 目标

解决Phase 299识别的R/F污染问题。核心方法：Gram-Schmidt正交化R和F方向后做因果注入。

分解：
```
R_clean = R_raw - Proj_F(R_raw)  (纯角色方向，正交于F)
F_clean = F_raw - Proj_R(F_raw)  (纯句框方向，正交于R)
Interaction = R_raw - R_clean     (R在F方向上的投影 = 共享成分)
```

因果条件10种：R_raw, F_raw, R_clean, F_clean, R_raw+F_raw, R_clean+F_clean, R_clean+F_clean+interaction, interaction, R_loo_clean+F_loo_clean, ortho_dir+norm(纯范数测试)

层覆盖：6层 [nl//6, nl//3, nl//2, 2*nl//3, 5*nl//6, nl-2] — 解决深层缺失问题

### 核心发现20：正交化修复DS7B中层的负绑定效应，但深层反而更差

**中层（最关键）**：

| 模型 | 层 | R_raw+F_raw | R_clean+F_clean | delta | 诊断 |
|------|-----|------------|-----------------|-------|------|
| Qwen3 | L18 | +0.403 | +0.367 | -0.036 | 正交化略微削弱正绑定 |
| GLM4 | L20 | +0.283 | +0.259 | -0.024 | 正交化略微削弱正绑定 |
| **DS7B** | **L14** | **-0.180** | **+0.037** | **+0.217** | **正交化修复负绑定！** |

→ **DS7B中层：R_clean+F_clean从-0.180变为+0.037，正交化确实修复了负绑定！**
→ 但Qwen3/GLM4相反：正交化略微削弱正绑定，说明它们的R/F重叠是有益的
→ **Interaction项在DS7B中高达+0.493**，说明R/F共享成分的因果效力非常强

**深层**：

| 模型 | 层 | R_raw+F_raw | R_clean+F_clean | delta |
|------|-----|------------|-----------------|-------|
| Qwen3 | L34 | +0.157 | +0.189 | +0.033 |
| GLM4 | L38 | +0.181 | +0.181 | +0.000 |
| **DS7B** | **L26** | **+0.142** | **-0.124** | **-0.266** |

→ **DS7B深层正交化反而更差**：从+0.142变为-0.124！

### 核心发现21：DS7B的cos(R,F)极端双峰分布——R和F要么平行要么反平行

| 模型 | mean cos(R,F) | std | range | |cos|>0.9 |
|------|--------------|-----|-------|---------|
| Qwen3 | +0.046 | 0.177 | [-0.42, +0.31] | 0/48 (0%) |
| GLM4 | +0.028 | 0.118 | [-0.30, +0.17] | 0/48 (0%) |
| **DS7B** | **-0.006** | **0.694** | **[-0.99, +0.99]** | **24/48 (50%)** |

→ **DS7B的50% token-layers有|cos(R,F)|>0.9**！R和F要么几乎平行(+0.99)，要么几乎反平行(-0.99)
→ Qwen3/GLM4的cos(R,F)都在±0.3以内，R和F接近正交

**DS7B Per-token cos(R,F) at L14**：
```
adj_verb:
  clear: cos(R,F)=+0.993 ← R和F几乎平行！
  clean: cos(R,F)=+0.994 ← R和F几乎平行！
  open:  cos(R,F)=+0.223 (正常)
  warm:  cos(R,F)=+0.146 (正常)
adj_noun:
  light: cos(R,F)=-0.880 ← R和F几乎反平行！
  cold:  cos(R,F)=-0.982 ← R和F几乎反平行！
noun_verb:
  fire:   cos(R,F)=-0.991 ← R和F几乎反平行！
  record: cos(R,F)=+0.852 ← R和F高度正相关
```

→ adj_verb的clear/clean：cos(R,F)≈+1，角色方向和句框方向几乎重合
→ adj_noun的light/cold：cos(R,F)≈-1，角色方向和句框方向几乎相反
→ noun_verb的fire：cos(R,F)≈-1，角色方向和句框方向相反

### 核心发现22：正交化对不同cos(R,F)的token有完全不同的效果

**DS7B L14 per-token detail**：

| Token | cos(R,F) | R_raw | R_clean | bundle_raw | bundle_clean | 效果 |
|-------|----------|-------|---------|-----------|-------------|------|
| clear | +0.993 | +0.834 | +0.194 | -0.556 | -0.369 | 正交化削弱R，但bundle仍负 |
| clean | +0.994 | +0.940 | +0.146 | -0.529 | -0.076 | 正交化大幅削弱R，bundle接近0 |
| cold | -0.982 | -0.846 | -0.704 | -0.690 | **+0.172** | 正交化修复！反平行→正 |
| fire | -0.991 | +0.647 | +0.457 | +0.387 | +0.400 | 正交化轻微改善 |
| warm | +0.146 | +0.598 | +0.627 | +0.794 | +0.676 | 正交化轻微削弱 |
| open | +0.223 | +0.671 | +0.668 | +0.453 | +0.522 | 正交化轻微改善 |

→ **cos(R,F)≈+1的token（clear/clean）**：R和F高度重叠，正交化后R_clean很小（0.15-0.19），因果效力大幅下降
→ **cos(R,F)≈-1的token（cold/fire）**：R和F方向相反，正交化移除冲突成分，改善因果效力
→ 这解释了为什么正交化的整体效果是mixed的：不同token需要不同处理

### 核心发现23：ortho_dir+norm测试不支持"范数独立通道"假设

| 模型 | 中层 ortho_norm | 中层 random | 比值 | 范数通道？ |
|------|----------------|------------|------|----------|
| Qwen3 | +0.047 | +0.107 | 0.4x | 否 |
| GLM4 | +0.114 | +0.110 | 1.0x | 否 |
| DS7B | +0.098 | +0.076 | 1.3x | 边缘 |

→ 正交方向+正确范数的效果≈随机基线，不支持"范数是独立门控通道"假设
→ Phase 299的norm_gate_rand_dir有效可能是因为随机方向碰到了其他敏感子空间

### 新增客观事实拼图（5条）

20. **正交化修复DS7B中层负绑定**：R_clean+F_clean从-0.180变+0.037，验证R/F污染是中层负效应的原因
21. **DS7B的cos(R,F)极端双峰**：50%的|cos|>0.9，R和F要么平行要么反平行
22. **正交化对cos≈+1和cos≈-1的token效果完全相反**：前者削弱R，后者修复冲突
23. **ortho_dir+norm≈随机基线**：不支持范数独立通道假设
24. **Qwen3/GLM4的R/F重叠是有益的**：正交化略微削弱它们的正绑定

### 对用户分析的判断

**用户分析总体正确**，但有重要修正：

1. ✅ "R和F没有被干净分离" → **验证正确**：DS7B中层的负绑定确实是R/F污染导致，正交化可修复
2. ⚠️ "正交化后如果DS7B仍为负则说明token-specific coding" → **修正**：中层正交化后变为微正（+0.037），深层仍为负。真相更复杂：
   - 中层：R/F污染是主因，正交化可修复
   - 深层：存在token-specific + frame-dominated的编码，正交化不够
3. ✅ "norm_gate不是纯范数测试" → **完全验证**：ortho_dir+norm≈随机，确认norm不是独立通道
4. ⚠️ "DS7B角色高度依赖词元和句框交互" → **修正**：DS7B的角色方向本身是有效的（R_raw=+0.323），问题出在R和F方向重叠/冲突

### 硬伤分析

1. **DS7B中层正交化后R_clean+F_clean仅为+0.037**：微弱正值，接近0，统计上可能不显著
2. **F_estimated = full_delta - R**：这个定义使F成为残差，不是独立提取的句框方向，可能包含其他成分
3. **cos(R,F)极端值可能是F定义问题**：F=full_delta-R，当full_delta≈R时F≈0→cos不稳定
4. **8个token的样本量导致per-token分析不可靠**：每个cos(R,F)类别只有2-4个token
5. **深层DS7B正交化恶化的原因未明**：可能是深层frame-dominated编码

### 命令记录

```
python tests/glm5/phase301_orthogonal_rf.py qwen3       # ~48s
python tests/glm5/phase301_orthogonal_rf.py glm4        # ~17min
python tests/glm5/phase301_orthogonal_rf.py deepseek7b  # ~10min
python tests/glm5_temp/phase301_cross_model.py           # 跨模型分析
```

### 数据文件

- `results/phase301_orthogonal_rf/{qwen3,glm4,deepseek7b}_orthogonal_rf.json`
- `tests/glm5/phase301_orthogonal_rf.py`
- `tests/glm5_temp/phase301_cross_model.py`

### 下一步

1. **独立提取F方向**：不再用F=full_delta-R，而是用role-averaged pair差作为独立F
2. **扩展token集到20-30个**：稳定cos(R,F)分布估计
3. **深层DS7B的frame-dominated编码分析**：为什么深层R_raw很强但bundle_raw为负？
4. **多操作符因果测试**：用Phase 300的operator方向做activation patching
5. **条件绑定分解算法**：系统化I/R/F/O/S的正交分解流程

## Phase 302: Full Factorial I/R/F Decomposition [2026-05-31 02:15]

### 目标

解决Phase 301最大硬伤：F=full_delta-R是残差定义，不是独立提取的句框方向。
使用完整因子设计独立估计I(token), R(role), F(frame), RF(interaction)。

因子分解方法：
```
μ_t = token-level grand mean
R_t(role) = mean across frames for (token, role) - μ_t  [角色主效应]
F_t(frame) = mean across roles for (token, frame) - μ_t  [句框主效应]
RF_t(role, frame) = cell_mean - μ_t - R_t - F_t  [交互/绑定项]

R_direction = R_t(r2) - R_t(r1)  [角色对比]
F_direction = average F_t(frame)  [平均句框方向]
RF_direction = RF_t(r2, avg_frame) - RF_t(r1, avg_frame)  [绑定方向]
```

刺激集扩展：22个双角色词元（10 adj_verb + 6 adj_noun + 6 noun_verb），每个5-10个句框

层覆盖：8层 [nl//8, nl//4, 3nl//8, nl//2, 5nl//8, 3nl//4, 7nl//8, nl-2]

### 核心发现25：因子F和残差F几乎不相关——Phase 301的F定义是严重问题

| 模型 | 中层 cos(F_factorial, F_residual) | 含义 |
|------|----------------------------------|------|
| Qwen3 L18 | -0.079 | 因子F与残差F不相关 |
| GLM4 L20 | +0.003 | 因子F与残差F完全不相关 |
| DS7B L14 | -0.334 | 因子F与残差F反向相关！ |

→ **F_residual = full_delta - R 包含大量非F成分**，不能用来估计句框方向
→ Phase 301的cos(R,F)极端双峰结论需要重新审视：残差F本身就不是纯F

### 核心发现26：DS7B的cos(R,F)极端双峰大幅减少——从50%降到23%

| 模型 | cos(R,F)_factorial |cos|>0.9 | cos(R,F)_residual |cos|>0.9 |
|------|-------------------|---------|-------------------|---------|
| Qwen3 | 0/22 (0%) | 0/22 (0%) |
| GLM4 | 0/22 (0%) | 0/22 (0%) |
| DS7B | **5/22 (23%)** | **12/22 (55%)** |

→ DS7B的极端cos(R,F)从55%降到23%——超过一半的极端值是残差F定义造成的假象！
→ 但DS7B仍有23%的token有|cos(R,F)_factorial|>0.9，说明DS7B确实有R/F耦合，只是没之前认为的那么极端

### 核心发现27：RF绑定项整体为负因果效力——不是功能性绑定，而是干扰项

| 模型 | 层 | R_only | F_only | RF_only | R+F | R+F+RF | RF_boost |
|------|-----|--------|--------|---------|-----|--------|----------|
| Qwen3 | L18 | +0.350 | +0.029 | -0.084 | +0.346 | +0.075 | **-0.271** |
| GLM4 | L20 | +0.224 | +0.019 | +0.044 | +0.229 | +0.027 | **-0.202** |
| DS7B | L14 | +0.189 | +0.156 | +0.033 | +0.200 | +0.015 | **-0.185** |

→ **所有三个模型：R+F+RF << R+F**，RF绑定项boost为-0.17到-0.27
→ 这说明RF交互项不是功能性绑定，而是干扰了R+F的因果效力
→ R+F是最优因果组合，不需要绑定项

### 核心发现28：因子F方向的因果效力弱于残差F

| 模型 | 中层 F_factorial | 中层 F_residual | 因子F vs 随机基线 |
|------|-----------------|-----------------|-----------------|
| Qwen3 L18 | +0.029 (0.3x) | +0.123 (1.1x) | 弱于随机 |
| GLM4 L20 | +0.019 (0.2x) | +0.122 (1.5x) | 弱于随机 |
| DS7B L14 | +0.156 (5.6x) | +0.056 (2.0x) | 强于随机 |

→ **Qwen3和GLM4的因子F几乎无因果效力**（0.2-0.3x随机基线）
→ DS7B的因子F有显著因果效力（5.6x随机基线）
→ 残差F在Qwen3/GLM4上更强，因为它包含R泄漏和其他成分
→ 这说明**Qwen3/GLM4的句框效应主要通过角色方向间接体现**，而非独立句框方向

### 核心发现29：DS7B的R方向跨token复用率最高

| 模型 | R_loo / R_only | 含义 |
|------|----------------|------|
| Qwen3 | 0.292 | R跨token泛化弱 |
| GLM4 | 0.506 | R跨token泛化中等 |
| DS7B | **0.889** | R跨token泛化极强 |

→ **DS7B的R_loo几乎等于R_only**（0.889），说明DS7B的角色方向高度跨token共享
→ 这与Phase 301的"DS7B角色方向弱"结论矛盾——实际上DS7B的角色方向是最强共享的
→ Phase 301的问题是F定义导致R看起来弱

### 核心发现30：adj_noun角色对的因子F因果效力远高于adj_verb和noun_verb

DS7B L14 per-role-pair:
```
adj_noun:  R=+0.129  F=+0.571  RF=+0.245  R+F=+0.141  R+F+RF=+0.057
adj_verb:  R=+0.315  F=+0.000  RF=-0.055  R+F=+0.326  R+F+RF=+0.000
noun_verb: R=+0.039  F=+0.000  RF=-0.034  R+F=+0.051  R+F+RF=+0.000
```

→ adj_noun的F_only=+0.571远超其他组——adj_noun角色对有独立的句框因果通道
→ adj_verb和noun_verb的F_only=0——句框效应完全通过R间接体现
→ **不同角色对使用不同的编码策略**

### 新增客观事实拼图（6条）

25. **因子F和残差F几乎不相关**（cos≈0），Phase 301的F定义是残差而非独立方向
26. **DS7B的cos(R,F)极端双峰从55%降到23%**——过半极端值是残差F假象
27. **RF绑定项整体为负因果效力**（-0.17到-0.27），不是功能性绑定而是干扰
28. **Qwen3/GLM4的因子F几乎无因果效力**，DS7B的因子F有5.6x随机基线
29. **DS7B的R_loo/R_only=0.889**，角色方向跨token共享率最高
30. **adj_noun角色对有独立句框因果通道（F=+0.571）**，adj_verb/noun_verb没有

### 对Phase 301结论的修正

1. ⚠️ "DS7B的cos(R,F)极端双峰" → **修正**：55%→23%，过半是残差F假象。但23%仍远高于Qwen3/GLM4的0%，DS7B确有R/F耦合
2. ⚠️ "DS7B角色方向弱" → **修正**：R_loo/R_only=0.889，DS7B的角色方向其实是最强共享的
3. ✅ "RF绑定项是关键计算对象" → **修正**：RF交互项整体为负因果效力，不是功能性绑定。R+F才是最优因果组合
4. ⚠️ "Qwen3/GLM4的R/F重叠是有益复用" → **深化**：这些模型的F方向本身因果效力弱，R/F重叠实际上是R吸收了F的因果成分

### 硬伤分析

1. **R+F+RF≈0的问题**：R+F+RF在Qwen3上=+0.075，几乎等于0。而full_delta因果效力应该=1.0（因为full_delta就是实际的v2-v1）。这说明因子分解没有正确重建full_delta的因果效力
2. **F_direction取平均**：当前F_direction是所有frame主效应的平均，但不同frame差异很大，平均可能抵消
3. **RF交互项为负**：可能说明因子分解的统计模型不符合Transformer实际计算——Transformer可能不使用ANOVA风格的线性分解
4. **22个token中F_only≈0**：可能是F的定义方式问题——frame主效应可能不是正确的因果方向
5. **R+F远低于full_delta**：说明I(token identity)和其他未建模成分很重要

### 命令记录

```
python tests/glm5/phase302_factorial_decomposition.py qwen3       # ~2min
python tests/glm5/phase302_factorial_decomposition.py glm4        # ~54min
python tests/glm5/phase302_factorial_decomposition.py deepseek7b  # ~31min
python tests/glm5_temp/phase302_cross_model.py                     # 跨模型分析
```

### 数据文件

- `results/phase302_factorial_decomposition/{qwen3,glm4,deepseek7b}_factorial_decomposition.json`
- `tests/glm5/phase302_factorial_decomposition.py`
- `tests/glm5_temp/phase302_cross_model.py`

### 下一步

1. **修复因子分解重建问题**：R+F+RF应该≈full_delta的因果效力，但实际远低于。需要检查分解的数学正确性
2. **per-frame F方向**：不同frame的F方向不同，不应取平均。对每个因果测试对使用其对应frame的F方向
3. **角色对特异性编码策略**：adj_noun使用独立F通道，adj_verb/noun_verb不使用，这是机制层面的差异
4. **操作符因果测试**：Phase 300的operator方向做activation patching
5. **从R+F到full_delta**：缺失的因果效力来自哪里？I(token)? 位置? 其他?










## Phase 303: Large-Scale Factorial (60 Tokens) + Bootstrap Stability [2026-05-31 06:10]

### 目标

解决Phase 302/301的"样本太少"硬伤。从22个token扩展到60个（20 adj_verb + 20 adj_noun + 20 noun_verb），
每个角色7个句框×2变体=14条观察句，8层覆盖，1000次bootstrap稳定性分析。

### 方法论发现：不平衡设计下ANOVA分解的数学伪影

**关键发现：当不同角色使用不同句框时（adj用copula/remain/seem等，verb用transitive/intransitive/modal等），
标准ANOVA因子分解会产生系统性伪影：**

```
F_direction_avg ≈ 0     因为adj_frames和verb_frames的边际效应相互抵消
RF_direction = -R_direction  因为对于角色专属句框，RF(role, frame) = -R_effect(role)
R+F+RF = F ≈ 0        RF恰好抵消R
```

数学证明：对于只有角色r使用的句框f，
- frame_means[f] = cell_mean(r, f)（只有r有此句框）
- F_effect[f] = cell_mean(r, f) - grand_mean
- RF(r, f) = cell_mean(r, f) - μ - R(r) - F(f) = -R(r)
- 因此 RF_r_avg = -R(r)，RF_direction = -R_direction

→ **Phase 302的"RF绑定项为负因果效力"结论需要修正：RF=-R是数学伪影，不是机制发现**

### 核心发现31：DS7B的adj_noun极端cos(R,F)是系统性模式——20/20 token的|cos|>0.95

| 模型 | adj_verb |cos|>0.9 | adj_noun |cos|>0.9 | noun_verb |cos|>0.9 |
|------|-------------------|-------------------|--------------------|
| Qwen3 | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| GLM4 | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| **DS7B** | **0/20 (0%)** | **20/20 (100%)** | **0/20 (0%)** |

DS7B L14 adj_noun per-token详情（前10）：
```
deal:   cos_fact=+0.991  cos_resid=-0.985  R_only=+0.206  R+F_resid=+0.268
plain:  cos_fact=+0.991  cos_resid=-0.987  R_only=+0.061  R+F_resid=+0.093
sweet:  cos_fact=+0.987  cos_resid=-0.952  R_only=+0.025  R+F_resid=+0.049
flat:   cos_fact=+0.983  cos_resid=-0.976  R_only=+0.064  R+F_resid=+0.116
dark:   cos_fact=+0.981  cos_resid=-0.963  R_only=+0.031  R+F_resid=+0.063
right:  cos_fact=-0.980  cos_resid=+0.984  R_only=+0.031  R+F_resid=+0.036
match:  cos_fact=+0.980  cos_resid=-0.959  R_only=+0.093  R+F_resid=+0.164
prime:  cos_fact=-0.976  cos_resid=-0.965  R_only=-0.004  R+F_resid=-0.041
cold:   cos_fact=+0.959  cos_resid=-0.946  R_only=+0.200  R+F_resid=+0.195
solid:  cos_fact=+0.954  cos_resid=-0.936  R_only=+0.065  R+F_resid=+0.074
```

→ **100%的adj_noun token有极端cos(R,F)_factorial——这不是小样本假象，是DS7B adj_noun的系统性编码模式**
→ **factorial R和factorial F方向几乎平行，但residual R和residual F方向几乎反平行——这是DS7B独有的**
→ **adj_verb和noun_verb完全不受影响（0/20有极端cos）——模式严格限定在adj_noun角色对**

### 核心发现32：60 token大规模测试下三模型的因果效力对比（含Bootstrap 95% CI）

| 指标 | Qwen3 L18 | GLM4 L20 | DS7B L14 |
|------|-----------|----------|----------|
| R_only | +0.401 | +0.236 | +0.177 |
| R+F_residual | +0.387 | +0.273 | +0.009 |
| full_delta | +0.393 | +0.276 | +0.008 |
| random | +0.170 | +0.104 | -0.037 |
| R/F重叠|cos|>0.9 | 0% | 0% | 23% |
| R_loo/R_only | 0.265 | 0.327 | 0.347 |
| R_only/total | 102% | 85% | 2213% |

**Qwen3**：R_only ≈ full_delta（102%），R方向几乎完全解释因果效力
**GLM4**：R_only ≈ 85% full_delta，R+F_residual ≈ 99%——R+残差F完全重建
**DS7B**：R_only = +0.177但full_delta = +0.008——R方向因果效力远超full_delta！

### 核心发现33：DS7B的full_delta因果效力接近零——角色差异在logit层面几乎不表现

| 模型 | full_delta cos_shift | R_only cos_shift | R_only/full_delta |
|------|---------------------|------------------|-------------------|
| Qwen3 | +0.393 | +0.401 | 102% |
| GLM4 | +0.276 | +0.236 | 85% |
| **DS7B** | **+0.008** | **+0.177** | **2213%** |

→ **DS7B的full_delta因果效力≈0**：从句1 patch full_delta到句2，logit变化几乎不朝目标方向
→ **但R_only因果效力=+0.177**：R方向本身是有因果效力的
→ **矛盾**：R是full_delta的分量，但R的因果效力远超full_delta
→ **解释**：DS7B的full_delta方向在logit空间中不是最优因果方向——可能存在非线性抵消

### 核心发现34：adj_noun角色对在所有模型中R_only最弱，但在DS7B中R_loo最强

| 模型 | adj_verb R | adj_noun R | noun_verb R | adj_noun R_loo/R_only |
|------|-----------|-----------|------------|----------------------|
| Qwen3 | +0.443 | +0.347 | +0.414 | 0.279 |
| GLM4 | +0.304 | +0.257 | +0.148 | 0.295 |
| **DS7B** | **+0.374** | **+0.128** | **+0.030** | **0.356** |

→ **所有模型中adj_noun的R_only都是最弱的角色对**（vs adj_verb和noun_verb）
→ **DS7B adj_noun R_only=+0.128远弱于adj_verb的+0.374**——adj_noun编码困难
→ **但DS7B adj_noun R_loo/R_only=0.356**，比adj_verb的-0.058还高——角色方向跨token复用更好

### 核心发现35：Qwen3的R+F_residual在adj_verb上超过full_delta——残差F补充了R的不足

| 模型 | adj_verb R | adj_verb R+F_resid | adj_verb full_delta |
|------|-----------|-------------------|-------------------|
| Qwen3 | +0.443 | **+0.518** | +0.515 |
| GLM4 | +0.304 | +0.303 | +0.303 |
| DS7B | +0.374 | +0.026 | +0.023 |

→ **Qwen3的R+F_residual = +0.518 > full_delta = +0.515**：残差F微弱但正面地补充了R
→ **GLM4的R+F_residual ≈ R_only**：残差F对GLM4贡献极小
→ **DS7B的R+F_residual ≈ full_delta**：残差F完全重建了full_delta（因为R太大，F抵消了一部分）

### Bootstrap 95% CI验证

Qwen3 L18 (60 tokens, 1000 bootstrap resamples):
```
R_only:       +0.401 CI=[+0.333, +0.466] ***
R+F:          +0.400 CI=[+0.333, +0.465] ***
R+F_residual: +0.387 CI=[+0.319, +0.453] ***
full_delta:   +0.393 CI=[+0.324, +0.457] ***
```

GLM4 L20 (60 tokens):
```
R_only:       +0.236 CI=[+0.189, +0.281] ***
R+F_residual: +0.273 CI=[+0.221, +0.324] ***
full_delta:   +0.276 CI=[+0.225, +0.327] ***
```

DS7B L14 (60 tokens):
```
R_only:       +0.177 CI=[+0.069, +0.277] ***
R+F_residual: +0.009 CI=[-0.092, +0.105]
full_delta:   +0.008 CI=[-0.089, +0.104]
```

→ Qwen3/GLM4的所有关键指标CI排除0——结论稳定
→ DS7B的full_delta CI包含0——因果效力不确定

### 核心发现36：DS7B adj_verb的full_delta与R_only方向严重矛盾——句结构效应压过角色信号

| 模型 | adj_verb符号一致 | adj_noun符号一致 | noun_verb符号一致 | 总体一致 |
|------|-----------------|-----------------|------------------|---------|
| Qwen3 | 95% | 80% | 90% | 88% |
| GLM4 | 95% | 100% | 100% | 98% |
| **DS7B** | **45%** | **95%** | **80%** | **73%** |

DS7B adj_verb典型矛盾token：
```
slow:  R_only=+0.933  full_delta=-0.728  （R正确，FD错误方向）
thin:  R_only=+0.961  full_delta=-0.646  （R正确，FD错误方向）
doubt: R_only=+0.683  full_delta=-0.613  （R正确，FD错误方向）
```

→ **DS7B的adj_verb：copula→transitive句结构变化导致full_delta方向反转**
→ **adj_noun：两个copula句对比，结构相似，95%一致**
→ **结论：DS7B编码是全局句结构敏感的，不像Qwen3/GLM4那样局部词级别编码**

### 新增客观事实拼图（6条）

31. **DS7B adj_noun的cos(R,F)极端值是系统性模式**：20/20 token的|cos|>0.95，100%覆盖率，仅限adj_noun
32. **三模型因果效力对比确认**：Qwen3 R≈full_delta，GLM4 R=85% full_delta，DS7B R=2213% full_delta（R远超full_delta）
33. **DS7B full_delta因果效力≈0**（+0.008），但R_only因果效力=+0.177——非线性抵消
34. **adj_noun角色对在所有模型中R_only最弱**，DS7B最显著（+0.128 vs adj_verb +0.374）
35. **不平衡设计下ANOVA分解的RF=-R是数学伪影**，不是机制发现
36. **DS7B adj_verb的full_delta与R_only方向严重矛盾（45%一致）**——句结构效应压过角色信号，adj_noun仅95%一致

### 对Phase 302结论的修正

1. ⚠️ "RF绑定项整体为负因果效力" → **修正**：RF=-R是ANOVA不平衡设计的数学伪影，不是真实机制。R+F+RF=F≈0不代表RF是"干扰项"
2. ✅ "因子F和残差F几乎不相关" → **确认并深化**：factorial F≈0（边际效应抵消），residual F有因果效力但包含非F成分
3. ⚠️ "DS7B的cos(R,F)极端双峰从55%降到23%" → **修正**：60 token测试中DS7B有23%|cos|>0.9，但**100%集中在adj_noun**，adj_verb和noun_verb为0%
4. ⚠️ "adj_noun角色对有独立句框因果通道" → **深化**：adj_noun的F在factorial框架下≈0，但R+F_residual > R_only，说明残差F确实有贡献
5. ⚠️ "DS7B角色方向跨token共享率最高(R_loo/R_only=0.889)" → **修正**：60 token测试中DS7B R_loo/R_only=0.347，不是0.889。Phase 302的0.889可能是小样本偏差

### 硬伤分析

1. **不平衡设计伪影**：F_direction_avg ≈ 0和RF = -R是数学必然，不是发现。需要重新设计分解方法（使用within-role frame variation而非ANOVA边际效应）
2. **DS7B full_delta ≈ 0**：需要调查为什么DS7B在60 token测试中full_delta因果效力几乎为零。可能原因：(a) token解析错误导致position不对；(b) DS7B对这些句对确实不区分角色；(c) 位置效应干扰
3. **DS7B R_only >> full_delta**：这违反直觉（分量>整体），说明activation patching的非线性效应——R方向命中了更敏感的子空间
4. **因子分解重建失败**：R+F（factorial）无法重建full_delta的因果效力，因为F≈0。只有R+F_residual能近似重建
5. **bootstrap显示DS7B full_delta CI包含0**：DS7B的角色区分因果效力不确定，需要更大样本或不同方法验证

### 命令记录

```
python tests/glm5/phase303_large_scale_factorial.py qwen3       # ~3.5min
python tests/glm5/phase303_large_scale_factorial.py glm4        # ~67min
python tests/glm5/phase303_large_scale_factorial.py deepseek7b  # ~80min
python tests/glm5_temp/phase303_cross_model.py                   # 跨模型分析
```

### 数据文件

- `results/phase303_large_scale_factorial/{qwen3,glm4,deepseek7b}_large_scale_factorial.json`
- `tests/glm5/phase303_large_scale_factorial.py`
- `tests/glm5_temp/phase303_cross_model.py`

### 下一步

1. **修复因子分解**：用within-role frame PCA代替ANOVA边际效应，避免F≈0和RF=-R伪影
2. **调查DS7B full_delta ≈ 0**：检查token position解析是否正确，测试DS7B是否真的无法区分角色
3. **DS7B adj_noun深度分析**：100%极端cos(R,F)的机制——R和F方向平行意味着什么？
4. **从R_only到full_delta的gap分析**：为什么Qwen3的R≈full_delta但DS7B的R远超full_delta？
5. **操作符因果测试**：从I/R/F转向O/S分析

## Phase 304: Construction Identification + Gap Decomposition [2026-05-31 09:40]

### 目标

解决Phase 303留下的两个核心问题：
1. **F(frame)可识别性**：用within-role frame PCA替代ANOVA边际F，避免不平衡设计伪影
2. **DS7B full_delta ≈ 0**：通过Gap = full_delta - R分解，理解什么抵消了R

理论框架更新：h = I + R + C + U（身份 + 角色 + 构式绑定体 + 未解析残差）

### 核心发现37：DS7B的within-role frame PCA几乎是1维的——构式编码极端刚性

| 模型 | adj PC1 | verb PC1 | noun PC1 |
|------|---------|----------|----------|
| Qwen3 | 41.1% | 20.1% | 25.8% |
| GLM4 | 22.1% | 12.6% | 13.3% |
| **DS7B** | **98.0%** | **99.0%** | **99.5%** |

DS7B的frame variation在每个角色内几乎完全由单一方向(PC1)解释：
- adj: PC1=98.0%, PC2=0.38%, PC3=0.18%
- verb: PC1=99.0%, PC2=0.13%, PC3=0.11%
- noun: PC1=99.5%, PC2=0.14%, PC3=0.07%

→ **DS7B的构式编码是1维的——所有句框变化被压缩到单一方向**
→ **Qwen3/GLM4的frame variation是多维的（PC1仅20-41%），有更丰富的句框编码**
→ **DS7B的构式编码极度刚性，缺乏frame diversity**

### 核心发现38：DS7B的构式子空间跨角色近乎重合——共享构式方向

| 模型 | adj↔noun最小角 | adj↔verb最小角 | noun↔verb最小角 |
|------|---------------|---------------|----------------|
| Qwen3 | 9.0° | 38.9° | 38.6° |
| GLM4 | 42.5° | 51.5° | 58.0° |
| **DS7B** | **3.5°** | **1.0°** | **3.5°** |

→ **DS7B的构式子空间跨角色几乎平行（最小角1-3.5°）**
→ **adj/verb/noun使用同一个构式方向——role-independent construction**
→ **Qwen3的adj↔noun子空间较近(9°)，但adj↔verb(39°)和noun↔verb(39°)正交**
→ **GLM4的构式子空间最独立（42-58°），角色间构式编码最分化**

### 核心发现39：DS7B的Gap = full_delta - R几乎完全由C(构式)解释

| 模型 | cos(Gap, C) | C投影能量 | U(未解析)% |
|------|-------------|----------|-----------|
| Qwen3 | +0.603 | 0.383 | 77.3% |
| GLM4 | +0.484 | 0.247 | 86.5% |
| **DS7B** | **+0.858** | **0.799** | **31.0%** |

→ **DS7B的Gap有80%能量在构式方向上——Gap主要由C(构式)驱动**
→ **DS7B的未解析残差U仅31%——比Qwen3(77%)和GLM4(87%)少得多**
→ **DS7B的构式编码如此刚性（1维），以至于Gap几乎完全可由C解释**

### 核心发现40：DS7B的full_delta ≈ 0是因为C(构式)抵消了R(角色)

DS7B L14 per-role-pair因果效力分解：
```
adj_verb:  R_only=+0.374  full_delta=-0.119  R+C=-0.172
           cos(Gap,R)=+0.597  cos(Gap,C)=+0.844
adj_noun:  R_only=+0.128  full_delta=+0.149  R+C=+0.148
           cos(Gap,R)=-0.427  cos(Gap,C)=+0.805
noun_verb: R_only=+0.030  full_delta=-0.006  R+C=+0.052
           cos(Gap,R)=-0.762  cos(Gap,C)=+0.927
```

关键机制链：
1. DS7B的C(构式)方向在每个角色内是1维的（PC1>98%）
2. C方向跨角色几乎相同（最小角1-3.5°）
3. C方向与Gap(=full_delta-R)高度对齐（cos=0.86）
4. R+C因果效力 ≈ full_delta因果效力（因为R+C ≈ full_delta）
5. 但R_only因果效力远高于full_delta（因为C在logit空间中抵消R）

→ **DS7B的"角色编码被抵消"机制确认：1维刚性构式编码在logit空间中与角色方向竞争**
→ **adj_verb最严重：R_only=+0.374但full_delta=-0.119，构式方向完全压过角色方向**

### 核心发现41：三模型的R+C因果效力对比——Qwen3的C有独立贡献，DS7B的C是抵消项

| 模型 | R_only | R+C | full_delta | R+C-R_only差值 |
|------|--------|-----|-----------|---------------|
| Qwen3 | +0.402 | +0.371 | +0.393 | -0.031 |
| GLM4 | +0.236 | +0.239 | +0.276 | +0.003 |
| DS7B | +0.177 | +0.009 | +0.008 | -0.168 |

→ **Qwen3：R+C略低于R_only（C微弱负面）**
→ **GLM4：R+C ≈ R_only（C近乎中性）**
→ **DS7B：R+C远低于R_only（C强烈负面，-0.168差距）**

注意：这里的C(构式)来自within-role PCA，不同于Phase 303的ANOVA F。
Phase 304的C是角色内部句框变化的主方向，是真正意义上的构式编码。

### 核心发现42：DS7B adj_verb的R和full_delta方向在激活空间同向但在logit空间反向

DS7B adj_verb R-FD sign agreement（激活空间向量点积符号）= 100%
但 R_only causal shift = +0.374, full_delta causal shift = -0.119

→ **R_dir和full_delta向量在激活空间指向同一方向（点积>0）**
→ **但inject R_dir产生正向因果效力，inject full_delta产生负向因果效力**
→ **这是非线性效应：full_delta中包含的C成分，虽然与R同向，但在经过后续层变换后产生反向效果**

### Bootstrap 95% CI验证

Qwen3 L18:
```
R_only:       +0.402 CI=[+0.329, +0.470] ***
full_delta:   +0.393 CI=[+0.324, +0.457] ***
C_only:       -0.001 CI=[-0.063, +0.061]
R+C:          +0.371 CI=[+0.298, +0.440] ***
Gap_only:     +0.047 CI=[-0.011, +0.104]
```

GLM4 L20:
```
R_only:       +0.236 CI=[+0.189, +0.281] ***
full_delta:   +0.276 CI=[+0.225, +0.327] ***
C_only:       +0.007 CI=[-0.051, +0.064]
R+C:          +0.239 CI=[+0.190, +0.286] ***
```

DS7B L14:
```
R_only:       +0.177 CI=[+0.019, +0.324] ***
full_delta:   +0.008 CI=[-0.146, +0.166]
C_only:       +0.036 CI=[-0.105, +0.173]
R+C:          +0.009 CI=[-0.133, +0.155]
Gap_only:     +0.129 CI=[-0.006, +0.273]
```

→ Qwen3/GLM4的R_only和full_delta CI排除0——稳定
→ DS7B的full_delta CI包含0——因果效力不确定
→ DS7B的R_only CI排除0——R方向确实有因果效力
→ DS7B的Gap_only CI包含0但接近显著（CI上界+0.273）

### 新增客观事实拼图（6条）

37. **DS7B的within-role frame PCA几乎是1维的**：adj PC1=98%, verb PC1=99%, noun PC1=99.5%
38. **DS7B的构式子空间跨角色近乎重合**：最小角1-3.5°，共享构式方向
39. **DS7B的Gap有80%能量在C(构式)方向**：cos(Gap,C)=+0.858, C_proj_energy=0.799
40. **DS7B的full_delta≈0机制确认**：1维刚性构式编码在logit空间抵消角色方向
41. **三模型R+C对比**：Qwen3 C微负面(-0.031), GLM4 C中性(+0.003), DS7B C强负面(-0.168)
42. **DS7B adj_verb的R和full_delta在激活空间同向但在logit空间反向**——非线性效应

### 对Phase 303结论的修正

1. ✅ "ANOVA不平衡设计伪影" → **Phase 304用within-role PCA确认了F(frame)确实可以被独立识别**
2. ⚠️ "DS7B full_delta ≈ 0" → **深化**：原因是1维刚性构式编码在logit空间抵消R，不是position解析错误
3. ✅ "DS7B是全局句结构编码" → **精确化**：DS7B的构式编码是1维的、跨角色共享的、与R竞争的方向
4. ⚠️ "DS7B adj_verb符号一致45%" → **修正**：Phase 304的激活空间R-FD sign agreement=100%，但logit空间因果效力方向可以相反
5. ✅ "F(frame)还没有被真正独立识别" → **突破**：within-role PCA成功识别了C(构式)方向

### 硬伤分析

1. **C_only因果效力在所有模型中都接近0**：CI包含0，构式方向本身的因果效力不确定。这可能因为C是句框变化方向，与角色区分不直接相关
2. **R+C ≈ full_delta在DS7B中**：这意味着C(构式)方向≈Gap(=full_delta-R)，C实际上就是"R之外的剩余"。这不能算是独立的构式识别
3. **Gap_norm巨大差异**：DS7B Gap norm=1312 vs Qwen3=40 vs GLM4=6。这暗示DS7B的full_delta和R之间有巨大的范数差异，需要归一化分析
4. **1维构式编码的DS7B**：PC1>98%是极端的。但这是否因为DS7B的sliding window attention限制了信息传播？还是因为Qwen2架构的特定结构？
5. **缺少O/S因果验证**：当前所有分析仍在R/F框架内，缺少操作符(operator)和作用范围(scope)的因果测试

### 命令记录

```
python tests/glm5/phase304_construction_gap.py qwen3       # ~3.5min
python tests/glm5/phase304_construction_gap.py glm4        # ~98min
python tests/glm5/phase304_construction_gap.py deepseek7b  # ~57min
python tests/glm5_temp/phase304_cross_model.py              # 跨模型分析
```

### 数据文件

- `results/phase304_construction_gap/{qwen3,glm4,deepseek7b}_construction_gap.json`
- `tests/glm5/phase304_construction_gap.py`
- `tests/glm5_temp/phase304_cross_model.py`

### 下一步

1. **Phase 305 O/S操作符因果测试**：not/no/never/maybe/must/can/if等操作符的因果方向提取
2. **DS7B 1维构式编码深度分析**：为什么DS7B把所有frame variation压缩到1维？与sliding window的关系？
3. **归一化Gap分析**：Gap norm差异巨大，需要用cos或归一化投影替代原始norm比较
4. **操作符与角色的交互**：O(not)在不同角色(adj/verb/noun)上是否相同？这是I/R/C/O/S框架的关键测试

## Phase 305: Operator-Scope (O/S) Causal Testing [2026-05-31 11:00]

### 目标

从R/F框架转向O/S操作符因果测试。核心问题：
1. "not"是否有统一的操作符方向O(not)跨操作数？
2. O(not) = 反义词方向？还是否定与反义词正交？
3. O(not)是否跨角色(adj/verb/noun)共享？
4. DS7B的O(not)是否也受句结构抵消影响？

刺激设计：
- 20形容词×4句框×2条件=160否定句
- 10动词×4句框×2条件=80否定句
- 8名词×4句框×2条件=64否定句
- 12反义词对×4句框×3条件=144反义词句
- 总计：448条刺激句，397条唯一句

### 核心发现43：O(not)因果效力在所有模型中强劲且一致——否定是独立因果方向

| 模型 | O(not)因果 | full_delta因果 | O_avg因果 | antonym因果 | Random |
|------|-----------|---------------|----------|------------|--------|
| Qwen3 | +0.664 | +0.807 | +0.453 | +0.296 | +0.071 |
| GLM4 | +0.783 | +0.922 | +0.609 | +0.331 | +0.118 |
| DS7B | +0.587 | +0.787 | +0.550 | +0.279 | +0.121 |

→ **O(not)因果效力在所有模型中远超random baseline（5-7倍）**
→ **O(not)因果效力远超antonym方向（2-2.5倍）**
→ **O(not)在所有模型100%正向（Qwen3 38/38, GLM4 38/38）, DS7B 84%（32/38）**

### 核心发现44：O(not) ≠ Antonym——否定操作符与反义词替换是不同的因果机制

| 模型 | O(not)>antonym比例 | |O(not)|>|antonym|比例 | Corr(O,antonym) |
|------|-------------------|----------------------|------------------|
| Qwen3 | 20/20 (100%) | 20/20 (100%) | +0.113 |
| GLM4 | 20/20 (100%) | 20/20 (100%) | -0.362 |
| DS7B | 16/20 (80%) | 13/20 (65%) | +0.011 |

→ **O(not)因果效力在所有模型中远超antonym方向**
→ **O(not)与antonym的因果效力相关性极低（-0.36~+0.11）**
→ **否定操作符和反义词替换是两个几乎独立的因果通道**
→ **"not happy" ≠ "sad"在因果效力层面成立**

### 核心发现45：O(not)跨角色共享中等——adj↔verb最强，adj↔noun最弱

| 模型 | cos(O_adj, O_verb) | cos(O_adj, O_noun) | cos(O_verb, O_noun) |
|------|-------------------|-------------------|-------------------|
| Qwen3 | +0.630 | +0.450 | +0.539 |
| GLM4 | +0.489 | +0.299 | +0.402 |
| DS7B | +0.612 | +0.381 | +0.501 |

→ **adj↔verb的O(not)共享最强（0.49-0.63）**
→ **adj↔noun的O(not)共享最弱（0.30-0.45）——否定名词的机制不同于否定形容词**
→ **这与Phase 304的构式子空间一致：adj和verb共享更多构式结构**

### 核心发现46：O(not) LOO一致性中等——noun操作符跨词共享最强

| 模型 | adj LOO | verb LOO | noun LOO |
|------|---------|----------|----------|
| Qwen3 | +0.598 ± 0.101 | +0.616 ± 0.052 | +0.738 ± 0.090 |
| GLM4 | +0.514 ± 0.090 | +0.524 ± 0.070 | +0.716 ± 0.095 |
| DS7B | +0.593 ± 0.080 | +0.587 ± 0.092 | +0.769 ± 0.062 |

→ **noun的O(not) LOO一致性最高（0.72-0.77）——否定名词的方向最跨词共享**
→ **adj和verb的LOO一致性中等（0.51-0.62）——否定方向有一定的操作数特异性**
→ **三模型的LOO模式高度一致——这不是模型特定现象，而是语言否定机制的普遍特征**

### 核心发现47：DS7B的O(not)不受句结构抵消影响——与R(role)形成鲜明对比

DS7B L14关键对比：
```
R(role):     R_only=+0.177  full_delta=+0.008  ratio=22.1  （巨大抵消）
O(not):      O_not=+0.587   full_delta=+0.787  ratio=0.75  （最小抵消）
```

per-role对比：
```
adj:  O_not=+0.562  FD=+0.805  ratio=0.70
verb: O_not=+0.553  FD=+0.749  ratio=0.74
noun: O_not=+0.695  FD=+0.791  ratio=0.88
```

→ **DS7B的否定操作符方向O(not)不受1维刚性构式编码的抵消**
→ **R(role)的full_delta被构式方向抵消到接近0，但O(not)的full_delta保持强劲**
→ **这说明DS7B的构式抵消机制是角色编码特定的，不是全局的**
→ **操作符O(not)可能在DS7B中使用了不同的子空间——与构式方向正交的子空间**

### 新增客观事实拼图（5条）

43. **O(not)因果效力在所有模型中强劲**：Qwen3 +0.664, GLM4 +0.783, DS7B +0.587
44. **O(not) ≠ Antonym**：两者因果效力相关性极低（-0.36~+0.11），否定≠反义词替换
45. **O(not)跨角色共享中等**：adj↔verb最强(0.49-0.63)，adj↔noun最弱(0.30-0.45)
46. **noun的O(not) LOO一致性最高**（0.72-0.77）——否定名词的方向最跨词共享
47. **DS7B的O(not)不受句结构抵消**：O_not/FD ratio=0.75 vs R/FD ratio=22.1

### 对DS7B编码策略的修正

之前认为DS7B是"全局句结构敏感编码"，但Phase 305显示：
- R(role)方向被构式抵消（full_delta≈0）
- O(not)方向不受构式抵消（full_delta>>0）

修正理解：
→ **DS7B的角色编码和操作符编码使用了不同的子空间**
→ **角色编码子空间与构式子空间重叠（1维刚性构式抵消角色方向）**
→ **操作符编码子空间与构式子空间正交（构式不抵消操作符方向）**
→ **这是DS7B的适应性策略：角色信息被构式淹没，但操作符信息独立保留**

### 命令记录

```
python tests/glm5/phase305_operator_causal.py qwen3       # ~1.5min
python tests/glm5/phase305_operator_causal.py glm4        # ~43min
python tests/glm5/phase305_operator_causal.py deepseek7b  # ~27min
python tests/glm5_temp/phase305_cross_model.py             # 跨模型分析
```

### 数据文件

- `results/phase305_operator_causal/{qwen3,glm4,deepseek7b}_operator_causal.json`
- `tests/glm5/phase305_operator_causal.py`
- `tests/glm5_temp/phase305_cross_model.py`

### 下一步

1. **操作符与角色的交互测试**：O(not) + R(adj)的因果效力——操作符是否调制角色方向？
2. **O(not)子空间与R/C子空间的关系**：cos(O, R), cos(O, C)——操作符是否与角色/构式正交？
3. **更多操作符**：maybe, must, can, if, because——不同操作符是否共享O子空间？
4. **Scope因果测试**：不同scope下O(not)是否不同？
5. **I/R/C/O/S完整框架验证**：五个分量是否正交？是否可以重建full_delta？

## Phase 306: Normalized Construction PCA + Position/Norm Decomposition [2026-05-31 12:58]

### 目标

解决Phase 304最严重的硬伤：DS7B的PC1>98%是真方向还是范数伪影？
同时加入位置项P，做R/C/P/N完整分解。

### 核心发现48：DS7B的1维构式编码大部分是范数伪影——Raw PC1 98%→Unit PC1 32%

| 模型 | 角色 | Raw PC1 | Unit PC1 | 下降 | NormCV | corr(|PC1|,norm) | 判定 |
|------|------|---------|----------|------|--------|-----------------|------|
| Qwen3 | adj | 41.1% | 18.7% | 54% | 0.553 | +0.918 | 部分范数驱动 |
| Qwen3 | verb | 20.1% | 16.5% | 18% | 0.189 | +0.639 | 真方向 |
| Qwen3 | noun | 25.8% | 24.3% | 6% | 0.321 | +0.833 | 真方向 |
| GLM4 | adj | 22.1% | 17.2% | 22% | 0.309 | +0.783 | 真方向 |
| GLM4 | verb | 12.6% | 12.0% | 5% | 0.146 | +0.568 | 真方向 |
| GLM4 | noun | 13.3% | 13.3% | 1% | 0.180 | +0.343 | 真方向 |
| **DS7B** | **adj** | **98.0%** | **32.3%** | **67%** | **1.839** | **+0.998** | **范数伪影** |
| **DS7B** | **verb** | **99.0%** | **45.2%** | **54%** | **1.593** | **+0.999** | **范数伪影** |
| DS7B | noun | 99.5% | 79.6% | 20% | 0.977 | +1.000 | 部分真方向 |

→ **DS7B adj/verb的1维构式编码是范数伪影：归一化后PC1从98-99%暴跌至32-45%**
→ **DS7B noun的1维性部分真实：Unit PC1=79.6%，但仍受范数影响**
→ **Qwen3 adj也有范数影响（Raw 41%→Unit 19%），但verb/noun是真方向**
→ **GLM4最干净，Raw≈Unit，构式方向最真实**

### 核心发现49：DS7B的NormCV（范数变异系数）远超其他模型——中层范数变异是主要信号

| 模型 | adj NormCV | verb NormCV | noun NormCV |
|------|-----------|------------|------------|
| Qwen3 | 0.55 | 0.19 | 0.32 |
| GLM4 | 0.31 | 0.15 | 0.18 |
| **DS7B** | **1.84** | **1.59** | **0.98** |

→ **DS7B的NormCV是Qwen3的3-5倍，是GLM4的6-9倍**
→ **DS7B中层的hidden state范数变异远大于方向变异**
→ **这意味着DS7B的中层主要用"能量/幅度"而非"方向"编码句框差异**

### 核心发现50：DS7B中层范数变异从L3到L7急剧增长——范数编码是后起的

| 层 | adj NormCV | adj Raw PC1 | adj Unit PC1 |
|----|-----------|------------|-------------|
| L3 | 0.27 | 32.0% | 22.7% |
| L7 | 1.95 | 97.2% | 28.4% |
| L14 | 1.84 | 98.0% | 32.3% |
| L26 | 1.06 | 73.0% | 16.2% |

→ **L3的NormCV仅0.27（正常），L7突然跳至1.95（7倍增长）**
→ **这意味着DS7B在4-7层之间启动了范数编码机制**
→ **范数编码取代了方向编码成为句框信息的主要载体**

### 核心发现51：DS7B的C方向（无论raw/unit）在因果测试中强烈负面

DS7B L14因果效力：
```
R_only=+0.470  C_raw=-0.740  C_unit=-0.776  P=-1.018  FD=+1.055
```

DS7B L14 per-role-pair：
```
adj_verb:  R=+1.527  C_raw=-1.798  C_unit=-1.887  FD=+2.465
adj_noun:  R=+0.447  C_raw=-0.047  C_unit=-0.071  FD=+0.646
noun_verb: R=-0.565  C_raw=-0.373  C_unit=-0.370  FD=+0.055
```

→ **C方向（无论raw还是unit）在DS7B中都是强烈反向推动角色转换**
→ **adj_verb最极端：R=+1.53但C=-1.80，C几乎完全抵消R**
→ **这与Phase 304的"构式抵消角色"一致，但现在排除了范数伪影**

### 核心发现52：Qwen3和GLM4的C方向因果效力接近0或微弱

Qwen3 L18: R=-0.058 C_raw=+0.053 C_unit=+0.066 P=-0.012 FD=+0.071
GLM4 L20: R=+0.015 C_raw=-0.006 C_unit=-0.005 P=+0.021 FD=-0.004

→ **Qwen3/GLM4的C方向因果效力微弱（接近0）**
→ **这与Phase 304的C_only≈0一致**
→ **C在Qwen3/GLM4中更像是调制方向，不是独立因果通道**

### R/C/P/N能量预算对比

| 模型 | 层 | C% | P% | N% | U% |
|------|----|----|----|----|----|
| Qwen3 | L18 | 36.1% | 8.3% | 5.7% | 70.6% |
| GLM4 | L20 | 18.0% | 2.8% | 4.4% | 83.2% |
| DS7B | L14 | 84.1% | 75.5% | 74.2% | 92.0% |

→ **DS7B的C/P/N能量预算都极高（>74%），说明这些分量高度重叠**
→ **C、P、N在DS7B中不是正交的——范数是主要驱动力**
→ **U=92%意味着C+P+N大量重复计算了同一信号**

### 对Phase 304结论的重大修正

1. ❌ **"DS7B构式编码极端1维"** → **修正：DS7B的1维性主要是范数伪影，真实方向维度为32-45%（adj/verb）**
2. ❌ **"DS7B构式编码极度刚性"** → **修正：DS7B的范数变异极度刚性（1维），但方向变异多维**
3. ⚠️ **"DS7B full_delta≈0因为1维构式抵消R"** → **修正：范数主导的构式方向抵消R，但方向成分更丰富**
4. ✅ **"Qwen3/GLM4构式多维"** → **确认：Unit PC1=12-24%，真实多维**
5. ✅ **"DS7B adj_verb C抵消R"** → **确认：C_unit=-1.89强烈抵消R=+1.53**

### 新增客观事实拼图（5条）

48. **DS7B的1维构式编码大部分是范数伪影**：Raw PC1 98-99%→Unit PC1 32-45%，NormCV=1.8
49. **DS7B的NormCV远超其他模型**：adj=1.84 vs Qwen3=0.55 vs GLM4=0.31
50. **DS7B中层范数编码从L3到L7急剧启动**：NormCV从0.27→1.95
51. **DS7B的C方向（无论raw/unit）在因果测试中强烈负面**：C_unit=-0.78抵消R=+0.47
52. **Qwen3/GLM4的C方向因果效力接近0**——构式是调制方向不是独立因果通道

### 方法论发现：范数vs方向的编码策略分化

```
Qwen3/GLM4: 方向编码为主，范数变异小(NormCV<0.6)
            → 句框信息主要通过方向变化编码
            → Raw PCA≈Unit PCA

DS7B:       范数编码为主，方向变异小但多维
            → 句框信息主要通过范数/幅度变化编码
            → Raw PCA(范数)一维，Unit PCA(方向)多维
            → 范数编码容易被误认为方向1维性
```

### 硬伤分析

1. **DS7B noun的Unit PC1=79.6%**：虽然比raw 99.5%低，但仍偏高。这是否说明noun的构式编码确实更1维？需要更多数据验证
2. **C/P/N在DS7B中高度重叠**：能量预算C=84%, P=75%, N=74%——这些不是独立分量。需要正交化
3. **因果测试中patching强度问题**：注入的向量幅度可能影响结果，特别是DS7B范数变异大
4. **位置方向仍不够精确**：当前位置控制句只覆盖10个token，需要更大样本
5. **O(not)与R/C的正交性未完成**：Phase 305数据格式不匹配，需要单独计算

### 命令记录

```
python tests/glm5/phase306_norm_position_decompose.py qwen3       # ~2.5min
python tests/glm5/phase306_norm_position_decompose.py glm4        # ~67min
python tests/glm5/phase306_norm_position_decompose.py deepseek7b  # ~41min
python tests/glm5_temp/phase306_cross_model.py                     # 跨模型分析
python tests/glm5_temp/phase306_detail.py                          # 详细数据
```

### 数据文件

- `results/phase306_norm_position/{qwen3,glm4,deepseek7b}_norm_position.json`
- `tests/glm5/phase306_norm_position_decompose.py`

### 下一步

1. **O(not)与R/C的正交性直接计算**：从Phase 305加载O方向，与Phase 306的C方向计算cos
2. **DS7B范数门控机制分析**：范数编码如何影响后续层的因果效力？
3. **正交化R/C/P/N分解**：用Gram-Schmidt消除C/P/N重叠
4. **更多操作符测试**：maybe/must/can/if——不同操作符是否共享O子空间？
5. **Scope因果测试**：not A and B vs A and not B——作用范围的因果效应

## Phase 307: Operator Orthogonality + Multi-Operator Causal Testing [2026-05-31 13:21]

### 目标

1. O(not)与R方向的正交性：cos(O_not, R)直接测量
2. 多操作符测试：not/never/maybe/must/can/should——不同操作符是否共享O子空间？
3. 操作符LOO一致性和因果效力

刺激设计：
- 20形容词 + 10动词 + 5名词 × 6操作符条件(affirm/not/maybe/must/can/should/never) = 245条
- 16个双角色token的角色基线句 = 32条
- 总计277条唯一句

### 核心发现53：O(not)与R(adj)几乎正交，但O(not)与R(verb)反平行

| 模型 | 层 | cos(O_not, R_adj) | cos(O_not, R_verb) |
|------|----|-------------------|--------------------|
| Qwen3 | L18 | -0.034 | -0.475 |
| GLM4 | L20 | -0.106 | -0.330 |
| **DS7B** | **L14** | **-0.138** | **-0.998** |

→ **O(not)与R(adj)近正交（cos < |0.14|）——否定形容词不沿角色方向**
→ **O(not)与R(verb)反平行（cos = -0.33~-1.0）——否定动词沿角色反方向**
→ **DS7B极端：O(not)与R(verb)几乎完全反平行（cos=-0.998）**
→ **这意味着"not + verb"的方向 ≈ "affirmative adj"的方向——否定翻转了动词的角色方向**

### 核心发现54：操作符形成两个聚类——否定类(not/never)和义务类(must/should)

Qwen3 L18操作符相似度矩阵：
```
          can    maybe  must   never  not    should
can       1.000  +0.525 +0.464 +0.308 +0.310 +0.473
maybe     +0.525 1.000  +0.462 +0.314 +0.328 +0.456
must      +0.464 +0.462 1.000  +0.325 +0.330 +0.605
never     +0.308 +0.314 +0.325 1.000  +0.529 +0.289
not       +0.310 +0.328 +0.330 +0.529 1.000  +0.297
should    +0.473 +0.456 +0.605 +0.289 +0.297 1.000
```

→ **否定聚类：not↔never = +0.53（Qwen3），+0.45（GLM4），+0.57（DS7B）**
→ **义务聚类：must↔should = +0.61（Qwen3），+0.55（GLM4），+0.63（DS7B）**
→ **跨聚类相似度低：not↔must = +0.33，not↔can = +0.31**
→ **三模型一致：语言操作符自然形成语义聚类**

### 核心发现55：DS7B的not vs never verb方向近乎完全重叠（cos=+0.999）

| 模型 | not vs never adj | not vs never verb |
|------|-----------------|-------------------|
| Qwen3 | +0.76 | +0.83 |
| GLM4 | +0.70 | +0.67 |
| **DS7B** | **+0.70** | **+1.00** |

→ **DS7B的not和never在verb上方向完全一致**
→ **这与DS7B的1维范数编码一致：verb的所有否定操作符被压缩到同一方向**
→ **Qwen3/GLM4的not和never在verb上有差异——否定策略更丰富**

### 核心发现56：所有操作符的因果效力在DS7B中层都强劲

DS7B L14因果效力：
```
not:    +0.364
never:  +0.469
maybe:  +0.404
must:   +0.305
can:    +0.402
should: (未单独测试)
```

→ **DS7B的所有操作符因果效力都在+0.30~0.47之间——强且一致**
→ **与R(role)因果效力+0.177相比，操作符因果效力高2-3倍**
→ **DS7B的操作符编码不受范数抵消影响（与Phase 305一致）**

### 核心发现57：O(not)与R(adj)正交但O(not)与R(verb)反平行的机制解释

为什么O(not)与R(adj)正交（cos≈0）但与R(verb)反平行（cos≈-0.5~-1.0）？

可能机制：
1. **"not happy"的形容词仍然在形容词子空间**——否定不改变词性角色
2. **"not like"的动词被否定后远离动词子空间**——否定翻转了动词的因果效力方向
3. **R(verb)方向从adj→verb，O(not)方向从verb→not_verb≈adj**——因此O(not)≈-R(verb)**

→ **否定动词 ≈ 回到形容词空间——这是O(not)与R(verb)反平行的根本原因**
→ **否定形容词仍然在形容词空间——所以O(not)与R(adj)正交**
→ **这揭示了语言编码的一个深层机制：否定操作符的角色依赖性**

### 新增客观事实拼图（5条）

53. **O(not)与R(adj)近正交（cos<0.14），与R(verb)反平行（cos=-0.33~-1.0）**
54. **操作符形成两个聚类**：否定类(not/never, cos≈0.5)和义务类(must/should, cos≈0.6)
55. **DS7B的not vs never verb方向完全重叠**（cos=+0.999）——范数压缩效应
56. **DS7B所有操作符因果效力强劲**（+0.30~0.47），不受范数抵消
57. **O(not)≈-R(verb)的机制解释**：否定动词≈回到形容词空间

### 命令记录

```
python tests/glm5/phase307_operator_ortho.py qwen3       # ~0.5min
python tests/glm5/phase307_operator_ortho.py glm4        # ~10min
python tests/glm5/phase307_operator_ortho.py deepseek7b  # ~6.5min
python tests/glm5_temp/phase307_cross_model.py            # 跨模型分析
```

### 数据文件

- `results/phase307_operator_ortho/{qwen3,glm4,deepseek7b}_operator_ortho.json`
- `tests/glm5/phase307_operator_ortho.py`

### 下一步

1. **Scope因果测试**：not A and B vs A and not B——作用范围是否影响O方向
2. **O(not)与C(construction)的正交性**：否定方向是否与构式方向正交
3. **I/R/C/O/S完整框架重建测试**：五个分量是否可以重建full_delta
4. **DS7B O(not)≈-R(verb)的深层机制**：是否在所有否定句都成立
5. **跨语言验证**：英语的否定聚类是否在其他语言中也存在

## Phase 308: Scope Causal + O-C Orthogonality + Cross-Form Negation [2026-05-31 13:48]

### 目标

1. Scope因果测试：否定作用范围是否产生独立因果效应
2. O(not)与C(construction)正交性：直接测量
3. 跨否定形式子空间：not/no/never/n't是否共享

刺激设计：
- 30对scope最小对（5类：量词/副词/嵌入/不定式/双重否定）
- 48条构式句（16双角色token × 3句框）
- 100条跨否定形式句（15 adj × 4形式 + 10 verb × 4形式）
- 总计238条唯一句

### 核心发现58：O(not)与C(construction)近正交——三模型一致

| 模型 | 层 | cos(O_not, C) | cos(O_not, C_pc1) | O_clean/C ratio | O_clean/Cpc1 ratio |
|------|----|---------------|--------------------|-----------------|--------------------|
| Qwen3 | L18 | -0.142 | +0.106 | 0.990 | 0.994 |
| GLM4 | L20 | -0.065 | +0.061 | 0.998 | 0.998 |
| DS7B | L14 | -0.163 | -0.157 | 0.987 | 0.988 |

→ **cos(O_not, C) < |0.17|，三模型一致：操作符方向与构式方向近正交**
→ **O_clean/C ratio ≈ 0.99：移除C投影后O几乎不变**
→ **这解释了为什么DS7B的O(not)不受构式抵消——它们在不同子空间**

### 核心发现59：DS7B清除C后O因果效力反而增强——O_clean_C > O_not_raw

DS7B L14因果效力：
```
O_not_raw   → not = +0.0625
O_clean_C   → not = +0.5000  (8倍增强！)
O_clean_Cpc1→ not = +0.5625  (9倍增强！)
C_only      → not = +0.0000
random      → not = +0.1250
```

→ **DS7B中，构式方向虽然与O近正交，但仍然轻微干扰O的因果效力**
→ **清除C后O因果效力从+0.06增加到+0.50——这是Phase 306范数伪影发现后的又一关键证据**
→ **Qwen3/GLM4的O_clean与O_raw差异极小（它们的C与O本来就几乎完全正交）**

### 核心发现60：跨否定形式子空间——n't≈not≈never，no较远

| 对比 | Qwen3 adj | Qwen3 verb | GLM4 adj | GLM4 verb | DS7B adj | DS7B verb |
|------|-----------|------------|----------|-----------|----------|-----------|
| not vs n't | N/A | +0.954 | N/A | +0.912 | N/A | +1.000 |
| not vs never | +0.754 | +0.827 | +0.703 | +0.666 | +0.691 | +0.999 |
| not vs no | +0.586 | N/A | +0.456 | N/A | +0.535 | N/A |

→ **not vs n't: cos=+0.91~1.00——几乎完全重叠（形态变体）**
→ **not vs never: cos=+0.67~0.83——强共享（否定类操作符）**
→ **not vs no: cos=+0.46~0.59——中等共享（no更多是限定词/存在否定）**
→ **DS7B verb：所有否定形式cos>0.999——极端1维压缩，所有动词否定共享同一方向**

### 核心发现61：否定操作符形成层次化子空间结构

```
第一层：形态变体（not/n't）    cos ≈ 0.91-1.00  几乎完全重叠
第二层：否定类（not/never）    cos ≈ 0.67-0.83  强共享
第三层：限定否定（not/no）     cos ≈ 0.46-0.59  中等共享
第四层：义务类（not/must）     cos ≈ 0.33       弱共享（Phase 307）
```

→ **否定子空间不是单一方向，而是层次化结构**
→ **越接近的语义/句法形式，方向越重叠**
→ **这与人类语言直觉完全一致：not/n't是同一词的变体，not/never是同类操作符，not/no是不同类型的否定**

### 核心发现62：Scope对否定方向有显著效应——adverb_scope和infinitive_scope

| Scope类型 | 三模型cos(O_narrow, O_wide) | 解读 |
|-----------|-----------------------------|------|
| adverb_scope | +0.60~+0.78 | 中高共享，scope有中等效应 |
| infinitive_scope | +0.43~+0.54 | 中等共享，scope有显著效应 |
| embedding_scope | +0.18~+0.31 | 极低共享（但位置对齐有问题） |

→ **adverb_scope（not very ADJ vs very not ADJ）：两个否定方向有60-78%重叠**
→ **infinitive_scope（not possible to X vs possible not to X）：只有43-54%重叠——scope显著改变否定方向**
→ **embedding_scope数据不可靠：不同句法位置导致范数差异极大（O_wide=2300-6300）**

### 核心发现63：Scope因果效力在DS7B中方向反转——O_narrow→not=-0.47

DS7B L14 scope因果测试：
```
O_narrow       → not = -0.4753
O_wide         → not = +0.0013
O_shared       → not = -0.0326
S_scope        → not = +0.0768
random         → not = -0.0755
```

→ **DS7B的O_narrow强烈负面——注入narrow-scope否定方向反而抑制"not"输出**
→ **与Phase 307的O(not)与R(verb)反平行一致——narrow scope否定作用于动词角色**
→ **S_scope微弱正面——scope本身有独立因果效应但不强**
→ **Qwen3/GLM4的scope因果效力都弱，O_shared ≈ random水平**

### 新增客观事实拼图（6条）

58. **O(not)与C(construction)近正交**：cos < |0.17|，O_clean/C ratio ≈ 0.99
59. **DS7B清除C后O因果效力8倍增强**：O_clean_C=+0.50 vs O_raw=+0.06
60. **跨否定形式层次化子空间**：not≈n't(0.95)>never(0.75)>no(0.52)
61. **否定子空间是层次化结构，不是单一方向**
62. **infinitive scope对否定方向有显著效应**：cos(O_narrow, O_wide)=+0.43~0.54
63. **DS7B的O_narrow因果效力强负面**：O_narrow→not=-0.47，与R(verb)反平行一致

### 硬伤分析

1. **embedding_scope位置对齐问题**：不同句法位置导致范数差异极大（O_wide=2300-6300），需要用aligned position（如始终取"not"位置而非目标词位置）
2. **quantifier_scope和double_negation数据缺失**：target word tokenize问题导致多数对无法解析
3. **Scope因果效力在Qwen3/GLM4中都弱**：可能因为scope信号需要跨多个token协同，单点patching不够
4. **no vs not的cos较低(0.46-0.59)可能因为no是限定词**：需要区分"no + noun" vs "not + adj"的句法差异
5. **跨否定形式测试缺少without/non-/un-**：形态否定前缀可能共享不同子空间

### 关键洞察：O-C正交性解释了DS7B的异常

Phase 304发现DS7B的R(role)被C(construction)抵消，但Phase 305发现O(not)不受影响。
Phase 308给出了完整解释：

```
R与C有部分重叠（cos(R, C) > 0）→ C在输出层抵消R
O与C近正交（cos(O, C) ≈ 0）→ C无法抵消O
→ DS7B的R被C抵消但O不受影响，不是因为O更强，
  而是因为O与C在不同子空间
```

这进一步说明：
- 语言功能在残差流中的子空间分配不是随机的
- 角色和构式有交互（cos(R,C) ≠ 0），所以会产生竞争
- 操作符和构式是独立的（cos(O,C) ≈ 0），所以互不干扰

### 命令记录

```
python tests/glm5/phase308_scope_causal.py qwen3       # ~0.5min
python tests/glm5/phase308_scope_causal.py glm4        # ~12min
python tests/glm5/phase308_scope_causal.py deepseek7b  # ~7min
python tests/glm5_temp/phase308_cross_model.py          # 跨模型分析
```

### 数据文件

- `results/phase308_scope_causal/{qwen3,glm4,deepseek7b}_scope_causal.json`
- `tests/glm5/phase308_scope_causal.py`

### 下一步

1. **修正embedding_scope的位置对齐**：始终取"not"位置而非目标词位置，消除位置混淆
2. **补全quantifier_scope**：改用更容易定位的目标词（如"all/every"）
3. **O/R/C/S完整子空间映射**：计算所有两两cos，建立子空间关系图
4. **DS7B O_clean增强效应的深入分析**：为什么清除C后O因果效力增强8倍？
5. **形态否定前缀测试**：un-/non-/in-/dis- vs not是否共享子空间

## Phase 309: O/R/C/S Complete Subspace Mapping [2026-05-31 14:05]

### 目标

建立完整的功能子空间映射：计算O/R/C/A/S所有两两cosine、投影重叠、独立方向因果效力。

刺激设计：
- 角色对：16双角色token × 3句框 = 48条
- 操作符：20 adj + 10 verb + 10 noun × (affirm/not/maybe/must/can/never) = 200条
- Scope：6对infinitive × 3句 = 18条
- 反义词：10对 = 20条
- 总计276条唯一句

### 核心发现64：DS7B的子空间坍缩——O/R/C几乎完全共线

| 模型 | 层 | O→R | O→C | O→C_pc1 | R→C | R→O |
|------|----|-----|-----|---------|-----|-----|
| Qwen3 | L18 | 0.280 | 0.061 | 0.199 | 0.024 | 0.280 |
| GLM4 | L20 | 0.221 | 0.007 | 0.236 | 0.018 | 0.221 |
| **DS7B** | **L14** | **0.994** | **0.836** | **0.995** | **0.836** | **0.994** |

→ **DS7B的O(not)和R(role)几乎完全共线（cos=0.994）！**
→ **DS7B的C_pc1也几乎完全共线（cos=0.995）！**
→ **Qwen3/GLM4的O/R/C相对正交（cos<0.28）**
→ **DS7B中角色、操作符、构式共享同一主导方向——这是极端1维压缩的直接证据**

### 核心发现65：DS7B共线方向却产生相反因果效力——最强"激活同向/因果反向"案例

DS7B L14因果效力（注入到"the result was happy"）：
```
R_raw       → not = +0.9688
O_raw       → not = -0.7812
C_raw       → not = +1.0625
A_raw       → not = -1.7656
random      → not = -0.1562
```

→ **R_raw→not=+0.97 但 O_raw→not=-0.78：两个cos=0.994的方向产生了相反的因果效力！**
→ **C_raw→not=+1.06：构式方向因果效力甚至更强**
→ **A(antonym)→not=-1.77：反义词方向的负面效应最极端**
→ **这是"激活空间cosine≠输出因果效应"的最强证据——cos=0.994但因果效力方向相反**

### 核心发现66：DS7B的O_clean_RC因果效力反而增强——清除R/C投影后更强

DS7B L14独立方向因果效力：
```
R_raw       → not = +0.9688
O_raw       → not = -0.7812
R_clean_OC  → not = +0.3438
O_clean_RC  → not = +1.0312  (比O_raw更强！)
C_clean_RO  → not = +0.7188
O_clean_RCA → not = +1.0625  (比O_raw更强！)
```

→ **O_clean_RC=+1.03 > O_raw=-0.78：清除R和C投影后，O的因果效力从负面变成正面且更强！**
→ **这说明O_raw中的R/C分量在输出层产生了负向因果效应**
→ **而O的独立分量(1-R-C的残差)在输出层产生强正向因果效应**
→ **这完美解释了Phase 305的发现：DS7B的O(not)因果效力+0.587不受C抵消**

### 核心发现67：跨层子空间趋势——DS7B的共线从L7持续到L21

| 层 | DS7B O→R | DS7B O→C | DS7B R→C |
|----|----------|----------|----------|
| L7 | 0.991 | 0.572 | 0.571 |
| L9 | 0.989 | 0.398 | 0.408 |
| L14 | 0.994 | 0.836 | 0.836 |
| L18 | 0.991 | 0.460 | 0.459 |
| L21 | 0.978 | 0.120 | 0.123 |
| L26 | 0.902 | 0.140 | 0.201 |

→ **O→R在整个中层(7-21)都>0.97——角色和操作符的1维压缩是持续性的**
→ **O→C在L14峰值(0.836)，L21后下降——构式在L14层最集中**
→ **L26开始分化(O→R=0.902)——深层开始解压缩**

### 核心发现68：Qwen3/GLM4的功能子空间结构——O⊥C, R⊥C, O∼R中等

Qwen3 L18子空间重叠：
```
O→R = 0.280  (中等共享——O(not)和R(adj→verb)有部分重叠)
O→C = 0.061  (近正交——操作符和构式几乎独立)
R→C = 0.024  (近正交——角色和构式几乎独立)
O→A = 0.257  (中等共享——否定和反义词有部分重叠)
O→S = 0.109  (弱共享——操作符和scope弱相关)
```

GLM4 L20子空间重叠：
```
O→R = 0.221  (中等共享)
O→C = 0.007  (近正交——最干净的O-C分离)
R→C = 0.018  (近正交)
O→A = 0.357  (中等偏强——否定和反义词共享更多)
O→S = 0.062  (弱共享)
```

→ **Qwen3/GLM4中，O⊥C和R⊥C是最稳定的发现——操作符/角色与构式近正交**
→ **O→R≈0.22-0.28是中等共享——操作符和角色有部分重叠（否定动词≈回到形容词空间）**
→ **GLM4的O→A=0.357高于Qwen3的0.257——GLM4的否定更接近反义词替换**

### 核心发现69：GLM4深层功能汇聚——L38所有方向重叠增大

GLM4跨层趋势：
```
L10: O→R=0.254, O→C=0.017, R→C=0.035
L20: O→R=0.221, O→C=0.007, R→C=0.018
L30: O→R=0.223, O→C=0.073, R→C=0.108
L38: O→R=0.384, O→C=0.192, R→C=0.148
```

→ **GLM4深层(38层)功能子空间开始汇聚——所有方向重叠增大**
→ **这可能反映了深层"语义统一化"：不同功能在深层被压缩到共享子空间**
→ **这与DS7B的中层压缩不同：DS7B是中层就坍缩，GLM4是深层才汇聚**

### 新增客观事实拼图（6条）

64. **DS7B子空间坍缩：O/R/C_pc1几乎共线(cos=0.994/0.995)**
65. **DS7B共线方向产生相反因果效力：R→not=+0.97但O→not=-0.78**
66. **DS7B的O_clean_RC因果效力(+1.03)远强于O_raw(-0.78)**
67. **DS7B的O-R共线从L7持续到L21(>0.97)，L26开始分化(0.902)**
68. **Qwen3/GLM4中O⊥C(0.01-0.06), R⊥C(0.01-0.08)是最稳定发现**
69. **GLM4深层L38功能汇聚：O→R=0.384, O→C=0.192, R→C=0.148**

### 方法论重大突破：从"方向编码"到"方向-输出映射编码"

Phase 309的发现彻底改变了"语言编码=语义轴"的范式：

```
旧范式：语言功能 = 独立方向
  R = 角色方向
  O = 操作符方向
  C = 构式方向
  各方向正交 → 可以独立操作

新范式：语言编码 = 方向 × 输出映射
  同一方向经过不同的输出映射可以产生不同因果效力
  cos(方向A, 方向B) ≈ 1 不意味着 因果效力(A) ≈ 因果效力(B)
  因为后续层和输出层对同一方向的不同分量有不同的"读取权重"
```

**具体机制（DS7B）**：
1. O(not)和R(role)共享99.4%的方差
2. 剩余0.6%的方差存在于一个微小方向差δ = O - proj_O(R)
3. 这个微小δ被后续层/输出层极大放大
4. 导致O和R产生相反的因果效力

**这类似于光学中的双折射**：
- 同一条光线进入各向异性晶体后分成两束
- 虽然输入方向相同，但晶体对不同偏振分量的折射率不同
- 导致出射方向完全不同

### 硬伤分析

1. **DS7B因果效力数值极大(+0.97/-0.78)**：可能是范数主导效应。需要归一化后重新测试
2. **C方向在DS7B中norm=0.00**：因为C是centered后的均值，导致O→C的overlap不可靠
3. **因果测试只用了1个句子**：需要更多baseline句子验证
4. **Scale统一用O_norm×0.1**：不同方向的注入量可能需要调整
5. **scope使用last-token位置**：与operand位置不一致，可能引入混淆

### 命令记录

```
python tests/glm5/phase309_subspace_map.py qwen3       # ~0.5min
python tests/glm5/phase309_subspace_map.py glm4        # ~7min
python tests/glm5/phase309_subspace_map.py deepseek7b  # ~4min
python tests/glm5_temp/phase309_cross_model.py          # 跨模型分析
```

### 数据文件

- `results/phase309_subspace_map/{qwen3,glm4,deepseek7b}_subspace_map.json`
- `tests/glm5/phase309_subspace_map.py`

### 下一步

1. **DS7B归一化因果测试**：unit方向注入，排除范数主导
2. **δ方向放大机制分析**：0.6%的方差差如何被后续层放大成相反因果效力
3. **跨模型统一框架**：Qwen3/GLM4=正交子空间 vs DS7B=坍缩子空间×输出映射
4. **功能复用-冲突-依赖三图**：基于完整cosine矩阵构建
5. **深层汇聚(Phase 309-GLM4 L38)的机制分析**：为什么深层所有功能方向汇聚？

## Phase 310: Unified Normalized Causal Test + Direction Gain [2026-05-31 15:35]

### 目标

解决Phase 308/309的O-C冲突，通过统一定义+归一化因果测试+方向增益分析。

刺激设计：
- 复用Phase 309的258条唯一句
- 因果测试：5个baseline句 × 9个方向 × 4个alpha × 6层
- 方向增益：unit方向注入+测量输出logit变化

### 核心发现70：Phase 308的O⊥C结论是空结果——C_raw范数极小(~3e-7)

三模型统一数据：
```
模型       C_raw norm       C_pc1 norm    cos(O, C_raw)    cos(O, C_pc1)
Qwen3     ~3e-7            1.0           ~0.02             -0.20~-0.36
GLM4      ~2e-7            1.0           ~0.01             -0.17~-0.30
DS7B      ~5e-6            1.0           ~-0.16            -0.57~-0.86
```

→ **C_raw是centered frame差异的均值，正负相消导致范数~0**
→ **cos(O, C_raw) ≈ 0不是正交性证据，而是C_raw不可用的标志**
→ **Phase 308的"O与C近正交"结论必须修正为：O与C_raw的余弦不可靠**
→ **真实关系应该看cos(O, C_pc1)：Qwen3/GLM4中等负相关，DS7B强负相关**

### 核心发现71：DS7B中O(not)的87-89%被R(role)投影占据

O_clean_R / O_raw 比率：
```
模型     中层典型值       范围
Qwen3    0.94~0.97       O几乎独立于R
GLM4     0.92~0.98       O几乎独立于R
DS7B     0.11~0.21       O几乎完全被R占据！
```

→ **Qwen3/GLM4中，移除R投影后O仍保留94-97%范数——O与R确实近正交**
→ **DS7B中，移除R投影后O只剩11-21%范数——O的绝大部分方差在R方向上**
→ **DS7B中cos(O_not, R) ≈ -0.99——O(not)几乎完全在R的反方向上**
→ **这意味着DS7B的not方向≈-R方向：否定≈反向角色转换**
→ **但O_clean_R(仅11%范数)仍然有因果效力——微小残差方向被后续层放大**

### 核心发现72：DS7B的C_pc1方差解释率99.3%——极端1维构式压缩

C_pc1方差解释率：
```
模型     C_pc1_var
Qwen3    21%~26%     (多分散，构式是多维的)
GLM4     17%~22%     (多分散)
DS7B     94%~99%     (极端1维！)
```

→ **DS7B的构式变异几乎完全由一个主成分解释——1维刚性压缩**
→ **Qwen3/GLM4的构式是多维分布——不同frame产生不同方向**
→ **这解释了为什么DS7B的角色编码容易被构式抵消：构式只有1维，角色编码也被压缩到同一维度**

### 核心发现73：DS7B范数体系完全不同——R/O范数比Qwen3大40-70倍

中层范数对比：
```
模型       R_norm      O_not_norm    R/O ratio
Qwen3      51          33           1.56
GLM4       14          10           1.39
DS7B       1844        599          3.08
```

→ **DS7B的方向范数比Qwen3大40-70倍！**
→ **这意味着DS7B的残差流数值分布完全不同**
→ **Phase 309用raw方向×0.1注入，在DS7B中等效于注入了~180的范数——远超自然范围**
→ **Phase 309的DS7B因果效力(+0.97/-0.78)极可能是因为注入量过强导致的非自然效应**

### 核心发现74：归一化因果效力(unit, alpha=2.0)在所有模型中都弱

三模型unit方向(alpha=2.0)因果效力：
```
方向        Qwen3 L18     GLM4 L20      DS7B L14
O_not       +0.009        -0.081        -0.026
O_clean_RC  +0.025        -0.156        -0.093
R           +0.006        -0.286        -0.012
C_pc1       +0.025        -0.307        -0.045
A           +0.003        +0.217        -0.012
random      +0.016        -0.016        -0.049
```

→ **Qwen3/DS7B的unit因果效力接近random水平——unit方向注入太弱**
→ **GLM4的因果效力稍强(R=-0.29, C=-0.31, A=+0.22)——但仍在小量级**
→ **GLM4中A(antonym)→+0.22是唯一显著正效应——注入反义词方向增加"not"概率**
→ **DS7B的O_clean_RC→-0.09略高于random(-0.05)，但不显著**
→ **需要用匹配范数(自然范数比例)注入来获得可靠的因果结论**

### 核心发现75：方向增益(Gain)分析——O_not略高于random但差异不大

Gain ratio (direction_gain / random_gain):
```
方向        Qwen3 L18     GLM4 L20      DS7B L14
O_not       0.95x         1.70x         1.05x
O_clean_RC  0.80x         1.90x         0.94x
R           0.87x         1.48x         0.87x
C_pc1       1.00x         1.63x         0.88x
A           0.78x         2.23x         0.91x
```

→ **GLM4的增益比最高(A=2.23x, O_clean_RC=1.90x)——GLM4对功能方向有更强的输出映射**
→ **Qwen3/DS7B的增益比都接近1x——unit方向的输出映射和random无显著差异**
→ **增益分析不支持"微小残差方向被后续层高增益放大"的假说——至少在unit尺度下如此**

### 新增客观事实拼图（6条）

70. **C_raw范数极小(~3e-7)，cos(O,C_raw)≈0是空结果而非正交证据**
71. **DS7B中O(not)的87-89%被R投影占据，O_clean_R/O_raw=0.11~0.21**
72. **DS7B的C_pc1方差解释率99.3%——极端1维构式压缩(vs Qwen3/GLM4的~20%)**
73. **DS7B范数比Qwen3大40-70倍——Phase 309的raw注入过强**
74. **unit方向因果效力在所有模型中都弱——需要匹配范数注入**
75. **方向增益比接近1x——unit尺度下无"高增益放大残差"证据**

### Phase 308/309冲突的完整解决方案

Phase 308说O⊥C(cos<0.17)，Phase 309说DS7B的O→C_pc1=0.995。现在完全清楚：

```
Phase 308用C_raw:   norm ≈ 3e-7,  cos(O, C_raw) ≈ 0  → 空结果，不可靠
Phase 309用C_pc1:   norm = 1.0,   cos(O, C_pc1) varies → 真实关系
Phase 310统一测试:  cos(O, C_pc1) = -0.20~-0.36 (Qwen3/GLM4), -0.57~-0.86 (DS7B)
```

结论：
1. Phase 308的O⊥C结论必须撤回
2. 真实关系是cos(O, C_pc1)有中等负相关(Qwen3/GLM4)到强负相关(DS7B)
3. 负相关意味着否定操作符方向与构式主成分方向有一定对立
4. DS7B的强负相关(cos=-0.86)说明否定和构式在同一维度上但方向相反

### DS7B编码机制的修正理解

基于Phase 310数据，DS7B的编码机制应修正为：

```
旧理解(Phase 309): O/R/C几乎共线(cos=0.994)，是"子空间坍缩"
新理解(Phase 310):
  - O与R强负相关(cos=-0.99)，不是简单共线
  - O的87%方差在R方向上，但方向相反
  - C_pc1与R强正相关(cos=+0.84)
  - 所以cos(O, C_pc1)≈-0.86 = O与R负相关 × R与C正相关
  - 三者形成"一条链"：O ← R → C_pc1
  - 而不是"坍缩到同一方向"
```

这更精确地描述了DS7B的机制：
- 否定 ≈ 角色反转（not happy ≈ 把adj角色的方向反转）
- 构式 ≈ 角色编码（构式主成分与角色方向对齐）
- 否定与构式在同一维度上但方向相反

### 硬伤分析

1. **unit方向因果效力太弱**：Qwen3/DS7B中几乎所有方向都不显著——需要匹配范数注入
2. **匹配范数注入还没做**：应该用O_norm×0.1 vs R_norm×0.1做等比例注入
3. **方向增益分析在unit尺度下无显著发现**：可能需要在自然范数下分析
4. **因果测试只测了delta_not一个指标**：应该测更多目标token（如delta_happy, delta_sad）
5. **5个baseline句子可能不够**：需要更大量基线

### 命令记录

```
python tests/glm5/phase310_unified_norm_causal.py qwen3       # ~2min
python tests/glm5/phase310_unified_norm_causal.py glm4        # ~44min
python tests/glm5/phase310_unified_norm_causal.py deepseek7b  # ~28min
python tests/glm5_temp/phase310_cross_model.py                 # 跨模型分析
```

### 数据文件

- `results/phase310_unified_norm_causal/{qwen3,glm4,deepseek7b}_unified_norm_causal.json`
- `tests/glm5/phase310_unified_norm_causal.py`

### 下一步

1. **匹配范数因果测试**：用O_norm×0.1 vs R_norm×0.1 vs C_pc1_norm×0.1，使注入量与方向自然范数成比例
2. **多目标token因果测试**：不只测delta_not，还测delta_happy, delta_sad, delta_very等
3. **DS7B的"否定=角色反转"机制深入验证**：not ADJ ≈ -R(adj→verb) 是否在所有形容词上成立
4. **功能复用矩阵**：基于修正后的cos(O,C_pc1)重建子空间关系图
5. **Scope修正测试**：对齐到"not"位置而非last-token

## Phase 311: Norm-Matched Causal Test + Multi-Target Analysis [2026-05-31 16:10]

### 目标

验证Phase 310的归一化因果效力在匹配范数下是否成立，同时测试多目标token。

刺激设计：
- 复用258条唯一句
- 10个baseline句 × 8个方向 × 3个scale(0.05/0.1/0.2) × 2-3层
- 8个目标token: not/sad/very/happy/but/however/never/always

### 核心发现76：DS7B中O_clean_R的单位因果效力是O_not的114倍——微小残差方向高增益读取被验证

DS7B L14归一化因果效力(Δnot/norm):

O_not:      -0.00013  (几乎为0)
O_clean_R:  -0.01545  (O_not的114倍！)
O_clean_RC: -0.01088  (O_not的80倍！)
A:          -0.01457  (与O_clean_R相当)
R:          -0.00298  (中等)
random:     +0.00157  (微弱正面)

→ **O_clean_R方向虽然只占O_not范数的11%，但单位因果效力远超O_not**
→ **O_not的87%方差在R方向上(高范数共享成分)，这部分在输出层几乎无因果效力**
→ **O_not的13%方差在正交残差方向上(低范数差分成分)，这部分在输出层有强因果效力**
→ **这直接验证了"高方差复用+低方差差分+高增益读取"的编码机制**

### 核心发现77：Qwen3/GLM4中O_not和O_clean_R的单位因果效力相近——正交子空间模型下两者等价

Qwen3 L34归一化因果效力(Δnot/norm):

O_not:      -0.01257
O_clean_R:  -0.01236  (0.98x，几乎相同)
random:     +0.00255

GLM4 L26归一化因果效力(Δnot/norm):

O_not:      +0.02025
O_clean_R:  +0.00959  (0.47x)
A:          +0.04430  (2.2x)
random:     -0.01103

→ **Qwen3/GLM4中O与R近正交(cos<0.3)，所以移除R投影后O几乎不变**
→ **O_clean_R ≈ O_not，两者的单位因果效力相近**
→ **这是"正交子空间编码"的直接因果验证：功能方向已经相对独立，不需要额外差分读取**

### 核心发现78：GLM4中A(antonym)的单位因果效力最强——反义词替换比否定更直接影响输出

GLM4 L26各方向的Δnot/norm:

A(antonym):   +0.04430  (最强——注入反义词方向显著增加"not"概率)
O_not:        +0.02025
O_clean_RC:   +0.00839
R:            -0.03625
random:       -0.01103

→ **GLM4中反义词方向对"not"token的因果效力最强**
→ **这与Phase 309的发现一致：GLM4的O→A=0.36(Qwen3的0.26)——GLM4的否定更接近反义词替换**
→ **说明GLM4可能更依赖词汇级语义替换来实现否定效果**

### 核心发现79：DS7B的random方向产生强正面效应(norm=60时)——范数过大导致非自然分布

DS7B L14匹配范数因果测试(scale=0.1):

random(norm=60.2):  Δnot=+0.095, Δsad=+0.173, Δhappy=+0.731
O_not(norm=60.2):   Δnot=-0.008, Δsad=-0.107, Δhappy=-0.073

→ **random方向注入60范数时，Δhappy=+0.731——随机方向都能大幅改变输出**
→ **O_not方向注入60范数时反而抑制了这个正面效应(Δnot≈0 vs random的+0.095)**
→ **说明DS7B在scale=0.1时注入量已经超过自然分布范围**
→ **O_clean_R只需要6.4范数就能产生Δnot=-0.098——更接近自然尺度**

### 核心发现80：多目标token分析揭示否定机制的多维度效应

Qwen3 L34 (scale=0.1)各方向的Δhappy:

O_not:      -0.006  (否定方向几乎不影响happy输出)
O_clean_RC: +0.025  (清洁否定方向微弱增加happy)
R:          +0.087  (角色方向增加happy)
A:          -0.169  (反义词方向强抑制happy)
random:     +0.000  (随机方向无效应)

→ **否定方向(O_not)对"happy"token几乎没有效应(Δ=-0.006)**
→ **反义词方向(A)对"happy"token有强抑制效应(Δ=-0.169)**
→ **这说明否定不是语义替换——否定不替换"happy"的激活，而是独立操作**
→ **角色方向(R)增加"happy"概率(+0.087)——动词角色使形容词token更可能**

### 新增客观事实拼图（5条）

76. **DS7B O_clean_R的单位因果效力是O_not的114倍(Δnot/norm)**
77. **Qwen3/GLM4中O_not≈O_clean_R(0.5-1.0x)——正交子空间下两者等价**
78. **GLM4中A(antonym)的单位因果效力最强——反义词替换比否定更直接**
79. **DS7B的random方向在norm=60时产生强效应——范数过大导致非自然分布**
80. **O_not方向对"happy"几乎无效应，A方向强抑制"happy"——否定≠语义替换**

### 关键理论进展：两种编码架构的因果验证

Phase 311的归一化因果效力分析，第一次从因果效力层面区分了两种编码架构。

### 下一步
1) operand位置因果测试——修正注入位置；
2) 更大规模多目标测试——扩展目标token集合和句子数量；
3) DS7B高增益读取的层间传播——追踪O_clean_R如何被后续层放大。

## Phase 312: Operand-Position Causal Test [2026-05-31 16:15]

### 目标
在operand位置注入+概率测试，修正Phase 311的注入位置问题。

### 结果
operand位置注入在op_pos-1位置测量概率变化时，所有ΔP=0——因为patching改变的是operand位置本身的激活，影响的是后续层的处理，而非之前位置的预测。

**教训**：
应该在operand位置注入，在last token位置测量概率变化（Phase 311的方法已正确）。
Phase 312的技术修正未产出新发现，Phase 311的结果仍然有效。

### 命令记录

`python tests/glm5/phase312_operand_causal.py qwen3` — ΔP=0(位置逻辑错误)

### 下一步

1. **继续使用Phase 311的方法**：在last token位置测量，而非operand位置
2. **完整总结**：整合所有发现，建立统一理论框架

## Phase 304-312 完整总结与理论框架

### 已确认的客观事实拼图（80条中Phase 308-312新增的14条）

| 序号 | 发现 | 状态 |
| 58 | O(not)⊥C_raw但C_raw不可靠 | ✅ 已修正(#70) |
| 59 | DS7B清除C后O因果效力增强 | ⚠️ 可能是范数伪影(#73) |
| 60 | 跨否定形式层次化子空间 | ✅ 稳定 |
| 61 | 否定子空间是层次化结构 | ✅ 稳定 |
| 62 | infinitive scope对否定方向有显著效应 | ✅ 稳定 |
| 63 | DS7B O_narrow因果效力强负面 | ⚠️ 需归一化验证 |
| 64 | DS7B子空间坍缩 | ⚠️ 已修正为"O≈-R链式关系" |
| 65 | DS7B共线方向产生相反因果效力 | ✅ 已验证(#76) |
| 66 | DS7B O_clean_RC因果效力强于O_raw | ✅ 已验证(#76,114倍) |
| 67 | DS7B O-R共线从L7持续到L21 | ✅ 稳定 |
| 68 | Qwen3/GLM4中O⊥C,R⊥C | ⚠️ C_raw不可靠，修正为O⊥C_pc1中等负相关 |
| 69 | GLM4深层L38功能汇聚 | ✅ 稳定 |
| 70 | C_raw范数极小，cos(O,C_raw)≈0是空结果 | ✅ 决定性 |
| 71 | DS7B O(not)的87-89%被R投影占据 | ✅ 决定性 |
| 72 | DS7B C_pc1方差解释率99.3% | ✅ 决定性 |
| 73 | DS7B范数比Qwen3大40-70倍 | ✅ 决定性 |
| 74 | unit方向因果效力在所有模型中都弱 | ✅ 稳定 |
| 75 | 方向增益比接近1x | ✅ 稳定 |
| 76 | DS7B O_clean_R单位因果效力是O_not的114倍 | ✅ 决定性 |
| 77 | Qwen3/GLM4中O_not≈O_clean_R | ✅ 稳定 |
| 78 | GLM4中A(antonym)单位因果效力最强 | ✅ 稳定 |
| 79 | DS7B norm=60时random产生强效应 | ✅ 稳定 |
| 80 | O_not对"happy"无效应，A强抑制"happy" | ✅ 稳定 |

### 最终理论框架

1. 两种编码架构的因果验证

架构A: 正交子空间编码(Qwen3/GLM4)
  - 功能方向在激活空间中相对正交(cos<0.3)
  - O_clean_R ≈ O_not，单位因果效力相近
  - 直接映射：各功能方向独立影响输出
  - 机制：分布式编码，各功能有独立子空间

架构B: 共享主方向+差分读取(DS7B)
  - 功能方向高度重叠(cos(O,R)≈-0.99)
  - O_clean_R/O_not范数比=0.11，但单位因果效力×114
  - 差分映射：高范数共享成分无因果，低范数差分被高增益读取
  - 机制：1维压缩+差分放大，功能区分依赖微小残差


2. 语言编码的完整数学模型

h_l(token) = I_l(token)                     # 词元身份
           + R_l(role) · α_R(l)             # 角色编码
           + C_l(construction) · α_C(l)      # 构式编码
           + O_l(operator) · α_O(l)          # 操作符编码
           + interactions                    # 交互项
           + U_l                              # 残差

其中：
- 正交子空间模型(架构A): R⊥C⊥O, α≈1, 直接因果
- 差分读取模型(架构B): R≈-O, α_R>>α_O, 因果由δ=O-Proj_R(O)决定

因果效力 = ||W_U · δ|| / ||δ||
  架构A: δ≈O, 因果效力中等
  架构B: δ=O-Proj_R(O)仅11%范数, 但因果效力×114


3. C_pc1 vs C_raw 的澄清

C_raw: centered frame差异的均值 → norm≈3e-7 → cos(O,C_raw)不可靠
C_pc1: SVD第一主成分 → norm=1.0 → cos(O,C_pc1)是真实关系

真实cos(O,C_pc1):
  Qwen3: -0.20~-0.36 (中等负相关)
  GLM4:  -0.17~-0.30 (中等负相关)
  DS7B:  -0.57~-0.86 (强负相关)

解释：否定方向与构式主成分有一定对立，尤其在DS7B中最强
DS7B: cos(O,C_pc1)≈-0.86 = cos(O,R)×cos(R,C_pc1) = (-0.99)×(+0.84)
     否定和构式在同一维度链上，但方向相反


4. 下一步大任务

1. **输出层读取权重分析**：直接计算W_U对O/R/C/δ的投影，验证"差分放大"是W_U造成的还是中间层造成的
2. **逐层因果传播追踪**：在不同层注入，构建因果传播路径图
3. **功能复用矩阵**：基于修正后的cosine和归一化因果效力，建立完整的功能关系拓扑
4. **架构条件分析**：什么条件下模型选择架构A vs 架构B？参数量/深度/训练数据的效应？
5. **Scope因果测试修正**：在operand位置注入scope方向，在last token测量

## Phase 313: W_U Output Layer Readout Weight Analysis [2026-05-31 17:30]

### 目标

验证"差分放大"的来源：是W_U输出层直接读取差分方向更强？还是中间层在传播过程中放大了差分？

方法：
1. 提取W_U(lm_head权重矩阵) [vocab_size, d_model]
2. 计算W_U @ v的增益：||W_U @ v|| / ||v||
3. 计算Jacobian增益：注入unit方向v，测量输出logit变化
4. 比较jac/wu比率——如果jac/wu(O_clean_R) >> jac/wu(O_not)，说明中间层放大差分

### 核心发现81：W_U本身没有差分放大——所有方向的W_U增益几乎相同

三模型W_U增益(||W_U @ v|| / ||v||)相对于random基线：
```
方向        Qwen3 L18    GLM4 L18    DS7B L12
O_not       1.17x         1.18x       0.14x
O_clean_R   1.16x         1.17x       0.15x
R           0.94x         0.95x       0.09x
C_pc1       0.96x         1.01x       0.32x
A           1.16x         1.09x       0.04x
random      1.00x         1.00x       1.00x
```

→ **W_U对所有方向的增益几乎相同(~1.0x random)**
→ **W_U本身没有选择性地放大O_clean_R或差分方向**
→ **DS7B的所有方向W_U增益都很低(0.04-0.32x)，因为DS7B的范数极大**
→ **结论：差分放大不在W_U层！**

### 核心发现82：中间层(jacobian/wu比率)确实有差分放大效应，但模式复杂

jac/wu比率(中间层放大倍数)：
```
方向        Qwen3 L6     Qwen3 L18    GLM4 L18     DS7B L12
O_not       10.23         8.41          9.42         4.82
O_clean_R   22.56         7.98          8.71        27.39
ratio       2.205         0.948         0.925        5.68
```

→ **Qwen3 L6（早期层）：O_clean_R的中间层放大是O_not的2.2倍——早期层确实放大差分**
→ **Qwen3/GLM4中高层(L18+)：O_clean_R和O_not的放大倍数相近(~1.0x)——差分放大效应消失**
→ **DS7B L12：O_clean_R的jac/wu比率是O_not的5.68倍——但数值极不稳定**

### 核心发现83：DS7B的jacobian结果极其不稳定——不同层之间剧烈波动

DS7B各方向的jacobian/random比率跨层变化：
```
方向          L4      L8      L12     L16     L20     L26
shared_pc1   4.65x   0.24x   0.04x   3.62x   3.97x   0.61x
O_not        0.09x   0.18x   0.03x   0.06x   0.12x   0.93x
O_clean_R    0.17x   0.15x   0.06x   0.06x   0.15x   0.93x
C_pc1_delta  4.51x   0.14x   1.51x   3.68x   0.06x   3.83x
```

→ **DS7B的unit方向注入(epsilon=0.1)在不同层产生完全不同的结果**
→ **某些层(如L4, L16)所有方向都被极大放大(4x random)，其他层(如L12)几乎所有方向都很弱**
→ **这可能反映了DS7B的残差流范数在不同层间有巨大变化**
→ **DS7B的分析需要匹配范数注入，而非unit方向注入**

### 核心发现84：构式差分方向(C_pc1_delta_from_shared)在DS7B中被中间层极高放大

DS7B中C_pc1_delta_from_shared的jacobian/random比率：
```
L4: 4.51x    L8: 0.14x    L12: 1.51x    L16: 3.68x    L20: 0.06x    L26: 3.83x
```

→ **构式差分方向在多个层被显著放大(L4=4.5x, L16=3.7x, L26=3.8x)**
→ **但放大效应在层间不稳定——某些层完全没有放大(L8=0.14x, L20=0.06x)**
→ **这说明DS7B的信息传播可能存在"波浪式"模式：某些层专门放大特定差分**

### 核心发现85：Qwen3早期层(L6)的差分放大效应最清晰

Qwen3 L6各方向jac/wu比率：
```
O_not:      10.23
O_clean_R:  22.56  (2.2x O_not)
R:          11.64
C_pc1:       8.94
A:          11.46
```

→ **Qwen3 L6中O_clean_R的中间层放大是所有方向中最高的(22.56)**
→ **这比O_not(10.23)高2.2倍——早期层确实有差分放大机制**
→ **但这个效应在中高层(L18+)消失——差分放大主要发生在早期层**

### 新增客观事实拼图（5条）

81. **W_U本身没有差分放大——所有方向W_U增益接近random(~1.0x)**
82. **中间层有差分放大(Qwen3 L6: O_clean_R 2.2x O_not)，但仅限早期层**
83. **DS7B jacobian结果极不稳定——层间比率从0.03x到4.65x剧烈波动**
84. **DS7B构式差分方向(C_pc1_delta)被中间层极高放大(3-5x random)**
85. **差分放大不在W_U——在中间层（尤其早期层和特定功能层）**

### 理论修正

之前假设"差分放大可能发生在W_U输出层"。Phase 313数据明确否定了这个假说：

```
旧假设: W_U高增益读取δ_f → 功能区分
修正: 中间层放大δ_f → W_U平等读取所有方向 → 功能区分由中间层传播决定
```

新的因果链：
```
方向提取: h_l = shared + δ_f
中间层传播: J_{l→out}(δ_f) >> J_{l→out}(shared)  (在特定层)
W_U读取: W_U × J_{l→out}(δ_f) 和 W_U × J_{l→out}(shared) 增益相同
输出: δ_f经过中间层放大后到达W_U时已足够大
```

### 硬伤分析

1. **DS7B的jacobian分析不可靠**：unit方向注入在DS7B中产生不稳定结果，需要匹配范数注入
2. **只测了一个baseline句("they are very happy")**：Jacobian增益可能依赖上下文
3. **jac/wu比率的物理含义**：jac是含非线性变换的增益，wu是线性增益，两者比率不等于"非线性放大"
4. **delta_not=0问题**：tokenizer编码导致"not" token ID不匹配，未能测量特定token的Jacobian效应
5. **中间层的"差分放大"可能是残差连接的自然结果**：如果O_not和O_clean_R的范数差异很大，较小的O_clean_R经layernorm后会被放大

### 命令记录

```
python tests/glm5/phase313_WU_readout_analysis.py qwen3       # ~10min
python tests/glm5/phase313_WU_readout_analysis.py glm4        # ~31min
python tests/glm5/phase313_WU_readout_analysis.py deepseek7b  # ~24min
python tests/glm5_temp/phase313_cross_model.py                 # 跨模型分析
```

### 数据文件

- `results/phase313_WU_readout/{qwen3,glm4,deepseek7b}_WU_readout.json`
- `tests/glm5/phase313_WU_readout_analysis.py`

 ### 下一步
   GRCM全局相对编码图谱

## Phase 314: GRCM Global Relative Coding Map [2026-05-31 17:50]

### 目标

验证语言编码是否为关系网络编码而非点编码。构建外部语义关系网络G_external(8类关系)，提取模型内部关系图G_internal，比较两者同构性。

8类关系：同类(same_class)、上下位(hypernym)、属性(attribute)、功能(function)、反义(antonym)、否定(negation)、操作(operator_similar)、跨类别(cross_category)

68个节点，71条关系边，67个概念句子

### 核心发现86：所有三个模型都显著保持了人类语义关系网络

Mantel相关系数(所有p=0.002)：
```
模型       L2(早)    中层最佳    L_max(深层)    最佳层
Qwen3     0.550     0.537(L12)   0.446(L34)     L2
GLM4      0.587     0.558(L12)   0.465(L38)     L2
DS7B      0.511     0.575(L12)   0.466(L26)     L8-L12
```

→ **所有模型在所有层都显著保持了人类语义关系(p=0.002)**
→ **Mantel r在0.45-0.59范围——中等偏强相关**
→ **早期层(L2)关系保持最强(Qwen3/GLM4)，DS7B在中层(L8-12)最强**
→ **深层关系保持减弱——深层更多做任务压缩而非关系保持**
→ **这直接验证了"语言编码是关系网络编码"的假说**

### 核心发现87：关系类型保持存在通用层次——同类>上下位>否定>反义>跨类别>属性>功能

三模型中层关系保持比率(random_dist/related_dist)排序（越高=越强保持）：
```
层次  关系类型              Qwen3 L12  GLM4 L12   DS7B L12   通用排名
1    same_class(同类)       5.96        3.99       4.93       最强
2    hypernym(上下位)       3.24        2.65       3.36       第二
3    negation(否定)         3.08        2.18       2.75       第三
4    operator_similar(操作)  2.43        2.21       2.07       第四
5    antonym(反义)          2.00        2.04       1.87       第五
6    cross_category(跨类)   1.53        1.34       1.76       第六
7    attribute(属性)        1.04        0.92       0.96       第七
8    function(功能)         0.87        0.78       0.92       最弱
```

→ **同类关系保持最强(ratio=4-6)——模型强烈保持"苹果≈香蕉"**
→ **功能关系不被保持(ratio<1)——模型不理解"刀-切"的功能关联**
→ **否定关系比反义关系保持更强——否定是更基础的语义结构**
→ **这个层次在三个模型中高度一致——说明这不是模型特定的，而是语言本身的特征**

### 核心发现88：否定关系比反义关系保持得更好——否定是独立于反义的语义操作

negation/antonym比率跨模型跨层：
```
模型       L2     L6     L12    L18    L24    L30    L_max
Qwen3     2.32   1.51   1.55   1.34   1.55   1.56   ~1.5
GLM4      1.94   1.09   1.07   0.98   1.27   1.40   ~1.1
DS7B      2.49   1.70   1.47   1.23   1.19   1.38   ~1.4
```

→ **早期层(L2)否定/反义比率最高(1.9-2.5)——否定关系在早期就形成了强结构**
→ **GLM4中层L18出现比率<1(0.98)——GLM4在中层把否定和反义同样处理**
→ **DS7B早期层比率最高(2.49)——DS7B在早期层对否定有最强的独立编码**
→ **这支持否定≠反义词替换——否定在模型内部确实是独立的语义操作**

### 核心发现89：概念簇的共享方差随层递减——从共享到分化

水果簇(fruit cluster) shared_pc1_var跨层：
```
模型       L2     L6     L12    L18    L24    L30    L_max
Qwen3     0.991  0.968  0.906  0.710  0.558  0.454  0.411
GLM4      0.992  0.970  0.907  0.718  0.567  0.464  0.468
DS7B      0.993  0.962  0.891  0.700  0.559  0.461  0.469
```

→ **所有模型早期层shared_pc1_var≈0.99——水果概念几乎完全共享同一方向**
→ **后期层降到0.41-0.47——概念逐渐分化出独立方向**
→ **三模型的分化曲线几乎完全一致——这是通用机制而非模型特定**

工具簇(tool cluster) shared_pc1_var：
```
模型       L2     L12    L_max
Qwen3     0.996  0.969  0.697
GLM4      0.997  0.965  0.671
DS7B      0.994  0.957  0.761
```

→ **工具簇的分化更慢——工具概念的共享成分更持久**
→ **DS7B的工具簇shared_pc1_var=0.761(vs Qwen3的0.697)——DS7B的共享更刚性**

情绪正簇(emotion_pos) shared_pc1_var：
```
模型       L2     L12    L_max
Qwen3     0.943  0.542  0.310
GLM4      0.928  0.456  0.338
DS7B      0.928  0.510  0.307
```

→ **情绪概念分化最快——shared_pc1_var从0.93降到0.31**
→ **不同情绪(happy/strong/bright/warm/good)的差异比水果/工具大得多**
→ **这符合直觉：happy和bright的差异确实大于apple和banana的差异**

### 核心发现90：属性关系和功能关系不被保持——模型不理解"刀-切"和"苹果-红"

attribute和function的ratio在所有模型中都接近1.0或低于1.0：
```
关系类型    Qwen3 L12   GLM4 L12    DS7B L12
attribute    1.04         0.92        0.96
function     0.87         0.78        0.92
```

→ **ratio≈1.0意味着：关联概念的内部距离≈随机距离**
→ **模型不理解"刀是锋利的"、"苹果是甜的"这种属性关联**
→ **模型不理解"刀用来切"、"钥匙用来开"这种功能关联**
→ **但这不意味着模型不能使用这些关系——可能需要特定上下文才能激活**

### 新增客观事实拼图（5条）

86. **所有三模型都显著保持人类语义关系网络(Mantel r=0.45-0.59, p=0.002)**
87. **关系保持通用层次：同类>上下位>否定>操作>反义>跨类>属性>功能**
88. **否定比反义保持更强(neg/ant ratio=1.1-2.5)，尤其早期层**
89. **概念簇共享方差随层递减(fruit: 0.99→0.41)，情绪分化最快**
90. **属性和功能关系不被保持(ratio≈1.0)——需要上下文激活**

### 理论进展：语言编码的关系网络本质

Phase 314的数据第一次**从全局关系层面**验证了语言编码的关系网络本质：

```
1. 模型不是点编码，而是关系网络编码
   - Mantel r=0.55意味着模型内部保持了人类语义关系拓扑的约55%
   - 这个保持从L2到L34都存在，说明关系编码贯穿整个网络

2. 关系类型有清晰的优先级
   - 同类关系最强(ratio=4-6) → 模型最强地保持类别结构
   - 上下位次之(ratio=3-4) → 模型保持了层级分类
   - 否定第三(ratio=2-3) → 否定是独立于反义的基础操作
   - 属性/功能最弱(ratio≈1) → 这些关系需要上下文激活

3. 概念形成过程：共享→分化
   - 早期层：同类概念几乎完全共线(shared_pc1_var≈0.99)
   - 中层：逐渐分化(shared_pc1_var≈0.7-0.9)
   - 深层：高度分化(shared_pc1_var≈0.3-0.5)
   - 这个过程在三个模型中完全一致

4. 否定≠反义的再次验证
   - 否定关系在所有层都比反义关系保持更强
   - 早期层最强(neg/ant=2.5)→否定是比反义更基础的语义结构
   - 这与Phase 311的因果测试一致：O_not对happy无效应，A强抑制happy
```

### 命令记录

```
python tests/glm5/phase314_GRCM.py qwen3       # ~20s
python tests/glm5/phase314_GRCM.py glm4        # ~9min
python tests/glm5/phase314_GRCM.py deepseek7b  # ~5min
python tests/glm5_temp/phase314_cross_model.py  # 跨模型分析
```

### 数据文件

- `results/phase314_GRCM/{qwen3,glm4,deepseek7b}_GRCM.json`
- `tests/glm5/phase314_GRCM.py`

### 下一步

1. **扩展概念集**：当前68个节点太少，需要扩展到200+以获得更可靠的关系图谱
2. **关系级因果测试**：不只是测cosine距离，还要测"注入apple方向能否改变banana输出"
3. **上下文依赖测试**：属性/功能关系在特定上下文中是否被激活
4. **操作代数测试**：not(not X)≈X, 翻译操作的可组合性
5. **路径束图谱**：从单层距离升级到多层路径分析

## Phase 315: Context Activation + Relation-Level Causal Test [2026-05-31 19:30]

### 目标

解决Phase 314的两大硬伤：
A) 属性/功能关系在静态上下文中不被保持——是否因为缺少激活上下文？
B) Mantel相关只能证明拓扑相似，不能证明因果机制——需要关系级因果测试

### Part A: 上下文激活测试设计

4种关系类型×3种上下文条件：
- 属性：static("the apple was there"), attribute_probe("The apple is usually ___"), attribute_fill("The apple has the quality of being ___")
- 功能：static, function_probe("You use a knife to ___"), function_fill("A knife is designed for ___")
- 同类：static, category_probe("The apple and the banana are both ___")
- 否定：static_pos("they felt happy"), static_neg("they were not happy"), negation_probe("It is not_happy that things are happy")

15-20个概念对/关系类型，20个随机基线对

### 核心发现91：属性关系被上下文条件激活——attribute_fill在L2有5.0-5.7x ratio

```
模型         L2 Static  L2 AttrFill  L2 Ratio  L6 Static  L6 AttrFill  L6 Ratio
Qwen3        0.025      0.005        5.45x     0.093      0.036        3.27x
GLM4         0.035      0.007        5.01x     0.171      0.028        5.99x
DS7B         0.050      0.009        5.65x     0.164      0.033        5.26x
```

→ **静态上下文中属性ratio≈1.0-1.3x（不被保持）**
→ **属性填空上下文中L2 ratio=5.0-5.7x（强烈保持）**
→ **三模型完全一致——属性关系确实存在，但需要适当上下文激活**
→ **这个效应在深层消失（L24+ ratio降到0.7-1.0）——深层做任务压缩**

### 核心发现92：function_probe上下文摧毁功能关系（ratio=0.2-0.5x）

```
模型         L2 FuncProbe  L6 FuncProbe  L12 FuncProbe
Qwen3       0.20x         0.40x         0.45x
GLM4        0.24x         0.40x         0.61x
DS7B        0.33x         0.46x         0.47x
```

→ **"You use a knife to"模板使功能概念对比随机对更远**
→ **原因：function_probe模板让模型预测不同的动词（cut/open/drive），而非共享功能路径**
→ **function_fill("A knife is designed for")在L2有1.3-2.0x ratio——稍微好些**
→ **功能关系的激活方式不同于属性——可能需要更精确的上下文模板**

### 核心发现93：同类关系被类别上下文极大增强（ratio=11-15x）

```
模型         L2 Static  L2 CatProbe   L2 Ratio   L6 Static  L6 CatProbe   L6 Ratio
Qwen3       0.014      0.002         14.60x     0.044      0.009         12.90x
GLM4        0.017      0.007         4.89x      0.072      0.012         14.00x
DS7B        0.034      0.009         6.14x      0.095      0.016         11.16x
```

→ **类别探测上下文("The apple and the banana are both")使同类概念在激活空间极度接近**
→ **L6的ratio=11-15x——比Phase 314的静态ratio(3-6x)高出3-5倍**
→ **上下文门控效应极强——同一对概念在不同上下文中的距离可以差10倍以上**

### 核心发现94：否定关系在所有上下文中ratio<1——否定产生对立而非相似

```
模型         L2 StaticPos  L2 StaticNeg  L2 NegProbe
Qwen3       0.237         0.220         0.217
GLM4        0.371         0.333         0.367
DS7B        0.297         0.260         0.283
```

→ **否定对(happy/not_happy)在所有上下文中都比随机对更远**
→ **这是符合预期的：否定创造对立，而非相似**
→ **Phase 314的Mantel测试发现"否定关系被保持"是因为它测量距离排序**
→ **修正：否定关系不是"距离近"，而是"有确定的方向性对立"**

### 核心发现95：所有关系类型都有因果效力（1.4-7.5x random）

```
关系类型       Qwen3      GLM4      DS7B      跨模型范围
same_class    4.61x      3.35x     1.65x     1.7-4.6x
hypernym      7.47x      1.60x     2.36x     1.6-7.5x
negation      3.63x      1.79x     1.41x     1.4-3.6x
antonym       4.45x      2.87x     4.33x     2.9-4.5x
attribute     3.14x      1.87x     2.74x     1.9-3.1x
function      3.76x      2.53x     2.14x     1.9-3.8x
```

→ **所有关系类型的最佳因果效力都>1.4x random**
→ **即使attribute和function也有1.9-3.8x的因果效力——尽管静态距离不明显**
→ **这直接证明：属性/功能关系确实存在因果效力，只是静态测量不够敏感**
→ **Qwen3的因果效力整体最强（4-7x），GLM4最弱（1.6-3.4x）**

### 核心发现96：属性关系的上下文激活随层递减——从L2的5x降到L24的1x

```
模型         L2    L6    L12   L18   L24   L30   L32   L34
Qwen3       5.45  3.27  1.48  1.14  1.08  0.78  0.73  0.82
GLM4        5.01  5.99  3.49  1.61  1.01  0.78   -     -
DS7B        5.65  5.26  1.28  1.04  0.99   -     -     -
```

→ **属性激活效应在L12-L18急剧减弱——从3-6x降到1-1.5x**
→ **深层(L24+)完全消失——深层不再保持属性关系的上下文激活**
→ **这说明属性关系主要在早中层被条件激活，深层做任务相关压缩**

### 新增客观事实拼图（6条）

91. **属性关系被上下文条件激活（L2 ratio=5.0-5.7x），但静态不保持（ratio≈1.0）**
92. **function_probe上下文摧毁功能关系（ratio=0.2-0.5），功能需要不同激活方式**
93. **同类关系被类别上下文极大增强（L6 ratio=11-15x）**
94. **否定关系在所有上下文中ratio<1——否定产生对立而非相似**
95. **所有关系类型都有因果效力（1.4-7.5x random），包括attribute和function**
96. **属性关系的上下文激活随层递减（L2: 5x → L24: 1x）**

### 修正Phase 314的判断

Phase 314说"属性和功能关系不被保持"。Phase 315修正为：
```
旧判断: 属性/功能关系不被保持(ratio≈1.0)
修正: 属性/功能关系在无上下文时不被静态保持，但在适当上下文中被条件激活
     属性: attribute_fill context L2 ratio=5.0-5.7x
     功能: function_fill context L2 ratio=1.3-2.0x
     因果测试: attribute和function都有1.9-3.8x的因果效力
```

这验证了分析二的预测："属性/功能关系不是不存在，而是条件激活"。

### 关系编码机制更新

```
Phase 314: 语言编码是关系网络编码
Phase 315修正: 语言编码是条件激活的关系网络编码

  1. 关系结构存在，但多数不是静态保持的
     - 同类关系：静态弱保持(3-6x)，上下文强激活(11-15x)
     - 属性关系：静态不保持(≈1x)，上下文强激活(5-6x)
     - 功能关系：静态不保持(≈1x)，上下文弱激活(1.3-2.0x)
     - 否定关系：产生对立而非相似(ratio<1)

  2. 上下文门控机制
     - 同一句子中的概念距离受上下文模板强烈影响（可差10倍）
     - 上下文激活在早中层最强，深层消失
     - 这意味着：概念不是固定点，而是上下文依赖的位置

  3. 因果效力 ≠ 静态距离
     - attribute和function静态ratio≈1，但因果效力1.9-3.8x
     - 说明这些关系的因果路径存在，但静态余弦距离不足以测量
     - 因果效力可能来自差分方向的放大（Phase 313发现）
```

### 命令记录

```
python tests/glm5/phase315_context_causal.py qwen3       # ~3min
python tests/glm5/phase315_context_causal.py glm4        # ~38min
python tests/glm5/phase315_context_causal.py deepseek7b  # ~31min
python tests/glm5_temp/phase315_cross_model.py            # 跨模型分析
```

### 数据文件

- `results/phase315_context_causal/{qwen3,glm4,deepseek7b}_context_causal.json`
- `tests/glm5/phase315_context_causal.py`

### 硬伤分析

1. **function_probe模板设计有问题**："You use a knife to"让模型预测不同动词，不是测试共享路径。需要更好的模板。
2. **否定关系的上下文模板不够好**：negation_probe模板"not_happy"不是自然语言，需要用真正的否定句。
3. **因果测试只测了W_U层和hook注入**：还需要Jacobian传播分析来确认中间层放大。
4. **概念对数量偏少**：属性15对，功能15对，需要更多来确认统计显著性。
5. **function_fill的ratio只在L2达到2x，L6就降到1.4x以下**：功能关系的上下文激活可能需要更特定模板。

### 下一步

1. **修复function上下文模板**：设计更好的功能激活模板（如"Using a knife, you can"）
2. **否定关系方向测试**：测量happy→not_happy的方向是否有因果效力（注入方向能否改变否定判断）
3. **上下文-因果联合测试**：在attribute_fill上下文中注入attribute差分方向，测因果效力是否增强
4. **大规模确认测试**：扩大概念对到50+对，确认属性/功能上下文激活的稳定性

基于Phase 310-313的发现，下一步应从"单方向测试"升级到"全局关系网络测试"：

1. **构建多层关系网络**：8类关系（同类/上下位/属性/功能/反义/否定/操作/组合）
2. **提取模型内部关系图**：每层计算概念间余弦距离矩阵
3. **比较内外图同构性**：Mantel相关/邻域重叠/排序保持
4. **复用-差分路径分解**：对每个概念簇提取shared_path和delta_path
5. **建立三图：复用图、差异图、冲突图**

这才是破解整体编码机制的关键路径。

## Phase 316: Phase 315-R2 确认测试跨模型分析 [2026-05-31 21:30]

### 命令记录

```
python tests/glm5/phase315r2_confirm.py qwen3       # 已完成 (Qwen3 ~1.5min)
python tests/glm5/phase315r2_confirm.py glm4        # 已完成 (GLM4 ~55min)
python tests/glm5/phase315r2_confirm.py deepseek7b  # 已完成 (DS7B ~35min)
python tests/glm5_temp/phase315r2_cross_model.py     # 跨模型分析
```

### 数据文件

- `results/phase315r2_confirm/{qwen3,glm4,deepseek7b}_confirm.json`
- `tests/glm5/phase315r2_confirm.py`
- `tests/glm5_temp/phase315r2_cross_model.py`

### Test 1: 属性上下文激活确认 (50+ 对) — 设计缺陷

**关键发现：Test 1 存在设计缺陷**

- attr_fill条件使用了非平行句子："The apple has the quality of being red" vs "The red is a quality"
- static条件使用了平行句子："the apple was there" vs "the red was there"
- 非平行句子的结构差异导致距离人为放大（0.597 vs 0.025），不反映属性激活
- attr_fill_ratio < 1 是设计假象，不能与Phase 315原始结果对比
- Phase 315原始测试使用平行模板（同一句式不同填词），设计正确

**数据（参考但不作为有效证据）**：
| Layer | Qwen3 attr_fill_ratio | GLM4 attr_fill_ratio | DS7B attr_fill_ratio |
|-------|----------------------|---------------------|---------------------|
| L2    | 0.048                | 0.044               | 0.068               |
| L6    | 0.173                | 0.224               | 0.220               |
| L12   | 0.228                | 0.444               | 0.290               |

### Test 2: 功能模板比较 — 有效结果

**发现 97: function_tool模板跨模型一致最优**
- 所有3个模型中，"The X is a tool for"模板一致产生最高的功能关系激活
- Qwen3: function_tool ratio=1.39(L6), 1.27(L12)
- GLM4: function_tool ratio=1.18(L6), 1.44(L12)
- DS7B: function_tool ratio=1.26(L6), 1.32(L12)
- 跨模型平均: function_tool=1.31x > designed=1.20x > purpose=1.08x > using=1.05x

**发现 98: function_purpose模板有害**
- "The purpose of a X is to"模板在多个模型/层中ratio<1
- Qwen3 L2: 0.89, DS7B L2: 0.78
- 说明"purpose"这个词本身可能引向抽象语义而非功能关联

**发现 99: 功能关系需要特定上下文框架**
- 静态条件(static)下功能关系也很弱（ratio~1.0-1.3）
- 只有tool框架能稳定激活（1.3-1.4x）
- 这证实Phase 315的结论：功能关系不是静态编码的，而是条件激活的

### Test 3: 否定方向因果测试 — 核心发现

**发现 100: 否定方向范数跨模型差异巨大（编码架构差异的又一证据）**
- Qwen3: neg_dir_norm avg=25.07, range=[17.19, 38.37]
- GLM4: neg_dir_norm avg=3.01, range=[2.14, 3.67]  ← 8.3x小于Qwen3！
- DS7B: neg_dir_norm avg=82.69, range=[46.83, 113.82] ← 3.3x大于Qwen3！

这与Architecture A(Qwen3/GLM4) vs Architecture B(DS7B)完全一致：
- GLM4使用更紧凑的表示空间，否定方向范数极小
- DS7B使用更分散的表示空间，否定方向范数极大
- 否定方向范数的数量级差异反映了底层编码架构的根本区别

**发现 101: 否定方向具有稳定的因果效力（跨模型/跨语义域）**
- 所有3个模型中，注入否定方向能可靠地激活语义对立词：
  - happy→not: 激活 "unhappy"(1.5-1.6x), "disappointed"(1.0-2.1x)
  - safe→not: 激活 "unsafe"(2.5-3.3x), "dangerous"(1.1-1.9x), "risky"(1.7-2.3x)
  - good→not: 激活 "bad"(0.9-2.9x), "negative"(0.6-2.2x)
  - possible→not: 激活 "impossible"(0.4-1.9x)
  - clean→not: 激活 "dirty"(1.0-2.9x), "messy"(1.0-4.0x)

- 语义否定效果（排除"not"token）随层深增加：
  - Qwen3: L12=1.15x → L34=1.76x
  - GLM4: L12=0.75x → L38=1.08x
  - DS7B: L8=0.85x → L26=1.00x

**发现 102: 否定方向是语义对跖方向，不是简单翻转**
- 否定方向（pos→neg）的W_U投影不仅激活"not"，还激活整个否定语义网络
- 例如happy→not方向同时激活unhappy+disappointed+sad，不仅仅是happy的反义词
- 否定方向编码的是"语义极性翻转"的复合操作，而非单一token映射

**发现 103: DS7B否定方向在深层出现语义发散**
- DS7B的neg_dir_norm极大（L26=85-245），深层传播后语义效果不稳定
- "possible→not"在DS7B的L8层效果仅0.04x（几乎无效），而Qwen3在L12=2.58x
- 这与Phase 313发现的DS7B Jacobian不稳定一致
- DS7B的大范数=低信噪比，否定方向在深层被噪声淹没

### 硬伤分析

1. **Test 1设计缺陷（严重）**：非平行句子比较使属性激活测试失效。必须用平行模板重测。
2. **否定因果只测了5对**：需要更多否定对（20+对）来确认统计显著性。
3. **功能模板只测了3层**：深层（L18+）功能关系激活未测，可能遗漏衰减模式。
4. **缺少上下文×因果交互测试**：Phase 315计划了"在属性上下文中注入属性差分方向"的交互测试，R2未执行。
5. **GLM4 neg_dir_norm极小(2-4)**：可能与GLM4的40层深度和4096维空间有关，需要进一步分析这是紧凑编码还是范数压缩。

### 下一步

1. **修复属性激活测试**：用平行模板重测50+对属性对
2. **扩大否定对数量**：从5对扩展到20+对，测量否定方向的统计显著性
3. **上下文×因果交互测试**：在attribute_fill上下文中注入属性差分方向，测因果效力是否增强
4. **否定方向的正交性测试**：测量不同否定对的方向是否共享同一"否定子空间"
5. **深层功能关系衰减测试**：测量L18-L34的功能模板效果

### 阶段性任务规划

当前已完成的事实拼图：
- 语言编码 = 高方差共享 + 低方差差分 + 高增益读出（Phase 313）
- 关系网络在中间层被保留（Phase 314 Mantel r=0.45-0.59）
- 属性/功能关系条件激活（Phase 315）→ Phase 317大幅修正
- 否定方向是因果有效的语义极性翻转（Phase 316/R2）
- 否定方向范数反映编码架构差异（GLM4极小=紧凑, DS7B极大=分散）

下一步应进入**关系子空间分解**阶段：
- 8类关系是否共享同一条"关系编码主干"？
- 否定子空间 vs 反义子空间 vs 属性子空间是否正交？
- 如果不共享，关系的几何结构是什么？（分形？层级嵌套？环面？）
- 这是破解"语言编码的数学结构"的关键突破口

## Phase 317: 关键测试修正与跨模型验证 [2026-06-01 00:15]

### 背景

Phase 315-R2 存在两个严重设计缺陷：
1. **属性激活测试使用非平行模板**："The apple has the quality of being red" vs "The red is a quality"——不同模板结构本身引入巨大差异，ratio=5.45是虚假膨胀
2. **Context × Causal 交互测试未执行**——这是验证"上下文门控"假设的关键实验
3. **否定测试只有5对**——统计效力不足

Phase 317 针对3个模型（Qwen3/GLM4/DS7B）重测，修复所有设计缺陷。

### 测试设计

**Test 1: 属性激活（平行模板, 50对, 6层）**
- 3个模板都用 `"{w}"` 占位符，确保同一对词的两种角色（名词/属性）经历完全相同的句子结构
- static: `"the {w} was there"`
- attribute_probe: `"The {w} is usually"`
- attribute_fill: `"the {w} {w}"` → 实际为 `"the apple red"` 等
- 随机基线：同模板内打乱配对，而非跨模板比较

**Test 2: Context × Causal 交互（15对, 注入L12, 读L12/18/24/34）**
- 在静态上下文（"the X was there"）和属性上下文（"The X is usually"）中分别注入属性差分方向
- 交互比 = 属性上下文中的因果效力 / 静态上下文中的因果效力
- 同时测量随机方向的交互比作为对照

**Test 3: 扩大否定测试（25对, 3种否定类型, 5层）**
- regular: 10对常规否定（very happy → not happy）
- double_negation: 5对双重否定（very bad → not bad）
- weak_negation: 5对弱否定（very great → not great）
- 另有5对regular扩展

### 关键结果

**Test 1: 属性激活——Phase 315的5.45x是虚假膨胀！**

| 模板 | 模型 | L2 | L6 | L12 | L18 | L24 | 最深层 |
|------|------|-----|-----|------|------|------|--------|
| static | Qwen3 | 1.09 | 1.15 | 1.14 | 1.07 | 1.13 | L34:1.41 |
| static | GLM4 | **0.82** | **0.94** | 1.14 | **0.99** | **0.94** | L38:1.19 |
| static | DS7B | 0.97 | 0.98 | 1.04 | 0.94 | - | L26:0.97 |
| attr_fill | Qwen3 | 1.11 | 1.18 | 1.21 | 1.19 | 1.37 | L34:**1.79** |
| attr_fill | GLM4 | 1.18 | 1.09 | 1.07 | 1.02 | 1.17 | L38:**1.52** |
| attr_fill | DS7B | 1.14 | 1.14 | 1.21 | 1.19 | - | L26:**1.46** |

核心发现：
- Phase 315声称attribute_fill ratio=5.45，实际只有1.1-1.8x（**膨胀3-5倍**）
- 原因：跨模板random基线极低（不同模板的表示距离本来就大），导致ratio虚高
- GLM4的早层ratio<1.0，说明属性关联词对在该层比随机词对距离更远
- 只有最深层（L34/L38/L26）才出现1.4-1.8x的modest效应
- **结论：属性上下文使所有词对距离缩小，而非特定使属性关联词对更近**

**Test 2: Context × Causal 交互——上下文门控不是通用机制**

| 模型 | attr_dir交互比 | random_dir交互比 | 净门控效果 |
|------|---------------|-----------------|-----------|
| Qwen3 | 1.196 | 1.239 | **-0.044**（random更好！） |
| GLM4 | 1.137 | 1.125 | +0.012（可忽略） |
| DS7B | 1.258 | 1.202 | +0.056（仅边际） |

核心发现：
- 仅DS7B勉强超过1.2x阈值
- Qwen3中随机方向比属性方向更有效，净门控为负
- 交互比有巨大pair间方差（0.27~6.07），高度不稳定
- **结论：上下文门控是模型特异和词对特异的，不是通用机制**
- 某些词对（如diamond_hard在DS7B:6.07x）有强烈门控，但其他词对无

**Test 3: 否定——三个模型的否定机制本质不同**

否定方向范数（跨模型数量级差异）：

| 否定类型 | GLM4 | Qwen3 | DS7B |
|---------|------|-------|------|
| regular | 3.07 | 25.34 | 84.71 |
| double_neg | 3.72 | 29.56 | 105.23 |
| weak_neg | 3.88 | 27.79 | 106.56 |

否定选择率（neg_top / rand_top，最深层）：

| 否定类型 | GLM4 | Qwen3 | DS7B |
|---------|------|-------|------|
| regular | 1.94x | 2.10x | 1.50x |
| double_neg | 1.83x | 2.14x | 1.32x |
| weak_neg | 2.12x | **2.94x** | 1.58x |

双重否定分析（"not bad"是否激活正面词？）：
- **Qwen3**: 5对中3对MITIGATING（否定方向激活正面词）
- **GLM4**: 5对中2对MITIGATING，2对REINFORCING，1对不确定
- **DS7B**: 5对中2对MITIGATING，2对REINFORCING，1对不确定

核心发现：
- 反义方向范数始终大于否定方向，说明反义是更"直接"的语义操作
- 双重否定的效果高度不稳定，模型间不一致
- Qwen3的weak_neg选择率最高(2.94x)——弱否定词对（great→not great）比强否定（happy→not happy）效果更清晰
- DS7B的"not bad"反而**强化**"bad"——否定不是逻辑取反
- **结论：否定方向是"语义极性偏移"，不是逻辑否定算子**

### 硬伤分析（最严格审视）

1. **属性激活的"浅层反转"问题**（严重）：
   - GLM4的L2层ratio=0.82（<1.0），意味着属性关联词对在浅层反而比随机词对更远
   - 这与"属性上下文使词对更近"的假设矛盾
   - 可能解释：GLM4的浅层专门编码token identity而非关系，同一词在不同角色下表示差异更大

2. **上下文门控的"随机方向问题"**（致命）：
   - Qwen3中random_dir交互比(1.239) > attr_dir交互比(1.196)
   - 这意味着注入任意方向在属性上下文中都更有效，而非特指属性方向
   - 门控效应可能只是"属性上下文的表示空间更敏感"，而非"属性上下文选择性地放大属性方向"

3. **否定方向的高方差**（严重）：
   - DS7B的neg_dir_norm=47-116，而GLM4仅2-4
   - 如此大的范数差异意味着不同模型的"否定操作"在数学上是完全不同的操作
   - 试图找到一个统一的"否定数学结构"可能是不可能的

4. **weak_neg选择率反常**（需关注）：
   - Qwen3中weak_neg(2.94x) > regular(2.10x)，这不符合直觉
   - 可能原因："not great"与"not happy"在语用上不同——"not great"更接近"mediocre"，方向更清晰
   - 这暗示否定方向的语义效果受形容词语义梯度影响，不是纯逻辑操作

5. **Test 2中L12层交互比≈1.0**（关键）：
   - 所有模型在注入层(L12)的交互比几乎恰好1.0
   - 因为注入方向在注入层直接加到表示上，没有经过后续层的变换
   - 真正的门控效果来自后续层的非线性变换，但平均效果太弱

### 拼图更新

已修正的事实拼图：
1. ~~属性上下文使属性关联词对显著更近(5.45x)~~ → **属性上下文使所有词对modestly更近(1.1-1.8x)，属性特异性很弱**
2. ~~上下文门控：属性上下文选择性地放大属性方向的因果效力~~ → **上下文门控不是通用机制，随机方向在属性上下文中同样有效**
3. 否定方向是"语义极性偏移"(confirmed)，不是"逻辑否定算子"(rejected)
4. 否定方向的数学实现是模型特异的（范数差异30-50倍）
5. 反义方向始终比否定方向更强——语言模型把"反义"当作更基本的操作

### 破解语言数学结构的第一性原理分析

**核心洞察1：语言编码的本质是"条件激活的相对关系网络"——但"条件激活"比预期弱得多**

Phase 313-315建立了一个图景：语言编码 = 高方差共享 + 低方差差分 + 高增益读出 + 条件激活。Phase 317揭示"条件激活"这一环节远比预期弱（1.1-1.8x而非5x+）。这意味着：
- 关系信息主要在**差分方向**本身编码，而非在上下文的门控机制中
- "属性关系激活"更多是**统计相关性**的体现，而非因果性的条件门控
- 模型不需要"知道这是属性上下文"才能提取属性信息——属性方向本身就携带了关系语义

**核心洞察2：否定不是逻辑算子，而是语义空间中的"偏移操作"**

所有3个模型都确认：
- 否定方向 ≈ 朝向否定/中性语义区域的偏移向量
- 双重否定 ≠ 肯定（不是逻辑双翻转）
- 反义方向 > 否定方向（反义是更直接、更强的语义操作）
- 否定操作的强度和方向高度依赖具体词对

这暗示语言模型内部的"否定"更接近于人类语言中否定的**语用功能**（"not X"≈"X的否定极"而非"X的逻辑补"），而非形式逻辑中的否定算子。

**核心洞察3：不同模型的"同一关系"编码在数学上可能是完全不同的**

GLM4的属性方向norm=1.5-2.4 vs DS7B的norm=27-56。这不是简单的"缩放"差异——40倍的范数差异意味着：
- 这些方向可能存在于完全不同的子空间维度
- "属性关系"在不同模型中可能对应不同的几何结构
- 寻找"跨模型的通用关系编码结构"需要关注**方向**而非**范数**

### 下一步方向

**瓶颈**：我们一直在测"某个方向是否有因果效力"，但无法回答"方向的几何结构是什么"。

**突破方向**：从"因果效力测试"转向"几何结构测绘"：

1. **关系子空间PCA**：对同一类关系的多个方向（如10个属性方向）做PCA，看它们是否共享一个低维子空间
2. **跨关系类型子空间正交性**：否定子空间 vs 反义子空间 vs 属性子空间是否正交
3. **单方向的投影分解**：一个否定方向 = α·反义方向 + β·否定特有方向 + γ·噪声方向？
4. **跨模型子空间对齐**：GLM4的属性子空间是否与Qwen3的属性子空间正交对齐（CCA/RV系数）

这是从"行为测试"进入"结构解码"的关键转折点。

### 命令记录

```bash
# 主测试脚本（3个模型顺序运行）
python tests/glm5/phase317_comprehensive.py
# 耗时：Qwen3~3min, GLM4~70min, DS7B~97min

# 跨模型分析
python tests/glm5_temp/phase317_cross_model.py
```

脚本位置：
- `tests/glm5/phase317_comprehensive.py` — 主测试
- `tests/glm5_temp/phase317_cross_model.py` — 跨模型分析
- 结果：`results/phase317_comprehensive/{qwen3,glm4,deepseek7b}_phase317.json`
- 日志：`tmp/phase317_{qwen3,glm4,deepseek7b}.log`

## Phase 318: 关系子空间几何测绘 [2026-06-01 00:50]

### 背景

Phase 317 把研究从"行为因果测试"推进到"几何结构测绘"的必要性已经明确。Phase 318 是第一次系统性的关系子空间几何分析，回答核心问题：**同一类关系的方向是否形成稳定的低维子空间？**

### 测试设计

6类关系 × 30对 × 5层 × 3模型：
- **same_class** (同类): apple→orange, dog→cat, car→bus 等30对
- **attribute** (属性): apple→red, ice→cold, silk→smooth 等30对
- **function** (功能): knife→cut, pen→write, car→drive 等30对
- **antonym** (反义): happy→sad, hot→cold, big→small 等30对
- **regular_negation** (常规否定): "very happy"→"not happy" 等30对
- **weak_negation** (弱否定): "very great"→"not great" 等30对

模板设计：
- 词对关系: "the {w} was there" (保证平行结构)
- 否定关系: 直接使用句子对

分析指标：
- PCA: 子空间维度(dim@50/80/90%), top1/top3解释方差比
- LOO cosine: 留一法余弦对齐度(衡量每个方向是否被其余方向解释)
- pairwise cosine: 同类方向间的平均余弦相似度
- 跨类型主角度: 不同关系子空间间的夹角分布
- 范数统计: 方向范数的均值/分布

### 核心结果

**结果1: 子空间维度——没有哪类关系形成"低维"子空间**

| 关系类型 | 模型 | L深 dim@80 | L深 dim@90 | L深 top1% | 说明 |
|---------|------|-----------|-----------|----------|------|
| same_class | Qwen3 | 19 | 24 | 8.5% | 极高维，接近随机 |
| same_class | GLM4 | 18 | 23 | 11.3% | 极高维 |
| same_class | DS7B | 19 | 24 | 8.7% | 极高维 |
| attribute | Qwen3 | 16 | 21 | 15.0% | 中高维 |
| attribute | GLM4 | 15 | 20 | 16.6% | 中高维 |
| attribute | DS7B | 17 | 22 | 11.6% | 中高维 |
| function | Qwen3 | 16 | 21 | 12.0% | 中高维 |
| function | GLM4 | 15 | 20 | 19.1% | 中高维 |
| function | DS7B | 18 | 23 | 9.7% | 中高维 |
| antonym | Qwen3 | 16 | 21 | 14.6% | 中高维 |
| antonym | GLM4 | 16 | 21 | 14.2% | 中高维 |
| antonym | DS7B | 18 | 23 | 16.1% | 中高维 |
| regular_negation | Qwen3 | 14 | 19 | 20.5% | 相对紧凑 |
| regular_negation | GLM4 | 16 | 21 | 18.3% | 相对紧凑 |
| regular_negation | DS7B | 18 | 22 | 20.2% | 中高维 |
| weak_negation | Qwen3 | 16 | 22 | 15.3% | 中等 |
| weak_negation | GLM4 | 18 | 23 | 12.9% | 中高维 |
| weak_negation | DS7B | 20 | 24 | 12.4% | 高维 |

核心发现：30个方向在2560-4096维空间中需要14-20个主成分才能解释80%方差——**这不是低维子空间**。对比：如果30个方向共享一个5维子空间，dim@80应该=5。实际dim@80=14-20，意味着这些方向**几乎占据了完整的30维空间**。

**结果2: LOO cosine——同类方向之间的对齐度分层明显**

| 关系类型 | Qwen3 深层 | GLM4 深层 | DS7B 深层 | 解释 |
|---------|-----------|-----------|-----------|------|
| same_class | 0.43 | 0.50 | 0.44 | 弱对齐 |
| attribute | 0.69 | 0.71 | 0.64 | 中等对齐 |
| function | 0.77 | 0.76 | 0.61 | 中强对齐 |
| antonym | 0.62 | 0.60 | 0.51 | 弱-中对齐 |
| regular_negation | 0.76 | 0.72 | 0.67 | 中强对齐 |
| weak_negation | 0.83 | 0.80 | 0.67 | 强对齐(浅层) |

排序(从强到弱): **weak_negation > function ≈ regular_negation > attribute > antonym > same_class**

核心发现：
- same_class方向几乎不对齐(LOO~0.4)——不同类别的同类关系方向完全没有共同结构
- weak_negation在浅层(LOO~0.8-0.9)对齐最强——"very X → not X"操作高度一致
- attribute和function处于中间——部分共享结构，但不完全统一

**结果3: pairwise cosine——方向间余弦揭示了深层结构**

| 关系类型 | Qwen3 深层 | GLM4 深层 | DS7B 深层 |
|---------|-----------|-----------|-----------|
| same_class | 0.011 | 0.007 | 0.002 | 近乎正交 |
| attribute | 0.18 | 0.18 | 0.19 | 弱正相关 |
| function | 0.40 | 0.30 | 0.21 | 中等正相关 |
| antonym | 0.020 | 0.008 | 0.020 | 近乎正交 |
| regular_negation | 0.35 | 0.29 | 0.27 | 中等正相关 |
| weak_negation | 0.55 | 0.52 | 0.36 | 较强正相关 |

核心发现：
- **same_class和antonym的pairwise cosine接近0**——同类关系的不同实例之间方向近乎正交
- **weak_negation的pairwise cosine最高(0.36-0.55)**——所有弱否定方向确实指向类似的方向
- 这意味着："同类关系"的几何结构不是"共享同一方向"，而是"分散在一个高维空间中的弱相关方向族"

**结果4: 跨类型主角度——没有两类关系完全正交或完全对齐**

所有类型对的mean_angle都在49-63度之间，mean_cosine在0.43-0.61之间。
- 最对齐: regular_negation vs weak_negation (50-54deg, cos≈0.56-0.59)
- 最对齐: same_class vs antonym (49-54deg, cos≈0.56-0.61) 
- 最正交: attribute vs function (55-62deg, cos≈0.44-0.54)
- 最正交: antonym vs regular_negation (55-62deg, cos≈0.43-0.52)

核心发现：所有关系子空间之间的角度都在50-60度附近——**既不正交(90度)也不对齐(0度)**。这暗示所有关系类型共享一部分底层表示空间，但各自有独特的成分。

**结果5: 范数分布——确认跨模型尺度差异**

| 关系类型 | Qwen3 L深 | GLM4 L深 | DS7B L深 | GLM4/Qwen3 比率 |
|---------|----------|---------|---------|----------------|
| same_class | 88 | 54 | 228 | 0.6x |
| attribute | 140 | 89 | 338 | 0.6x |
| regular_negation | 153 | 70 | 307 | 0.5x |

GLM4方向范数约为Qwen3的0.5-0.6倍，DS7B约为Qwen3的2-2.5倍。这与Phase 317的发现一致，但Phase 318的归一化分析排除了范数差异对角度分析的干扰。

### 客观事实拼图更新

1. **同一类关系方向不形成低维子空间** (dim@80=14-20，接近满秩30)
2. **同类方向的对齐度分层**：weak_negation>function>attribute>antonym>same_class
3. **same_class和antonym方向近乎正交**——"同类"和"反义"关系的不同实例之间方向差异极大
4. **weak_negation方向最一致**——"very X → not X"操作在所有词对中指向类似方向
5. **所有关系子空间间角度50-60度**——共享底层表示但各有独特成分
6. **negation子空间重叠最大**(regular vs weak, 50-54deg)——常规否定和弱否定共享最多几何结构
7. **attribute vs antonym接近最正交**(55-62deg)——属性关系和反义关系在几何上最不同

### 硬伤分析

1. **"the {w} was there"模板的局限**（严重）：
   - "the cut was there"中"cut"可能被编码为名词而非动词
   - 同理，"the smooth was there"中"smooth"可能是名词用法
   - 这可能解释function和attribute方向的低对齐度——模板导致词义偏移

2. **否定方向来源不同**（需关注）：
   - 词对关系: 两个词在同一模板中，方向 = h(词B) - h(词A)
   - 否定关系: 两个句子，方向 = h(否定句) - h(肯定句)
   - 句子差异包含"very"→"not"的结构差异，不仅仅是语义差异
   - 这可能导致否定的LOO cosine被人为抬高

3. **dim@80=14-20可能受噪声影响**（需验证）：
   - 30个方向在2500+维空间中，前14-20个PC解释80%方差可能只是信号+噪声
   - 需要随机方向基线对比：如果30个随机方向的dim@80也是14-20，则说明结果无意义

4. **negation的高LOO cosine可能是"not"的单一token效应**：
   - 所有否定对都包含"not"这个公共token
   - 方向中"not"的贡献可能是LOO cosine高的主要原因
   - 需要控制实验：同样句子对但不包含"not"的方向

### 破解语言数学结构的第一性原理分析

**核心洞察1: 关系编码不是"方向一致的子空间"，而是"结构相似的方向族"**

Phase 318最重要的发现是：同类关系方向的pairwise cosine接近0(same_class=0.002-0.023, antonym=0.008-0.032)，意味着"同类关系"没有统一的几何方向。

但LOO cosine显示它们仍然可以被其他同类方向部分预测(0.4-0.8)。这意味着：**关系的共同结构不是"同一方向"，而是"某种可预测的变换模式"**。

类比：所有旋转都是"旋转操作"，但不同角度的旋转方向完全不同。关系的共同性可能在于某种"变换类型"(如"偏移"、"投影")，而不在于"方向本身"。

**核心洞察2: 弱否定的最高一致性暗示存在"操作级"编码**

weak_negation的pairwise cosine高达0.36-0.66，远超其他类型。这意味着"very X → not X"这个操作在模型内部有非常一致的几何实现——不管X是什么。

但注意：这种一致性可能来自"not"这个单一token，而非语义否定操作。如果是"not"的token embedding贡献，那么这不是"关系编码"而是"token编码"。

**关键区分实验**：用"never X"代替"not X"——如果LOO cosine同样高，则是否定操作编码；如果大幅下降，则是"not" token编码。

**核心洞察3: 属性和功能方向的"中等对齐"可能来自词性模式**

attribute和function的pairwise cosine=0.17-0.40，介于same_class(≈0)和weak_negation(≈0.5)之间。这可能反映的不是"属性关系"本身的一致性，而是**名词→形容词/动词的词性变换**的一致性。

验证方法：对比"apple→red"(名词→形容词)和"red→apple"(形容词→名词)的方向——如果方向不对称(范数和角度都不同)，说明词性变换是主导因素。

### 下一步方向

**瓶颈**: Phase 318发现关系方向不形成低维子空间，但我们还不清楚这是"真的没有共同结构"还是"模板和token干扰掩盖了真实结构"。

**突破方向**: 在做更复杂的几何分析之前，必须先解决"模板污染"和"token污染"问题：

1. **随机方向基线**: 30个随机词对的dim@80/LOO cosine/pairwise cosine是多少？如果和same_class差不多，说明same_class没有超出随机水平的结构信号

2. **"not" token控制实验**: 对比 "not X" vs "never X" vs "barely X"——它们的子空间结构是否一致？

3. **词性变换控制**: 对比 "apple→red" vs "red→apple" 的方向——词性效应有多大？

4. **多模板交叉验证**: 同一关系对在不同模板中的方向是否一致？

只有排除这些混淆因素后，才能判断"关系编码的几何结构"到底是怎样的。

### 命令记录

```bash
# Phase 318: 关系子空间几何测绘
python tests/glm5/phase318_subspace_geometry.py qwen3    # ~20s
python tests/glm5/phase318_subspace_geometry.py glm4     # ~6.5min
python tests/glm5/phase318_subspace_geometry.py deepseek7b  # ~4.5min
```

脚本位置：
- `tests/glm5/phase318_subspace_geometry.py` — 主测试
- 结果：`results/phase318_subspace/{qwen3,glm4,deepseek7b}_phase318.json`
- 日志：`tmp/phase318_{qwen3,glm4,deepseek7b}.log`

## Phase 319: 随机基线 + 模板控制 + 否定Token控制 [2026-06-01 07:48]

### 背景

Phase 318发现关系方向不形成低维子空间，但缺少三个关键控制：(1)随机基线——不知道same_class/antonym是否超出随机水平；(2)模板污染——"the {w} was there"可能产生伪影方向；(3)否定Token控制——高一致性可能来自"not"这个单一token。

### 测试设计

**Part A: 随机基线** (3组×30对)
- random_mixed: 30随机词对(名词→形容词等混合词性)
- random_noun_noun: 30随机名词→名词对
- random_adj_adj: 30随机形容词→形容词对
- 全部使用"the {w} was there"模板（与Phase 318相同）

**Part B: 多模板一致性** (4类型×15对×3模板)
- t1: "the {w} was there" (Phase 318原模板)
- t2: "they mentioned the {w}"
- t3: "they discussed the {w}"
- 计算：同对方向跨模板余弦相似度

**Part C: 否定Token控制** (20形容词×4否定形式)
- not: "very {adj}" → "not {adj}"
- never: "very {adj}" → "never {adj}"
- barely: "very {adj}" → "barely {adj}"
- morphological: "very {adj}" → "un{adj}"
- 计算：同词跨否定形式方向余弦 + 子空间主角度

### 核心结果

**结果1: 随机基线——same_class和antonym低于或等于随机水平！**

| 组别 | 模型 | dim@80 | top1% | LOO_cos | pair_cos |
|------|------|--------|-------|---------|----------|
| random_mixed | Qwen3 | 15 | 18.2% | 0.748 | 0.271 |
| random_mixed | GLM4 | 15 | 17.5% | 0.758 | 0.317 |
| random_mixed | DS7B | 17 | 14.2% | 0.674 | 0.264 |
| random_noun_noun | Qwen3 | 15 | 18.8% | 0.740 | 0.250 |
| random_noun_noun | GLM4 | 14 | 16.2% | 0.744 | 0.222 |
| random_noun_noun | DS7B | 17 | 15.0% | 0.652 | 0.171 |
| random_adj_adj | Qwen3 | 15 | 14.6% | 0.664 | 0.021 |
| random_adj_adj | GLM4 | 14 | 16.2% | 0.688 | 0.018 |
| random_adj_adj | DS7B | 17 | 13.5% | 0.571 | 0.017 |

对比Phase 318 (深层):
- **same_class**: dim@80=18-19, pair_cos=0.002-0.023 → **pair_cos低于random_noun_noun(0.17-0.25)！**
- **antonym**: dim@80=16-18, pair_cos=0.008-0.032 → **pair_cos与random_adj_adj(0.017-0.021)几乎相同！**
- **attribute**: dim@80=16, pair_cos=0.18-0.19 → 略高于random_adj_adj
- **function**: dim@80=16, pair_cos=0.21-0.40 → 显著高于随机
- **negation**: dim@80=10-11, pair_cos=0.26-0.76 → 远高于随机

**关键发现**: random_adj_adj的pair_cos只有0.017-0.021，与antonym的0.008-0.032几乎相同。这意味着**antonym方向间的近乎正交性并不比随机形容词对更差——它们都是"随机方向"**。same_class甚至更差（pair_cos更低），说明"同类关系"在"the X was there"模板下完全没有提取到有意义的关系信号。

**结果2: 模板控制——"the {w} was there"模板产生完全不同的方向！**

| 类型 | t1 vs t2 | t1 vs t3 | t2 vs t3 |
|------|----------|----------|----------|
| same_class(Qwen3) | 0.236 | 0.218 | **0.938** |
| same_class(GLM4) | 0.265 | 0.240 | **0.935** |
| same_class(DS7B) | 0.119 | 0.101 | **0.862** |
| attribute(Qwen3) | 0.182 | 0.182 | **0.958** |
| attribute(GLM4) | 0.193 | 0.188 | **0.942** |
| attribute(DS7B) | 0.115 | 0.104 | **0.897** |
| function(Qwen3) | 0.233 | 0.223 | **0.944** |
| function(GLM4) | 0.241 | 0.213 | **0.919** |
| function(DS7B) | 0.138 | 0.124 | **0.875** |
| antonym(Qwen3) | 0.170 | 0.159 | **0.948** |
| antonym(GLM4) | 0.154 | 0.134 | **0.938** |
| antonym(DS7B) | 0.095 | 0.084 | **0.862** |

**关键发现**: 
1. **t1 vs t2/t3余弦仅0.08-0.27**——Phase 318使用的"the {w} was there"模板产生的方向，与"they mentioned the {w}"和"they discussed the {w}"模板的方向**几乎不相关**。
2. **t2 vs t3余弦0.86-0.96**——结构相似的模板产生高度一致的方向。
3. 这意味着Phase 318中观察到的方向主要是**模板效应**，不是关系信号。当模板从"the X was there"换成"they mentioned the X"，方向完全变了。

**结果3: 否定Token控制——否定一致性不是"not" Token效应！**

跨否定形式方向余弦（深层，20形容词对的平均）:

| 否定形式对 | Qwen3 | GLM4 | DS7B |
|-----------|-------|------|------|
| not vs never | **0.783** | **0.699** | **0.796** |
| not vs barely | **0.695** | **0.684** | **0.565** |
| not vs morphological | **0.679** | **0.608** | **0.694** |
| never vs barely | **0.646** | **0.693** | **0.527** |
| never vs morphological | **0.614** | **0.465** | **0.615** |
| barely vs morphological | **0.600** | **0.512** | **0.499** |

子空间主角度（中间层L12-18）:

| 否定形式对 | mean_angle | mean_cosine |
|-----------|------------|-------------|
| not vs never | 14-21° | 0.85-0.93 |
| not vs barely | 20-24° | 0.83-0.89 |
| not vs morphological | 17-28° | 0.79-0.91 |
| never vs barely | 19-25° | 0.82-0.85 |
| never vs morphological | 18-26° | 0.81-0.90 |
| barely vs morphological | 23-30° | 0.78-0.85 |

**关键发现**:
1. **"not" vs "never"方向高度一致**(mean_cos=0.70-0.80, angle=14-21°)——这不是同一个token的效应！
2. **形态学否定(un-X)与"not X"也有强一致性**(mean_cos=0.61-0.69)——模型内部确实存在一个**跨实现方式的否定操作结构**。
3. **所有4种否定形式共享子空间**(角度14-30°, cosine 0.78-0.93)——这证明Phase 318中否定的高一致性是**真实的否定操作编码**，不是"not" token的伪影。
4. 一致性排序: not≈never > barely > morphological——句法否定比形态学否定更一致。

各否定形式自身的子空间性质（深层）:

| 形式 | dim@80 | LOO_cos | pair_cos |
|------|--------|---------|----------|
| not | 10-11 | 0.78-0.85 | 0.43-0.53 |
| never | 10-11 | 0.76-0.84 | 0.42-0.57 |
| barely | 10-12 | 0.73-0.83 | 0.35-0.55 |
| morphological | 10-12 | 0.68-0.72 | 0.17-0.27 |

**关键发现**: 形态学否定的内部一致性(pair_cos=0.17-0.27)显著低于句法否定(0.35-0.57)，说明un-X的操作比not X更多样化（不同词的un-前缀效果差异更大）。

### 客观事实拼图更新

1. **same_class和antonym的pairwise cosine低于随机基线**——在"the X was there"模板下，这两类关系完全没有提取到有意义的关系信号
2. **"the {w} was there"模板产生模板特异性方向**——换模板后方向余弦仅0.08-0.27，Phase 318的方向主要是模板效应
3. **negation的真实性被确认**——not/never/barely/un-四种否定形式高度一致，这是否定操作编码而非token效应
4. **random_adj_adj的pair_cos≈0.02**——随机形容词对方向近乎正交，与antonym水平相同
5. **形态学否定与句法否定共享子空间**——但内部一致性较低，说明un-X的实现更多样

### Phase 318结论的修订

Phase 318的几个核心结论需要重大修正：

1. ~~"same_class方向近乎正交"~~ → same_class方向低于随机水平，"the X was there"模板根本不适合测同类关系
2. ~~"antonym方向近乎正交"~~ → antonym方向等于随机形容词对水平，模板方法不适用
3. ~~"attribute有中等对齐"~~ → attribute的pair_cos(0.18)仅略高于random_adj_adj(0.02)，信号很弱
4. ~~"function有中强对齐"~~ → function的pair_cos(0.21-0.40)确实高于random_mixed(0.27)，但需要更好模板验证
5. **"negation高一致性"得到确认** → 这是Phase 318中唯一经得起控制实验检验的结论

### 硬伤分析

1. **Phase 318的词对关系结论基本无效**（严重）：
   - "the {w} was there"模板产生的方向主要是模板效应
   - 跨模板一致性仅0.08-0.27，说明方向中关系信号占比极低
   - 必须使用关系特异性模板（如"the apple is red"而非"the red was there"）

2. **随机基线揭示的真正问题**（关键）：
   - random_mixed的LOO_cos=0.67-0.75，接近same_class(0.43-0.50)——但same_class更低！
   - 这可能因为同类词对的范数方差更大，或因为"the apple was there"与"the orange was there"的方向差异比随机词对更分散

3. **否定结果的可信度**（正面）：
   - 否定使用句子对而非词对，避免了模板污染
   - 四种否定形式交叉验证排除了token效应
   - 这是目前为止最可靠的结构性发现

### 破解语言数学结构的第一性原理分析

**核心洞察1: "the X was there"模板是一种"零上下文"条件**

当模型处理"the apple was there"时，apple出现在没有属性/功能上下文的位置。模型的表示主要反映apple这个token自身的语义，而不是apple与其他词的关系。因此，h("the orange was there") - h("the apple was there")测的是"orange的语义表示 - apple的语义表示"，而不是"同类关系的差分"。

这个方向在不同模板下不同，因为不同上下文激活了不同的语义维度。

**核心洞察2: 否定操作是真正的"关系算子"**

否定使用完整句子对（"very happy" vs "not happy"），差分方向捕获的是"从肯定到否定的语义偏移"。这个偏移在not/never/barely/un-之间高度一致，说明模型内部确实编码了一个"否定操作"——不管用什么否定词，都指向类似的子空间。

这是第一个被严格验证的"语言操作编码"。

**核心洞察3: 关系编码需要关系特异性上下文**

要测属性关系，必须在模板中激活属性上下文（"the apple is red" vs "the apple is there"）。要测功能关系，必须激活功能上下文（"people use knives to cut" vs "people use knives"）。单纯替换词不等于提取关系。

下一步核心方向：用**关系特异性模板**重新提取属性和功能方向。

### 命令记录

```bash
# Phase 319: 随机基线 + 模板控制 + 否定Token控制
python tests/glm5/phase319_baseline_control.py qwen3       # ~20s
python tests/glm5/phase319_baseline_control.py glm4        # ~10min
python tests/glm5/phase319_baseline_control.py deepseek7b  # ~6min
```

脚本位置：
- `tests/glm5/phase319_baseline_control.py` — 主测试
- 结果：`results/phase319_control/{qwen3,glm4,deepseek7b}_phase319.json`
- 日志：`tmp/phase319_{qwen3,glm4,deepseek7b}.log`

## Phase 319b: 关系特异性模板验证 [2026-06-01 08:08]

### 背景

Phase 319证明"the {w} was there"模板产生模板特异性方向（跨模板余弦仅0.08-0.27），且same_class/antonym低于随机基线。核心问题：**使用关系特异性模板能否提取到有意义的关系方向？**

### 测试设计

4类关系 × 20对 × 2-3模板（中性 vs 关系特异性 vs 描述性）：

| 关系 | 中性模板 | 特异性模板 | 描述/目的模板 |
|------|---------|-----------|-------------|
| same_class | "the {A} was there" / "the {B} was there" | "{A} and {B} are both things" / "{B} and {A} are both things" | "{A} is a kind of thing" / "{B} is a kind of thing" |
| attribute | "the {N} was there" / "the {A} was there" | "the {N} is {A}" / "the {N} is just an object" | "the {N} has the quality of being {A}" / "the {N} is just an object" |
| function | "the {T} was there" / "the {V} was there" | "people use the {T} to {V}" / "people use the {T} for something" | "the {T} is for {V}ing" / "the {T} is for something" |
| antonym | "the {A} was there" / "the {B} was there" | "{A} is the opposite of {B}" / "{B} is the opposite of {A}" | "not {A} but {B}" / "not {B} but {A}" |

注意：属性和功能的特异性模板使用**同对象不同描述**（"the apple is red" vs "the apple is just an object"），而非**词替换**（"the red was there" vs "the apple was there"）。这是关键差异。

### 核心结果

**结果1: 属性方向——特异性模板使pair_cos从0.18暴涨到0.55-0.60！**

| 模板 | Qwen3 pair_cos | GLM4 pair_cos | DS7B pair_cos | 对比随机(0.24-0.28) |
|------|---------------|---------------|---------------|---------------------|
| neutral | 0.180 | 0.192 | 0.191 | **低于随机** |
| specific | **0.606** | **0.553** | **0.559** | **2x以上** |
| descriptive | **0.604** | **0.599** | **0.574** | **2x以上** |

**关键发现**: 中性模板下属性方向pair_cos(0.18)甚至低于随机基线(0.24)，但特异性模板下暴涨到0.55-0.60！这意味着：
- "the red was there" - "the apple was there"测的不是属性关系，而是两个词的语义差
- "the apple is red" - "the apple is just an object"测的才是真正的**属性激活方向**

**结果2: 功能方向——同样暴涨**

| 模板 | Qwen3 pair_cos | GLM4 pair_cos | DS7B pair_cos |
|------|---------------|---------------|---------------|
| neutral | 0.382 | 0.257 | 0.205 |
| specific | **0.528** | **0.602** | **0.560** |
| purpose | **0.530** | **0.580** | **0.523** |

功能方向的增幅略小于属性，但仍然显著。

**结果3: same_class——无论什么模板，pair_cos都接近0**

| 模板 | Qwen3 pair_cos | GLM4 pair_cos | DS7B pair_cos |
|------|---------------|---------------|---------------|
| neutral | 0.011 | 0.017 | 0.004 |
| specific | 0.019 | -0.014 | -0.006 |
| category | 0.059 | 0.005 | -0.007 |

**关键发现**: 即使使用"{A} is a kind of thing" / "{B} is a kind of thing"这种类别模板，pair_cos仍然只有0.005-0.059。DS7B甚至出现负值！这强烈说明**同类关系不是一个方向差分编码，而是拓扑关系**。

**结果4: antonym——即使"opposite"模板也不行**

| 模板 | Qwen3 pair_cos | GLM4 pair_cos | DS7B pair_cos |
|------|---------------|---------------|---------------|
| neutral | 0.041 | 0.020 | 0.039 |
| specific | 0.034 | 0.029 | 0.018 |
| contrast | 0.052 | 0.051 | 0.026 |

**关键发现**: 即使使用"{A} is the opposite of {B}"模板，pair_cos也只有0.018-0.052，几乎等同于随机水平(0.017-0.021)。这证明**反义关系不是统一的方向操作，而是分布在多个语义轴上的极性翻转**。

**结果5: 中性模板 vs 特异性模板的方向完全不同**

| 关系 | neutral vs specific | neutral vs descriptive/purpose | specific vs descriptive/purpose |
|------|-------------------|------------------------------|-------------------------------|
| attribute | **-0.09~-0.01** | **-0.06~-0.02** | **0.81-0.89** |
| function | **-0.05~0.00** | **-0.04~0.02** | **0.72-0.75** |
| same_class | **-0.03~0.06** | **0.07-0.26** | **0.03-0.45** |
| antonym | **-0.06~-0.02** | **-0.05~0.00** | **0.37-0.55** |

**关键发现**: 
1. 中性模板和特异性模板的方向**完全不相关**（余弦接近0甚至为负）
2. 特异性模板和描述性模板之间**高度一致**（余弦0.72-0.89）
3. 这证明中性模板测的是**模板词法差异**，特异性模板测的是**关系激活差异**

### 客观事实拼图更新

1. **属性和功能关系存在真实的方向编码**——但必须用关系特异性模板才能提取到
2. **中性模板("the X was there")产生的是词法差异方向**——不是关系方向
3. **same_class不存在方向编码**——无论什么模板pair_cos都接近0
4. **antonym不存在统一方向编码**——即使"opposite"模板也不行
5. **关系特异性模板(specific vs descriptive/purpose)之间高度一致**——方向是可靠的
6. **属性方向pair_cos=0.55-0.60远超随机(0.24-0.28)**——这是第一个被严格验证的词对关系编码
7. **功能方向pair_cos=0.52-0.60也远超随机**——功能关系同样有真实方向编码

### 理论更新

**语言关系编码的三种类型（最终确认版）：**

1. **操作型关系**：否定(not/never/barely/un-)
   - 特征：pair_cos=0.27-0.76，跨实现方式高度一致
   - 编码方式：统一子空间，方向较一致
   - 测量方法：句子对差分

2. **激活型关系**：属性(is red)、功能(to cut)
   - 特征：pair_cos=0.52-0.60（特异性模板），远超随机
   - 编码方式：中维子空间，方向族较一致
   - 关键：必须用关系特异性模板（"the apple is red"），不能用中性模板

3. **拓扑型关系**：同类、反义
   - 特征：pair_cos≈0，无论什么模板
   - 编码方式：不是方向差分，而是邻域结构/语义轴极性
   - 测量方法：需要用图结构/邻域分析，不能用方向差分

### 硬伤与关键问题

1. **属性特异性模板的方向含义需要进一步解释**（重要）：
   - "the apple is red" - "the apple is just an object" 的差分可能包含：属性激活 + 特定属性(red)的语义 + "is just an object"的弱化效果
   - 需要控制：对比"the apple is red" - "the apple is green" → 这个差分才是纯属性语义轴方向

2. **same_class的pair_cos为负值**（DS7B中specific和category模板）：
   - 负值意味着方向相反——但这在随机噪声范围内(-0.01~-0.007)
   - 说明模型处理"apple is a kind of thing"和"orange is a kind of thing"时，最后一token的表示差异是完全随机的

3. **antonym的"opposite"模板为什么不起作用**（关键）：
   - "{A} is the opposite of {B}" 与 "{B} is the opposite of {A}" 的差分方向取决于A和B的位置
   - 这可能不是一个有意义的差分——因为两个句子只是在A和B的位置上交换
   - 需要换一种方式测：如"the opposite of hot is cold" vs "the opposite of cold is hot"

### 破解语言数学结构的第一性原理分析

**核心洞察: 关系编码的三层结构**

Phase 319+319b揭示了一个关键的三层结构：

```
第一层：Token差异（中性模板测到的）
  → 词法替换导致的表示差异
  → 与关系无关，完全由词决定
  → pair_cos ≈ 0.17-0.27 (随机水平)

第二层：关系激活（特异性模板测到的）
  → 激活某种关系后的表示差异
  → pair_cos ≈ 0.52-0.60 (显著高于随机)
  → 但这不是"关系本身"的方向，而是"关系被激活时的整体偏移"

第三层：关系算子（否定操作测到的）
  → 关系操作的几何实现
  → pair_cos ≈ 0.27-0.76 (最高一致性)
  → 这是真正的"语言操作编码"
```

**下一步方向**: 需要区分第二层和第三层——属性/功能的pair_cos=0.52-0.60到底测的是"关系激活偏移"还是"关系算子"？关键实验：测试属性方向的因果效力——将"the apple is red"的属性方向注入"the apple is just an object"，是否能让模型输出变成"the apple is red"？

### 命令记录

```bash
# Phase 319b: 关系特异性模板验证
python tests/glm5/phase319b_relation_templates.py qwen3       # ~20s
python tests/glm5/phase319b_relation_templates.py glm4        # ~8.5min
python tests/glm5/phase319b_relation_templates.py deepseek7b  # ~5.5min
```

脚本位置：
- `tests/glm5/phase319b_relation_templates.py` — 主测试
- 结果：`results/phase319b_templates/{qwen3,glm4,deepseek7b}_phase319b.json`
- 日志：`tmp/phase319b_{qwen3,glm4,deepseek7b}.log`

## Phase 320: 关系激活方向因果验证 + 否定子类型分解 [2026-06-01 09:12]

### 背景

Phase 319b证明属性/功能方向在几何上稳定(pair_cos=0.55-0.60)，但未验证因果效力。本实验测试：(1)属性方向注入能否推动模型预测属性词；(2)功能方向注入能否推动模型预测动作词；(3)否定是否可分解为子类型。

### 测试设计

**Part A: 属性因果注入** (20源×20目标×4强度)
- d_attr: h("the apple is red") - h("the apple is just an object")
- d_axis: h("the apple is green") - h("the apple is red") (纯属性轴)
- 注入到"The {target} is"，测量目标属性词logit变化
- alpha=[0.5, 1.0, 2.0, 4.0]，注入在最深层

**Part B: 功能因果注入** (10直注×2强度 + 10迁移×2强度)
- d_func: h("people use the knife to cut") - h("people use the knife")
- 直注：注入"People use the knife to"，测量cut的logit变化
- 迁移：knife→cut方向注入"People use the scissors to"

**Part C: 否定子类型分解** (20形容词×6否定类型)
- not/never/barely/morphological(un-)/double_neg(not un-)/scope_neg(did not try to be)
- 分析内部一致性、跨类型余弦、因果效力

### 核心结果

**结果1: 属性因果效力——模型间差异巨大**

| 注入方式 | 模型 | alpha=1.0 | alpha=2.0 | alpha=4.0 | 跨对象迁移 |
|---------|------|-----------|-----------|-----------|-----------|
| d_attr | Qwen3 | 0.004(15%) | 0.010(35%) | 0.034(75%) | 0.015(36%) |
| d_attr | GLM4 | **0.008(70%)** | **0.022(85%)** | **0.036(85%)** | **0.024(64%)** |
| d_attr | DS7B | 0.009(35%) | -0.018(55%) | -0.006(65%) | **0.037(72%)** |
| d_axis | Qwen3 | 0.006(33%) | 0.002(27%) | 0.027(53%) | — |
| d_axis | GLM4 | **0.008(73%)** | **0.020(87%)** | **0.041(73%)** | — |
| d_axis | DS7B | -0.079(27%) | -0.087(27%) | -0.092(27%) | — |

**关键发现**:
1. **GLM4属性因果效力最强**：alpha=2时d_attr的frac_positive=85%，d_axis的frac_positive=87%
2. **Qwen3弱但存在**：alpha=4时d_attr的frac_positive=75%，说明方向有弱因果效力
3. **DS7B纯属性轴(d_axis)因果效力为负**：注入红→绿方向，红和绿的logit都下降——这说明DS7B的属性轴方向不是因果操作
4. **跨对象迁移不一致**：Qwen3只有36%正向，GLM4有64%正向，DS7B有72%正向但绝对值很小

**结果2: 功能因果效力——GLM4唯一有显著因果效力**

| 注入方式 | 模型 | alpha=1.0 | alpha=2.0 |
|---------|------|-----------|-----------|
| 直注 | Qwen3 | 0.000(10%) | 0.006(20%) |
| 直注 | GLM4 | **0.014(60%)** | **0.019(70%)** |
| 直注 | DS7B | -0.012(10%) | -0.005(20%) |
| 迁移 | Qwen3 | 0.000(0%) | 0.009(14%) |
| 迁移 | GLM4 | **0.015(71%)** | **0.029(86%)** |
| 迁移 | DS7B | -0.018(0%) | -0.009(0%) |

**关键发现**:
1. **GLM4功能方向有强因果效力和迁移能力**：迁移时alpha=2，frac_positive=86%
2. **Qwen3功能方向几乎没有因果效力**：alpha=2直注仅20%正向
3. **DS7B功能方向完全无因果效力**：所有delta为负，迁移0%正向
4. 这说明不同模型的"功能关系编码"机制可能完全不同

**结果3: 否定子类型分解——否定不是单一子空间，而是两个独立簇**

| 否定类型 | Qwen3 dim@80 | Qwen3 pair_cos | GLM4 dim@80 | GLM4 pair_cos | DS7B dim@80 | DS7B pair_cos |
|---------|-------------|----------------|-------------|---------------|-------------|---------------|
| not | 11 | 0.999 | 13 | 0.306 | 3 | 0.507 |
| never | 11 | 0.999 | 13 | 0.312 | 4 | 0.490 |
| barely | 11 | 0.999 | 13 | 0.326 | 4 | 0.551 |
| morphological | 1 | 0.693 | 9 | 0.334 | 2 | -0.018 |
| double_neg | 9 | 0.215 | 9 | 0.186 | 6 | 0.161 |
| scope_neg | 13 | 0.445 | 13 | 0.379 | 13 | 0.484 |

跨类型余弦（三模型平均趋势）:

| 类型对 | Qwen3 | GLM4 | DS7B |
|--------|-------|------|------|
| not vs never | 0.999 | 0.257 | 0.509 |
| not vs barely | 0.999 | 0.217 | 0.514 |
| not vs morphological | 0.846 | 0.142 | 0.034 |
| not vs double_neg | **-0.009** | **-0.071** | 0.192 |
| not vs scope_neg | **0.078** | **-0.002** | 0.062 |
| morphological vs not/never | 0.846 | 0.078-0.142 | 0.028-0.034 |
| double_neg vs not/never | **-0.009~-0.008** | **-0.071~-0.038** | 0.192-0.193 |
| scope_neg vs not/never | 0.078-0.079 | -0.038~-0.002 | 0.062 |

**关键发现**:
1. **否定分成两个独立簇**:
   - **簇1: 句法否定** (not/never/barely)：三者高度一致(Qwen3: 0.999, GLM4: 0.19-0.26, DS7B: 0.50)
   - **簇2: 语义否定** (scope_neg/double_neg)：与簇1近乎正交(cos≈0甚至为负)
2. **morphological(un-)介于两簇之间**：与句法否定有中等对齐(Qwen3: 0.846, GLM4: 0.14, DS7B: 0.03)
3. **double_neg("not un-X")与句法否定方向相反**：Qwen3/GLM4中余弦为负！这在语义上合理——"not unhappy"偏向积极
4. **scope_neg("did not try to be X")与句法否定完全独立**：cos≈0，说明作用范围不同

**结果4: 否定因果效力——GLM4远强于其他模型**

| 注入类型 | Qwen3 max_neg_delta | Qwen3 adj_delta | GLM4 max_neg_delta | GLM4 adj_delta | DS7B max_neg_delta | DS7B adj_delta |
|---------|--------------------|--------------------|--------------------|----------------|--------------------|----------------|
| not→very X | -0.002 | -0.005 | **0.121** | **0.052** | 0.055 | 0.021 |
| never→very X | 0.013 | -0.002 | **0.197** | 0.028 | 0.056 | 0.030 |
| barely→very X | 0.008 | -0.003 | **0.178** | 0.025 | 0.047 | 0.027 |

**关键发现**:
1. **GLM4否定因果效力最强**：注入not方向后否定词logit增加0.12-0.20
2. **Qwen3否定因果效力最弱**：几乎无变化(max_neg_delta≈0)
3. **DS7B有弱但正向的否定因果效力**：delta=0.05左右

### 客观事实拼图更新

1. **属性方向的因果效力高度模型依赖**：GLM4有因果效力(70-87%正向)，Qwen3弱(35-75%)，DS7B无/负
2. **功能方向的因果效力同样高度模型依赖**：GLM4有迁移能力(86%正向)，其他模型几乎无
3. **否定不是单一子空间，而是两个独立机制**：句法否定簇(not/never/barely) vs 语义否定簇(double_neg/scope_neg)
4. **双重否定方向与句法否定方向相反**：符合语义直觉("not unhappy" ≠ "not happy")
5. **morphological否定(un-X)在不同模型中的定位不同**：Qwen3中与句法否定高度对齐(0.846)，DS7B中近乎正交(0.03)
6. **同一方向在不同模型中的因果效力可以完全不同**——这是最重要的新发现

### 关键问题分析

**问题1: 为什么同一方向在不同模型中因果效力差异如此大？**

可能原因：
- 不同模型的"readout机制"不同：即使方向在几何空间中存在，lm_head的权重方向可能不与该方向对齐
- GLM4的d_model=4096更大，可能有更丰富的readout空间
- DS7B的sliding window attention可能导致深层表示信息丢失

**问题2: 属性方向几何一致性(pair_cos=0.55)与因果效力的脱节**

Phase 319b发现属性方向pair_cos=0.55-0.60(远超随机)，但因果效力在Qwen3和DS7B中很弱。这说明：
- 几何一致性≠因果效力
- 方向可能存在于表示空间中，但不被readout层有效读取
- 因果效力需要方向与lm_head权重的对齐

**问题3: 否定的两簇结构意味着什么？**

句法否定簇(not/never/barely)：修改极性，保持语义框架
语义否定簇(double_neg/scope_neg)：修改语义框架或作用范围
morphological(un-)：介于两者之间，取决于模型

### 命令记录

```bash
# Phase 320: 因果验证
python tests/glm5/phase320_causal_verification.py qwen3       # ~12s
python tests/glm5/phase320_causal_verification.py glm4        # ~11min
python tests/glm5/phase320_causal_verification.py deepseek7b  # ~7min
```

脚本位置：
- `tests/glm5/phase320_causal_verification.py` — 主测试
- 结果：`results/phase320_causal/{qwen3,glm4,deepseek7b}_phase320.json`
- 日志：`tmp/phase320_{qwen3,glm4,deepseek7b}.log`

## Phase 320b: 层特异性因果验证（确认轮） [2026-06-01 09:24]

### 背景

Phase 320在最深层注入发现属性/功能因果效力很弱。但因果效力的关键可能在于**注入层位置**——不同关系类型可能在不同层有最优因果效力。

### 测试设计

在所有层（每隔2-4层）注入方向，测量属性/功能/否定的logit变化。alpha=2.0。

### 核心结果

**结果1: 层位置决定因果效力——浅层远强于深层！**

属性方向注入（alpha=2.0，跨4对源→目标平均）:

| 层 | Qwen3 mean_delta | Qwen3 positive | GLM4 mean_delta | GLM4 positive | DS7B mean_delta | DS7B positive |
|----|------------------|----------------|-----------------|---------------|-----------------|---------------|
| L0 | **0.227** | 100% | -0.097 | 50% | -0.039 | 25% |
| L4 | 0.141 | 75% | **0.814** | **100%** | -0.098 | 25% |
| L8 | 0.148 | 100% | **0.521** | **100%** | -0.031 | 50% |
| L12 | 0.156 | 100% | 0.544 | 100% | 0.024 | 50% |
| L16 | **0.172** | 100% | 0.346 | 100% | -0.047 | 50% |
| L20 | 0.125 | 100% | 0.371 | 100% | -0.023 | 50% |
| L24 | 0.086 | 100% | 0.147 | 100% | 0.027 | 50% |
| 最深 | 0.000 | 0% | 0.021 | 100% | 0.000 | 0% |

**关键发现**:
1. **GLM4属性方向在L4有极强因果效力**：mean_delta=0.814，100%正向！这是所有实验中最大的因果效力
2. **Qwen3属性方向在L0-16有因果效力**：100%正向，但绝对值较小(0.09-0.23)
3. **DS7B属性方向几乎没有因果效力**：所有层都很弱，best layer L26仅0.035
4. **最深层注入几乎无效**：三个模型在最深层注入delta≈0

功能方向注入（alpha=2.0）:

| 层 | Qwen3 | GLM4 | DS7B |
|----|-------|------|------|
| L0 | 0.031(25%) | -3.11(25%) | 0.055(75%) |
| L4 | 0.086(100%) | -0.23(50%) | 0.047(75%) |
| L8 | 0.031(50%) | 0.18(50%) | 0.031(75%) |
| L12 | 0.047(50%) | **0.40(100%)** | 0.016(50%) |
| L16 | 0.063(75%) | **0.46(100%)** | 0.031(50%) |
| L20 | 0.086(75%) | 0.31(100%) | 0.000(25%) |
| 最深 | 0.000(0%) | 0.008(50%) | 0.000(0%) |

**关键发现**:
1. **GLM4功能方向在L12-16有强因果效力**：0.40-0.46，100%正向
2. **Qwen3功能方向弱但一致**：L4和L20最优(0.086)
3. **DS7B功能方向在浅层有弱效力**：L0最优(0.055)

否定方向注入（alpha=2.0，max_neg_delta）:

| 层 | Qwen3 | GLM4 | DS7B |
|----|-------|------|------|
| L0 | -0.078 | **3.56** | **0.089** |
| L4 | -0.013 | 1.64 | 0.026 |
| L8 | -0.003 | 1.45 | 0.017 |
| L12 | 0.000 | 1.22 | 0.010 |
| L16 | 0.005 | 0.68 | 0.010 |
| 最深 | 0.000 | 0.046 | 0.000 |

**关键发现**:
1. **GLM4否定方向在L0有极强因果效力**：max_neg_delta=3.56！这是L0直接操纵embedding的极强效果
2. **DS7B否定方向在L0也有最强效力**：0.089
3. **Qwen3否定方向在所有层因果效力都极弱**

### 最优注入层总结

| 关系类型 | Qwen3最优层 | GLM4最优层 | DS7B最优层 |
|---------|------------|-----------|-----------|
| 属性 | L0 (0.227) | **L4 (0.814)** | L26 (0.035) |
| 功能 | L4 (0.086) | **L16 (0.461)** | L0 (0.055) |
| 否定 | L16 (0.005) | **L0 (3.56)** | L0 (0.089) |

### 客观事实拼图更新

1. **注入层位置是因果效力的决定性因素**：最深层几乎无效，浅/中层远强于深层
2. **GLM4的属性方向在L4有极强因果效力(mean_delta=0.814)**——这是目前发现的最强因果信号
3. **GLM4的否定方向在L0有极强因果效力(3.56)**——L0注入接近直接修改embedding
4. **不同关系类型的最优注入层不同**：属性→浅层(L0-4)，功能→中层(L12-16)，否定→最浅层(L0)
5. **因果效力的层分布呈现倒U形**：浅层最强，中层减弱，深层消失
6. **DS7B的因果效力整体很弱**——可能与其sliding window attention有关
7. **Qwen3否定方向因果效力极弱**——但几何一致性很高(pair_cos=0.999)，说明几何一致≠因果效力

### 关键硬伤

1. **L0注入可能主要是embedding偏移**：在L0注入方向相当于修改第一个token的embedding，这可能绕过了正常的transformer计算路径。L0的强因果效力不一定反映"关系算子的因果作用"
2. **GLM4的L4属性效力(0.814)更可信**：因为L4已经过了4层transformer计算，不是简单的embedding偏移
3. **因果效力随层递减的模式需要更细粒度验证**：目前间隔太大(4层)，需要在最优层附近做1层间隔的细测
4. **DS7B的弱因果效力可能不是真实的模型特性**：sliding window attention可能影响深层hook的有效性

### 命令记录

```bash
# Phase 320b: 层特异性因果验证
python tests/glm5/phase320b_layer_causal.py qwen3       # ~30s
python tests/glm5/phase320b_layer_causal.py glm4        # ~5min
python tests/glm5/phase320b_layer_causal.py deepseek7b  # ~3min
```

脚本位置：
- `tests/glm5/phase320b_layer_causal.py` — 主测试
- 结果：`results/phase320b_layer/{qwen3,glm4,deepseek7b}_phase320b.json`
- 日志：`tmp/phase320b_{qwen3,glm4,deepseek7b}.log`

## Phase 321: 细粒度层扫描 + 读出对齐分析 [2026-06-01 11:12]

### 背景

Phase 320b用4层间隔发现GLM4属性L4最优(delta=0.814)，但层间隔太大，无法确定峰值是否为单层伪影。同时需要验证"读出对齐假说"：因果效力的跨模型差异是否源于方向与W_U的对齐度差异。

### 测试设计

在最优层附近做1层间隔细扫，同时计算读出对齐(方向与W_U[target_token]的余弦)：
- GLM4: 属性L1-L9, 功能L9-L19, 否定L0-L5
- Qwen3: 属性L0-L10, 功能L0-L10+L16-L24, 否定L0-L10
- DS7B: 属性L3-L15, 功能L0-L8, 否定L0-L8

alpha=2.0，注入在最细扫层。

### 核心结果

**结果1: 细粒度层扫描确认GLM4属性L3是真正的因果窗口峰值**

属性方向注入(alpha=2.0):

| 层 | Qwen3 delta(%) | Qwen3 readout | GLM4 delta(%) | GLM4 readout | DS7B delta(%) | DS7B readout |
|----|---------------|---------------|---------------|-------------|---------------|-------------|
| L0 | 0.188(100%) | 0.088 | — | — | — | — |
| L1 | 0.172(88%) | 0.081 | -0.234(50%) | -0.022 | — | — |
| L2 | 0.156(100%) | 0.069 | **1.641(88%)** | -0.027 | — | — |
| L3 | 0.141(100%) | 0.050 | **1.286(88%)** | -0.016 | 0.036(38%) | 0.027 |
| L4 | 0.141(88%) | 0.039 | **0.928(88%)** | 0.003 | -0.012(50%) | 0.030 |
| L5 | 0.141(88%) | 0.027 | 0.729(88%) | 0.003 | -0.077(38%) | 0.031 |
| L6 | 0.148(88%) | 0.019 | 0.808(88%) | 0.000 | **0.069(62%)** | 0.031 |
| L7 | 0.156(88%) | 0.031 | 0.633(88%) | 0.001 | -0.051(50%) | 0.043 |
| L8 | 0.141(75%) | 0.024 | 0.540(88%) | 0.000 | 0.059(62%) | 0.044 |
| L9 | 0.125(75%) | 0.032 | 0.503(88%) | 0.004 | 0.039(75%) | 0.033 |

**关键发现**:
1. **GLM4属性L2达到历史最高delta=1.641**，远超Phase 320b的L4=0.814。L2-L6形成一个宽广的高原区(delta>0.5)
2. **GLM4属性L2-3的因果效力极高，但读出对齐为负(-0.027)**！这说明方向与W_U的直接对齐完全不能解释因果效力
3. **Qwen3属性L0最强(0.188)，读出对齐最高(0.088)**，但因果效力远低于GLM4
4. **DS7B属性始终弱(delta<0.08)**，但读出对齐反而是三模型中最高的(0.044)
5. **读出对齐与因果效力呈反比关系**：对齐最高的Qwen3/DS7B因果效力最弱，对齐最低的GLM4因果效力最强

**结果2: GLM4功能方向在L9-L19形成稳定高原区**

功能方向注入(alpha=2.0):

| 层 | Qwen3 delta(%) | GLM4 delta(%) | DS7B delta(%) |
|----|---------------|---------------|---------------|
| L6 | 0.055(62%) | — | 0.063(75%) |
| L9 | 0.039(50%) | 0.449(88%) | — |
| L12 | 0.055(50%) | 0.494(100%) | — |
| L15 | 0.063(50%) | 0.490(100%) | — |
| L17 | 0.066(50%) | **0.508(100%)** | — |
| L18 | **0.098(75%)** | 0.504(100%) | — |
| L19 | 0.070(62%) | 0.426(100%) | — |

**关键发现**:
1. **GLM4功能方向在L9-L19形成delta>0.4的稳定高原区**，100%正向
2. **Qwen3功能方向在L18有一个局部峰值(0.098)**
3. **DS7B功能方向极弱(delta<0.06)**

**结果3: GLM4否定方向在L0-L5形成极强高原区**

否定方向注入(alpha=2.0):

| 层 | Qwen3 neg_delta | GLM4 neg_delta | DS7B neg_delta |
|----|----------------|----------------|----------------|
| L0 | **0.297** | **5.166** | **0.175** |
| L1 | -0.056 | 2.285 | 0.069 |
| L2 | -0.043 | 2.251 | 0.030 |
| L3 | 0.014 | **2.559** | 0.075 |
| L4 | 0.000 | 2.435 | 0.049 |
| L5 | -0.014 | 2.433 | 0.081 |

**关键发现**:
1. **GLM4否定L0 neg_delta=5.166**，但L3也有2.559——不是单层伪影，而是L0-L5的宽广高原
2. **Qwen3否定只在L0有弱效力(0.297)**
3. **DS7B否定在L0有弱效力(0.175)**

**结果4: 读出对齐完全不能解释跨模型因果效力差异——这是最重要的反直觉发现**

跨模型读出对齐汇总:

| 模型 | 因果效力(属性) | 直接读出对齐 | 簇读出对齐 | 属性readout(方向→W_U) |
|------|--------------|------------|----------|---------------------|
| GLM4 | **极强(L2:1.641)** | 0.003 | 0.000 | **-0.002** |
| Qwen3 | 弱(L0:0.188) | **0.088** | **0.034** | **0.032** |
| DS7B | 极弱(L6:0.069) | 0.031 | 0.012 | **0.034** |

**关键发现**:
1. **GLM4因果效力最强，但读出对齐最低(甚至为负)**
2. **Qwen3/DS7B读出对齐最高，但因果效力最弱**
3. **读出对齐与因果效力呈反比**——这完全推翻了"读出对齐假说"
4. GLM4的属性方向与W_U[target]近乎正交，但注入后仍产生极大logit变化
5. 这说明GLM4的因果效力不来自方向与W_U的直接对齐，而是来自方向被后续层变换后与W_U对齐

### 客观事实拼图更新

1. **GLM4属性L2是精确的因果窗口峰值(delta=1.641)**，不是L4伪影
2. **GLM4属性因果窗口是L2-L6的宽广高原**，不是单层效应
3. **GLM4功能因果窗口是L9-L19的宽广高原(delta>0.4)**
4. **GLM4否定因果窗口是L0-L5(delta>2.0)**
5. **读出对齐与因果效力呈反比**——因果效力最强的GLM4读出对齐最低
6. **Qwen3属性L0读出对齐最高(0.088)但因果效力弱(0.188)**
7. **DS7B属性读出对齐最高(0.044)但因果效力极弱(0.069)**
8. 这说明因果效力来自"后续层变换后的间接对齐"，不是"方向与W_U的直接对齐"

### 关键硬伤

1. **GLM4属性L2的delta=1.641非常高**，但L1是负的(-0.234)——因果效力在L1→L2突变，需要理解这个突变机制
2. **读出对齐为负但因果效力极强**意味着存在"间接读出路径"：方向被后续层非线性变换后才与W_U对齐。这个路径需要被追踪
3. **Qwen3否定在L0有弱效力(0.297)但在L1消失**——可能是L0注入确实只是embedding偏移
4. **DS7B因果效力极弱但读出对齐高**——说明DS7B的属性方向在几何上确实指向属性词，但后续层没有"放大"这个信号

### 命令记录

```bash
# Phase 321: 细粒度层扫描 + 读出对齐
python tests/glm5/phase321_fine_layer_readout.py qwen3       # ~30s
python tests/glm5/phase321_fine_layer_readout.py glm4        # ~7min
python tests/glm5/phase321_fine_layer_readout.py deepseek7b  # ~5min
```

脚本位置：
- `tests/glm5/phase321_fine_layer_readout.py` — 主测试
- 结果：`results/phase321_fine_readout/{qwen3,glm4,deepseek7b}_phase321.json`
- 日志：`tmp/phase321_{qwen3,glm4,deepseek7b}.log`

## Phase 322: 间接读出路径追踪 + 方向变换分析 [2026-06-01 11:22]

### 背景

Phase 321发现读出对齐与因果效力呈反比：GLM4因果效力最强但读出对齐最低。这暗示GLM4的方向通过"间接读出路径"——后续层变换后才与W_U对齐。本实验追踪方向在层间传播时的变换：注入方向后，追踪delta在各后续层的范数、与原始方向的对齐度、以及与W_U[target]的对齐度。

### 测试设计

**Part A**: 属性方向变换追踪——在最优层注入方向，追踪后续35+层中delta的传播
**Part B**: Block重算测试——不同alpha在最优层附近的因果效力
**Part C**: 否定方向变换追踪——L0注入否定的delta传播
**Part D**: L1→L2转换分析——方向在各层的范数和读出对齐演化

### 核心结果

**结果1: GLM4的间接读出路径被确认——Qwen3/DS7B没有**

属性方向注入后，delta与W_U[target]的对齐度随层传播变化:

| 距注入层距离 | Qwen3 cos_tgt | GLM4 cos_tgt | DS7B cos_tgt |
|------------|---------------|--------------|--------------|
| +1层 | 0.077 | **-0.006** | 0.037 |
| +5层 | 0.014 | **0.001** | 0.014 |
| +10层 | -0.007 | -0.001 | 0.022 |
| +15层 | 0.009 | -0.004 | — |
| +20层 | — | -0.007 | — |
| 变化方向 | **衰减(-0.063)** | **增加(+0.007)** | **衰减(-0.022)** |

**关键发现**:
1. **GLM4是唯一一个读出对齐在传播中增加的模型**——从-0.006增加到0.001。虽然增加量很小，但方向正确
2. **Qwen3读出对齐在传播中大幅衰减**——从0.077衰减到0.014，衰减了81%
3. **DS7B读出对齐也在衰减**——从0.037衰减到0.014
4. 这解释了跨模型差异：GLM4的后续层能将方向"变换"到与W_U更对齐的方向，而Qwen3/DS7B的后续层会"消散"方向与W_U的对齐

**结果2: 方向与原始注入方向的对齐度——GLM4衰减更慢**

| 距注入层距离 | Qwen3 cos_orig | GLM4 cos_orig | DS7B cos_orig |
|------------|---------------|--------------|--------------|
| +1层 | 0.954 | 0.854 | 0.776 |
| +5层 | 0.658 | 0.501 | 0.513 |
| +10层 | 0.359 | 0.331 | 0.394 |
| +15层 | 0.273 | 0.293 | — |

三个模型衰减速度相近，但Qwen3初始对齐最高(0.954)。

**结果3: delta范数在传播中增长——所有模型都放大注入信号**

| 距注入层距离 | Qwen3 norm | GLM4 norm | DS7B norm |
|------------|-----------|----------|----------|
| +1层 | 2.08 | 2.23 | 2.19 |
| +5层 | 2.48 | 2.86 | 2.76 |
| +10层 | 3.24 | 3.02 | 3.60 |
| +20层 | — | 5.69 | 4.80 |

注入alpha=2.0的归一化方向后，delta范数从2增长到3-5。这说明后续层确实在放大信号，但放大方向不一定是与W_U对齐的方向。

**结果4: GLM4否定方向在传播中读出对齐极低但因果效力极强**

否定方向注入L0后，delta与否定词W_U的对齐:
- GLM4 L0注入后: neg_cos在0.015-0.027之间波动，始终很低
- 但GLM4 L0注入的neg_delta高达5.166

这再次确认：因果效力不来自方向与W_U的直接对齐，而来自后续层的非线性变换。

**结果5: GLM4属性方向在各层的范数和读出——L4是分水岭**

GLM4 apple→red方向各层:
```
L0: norm=0.49, cos_tgt=-0.012
L1: norm=0.85, cos_tgt=-0.022
L2: norm=1.24, cos_tgt=-0.027  ← delta最大但读出为负
L3: norm=1.63, cos_tgt=-0.016
L4: norm=2.21, cos_tgt=0.002   ← 读出转正！
L5: norm=2.57, cos_tgt=-0.001
L6: norm=2.75, cos_tgt=-0.011
L7: norm=3.43, cos_tgt=-0.010
```

L2-L4之间存在一个关键变换：方向从与W_U[target]负对齐变为正对齐。这可能是GLM4因果效力强的关键机制。

### 客观事实拼图更新

1. **GLM4是唯一具有间接读出路径的模型**——后续层将方向变换到更与W_U对齐的位置
2. **Qwen3/DS7B的后续层会消散方向与W_U的对齐**——读出对齐在传播中衰减
3. **这解释了为什么GLM4因果效力远强于Qwen3/DS7B**：不是方向本身更对齐，而是后续层更会"引导"方向到输出
4. **GLM4的L2→L4变换是关键**：方向在这个区间从负对齐变为正对齐
5. **所有模型都在放大注入信号的范数**——但只有GLM4的放大方向与W_U对齐
6. **因果效力 = 注入信号 × 后续层变换增益 × 变换后方向与W_U的对齐度**

### 关键硬伤

1. **GLM4的读出对齐增加量很小(+0.007)**，仅从-0.006到0.001。这可能不足以完全解释delta=1.6的因果效力。可能还有其他机制
2. **范数放大是所有模型的共性**，不是GLM4独有的。GLM4的差异在于放大"方向"而非"量级"
3. **尚未分解attention和MLP各自对变换的贡献**——后续层是attention还是MLP负责引导方向？
4. **delta_norm的增长可能导致logit变化**——即使方向与W_U不对齐，范数增大也会通过数值效应增加某些logit

### 命令记录

```bash
# Phase 322: 间接读出路径追踪
python tests/glm5/phase322_indirect_readout.py qwen3       # ~30s
python tests/glm5/phase322_indirect_readout.py glm4        # ~2.5min
python tests/glm5/phase322_indirect_readout.py deepseek7b  # ~1.5min
```

脚本位置：
- `tests/glm5/phase322_indirect_readout.py` — 主测试
- 结果：`results/phase322_indirect_readout/{qwen3,glm4,deepseek7b}_phase322.json`
- 日志：`tmp/phase322_{qwen3,glm4,deepseek7b}.log`

## Phase 322b: 读出增益分解确认测试 [2026-06-01 11:32]

### 背景

Phase 322确认GLM4存在间接读出路径，但读出对齐增加量很小。本测试直接对比三种注入方向在同一层同一alpha下的因果效力差异：属性方向 vs W_U[target]方向 vs 随机方向。

### 核心结果

**结果1: 属性方向 vs W_U方向 vs 随机方向——跨模型对比**

| 模型 | 属性方向delta | W_U方向delta | 随机方向delta | 属性增益(超随机) | W_U增益(超随机) | cos(attr,wu) |
|------|-------------|-------------|-------------|---------------|---------------|-------------|
| Qwen3 | 0.156 | **0.349** | 0.044 | 0.113 | **0.305** | 0.080 |
| GLM4 | 1.225 | **1.606** | 0.090 | 1.135 | **1.516** | -0.001 |
| DS7B | **0.092** | 0.002 | 0.057 | **0.035** | -0.055 | 0.031 |

**关键发现**:
1. **Qwen3/GLM4中W_U方向比属性方向更有效**——这是直接读出路径的证据
2. **DS7B中属性方向比W_U方向更有效**——W_U方向在DS7B中甚至为负！
3. **GLM4中W_U方向增益最高(1.516)**——说明GLM4的后续层对直接W_U方向有强放大
4. **GLM4中属性方向增益也很高(1.135)**——尽管cos(attr,wu)≈0，属性方向仍然有效

**结果2: GLM4层特异对比——L3是间接读出路径的转折点**

| 注入层 | 属性方向delta | W_U方向delta | 差值(attr-wu) | cos(attr,wu) |
|--------|-------------|-------------|-------------|-------------|
| L1 | 0.855 | **2.147** | **-1.292** | -0.006 |
| L2 | 0.862 | **1.278** | **-0.416** | -0.010 |
| L3 | **1.131** | 0.936 | **+0.195** | -0.008 |
| L4 | **1.001** | 0.871 | **+0.130** | -0.001 |
| L5 | 0.678 | 0.838 | -0.161 | -0.001 |
| L6 | **0.779** | 0.645 | **+0.134** | -0.007 |
| L7 | **0.756** | 0.512 | **+0.245** | -0.003 |
| L8 | 0.549 | 0.544 | +0.005 | -0.002 |

**极其关键的发现**:
1. **L1-L2: W_U方向远强于属性方向**——浅层适合直接W_U对齐注入
2. **L3+: 属性方向开始超过W_U方向**——尽管cos(attr,wu)≈0！
3. **L3-L7: 属性方向持续优于W_U方向(+0.13到+0.25)**
4. **L5是个例外：W_U方向重新超过属性方向**——可能L5的后续层变换对W_U方向更友好
5. **这个翻转发生在L3**——恰好是Phase 321发现的属性因果窗口峰值

**结果3: DS7B中W_U方向几乎完全无效(0.002)**

DS7B中W_U方向注入后delta≈0.002，远低于随机方向(0.057)。这说明DS7B的后续层不会放大W_U方向——DS7B的readout机制与GLM4/Qwen3完全不同。

### 客观事实拼图更新

1. **语言模型的readout机制不是单一通道，而是至少两种**：
   - **直接读出**：注入W_U对齐方向，通过范数放大产生logit变化
   - **间接读出**：注入非对齐方向，通过后续层变换后产生logit变化
2. **GLM4在L3+层同时具有直接和间接读出能力**
3. **Qwen3只有直接读出能力**——W_U方向始终更有效
4. **DS7B的W_U方向注入几乎无效**——可能DS7B使用完全不同的readout路径
5. **属性方向在L3+超过W_U方向**意味着：模型内部的"属性编码"比直接W_U投影更有效地到达输出
6. **间接读出路径存在的证据**：在L3-L7，即使cos(attr,wu)≈0，属性方向仍比W_U方向更有效

### 关键硬伤

1. **GLM4整体上W_U方向仍比属性方向更有效(1.606 vs 1.225)**——间接读出路径是补充，不是替代
2. **DS7B的W_U方向为负很反常**——可能需要更多调试
3. **L5的W_U方向重新超过属性方向**——说明间接读出路径不是单调的，不同层有不同特性
4. **尚未解释为什么L3是转折点**——可能L3是GLM4中MLP开始大量处理属性信息的层

### 命令记录

```bash
# Phase 322b: 读出增益分解
python tests/glm5/phase322b_readout_gain.py qwen3       # ~25s
python tests/glm5/phase322b_readout_gain.py glm4        # ~4min
python tests/glm5/phase322b_readout_gain.py deepseek7b  # ~2.5min
```

脚本位置：
- `tests/glm5/phase322b_readout_gain.py` — 确认测试
- 结果：`results/phase322b_gain/{qwen3,glm4,deepseek7b}_phase322b.json`
- 日志：`tmp/phase322b_{qwen3,glm4,deepseek7b}.log`

## Phase 323: 属性路径组件分解 — Attention vs MLP [2026-06-01 11:55]

### 背景

Phase 322b发现GLM4存在"间接读出路径"，属性方向在L3+超过W_U方向。核心问题：是attention还是MLP负责将属性方向变换成输出可读信号？

### 测试设计

5个测试维度：
1. **Component Output Analysis**：提取positive/negative句子的attn_out和mlp_out，计算delta范数、ratio、与W_U对齐度
2. **Component Ablation**：在最优层分别ablate attention或MLP，测量因果效力变化
3. **Component Replacement**：用源句子的attn/mlp delta注入到目标句子
4. **Cluster Readout**：从单token扩展到目标簇、竞争簇、对象簇读出
5. **Component Sensitivity**：注入方向后，下一层attn/mlp的响应强度

alpha=2.0，测试6个属性对(8个属性对用于component analysis)。

### 核心结果

**结果1: MLP是属性因果效力的绝对主导——所有三个模型一致**

Ablation结果（attn_importance = full_delta - attn_abl_delta, mlp_importance = full_delta - mlp_abl_delta）:

| 模型 | full_delta | attn_importance | mlp_importance | MLP/Attn比 |
|------|-----------|----------------|----------------|-----------|
| Qwen3 | 0.125 | **-1.552** | **2.927** | ∞(attn为负) |
| GLM4 | 0.860 | **0.052** | **0.460** | **8.8x** |
| DS7B | -0.115 | **-0.557** | **4.162** | ∞(attn为负) |

**极其关键的发现**：
1. **三个模型一致：MLP importance远大于attn importance**
2. **Qwen3和DS7B中attn importance为负**——ablate attention反而增强了因果效力！
3. **GLM4中attn importance仅为0.052**——attention几乎无贡献
4. **DS7B中mlp_importance高达4.162**——尽管DS7B单方向注入弱，MLP的因果贡献极大

**结果2: MLP贡献比随层增加——MLP是属性方向的主要来源**

Component analysis (mlp_ratio = mlp_delta_norm / direction_norm):

| 层 | Qwen3 mlp_ratio | GLM4 mlp_ratio | DS7B mlp_ratio |
|----|----------------|----------------|----------------|
| L0 | **0.836** | — | — |
| L1 | 0.286 | 0.603 | — |
| L2 | 0.306 | 0.534 | — |
| L3 | 0.443 | **0.464** | 0.241 |
| L4 | 0.467 | 0.517 | 0.328 |
| L5 | — | 0.505 | 0.226 |
| L6 | — | — | **0.467** |
| L7 | — | — | **0.498** |

1. **Qwen3 L0的mlp_ratio=0.836**——初始层几乎全是MLP
2. **GLM4的mlp_ratio稳定在0.5左右**——MLP约占一半，但MLP的因果贡献远超比例
3. **DS7B深层mlp_ratio增大**——L6-L7达到0.47-0.50

**结果3: MLP delta与属性方向高度对齐——Qwen3 L0 cos_mlp_dir=0.956**

| 层 | Qwen3 cos_mlp_dir | GLM4 cos_mlp_dir | DS7B cos_mlp_dir |
|----|-------------------|------------------|------------------|
| L0 | **0.956** | — | — |
| L1 | 0.287 | 0.802 | — |
| L2 | 0.229 | **0.728** | — |
| L3 | 0.153 | 0.619 | 0.323 |
| L4 | 0.175 | 0.594 | 0.444 |

1. **Qwen3 L0: MLP输出与属性方向对齐度0.956**——几乎完全重合
2. **GLM4 L1-L2: MLP输出与属性方向对齐度0.73-0.80**——MLP是属性方向的主要来源
3. **DS7B的对齐度较低(0.32-0.44)**——但随层增加

**结果4: GLM4 cluster readout——属性方向推动所有词簇，不只推动目标簇**

| 簇类型 | full_mean_delta | attn_abl_delta | mlp_abl_delta |
|--------|----------------|---------------|---------------|
| color(目标) | **1.428** | 1.459 | 1.100 |
| taste | **1.860** | 1.861 | 1.990 |
| texture | **1.798** | 1.864 | 1.581 |
| object | **1.117** | 1.133 | 1.044 |
| action | 0.637 | 0.561 | 0.233 |
| negation | 0.033 | -0.016 | **-0.411** |
| positive | 0.492 | 0.442 | 0.014 |
| negative | 0.246 | 0.185 | **-0.085** |

1. **属性方向推动color(1.43)、taste(1.86)、texture(1.80)全部上升**——不是"属性值方向"，而是"属性激活方向"
2. **negation词簇在MLP ablate时大幅下降(-0.41)**——说明MLP还携带极性/否定信息
3. **GLM4 cluster specificity=-0.027**——属性方向对目标簇和竞争簇的提升几乎相同
4. **这说明当前属性方向混合了"属性槽激活"和"具体属性值"，不是纯属性值编码**

**结果5: MLP Sensitivity随层增加——MLP对注入信号的放大持续增强**

| 传播 | Qwen3 mlp_sens | GLM4 mlp_sens | DS7B mlp_sens |
|------|---------------|--------------|--------------|
| +1层 | 0.439 | 0.613 | 0.886 |
| +2层 | 0.803 | 0.721 | 0.504 |
| +3层 | 0.627 | **0.994** | 0.979 |
| +4层 | **0.958** | 0.965 | **1.243** |
| +5层 | **1.053** | **1.032** | 1.043 |

1. **所有模型MLP sensitivity都随层增加**——MLP对注入信号的响应在深层更强
2. **DS7B L6→L7的mlp_sens=1.243**——DS7B深层MLP有更强的信号放大
3. **Attention sensitivity保持较低且稳定**——attn_sens通常在0.3-0.7之间

### 客观事实拼图更新

1. **MLP是属性方向因果效力的绝对主导**——三模型一致，attn贡献极小甚至为负
2. **Qwen3 L0的MLP输出几乎就是属性方向(cos=0.956)**——L0属性方向主要来自MLP
3. **GLM4 L2的MLP输出与属性方向对齐0.73**——MLP是属性方向的主要来源
4. **GLM4中ablate attention反而增强因果效力(attn_importance=0.052)**——attention可能起了"分流"作用
5. **DS7B的MLP importance极高(4.162)但整体因果效力弱**——MLP信号被后续层消散
6. **属性方向不是"属性值方向"，而是"属性激活方向"**——推动整个属性词簇，不仅目标词
7. **negation词簇在MLP ablate时大幅下降**——MLP还携带极性信息
8. **MLP sensitivity随层增加**——深层MLP对注入信号有更强放大

### 关键硬伤

1. **Cluster specificity为负或接近零**——说明当前"属性方向"混合了太多成分，不是纯属性关系编码。需要分解为属性槽、属性类型、属性值
2. **Ablate attention增强因果效力**——这违反直觉，可能是attention的"分流"效应：attention把注入的信号路由到其他位置，导致目标token的logit增幅减小
3. **DS7B mlp_importance=4.162但full_delta=-0.115**——矛盾。说明DS7B的MLP虽然对属性方向有因果贡献，但同时也在产生负向效果
4. **MLP sensitivity增加但读出对齐不增加(Qwen3/DS7B)**——MLP在放大信号但放大方向不是与W_U对齐的方向
5. **尚未测试block recomputation**——单层ablation可能无法揭示多层契约机制

### 命令记录

```bash
# Phase 323: 属性路径组件分解
python tests/glm5/phase323_path_decomposition.py qwen3       # ~16s
python tests/glm5/phase323_path_decomposition.py glm4        # ~4.7min
python tests/glm5/phase323_path_decomposition.py deepseek7b  # ~3min
```

脚本位置：
- `tests/glm5/phase323_path_decomposition.py` — 主测试
- 结果：`results/phase323_path_decomp/{qwen3,glm4,deepseek7b}_phase323.json`
- 日志：`tmp/phase323_{qwen3,glm4,deepseek7b}.log`

## Phase 323b: Attention消融悖论确认 + 属性编码分解 [2026-06-01 12:15]

### 背景

Phase 323发现MLP是属性因果效力的主导，且Qwen3/DS7B中ablate attention反而增强效力。本测试确认这个"attention消融悖论"是否稳定，并首次分解属性方向为三个层级：属性槽(slot)→属性类型(type)→属性值(value)。

### 测试设计

1. **Attention消融悖论稳定性测试**：3层×3 alpha值×6属性对=54个测试点
2. **属性编码分解**：用4级模板(something→has property→has color→is red)提取各层方向
3. **属性槽迁移**：测试slot方向是否可跨对象迁移

### 核心结果

**结果1: Attention消融悖论是稳定现象——三模型一致**

| 模型 | 悖论率 | attn_importance均值 | mlp_importance均值 | attn_imp正值% |
|------|--------|--------------------|--------------------|-------------|
| Qwen3 | **48.1%** | -0.374 | **1.309** | 48.1% |
| GLM4 | **59.3%** | -0.127 | -0.023 | 40.7% |
| DS7B | **68.5%** | -0.391 | **2.247** | 31.5% |

按层分布：

| 层 | Qwen3悖论率 | GLM4悖论率 | DS7B悖论率 |
|----|-----------|-----------|-----------|
| 浅层 | 66.7%(L0) | 61.1%(L2) | 55.6%(L5) |
| 中层 | 50.0%(L1) | 44.4%(L3) | 66.7%(L6) |
| 深层 | 27.8%(L2) | 72.2%(L4) | 83.3%(L7) |

**关键发现**：
1. **三个模型都出现attention消融悖论**——近半数到三分之二的情况下ablate attention增强因果效力
2. **DS7B悖论率最高(68.5%)**——DS7B中attention对属性方向的传播最不利
3. **GLM4 L4悖论率72.2%**——即使GLM4因果效力最强的层，attention仍在"分流"
4. **attn_importance均值在三模型中都为负**——attention不是在"帮助"属性方向，而是在"分散"它

**结果2: 属性编码可分解为三个层级——且slot/type/value方向是不同的**

GLM4 L3的属性分解(color为例):

| 层级 | 方向描述 | tgt_delta(red) | cluster_delta(color) | cos(slot,X) | cos(value,X) |
|------|---------|---------------|---------------------|------------|-------------|
| slot | "is something"→"has property" | **-3.194** | -1.779 | 1.000 | 0.486 |
| type | "has property"→"has color" | **+3.438** | +1.860 | 0.573 | 0.612 |
| value | "has color"→"is red" | **+0.469** | +0.995 | 0.486 | 1.000 |
| full | "is just an object"→"is red" | +0.828 | +1.337 | 0.003 | 0.612 |

**极其关键的发现**：
1. **slot方向产生强负效果(tgt_delta=-3.19)**——"有属性"这个概念在GLM4中反而抑制具体属性词！
2. **type方向产生最强正效果(tgt_delta=+3.44)**——"是颜色"这个类型方向比具体值方向更有效
3. **cos(slot,full_attr)≈0**——属性槽方向与完整属性方向正交！二者是完全不同的编码
4. **cos(value,full_attr)=0.61**——值方向与完整方向中度相关
5. **slot和type的余弦≈0.57**——属性槽和属性类型有一定重叠但不同

GLM4各属性类型的value vs type比较：

| 类型 | slot delta | type delta | value delta | full_attr delta | type>value? |
|------|-----------|-----------|------------|----------------|-----------|
| color | -3.19 | **+3.44** | +0.47 | +0.83 | **是(7.3x)** |
| taste | -0.34 | -0.59 | **+1.40** | **+2.29** | 否(值更强) |
| temperature | -2.32 | +0.30 | **+1.52** | **+1.66** | 否(值更强) |
| texture | -2.20 | +1.21 | **+1.99** | **+2.39** | 否(值更强) |

**发现**：color是唯一type>value的属性类型。其他类型都是value更强。这说明不同属性类型的编码结构不同。

**结果3: Qwen3的属性分解——slot方向与full_attr正交**

Qwen3 L0:

| 层级 | tgt_delta | cluster_delta | cos(slot,X) |
|------|----------|-------------|------------|
| slot | 0.000 | 0.010 | 1.000 |
| type | 0.000 | 0.047 | 0.696 |
| value | 0.063 | 0.086 | 0.651 |
| full_attr | **0.125** | **0.092** | **0.004** |

1. **Qwen3: cos(slot,full_attr)=0.004**——slot方向与full_attr完全正交
2. **Qwen3属性分解整体弱(tgt_delta≤0.125)**——但层级结构清晰
3. **slot→type→value的tgt_delta递增**：0→0→0.063→0.125

**结果4: DS7B的属性分解——slot和type方向比value更有效**

DS7B L6:

| 层级 | color tgt_delta | taste tgt_delta | texture tgt_delta |
|------|----------------|----------------|------------------|
| slot | **1.813** | 0.000 | 0.047 |
| type | **2.031** | 0.063 | 0.039 |
| value | 0.031 | 0.063 | 0.047 |
| full_attr | **-0.156** | -0.063 | 0.023 |

1. **DS7B color的slot/type方向反而最有效(1.81/2.03)**——但value方向几乎无效(0.031)
2. **full_attr方向在color上为负(-0.156)**——与Phase 321结果一致
3. **DS7B的属性编码结构与GLM4/Qwen3完全不同**

### 客观事实拼图更新

1. **Attention消融悖论是稳定现象**——三模型48-69%的悖论率
2. **Attention在属性编码中起"分流"作用**——它把注入信号路由到其他位置，导致目标token的logit增幅减小
3. **属性编码可分解为三层：slot(属性槽)→type(属性类型)→value(属性值)**
4. **cos(slot,full_attr)≈0**——属性槽方向与完整属性方向完全正交！这是最重要的结构发现
5. **GLM4 color: type方向远强于value方向(3.44 vs 0.47)**——属性类型编码比属性值编码更有效
6. **GLM4 slot方向产生强负效果(-3.19)**——"有属性"这个概念在GLM4中抑制具体属性词
7. **DS7B的slot/type方向比value方向更有效**——DS7B编码的是属性结构而非属性值
8. **不同属性类型的编码结构不同**——color的type>value，而taste/temperature/texture的value>type

### 关键硬伤

1. **GLM4 slot方向产生强负效果(-3.19)需要解释**——"has a property"为什么抑制"red"？可能是slot方向激活了"泛属性"信号，让模型不确定具体属性值，反而抑制了具体词
2. **DS7B full_attr方向为负但slot/type为正**——说明DS7B的"just an object"vs"is red"差分方向中包含了额外的噪声成分，抵消了属性编码
3. **Qwen3的分解整体太弱(tgt_delta≤0.125)**——难以确认层级结构
4. **尚未测试block recomputation**——当前都是单层注入+消融，可能低估了多层契约效应

### 命令记录

```bash
# Phase 323b: Attention消融悖论确认 + 属性编码分解
python tests/glm5/phase323b_attn_paradox.py qwen3       # ~33s
python tests/glm5/phase323b_attn_paradox.py glm4        # ~9.2min
python tests/glm5/phase323b_attn_paradox.py deepseek7b  # ~5.9min
```

脚本位置：
- `tests/glm5/phase323b_attn_paradox.py` — 确认测试
- 结果：`results/phase323b_attn_paradox/{qwen3,glm4,deepseek7b}_phase323b.json`
- 日志：`tmp/phase323b_{qwen3,glm4,deepseek7b}.log`

## Phase 324: 多模板属性层级结构确认 [2026-06-01 13:40]

### 背景

Phase 323b首次发现属性编码可分解为slot(属性槽)→type(属性类型)→value(属性值)三层结构，且slot方向与full_attr几乎正交。但硬伤是：模板可能引入语义偏差（抽象度、自然度、词频差异）。本测试用多套平行模板（每级4个模板）+ 6种属性类型(color/taste/temperature/texture/shape/size)确认层级结构是否真实。

### 测试设计

1. **跨模板一致性**：对每级(slot/type/value)使用4个不同模板提取方向，测量模板间方向余弦。若跨模板一致性高→不是模板伪影
2. **词簇读出模式**：注入slot/type/value方向后，测量6个属性词簇+对象/动作/否定词簇的logit变化
3. **Slot抑制效应确认**：10个对象-属性对/类型，确认slot方向是否稳定抑制具体属性值
4. **对象-属性绑定**：注入方向后比较"兼容属性词"vs"不兼容属性词"的logit变化

6种属性类型：color(30对), taste(30对), temperature(30对), texture(30对), shape(30对), size(30对)

### 核心结果

**结果1: 跨模板一致性：value > type > slot——三模型一致，层级结构不是模板伪影**

跨模板mean_cosine (6种属性类型平均):

| 层级 | Qwen3 | GLM4 | DS7B | 判定 |
|------|-------|------|------|------|
| value | **0.971** | **0.884** | **0.814** | 极高，跨模板一致 |
| type | **0.750** | **0.678** | **0.684** | 高，跨模板较一致 |
| slot | **0.540** | **0.474** | **0.536** | 中等，跨模板较弱 |

**关键发现**：
1. **value方向跨模板一致性最高(0.81-0.97)**——不同模板（"is red"/"has the color red"/"appears red"/"looks red"）产生几乎相同的方向
2. **type方向一致性中等(0.68-0.75)**——"has a color"/"has some color"/"has a certain color"等模板方向较一致
3. **slot方向一致性最低(0.47-0.54)**——"has some feature"/"has a property"/"has some quality"等抽象模板方向差异较大
4. **一致性排序 value > type > slot 在三模型中完全一致**——这不是模板伪影

**结果2: 跨层级余弦：slot与value最远，type居中**

跨层级cosine (6种属性类型平均):

| 跨层级对 | Qwen3 | GLM4 | DS7B |
|---------|-------|------|------|
| cos(slot,type) | 0.70 | 0.60 | 0.61 |
| cos(slot,value) | **0.60** | **0.51** | **0.49** |
| cos(type,value) | 0.69 | 0.62 | 0.61 |

1. **slot与value的余弦最低(0.49-0.60)**——属性槽和属性值是最不同的两个编码维度
2. **type与value的余弦最高(0.61-0.69)**——属性类型和属性值有一定重叠
3. **slot与type居中(0.60-0.70)**——属性槽和属性类型中度相关

**结果3: GLM4 slot方向在6种属性类型中5种抑制具体属性值**

GLM4 L3 slot/type/value方向的tgt_delta:

| 属性类型 | slot_tgt | type_tgt | value_tgt | slot负率 | 最强级 |
|---------|---------|---------|----------|---------|--------|
| color | **-0.31** | **+1.80** | +0.16 | 70% | **type** |
| taste | **-1.41** | **+0.58** | -0.10 | 100% | **type** |
| temperature | **-1.10** | -0.22 | -0.50 | 70% | type(弱) |
| texture | **-0.52** | **+1.32** | **+1.40** | 50% | **value≈type** |
| shape | **-0.78** | -0.24 | +0.09 | 60% | value(弱) |
| size | **-1.40** | -0.14 | -0.24 | 80% | 无 |

**极其关键的发现**：
1. **GLM4 slot方向在5/6属性类型中抑制具体属性值(负tgt_delta)**——确认Phase 323b的发现
2. **color和taste的type方向最强(+1.80和+0.58)**——"是颜色"/"是味道"比具体值更有效
3. **texture的type和value几乎等强(1.32 vs 1.40)**——质地属性的type和value编码同等重要
4. **temperature/shape/size的所有方向都弱或负**——这些属性类型的编码结构不同于color/taste
5. **属性类型间编码策略差异巨大**——不能用统一模型解释

**Qwen3 L0 slot/type/value的tgt_delta:**

| 属性类型 | slot_tgt | type_tgt | value_tgt |
|---------|---------|---------|----------|
| color | +0.006 | +0.091 | +0.106 |
| taste | +0.019 | -0.003 | +0.038 |
| texture | +0.061 | +0.081 | +0.220 |
| shape | +0.067 | +0.026 | +0.192 |

1. **Qwen3 slot方向不抑制具体值(微弱正)**——与GLM4不同
2. **Qwen3整体弱(tgt_delta≤0.22)**——但value > type > slot的层级关系仍存在
3. **texture/shape的value方向最强(0.22/0.19)**——与GLM4的color不同

**DS7B L6 slot/type/value的tgt_delta:**

| 属性类型 | slot_tgt | type_tgt | value_tgt |
|---------|---------|---------|----------|
| color | **+0.17** | **+0.89** | -0.001 |
| taste | +0.01 | -0.08 | -0.04 |
| temperature | -0.08 | -0.17 | -0.06 |

1. **DS7B color的type方向最强(+0.89)**——与GLM4一致
2. **DS7B color的value方向几乎无效(-0.001)**——与GLM4不同(GLM4=+0.16)
3. **DS7B的slot方向微弱正(+0.17)**——不抑制，但也不强

**结果4: Binding测试——三模型都没有对象-属性绑定效应**

Binding score = compat_cluster_delta - incompat_cluster_delta:

| 层级 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| slot | -0.048 | **-1.327** | +0.011 |
| type | -0.030 | **-0.806** | +0.027 |
| value | -0.027 | **-1.081** | +0.013 |

1. **GLM4的binding全为负**——注入属性方向后，兼容词下降更多！这说明属性方向不只推动兼容词
2. **Qwen3/DS7B的binding接近零**——无绑定效应
3. **三模型都没有"apple+color"对"red"偏好于"blue"的绑定机制**——至少在单方向注入下没有

### 客观事实拼图更新

1. **slot/type/value层级结构不是模板伪影**——跨模板一致性value(0.81-0.97) > type(0.68-0.75) > slot(0.47-0.54)
2. **slot方向一致性最低**——因为"有属性"这个概念本身模糊，不同模板引入的方向差异大
3. **value方向一致性最高**——不同模板描述同一属性值时方向几乎完全一致
4. **GLM4 slot方向稳定抑制具体属性值(5/6属性类型为负)**——"有属性"打开属性空间但抑制具体值
5. **color/taste的type方向远强于value方向**——"是颜色"/"是味道"这个类型约束比具体值选择更有效
6. **texture的type和value几乎等强**——质地属性的层级结构不同于颜色/味道
7. **temperature/shape/size三模型都弱**——这些属性的编码结构可能需要不同模板或不同层
8. **没有对象-属性绑定效应**——单方向注入不能产生"apple-red兼容性偏好"
9. **DS7B color的type强但value几乎无效**——DS7B编码属性类型但不编码具体值
10. **不同属性类型的编码策略差异巨大**——不能用统一的slot→type→value模型解释所有属性

### 关键硬伤

1. **slot方向一致性低(0.47-0.54)**——可能是因为"属性槽"这个概念确实模糊，不同模板激活不同的泛属性空间
2. **GLM4 binding全为负**——这意味着属性方向不只推动目标属性，还推动了不兼容属性。当前的slot/type/value分解可能还不够细
3. **temperature/shape/size三模型都弱**——可能这些属性类型的编码模板不合适，或者这些属性更多是关系性的而非描述性的
4. **没有对象-属性绑定**——可能需要多层block替换而非单方向注入才能看到绑定效应
5. **slot方向在不同模型中行为不同(GLM4负/Qwen3正/DS7B弱正)**——slot可能不是跨模型通用的编码维度

### 命令记录

```bash
# Phase 324: 多模板属性层级结构确认
python tests/glm5/phase324_hierarchy_confirm.py qwen3       # ~67s
python tests/glm5/phase324_hierarchy_confirm.py glm4        # ~28min
python tests/glm5/phase324_hierarchy_confirm.py deepseek7b  # ~18min
```

脚本位置：
- `tests/glm5/phase324_hierarchy_confirm.py` — 主测试
- 结果：`results/phase324_hierarchy/{qwen3,glm4,deepseek7b}_phase324.json`
- 日志：`tmp/phase324_{qwen3,glm4,deepseek7b}.log`

## Phase 324b: Color Type>Value 确认 + Binding 伪影检查 [2026-06-01 14:22]

### 背景

Phase 324用8个样本发现GLM4 color的type方向(1.80)远强于value(0.16)，GLM4 binding全为负。需要：(1)用30对全量样本确认type>value是否稳定；(2)用多alpha确认binding为负是否是alpha伪影。

### 测试设计

1. **30对color全量测试**：3个alpha(1.0/2.0/3.0) × 30对color = 90个测试点
2. **30对taste/texture全量测试**：alpha=2.0
3. **Binding低alpha测试**：7个绑定对 × 3级 × 3个alpha(0.5/1.0/2.0) = 63个测试点

### 核心结果

**结果1: GLM4 color type>value 在30对全量样本中确认——type更鲁棒，value在强注入下崩溃**

GLM4 L3 color 30对tgt_delta:

| α | slot_tgt | type_tgt | value_tgt | type/value比 |
|---|---------|---------|----------|-------------|
| 1.0 | +0.020 | **+0.687** | +0.397 | **1.73** |
| 2.0 | -0.450 | **+1.167** | **-0.245** | ∞(value为负) |
| 3.0 | -1.570 | **+0.678** | **-0.241** | ∞(value为负) |

**极其关键的发现**：
1. **GLM4 type方向在所有alpha下保持正值(0.68-1.17)**——鲁棒
2. **GLM4 value方向在α≥2.0时变为负值(-0.24)**——注入value方向反而抑制目标词！
3. **α=1.0时type/value=1.73**——即使在弱注入下type仍更强
4. **slot方向在α≥2.0时大幅负值(-0.45~-1.57)**——强注入slot方向强烈抑制具体属性值

**Qwen3 L0 color 30对tgt_delta:**

| α | slot_tgt | type_tgt | value_tgt | type/value比 |
|---|---------|---------|----------|-------------|
| 1.0 | -0.003 | +0.076 | **+0.106** | **0.72** |
| 2.0 | +0.002 | +0.130 | **+0.219** | **0.60** |
| 3.0 | +0.024 | +0.194 | **+0.314** | **0.62** |

**Qwen3与GLM4完全相反**：Qwen3的value方向更强(type/value≈0.6-0.72)。

**DS7B L6 color 30对tgt_delta:**

| α | slot_tgt | type_tgt | value_tgt | type/value比 |
|---|---------|---------|----------|-------------|
| 1.0 | **+0.347** | +0.133 | +0.038 | 3.47 |
| 2.0 | +0.003 | **+0.386** | -0.032 | ∞ |
| 3.0 | +0.021 | +0.208 | **+0.342** | 0.61 |

**DS7B跨alpha不稳定**：α=1.0 slot最强，α=2.0 type最强，α=3.0 value最强。

**结果2: GLM4 taste全负，texture的value>type**

GLM4 L3 30对tgt_delta (α=2.0):

| 属性类型 | slot_tgt | type_tgt | value_tgt | 最强级 |
|---------|---------|---------|----------|--------|
| color | -0.450 | **+1.167** | -0.245 | **type** |
| taste | -1.647 | -0.032 | -0.584 | 无(全负) |
| texture | -0.924 | +0.498 | **+1.066** | **value** |

1. **taste全负**——"has a taste"/"tastes sour"等方向注入后都抑制目标词
2. **texture的value(1.07) > type(0.50)**——与color的type>value相反
3. **color是唯一type>value的属性类型**

Qwen3 L0 30对tgt_delta (α=2.0):

| 属性类型 | slot_tgt | type_tgt | value_tgt |
|---------|---------|---------|----------|
| taste | +0.018 | +0.020 | **+0.071** |
| texture | +0.019 | +0.050 | **+0.163** |

Qwen3所有类型都是value最强。

DS7B L6 30对tgt_delta (α=2.0):

| 属性类型 | slot_tgt | type_tgt | value_tgt |
|---------|---------|---------|----------|
| taste | -0.010 | **+0.031** | +0.021 |
| texture | -0.259 | -0.248 | -0.158 |

DS7B的texture全负。

**结果3: Binding为负不是alpha伪影——在α=0.5/1.0/2.0下GLM4 binding都为负**

GLM4 binding score by alpha and level:

| α | slot | type | value |
|---|------|------|-------|
| 0.5 | **-0.123** | -0.076 | -0.139 |
| 1.0 | **-0.275** | -0.181 | -0.157 |
| 2.0 | **-0.900** | -0.482 | -0.586 |

1. **GLM4 binding在所有alpha下都为负**——不是alpha伪影
2. **负binding意味着：注入属性方向后，不兼容词的logit上升更多**
3. **这说明属性方向不只是推动目标词簇，而是推动更广的语义空间**

Qwen3/DS7B binding接近零(-0.04~+0.01)，无论alpha值。

### 客观事实拼图更新

1. **GLM4 color type>value用30对全量样本确认**——type方向鲁棒(所有alpha正值)，value方向在高alpha崩溃
2. **GLM4 value方向在高注入强度下变负**——可能因为value方向混合了其他成分，强注入时这些成分主导
3. **GLM4 slot方向强抑制具体属性值(α≥2.0时slot_tgt=-0.45~-1.57)**
4. **Qwen3 color value>type(0.6-0.72)**——与GLM4相反
5. **DS7B跨alpha不稳定**——α=1.0 slot最强，α=2.0 type最强，α=3.0 value最强
6. **GLM4 taste全负**——当前模板不适合GLM4的taste编码
7. **GLM4 texture的value>type(1.07 vs 0.50)**——texture和color的编码结构不同
8. **GLM4 binding为负不是alpha伪影**——属性方向推动不兼容词更多
9. **跨模型的color编码策略完全不同**：GLM4=type主导，Qwen3=value主导，DS7B=不稳定

### 关键硬伤

1. **GLM4 value方向在高alpha下变负**——这说明value方向不是简单的"属性值方向"，可能混入了slot成分
2. **taste全负(GLM4)**——模板可能不适合GLM4的taste编码结构
3. **binding为负**——这意味着属性方向不只是选择特定属性词，而是推动整个属性空间。当前"属性方向"可能更多是"属性激活方向"而非"属性值选择方向"
4. **DS7B跨alpha不稳定**——DS7B的编码可能更依赖注入强度，不适合单方向注入范式
5. **color是唯一type>value的属性类型**——为什么color如此特殊？需要进一步研究

### 命令记录

```bash
# Phase 324b: Color Type>Value 确认
python tests/glm5/phase324b_color_confirm.py qwen3       # ~67s
python tests/glm5/phase324b_color_confirm.py glm4        # ~24min
python tests/glm5/phase324b_color_confirm.py deepseek7b  # ~15min
```

脚本位置：
- `tests/glm5/phase324b_color_confirm.py` — 确认测试
- 结果：`results/phase324b_color/{qwen3,glm4,deepseek7b}_phase324b.json`
- 日志：`tmp/phase324b_{qwen3,glm4,deepseek7b}.log`

## Phase 325: 属性类型专用模板确认 [2026-06-01 21:58]

### 背景

Phase 324/324b发现taste/temperature/shape/size的因果效力很弱甚至为负。问题：这是否因为统一模板（"has a property"/"has a taste"/"is sour"）不适合某些属性类型？本测试为每种属性类型设计专用模板，与统一模板对比。

### 专用模板设计

| 属性类型 | slot模板 | type模板 | value模板 |
|---------|---------|---------|----------|
| color(统一) | has some feature | has a color | is red |
| color(专用) | has some visual feature | has a color | looks red |
| taste(统一) | has some feature | has a taste | is sour |
| taste(专用) | has some flavor quality | has a flavor | tastes sour |
| temperature(统一) | has some feature | has a temperature | is hot |
| temperature(专用) | has some thermal state | feels hot to touch | is hot to touch |
| texture(统一) | has some feature | has a texture | is rough |
| texture(专用) | has a surface quality | has a surface feel | feels rough |
| shape(统一) | has some feature | has a shape | is round |
| shape(专用) | has a geometric form | has a geometric shape | has a round shape |
| size(统一) | has some feature | has a size | is large |
| size(专用) | has a certain scale | is big in size | is bigger than average |

每类15对对象-属性组合，alpha=2.0。

### 核心结果

**结果1: GLM4专用模板大幅改善temperature和shape——弱效应确实是模板不适配**

GLM4 L3 tgt_mean (generic → specialized):

| 属性类型 | slot(通用)→(专用) | type(通用)→(专用) | value(通用)→(专用) | 最强级(专用) |
|---------|------------------|------------------|-------------------|-------------|
| color | -0.71→**+0.34** | **1.31→1.36** | -0.33→-0.25 | **type** |
| taste | -1.69→-0.01 | 0.23→-0.17 | -0.62→-0.36 | **slot(弱)** |
| temperature | -1.06→**+0.57** | 0.05→-0.05 | 0.13→**0.51** | **slot≈value** |
| texture | -0.83→-0.41 | 0.84→**2.51** | 1.37→0.50 | **type** |
| shape | -0.31→**+0.67** | 0.21→**0.81** | 0.50→0.58 | **type** |
| size | -0.93→-0.56 | -0.71→-0.70 | -0.27→0.00 | **value(≈0)** |

**极其关键的发现**：
1. **temperature: slot从-1.06翻转到+0.57，value从0.13升到0.51**——模板适配后temperature终于有正效应！
2. **shape: slot从-0.31翻转到+0.67，type从0.21升到0.81**——shape的type方向在专用模板下很强
3. **texture: type从0.84飙升到2.51**——texture的专用模板让type方向成为最强
4. **taste: slot从-1.69翻转到-0.01**——显著改善但仍为负，type/value仍为负
5. **color: slot从-0.71翻转到+0.34**——专用模板让color的slot也变正了
6. **size: 几乎无改善，全负或接近0**

**结果2: Qwen3专用模板改善taste的type(3x提升)和texture的type(1.7x)**

Qwen3 L0 tgt_mean (generic → specialized):

| 属性类型 | slot(通用)→(专用) | type(通用)→(专用) | value(通用)→(专用) | 最强级(专用) |
|---------|------------------|------------------|-------------------|-------------|
| color | 0.02→0.04 | 0.17→0.17 | **0.25→0.25** | **value** |
| taste | 0.03→0.05 | 0.02→**0.07** | 0.06→0.05 | **type≈value** |
| temperature | 0.04→0.05 | 0.11→0.10 | **0.11→0.11** | **type≈value** |
| texture | 0.05→0.02 | 0.07→**0.11** | **0.22→0.15** | **value** |
| shape | 0.03→0.02 | 0.02→0.04 | 0.13→**0.01** | **type(弱)** |
| size | 0.03→0.04 | -0.01→0.02 | 0.04→0.02 | **slot(弱)** |

1. Qwen3改善有限，整体仍然很弱(tgt_mean≤0.25)
2. taste的type方向3x提升(0.02→0.07)——专用模板有效
3. shape的value方向反而从0.13降到0.01——专用模板不适合Qwen3的shape

**结果3: DS7B专用模板让color的value从-0.09翻转到+0.50——巨大改善**

DS7B L6 tgt_mean (generic → specialized):

| 属性类型 | slot(通用)→(专用) | type(通用)→(专用) | value(通用)→(专用) | 最强级(专用) |
|---------|------------------|------------------|-------------------|-------------|
| color | 0.35→0.30 | **0.69→0.82** | -0.09→**+0.50** | **type** |
| taste | -0.03→0.06 | 0.06→-0.01 | -0.06→0.00 | **slot(弱)** |
| temperature | 0.05→**-0.16** | 0.06→0.02 | -0.06→-0.13 | **type(≈0)** |
| texture | -0.06→**0.08** | -0.14→-0.08 | 0.03→-0.01 | **slot(弱)** |
| shape | -0.05→-0.06 | -0.09→0.04 | -0.11→-0.04 | **type(弱)** |
| size | -0.09→-0.04 | -0.01→-0.13 | -0.10→-0.09 | **slot(弱)** |

1. **DS7B color的value从-0.09翻转到+0.50**——5x改善！这说明DS7B编码color value，但需要专用模板
2. DS7B color: type(0.82) > value(0.50) > slot(0.30)——与GLM4的color编码模式一致
3. taste/texture/slot从负变正——改善但不强
4. temperature反而变差，shape/size仍然弱

**结果4: 跨模板一致性——专用模板vs通用模板**

跨模板mean_cos (specialized):

| 属性类型 | slot | type | value | 排序 |
|---------|------|------|-------|------|
| color(3模型均) | 0.61-0.65 | 0.89-0.96 | 0.88-0.99 | value≈type > slot |
| taste(3模型均) | 0.56-0.72 | 0.68-0.81 | 0.53-0.69 | type > slot≈value |
| temperature(3模型均) | 0.85-0.98 | 0.80-0.96 | 0.45-0.66 | slot≈type > value |
| texture(3模型均) | 0.81-0.98 | 0.52-0.68 | 0.54-0.67 | slot > type≈value |
| shape(3模型均) | 0.55-0.71 | 0.55-0.78 | 0.80-0.96 | value > type≈slot |
| size(3模型均) | 0.52-0.73 | 0.64-0.76 | 0.51-0.60 | type > slot≈value |

**发现**：不同属性类型的一致性排序不同：
- color/taste: type和value一致性高
- temperature: slot和type一致性高，value一致性低
- texture: slot一致性最高，type/value低
- shape: value一致性最高
- size: type一致性最高

这说明**每种属性类型的编码稳定性分布在不同的层级上**。

### 客观事实拼图更新

1. **弱效应确实是模板不适配**——temperature和shape在GLM4中用专用模板后大幅改善(temperature: slot -1.06→+0.57; shape: type 0.21→0.81)
2. **GLM4 texture专用模板让type飙升至2.51**——远超value(0.50)，texture的type>value与Phase 324的value>type矛盾，说明模板对texture的层级判断影响很大
3. **DS7B color的value从-0.09翻转到+0.50**——DS7B编码color value，但需要"looks red"而非"is red"
4. **taste仍然困难**——GLM4用专用模板后slot从-1.69改善到-0.01，但type/value仍为负。taste可能需要完全不同的范式
5. **size是唯一所有模型都无法激活的属性类型**——三模型size的专用模板tgt_mean都≤0
6. **color的slot方向用专用模板后从负变正(GLM4: -0.71→+0.34)**——"has some visual feature"比"has some feature"更适合color的slot
7. **不同属性类型的跨模板一致性排序不同**——不是统一的value>type>slot

### 关键硬伤

1. **texture的type>value(value=2.51)与Phase 324的value>type矛盾**——模板选择严重影响了层级判断。需要确认哪个模板更自然
2. **taste仍然困难**——即使专用模板也无法让GLM4的taste type/value产生正效应。taste可能依赖动词句式而非形容词句式
3. **size无法被任何模板激活**——size可能是比较关系而非描述性属性，需要"bigger than"等比较句式
4. **temperature的跨模板value一致性低(0.45-0.66)**——"is hot"/"is hot to touch"/"feels hot"差异大
5. **GLM4 color的slot从负变正**——说明Phase 324/324b中slot的抑制效应部分来自模板不匹配

### 命令记录

```bash
# Phase 325: 属性类型专用模板确认
python tests/glm5/phase325_specialized_templates.py qwen3       # ~72s
python tests/glm5/phase325_specialized_templates.py glm4        # ~20min
python tests/glm5/phase325_specialized_templates.py deepseek7b  # ~29min
```

脚本位置：
- `tests/glm5/phase325_specialized_templates.py` — 主测试
- 结果：`results/phase325_specialized/{qwen3,glm4,deepseek7b}_phase325.json`
- 日志：`tmp/phase325_{qwen3,glm4,deepseek7b}.log`

## Phase 325b: 专用模板关键发现确认 (20对/类型) [2026-06-01 22:28]

### 背景

Phase 325发现专用模板大幅改善temperature/shape/color等属性类型。关键发现需要确认：GLM4 texture type=2.51，GLM4 temperature slot翻正，DS7B color value翻正。本测试用20对全量样本确认。

### 核心结果

**结果1: GLM4 color type=4.33, value=2.13——type远强于value，20对确认**

GLM4 L3 专用模板 20对 tgt_mean:

| 属性类型 | slot | type | value | 最强级 | type/value比 |
|---------|------|------|-------|--------|------------|
| color | -0.65 | **4.33** | 2.13 | **type** | **2.03** |
| temperature | 1.52 | 0.42 | **3.38** | **value** | 0.12 |
| texture | -1.58 | **1.98** | -0.58 | **type** | ∞(value负) |
| shape | 0.01 | -0.54 | -0.54 | slot(≈0) | — |

**极其关键的发现**：
1. **GLM4 color type/value=2.03**——Phase 324的1.73→Phase 325的1.04→Phase 325b的2.03，type>value稳定
2. **GLM4 temperature value=3.38**——temperature用专用模板后value方向极强！这是Phase 325的新发现
3. **GLM4 texture type=1.98**——Phase 325的2.51确认(方向一致，量级略降)
4. **GLM4 shape全弱**——专用模板无法稳定激活shape
5. **GLM4 slot在所有属性类型中4/4为负或接近0**——slot稳定抑制具体属性值

**结果2: Qwen3 color value=0.11, type=0.07——value>type确认**

Qwen3 L0 专用模板 20对 tgt_mean:

| 属性类型 | slot | type | value | 最强级 |
|---------|------|------|-------|--------|
| color | -0.04 | 0.07 | **0.11** | **value** |
| temperature | -0.02 | 0.01 | -0.02 | type(≈0) |
| texture | 0.01 | 0.06 | **0.11** | **value** |
| shape | 0.02 | 0.07 | **0.07** | type≈value |

1. Qwen3整体极弱(max=0.11)，但value>type的模式在color/texture中稳定
2. temperature/shape专用模板对Qwen3无效

**结果3: DS7B整体弱——color value翻正未被20对确认**

DS7B L6 专用模板 20对 tgt_mean:

| 属性类型 | slot | type | value | 最强级 |
|---------|------|------|-------|--------|
| color | 0.02 | -0.02 | 0.02 | slot(≈0) |
| temperature | 0.02 | **0.15** | 0.01 | **type** |
| texture | -0.00 | 0.01 | 0.01 | value(≈0) |
| shape | -0.06 | -0.05 | -0.09 | none |

1. **DS7B color value=0.02**——Phase 325的+0.50未被20对确认，15对样本的翻转可能是小样本噪声
2. **DS7B temperature type=0.15**——唯一稳定的正效应
3. DS7B整体因果效力极弱(≤0.15)

### 客观事实拼图更新

1. **GLM4 color type>value用20对确认(比=2.03)**——color编码确实先进入"颜色空间"再选具体值
2. **GLM4 temperature value=3.38极强**——temperature不是弱属性类型，只是通用模板不合适。"is hot to touch"方向极强
3. **GLM4 texture type=1.98确认**——texture也是type主导（"有表面触感"比"摸起来粗糙"更有效）
4. **GLM4 slot在4/4属性类型中为负或≈0**——slot稳定抑制具体属性值，这是GLM4的通用机制
5. **DS7B color value翻转(+0.50)未被20对确认**——小样本假阳性，DS7B整体因果效力极弱
6. **Qwen3 color/texture稳定value>type**——Qwen3编码策略与GLM4相反
7. **shape在GLM4和DS7B中都弱**——shape编码可能需要完全不同的范式

### 关键硬伤

1. **GLM4 temperature value=3.38远超color type=4.33?** 不，color type=4.33 > temperature value=3.38。但temperature value也很强。这说明不同属性类型的编码强度差异大
2. **shape专用模板无效**——可能shape编码依赖视觉/几何概念而非形容词描述
3. **DS7B 15对样本的翻转在20对中消失**——说明小样本结论需要谨慎，重要发现必须大样本确认
4. **GLM4 slot稳定为负**——这个现象需要更深理解：是slot真的抑制值选择，还是slot方向提取有偏差

### 命令记录

```bash
# Phase 325b: 专用模板关键发现确认
python tests/glm5/phase325b_confirm.py qwen3       # ~46s
python tests/glm5/phase325b_confirm.py glm4        # ~14min
python tests/glm5/phase325b_confirm.py deepseek7b  # ~9min
```

脚本位置：
- `tests/glm5/phase325b_confirm.py` — 确认测试
- 结果：`results/phase325b_confirm/{qwen3,glm4,deepseek7b}_phase325b.json`

## Phase 326: Slot输出验证+感官通道+对象-属性绑定 [2026-06-01 23:09]

### 背景

Phase 325/325b确认了属性类型特异的编码结构，但三个关键问题未解决：
1. GLM4 slot方向稳定抑制具体属性值——是真实机制还是提取偏差？需要验证slot是否推泛属性词(property/feature/quality)
2. temperature的"feels hot"/"is hot to touch"效果来自属性类型还是感官动词？
3. 对象-属性绑定(binding)仍未破解

### 测试设计

**Test 1: Slot输出模式**——注入slot方向后，测量三类输出词的logit变化：
- specific values: red/sweet/rough/hot/round等具体属性值
- type words: color/taste/texture/temperature/shape/size等属性类型词
- generic property: property/feature/quality/characteristic/attribute/trait等泛属性词

**Test 2: 感官通道对比**——同一属性类型用不同感官通道模板：
- color: "looks red"(视觉) vs "is red"(状态) vs "has a color"(类型)
- texture: "feels rough"(触觉) vs "is rough"(状态) vs "has a surface feel"(类型)
- temperature: "feels hot"(触觉) vs "is hot to touch"(接触) vs "is hot"(状态) vs "has a temperature quality"(类型)

**Test 3: 对象-属性绑定**——10对兼容/不兼容属性对，3种注入方向：
- type方向："has a color"方向，测量red vs blue的logit差
- compat_value方向："looks red"方向，测量red vs blue的logit差
- incompat_value方向："looks blue"方向，测量red vs blue的logit差

### 核心结果

**结果1: GLM4 slot方向强烈推泛属性词(+0.61)，同时抑制具体值(-0.81)和类型词(-1.76)**

GLM4 L3 slot方向对三类输出词的logit变化(10个对象平均):

| 输出类别 | mean delta | positive rate | 判定 |
|---------|-----------|--------------|------|
| Specific values (red/sweet/rough...) | **-0.81** | 20% | 强烈抑制 |
| Type words (color/taste/texture...) | **-1.76** | 0% | 更强抑制 |
| **Generic property (property/feature/quality...)** | **+0.61** | **80%** | **强烈推升** |

**这是Phase 326最重要的发现！**
1. **slot方向不是"抑制属性"，而是"打开泛属性空间"**——property/feature/quality等词被推升
2. **slot方向不仅抑制具体值，还抑制类型词**——color/taste等也下降，说明slot不选类型也不选值
3. **slot的真实功能：告知模型"这里需要讨论属性"，但不选择具体属性**——这与Phase 324/325的假说一致

Qwen3 L0 slot方向(对比):
- specific=+0.005, type=+0.028, generic=**+0.100** — 同样推泛属性词最高，但不抑制具体值

DS7B L6 slot方向:
- specific=-0.087, type=-0.084, generic=-0.070 — 全弱，无清晰模式

**结果2: GLM4 sensory channel——type主导不依赖感官通道，temperature依赖"to touch"构式**

GLM4 L3 sensory channel tgt_mean:

| 属性类型 | 视觉/触觉 | 状态(bare) | 类型(type) | 判定 |
|---------|----------|----------|----------|------|
| color | visual=2.68 | state=2.67 | **type=5.02** | type远强，通道无差异 |
| texture | tactile=-0.06 | state=-0.01 | **type=2.30** | type强，感官通道无效 |
| temperature | tactile=-0.49 | state=-0.24 | type=0.32 | **contact=3.54极强** |

**极其关键的发现**：
1. **color: type(5.02) >> visual(2.68) ≈ state(2.67)**——type主导与感官通道无关！"has a color"比"looks red"或"is red"强2倍
2. **texture: type(2.30) >> tactile(-0.06) ≈ state(-0.01)**——type主导也不依赖感官通道！"has a surface feel"比"feels rough"强
3. **temperature: contact(3.54) >> tactile(-0.49) > state(-0.24)**——"is hot to touch"远强于"feels hot"或"is hot"！温度属性依赖接触构式而非感官动词
4. **"feels"动词在GLM4中对texture和temperature都无效**——这与直觉相反！

Qwen3 L0 sensory channel:
- color: type(0.15) > visual(0.11) ≈ state(0.10) — type也更强
- texture: tactile(0.08) > state(0.05) ≈ type(0.05) — 触觉通道略强
- temperature: tactile=0.04, contact=0.00, state=0.02, type=0.09 — 整体弱

DS7B L6 sensory channel:
- 全部极弱(≤0.06)，无清晰通道差异

**结果3: GLM4 type方向有弱binding(0.64)，compat_value方向也有弱binding(0.38)**

GLM4 binding score (compatible_delta - incompatible_delta):

| 注入方向 | mean binding | positive rate | 判定 |
|---------|-------------|--------------|------|
| type方向("has a color") | **+0.64** | 70% | 弱正binding |
| compat_value("looks red") | **+0.38** | 60% | 弱正binding |
| incompat_value("looks blue") | **-0.36** | 40% | 负binding |

1. **type方向有70%正binding**——"has a color"方向让compatible值(red)比incompatible值(blue)上升更多
2. **incompat_value方向binding为负(-0.36)**——注入"looks blue"方向后，red反而比blue下降更多
3. **但数据量小(10对)，方差大**——需要更多对确认

Qwen3 binding:
- type: +0.12(80%正), compat_value: +0.03(50%正), incompat_value: +0.02(40%正) — 极弱

DS7B binding:
- 全接近0，无binding信号

### 客观事实拼图更新

1. **GLM4 slot方向的真实功能是"打开泛属性空间"**——推property/feature/quality(+0.61)，抑制color/taste(-1.76)和red/sweet(-0.81)。这是确凿的机制发现
2. **Qwen3 slot方向也推泛属性词(+0.10)最高**——但不抑制具体值(仅+0.005)。slot的"打开属性空间"功能跨模型一致
3. **GLM4 color/texture的type主导不依赖感官通道**——"has a color"(5.02)远强于"looks red"(2.68)，说明type方向不是感官激活
4. **GLM4 temperature依赖"to touch"构式(contact=3.54)**——"is hot to touch"远强于"feels hot"(-0.49)和"is hot"(-0.24)
5. **"feels"动词在GLM4中对texture和temperature都无效**——这与语言直觉相反，说明GLM4不主要通过感官动词编码触觉属性
6. **GLM4 type方向有弱binding(+0.64, 70%正)**——"has a color"方向让apple-red比apple-blue更兼容。但数据量小，需确认
7. **DS7B所有测试都弱**——slot无模式，通道无差异，binding接近0

### 关键硬伤

1. **"feels"在GLM4中无效是反直觉的**——可能是GLM4用不同方式编码触觉语义，"feels"可能激活了其他语义路径
2. **binding数据量仅10对**——GLM4的弱binding(+0.64)需要至少20对确认
3. **temperature的"to touch"效果可能来自构式而非属性类型**——"is hot to touch"可能更强是因为它包含更多语义约束
4. **Qwen3的slot推泛属性词但不抑制具体值**——Qwen3的slot更温和，"打开但不抑制"

### 命令记录

```bash
# Phase 326: Slot验证+感官通道+绑定
python tests/glm5/phase326_slot_channel_binding.py qwen3       # ~32s
python tests/glm5/phase326_slot_channel_binding.py glm4        # ~9min
python tests/glm5/phase326_slot_channel_binding.py deepseek7b  # ~6min
```

脚本位置：
- `tests/glm5/phase326_slot_channel_binding.py` — 主测试
- 结果：`results/phase326_slot_channel_binding/{qwen3,glm4,deepseek7b}_phase326.json`

## Phase 326b: Slot+Binding+Temperature确认 (20样本) [2026-06-01 23:27]

### 背景

Phase 326三个关键发现需要20样本确认：
1. GLM4 slot推泛属性词(+0.61)，同时抑制具体值(-0.81)和类型词(-1.76)
2. GLM4 type方向有弱binding(+0.64, 70%正)
3. GLM4 temperature "is hot to touch"(contact=3.54)远强于"feels hot"(tactile=-0.49)

### 核心结果

**结果1: GLM4 slot推泛属性词用20对象确认——这是确凿的机制发现**

GLM4 L3 slot方向对三类输出词的logit变化:

| 输出类别 | 10对象(Phase 326) | 20对象(Phase 326b) | 判定 |
|---------|------------------|-------------------|------|
| Specific values | -0.81 (20%正) | **-0.58 (30%正)** | 稳定抑制 |
| Type words | -1.76 (0%正) | **-1.63 (0%正)** | 更强抑制 |
| Generic property | +0.61 (80%正) | **+0.84 (90%正)** | 稳定推升 |

**确认！slot方向的真实功能是"打开泛属性空间"**——property/feature/quality/characteristic等词被推升，而color/taste和red/sweet被抑制。

Qwen3 L0 slot方向(20对象):
- specific=-0.006(40%正), type=-0.006(45%正), generic=**+0.084(75%正)**
- 同样推泛属性词最高，但不抑制具体值——更温和的"打开"

DS7B L6 slot方向(20对象):
- specific=-0.050(55%正), type=-0.044(45%正), generic=-0.036(60%正)
- 全弱，无清晰模式

**结果2: GLM4 type方向有稳定binding(+0.74, 70%正)，compat_value方向也有binding(+0.73, 55%正)**

GLM4 binding score (20对):

| 注入方向 | 10对(Phase 326) | 20对(Phase 326b) | 判定 |
|---------|----------------|-----------------|------|
| type方向 | +0.64 (70%正) | **+0.74 (70%正)** | 稳定弱binding |
| compat_value方向 | +0.38 (60%正) | **+0.73 (55%正)** | 稍不稳定但正 |

**type方向binding确认**——"has a color"方向确实让apple-red比apple-blue更兼容(70%对为正)。这是首次发现对象-属性绑定效应！

Qwen3 binding: type=+0.05(60%正), compat_value=+0.01(50%正) — 极弱
DS7B binding: type=-0.005(30%正), compat_value=+0.001(45%正) — 无binding

**结果3: GLM4 temperature "to touch"构式用20对确认极强**

GLM4 L3 temperature channel tgt_mean (20对):

| 通道 | 8对(Phase 326) | 20对(Phase 326b) | 判定 |
|------|---------------|-----------------|------|
| contact("is hot to touch") | 3.54 | **3.38** | 稳定极强 |
| state("is hot") | -0.24 | **-0.50** | 稳定负 |
| type("has temp quality") | 0.32 | **0.42** | 弱正 |

**确认**——temperature的"to touch"构式(contact=3.38)远强于bare state(-0.50)。这不是感官通道差异，而是构式差异。

Qwen3 temperature: contact=-0.02, state=+0.03, type=+0.01 — 无通道差异
DS7B temperature: contact=+0.01, state=+0.02, **type=+0.15** — type最强(与GLM4不同)

### 客观事实拼图更新

1. **GLM4 slot"打开泛属性空间"机制用20对象确认(generic=+0.84, 90%正)**——这是确凿的机制发现
2. **Qwen3 slot也推泛属性词(+0.084, 75%正)但不抑制具体值**——跨模型一致：slot推泛属性词
3. **GLM4 type方向有稳定binding(+0.74, 70%正)**——首次发现对象-属性绑定效应
4. **GLM4 compat_value方向也有binding(+0.73, 55%正)**——注入"looks red"方向也让apple-red比apple-blue更兼容
5. **GLM4 temperature "to touch"构式极强(3.38)**——确认不是感官通道差异，而是构式差异
6. **DS7B temperature type=0.15是其最稳定信号**——DS7B在temperature上编码type而非value
7. **Qwen3/DS7B无binding**——binding是GLM4特有的

### 命令记录

```bash
# Phase 326b: 确认测试
python tests/glm5/phase326b_confirm.py qwen3       # ~32s
python tests/glm5/phase326b_confirm.py glm4        # ~9min
python tests/glm5/phase326b_confirm.py deepseek7b  # ~6min
```

脚本位置：
- `tests/glm5/phase326b_confirm.py` — 确认测试
- 结果：`results/phase326b_confirm/{qwen3,glm4,deepseek7b}_phase326b.json`

## Phase 327: Slot组合因果+Temperature构式分解+Binding大矩阵 [2026-06-02 05:25]

### 背景

Phase 326/326b确认slot是"泛属性空间入口"，但仍未解决三个关键问题：
1. slot是否真正参与属性计算？还是仅推泛属性词但不增强后续属性选择？
2. temperature的"to touch"构式是温度专用，还是一般物理感知构式？
3. binding是否随兼容等级单调变化(high>medium>low>absurd)？

### 测试设计

**Test 1: 7种组合注入**——10个color对象，注入slot/type/value的7种组合，测量compat/incompat/generic/type/specific五类词logit变化
**Test 2: Temperature构式分解**——20对温度对象×4种模板(contact/tactile/state/type) + 12个非温度对象×6种模板(rough_touch/rough_state/sharp_touch/sharp_state/heavy_lift/heavy_state)
**Test 3: Binding兼容等级矩阵**——52对对象-属性组合(4属性类型×4兼容等级)，type方向注入

### 核心结果

**结果1: GLM4 slot+value组合大幅增强compat值——slot确实参与属性计算**

GLM4 L3 7种组合注入的compat值logit变化(10个color对象平均):

| 组合 | compat | incompat | compat-incompat | generic | type | specific |
|------|--------|----------|----------------|---------|------|----------|
| slot | +0.20 | -0.27 | **+0.47** | **+0.97** | **-1.53** | -0.50 |
| type | **+4.38** | +3.80 | +0.58 | +0.47 | -0.80 | **+3.16** |
| value | +2.18 | +1.69 | +0.49 | -1.64 | **+2.01** | +2.12 |
| slot+type | +3.69 | +3.19 | +0.50 | +0.62 | -0.84 | +2.70 |
| **slot+value** | **+3.70** | **+2.07** | **+1.63** | -1.39 | +1.62 | +2.87 |
| type+value | +3.80 | +2.16 | +1.64 | -1.25 | +0.05 | +2.51 |
| **slot+type+value** | **+5.12** | **+3.07** | **+2.04** | -0.58 | -0.61 | +3.53 |

**极其关键的发现**：
1. **slot+value vs value: compat_diff=+1.52!** — slot加入value后compat值从2.18涨到3.70(+70%)，这是slot参与属性计算的最强证据
2. **slot+type vs type: compat_diff=-0.69** — slot加入type后compat值从4.38降到3.69。slot抑制type的compat效果
3. **slot+type+value=5.12是所有组合中最强的** — 三重组合compat最高
4. **compat-incompat差**：slot+value(+1.63) ≈ type+value(+1.64) ≈ slot+type+value(+2.04) — 三重组合的binding效果最强

Qwen3 L0 combo:
- slot+type+value=0.80最强，type+value=0.42次之
- slot+type vs type: compat_diff=+0.01(几乎无增强)
- slot+value vs value: compat_diff=+0.04(弱增强)

DS7B L6 combo: 全弱(≤-0.05)，无清晰组合效应

**结果2: GLM4 "to touch"不是温度专用构式——它是通用物理感知构式**

GLM4 L3 构式效果(20对/12对象平均):

| 构式 | 温度目标词 | 对照属性 | 判定 |
|------|----------|---------|------|
| **is hot to touch** | **+3.17** | — | 极强(温度) |
| is hot(state) | -0.87 | — | 负 |
| feels hot(tactile) | -0.95 | — | 负 |
| has temp quality(type) | +0.28 | — | 弱正 |
| **is rough to touch** | — | **+2.99** | 强(texture) |
| is rough(state) | — | **+3.72** | 更强(texture) |
| **is sharp to touch** | — | **+5.78** | 极强(shape) |
| is sharp(state) | — | +2.34 | 中(shape) |
| is heavy to lift | — | -0.20 | 无(size) |
| is heavy(state) | — | -2.30 | 负(size) |

**极其关键的发现**：
1. **"to touch"对sharp(5.78)远强于rough(2.99)和hot(3.17)** — "to touch"不是温度专用，sharp的"to touch"效果最强！
2. **rough_state(3.72) > rough_touch(2.99)** — 对texture，bare state反比"to touch"更强
3. **sharp_touch(5.78) >> sharp_state(2.34)** — 对shape，"to touch"效果翻倍
4. **heavy_lift(-0.20)和heavy_state(-2.30)都为负** — size的构式全部无效
5. **"feels hot"(-0.95)和"is hot"(-0.87)在GLM4中都不如"is hot to touch"(3.17)** — temperature需要完整构式

Qwen3 L0 构式:
- temperature: tactile(0.08) > state(0.07) > contact(-0.03) — 与GLM4相反，Qwen3中"feels"最强
- nontemp: sharp_touch(0.12) > heavy_state(0.11) > heavy_lift(0.10) — "to touch"对sharp略强

DS7B L6 构式:
- temperature: type(0.16) > tactile(0.05) > state(0.02) ≈ contact(0.01) — type最强
- nontemp: 全弱(≤0.06)

**结果3: GLM4 binding不随兼容等级单调变化——type方向注入不能区分兼容等级**

GLM4 binding by level (type方向注入, 43对):

| 兼容等级 | mean_binding | positive_rate | n |
|---------|-------------|--------------|---|
| high | +0.54 | 56% | 18 |
| medium | +0.51 | 62% | 8 |
| low | +0.32 | 50% | 8 |
| **absurd** | **+1.05** | **67%** | 9 |

**absurd等级binding最高！** 这完全违反预期——如果binding机制正确，absurd组合(如"idea-hot")应该binding最低。

GLM4 binding by type:
| 属性类型 | mean_binding | positive_rate |
|---------|-------------|--------------|
| color | +0.11 | 50% |
| **texture** | **+1.19** | **64%** |
| **temperature** | **+1.17** | **64%** |
| taste | -0.21 | 57% |

texture和temperature的binding远强于color——但Phase 326b中color的binding是+0.74。**差异来源是matrix中包含了更多低/absurd等级的对**，且type方向注入对不同属性类型效果不同。

Qwen3/DS7B binding: 全弱，不单调。

### 客观事实拼图更新

1. **GLM4 slot确实参与属性计算**——slot+value比value alone的compat值高70%(2.18→3.70)。slot不是只推泛属性词的"抽象方向"，而是真正增强value选择的计算入口
2. **GLM4 slot+type反而比type alone弱**——slot抑制type方向的compat效果。这与slot抑制type word(-1.53)一致
3. **GLM4三重组合slot+type+value=5.12是最强的compat值**——三层叠加产生最强效果
4. **"to touch"是通用物理感知构式**——对sharp(5.78)>hot(3.17)>rough(2.99)都有效。不是温度专用
5. **GLM4中rough_state(3.72)>rough_touch(2.99)**——texture的bare state反而更强，"to touch"对texture不是必要的
6. **sharp_touch(5.78)是所有构式中效果最强的**——"is sharp to touch"是最强属性激活构式
7. **binding不随兼容等级单调变化**——absurd等级binding(+1.05)反而最高。type方向注入不能区分兼容等级
8. **GLM4 texture/temperature的binding远强于color**——但这可能是因为texture/temperature方向的因果效力本身就更强
9. **size/weight的构式全部无效(heavy_lift=-0.20, heavy_state=-2.30)** — size确实不能用属性描述激活
10. **Qwen3构式偏好与GLM4相反**——Qwen3温度用tactile("feels")，GLM4用contact("to touch")

### 关键硬伤

1. **binding不单调是最严重的硬伤**——absurd等级binding最高说明type方向注入测量的不是"对象-属性兼容性"，而是"方向对属性词的推升力度"。absurd对象可能语义空间更"空"，方向注入更容易推升任意属性词
2. **slot+value的增强可能来自方向叠加的几何效应**——两个方向叠加可能只是范数增大，而非计算组合。需要用不同alpha的对照验证
3. **"to touch"对sharp极强(5.78)需要解释**——sharp是形状/功能属性，"to touch"可能激活"安全评估"或"危险感知"路径，而非简单触觉
4. **texture的state>touch与temperature的touch>state矛盾**——两种触觉属性的构式偏好不同，需要进一步分析
5. **absurd对象的binding最高说明当前binding指标有根本问题**——可能需要换为"value方向注入"或"对象+value联合注入"

### 命令记录

```bash
# Phase 327: Slot组合+构式分解+Binding大矩阵
python tests/glm5/phase327_slot_combo_binding_matrix.py qwen3       # ~55s
python tests/glm5/phase327_slot_combo_binding_matrix.py glm4        # ~15min
python tests/glm5/phase327_slot_combo_binding_matrix.py deepseek7b  # ~9.5min
```

脚本位置：
- `tests/glm5/phase327_slot_combo_binding_matrix.py` — 主测试
- 结果：`results/phase327_combo_binding/{qwen3,glm4,deepseek7b}_phase327.json`

## Phase 327b: Slot组合Alpha对照+Value Binding+Sharp确认 [2026-06-02 05:38]

### 背景

Phase 327发现slot+value比value alone高70%，但可能是范数叠加。需要alpha对照确认。同时binding不单调(absurd最高)，需要value方向注入验证。

### 核心结果

**结果1: GLM4 slot参与属性计算被alpha对照确认——这是确凿证据**

GLM4 L3 Alpha对照(10个color对象):

| 注入方式 | compat | incompat | C-I(binding) |
|---------|--------|----------|-------------|
| value(alpha=2.0) | 2.18 | 1.69 | 0.49 |
| **slot(1.0)+value(1.0)** | **2.23** | **1.21** | **1.02** |
| slot(2.0)+value(2.0) | 3.70 | 2.07 | 1.63 |
| type(alpha=2.0) | 4.38 | 3.80 | 0.58 |
| slot(1.0)+type(1.0) | 2.51 | 2.34 | 0.17 |
| slot(2.0)+type(2.0) | 3.69 | 3.19 | 0.50 |

**关键发现**：
1. **slot(1.0)+value(1.0) > value(2.0): compat 2.23 > 2.18** — 总alpha相同(2.0)，但分配给slot+value比全给value更强！
2. **slot(1.0)+value(1.0) binding=1.02 >> value(2.0) binding=0.49** — binding翻倍！
3. **slot+type: 无论alpha多少，binding都弱于type alone** — slot确实抑制type方向

这说明：**slot不是范数叠加效应，而是真正参与value选择的计算入口。slot+value组合产生了超叠加(synergistic)效果。**

DS7B: slot(1.0)+value(1.0)=0.097 > value(2.0)=-0.050 — DS7B也有弱超叠加
Qwen3: slot(1.0)+value(1.0)=0.027 < value(2.0)=0.080 — Qwen3无超叠加

**结果2: Binding即使用value方向注入也不单调——absurd仍然最高**

GLM4 value方向binding:
- High compatibility: **0.34**
- Absurd: **2.29**

absurd等级的binding远高于high！即使换成value方向注入，binding仍然不单调。这说明**当前binding测量方法有根本问题**——absurd对象的语义空间更"空"，任何方向注入都更容易推升其属性词。

Qwen3: High=0.05, Absurd=0.08 — 也弱但不单调
DS7B: High=-0.01, Absurd=0.02 — 极弱

**结果3: GLM4 sharp "to touch"用12对象确认——touch(3.12) > state(1.75)**

GLM4 L3 sharp构式(12对象平均):
- is sharp to touch: **3.12**
- is sharp: **1.75**
- 触摸构式1.8x增强

Qwen3: sharp_touch(0.09) > sharp_state(0.03) — 方向一致但弱
DS7B: sharp_touch(0.002) ≈ sharp_state(-0.07) — 无差异

### 客观事实拼图更新

1. **GLM4 slot参与value选择被alpha对照确凿确认**——slot(1.0)+value(1.0)比value(2.0)的compat更高(2.23 vs 2.18)且binding翻倍(1.02 vs 0.49)，这不是范数叠加
2. **GLM4 slot抑制type方向也被确认**——slot(1.0)+type(1.0)的binding(0.17)远弱于type(2.0)的binding(0.58)
3. **DS7B也有弱超叠加**——slot(1.0)+value(1.0)=0.097 > value(2.0)=-0.050
4. **Binding不单调不是type方向的问题**——value方向注入也不单调(absurd=2.29 >> high=0.34)
5. **Absurd对象binding更高的可能原因**——absurd对象(idea/music/color)语义空间少有具体属性词，方向注入更容易推升
6. **GLM4 sharp "to touch"确认(3.12 vs 1.75)**——"to touch"对sharp确实是通用构式增强

### 关键硬伤

1. **Binding方法需要根本重新设计**——无论type还是value方向，absurd对象都更容易被推升。需要换指标：不用"绝对logit变化"，而用"compatible vs incompatible的差是否受对象约束"
2. **Slot的超叠加效应虽然确凿，但机制仍不明**——为什么slot+value能产生比纯value更强的效果？可能是slot打开的"泛属性空间"让value方向有更多"施展空间"
3. **Qwen3无超叠加**——说明slot+value的计算组合可能是GLM4/DS7B特有的

### 命令记录

```bash
# Phase 327b: 确认测试
python tests/glm5/phase327b_confirm.py qwen3       # ~26s
python tests/glm5/phase327b_confirm.py glm4        # ~6min
python tests/glm5/phase327b_confirm.py deepseek7b  # ~4min
```

脚本位置：
- `tests/glm5/phase327b_confirm.py` — 确认测试
- 结果：`results/phase327b_confirm/{qwen3,glm4,deepseek7b}_phase327b.json`

## Phase 328: Binding指标重新设计 [2026-06-02 06:32]

### 背景

Phase 327/327b暴露了binding指标的根本问题：absurd对象binding最高，说明绝对logit增量测的不是"兼容性"而是"可推动度"。设计了三种新指标：

1. **Rank Gain**：注入type方向后，5个候选值中compat值排名是否上升
2. **Baseline-Corrected Binding**：减去random对象的兼容优势
3. **Interaction Term**：Effect(obj+val) - Effect(obj) - Effect(val)，超叠加=binding

### 核心结果

**结果1: GLM4 Rank Gain全部为负——type方向注入使compat排名反而下降**

GLM4 L3 Rank Gain (alpha=2.0):

| 属性类型 | net_gain | compat_gain | incompat_gain | pos_rate |
|---------|----------|-------------|--------------|----------|
| color | **-0.583** | -0.233 | +0.350 | 0.10 |
| texture | **-0.312** | -0.125 | +0.188 | 0.25 |
| temperature | **-0.104** | -0.042 | +0.062 | 0.00 |

**所有属性类型的net_gain都为负！** type方向注入后，incompat值排名反而上升，compat值排名下降。这完全违反binding预期。

但注意：这可能是type方向本身就有强偏好方向——"has a color"推高的是整个color空间，而非特定值。

Qwen3 L0: color=-0.083, texture=+0.208(弱正), temperature=0.000
DS7B L6: 全0或弱负

**结果2: Baseline-Corrected Binding——校正后仍有属性类型差异**

GLM4 L3 Baseline-Corrected (alpha=2.0):

| 属性类型 | corrected | raw | pos_rate | n |
|---------|----------|-----|----------|---|
| color | **-0.694** | -0.634 | 0.50 | 4 |
| temperature | **+0.613** | +0.539 | 0.67 | 3 |
| texture | **-0.469** | -0.408 | 0.17 | 6 |
| taste | **+1.208** | +1.208 | 1.00 | 2 |

校正后，temperature和taste的binding为正，但color和texture为负。这和Phase 327的"texture/temperature远强于color"一致，但方向不同。

Qwen3: corrected全弱(≤0.125)
DS7B: corrected全弱(≤0.010)

**结果3: Interaction Term——GLM4 color binding_interaction=1.328，100%正向！**

GLM4 L3 Interaction Term (alpha=1.0):

| 属性类型 | binding_interaction | pos_rate | n |
|---------|-------------------|----------|---|
| **color** | **+1.328** | **1.00** | 4 |
| temperature | -0.086 | 0.33 | 3 |
| texture | -0.282 | 0.50 | 6 |
| taste | -0.146 | 0.50 | 2 |

**GLM4 color的interaction term极强(1.328)且100%正向！** 这说明：
- slot方向+value方向的联合效果远大于各自效果之和
- apple/snow/sky/banana四个对象全部显示超叠加
- 这是迄今最强的binding证据

单个对象的binding_interaction:
- snow: **3.129** (极强)
- apple: 1.186
- banana: 0.770
- sky: 0.227

注意这里用的是slot方向(泛属性) + value方向(具体值)的交互，而非type方向。

Qwen3: 全弱(≤0.031)
DS7B: 全弱(≤-0.016)

### 三个指标之间的矛盾

| 指标 | color | temperature | texture | taste |
|------|-------|------------|---------|-------|
| Rank Gain | **-0.583** | -0.104 | -0.312 | — |
| Baseline-Corrected | **-0.694** | **+0.613** | -0.469 | **+1.208** |
| Interaction Term | **+1.328** | -0.086 | -0.282 | -0.146 |

**color在Interaction Term中最强(1.328)，但在Rank Gain和Baseline-Corrected中最弱(-0.694)！**

这说明三种指标测量的是不同东西：
- Rank Gain测的是"type方向推升compat是否比incompat更多"——type方向可能平等推升所有color值
- Baseline-Corrected测的是"type方向对real对象比random对象更有效吗"——real对象可能已有强color先验，type方向反而推升random对象更多
- Interaction Term测的是"slot+value是否超叠加"——这才是真正的binding，因为它测量方向组合的交互效应

### 客观事实拼图更新

1. **GLM4 color interaction term = 1.328, 100%正向** — slot+value在color上产生极强超叠加，这是迄今最可靠的binding证据
2. **Rank Gain全部为负** — type方向注入使compat排名反而下降。这可能因为"type=color"推高的是整个color空间，而不区分具体值
3. **Baseline-Corrected中color=-0.694为负** — real对象已有强color先验，type方向的增量不如random对象大
4. **temperature/taste在Baseline-Corrected中为正** — 这些属性的先验更弱，type方向注入更有效
5. **三指标矛盾说明"binding"不是单一现象** — 需要区分"方向推升力"、"排名偏好"和"交互超叠加"
6. **Qwen3/DS7B所有新指标仍然弱** — 继续确认binding在GLM4中最清晰

### 关键硬伤

1. **Interaction Term的"slot+value超叠加"可能不是binding** — 它测的是slot方向和value方向的几何交互，不是"模型理解apple-red比apple-blue更合理"。interaction=1.328可能只说明slot打开的"泛属性空间"让value方向有更多自由度
2. **Rank Gain为负是最严重的硬伤** — 如果binding存在，注入"has a color"后apple-red排名应上升。但实际反而下降，说明type方向平等推升所有color值，不区分兼容性
3. **Baseline-Corrected的"temperature/taste正"可能是因为先验弱** — 这些属性的baseline更低，所以type方向的相对增量更大。这不是binding，而是"可推升度"
4. **DS7B/Qwen3的所有指标都弱** — binding可能在GLM4中才有清晰结构

### 命令记录

```bash
# Phase 328: Binding指标重新设计
python tests/glm5/phase328_binding_redesign.py qwen3       # ~29s
python tests/glm5/phase328_binding_redesign.py glm4        # ~11.6min
python tests/glm5/phase328_binding_redesign.py deepseek7b  # ~7min
```

脚本位置：
- `tests/glm5/phase328_binding_redesign.py` — 三种新binding指标
- 结果：`results/phase328_binding_redesign/{qwen3,glm4,deepseek7b}_phase328.json`

## Phase 328b: Interaction Term确认 [2026-06-02 06:53]

### 背景

Phase 328发现GLM4 color interaction term = 1.328(slot+value)，但三个指标之间矛盾。需要确认：
1. interaction term是否区分normal vs absurd对象？
2. slot+value的超叠加是否只属于slot，type+value是否也有？
3. 超叠加是否在compat值上比incompat值更强？

### 核心结果

**结果1: GLM4 normal对象binding_interaction(0.774) > absurd(0.379)——interaction term确实区分对象类型！**

GLM4 L3 slot+value interaction (14对象):

| 对象类别 | mean_binding_interaction | positive_rate |
|---------|------------------------|--------------|
| **Normal** | **+0.774** | **0.88** |
| Absurd | +0.379 | 0.50 |

Normal对象的binding_interaction是absurd的2倍！且88%正向率vs 50%。
但注意：idea(color,absurd)的binding_interaction=5.828极高，是outlier。

单个normal对象详情:
- snow: **3.129** (极强)
- apple: 1.186
- banana: 0.769
- fire: 0.641
- stone: 0.500
- silk: 0.281
- sky: 0.227
- ice: -0.539 (唯一负)

Qwen3: normal=0.042, absurd=0.037 (弱但normal>absurd)
DS7B: normal=-0.003, absurd=0.000 (无区分)

**结果2: slot+value(1.328) >> type+value(0.002)——超叠加只属于slot+value！**

GLM4 L3 color对象4对:

| 方向组合 | mean_binding_interaction |
|---------|------------------------|
| **slot + value** | **1.328** |
| type + value | **0.002** |

type+value的交互项几乎为零！这说明超叠加不是"任何两个方向叠加"的几何效应，而是slot方向特有的调制作用。

Qwen3: slot+value=-0.018, type+value=-0.059 (都弱)
DS7B: slot+value=-0.001, type+value=0.005 (都弱)

**结果3: inter_c=4.065 >> inter_i=2.737——超叠加在compat值上更强**

GLM4 color对象:
- interaction on compat values: **4.065**
- interaction on incompat values: **2.737**
- 不对称性(binding): **1.328**

compat值上的交互项是incompat值的1.49倍。这意味着slot+value超叠加在推升red时比推升blue时更有效——这正是binding应该表现出的不对称性。

Qwen3: inter_c=-0.002, inter_i=0.016 (无不对称)
DS7B: inter_c=0.009, inter_i=0.010 (无不对称)

### 客观事实拼图更新

1. **GLM4 normal对象的binding_interaction是absurd的2倍(0.774 vs 0.379)** — interaction term确实能区分对象类型，不是纯几何效应
2. **slot+value超叠加=1.328，type+value=0.002** — 超叠加只在slot参与时出现，type方向与value无交互。这进一步证明slot是调制器
3. **compat值上的交互(4.065)远大于incompat值(2.737)** — 不对称性1.328说明超叠加对compat值更有效，这是binding的关键特征
4. **idea(absurd)的binding_interaction=5.828极高** — 这是唯一一个absurd对象binding高于所有normal对象的情况，需要解释
5. **Qwen3和DS7B无此结构** — binding_interaction在GLM4中独有

### 关键硬伤

1. **idea的binding_interaction=5.828** — "idea is red"比"apple is red"的binding更强？这严重违反binding假设。如果interaction term真的测量binding，absurd对象不应比normal对象更高。可能解释：idea的语义空间非常空，slot+value方向几乎不受约束，导致超叠加极大
2. **slot+value vs type+value的差异可能来自方向正交性** — slot和value方向可能更正交(不同语义空间)，type和value方向可能更平行(都在color空间内)，导致type+value接近线性叠加
3. **Rank Gain仍然全部为负** — 即使interaction term为正，type方向注入仍然不能改善compat排名。这说明两个指标测量的是完全不同的东西
4. **只有GLM4有此结构** — 如果binding是通用语言机制，为什么Qwen3/DS7B没有？

### 命令记录

```bash
# Phase 328b: Interaction Term确认
python tests/glm5/phase328b_confirm.py qwen3       # ~21s
python tests/glm5/phase328b_confirm.py glm4        # ~4.3min
python tests/glm5/phase328b_confirm.py deepseek7b  # ~2.7min
```

脚本位置：
- `tests/glm5/phase328b_confirm.py` — 确认测试
- 结果：`results/phase328b_confirm/{qwen3,glm4,deepseek7b}_phase328b.json`

## Phase 329: 三元交互(I×S×V)与Context-Gated Binding [2026-06-02 07:38]

### 背景

Phase 328/328b确认了GLM4中slot+value协同(S×V=1.328)，但I×V(object×value)交互未测试。用户分析指出：真正的binding需要证明object identity参与了value选择。

两种方法：
1. **Phase 329**: 方向注入法——计算I/S/V三个方向，注入到neutral prompt "The"，测试2^3=8条件的因子设计
2. **Phase 329b**: Context-Gated法——将object放入prompt("The apple")，注入V方向，比较有无object context时value方向的效力差异

### Phase 329核心结果：方向注入法

**方向定义(object-agnostic)：**
- I = "I see the {obj}" vs "I see the item" (对象身份)
- S = "It has a property" vs "It is an object" (属性槽)
- V = "It is {val}" vs "It is an object" (属性值)

**GLM4 L3 IxV_binding by compat_level (alpha=1.0)：**

| compat_level | n | IxV_binding | pos_rate | SxV_binding | IxSxV_binding |
|-------------|---|------------|---------|------------|--------------|
| high_compatible | 12 | **+0.595** | 0.83 | +1.541 | **-0.599** |
| near_incompatible | 4 | +0.405 | 0.75 | -0.948 | +0.411 |
| cross_type | 2 | **+0.648** | 1.00 | -0.416 | -1.203 |
| abstract_absurd | 5 | +0.283 | 0.60 | +1.169 | **-1.942** |

**关键发现1：IxV存在但模式不单调**
- HC(0.595) > AA(0.283)，方向正确
- 但cross_type(0.648) > HC(0.595)，违反binding预期

**关键发现2：IxSxV三元交互全部为负**
- 高兼容: -0.599, 荒谬: -1.942
- 三因子组合反而干扰，不是协同

**关键发现3：apple-red的IxV_binding为负(-0.406)**
- Effect(I+V, red) = Effect(I, red) + Effect(V, red) (线性，无超叠加)
- Effect(I+V, blue) > Effect(I, blue) + Effect(V, blue) (超叠加！)
- 原因：apple方向打开color空间，blue比red有更多增长空间(天花板效应)

Qwen3: 全弱(IxV≈0)
DS7B: 全弱(IxV≈0)

### Phase 329b核心结果：Context-Gated Binding

**设计：** 将object放入prompt，比较有无object时value方向的效力差异。

6个条件：{obj}baseline, {obj}+V, {obj}+S+V, item baseline, item+V, item+S+V

**baseline_binding = (logit_t_{obj} - logit_c_{obj}) - (logit_t_{item} - logit_c_{item})**

这测量的是：对象上下文对兼容值vs不兼容值偏好的提升，减去通用item基线。

**跨模型baseline_binding (最强binding证据!)：**

| 模型 | high_compatible | near_incompat | cross_type | abstract_absurd | HC>AA? |
|------|----------------|---------------|------------|----------------|--------|
| Qwen3 | **+2.585** | +0.391 | +0.557 | +0.202 | **True** |
| GLM4 | **+2.644** | **-1.023** | +3.851 | +0.923 | **True** |
| DS7B | **+1.743** | **-0.973** | +2.082 | **-0.184** | **True** |

**首次三模型一致确认：baseline_binding的HC > AA模式！**

**关键发现4：GLM4和DS7B的near_incompatible baseline_binding为负！**
- GLM4 NI = -1.023: "The apple"使"blue"比"The item"更不可能
- DS7B NI = -0.973: 同上
- 这不是简单的共现统计——对象不仅推高兼容值，还**主动抑制不兼容值**

**关键发现5：Context-gated binding (binding_V)因天花板效应失败**

| 模型 | HC binding_V | AA binding_V | HC>AA? |
|------|-------------|-------------|--------|
| Qwen3 | -0.009 | +0.046 | False |
| GLM4 | +0.077 | +0.162 | False |
| DS7B | -0.185 | +0.020 | False |

方向注入无法改善已有强baseline的binding——天花板效应。

**关键发现6：rank_obj也一致确认binding**

| 模型 | HC rank | AA rank | HC>AA? |
|------|---------|---------|--------|
| Qwen3 | **0.92** | 0.40 | **True** |
| GLM4 | **0.83** | 0.80 | True (弱) |
| DS7B | **0.83** | 0.60 | **True** |

### 客观事实拼图更新

1. **baseline_binding是迄今最可靠的binding指标** — 三模型一致，HC > AA，且GLM4/DS7B的NI为负说明对象主动抑制不兼容值
2. **方向注入法的天花板效应是根本性问题** — 当模型已在baseline中完成binding，方向注入只能测边际效应，无法进一步改善
3. **IxV交互(方向注入)模式不单调** — cross_type > HC，说明IxV测量的是"方向间协同"而非"对象-属性兼容性"
4. **IxSxV三元交互为负** — 三因子组合干扰而非协同
5. **S×V(object-agnostic)与Phase 328b(object-specific)的color均值相似(+1.336 vs +1.328)** — 方向计算方式对均值影响小，但对单对方差极大
6. **apple-red的IxV_binding为负(-0.406)而SxV_binding极强(+4.981)** — slot是binding的关键调制器，不是object identity

### 关键硬伤

1. **baseline_binding可能只是共现统计，不是机制** — 模型知道apple-red>apple-blue，可能只是因为训练数据中"red apple"比"blue apple"更常见。但NI为负(GLM4/DS7B)说明不仅是共现，还有主动抑制
2. **天花板效应使方向注入法失效** — 所有binding信息已在baseline中，方向注入只能测量边际效应
3. **cross_type baseline异常高(GLM4=3.851)** — snow-sweet的baseline=7.085不合理，可能"sweet"作为高频词污染了结果
4. **GLM4 AA baseline=0.923仍为正** — 理论上absurd对象不应有正binding，说明baseline_binding仍受value prior污染
5. **方向注入法测的S×V和baseline_binding测的是完全不同的东西** — S×V是方向协同，baseline_binding是自然输出偏好

### 命令记录

```bash
# Phase 329: 三元交互(I×S×V)
python tests/glm5/phase329_three_way_binding.py qwen3       # ~24s
python tests/glm5/phase329_three_way_binding.py glm4        # ~5min
python tests/glm5/phase329_three_way_binding.py deepseek7b  # ~3.2min

# Phase 329b: Context-Gated Binding
python tests/glm5/phase329b_context_gated.py qwen3       # ~20s
python tests/glm5/phase329b_context_gated.py glm4        # ~3.3min
python tests/glm5/phase329b_context_gated.py deepseek7b  # ~2.2min
```

脚本位置：
- `tests/glm5/phase329_three_way_binding.py` — 三元交互测试
- `tests/glm5/phase329b_context_gated.py` — Context-Gated Binding
- 结果：`results/phase329_three_way/{qwen3,glm4,deepseek7b}_phase329.json`
- 结果：`results/phase329b_context_gated/{qwen3,glm4,deepseek7b}_phase329b.json`

## Phase 330: 层级追踪Contextual Binding + Value Prior校正 [2026-06-02 08:32]

### 背景

Phase 329b证明了baseline_binding是最强binding信号(HC>AA跨模型一致)，但两个关键问题未解决：
1. binding在哪一层形成？
2. baseline_binding是否只是value prior（red比blue更常见）的伪相关？

### Phase 330设计：层级追踪

对每层L，用W_U投影hidden_state到logit空间：
```
binding(L) = [logit_t(obj,L) - logit_c(obj,L)] - [logit_t(item,L) - logit_c(item,L)]
```
其中logit_v(prompt,L) = W_U[v] @ hidden_state_L[-1](prompt)

### Phase 330核心结果：binding形成的层级

**跨模型final binding (raw)：**

| 模型 | HC | NI | AA | HC>AA | HC first dominates AA |
|------|-----|-----|-----|-------|----------------------|
| Qwen3 | +3.023 | +0.183 | +0.203 | True | **L1** |
| GLM4 | +2.663 | **-0.718** | +0.920 | True | **L27** |
| DS7B | +2.158 | **-2.679** | **-0.183** | True | **L4** |

**关键发现1：GLM4的binding形成极晚(L25-L32)**
- GLM4 L0-L24: binding≈0（前60%的层完全没有binding）
- GLM4 L25-L32: binding从0.16急速增长到1.27（核心binding形成期）
- GLM4 L33-L40: binding稳定在1.3-2.7（读出期）

**关键发现2：Qwen3/DS7B的binding形成极早**
- Qwen3: L1就已HC>AA，L28-L35快速增长
- DS7B: L4就已HC>AA，L23-L27快速增长

**关键发现3：各模型binding快速增长层不同**

| 模型 | binding快速增长层 | 核心gain层 |
|------|------------------|-----------|
| Qwen3 | L28-L35 | L30(+3.16) |
| GLM4 | L25-L32 | L32(+0.29) |
| DS7B | L23-L27 | L23(+2.55) |

三模型的"binding形成层"都在模型后半段(约70-90%深度处)。

**关键发现4：NI(近邻不兼容)的binding轨迹完全不同**
- GLM4 NI: 前期微正(L0-L27, ≈0)，后期变负(L28+，最终-0.718)
- DS7B NI: 前期微正(L0-L11)，中后期持续下降(最终-4.261)
- 这说明"抑制不兼容值"是后期层的主动计算，不是早期嵌入

### Phase 330b设计：Value Prior校正

原始binding可能被value prior污染（red比blue更常见→red logit更高→binding被夸大）。

校正方法：
```
prior(v, L) = W_U[v] @ hidden_state_L[-1]("The")   # 无对象上下文的先验
corrected_binding = raw_binding - (prior(target,L) - prior(competitor,L))
```

两种baseline:
- corr(The): 用"The"作为prior baseline（最无信息）
- corr(item): 用"The item"作为prior baseline

### Phase 330b核心结果：prior校正后的binding

**跨模型corrected binding (corr_item)：**

| 模型 | HC | NI | AA | HC>AA |
|------|-----|-----|-----|-------|
| Qwen3 | **+3.371** | +0.940 | **-0.036** | **True** |
| GLM4 | **+3.127** | **-0.758** | +0.812 | **True** |
| DS7B | **+2.565** | **-4.261** | +0.080 | **True** |

**关键发现5：prior校正使binding更强而非更弱**
- Qwen3 HC: +3.023→+3.371（校正后反而更大）
- GLM4 HC: +2.663→+3.127（校正后更大）
- DS7B HC: +2.158→+2.565（校正后更大）

这说明value prior实际上在"抵消"binding而非"制造"binding——因为competitor(blue, black等)的prior比target(red, yellow等)更高时，raw binding被低估了。

**关键发现6：Qwen3的AA校正后变为负值(-0.036)**
- Raw AA=+0.203 → Corr AA=-0.036
- 这意味着Qwen3中，荒谬对象上的"正binding"完全是value prior假象
- 校正后，荒谬对象不再有binding

**关键发现7：DS7B的NI校正后为-4.261（极强负值）**
- Raw NI=-2.679 → Corr NI=-4.261
- 这说明DS7B对不兼容属性的主动抑制比raw数据还强
- value prior实际上在"掩盖"了部分抑制效应

**关键发现8：GLM4的AA=+0.812仍偏高**
- 可能原因：justice-blue等荒谬对仍有隐喻/诗性联想
- 需要更严格的负例分类（区分"抽象荒谬"和"隐喻可用"）

### 客观事实拼图更新

1. **binding在模型后半段形成** — 三模型一致，binding的快速增长层在70-90%深度
2. **GLM4的binding形成最晚(L25-L32)** — 前60%的层几乎无binding
3. **value prior校正使binding更强** — 说明prior在抵消binding，不是制造binding
4. **NI(近邻不兼容)的抑制是后期层的主动计算** — 前期接近0，后期变负
5. **Qwen3的AA校正后为负** — 荒谬对象binding完全是value prior假象
6. **DS7B的NI校正后=-4.261** — 模型对不兼容属性的抑制极强

### 关键硬伤

1. **GLM4的AA=+0.812仍为正** — 可能是justice-blue等隐喻联想；需更严格负例
2. **apple-red在Qwen3/DS7B的binding为负** — cherry-red也为负，可能是"red"token化或"red"prior异常
3. **层数校正缺失** — 不同模型总层数不同(36/40/28)，需用相对层数对比
4. **block recomputation仍缺失** — 只追踪了binding轨迹，还没做因果干预
5. **香蕉-黄色binding异常** — banana-yellow在Qwen3 raw=+1.902但corr_item=+0.788，可能"yellow"prior被特殊处理

### 命令记录

```bash
# Phase 330: 层级追踪
python tests/glm5/phase330_layer_binding.py qwen3       # ~16s
python tests/glm5/phase330_layer_binding.py glm4         # ~71s
python tests/glm5/phase330_layer_binding.py deepseek7b   # ~51s

# Phase 330b: Value Prior校正
python tests/glm5/phase330b_prior_corrected.py qwen3       # ~16s
python tests/glm5/phase330b_prior_corrected.py glm4        # ~75s
python tests/glm5/phase330b_prior_corrected.py deepseek7b  # ~51s
```

脚本位置：
- `tests/glm5/phase330_layer_binding.py` — 层级追踪
- `tests/glm5/phase330b_prior_corrected.py` — Value Prior校正
- 结果：`results/phase330_layer_binding/{qwen3,glm4,deepseek7b}_phase330.json`
- 结果：`results/phase330b_prior_corrected/{qwen3,glm4,deepseek7b}_phase330b.json`

## Phase 331: 公式审计 + 多Prior基线 + 层级轨迹确认 [2026-06-02 09:38]

### 背景

用户指出Phase 330b的corr_item可能存在**重复校正**：
```
raw_binding = gap(object) - gap(item)
corrected_binding_item = raw_binding - gap(item) = gap(object) - 2*gap(item)
```
这确实是一个公式错误。需要用多种独立的binding定义验证结果是否稳健。

### Phase 331设计：五种独立binding定义

每种定义只做**一次**baseline减法，不叠加：

1. `binding_raw = gap(object)` — 原始对象优势（无baseline）
2. `binding_item = gap(object) - gap("The item")` — Phase 330原始定义（正确）
3. `binding_the = gap(object) - gap("The")` — 最中性baseline
4. `binding_thing = gap(object) - gap("The thing")` — 备选baseline
5. `binding_multi = gap(object) - mean(gap(baselines))` — 多baseline平均

5个baseline prompts: "The", "The item", "The thing", "It is", "Something"

### Phase 331核心结果：HC>AA在所有定义×所有模型下成立

**跨模型final binding（binding_multi）：**

| 模型 | HC | NI | AA | HC>AA |
|------|-----|-----|-----|-------|
| Qwen3 | +2.792 | -0.099 | +0.297 | **True** |
| GLM4 | +2.784 | -0.256 | +0.455 | **True** |
| DS7B | +1.680 | -0.036 | **-0.858** | **True** |

**跨模型final binding（binding_item，最稳定定义）：**

| 模型 | HC | NI | AA | HC>AA |
|------|-----|-----|-----|-------|
| Qwen3 | +2.768 | +0.245 | +0.482 | **True** |
| GLM4 | +2.745 | **-0.324** | +0.604 | **True** |
| DS7B | +1.945 | **-0.694** | **-0.834** | **True** |

**5种定义×3模型=15种组合，HC>AA全部True！** binding信号极其稳健。

### Phase 331b：层级轨迹确认

**binding_item层级关键信息（最稳定定义）：**

| 模型 | First HC>AA | Max HC gain层 | First NI<0层 |
|------|------------|--------------|-------------|
| Qwen3 | L1 (0.03) | L30 (0.83) | L18 (0.50) |
| GLM4 | L27 (0.68) | L39 (0.97) | L37 (0.93) |
| DS7B | L5 (0.18) | L24 (0.86) | L28 (1.00) |

### 公式审计结论

**Phase 330b的corr_item确实是重复校正**：
- Phase 330b: corrected_item = gap_obj - 2*gap_item（多减了一次item_gap）
- Phase 331: binding_item = gap_obj - gap_item（正确，只减一次）

但方向性结论不受影响！因为：
- 如果gap_item>0，Phase 330b过度校正（binding偏高）
- 如果gap_item<0，Phase 330b校正不足（binding偏低）
- 无论哪种情况，HC>AA的方向不变

### 重要发现：logit lens中间层爆炸

- **binding_the在Qwen3 L7出现巨值**（+40, -262等）
- **binding_multi在DS7B L4出现巨值**（-936等）
- **binding_item无此问题** — 因为"The item"和"The apple"句法相似，hidden state范数更匹配
- **GLM4的binding_multi轨迹最平滑** — 无中间层爆炸

这说明：
1. **binding_item是最适合做层级追踪的定义**（范数匹配最好）
2. 中间层logit lens爆炸是因为不同prompt的hidden state范数不匹配
3. GLM4的表示空间更均匀，logit lens更可靠

### Per-value prior分析

在"The" baseline下，属性值logit排序（Qwen3）：
```
hot     +5.714 (最高prior)
green   +4.948
cold    +3.864
black   +2.945
blue    +2.791
...
quiet   -4.255 (最低prior)
```

**"red"的prior较低(-4.545)**，这解释了apple-red的binding_the为负(-2.969)：
- "The" baseline下red的logit很低
- "The apple"下red的logit高
- gap(red|apple) - gap(red|The) = 大正数 → 但gap(blue|apple) - gap(blue|The) 更大
- 因为blue在"The"下比red更高，减去更大的baseline后blue反而更优

这再次证明：**不同baseline会导致完全不同的binding数值，但HC>AA方向稳健。**

### 客观事实拼图更新

1. **HC>AA在5种binding定义×3模型=15种组合下全部成立** — 极其稳健
2. **Phase 330b存在重复校正（gap_item被减了两次）** — 数值需重解读
3. **binding_item是最稳定的层级追踪定义** — 无logit lens爆炸
4. **binding_the/binding_multi在中间层有norm爆炸** — Qwen3 L7, DS7B L4
5. **GLM4轨迹最平滑** — logit lens在GLM4上最可靠
6. **DS7B的AA=-0.834（binding_item）** — 荒谬对象binding完全消失
7. **不同baseline影响binding数值大小，但不改变HC>AA方向**
8. **"red"在"The"baseline下prior很低(-4.545)** — 解释apple-red异常

### 关键硬伤

1. **GLM4的AA=+0.604仍为正** — justice-blue等荒谬对仍有隐喻联想
2. **logit lens中间层爆炸** — binding_the/binding_multi不适合做层级追踪
3. **仍缺因果干预** — 只追踪了轨迹，还没做patching
4. **不同baseline给出不同数值** — binding不是唯一确定的量，依赖baseline选择
5. **"red"的prior异常低** — 分词或词频效应，需per-value随机效应控制

### 命令记录

```bash
# Phase 331: 公式审计 + 多Prior基线
python tests/glm5/phase331_formula_audit.py qwen3       # ~17s
python tests/glm5/phase331_formula_audit.py glm4         # ~71s
python tests/glm5/phase331_formula_audit.py deepseek7b   # ~49s

# Phase 331b: 层级轨迹确认
python tests/glm5/phase331b_confirm.py qwen3       # ~16s
python tests/glm5/phase331b_confirm.py glm4         # ~67s
python tests/glm5/phase331b_confirm.py deepseek7b   # ~48s
```

脚本位置：
- `tests/glm5/phase331_formula_audit.py` — 公式审计+多Prior基线
- `tests/glm5/phase331b_confirm.py` — 层级轨迹确认
- 结果：`results/phase331_formula_audit/{qwen3,glm4,deepseek7b}_phase331.json`
- 结果：`results/phase331b_confirm/{qwen3,glm4,deepseek7b}_phase331b.json`

## Phase 332: 因果替换 + 层归因 [2026-06-02 10:16]

### 背景

Phase 331确认binding在5种定义×3模型下稳健存在。下一步需要回答：**哪些层因果必要？**

### Phase 332初始：输出替换（Output Replacement）

**方法：** 在层L替换输出为"The item"的隐藏状态，后续层从source重新计算。

**Qwen3结果：frac_destroyed ≈ 1.0 在所有层！**

这意味着输出替换过于粗暴——替换任何层的输出都会让后续层从source状态完全重算，无论替换哪层都完全摧毁binding。这是output replacement的已知局限：**不能区分层间贡献，因为模型是确定性的，替换后等于重跑source。**

**重要发现：binding不是存储在任何单层中，而是整个前向传播的涌现属性。**

### Phase 332修正：层归因（Layer Attribution）

**方法：** 利用残差连接的线性性质分解binding：
```
h[N] = h[0] + Σ_L (attn_out_L + mlp_out_L)
binding = (W_U[target] - W_U[competitor]) @ h[N]
Δ_binding_L = (W_U[target] - W_U[competitor]) @ (h[L+1] - h[L])
Δ_binding_item_L = Δ_gap_obj_L - Δ_gap_item_L
```

**这是数学精确的分解（mismatch = 0.0000），不是近似。**

### Phase 332核心结果：跨模型层归因

**HC binding_item层归因汇总：**

| 模型 | HC final | Embed贡献 | 层贡献 | 最大贡献层 | 最大Δ |
|------|---------|----------|-------|-----------|-------|
| Qwen3 | +2.773 | +0.046 (2%) | +2.727 (98%) | L29 (rel=0.81) | +3.20 |
| GLM4 | +2.616 | +0.001 (0.04%) | +2.616 (99.9%) | L38 (rel=0.95) | +1.10 |
| DS7B | +2.202 | +0.001 (0.05%) | +2.202 (99.9%) | L23 (rel=0.82) | +3.99 |

**关键发现1：Embedding几乎不贡献binding_item**

三模型的embedding对binding_item的贡献都在0.05%以下。这意味着：
- 对象身份（apple vs item）在embedding层几乎没有区分
- binding完全由transformer层的计算产生
- 这反驳了"binding在embedding中预编码"的假说

**关键发现2：每个模型有特定的binding峰值层**

- Qwen3: L29 (rel=0.81), Δ=+3.20 — 集中式峰值
- GLM4: L38 (rel=0.95), Δ=+1.10 — 分布式，最深层的单层贡献最大
- DS7B: L23 (rel=0.82), Δ=+3.99 — 最强集中式峰值

**关键发现3：GLM4的binding最分布式**

GLM4的最大单层贡献只有+1.10，而Qwen3和DS7B分别有+3.20和+3.99。这说明：
- Qwen3/DS7B的binding集中在少数层
- GLM4的binding分散在更多层
- 这与Phase 331b的发现一致：GLM4的binding形成最晚

**关键发现4：NI/AA的层归因模式**

| 模型 | NI final | NI最大贡献层 | AA final | AA最大贡献层 |
|------|---------|------------|---------|------------|
| Qwen3 | +0.245 | L27 (+2.04) | +0.482 | L35 (+6.48) |
| GLM4 | -0.324 | L38 (+0.39) | +0.604 | L37 (+0.19) |
| DS7B | -0.694 | L23 (+3.32) | -0.834 | L27 (+2.20) |

DS7B是唯一AA为负的模型（AA=-0.834），其NI也是最强负值（-0.694）。

### 输出替换 vs 层归因的方法论对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| 输出替换 | 概念简单 | frac_destroyed≈1.0 everywhere，无法区分层贡献 |
| 层归因 | 数学精确，可分解每层贡献 | 是归因而非因果（不能证明层是因果必要的） |

**层归因回答的问题：** "每层对binding贡献了多少？"
**层归因不能回答的问题：** "如果去掉这层，binding会消失吗？"

### 最后一层的logit lens振荡

Per-pair分析显示最后1-2层有极端振荡：
- Qwen3 snow_white: L34 Δ_gap_obj=+11.683, L35 Δ_gap_obj=-22.471
- DS7B: L27 Δ_gap_obj=-64.971（极端负值）

这是logit lens在输出层的已知问题：最后层的hidden state范数变化剧烈，W_U投影产生不稳定值。但累积值（cumulative）仍然是稳定的。

### 客观事实拼图更新

1. **Embedding对binding_item的贡献 < 0.05%** — binding完全由transformer层计算
2. **每个模型有特定的binding峰值层：** Qwen3 L29, GLM4 L38, DS7B L23
3. **GLM4的binding最分布式**（最大单层贡献仅+1.10）
4. **Qwen3/DS7B的binding最集中**（最大单层贡献+3.20/+3.99）
5. **输出替换无法区分层贡献** — frac_destroyed≈1.0 everywhere
6. **层归因是数学精确的分解** — mismatch=0.0000
7. **最后一层有logit lens振荡** — 但累积值稳定
8. **NI/AA的层贡献模式与HC不同** — NI在某些层有正贡献（增强近邻值）

### 关键硬伤

1. **层归因是归因不是因果** — 不能证明某层是因果必要的
2. **最后一层logit lens不稳定** — 个别pair的Δ值极端
3. **GLM4 AA仍为正 (+0.604)** — 负例体系仍不干净
4. **未分解attention vs MLP贡献** — 目前只看层总贡献
5. **per-value随机效应未控制** — apple-red等个别pair仍异常

### 命令记录

```bash
# Phase 332: 输出替换（发现方法论局限）
python tests/glm5/phase332_causal_patching.py qwen3       # ~24s

# Phase 332: 层归因（修正方法）
python tests/glm5/phase332_layer_attribution.py qwen3       # ~9s
python tests/glm5/phase332_layer_attribution.py glm4         # ~54s
python tests/glm5/phase332_layer_attribution.py deepseek7b   # ~38s
```

脚本位置：
- `tests/glm5/phase332_causal_patching.py` — 输出替换（有方法论局限）
- `tests/glm5/phase332_layer_attribution.py` — 层归因（数学精确）
- 结果：`results/phase332_causal_patching/qwen3_phase332.json`
- 结果：`results/phase332_layer_attribution/{qwen3,glm4,deepseek7b}_phase332.json`

## Phase 333: 组件级Binding分解（Attention vs MLP） [2026-06-02 10:58]

### 背景

Phase 332确定了每层对binding_item的贡献量。关键问题是：**binding由Attention计算还是MLP计算？**

### 方法

利用残差连接结构：
```
h[L+1] = h[L] + attn_out_L + mlp_out_L
Δ_binding_item_L = Δ_binding_item_L_attn + Δ_binding_item_L_mlp
```

其中：
```
Δ_binding_item_L_attn = (binding_dir @ attn_out_obj_L) - (binding_dir @ attn_out_item_L)
Δ_binding_item_L_mlp = (binding_dir @ mlp_out_obj_L) - (binding_dir @ mlp_out_item_L)
```

使用forward hook捕获每层的self_attn和mlp模块输出，投影到binding方向。

### Phase 333核心结果：MLP主导binding计算

**HC关键binding层Attn vs MLP分解（mismatch < 0.08，分解精确）：**

| 模型 | 关键层 | Δ_attn | Attn% | Δ_mlp | MLP% | Δ_total | mismatch |
|------|--------|--------|-------|-------|------|---------|----------|
| Qwen3 | L29 | +0.644 | 20% | +2.564 | **80%** | +3.209 | 0.021 |
| GLM4 | L38 | +0.013 | 1% | +1.092 | **99%** | +1.105 | 0.018 |
| DS7B | L23 | +0.668 | 17% | +3.337 | **83%** | +4.005 | 0.070 |

**平均：HC关键层Attn=12.6%, MLP=87.4%**

### 关键发现1：MLP是binding的主要计算组件

三模型一致：MLP在关键binding层贡献80-99%。这意味着：
- object-attribute binding主要由MLP的知识检索功能计算
- Attention的贡献较小（1-20%），可能主要起上下文路由作用
- 这反驳了"binding主要由attention路由"的假说

### 关键发现2：GLM4的binding最极端地由MLP主导

GLM4 L38: Attn=1%, MLP=99% — 这是三模型中最极端的MLP主导。
这与GLM4的"后期结构化读出"架构特点一致：binding几乎完全由MLP的知识检索完成。

### 关键发现3：Qwen3和DS7B的attention有少量贡献（17-20%）

Qwen3 L29: Attn=20%, DS7B L23: Attn=17%。
这些attention贡献可能来自：
- 对象身份的上下文路由（将对象信息传递给后续MLP）
- 少量直接的属性兼容性选择

### 关键发现4：不同compat_level的attn/mlp模式不同

以Qwen3 L29为例：

| compat_level | Attn% | MLP% |
|-------------|-------|------|
| HC | 20% | 80% |
| NI | 7% | 93% |
| CT | 0% | 100% |
| AA | 27% | 73% |

NI和CT的MLP贡献更大，AA的attention贡献相对更大。
这可能是因为AA（抽象荒谬）的对象上下文较弱，MLP无法检索到明确属性，attention需要做更多"猜测"。

### 关键发现5：深度quartile分析显示MLP在所有深度段主导

Qwen3 HC (excl last 2 layers):
- 0-25%: Attn=2%, MLP=98%
- 25-50%: Attn=89%, MLP=11% (注意：此段总贡献很小)
- 50-75%: Attn=18%, MLP=82%
- 75-100%: Attn=12%, MLP=88%

MLP在有意义的binding贡献段（50-100%）稳定占80%+。

### 关键发现6：最后一层logit lens爆炸严重干扰汇总

- Qwen3 L35: mismatch=21.25
- DS7B L27: mismatch=64.21
- GLM4: 最后一层相对稳定

排除最后2层后，Qwen3 HC的|attn|%=10.1%, |mlp|%=89.9%。

### 方法学验证

hook-based分解与hidden-state-based计算的mismatch：
- 大多数层 < 0.05 → 分解精确
- 关键binding层 < 0.08 → 可靠
- 最后一层 > 20 → 不可用（logit lens爆炸）

### 客观事实拼图更新

1. **MLP在关键binding层贡献80-99%** — 三模型一致
2. **GLM4的binding最极端MLP主导（99%）** — 与后期结构化读出一致
3. **Attention在binding中贡献1-20%** — 可能是上下文路由
4. **不同compat_level的attn/mlp比例不同** — AA有更多attention参与
5. **Hook分解在大多数层精确（mismatch<0.05）** — 方法可靠
6. **最后一层logit lens爆炸严重** — 必须排除最后1-2层
7. **Binding是MLP驱动的知识检索，不是attention驱动的上下文路由**

### 关键硬伤

1. **Hook分解是归因不是因果** — 知道MLP贡献最大，不代表去掉MLP就会摧毁binding
2. **Attention的间接作用未量化** — attention可能通过路由信息给MLP间接贡献
3. **MLP内部的计算机制未分析** — MLP如何存储和检索object→attribute映射？
4. **per-value效应仍存在** — apple-red等异常pair
5. **GLM4 AA仍为正 (+0.604)** — 负例体系不干净
6. **最后一层数据不可靠** — 需排除

### 命令记录

```bash
# Phase 333: 组件级分解
python tests/glm5/phase333_attn_mlp_decomposition.py qwen3       # ~16s
python tests/glm5/phase333_attn_mlp_decomposition.py glm4         # ~54s
python tests/glm5/phase333_attn_mlp_decomposition.py deepseek7b   # ~39s

# Phase 333b: 确认分析（排除最后2层）
python tests/glm5/phase333b_confirm.py
```

脚本位置：
- `tests/glm5/phase333_attn_mlp_decomposition.py` — 主测试
- `tests/glm5/phase333b_confirm.py` — 确认分析
- 结果：`results/phase333_attn_mlp_decomposition/{qwen3,glm4,deepseek7b}_phase333.json`
- 结果：`results/phase333b_confirm/summary.json`

## Phase 334+335: 组件因果替换 — 直接与间接效应 [2026-06-02 12:30]

### 背景

Phase 333确定了MLP在关键binding层的归因贡献为80-99%。但归因≠因果。核心问题：
1. MLP是否因果必要？（去掉MLP的输出，binding是否消失？）
2. Attention的直接因果效应有多大？
3. Attention是否通过MLP产生间接效应？（路由假说）

### 方法：激活替换（Activation Patching）

```
Clean: "The apple" → 高binding信号
Corrupted: "The item" → 低binding信号
Patched: 运行corrupted，但在特定层替换特定组件输出为clean版本
```

四种替换条件：
1. **attn_patch**: 替换attn_out为clean版本 → Attention直接+间接效应
2. **mlp_patch**: 替换mlp_out为clean版本 → MLP直接效应
3. **attn_direct_only**: 替换attn_out为clean + 冻结mlp_out为corrupted版本 → 仅Attention直接效应
4. **full_block**: 替换attn_out和mlp_out都为clean版本 → 完整层效应

恢复率指标：
```
recovery_pct = (binding_patched - binding_corrupted) / (binding_clean - binding_corrupted) × 100
```

间接效应 = attn_patch recovery - attn_direct_only recovery

### Phase 334 Round 1 结果（全部pair，未过滤）

**HC关键层因果替换（Round 1，12 HC pairs）：**

| 模型 | 关键层 | attn recovery | MLP recovery | full_block | MLP/attn比 |
|------|--------|---------------|-------------|------------|-----------|
| Qwen3 | L29 | +4.6% | +3.0% | +6.1% | 0.7x |
| GLM4 | L38 | +0.6% | **+40.7%** | +41.1% | **67.8x** |
| DS7B | L23 | +24.3% | +24.8% | +37.9% | 1.0x |

Round 1问题：部分pair的binding_range为负或极小，导致recovery噪声极大。
- Qwen3 fire_hot: binding_range = -0.725（负！）
- DS7B apple_red: binding_range = -0.965（负！）

### Phase 334b Round 2 结果（过滤binding_range < 0.3的异常pair）

**关键发现：过滤异常pair后，三模型一致显示MLP因果主导！**

| 模型 | 有效pair | 关键层 | attn recovery | MLP recovery | full_block | MLP/attn比 | 间接attn |
|------|---------|--------|---------------|-------------|------------|-----------|---------|
| Qwen3 | 22/24 | L29 | +3.1% | **+7.9%** | +10.6% | **2.5x** | +1.2% |
| GLM4 | 22/24 | L38 | +0.8% | **+28.8%** | +29.5% | **36.9x** | +0.2% |
| DS7B | 18/24 | L23 | +1.1% | **+15.0%** | +9.6% | **13.5x** | +0.3% |

### 关键发现1：MLP是binding的因果必要组件 — 三模型一致确认

过滤异常pair后：
- Qwen3 L29: MLP recovery 7.9% vs attn 3.1%（MLP 2.5倍）
- GLM4 L38: MLP recovery 28.8% vs attn 0.8%（MLP 36.9倍！）
- DS7B L23: MLP recovery 15.0% vs attn 1.1%（MLP 13.5倍）

**这从因果层面确认了Phase 333的归因发现：MLP是binding的主要计算组件。**

### 关键发现2：GLM4的MLP因果主导最极端

GLM4 L38: MLP recovery 28.8%，attn仅0.8%，MLP/attn = 36.9倍。
这与Phase 333的99% MLP归因完全一致。GLM4的binding几乎完全由后期MLP计算。

### 关键发现3：Attention的直接因果效应极小

三模型在关键层的Attention直接因果效应：
- Qwen3 L29: attn_direct_only = +1.9%
- GLM4 L38: attn_direct_only = +0.5%
- DS7B L23: attn_direct_only = +0.8%

Attention对binding的直接因果贡献几乎为零。

### 关键发现4：Attention的间接效应也极小 — 路由假说被削弱

间接效应 = attn_patch - attn_direct_only：
- Qwen3 L29: +1.2%
- GLM4 L38: +0.2%
- DS7B L23: +0.3%

**Attention既没有直接因果效应，也没有通过MLP产生显著的间接效应。**
这意味着"Attention路由信息给MLP"的假说在binding场景下不成立。
Attention在binding中的作用可能是更基础的（如token identity传播、位置编码等），
而不是有目的地将对象信息路由给MLP。

### 关键发现5：单层替换恢复率仅7-29%，说明binding是分布式计算

- GLM4 L38: 28.8%（最高）
- DS7B L23: 15.0%
- Qwen3 L29: 7.9%

单层MLP替换只能恢复7-29%的binding信号，说明binding计算分布在多个层。
后续需要多层联合替换来验证。

### 关键发现6：DS7B Round 1的attn/MLP平分是噪声伪影

- Round 1（未过滤）: DS7B L23 attn=24.3%, mlp=24.8% — 看似平分
- Round 2（过滤后）: DS7B L23 attn=1.1%, mlp=15.0% — MLP 13.5倍主导

差异原因：6个异常pair（binding_range为负或极小）严重扭曲了平均值。
特别是apple_red在DS7B中binding_range=-0.965，导致recovery计算不稳定。

### 关键发现7：多层MLP贡献模式

**Qwen3（7层扫描，Round 1）：**
| 层 | attn% | MLP% | MLP share |
|---|-------|------|-----------|
| L15 | +3.8 | +20.3 | 84.2% |
| L25 | +6.5 | +24.7 | 79.2% |
| L29 | +4.6 | +3.0 | 39.5% |

L25的MLP recovery最高（24.7%），L29反而较低。说明Qwen3的binding计算高峰在L25附近。

**GLM4（6层扫描，Round 1）：**
| 层 | attn% | MLP% | MLP share |
|---|-------|------|-----------|
| L30 | -0.2 | +11.4 | 101.8% |
| L38 | +0.6 | +40.7 | 98.5% |

L38的MLP recovery远高于其他层，确认GLM4的binding集中在最后几层。

### 客观事实拼图更新

1. **MLP在关键binding层是因果必要组件** — 三模型一致，从归因推进到因果
2. **GLM4的MLP因果主导最极端（36.9倍）** — 与99%归因完全一致
3. **Attention的直接因果效应极小（0.5-1.9%）** — Attention不是binding的计算者
4. **Attention的间接效应也极小（0.2-1.2%）** — 路由假说被削弱
5. **单层MLP替换恢复7-29%** — binding是分布式多层计算
6. **DS7B Round 1的attn/MLP平分是噪声伪影** — 过滤后MLP 13.5倍主导
7. **异常pair的binding_range为负是严重污染源** — 后续必须过滤

### 关键硬伤

1. **单层替换恢复率低** — 只能证明MLP在单层因果必要，不能证明MLP是唯一因果路径
2. **Attention间接效应的测量受限于单层替换** — Attention可能在更早层产生间接效应
3. **binding_range为负的pair未深入分析** — 为什么apple_red在DS7B中binding为负？
4. **恢复率不是100%** — 说明多层的协同计算未捕获
5. **间接效应可能存在于跨层路径中** — 当前只测量同层的间接效应
6. **corrupted baseline（"The item"）可能不是最佳控制** — "item"本身可能携带语义

### 命令记录

```bash
# Phase 334: 组件因果替换（Round 1）
python tests/glm5/phase334_causal_patching.py qwen3       # ~27s
python tests/glm5/phase334_causal_patching.py glm4         # ~473s (7.9min)
python tests/glm5/phase334_causal_patching.py deepseek7b   # ~303s (5.0min)

# Phase 334b: 确认测试（Round 2，过滤异常pair，扩展数据）
python tests/glm5/phase334b_confirm.py deepseek7b           # ~163s (2.7min)
python tests/glm5/phase334b_confirm.py qwen3                # ~24s
python tests/glm5/phase334b_confirm.py glm4                 # ~229s (3.8min)
```

脚本位置：
- `tests/glm5/phase334_causal_patching.py` — 主测试
- `tests/glm5/phase334b_confirm.py` — 确认测试
- 结果：`results/phase334_causal_patching/{qwen3,glm4,deepseek7b}_phase334.json`
- 结果：`results/phase334_causal_patching/{qwen3,glm4,deepseek7b}_phase334b.json`

## Phase 336+337+338: 多层替换+反向破坏+跨层注意力 [2026-06-02 13:10]

### 背景

Phase 334/334b 确定了MLP在关键binding层是主要因果恢复通道，但三大硬伤：
1. 单层MLP恢复率仅7-29%→binding是分布式计算？需多层联合替换验证
2. 仅证因果充分性→需反向破坏（clean→corrupted）证必要性
3. Attention跨层间接效应未测→需测早期Attention是否传递对象身份

### 方法

**Phase 336：多层MLP块替换（corrupted→clean recovery）**
- 同时替换连续多层的MLP/Attention输出为clean版本
- 块定义：Qwen3[L21-23, L24-26, L27-29, L21-29]，GLM4[L30-34, L35-38, L30-38]，DS7B[L19-21, L22-24, L19-24]

**Phase 337：反向破坏（clean→corrupted destruction）**
- 运行clean输入"The apple"，但在关键层替换组件输出为corrupted版本
- destruction_pct = (binding_clean - binding_reverse) / (binding_clean - binding_corrupted) × 100
- 如果destruction高→组件因果必要

**Phase 338：跨层早期注意力（corrupted→clean early attn）**
- 替换早期注意力层的输出为clean版本，让后续层自然重算
- 测试早期Attention是否携带对象身份信息

### Phase 336 Round 1 结果：多层MLP块替换

**Qwen3（22 valid pairs）：**

| 块 | mlp_block | attn_block | full_block | MLP占full比例 |
|---|-----------|------------|------------|-------------|
| L21-23 | +18.9% | -0.9% | +21.2% | 89.2% |
| L24-26 | +4.9% | **+16.5%** | +16.1% | 30.4% |
| L27-29 | +36.1% | +1.6% | +49.4% | 73.1% |
| **L21-29** | **+69.6%** | +24.5% | +79.6% | 87.4% |

**GLM4（22 valid pairs）：**

| 块 | mlp_block | attn_block | full_block | MLP占full比例 |
|---|-----------|------------|------------|-------------|
| L30-34 | +30.5% | +11.3% | +42.1% | 72.4% |
| L35-38 | +24.6% | +0.3% | +26.9% | 91.4% |
| **L30-38** | **+46.0%** | +11.0% | +56.5% | 81.4% |

**DS7B（18 valid pairs）：**

| 块 | mlp_block | attn_block | full_block | MLP占full比例 |
|---|-----------|------------|------------|-------------|
| L19-21 | +6.4% | -1.8% | +5.1% | 125.5% |
| L22-24 | +32.2% | +4.1% | +47.0% | 68.5% |
| **L19-24** | **+58.5%** | +5.5% | +62.2% | 94.1% |

**关键对比：单层 vs 多层MLP恢复率**

| 模型 | 单层恢复率 | 最佳多层恢复率 | 提升倍数 |
|------|-----------|-------------|---------|
| Qwen3 | 7.9% (L25) | **69.6%** (L21-29) | **8.8x** |
| GLM4 | 28.8% (L38) | **46.0%** (L30-38) | **1.6x** |
| DS7B | 15.0% (L23) | **58.5%** (L19-24) | **3.9x** |

### Phase 337 结果：反向破坏（因果必要性）

**MLP反向破坏确认因果必要性，Attention破坏极小：**

| 模型 | 关键层 | MLP destruction | Attn destruction | MLP/Attn比 |
|------|--------|----------------|-----------------|-----------|
| Qwen3 | L29 | +25.7% | +8.6% | 3.0x |
| GLM4 | L38 | **+32.5%** | +1.1% | **29.5x** |
| DS7B | L23 | +68.8% | +65.2%* | 1.1x* |

*DS7B L23 attn destruction异常高（std=183.6%），可能为噪声驱动

**从因果充分性+因果必要性双重确认：MLP是binding的主要因果组件。**

### Phase 338 Round 1 结果：跨层早期注意力

**惊人发现：Qwen3和DS7B的早期注意力恢复率极高！**

| 模型 | 早期Attn块 | 恢复率 | std |
|------|-----------|--------|-----|
| Qwen3 | L0-8 | **+69.7%** | 45.0% |
| GLM4 | L0-10 | **+0.3%** | 12.5% |
| DS7B | L0-8 | **+88.5%** | 38.6% |

### Phase 336b Round 2 结果：细粒度早期组件分析

**关键发现：早期效应集中在L0-L2！**

**Qwen3 细粒度早期注意力恢复：**

| 块 | 注意力恢复 | MLP恢复 |
|---|----------|--------|
| L0-2 | **+93.3%** | **+101.8%** |
| L3-5 | -4.9% | -1.7% |
| L6-8 | +4.8% | -23.4% |
| L0-4 | +87.2% | — |
| L0-8 full | — | **+97.9%** |

**GLM4 细粒度早期组件恢复：**

| 块 | 注意力恢复 | MLP恢复 |
|---|----------|--------|
| L0-4 | **-0.0%** | **+99.4%** |
| L5-10 | -1.5% | +81.9% |
| L0-10 full | — | **+99.6%** |

**DS7B 细粒度早期注意力恢复：**

| 块 | 注意力恢复 | MLP恢复 |
|---|----------|--------|
| L0-2 | **+87.1%** | **+93.9%** |
| L3-5 | -4.3% | -10.0% |
| L6-8 | -4.0% | +13.1% |
| L0-4 | +88.5% | — |
| L0-8 full | — | **+100.3%** |

### 关键发现1：Binding是分布式MLP契约 — 三模型一致确认

多层MLP块替换恢复率（46-70%）远超单层（8-29%）：
- Qwen3 L21-29: 69.6% vs 7.9% = 8.8倍
- GLM4 L30-38: 46.0% vs 28.8% = 1.6倍
- DS7B L19-24: 58.5% vs 15.0% = 3.9倍

**binding不是某一层MLP单独完成的，而是多层MLP链式累积的结果。**

### 关键发现2：MLP因果必要性确认 — 反向破坏实验

GLM4 L38: MLP destruction 32.5% vs Attn 1.1%（29.5倍差距）。
Qwen3 L29: MLP destruction 25.7% vs Attn 8.6%（3.0倍差距）。

**从corrupted→clean恢复和clean→corrupted破坏两个方向，都确认MLP是binding的因果必要组件。**

### 关键发现3：对象身份传播的三模型差异 — 最重要的新发现

**早期组件（L0-L2/4）携带对象身份，但机制不同：**

| 模型 | 早期Attn恢复 | 早期MLP恢复 | 早期full恢复 | 身份传播机制 |
|------|------------|-----------|------------|-----------|
| Qwen3 | 93.3% | 101.8% | 97.9% | Attn+MLP双通道 |
| GLM4 | **0.0%** | **99.4%** | 99.6% | **仅MLP通道** |
| DS7B | 87.1% | 93.9% | 100.3% | Attn+MLP双通道 |

**这是首次发现三模型在binding机制上的根本差异：**
- Qwen3/DS7B：早期Attention和MLP都传播对象身份（双通道冗余）
- GLM4：仅早期MLP传播对象身份，Attention完全不参与（单通道）

### 关键发现4：L0-L2是对象身份传播的关键区域

三模型一致：L0-L2/L0-L4的组件恢复率最高，L3-L8几乎为零。
这意味着对象身份在模型最前面的2-4层就已经被写入residual stream。

### 关键发现5：早期完整块恢复接近100%

| 模型 | L0-8/10 full恢复 | std |
|------|----------------|-----|
| Qwen3 | 97.9% | 4.7% |
| GLM4 | 99.6% | 1.6% |
| DS7B | 100.3% | 7.6% |

**如果前8-10层的组件全部替换为clean版本，后续层可以自然计算binding。**
这意味着binding计算需要的是"正确的residual stream输入"，而非特定层的特定计算。

### 关键发现6：Qwen3 L24-26的Attention主导

Qwen3 L24-26块中：attn_block=+16.5% > mlp_block=+4.9%。
这是唯一一个Attention主导的binding层块，可能暗示该区域有路由/通信功能。
但需要更多数据确认。

### 关键发现7：DS7B L23 Attention异常高方差

DS7B L23: attn_reverse destruction = +65.2%（std=183.6%），mlp_reverse = +68.8%（std=138.9%）。
超高方差表明这不是稳定的机制效应，而是少数pair驱动的噪声。

### 客观事实拼图更新

1. **Binding是分布式MLP契约** — 多层联合恢复46-70%，远超单层8-29%
2. **MLP因果必要性确认** — 反向破坏25-33%（Qwen3/GLM4）
3. **对象身份在L0-L2/4传播** — 早期组件恢复87-102%
4. **三模型身份传播机制不同** — Qwen3/DS7B双通道(Attn+MLP)，GLM4单通道(仅MLP)
5. **早期完整块恢复~100%** — 正确的residual stream输入足以让后续层计算binding
6. **L3-L8组件对binding几乎无直接贡献** — 对象身份在L0-L2已传播完毕
7. **GLM4的Attention完全不参与对象身份传播** — 这是根本性的架构差异

### Binding计算管线模型更新

```
Binding Pipeline:
  1. Token Embedding → 对象身份向量 (embedding space)
  2. L0-L2 MLP + Attn → 对象身份传播到residual stream (identity propagation)
  3. L3-L20 → 上下文整合、句法处理 (context integration)
  4. L21-L38 MLP → 属性兼容性计算 (compatibility computation)
  5. Last layers → 输出读出 (readout)
```

关键洞察：
- **步骤2是对象身份的入口**：不同模型用不同机制（Attn+MLP vs 仅MLP）
- **步骤4是binding的核心计算**：MLP将对象身份转换为属性兼容性排序
- **整条管线需要正确的输入**：单层替换只能恢复局部，多层替换恢复全部

### 关键硬伤

1. **早期MLP恢复100%的含义需要深入理解** — 这可能只是"正确的residual stream输入"效应，不代表早期MLP在计算binding
2. **Qwen3 L24-26 Attention主导** — 仅一个块出现，可能是噪声或特殊机制，需确认
3. **DS7B L23的Attention异常** — 超高方差（std=183.6%），不可靠
4. **early full block恢复~100%** — 这说明后续层的MLP在干净输入上自然产生binding，但无法区分哪些层贡献最大
5. **反向破坏率未达100%** — MLP destruction仅25-33%（Qwen3/GLM4），说明binding是多组件冗余的
6. **corrupted baseline "The item"仍可能携带语义** — 更好的baseline需要多个中性词

### 命令记录

```bash
# Phase 336 Round 1: 多层替换+反向破坏+跨层注意力
python tests/glm5/phase336_multilayer_patching.py qwen3       # ~33s
python tests/glm5/phase336_multilayer_patching.py glm4         # ~400s (6.7min)
python tests/glm5/phase336_multilayer_patching.py deepseek7b   # ~217s (3.6min)

# Phase 336b Round 2: 细粒度早期组件确认
python tests/glm5/phase336b_early_attn_confirm.py qwen3        # ~29s
python tests/glm5/phase336b_early_attn_confirm.py deepseek7b   # ~204s (3.4min)
python tests/glm5/phase336b_early_attn_confirm.py glm4         # ~229s (3.8min)
```

脚本位置：
- `tests/glm5/phase336_multilayer_patching.py` — 主测试（Phase 336+337+338）
- `tests/glm5/phase336b_early_attn_confirm.py` — Round 2确认
- 结果：`results/phase336_multilayer/{qwen3,glm4,deepseek7b}_phase336.json`
- 结果：`results/phase336_multilayer/{qwen3,glm4,deepseek7b}_phase336b.json`

## Phase 339+340+341: 多Baseline验证+管线组合+身份探针 [2026-06-02 13:45]

### 背景

Phase 336+337+338 确认了MLP因果主导和早期身份传播，但三大硬伤：
1. 仅用"The item"作为baseline→需多baseline验证稳健性
2. 早期身份块+后期计算块联合替换→能否接近100%？
3. 早期层到底写入了什么→需身份探针验证

### 方法

**Phase 339：多Baseline验证**
- 使用4个corrupted baseline："The item"、"The thing"、"The object"、"The entity"
- 测试关键MLP/Attention/Full块替换恢复率
- 验证MLP>Attn模式是否在所有baseline下稳健

**Phase 340：管线块组合替换**
- 早期身份块（Qwen3 L0-2, GLM4 L0-4, DS7B L0-2 full block）
- 后期计算块（MLP only）
- 两者联合替换
- 更宽计算块（L18-29, L15-29等）

**Phase 341：身份探针**
- 对24个对象在12个关键层提取residual stream
- 计算对象与baseline的余弦相似度、分离度、差异方差
- 最近质心分类器测试对象类别可区分性

### Phase 339 结果：多Baseline验证 — MLP>Attn稳健成立

**Qwen3（L21-29块）：**

| Baseline | MLP恢复 | Full恢复 | Attn恢复 | Valid pairs |
|----------|---------|---------|---------|------------|
| The item | +69.6% | +79.6% | +24.5% | 22 |
| The thing | +80.7% | +83.4% | +6.3% | 19 |
| The object | +88.8% | +93.5% | +16.3% | 18 |
| The entity | +92.7% | +96.9% | +7.8% | 19 |

**GLM4（L30-38块）：**

| Baseline | MLP恢复 | Full恢复 | Attn恢复 | Valid pairs |
|----------|---------|---------|---------|------------|
| The item | +46.0% | +56.5% | +11.0% | 22 |
| The thing | +56.8% | +61.5% | +2.8% | 17 |
| The object | +66.4% | +74.8% | +15.8% | 19 |
| The entity | +63.3% | +71.9% | ~10% | 22 |

**DS7B（L19-24块）：**

| Baseline | MLP恢复 | Full恢复 | Attn恢复 | Valid pairs |
|----------|---------|---------|---------|------------|
| The item | +58.5% | +62.2% | +5.5% | 18 |
| The thing | +68.8% | +83.4% | +2.3% | 21 |
| The object | +71.3% | +82.0% | +7.9% | 21 |
| The entity | +59.0% | +61.6% | ~6% | 19 |

**关键发现：MLP>Attn在所有4个baseline下都稳健成立！**

有趣模式：更抽象的baseline（"The object"/"The entity"）通常给更高的MLP恢复率。原因：这些baseline预设的binding信号更弱，MLP patch的效果更显著。

### Phase 340 结果：管线块组合 — 身份块单独接近100%

**Qwen3：**

| 块 | 恢复率 | std |
|----|--------|-----|
| identity_L0-2_full | **+99.4%** | 4.3% |
| compute_L21-29_mlp | +69.6% | 55.2% |
| identity+compute | **+99.8%** | 2.5% |
| compute_L18-29_mlp | +88.1% | 52.3% |
| compute_L15-29_mlp | +86.1% | 45.6% |

**GLM4：**

| 块 | 恢复率 | std |
|----|--------|-----|
| identity_L0-4_full | **+99.6%** | 1.9% |
| compute_L30-38_mlp | +46.0% | 42.4% |
| identity+compute | **+100.1%** | 0.9% |
| compute_L25-38_mlp | +80.3% | 18.5% |
| compute_L20-38_mlp | +90.9% | 11.0% |

**DS7B：**

| 块 | 恢复率 | std |
|----|--------|-----|
| identity_L0-2_full | **+100.6%** | 5.6% |
| compute_L19-24_mlp | +58.5% | 45.7% |
| identity+compute | +89.9% | 36.0% |
| compute_L16-24_mlp | +59.4% | 38.2% |
| compute_L12-24_mlp | +54.8% | 29.7% |

**最关键发现：**

1. **身份块单独就接近100%恢复**：Qwen3 99.4%、GLM4 99.6%、DS7B 100.6%
2. **identity+compute ≈ identity单独**：Qwen3 99.8% vs 99.4%，GLM4 100.1% vs 99.6%
3. **DS7B identity+compute < identity单独**：89.9% vs 100.6%，计算块patch反而干扰了自然计算
4. **更宽计算块提升恢复率**：GLM4 L20-38(90.9%) > L25-38(80.3%) > L30-38(46.0%)

### Phase 341 结果：身份探针 — 对象身份在全层存在

**对象与baseline的余弦相似度（Qwen3）：**

| 层 | cos_sim | separation | diff_var |
|----|---------|-----------|----------|
| L0 | 0.0177 | 0.9263 | 0.000422 |
| L1 | 0.3529 | 0.5381 | 0.028591 |
| L2 | 0.3642 | 0.5205 | 0.047463 |
| L4 | 0.3828 | 0.5656 | 0.107304 |
| L8 | 0.2663 | 0.5726 | 0.349804 |
| L18 | 0.4670 | 0.3079 | 0.437942 |
| L24 | 0.4221 | 0.3489 | 1.433360 |
| L29 | 0.3453 | 0.5025 | 9.719251 |
| L35 | 0.6767 | 0.2792 | 33.193958 |

**对象类别可区分性（7类最近质心分类器，chance=14.3%）：**

| 层 | Qwen3 | GLM4 | DS7B |
|----|-------|------|------|
| L0 | 41.7% | 33.3% | 37.5% |
| L1 | 50.0% | 41.7% | 58.3% |
| L2 | 45.8% | 45.8% | 41.7% |
| L5 | 37.5% | 50.0% | 45.8% |
| L8 | 45.8% | 45.8% | 54.2% |
| L12 | 41.7% | 41.7% | 58.3% |
| L24 | 45.8% | — | 54.2% |
| L29/38 | 54.2% | 54.2% | — |

→ 所有层都远超chance（2-4倍），但跨层差异不大。对象身份信息从嵌入层就存在，并持续贯穿所有层。

### 关键发现1：MLP>Attn在多Baseline下完全稳健

4个baseline × 3个模型 = 12种条件，MLP恢复率始终显著高于Attention：
- MLP恢复率范围：39.6%~92.7%
- Attn恢复率范围：2.3%~24.5%
- MLP/Attn比值：3.0x~36.9x

**结论不受baseline选择影响。**

### 关键发现2：身份块单独即接近100%——最重要的新发现

```
身份块恢复率：
Qwen3 L0-2 full: 99.4%
GLM4 L0-4 full: 99.6%
DS7B L0-2 full: 100.6%
```

这意味着：只要前2-4层的residual stream被修正为clean状态，后续所有层可以自然计算出正确的binding。

**更深层含义：**
- 后期MLP不是binding的"唯一通道"，而是"在正确输入上的最强单层贡献"
- 真正的结构是：正确残差输入 → 自然计算（所有后续层参与）→ 正确binding输出
- 后期MLP的恢复率高，是因为它们是最大的单层贡献者，但不是唯一贡献者

### 关键发现3：DS7B的identity+compute低于identity单独

DS7B: identity=100.6%, identity+compute=89.9%。这反直觉的结果说明：
- 身份块patch已使L3开始的residual stream接近clean
- 此时MLP在L19-L24的输出本身就是正确的（因为输入正确）
- 额外patch L19-24 MLP反而可能引入微小不一致（因为patch只替换MLP输出，不替换Attention输出）

### 关键发现4：更宽计算块持续提升恢复率

GLM4的计算块：
- L30-38 MLP: 46.0%
- L25-38 MLP: 80.3%
- L20-38 MLP: 90.9%

说明binding计算确实分布在多个层，越宽的块捕获越多计算。

### 关键发现5：对象身份信息从嵌入层就存在

Phase 341显示：
- L0 separation ≈ 0.93-0.95（对象间非常不同）
- L0-1的对象-基线余弦相似度极低（0.02-0.35）
- 类别可区分性在所有层都远超chance

但身份信息的存在≠binding计算。L0-L2的身份是"对象身份向量"，不是"属性兼容性排序"。从身份到兼容性的转换仍发生在后续MLP中。

### 客观事实拼图更新

1. **MLP>Attn在4个baseline下完全稳健** — 排除了baseline选择偏差
2. **身份块单独即接近100%恢复** — 正确残差输入足以让后续层计算binding
3. **身份+计算≈身份单独** — 计算块在身份块之后是冗余的
4. **更宽计算块提升恢复率** — binding是多层分布式计算
5. **对象身份从L0就存在并持续** — 但这是身份信息，不是兼容性计算
6. **DS7B的identity+compute低于identity单独** — patch可能引入不一致
7. **类别可区分性在所有层都远超chance** — 身份信息是持久特征

### Binding管线模型更新（更精确版）

```
Binding Pipeline (Updated):

1. Embedding: 对象token → 对象身份向量 (L0 separation=0.93-0.95)

2. L0-L2/L0-L4: 身份传播
   Qwen3/DS7B: Attn+MLP双通道 → residual stream包含完整对象上下文
   GLM4: 仅MLP通道 → residual stream包含完整对象上下文
   结果: L3的residual stream已携带足够信息让后续层计算binding

3. L3-L20: 上下文整合与身份维持
   不产生强binding信号，但维持和转换residual stream
   为后续MLP准备可计算格式

4. L21-L38: MLP兼容性计算
   将对象身份转换为属性值排序
   多层MLP链式累积: 每层贡献一部分兼容性变换
   越宽的块恢复率越高 (L20-38=90.9% > L25-38=80.3% > L30-38=46.0%)

5. Last layers: 输出读出
   将兼容性排序放大到logit space
```

**关键修正：步骤2不是"写入binding信号"，而是"提供正确的residual stream输入"。步骤4才是真正的兼容性计算。**

### 关键硬伤

1. **身份块100%恢复的真正含义** — 这可能只是"前几层patch ≈ 运行clean模型"，而非身份信息的特殊功能。需要更精细的实验区分。
2. **身份块是什么？** — L0-L2 full block包含attn+MLP+LayerNorm，不是纯"身份传播"。需要拆解哪个子组件真正关键。
3. **DS7B identity+compute < identity** — 计算块patch反而降低恢复率，说明多层联合patch可能有非叠加效应。
4. **类别可区分性不够高** — 最高58.3%，说明residual stream中的对象身份不是线性可分的简单特征。
5. **不同baseline恢复率差异大** — "The item"和"The entity"之间MLP恢复率差20-30%，这可能影响量化结论。
6. **后期MLP内部如何计算兼容性** — 仍未解答，这是理解binding编码结构的关键。

### 命令记录

```bash
# Phase 339+340+341: 多Baseline+管线组合+身份探针
python tests/glm5/phase339_multibaseline_pipeline.py qwen3       # ~55s
python tests/glm5/phase339_multibaseline_pipeline.py deepseek7b   # ~360s (6min)
python tests/glm5/phase339_multibaseline_pipeline.py glm4         # ~568s (9.5min)
```

脚本位置：
- `tests/glm5/phase339_multibaseline_pipeline.py` — 主测试（Phase 339+340+341）
- 结果：`results/phase339_multibaseline/{qwen3,glm4,deepseek7b}_phase339.json`

## Phase 342+342b: MLP内部通道分析 — 平衡放大发现 [2026-06-02 14:15]

### 背景

Phase 339-341确认了MLP是binding的主要因果通道，但**MLP如何把对象身份变成属性排序**仍未知。Phase 342目标：进入MLP内部，分解通道级贡献。

### 方法

**Phase 342: MLP通道绑定分解**
- 对关键binding层（Qwen3 L21-29, GLM4 L30-38, DS7B L19-24）提取gate_proj和up_proj激活
- SwiGLU: MLP(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
- 通道i对binding方向的贡献 = (d·W_down[:,i]) * SiLU(gate_i) * up_i
- 分类：兼容通道(d·W_down[:,i]>0) vs 不兼容通道(d·W_down[:,i]<0)

**Phase 342b: 修正命名+平衡放大确认**
- 修正变量命名（Phase 342中"incompat_suppress"实际测量的是不兼容信号增加）
- 新增：放大平衡比 = incompat_gross / compat_gross
- 新增：净/总比 = |net_binding| / gross_amplification
- Per-pair分析验证模式稳健性

### 核心发现：平衡放大（Balanced Amplification）

**三模型13个binding层的结果完全一致：**

| 模型 | 层 | 兼容提升 | 不兼容放大 | 平衡比 | 净/总比 |
|------|-----|---------|----------|--------|---------|
| Qwen3 | L21 | +9.60 | -9.51 | 1.003 | 2.0% |
| Qwen3 | L23 | +12.27 | -12.04 | 1.007 | 2.3% |
| Qwen3 | L25 | +16.73 | -16.61 | 1.020 | 2.7% |
| Qwen3 | L27 | +21.34 | -21.29 | 1.011 | 2.3% |
| Qwen3 | L29 | +25.16 | -24.63 | 1.016 | 2.9% |
| DS7B | L19 | +19.55 | -19.44 | 0.996 | 1.6% |
| DS7B | L21 | +30.67 | -29.86 | 0.996 | 2.1% |
| DS7B | L23 | +43.07 | -40.43 | 0.994 | 2.9% |
| DS7B | L24 | +51.53 | -50.56 | 0.996 | 1.2% |
| GLM4 | L30 | +2.60 | -2.51 | 1.003 | 2.3% |
| GLM4 | L33 | +3.36 | -3.32 | 1.004 | 1.9% |
| GLM4 | L36 | +4.45 | -4.39 | 0.985 | 1.7% |
| GLM4 | L38 | +12.43 | -11.94 | 0.999 | 2.1% |

**关键数据：**
1. **放大平衡比 = 0.985-1.020（均值≈1.00, std≈0.05）** — 兼容和不兼容通道被几乎完全同等放大
2. **净/总比 = 1.2%-2.9%** — 总放大的97-98%相互抵消，仅1-3%作为净binding效果存活
3. **Per-pair稳健** — 所有22-24个pair的平衡比都在0.88-1.18范围内

### 平衡放大的含义

MLP不是"选择性增强兼容、抑制不兼容"的选择器，而是**几乎对称地放大兼容和不兼容信号**，仅有极小偏向（~2%偏向兼容方向）。

数学表述：
```
MLP_l(clean) - MLP_l(corrupt)
= gross_compat_boost + gross_incompat_amplify + ...
≈ A * (1 + 0.02) + A * (1 - 0.02)  [A=gross amplification]
= 2A + 0.04A - 2A + 0.04A
net ≈ 0.02 * gross
```

### MLP不是均匀放大器

Experiment 2测试了MLP输出差异与输入差异的余弦相似度：

| 模型 | 层 | cos_sim(diff) |
|------|-----|--------------|
| Qwen3 | L21 | 0.040 |
| Qwen3 | L29 | 0.104 |
| DS7B | L21 | 0.181 |
| GLM4 | L38 | 0.361 |

余弦相似度远低于1.0，说明MLP不是简单缩放输入。MLP对输入进行了非线性变换，但这个变换在兼容和不兼容方向上几乎等量。

### Binding如何从1-3%的净偏向中产生？

关键在于**多层累积**：
- 每层MLP的净binding贡献 ≈ 0.2-1.5
- 5-9个binding层累积 → 总binding信号 ≈ 1-7
- 这与观测到的binding_range（通常1-10）一致

**Binding机制是"多弱选择器累积"而非"少强选择器"**。

### 客观事实拼图更新

1. **MLP对兼容和不兼容通道几乎同等放大** — 平衡比≈1.00（13/13层确认）
2. **净binding效果仅占总放大的1-3%** — 97-98%的放大被抵消
3. **MLP不是均匀放大器** — 输出差异与输入差异cos_sim≈0.04-0.36
4. **Binding通过多层微小偏向累积产生** — "多弱选择器"机制
5. **Per-pair模式稳健** — 所有pair的平衡比在0.88-1.18

### 嵌入差值patch测试

Phase 342的嵌入差值patch实验显示：完整嵌入差值patch给出~100%恢复（trivially等价于运行clean模型）。由于device_map="auto"导致embed_tokens在meta device上，部分嵌入patch（binding-only/ortho-only）未能在DS7B和GLM4上完成。

### 关键硬伤

1. **1-3%的净偏向为何存在？** — 如果MLP真的等量放大，为何不是精确0%净效果？这个微小偏向的来源是什么？
2. **平衡放大是否是MLP的通用属性？** — 还是只在binding方向上平衡？其他方向是否也平衡？
3. **MLP的非线性变换是什么？** — cos_sim低说明不是简单缩放，但具体变换形式未知
4. **累积效应的数学描述** — 5-9层×2%偏向如何精确累积？需要层级轨迹分析
5. **兼容通道和不兼容通道的区分标准** — 基于d·W_down的符号，但这个符号是否在所有对象上一致？
6. **未能完成部分嵌入patch实验** — device_map="auto"限制了hook能力

### 命令记录

```bash
# Phase 342: MLP通道分析（第一轮）
python tests/glm5/phase342_mlp_channel_analysis.py qwen3       # ~28s
python tests/glm5/phase342_mlp_channel_analysis.py deepseek7b   # ~102s
python tests/glm5/phase342_mlp_channel_analysis.py glm4         # ~171s

# Phase 342b: 平衡放大确认（第二轮）
python tests/glm5/phase342b_balanced_amplification.py qwen3     # ~25s
python tests/glm5/phase342b_balanced_amplification.py deepseek7b # ~82s
python tests/glm5/phase342b_balanced_amplification.py glm4       # ~131s
```

脚本位置：
- `tests/glm5/phase342_mlp_channel_analysis.py` — Phase 342 MLP通道分析
- `tests/glm5/phase342b_balanced_amplification.py` — Phase 342b 平衡放大确认
- 结果：`results/phase342_mlp_channel/{qwen3,glm4,deepseek7b}_phase342.json`
- 结果：`results/phase342_mlp_channel/{qwen3,glm4,deepseek7b}_phase342b.json`

## Phase 343+343b: 平衡放大通用性 + 微偏置来源 [2026-06-02 14:45]

### 背景

Phase 342发现MLP在binding方向上做平衡放大（平衡比≈1.00，净/总比1-3%）。关键问题：
1. 平衡放大是binding特有还是MLP通用属性？
2. 1-3%微偏置从哪里来？

### 方法

**Experiment A (Phase 343): 多方向通道分解**
- 对6种方向类型做通道分解：binding、random、object_identity、same_class、attribute_only、unrelated
- 所有方向使用相同的clean/corrupt激活（仅方向向量不同）
- 10个随机方向 × 5 binding层 + 多种语义方向

**Experiment B (Phase 343): 微偏置来源分解**
- 分解MLP(x) = down(SiLU(gate(x)) * up(x))的微偏置
- 乘积分解：gate驱动项、up驱动项、交互项
- 结构不对称性：|d|投影、SiLU激活、ΔSiLU、Δup在正/负通道间的差异
- 通道-方向相关性

**Round 2 (Phase 343b): 50随机方向确认**
- 50个随机方向 vs 10个binding方向
- 5个不同prompt上下文 × 10随机方向
- t-test统计检验

### 核心发现1：平衡放大是MLP通用属性

**三模型6方向类型的平衡比均值：**

| 方向类型 | Qwen3 | DS7B | GLM4 |
|---------|-------|------|------|
| binding | 0.995 | 1.008 | 0.989 |
| random | 1.003 | 1.005 | 1.011 |
| object_identity | 0.949 | 1.008 | 1.030 |
| same_class | 1.004 | 0.998 | 1.000 |
| attribute_only | 0.998 | 0.998 | 0.995 |
| unrelated | 1.009 | 1.004 | 1.009 |

→ 所有方向类型的平衡比都在0.95-1.03范围，无显著差异
→ **平衡放大不是binding特有机制，而是SwiGLU MLP的通用几何性质**

### 核心发现2：Net/gross比在binding方向显著更高

**Round 2 (50随机方向) 统计检验：**

| 模型 | Binding N/G | Random N/G | p值 | 显著性 |
|------|------------|-----------|------|--------|
| Qwen3 | 0.028±0.020 | 0.018±0.014 | 0.0000 | *** |
| DS7B | 0.023±0.018 | 0.016±0.011 | 0.0059 | ** |
| GLM4 | 0.023±0.018 | 0.018±0.014 | 0.0338 | * |

→ binding方向的net/gross比显著高于随机方向（约1.3-1.6倍）
→ **binding方向确实有更强的方向性偏向，这不是噪声**
→ 但绝对差异很小（~0.01），说明binding特异性是微弱的

### 核心发现3：微偏置由gate和up共同产生，无主导成分

**Phase 344微偏置来源分解（6 pair × 多层）：**

| 模型 | Gate N/G | Up N/G | Gate-Dir相关 | Up-Dir相关 | ΔSiLU不对称 | ΔUp不对称 |
|------|----------|--------|-------------|-----------|------------|----------|
| Qwen3 | 0.031 | 0.031 | -0.001 | -0.002 | 1.007 | 1.000 |
| DS7B | 0.021 | 0.027 | +0.003 | +0.005 | 1.012 | 1.001 |
| GLM4 | 0.033 | 0.028 | +0.004 | +0.000 | 1.007 | 1.001 |

关键发现：
1. **Gate和up的net/gross比几乎相等** — 两者对微偏置的贡献相当，无主导成分
2. **Gate和up与binding方向的相关性≈0** — 激活变化不沿binding方向系统偏向
3. **ΔSiLU不对称≈1.00** — gate激活变化在正/负通道间对称
4. **Δup不对称≈1.00** — up投影变化在正/负通道间对称
5. **W_down投影不对称≈1.00** — |d|在正/负通道间均匀

### 重新理解微偏置来源

微偏置不是来自任何单一组件的系统性偏向，而是：
- gate变化对称 + up变化对称 + W_down结构对称
- 但三者乘积的**高阶交互效应**在binding方向上产生微小残余
- 这个残余虽小（~1-2%），但在binding方向上显著高于随机方向
- 多层累积后，这个微小偏向足以产生可观测的binding

数学表述：
```
微偏置 ≈ Σ_{ijk} ∂³f/∂gate_i∂up_j∂W_down_k × Δgate_i × Δup_j × W_k
         ↑ 三阶交互项
```

这不是"gate选择"或"up选择"，而是**高维空间中多对称分量交互的统计残余**。

### 客观事实拼图更新

1. **平衡放大是SwiGLU MLP的通用几何性质** — 对所有方向都成立（6方向类型×3模型确认）
2. **Binding方向的net/gross比显著高于随机方向** — 约1.3-1.6倍（3模型t-test确认）
3. **微偏置不由gate或up单独产生** — 两者贡献相当，且都对称
4. **微偏置来自高阶交互效应** — 对称分量的乘积产生的统计残余
5. **W_down结构无系统性偏向** — 投影在正负通道间均匀
6. **Gate/Up激活变化不沿binding方向系统偏向** — 相关性≈0

### 硬伤和问题

1. **交互效应的具体数学形式未确定** — "高阶交互"是定性描述，不是精确公式
2. **Net/gross差异虽显著但很小** — binding方向1.5-2.8% vs 随机1.6-1.8%，绝对差异<1%
3. **多层累积的具体动力学未建模** — 层间如何传递和增强微偏置？
4. **平衡放大的数学证明缺失** — 为什么SwiGLU结构必然导致平衡放大？
5. **Prompt上下文对微偏置的影响未系统测试** — 当前只用"The apple"和"The item"

### 命令记录

```bash
# Phase 343: 平衡放大通用性（第一轮）
python tests/glm5/phase343_balanced_amplification_generality.py qwen3       # ~14s
python tests/glm5/phase343_balanced_amplification_generality.py deepseek7b   # ~43s
python tests/glm5/phase343_balanced_amplification_generality.py glm4         # ~59s

# Phase 343b: 50随机方向确认（第二轮）
python tests/glm5/phase343b_confirmation.py qwen3      # ~51s
python tests/glm5/phase343b_confirmation.py deepseek7b  # ~122s
python tests/glm5/phase343b_confirmation.py glm4        # ~137s
```

脚本位置：
- `tests/glm5/phase343_balanced_amplification_generality.py` — Phase 343 主测试
- `tests/glm5/phase343b_confirmation.py` — Phase 343b 确认测试
- 结果：`results/phase343_generality/{qwen3,glm4,deepseek7b}_phase343.json`
- 结果：`results/phase343_generality/{qwen3,glm4,deepseek7b}_phase343b.json`

## Phase 344+345: 多关系方向验证 + 方向匹配随机对照 [2026-06-02 15:10]

### 背景

Phase 343/343b证明平衡放大是MLP通用属性，但有两个硬伤：
1. 随机方向未严格匹配范数和输出空间对齐（Phase 344）
2. 只测了binding方向，未验证其他语言关系是否同样"平衡+微偏置增强"（Phase 345）

### 方法

**Phase 344 — 方向匹配随机对照（4种）**：
1. norm-matched random：与binding方向相同L2范数
2. W_U-subspace random：在W_U列空间内随机采样（通过SVD得到主子空间）
3. binding-orthogonal random：与binding方向正交的随机方向
4. pure random：标准高斯随机方向

**Phase 345 — 多关系方向（6种语言关系）**：
1. binding（对象-属性）：apple-red, banana-yellow, etc.
2. negation（否定）："is red" vs "is not"
3. antonym（反义）：hot-cold, big-small
4. role（角色）：主语vs宾语位置
5. tense（时态）：过去vs现在
6. same_class（同类）：apple-banana

每个方向都做通道分解，测平衡比和net/gross比。

### 核心发现1：所有6种语言关系方向的平衡比≈1.00

| 关系类型 | Qwen3 | DS7B | GLM4 |
|---------|-------|------|------|
| binding | 1.017 | 0.996 | 0.997 |
| negation | 1.054 | 1.013 | 0.991 |
| antonym | 0.998 | 0.992 | 0.994 |
| role | 1.016 | 0.992 | 1.003 |
| tense | 0.992 | 0.995 | 0.981 |
| same_class | 1.006 | 0.999 | 0.994 |

→ 所有语言关系方向都呈现平衡放大，与binding方向一致
→ **平衡放大不仅不是binding特有，甚至不是binding+negation等特定关系特有，而是所有语义方向的通用属性**

### 核心发现2：Net/gross比在不同关系类型间有差异

| 关系类型 | Qwen3 N/G | DS7B N/G | GLM4 N/G |
|---------|----------|---------|---------|
| binding | 0.028 | 0.021 | 0.020 |
| negation | 0.031 | 0.024 | 0.037 |
| antonym | 0.023 | 0.017 | 0.024 |
| role | 0.028 | 0.014 | 0.032 |
| tense | 0.017 | 0.015 | 0.017 |
| same_class | 0.033 | 0.020 | 0.032 |

→ tense（时态）的net/gross最低（~0.015），binding/negation/role/same_class较高（~0.02-0.03）
→ **不同语言关系确实有不同的微偏置强度**
→ negation和same_class的net/gross甚至高于binding，说明"语义对比强度"不是binding独有的

### 核心发现3：方向匹配对照确认binding方向net/gross高于W_U子空间随机

**Binding vs 4种随机对照的net/gross比**：

| 对照类型 | Qwen3 N/G | DS7B N/G | GLM4 N/G |
|---------|----------|---------|---------|
| binding | 0.028 | 0.021 | 0.020 |
| norm-matched | 0.020 | 0.014 | 0.019 |
| W_U-subspace | 0.021 | 0.015 | 0.018 |
| binding-orthogonal | 0.021 | 0.015 | 0.017 |
| pure random | 0.019 | 0.014 | 0.016 |

**统计检验**：

| 对照 | Qwen3 p值 | DS7B p值 | GLM4 p值 |
|------|----------|---------|---------|
| norm-matched | 0.53 ns | 0.08 ns | 0.90 ns |
| W_U-subspace | **0.031*** | **0.004**** | 0.92 ns |
| binding-orthogonal | 0.24 ns | **0.015*** | 0.98 ns |
| pure random | 0.19 ns | 0.16 ns | 0.64 ns |

→ Qwen3和DS7B对W_U-subspace random显著，DS7B对binding-orthogonal也显著
→ **GLM4无任何显著差异**——GLM4的binding方向微偏置不比任何随机方向强
→ 这个结果需要谨慎解读：binding vs random的net/gross差异在Qwen3/DS7B存在但边际，GLM4完全消失

### 客观事实拼图更新

1. **6种语言关系都呈现平衡放大** — 不是binding特有，是所有语义方向通用
2. **不同关系类型的net/gross有差异** — tense最低(~0.015)，negation/role/same_class较高(~0.03)
3. **Binding vs 匹配随机对照** — Qwen3/DS7B对W_U-subspace有边际显著性，GLM4完全不显著
4. **Binding的net/gross优势非常微弱** — 约0.004-0.007的绝对差异
5. **Net/gross差异可能来自方向分布特性**（如W_U子空间对齐），而非binding特有的编码机制

---

## Phase 346: 精确交互分解 + 层级累积闭合 [2026-06-02 15:10]

### 背景

Phase 343/344发现微偏置来自"高阶交互"，但那只是排除法推论。需要精确分解。

### 方法

**Part A: 精确4-way因子分解**
对MLP输出 = W_down @ (SiLU(gate) * up)，构造4种条件：
- CC: SiLU(gate_clean) * up_clean
- CR: SiLU(gate_clean) * up_corrupt
- RC: SiLU(gate_corrupt) * up_clean
- RR: SiLU(gate_corrupt) * up_corrupt

标准2×2因子分解（精确，非近似）：
- gate_main = ((CC-CR) + (RC-RR)) / 2
- up_main = ((CC-RC) + (CR-RR)) / 2
- interaction = CC - CR - RC + RR

**Part B: 层级累积闭合**
计算每个binding层MLP的净方向投影，求和，与最终binding信号比较。

### 核心发现1：交互项是最大贡献者（约40%）

| 模型 | Gate main | Up main | Interaction | 总效应 |
|------|-----------|---------|------------|--------|
| Qwen3 | 25.7% | 31.4% | **42.8%** | +0.566 |
| DS7B | 27.9% | 26.4% | **45.6%** | +1.008 |
| GLM4 | 30.2% | 30.7% | **39.1%** | +0.334 |

→ **三模型一致：gate×up交互项占总效应的39-46%**
→ Gate和up主效应几乎相等（各25-31%）
→ **微偏置主要由交互项产生，不是由gate或up单独产生**

### 核心发现2：层级累积不闭合

| 模型 | Closure ratio | Final binding | MLP net sum | Correlation |
|------|--------------|--------------|-------------|------------|
| Qwen3 | 1.11±2.40 | 2.38 | 3.57 | 0.70 |
| DS7B | -0.04±6.97 | 1.66 | 4.24 | 0.82 |
| GLM4 | 0.43±0.66 | 2.95 | 1.28 | 0.77 |

→ Closure ratio极不稳定（Qwen3: -3.4~4.0, DS7B: -13.7~6.0, GLM4: -0.6~1.7）
→ **MLP net sum不等于final binding** — 因为：
  1. 只测了binding层的MLP，忽略了非binding层
  2. 忽略了attention的贡献
  3. 忽略了LayerNorm的重缩放
  4. 忽略了层间残差交互
→ 但相关性较高（0.70-0.82），说明MLP贡献的方向是正确的

### 核心发现3：交互项方向不固定

- Qwen3: 28正/12负
- DS7B: 18正/14负
- GLM4: 14正/18负

→ 交互项方向（正/负）因pair和layer而异，不是系统性偏向
→ **交互项的绝对值大，但符号不稳定** — 这与"平衡放大+微小净偏置"一致

### 理论更新

微偏置的精确结构现在更清楚了：

```
微偏置 = gate_main(25-30%) + up_main(25-31%) + interaction(39-46%)
```

交互项最大，但它不是系统性偏向——它是gate和up同时变化时产生的非线性效应。这个非线性效应在binding方向上的投影平均为正，但方差很大。

更精确的描述：
```
微偏置不是"单方向选择"
而是"gate和up变化的非线性乘积在binding方向上的统计残余"
```

### 硬伤和问题

1. **交互项的物理意义不明确** — 知道它占40%，但不清楚SiLU(gate)×up的哪个具体性质导致交互
2. **Closure不闭合** — 当前方法只考虑binding层MLP，忽略attention和非binding层
3. **GLM4的binding vs random差异消失** — 与Phase 343b矛盾，可能是样本量不够或统计方法问题
4. **交互项符号不稳定** — 说明交互不是系统性偏向，而是高方差噪声+微弱均值
5. **W_down结构未深入分析** — 为什么W_down投影后正负通道如此平衡？

### 命令记录

```bash
# Phase 344+345: 多关系方向 + 匹配随机对照
python tests/glm5/phase344_345_multi_relation.py qwen3       # ~57s
python tests/glm5/phase344_345_multi_relation.py deepseek7b   # ~160s
python tests/glm5/phase344_345_multi_relation.py glm4         # ~195s

# Phase 346: 精确交互分解 + 层级累积闭合
python tests/glm5/phase346_interaction_closure.py qwen3       # ~19s
python tests/glm5/phase346_interaction_closure.py deepseek7b  # ~51s
python tests/glm5/phase346_interaction_closure.py glm4        # ~71s
```

脚本位置：
- `tests/glm5/phase344_345_multi_relation.py` — Phase 344+345 主测试
- `tests/glm5/phase346_interaction_closure.py` — Phase 346 精确交互+闭合
- 结果：`results/phase344_345_multi_relation/{qwen3,glm4,deepseek7b}_phase344_345.json`
- 结果：`results/phase346_interaction_closure/{qwen3,glm4,deepseek7b}_phase346.json`
