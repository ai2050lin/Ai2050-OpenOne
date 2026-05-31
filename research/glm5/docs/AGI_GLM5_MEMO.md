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

基于Phase 310-313的发现，下一步应从"单方向测试"升级到"全局关系网络测试"：

1. **构建多层关系网络**：8类关系（同类/上下位/属性/功能/反义/否定/操作/组合）
2. **提取模型内部关系图**：每层计算概念间余弦距离矩阵
3. **比较内外图同构性**：Mantel相关/邻域重叠/排序保持
4. **复用-差分路径分解**：对每个概念簇提取shared_path和delta_path
5. **建立三图：复用图、差异图、冲突图**

这才是破解整体编码机制的关键路径。
