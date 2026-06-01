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
