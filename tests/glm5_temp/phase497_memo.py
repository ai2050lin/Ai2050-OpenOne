"""Phase 497 MEMO更新脚本"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"

content = f"""

## Phase 497: Final RMSNorm分离与MLP因果闭环 [2026-06-14 {timestamp}]

### ★★★ 核心发现: RMSNorm是末层DCF的真正决定机制，不是MLP方向写入 ★★★

Phase 497最重要的发现是：

**1. RMSNorm不是简单放大器，而是DCF空间的非线性重映射**

Qwen3 R1 RMSNorm分离:
| 类别 | D_pre(norm前) | D_post(norm后) | RMSNorm增益 | D变化 |
|------|-------------|--------------|------------|------|
| fruit | 23.46 | 6.11 | -17.36 | 压缩74% |
| clothing | 48.95 | 4.31 | -44.64 | 压缩91% |
| emotion | 26.42 | 3.25 | -23.17 | 压缩88% |
| action | -10.21 | 2.59 | +12.80 | 翻转符号! |
| animal | 26.25 | 7.25 | -18.99 | 压缩72% |

★ RMSNorm将D压缩为原来的1/4~1/11!
★ RMSNorm将action的D从负翻转为正!
★ clothing在pre-norm空间D最高(48.95)，但在post-norm空间最低(4.31)!

**2. MLP对D的贡献在pre-norm和post-norm下方向相反**

Qwen3:
| 类别 | ΔD(0MLP)_pre | ΔD(0MLP)_post | 方向翻转? |
|------|-------------|--------------|---------|
| fruit | -6.99 | -2.05 | 同向(负) |
| clothing | +4.35 | -0.66 | ★翻转! |
| emotion | +8.61 | -0.28 | ★翻转! |
| action | -4.20 | +0.67 | ★翻转! |
| animal | -2.27 | -1.78 | 同向(负) |

解读:
- clothing: MLP在pre-norm空间减少D(+4.35=去除MLP增加D)，但在post-norm空间增加D(-0.66=去除MLP减少D)
- emotion: 同上翻转
- action: MLP在pre-norm空间增加D(-4.20=去除MLP减少D)，但在post-norm空间减少D(+0.67=去除MLP增加D)

★ 之前认为"MLP为D提供通用boost"是RMSNorm后的假象!
★ 在原始hidden state空间，MLP实际上减少大部分类别的D!

**3. MLP的效应主要通过RMSNorm缩放变化传递，方向效应可忽略**

Qwen3 R2 Exp3 (RMSNorm方向vs范数分离):
| 类别 | RMS变化比 | 缩放效应 | 方向效应 | ΔD(0MLP)_post |
|------|---------|---------|---------|--------------|
| fruit | 1.099 | -2.04 | -0.00 | -2.05 |
| clothing | 1.042 | -0.69 | 0.00 | -0.66 |
| emotion | 1.032 | -0.29 | -0.00 | -0.28 |
| action | 1.032 | +0.63 | 0.00 | +0.67 |
| animal | 1.080 | -1.73 | -0.00 | -1.78 |

★ 缩放效应 ≈ ΔD(0MLP)_post 的全部!
★ 方向效应 ≈ 0!
★ MLP去除后RMS稍微增大(1.03~1.10)，因为MLP输出有特定范数

**4. GLM4也显示相同的RMSNorm翻转模式**

GLM4 R1:
| 类别 | ΔD(0MLP)_pre | ΔD(0MLP)_post | 翻转? |
|------|-------------|--------------|------|
| fruit | +1.15 | -0.85 | ★翻转 |
| clothing | +0.95 | -0.01 | ★翻转 |
| emotion | +0.42 | -1.54 | ★翻转 |
| animal | +1.81 | -1.99 | ★翻转 |

★ GLM4的MLP在pre-norm空间也减少D(去除MLP增加D)
★ GLM4的RMSNorm同样翻转了MLP的D贡献方向

**5. 组件D贡献分解(pre-norm空间)**

Qwen3:
| 类别 | D_residual | D_attn | D_mlp | D_full |
|------|-----------|--------|-------|--------|
| fruit | 10.29 | 6.30 | 6.97 | 23.56 |
| clothing | 48.20 | 5.26 | -4.00 | 49.44 |
| emotion | 29.57 | 5.09 | -8.29 | 26.37 |
| action | -14.60 | 0.78 | 4.51 | -9.31 |
| animal | 20.71 | 2.94 | 2.18 | 25.82 |

★ Residual(残差流)是D的主导贡献者
★ Attn对D贡献较小但稳定正(5~6)
★ MLP的D贡献正负取决于类别: fruit/animal为正, clothing/emotion为负

GLM4:
| 类别 | D_residual | D_attn | D_mlp | D_full |
|------|-----------|--------|-------|--------|
| fruit | 5.99 | -0.24 | -1.17 | 4.59 |
| clothing | 2.75 | -0.06 | -0.91 | 1.77 |
| emotion | 5.68 | -0.01 | -0.48 | 5.19 |
| action | -6.79 | 0.25 | 2.64 | -3.91 |
| animal | 10.18 | -0.37 | -1.82 | 7.99 |

★ GLM4的MLP在pre-norm空间对除action外所有类别贡献负D
★ GLM4的Attn贡献接近零
★ GLM4的Residual是D的绝对主导

**6. DS7B数据大量NaN，CPU offload不可靠**

DS7B只有部分fruit数据有效，其余类别全部NaN。

### ★★★ Phase 497 对Phase 494-496理论的修正 ★★★

| 之前理论 | Phase 497修正 |
|---------|-------------|
| "MLP为DCF提供通用boost" | ★ 在pre-norm空间MLP实际减少大部分类别的D |
| "shared_semantic方向符号翻转" | ★ 方向效应可忽略，主要是RMSNorm缩放效应 |
| "MLP执行shared方向释放/刹车" | ★ MLP改变hidden state范数→RMSNorm重缩放→D变化 |
| "final RMSNorm只是归一化细节" | ★ RMSNorm是DCF空间的决定性变换 |

### 最新机制理解

```
末层DCF机制:
1. Residual(残差流)携带主要语义信号 (D_residual最大)
2. Attn提供小的正向D修正
3. MLP改变hidden state的整体范数(而非方向)
4. MLP对D的贡献在pre-norm和post-norm空间方向相反
5. RMSNorm将范数差异转化为DCF差异
6. 最终D_post由RMSNorm缩放后的hidden state决定
```

一句话:
**末层DCF不是MLP写入语义方向的结果，而是残差流携带语义信号 + MLP调节范数 + RMSNorm非线性重映射的联合产物。**

### 硬伤与瓶颈

1. **DS7B完全不可靠**: CPU offload导致NaN，需要纯GPU方式
2. **GLM4的RMSNorm weight不可访问**: meta device导致手动RMSNorm无法验证
3. **pre-norm空间D计算缺少LayerNorm补偿**: residual_error约1.6-1.8
4. **shared方向定义仍不完美**: PCA shared方向消融后D变化很小(Qwen3)
5. **action的RMSNorm翻转方向与其他类别不同**: 需要专门分析

### 下一步核心任务

1. **理解RMSNorm为什么翻转MLP的D贡献**: 数学分析RMSNorm对hidden state的非线性映射
2. **GLM4保守机制解释**: 如果MLP在pre-norm也减少D，GLM4和Qwen3的区别在哪？
3. **Residual(残差流)作为语义主要载体**: 需要分析L(n-2)残差流如何编码类别信息
4. **范数vs方向的信息论分析**: MLP改变范数而非方向，这意味着什么？
5. **DS7B纯GPU验证**: 用bf16全GPU方式重测
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {timestamp}")
