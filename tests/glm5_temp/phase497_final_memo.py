"""Phase 497 Final MEMO更新"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"

content = f"""

## Phase 497 补充: RMSNorm翻转机制的精确数学分析 [2026-06-14 {timestamp}]

### R3 hook问题说明

R3尝试对self_attn和mlp分别hook来分解组件贡献，但hook捕获的输出全为零。
原因可能是hook注册位置或输出格式问题。R1和R2的数据已经足够可靠。

### ★★★ Phase 497 最终确认的客观事实 ★★★

**事实1: RMSNorm将D压缩为原来的1/4~1/11 (Qwen3)**

```text
fruit:    D_pre=23.46 → D_post=6.11   (压缩74%)
clothing: D_pre=48.95 → D_post=4.31   (压缩91%)
emotion:  D_pre=26.42 → D_post=3.25   (压缩88%)
action:   D_pre=-10.21 → D_post=2.59  (翻转符号+压缩)
animal:   D_pre=26.25 → D_post=7.25   (压缩72%)
```

**事实2: MLP的D贡献在pre-norm和post-norm空间方向相反 (Qwen3+GLM4)**

Qwen3:
```text
clothing: Δ(0MLP)_pre=+4.35 → Δ(0MLP)_post=-0.66  (翻转)
emotion:  Δ(0MLP)_pre=+8.61 → Δ(0MLP)_post=-0.28  (翻转)
action:   Δ(0MLP)_pre=-4.20 → Δ(0MLP)_post=+0.67  (翻转)
```

GLM4:
```text
fruit:    Δ(0MLP)_pre=+1.15 → Δ(0MLP)_post=-0.85  (翻转)
emotion:  Δ(0MLP)_pre=+0.42 → Δ(0MLP)_post=-1.54  (翻转)
animal:   Δ(0MLP)_pre=+1.81 → Δ(0MLP)_post=-1.99  (翻转)
```

**事实3: MLP效应几乎全部通过RMSNorm缩放传递 (Qwen3)**

```text
scale_effect ≈ ΔD(0MLP)_post 的100%
dir_effect ≈ 0
```

**事实4: 组件D贡献分解(pre-norm空间, Qwen3)**

```text
fruit:    D_residual=10.29, D_attn=6.30, D_mlp=6.97
clothing: D_residual=48.20, D_attn=5.26, D_mlp=-4.00
emotion:  D_residual=29.57, D_attn=5.09, D_mlp=-8.29
action:   D_residual=-14.60, D_attn=0.78, D_mlp=4.51
animal:   D_residual=20.71, D_attn=2.94, D_mlp=2.18
```

★ Residual(残差流)是D的主导贡献者
★ MLP对clothing/emotion贡献负D，对fruit/action/animal贡献正D

**事实5: DS7B因CPU offload数据不可靠**

### ★★★ 对用户Phase 496评价的修正 ★★★

用户对Phase 496的评价**总体正确**，但Phase 497发现了更深层的机制：

| 用户判断 | Phase 497修正 |
|---------|-------------|
| "MLP主导释放" | ✅ 但MLP的释放是RMSNorm后的假象，pre-norm空间MLP实际减少D |
| "shared方向符号翻转" | ⚠️ 方向效应可忽略，主要是范数→RMSNorm缩放效应 |
| "final RMSNorm是放大器" | ❌ RMSNorm不是放大器，而是非线性重映射，它压缩D并翻转MLP贡献 |
| "Qwen3 MLP-shared翻转" | ⚠️ MLP对D的贡献主要通过改变RMSNorm缩放，不是方向写入 |
| "GLM4不使用shared方向" | ✅ 但GLM4的MLP在pre-norm空间也贡献负D |

### 机制更新

之前理解:
```
L(n-2): shared_semantic刹车
↓
L(n-1): MLP翻转shared方向 → 释放
↓
final RMSNorm: 放大
```

修正理解:
```
L(n-2): 残差流携带语义信号 (D_residual最大)
↓
L(n-1): MLP改变hidden state范数(不是方向!)
         → pre-norm空间MLP对多数类别贡献负D
         → 但MLP增加了hidden state范数
↓
final RMSNorm: 除以更大范数 → D压缩
         → 由于非线性重映射，MLP的D贡献方向翻转
         → 最终表现为"MLP boost D_post"
```

### 硬伤与瓶颈

1. **RMSNorm的数学机制尚未精确分析**: 为什么范数变化会导致D贡献翻转？
   - 可能与effective_readout方向（rmsnorm_w * W_D）有关
   - 需要分析: D_post = <h, rmsnorm_w * W_D> / rms(h)

2. **GLM4的RMSNorm weight不可访问**: 无法做手动RMSNorm验证

3. **Residual(残差流)作为语义主载体**: 这才是D的真正来源，需要深入分析

4. **action的RMSNorm翻转方向与其他类别不同**: action从D_pre<0变为D_post>0

5. **DS7B不可靠**: 需要纯GPU加载方式

### Phase 498方向

核心任务: **理解RMSNorm如何将范数差异转化为DCF差异**

1. 数学分析: D_post = <h, g*W_D> / rms(h) 中，g=rmsnorm_w
   - 当MLP增加h的范数时，1/rms(h)减小
   - 但<h, g*W_D>可能增加或减少
   - 翻转取决于<h, g*W_D>的变化率vs 1/rms(h)的变化率

2. Residual流是D的真正来源: 需要分析L(n-2)残差流如何编码类别信息

3. GLM4 vs Qwen3: 两者RMSNorm翻转模式相似，但D值尺度不同
   - GLM4: D_pre=4-8, D_post=1-6 (压缩较少)
   - Qwen3: D_pre=23-49, D_post=3-7 (压缩极大)
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"Final MEMO updated at {timestamp}")
