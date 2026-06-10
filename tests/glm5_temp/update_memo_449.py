"""Update MEMO with Phase 449 R1 results"""
import time

now = time.strftime('%Y-%m-%d %H:%M')
header = f"\n\n## Phase 449: 对象解锁门控 + Shared/Private因果验证 [2026-06-10 12:48]"

content = f"""

### 核心实验

1. **Exp1: 对象解锁门控验证** — 6类模板精细控制(T0无对象/T1有对象/T2重复/T3冲突/T4冲突对象近/T5替换)
2. **Exp2: MLP内部组件消融** — gate/up/down分别消融(结果有bug,gate/up/down返回相同值)
3. **Exp3: Shared/Private因果注入** — 分解delta为shared+private,分别注入测因果

### Exp1 关键发现: 对象解锁机制三模型对比

| 指标 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| avg_unlock | +2.39 | +3.31 | +2.03 |
| avg_conflict_recovery | +0.45 | +0.34 | **-1.66** |
| avg_repeat_ctrl | -0.03 | -0.52 | -0.82 |
| avg_replace_ctrl | -1.03 | **-2.31** | -0.25 |

**GLM4的对象替换控制最负(-2.31)** — 当冲突句中读出对象被替换为something,属性logit大幅下降,证明GLM4的属性释放**强依赖具体对象身份**。

**DS7B的冲突恢复为负(-1.66)** — 冲突模板导致属性logit下降,对象知识无法恢复。

**Qwen3冲突恢复为正(+0.45)** — 中等恢复力。

**GLM4的类别依赖**: fruit对象冲突恢复+2.03,animal/tool为负(-1.20/-0.19) — **类别依赖的冲突恢复**。

### Exp3 关键发现: Shared/Private因果验证(最关键结果)

**这是首次对shared/private分解做因果验证,而非统计观察。**

| 模型 | Private增长倍数 | Shared衰减倍数 | 最后一层private/cat比 |
|------|----------------|---------------|---------------------|
| Qwen3 | **10.5x** | 0.03x | **0.808** |
| GLM4 | **3.1x** | **0.99x** | 0.037 |
| DS7B | **22.6x** | 0.45x | **0.917** |

详细因果效应:

**Qwen3**: shared→cat从L0=+1.99衰减到L35=+0.05, private→cat从L0=+0.02增长到L35=+0.22
- 模式: 共享入口(早层)→逐步私有化(晚层),private最终占比0.808

**GLM4**: shared→cat从L0=+2.54到L39=+2.51几乎不变!, private→cat始终很小(0.03~0.26)
- **GLM4的shared分量几乎不衰减(0.99x)**,private只增长3倍
- L26出现shared→cat负值(-1.05),说明中层shared分量有反转
- **这解释了GLM4为何保持高shared_ratio: shared通道始终畅通**

**DS7B**: shared→cat先增后减(L0=-1.15→L18=+2.67→L27=+0.51), private→cat剧增(L0=-0.25→L27=+5.65)
- **Private增长22.6倍,最后一层private完全主导(0.917)**
- 这解释了DS7B深层极端私有化

### 因果验证结论

1. **"共享入口+深层私有化"是真实因果结构,不是统计假象**
2. **GLM4的shared通道几乎不衰减,这是它高shared_ratio的根本原因**
3. **DS7B的private增长22.6倍,深层完全由private主导**
4. **Qwen3居中,private增长10倍,shared衰减97%**
5. **GLM4 L26的shared负效应可能是一种"类别抑制门控"机制**

### Exp2 问题

MLP内部组件(gate/up/down)消融的hook实现有bug,三组件返回相同值。原因:
- register_forward_hook在子模块(gate_proj)上,但MLP整体输出通过残差连接已经绕过了子模块hook
- 需要改为在MLP层级别做forward修改,而不是子模块级别

### 新发现: GLM4 L26 shared→cat负效应

GLM4在L26注入shared分量时,类别读出反而下降(-1.05),这是其他模型没有的现象。可能解释:
1. L26是GLM4的"类别抑制门控层",当shared分量过强时反而抑制类别读出
2. 或者GLM4中层的shared方向含义与输入层不同,经过多层变换后已经反转
3. 这与Phase 448发现的GLM4负先验一致 — GLM4有一种"抑制-解锁"的类别门控机制

### 理论升级

最新理论应升级为:

```
语言编码是条件化关系槽位-对象解锁门控-类别共享通道维持(GLM4)/衰减(Qwen3/DS7B)-MLP绑定更新-注意力校准-候选读出动力系统
```

关键新增: GLM4的shared通道不衰减是它与Qwen3/DS7B的最根本差异,不是"共享程度高",而是"共享通道不关闭"。

### 硬伤与瓶颈

1. **Exp2 MLP内部消融实现有bug** — 需要用model.forward修改而非子模块hook
2. **GLM4 L26 shared负效应需要更精细定位** — 是哪一层的MLP还是Attn导致的?
3. **注入强度(inject_beta=2.0)是否在自然区间?** — 可能偏强,需要更小beta验证
4. **private分量的对象特异性** — 当前只测了apple的private,需要测更多对象
5. **DS7B L0 shared→cat为负(-1.15)** — 异常,可能是DS7B的embedding层问题

时间: {now}
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(header)
    f.write(content)

print(f"MEMO updated at {now}")
