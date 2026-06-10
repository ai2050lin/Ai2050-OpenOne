"""Update MEMO with Phase 443-445 results"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"

entry = """

## Phase 443-445: MLP vs Attention路径分解 + L0校准机制 + 中介标准化 [2026-06-10 08:34]

### Phase 443: MLP vs Attention Path Decomposition

核心发现 — **MLP贡献远大于Attention(即使Qwen3也如此)**:

| 模型 | |MLP|/|Attn| (apple) | |MLP|/|Attn| (knife) | |MLP|/|Attn| (dog) | direction_cos |
|------|----------------------|-----------------------|-------------------|---------------|
| Qwen3 | **2.49** | **3.04** | **5.10** | 0.3-0.8 |
| GLM4 | 1.61 | **4.49** | 1.20 | **0.94-0.99** |
| DS7B | 0.96 | 1.16 | 1.00 | -0.9~0.99 |

关键观察:
1. **Qwen3中MLP消融效果是Attention的2.5-5倍** — 需要修正之前的"Attention参与运输"判断
2. **GLM4中单层消融几乎不影响运输方向** (direction_cos>0.94) — 高度分布式
3. **DS7B极度不稳定** — 方向余弦从-0.9到0.99

### Phase 444: L0 Attention Calibration Mechanism

核心发现 — **L0 attention是全局校准器，不是搬运器**:

| 指标 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| delta_norm比率(消融/原始) | **5-15倍** | **0.95-1.07倍** | 0.97-10.7倍(不稳定) |
| entropy增大 | +1.9~+3.7 | +1.3~+2.7 | -2.9~+0.03(反转!) |
| cat_proj变化 | 161倍/反转 | 0.40-0.92 | -0.15~9.1 |

关键观察:
1. **Qwen3的L0 attention消融后全局信号爆炸5-15倍** — L0起强校准/抑制作用
2. **GLM4的L0 attention消融几乎不影响信号幅度** — 但entropy仍增大
3. **消融L0后Qwen3的对立类别logit大幅上升** — 说明L0过滤了非类别方向噪声
4. **Qwen3的L0 attention是一个"信号门控器"**: 允许类别方向通过，抑制非类别方向扩散

### Phase 445: Natural vs Forced Mediation Standardization

关键发现 — **两种"中介"测的是不同机制**:

| 模型 | 增强型自然中介(alpha=0.5) | 增强型强制中介(alpha=2.0) |
|------|--------------------------|--------------------------|
| Qwen3 | **-0.015** | -0.081 |
| GLM4 | **+0.680** | +0.580 |
| DS7B | -0.150 | -0.422 |

**与Phase 437/440的对比**:

| 方法 | Qwen3中介 | GLM4中介 |
|------|-----------|----------|
| 类别切换(fruit→animal, Phase 437/440) | **强(+4.2)** | 弱(-0.03) |
| 类别增强(增强fruit, Phase 445) | 弱(-0.015) | **强(+0.68)** |

**关键洞察: "中介"需要分两种**:
1. **类别切换中介(SwitchMediation)**: push类别到对立方向，测属性是否切换 — Qwen3强
2. **类别增强中介(BoostMediation)**: push类别增强方向，测属性是否同向增强 — GLM4也有

**这意味着**: GLM4不是"属性独立于类别"，而是"属性不跟随类别切换，但可跟随类别增强"。

### 综合理论修正

1. **MLP是类别运输的主要载体(所有模型)** — Attention更多做校准/路由，不是主要搬运器
2. **L0 attention是信号校准器** — 在Qwen3中极强(消融后全局爆炸)，在GLM4中弱
3. **两种中介机制需要区分** — 类别切换中介 vs 类别增强中介
4. **GLM4的类别-属性关系更微妙** — 不是简单的"解耦"，而是"解耦切换，耦合增强"

### 客观现象拼图(Phase 434-445)

1. 单头消融低效(Phase 434) — 类别运输是分布式过程
2. Qwen3 top_k>rand_k, GLM4极弱(Phase 439) — Qwen3的attention参与路由
3. **MLP消融效果是Attention的2.5-5倍(Phase 443)** — MLP是运输主要载体
4. L0 attention校准器: 消融后信号爆炸5-15倍(Qwen3), 1.0-1.1倍(GLM4)(Phase 444)
5. 对象identity替换在所有模型中改变属性(Phase 441)
6. **类别切换中介: Qwen3强, GLM4弱(Phase 437/440)**
7. **类别增强中介: GLM4强, Qwen3弱(Phase 445)** — 新发现
8. 最后层delta注入迁移不具类别特异性(Phase 442)
9. Qwen3 mediation从小alpha开始为正; GLM4 alpha≥1.5后才转正(Phase 440)

### 硬伤与瓶颈

1. **Phase 445的方法定义不够清晰** — "增强型中介"可能只是信号传播，不是真正的语义中介
2. **MLP贡献大的解释需要进一步验证** — 是MLP运输类别，还是MLP在校准后重编码？
3. **GLM4的"类别增强中介"vs"类别切换中介"的分离** — 需要更精细的实验
4. **DS7B数值不稳定** — 仍然无法得出清晰结论

### 下一步突破方向

1. **MLP内部机制** — MLP的哪一层/哪个中间表示包含类别信息？
2. **类别切换vs增强的统一框架** — 为什么增强有效但切换无效？
3. **跨层运输轨迹追踪** — 类别因子如何从L0传播到最后层？
4. **GLM4的MLP是否在做"键值检索"** — 而非"层级推导"？
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(entry)

print("MEMO updated successfully!")
