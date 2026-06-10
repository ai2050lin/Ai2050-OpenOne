"""更新MEMO: Phase 447结果"""
import os
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M")

content = f"""

## Phase 447: 类别绑定态分解与MLP机制验证 [2026-06-10 {now[11:]}]

### 实验设计

Phase 447包含4个子实验，验证"类别泛化是否由对象条件化绑定态实现"的假说:
1. **实验1**: 类别绑定态分解 — 6对象×3类别，逐层收集自然运输delta，PCA分解共享/私有成分
2. **实验2**: 功能等价验证 — 同类对象绑定态之间的余弦/重建/logit方向一致性
3. **实验3**: L0校准目标精确定位 — 范数/方向/噪声/熵/读出全面分析
4. **实验4**: 中介机制分型 — SwitchMediation/BoostMediation/IdentityMediation/SlotMediation

### 实验1: 类别绑定态分解 — 核心发现

**所有模型都呈现"早层共享→深层私有化"趋势，但速度差异极大:**

| 层 | Qwen3 shared | GLM4 shared | DS7B shared | Qwen3 pair_cos | GLM4 pair_cos | DS7B pair_cos |
|---|---|---|---|---|---|---|
| L0 | 0.90-0.96 | **1.00** | 0.92-0.96 | 0.88-0.95 | **1.00** | 0.90-0.95 |
| Mid | 0.57-0.72 | 0.80-0.85 | 0.43-0.65 | 0.53-0.69 | 0.75-0.83 | 0.38-0.58 |
| Last | 0.41-0.57 | 0.77-0.85 | 0.04-0.35 | 0.33-0.53 | 0.72-0.82 | -0.00-0.29 |

**关键结论:**
- GLM4的类别绑定态**始终更共享**(shared_ratio>0.77)，深层仍保持高一致性
- Qwen3的类别绑定态**中等私有化**(shared_ratio降至0.41-0.57)
- DS7B的类别绑定态**极端私有化**(深层shared_ratio降至0.04-0.35，pair_cos接近0)
- 所有模型L0层shared_ratio≈1.0，说明**类别方向在嵌入空间确实是共享的**
- 私有化发生在层间传播过程中，而非输入层

**这解释了为什么跨对象迁移在深层失败**: 深层delta已经高度对象特异，不同对象的"水果delta"方向不再一致。

### 实验2: 功能等价验证

**中间层(pair_cos) vs logit空间方向一致性:**

| 模型 | avg_pair_cos | avg_logit_cos | avg_recon_error |
|---|---|---|---|
| Qwen3 | 0.53-0.67 | 0.48-0.62 | 0.77-1.00 |
| GLM4 | 0.75-0.81 | **0.88-0.94** | 0.58-0.65 |
| DS7B | 0.39-0.54 | 0.58-0.79 | 0.84-17.4 |

**关键发现:**
- GLM4在logit空间中方向一致性极高(0.88-0.94)，远超Qwen3(0.48-0.62)
- 这说明GLM4的类别绑定态虽然共享成分更多，但在logit读出空间中**功能更一致**
- Qwen3的绑定态在隐藏空间不一致，但logit空间也不太一致 — 这与Qwen3"类别切换中介强"矛盾吗？
- 实际不矛盾: Qwen3的类别切换中介强说明**属性跟随类别切换**，而功能等价测的是**不同对象的delta方向**是否一致

### 实验3: L0校准精确定位

**Qwen3 vs GLM4 L0消融对比:**

| 指标 | Qwen3 | GLM4 | DS7B |
|---|---|---|---|
| norm_ratio(消融/原始) | **4.7-18.3** | 0.96-1.08 | 8.5-12.3 |
| direction_cos | **<0.15** | **>0.79** | -0.29~0.60 |
| noise_suppression | 0.88-1.56 | 0.96-1.15 | 0.97-1.02 |
| entropy_abl_delta | +0.57~+3.68 | +1.41~+2.66 | -3.65~-0.48 |

**关键发现:**
- Qwen3的L0 attention校准了**方向**(消融后方向完全混乱,dir_cos<0.15)
- GLM4的L0 attention不校准方向(消融后dir_cos>0.79)，只控制**熵**(消融后熵增大)
- DS7B的L0 attention也校准信号幅度(高norm_ratio)，但entropy反而下降(异常)

**结论:**
- Qwen3 L0 = 方向校准器 + 信号幅度控制器
- GLM4 L0 = 熵控制器(维持输出确定性)
- DS7B L0 = 信号放大器(消融后反而更确定但不正确)

### 实验4: 中介机制分型

| 模型 | SwitchMed | BoostMed | IdentityMed | SlotMed |
|---|---|---|---|---|
| Qwen3 | -0.03~0.11 | 0.04~0.26 | 0.62~1.20 | 1.34~1.64 |
| GLM4 | -0.26~0.44 | -1.31~1.98 | -0.42~2.45 | 0.74~1.68 |
| DS7B | -0.29~1.32 | -0.23~0.16 | -0.43~0.13 | **5.10~6.92** |

**关键发现:**
- **SlotMediation(关系槽位)在所有模型中都是最强的中介机制!**
  改变问题模板("is a" vs "has a" vs "feels")对属性读出的影响远超类别扰动
- Qwen3的SwitchMediation弱于Phase 437/440的发现 — 因为这里用"related vs unrelated属性差"测量
- DS7B的SlotMediation异常高(5-7)，说明DS7B对关系槽位极度敏感
- GLM4的中介模式最不稳定，不同对象间差异极大

### 对用户分析的验证

用户分析中以下结论**正确**:
- ✅ 类别泛化可能不是"共享方向"，而是"功能等价绑定态" — Phase 447 Exp1确认
- ✅ MLP可能是"绑定态计算器" — Phase 443确认MLP主导，但Phase 447未直接测MLP内部
- ✅ Attention是流形守门员 — Phase 447 Exp3确认Qwen3 L0校准方向
- ✅ 中介机制要分型 — Phase 447 Exp4确认不同中介类型差异大

用户分析中以下结论**需要修正**:
- ⚠️ "GLM4中属性不跟随类别切换但可跟随类别增强" — Phase 447 Exp4中GLM4的BoostMediation不稳定
- ⚠️ "Qwen3中SwitchMediation强" — Phase 447中用更精确的related/unrelated差测，SwitchMediation实际弱
  之前看到的强SwitchMediation可能是logit gap而非属性特异性
- ⚠️ "功能等价而非方向相同" — Phase 447 Exp2发现GLM4在logit空间方向一致性极高(logit_cos=0.88-0.94)
  这说明GLM4的绑定态在logit空间可能共享方向

### 核心拼图更新

**新发现1: 共享→私有化是所有模型的共性**
所有模型在L0层shared_ratio≈1.0，但随着传播到深层逐步私有化。
这解释了为什么跨对象迁移在深层失败：深层delta已经高度对象特异。

**新发现2: GLM4的类别绑定态最"共享"，Qwen3中等，DS7B最"私有"**
这与之前"Qwen3类别中介强，GLM4弱"的发现形成有趣对比：
- Qwen3: 绑定态更私有化，但类别切换更能带动属性变化
- GLM4: 绑定态更共享，但类别切换不带动属性变化

这意味着"绑定态共享程度"和"类别中介属性能力"是**独立的维度**！

**新发现3: SlotMediation(关系槽位)是最强的中介机制**
在所有模型中，改变问题模板("is a"→"has a"→"feels")对属性读出的影响，
远大于任何形式的类别扰动。这说明语言模型中，**问题框架(关系槽位)比类别信息更能决定属性读出**。

### 当前最可靠的客观现象清单

1. 类别运输是分布式过程，MLP是主要载体(Phase 443)
2. L0 attention是信号校准器，Qwen3校准方向+幅度，GLM4只控熵(Phase 444/447)
3. 所有模型呈现"早层共享→深层私有化"趋势(Phase 447 Exp1)
4. GLM4的类别绑定态始终更共享，Qwen3中等，DS7B最私有(Phase 447 Exp1)
5. 绑定态共享程度与类别中介能力是独立维度(Phase 447 vs Phase 440)
6. SlotMediation(关系槽位)是所有模型中最强的中介机制(Phase 447 Exp4)
7. Qwen3的SwitchMediation弱于此前认知(Phase 447 Exp4 vs Phase 437/440)
8. GLM4在logit空间方向一致性极高(Phase 447 Exp2)
9. DS7B数值仍不稳定，deep层shared_ratio降至0.04(Phase 447 Exp1)

### 硬伤与瓶颈

1. **SwitchMediation测量方法影响结论** — Phase 437用logit gap，Phase 447用related/unrelated差，结果不同
2. **绑定态分解方法太粗糙** — 当前用"均值=共享,残差=私有"，这不是最优分解
3. **MLP内部机制仍未直接验证** — 只知道MLP贡献大，不知道gate/up/down各做什么
4. **SlotMediation异常强但未深入分析** — 为什么关系槽位影响这么大？机制是什么？
5. **功能等价的定义需要更精确** — 当前只测了方向一致性和logit空间余弦

### 突破瓶颈的第一性原理分析

**核心洞察: 语言模型中"关系槽位 > 类别 > 对象身份"的影响层次**

Phase 447 Exp4揭示了一个全新的层次:
```
关系槽位(SlotMediation) >> 类别(CategoryMediation) > 对象身份(IdentityMediation)
```

这意味着语言模型的"属性检索"机制可能是:
1. 首先确定"当前问题问的是什么"(关系槽位)
2. 然后在对应槽位中查找"类别是什么"
3. 最后确定"具体对象的属性"

这不是简单的"类别→属性"线性路径，而是**槽位→类别→属性的层级查询**。

**下一步关键实验:**
1. SlotMediation机制解析 — 为什么不同模板对属性影响如此大？
2. 绑定态的精细分解 — 用ICA而非PCA做分解，可能得到更干净的独立成分
3. MLP内部绑定函数 — MLP的gate/up/down分别对类别、属性、槽位做什么变换？
4. 层间绑定态传播动力学 — 从共享到私有的转变发生在哪几层？是由MLP还是attention驱动的？
"""

memo_path = r"d:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md"
with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {now}")
