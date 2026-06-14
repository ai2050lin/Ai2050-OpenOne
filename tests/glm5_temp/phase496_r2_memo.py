"""Phase 496 R2 MEMO updater"""
import sys, os, time
sys.path.insert(0, 'tests/glm5')

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
with open(memo_path, "a", encoding="utf-8") as f:
    f.write(f"""
## Phase 496 R2: MLP shared vs nonshared方向分解确认 [2026-06-14 22:19]

### 实验目标
解释Phase 496 R1的"表面矛盾": 为什么零化MLP后所有类别D都大幅下降（释放方向），但Phase 495发现释放类和刹车类的MLP shared贡献不同？

### 核心方法
在L(n-1)捕获MLP输出，分解为shared分量和非shared分量，分别计算对D的贡献。

### ★★★ 三模型MLP shared方向D贡献完整对比 ★★★

| 模型 | fruit shared | clothing shared | emotion shared | action shared | animal shared |
|------|-------------|-----------------|----------------|---------------|---------------|
| Qwen3 | **+0.32**(释放) | **+2.20**(释放) | **-6.51**(刹车) | **-1.58**(刹车) | **+0.84**(释放) |
| GLM4 | -0.08(近零) | -0.05(近零) | -0.06(近零) | -0.06(近零) | +0.01(近零) |
| DS7B | -4.18(刹车?) | -50.94(强刹车) | +5.47(释放?) | -48.29(强刹车) | +12.04(释放) |

### ★★★ Phase 496 R2 四大核心客观发现 ★★★

发现1: ★★★ Qwen3: shared方向是释放/刹车的唯一区分维度 ★★★
- 释放类(fruit/clothing/animal): D_shared > 0
- 刹车类(emotion/action): D_shared < 0
- 非shared方向对所有类别都贡献负D（通用抑制），不区分释放/刹车
- **shared方向是语言编码的关键维度**

发现2: ★★★ GLM4: MLP几乎不通过shared方向贡献，全部通过nonshared方向 ★★★
- D_shared在所有类别都接近零(-0.05~-0.08)
- GLM4的MLP在L(n-1)不使用shared语义方向来区分类别
- 这解释了GLM4的保守策略: MLP不翻转shared方向，保持刹车

发现3: ★★★ DS7B: shared方向的D贡献符号与Qwen3相反 ★★★
- fruit: D_shared = -4.18(刹车) vs Qwen3的+0.32(释放)
- emotion: D_shared = +5.47(释放) vs Qwen3的-6.51(刹车)
- DS7B的MLP对shared方向的使用方式与Qwen3完全不同
- 可能DS7B的shared方向定义/计算有问题（CPU offload导致不一致）

发现4: D_mlp与ΔD(zeroMLP)严重不匹配
- Qwen3 fruit: D_mlp=-0.36 vs ΔD(zeroMLP)=-39.08
- 原因: final LayerNorm(RMSNorm)的非线性放大效应
- 零化MLP → 残差流改变 → final RMSNorm重新缩放 → D大幅变化
- **直接D贡献计算忽略了final LayerNorm的间接效应**

### 对Phase 494评价的最终验证

用户对Phase 494的评价基本正确，但需要以下修正:

1. ✅ "跨层因果传递成立" — Phase 495逐样本确认
2. ✅ "shared_semantic主轴符号翻转是核心机制" — Qwen3确认
3. ❌ "三模型策略: Qwen3释放/GLM4刹车/DS7B强释放" — 需要修正:
   - Qwen3: MLP通过shared方向执行释放/刹车区分
   - GLM4: MLP不使用shared方向区分，shared方向在MLP中近零
   - DS7B: shared方向模式与Qwen3相反，可能因CPU offload不可靠
4. ⚠️ "action反转" — Qwen3 action的D_shared=-1.58(刹车)，与跨层因果一致

### 硬伤与瓶颈

1. DS7B的shared方向贡献模式与Qwen3相反（可能CPU offload导致）
2. final LayerNorm的间接效应占ΔD(zeroMLP)的主导，直接D贡献分析不够
3. GLM4的MLP完全不用shared方向，那GLM4的"保守刹车"机制是什么？需要另找
4. 非shared方向在Qwen3中贡献负D（通用抑制），其性质需要进一步分析

### 下一步核心任务

1. **控制final LayerNorm的因果实验**: 在final RMSNorm之前和之后分别零化MLP，分离直接效应和间接效应
2. **GLM4保守机制的真正来源**: 如果MLP不用shared方向，那L(n-1)的shared方向贡献来自哪里？（可能是Attn或残差流本身）
3. **DS7B的可靠性验证**: 换用纯GPU方式加载（如8bit全GPU），重测shared方向贡献
4. **非shared方向的性质分析**: 它是否是"通用抑制"方向？是否跨类别一致？
5. **MLP gate/up子模块因果干预**: 阻断gate_proj或up_proj输出后forward
""")

print(f"MEMO updated at {time.strftime('%H:%M:%S')}")
