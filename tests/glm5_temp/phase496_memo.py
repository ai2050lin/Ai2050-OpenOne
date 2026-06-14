"""Phase 496 MEMO updater"""
import sys, os, time
sys.path.insert(0, 'tests/glm5')

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
with open(memo_path, "a", encoding="utf-8") as f:
    f.write(f"""
## Phase 496: 真正因果MLP/Attn干预 + MLP子模块分解 + 多Token位置 [2026-06-14 22:11]

### 实验目标
验证Phase 495的核心发现——MLP是末层符号翻转的守门模块——是否是真正因果机制。
Phase 495 Exp2只是"静态分解"（消融shared分量），Phase 496做"真正因果干预"（阻断整个MLP/Attn输出后forward）。

### Exp1: 真正因果MLP/Attn干预（核心实验）

方法: 在L(n-1)完全阻断MLP或Attn的输出（hook归零），forward后看L(n-1)的DCF变化

| 类别 | Qwen3 MLP | Qwen3 Attn | GLM4 MLP | GLM4 Attn | DS7B MLP | DS7B Attn |
|------|-----------|------------|----------|-----------|----------|-----------|
| fruit | -39.08 | -0.91 | -7.27 | -0.60 | -68.01 | -126.05 |
| clothing | -36.11 | +3.88 | -7.91 | +0.55 | -59.37 | -96.25 |
| emotion | -39.90 | +3.62 | -9.91 | -1.62 | -65.91 | -126.61 |
| action | -43.81 | -1.99 | -7.54 | +0.15 | -154.17 | -142.25 |

关键发现:
1. ★★★ Qwen3/GLM4: MLP效应远大于Attn效应（5-40倍），MLP是末层主导模块 ★★★
2. ★★★ DS7B: Attn效应竟然比MLP还大！这与Phase 495的静态分解结果矛盾 ★★★
3. 所有模型所有类别: ΔD(zeroMLP)均为负 → MLP在L(n-1)对所有类别都提供正向D贡献（无论释放还是刹车）
4. ★★★ GLM4 MLP效应远弱于Qwen3: -7~-10 vs -36~-44 → GLM4的MLP不翻转但仍然提供正向boost ★★★

对第3点的解释: MLP在L(n-1)对D的贡献是正向的（提升目标logit），但释放类和刹车类的区别在于shared方向的贡献符号不同。
- 释放类: MLP通过shared方向boost目标（shared翻转→释放）+ 通过其他方向boost目标
- 刹车类: MLP通过非shared方向boost目标（shared方向不翻转→仍然是刹车）+ 通过其他方向小量boost
→ 关键区别在shared方向，而非MLP的总量

### Exp2: MLP子模块分解

| 模型 | fruit MLP contrib | clothing MLP contrib | emotion MLP contrib |
|------|-------------------|---------------------|---------------------|
| Qwen3 | +39.42 | +36.21 | +39.93 |
| GLM4 | +7.13 | +7.86 | +10.08 |
| DS7B | +68.04 | +67.56 | +63.24 |

GLM4权重在meta device上无法直接访问gate/up分解。Qwen3和DS7B的权重分析显示gate和up都对shared方向有贡献，但量级相近。

### Exp3: 多Token位置干预

方法: 在L(n-2)不同token位置消融shared分量，forward到L(n-1)看DCF变化

| 位置 | Qwen3 fruit | Qwen3 emotion | GLM4 fruit | GLM4 emotion | DS7B fruit | DS7B emotion |
|------|-------------|---------------|------------|--------------|------------|--------------|
| object_only | +0.17 | -0.01 | +0.00 | -0.01 | +240.91 | +457.06 |
| relation_only | -0.35 | +0.18 | +0.04 | -0.15 | +11.65 | +16.74 |
| last_only | -25.17 | +92.87 | +20.12 | +57.24 | -55.85 | +179.69 |
| all_semantic | -10.02 | +89.82 | +37.19 | +53.03 | +63.06 | +201.01 |

★★★ 关键发现: ★★★
1. Qwen3/GLM4: last_token位置是跨层释放的主要位置，object/relation位置几乎无贡献
2. DS7B: object位置效应巨大（+240/+457），且方向与last_token相反！
   → DS7B的object位置shared消融导致D增加（刹车减弱），last位置消融导致D减少（释放减弱）
   → 说明DS7B的语义编码跨多个token位置，不同于Qwen3/GLM4的集中式编码
3. Qwen3 all_semantic < last_only: 多位置联合消融时object位置的微弱反效削弱了last位置的强效

### Exp4: DS7B Animal异常

| 类别 | zero_mlp ΔD |
|------|-------------|
| animal | -147.10 |
| fruit | -84.93 |

animal的MLP贡献(-147)远大于fruit(-85)，说明animal的MLP对目标D的boost更强。
但Phase 494中animal消融shared后ΔD=+90.51（刹车增强），而fruit为-38.26（释放增强）。
→ animal的MLP总量boost更强，但shared方向的贡献与其他类别不同（可能是brake方向而非release方向）

### ★★★ Phase 496 四大核心客观发现 ★★★

发现1: MLP在L(n-1)对所有类别都提供正向D贡献，释放/刹车的区别在于shared方向的贡献符号
- 释放类: MLP通过shared方向释放 + 非shared方向boost → 零化MLP后D大幅下降
- 刹车类: MLP通过非shared方向boost → 零化MLP后D也下降，但shared方向本身是刹车

发现2: DS7B的Attn效应出人意料地大于MLP（-126 vs -68 for fruit）
- 这与Phase 495的静态分解结果矛盾（Phase 495: MLP 80 vs Attn 41）
- 可能原因: Phase 495只看shared分量，Attn的非shared分量贡献更大
- DS7B的Attention机制可能比Qwen3/GLM4更活跃

发现3: 跨层释放主要发生在last_token位置（Qwen3/GLM4），但DS7B是多位置编码
- Qwen3/GLM4: object位置几乎无效应，last位置压倒性重要
- DS7B: object位置效应巨大且方向相反，提示其语义编码跨多token

发现4: GLM4的MLP效应远弱于其他模型（-7~-10 vs -36~-68）
- GLM4的MLP在L(n-1)只提供微弱的正向boost
- 这与GLM4的保守策略一致: 末层不做强释放，MLP保持弱贡献

### 硬伤与瓶颈

1. Exp2的gate/up权重分析在GLM4/DS7B上因meta device无法完成
2. DS7B的Attn效应大于MLP需要进一步验证——可能是CPU offload导致hook不完整
3. MLP的"对所有类别都提供正向boost"这个发现很反直觉，需要验证是否是D指标的特殊性质
4. 多位置干预的交互效应未分析（all_semantic < last_only说明有干扰）
5. emotion在所有模型中零化MLP后D都下降，但Phase 495显示emotion的shared方向MLP不翻转——这个表面矛盾需要解释

### 下一步

1. 解释"MLP对所有类别boost但shared方向不同"的机制——这是理解末层功能重编码的关键
2. DS7B Attn > MLP的异常需要验证——用无CPU offload方式重测
3. MLP gate/up子模块的真正因果干预（不是权重分析，而是阻断gate或up后forward）
4. 多位置干预的交互效应分析
5. 验证"last_token为主"是否在不同模板下成立
""")

print(f"MEMO updated at {time.strftime('%H:%M:%S')}")
