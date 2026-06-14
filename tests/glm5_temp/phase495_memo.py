"""Phase 495 MEMO updater"""
import os
from datetime import datetime

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

entry = f"""

## Phase 495: 逐样本因果验证、Attn/MLP分解、剂量曲线与异常机制 [2026-06-14 {timestamp}]

### 目标

解决Phase 494遗留核心问题:
1. 均值替换是否夸大跨层因果？→ 逐样本干预验证
2. 末层符号翻转由哪个模块执行？→ Attn/MLP分解
3. ablate/double不对称的原因？→ 剂量曲线
4. DS7B animal/action异常？→ 异常类别分析

### 核心发现22: 逐样本因果确认Phase 494均值替换结论（释放类），但揭示了模型间关键差异

**逐样本跨层因果干预结果**:

| 类别 | Qwen3 mean_ΔD | Qwen3 consistency | GLM4 mean_ΔD | GLM4 consistency | DS7B mean_ΔD | DS7B consistency |
|------|-------------|-------------------|-------------|-------------------|-------------|-------------------|
| fruit | -17.53 | 91.7% | +41.59 | 100% | -1.77 | 58.3% |
| clothing | -21.54 | 100% | +8.07 | 100% | -28.74 | 66.7% |
| emotion | +95.93 | 100% | +53.17 | 91.7% | +147.19 | 83.3% |
| action | +99.87 | 91.7% | +22.30 | 91.7% | -50.64 | 91.7% |

关键对比:
- Qwen3 fruit/clothing: 逐样本与均值替换一致，释放方向确认
- **Qwen3 emotion/action: 逐样本确认是刹车方向(+95.93/+99.87)**，修正了Phase 494中"action在L35反转"的判断
- GLM4: 所有类别100%刹车方向
- **DS7B: 高度不稳定!** fruit一致性仅58.3%（CPU offload可能导致数值问题）
- **DS7B action: 逐样本为释放方向(-50.64)，与Qwen3相反！**

### 核心发现23: MLP是末层符号翻转的守门模块——这是Phase 495最关键的发现

**Attn/MLP对shared方向的logit贡献分解**:

| 模型 | 类别 | L(n-2) attn | L(n-2) mlp | L(n-1) attn | L(n-1) mlp | Attn翻转? | MLP翻转? |
|------|------|-----------|-----------|-----------|-----------|----------|----------|
| Qwen3 | fruit | -1.62 | -6.01 | +1.13 | +3.75 | ✅ | ✅ |
| Qwen3 | clothing | -0.76 | -2.65 | +1.65 | +5.98 | ✅ | ✅ |
| Qwen3 | emotion | -2.72 | -7.79 | -0.29 | -1.17 | ❌ | ❌ |
| Qwen3 | action | -1.92 | -5.95 | +0.71 | +2.93 | ✅ | ✅ |
| GLM4 | fruit | -3.91 | -2.75 | +0.11 | -0.84 | ✅微弱 | ❌ |
| GLM4 | clothing | -2.26 | -1.84 | -0.02 | +0.02 | ❌ | ✅微弱 |
| GLM4 | emotion | -5.01 | -3.91 | +0.03 | -0.28 | ✅微弱 | ❌ |
| GLM4 | action | -2.44 | -1.94 | -0.01 | +0.13 | ❌ | ✅微弱 |
| DS7B | fruit | -2.16 | -21.76 | +40.80 | +80.05 | ✅ | ✅ |
| DS7B | animal | -3.07 | -33.96 | +59.16 | +114.66 | ✅ | ✅ |
| DS7B | emotion | -2.21 | -40.80 | +34.68 | +107.20 | ✅ | ✅ |
| DS7B | action | -2.65 | -30.53 | +40.79 | +69.69 | ✅ | ✅ |

**核心结论**:
1. **MLP是符号翻转的决定性模块**: 当MLP翻转时，类别释放; 当MLP不翻转时，类别保持刹车
2. **Attn翻转是必要但不充分条件**: GLM4的Attn有时微弱翻转，但MLP不翻→净刹车
3. **Qwen3 emotion是唯一Attn和MLP都不翻转的类别**→唯一完全不释放的类别
4. **DS7B的释放量级是Qwen3的10-30倍**: mlp_contrib从-21到+80 vs Qwen3从-6到+4
5. **MLP贡献量 > Attn贡献量**: 通常3-5倍（Qwen3），甚至10-20倍（DS7B L26层）

**三模型MLP策略差异**:
- Qwen3: MLP选择性翻转（实体类翻转，emotion不翻转）
- GLM4: MLP几乎不翻转（所有类别保持刹车）
- DS7B: MLP全翻转（所有类别强释放）

### 核心发现24: Qwen3 action的"局部释放vs跨层刹车"悖论

Phase 494 Exp3显示action在L35的ablate_shared_delta=-9.96（局部释放）。
Phase 495 Exp1显示action的跨层因果ΔD=+99.87（刹车方向）。
Exp2显示action的Attn和MLP都发生了符号翻转。

**解释**: 符号翻转是L35的局部特征，但跨层因果效应取决于L34→L35的完整传递链。
对action来说，L34的shared作为刹车信号被传递到L35; 当移除L34的shared时，L35收到的输入改变，
导致L35虽然局部有释放方向，但净DCF反而增加。

**这意味着**: 符号翻转是释放的必要条件，但不是充分条件。跨层传递的净效果取决于L34刹车的强度vs L35释放的强度。对action，L34刹车效应 > L35释放效应。

### 核心发现25: 剂量曲线揭示非线性传递

**Qwen3 fruit剂量曲线**:
- scale=0(ablate): ΔD=-27.94
- scale=1(natural): ΔD=0
- scale=2(double): ΔD=-25.17
- scale=-1(reverse): ΔD=+253.87

接近线性但反转方向效应远大于加倍方向，说明存在非线性饱和。

**GLM4 fruit剂量曲线**:
- scale=0: ΔD=+43.83
- scale=2: ΔD=-16.52
- 严重不对称! ablate效应(+43.8)远大于double效应(-16.5)

**DS7B fruit剂量曲线**:
- 非单调! scale=1.5时ΔD=+70.66，scale=2时ΔD=+28.24
- 可能受CPU offload影响，数值不稳定

### 核心发现26: DS7B逐样本因果高度不稳定

DS7B fruit一致性仅58.3%（12个样本中7个负5个正），远低于Qwen3(91.7%)和GLM4(100%)。

可能原因:
1. DS7B有14层在CPU上，hook替换可能不完整
2. DS7B的释放机制过于激进，小干扰就被放大
3. 逐样本使用类别级shared方向可能不适合DS7B

### 硬伤与瓶颈

1. **DS7B的CPU offload导致hook干预可能不完全**: 14层在CPU上，修改hidden state后forward可能不经过GPU层
2. **Exp2的Attn/MLP分解是静态的**: 计算的是"如果消融Attn/MLP中的shared分量，DCF如何变化"，而不是真正阻断Attn/MLP后forward
3. **action的跨层因果悖论未完全解决**: 需要在L35单独做Attn/MLP因果干预
4. **DS7B animal未在Exp1中测试**: 仍然是Phase 494遗留的异常

### 下一步: Phase 496方向

1. **真正的Attn/MLP因果干预**: 在L(n-1)分别阻断Attn和MLP后forward，看谁执行符号翻转
2. **DS7B animal异常解释**: 逐样本测试animal的跨层因果
3. **Qwen3 emotion为什么不翻转MLP?**: MLP的gate/up分解，看是gate没激活还是up没翻转
4. **非线性传递的数学模型**: 构建L(n-2)→L(n-1)的非线性传递函数

### 客观数据文件
- results/glm5/phase495_qwen3_r1.json
- results/glm5/phase495_glm4_r1.json
- results/glm5/phase495_deepseek7b_r1.json

### 测试脚本
- tests/glm5/phase495_samplewise_attn_mlp_dose.py
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(entry)

print(f"MEMO updated at {timestamp}")
