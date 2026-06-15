"""
Phase 498 MEMO: 自动整理关键结果并追加到MEMO文件
"""
import json
from pathlib import Path
from datetime import datetime

MEMO_FILE = Path("research/glm5/docs/AGI_GLM5_MEMO.md")
RESULTS_DIR = Path("results/glm5")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# 读取结果
def load_results(model, round_num):
    f = RESULTS_DIR / f"phase498_{model}_r{round_num}.json"
    if f.exists():
        with open(f, encoding='utf-8') as fh:
            return json.load(fh)
    return None

qwen3 = load_results("qwen3", 1)
glm4 = load_results("glm4", 1)

# 整理Exp1关键数据
def extract_exp1(data):
    if not data or "exp1_rmsnorm_math" not in data:
        return "N/A"
    lines = []
    for cat in ["fruit", "clothing", "emotion", "action", "animal"]:
        if cat not in data["exp1_rmsnorm_math"]:
            continue
        d = data["exp1_rmsnorm_math"][cat]
        lines.append(
            f"  {cat}: num_eff={d['mean_numerator_effect']:.3f}, "
            f"den_eff={d['mean_denominator_effect']:.3f}, "
            f"interact={d['mean_interaction']:.3f}, "
            f"gain_eff={d['mean_gain_effect']:.3f}"
        )
    return "\n".join(lines)

def extract_exp2(data):
    if not data or "exp2_fixed_rmsnorm" not in data:
        return "N/A"
    lines = []
    for cat in ["fruit", "clothing", "emotion", "action", "animal"]:
        if cat not in data["exp2_fixed_rmsnorm"]:
            continue
        d = data["exp2_fixed_rmsnorm"][cat]
        lines.append(
            f"  {cat}: D_normal={d['mean_D_normal']:.2f}, "
            f"D_fixed_denom={d['mean_D_fixed_denom']:.2f}, "
            f"D_no_gain={d['mean_D_no_gain']:.2f}, "
            f"D_no_norm={d['mean_D_no_norm']:.2f}, "
            f"rms_denom_eff={d['mean_rms_denom_effect']:.4f}, "
            f"gain_weight_eff={d['mean_gain_weight_effect']:.3f}, "
            f"norm_scale_eff={d['mean_norm_scale_effect']:.2f}"
        )
    return "\n".join(lines)

def extract_exp3(data):
    if not data or "exp3_mlp_norm_channel" not in data:
        return "N/A"
    lines = []
    for cat in ["fruit", "clothing", "emotion", "action", "animal"]:
        if cat not in data["exp3_mlp_norm_channel"]:
            continue
        d = data["exp3_mlp_norm_channel"][cat]
        lines.append(
            f"  {cat}: mlp_norm={d['mean_mlp_norm']:.1f}, "
            f"Δzero={d['mean_delta_zero']:.3f}, "
            f"Δ0.5x={d['mean_delta_scale_05']:.3f}, "
            f"Δ2x={d['mean_delta_scale_2']:.3f}, "
            f"Δortho={d['mean_delta_ortho']:.3f}, "
            f"Δaligned={d['mean_delta_aligned']:.3f}"
        )
    return "\n".join(lines)

def extract_exp4(data):
    if not data or "exp4_action_sign_flip" not in data:
        return "N/A"
    results = data["exp4_action_sign_flip"]
    if not results:
        return "N/A"
    n_flipped = sum(1 for r in results if r.get("sign_flipped", False))
    lines = [f"  翻转率: {n_flipped}/{len(results)}"]
    for r in results[:4]:
        lines.append(
            f"  {r['obj']}: D_pre={r['D_pre']:.2f}, D_post={r['D_post']:.2f}, "
            f"flipped={r['sign_flipped']}, target_logit: {r['target_logit_pre']:.2f}→{r['target_logit_post']:.2f}, "
            f"comp_logit: {r['comp_logit_pre']:.2f}→{r['comp_logit_post']:.2f}"
        )
    return "\n".join(lines)

memo_text = f"""

## Phase 498: RMSNorm读出几何分解与范数通道闭环 [{timestamp}]

### 本轮执行命令
- `python tests/glm5/phase498_rmsnorm_decomposition.py qwen3 1`
- `python tests/glm5/phase498_rmsnorm_decomposition.py glm4 1`
- `python tests/glm5/phase498_rmsnorm_decomposition.py deepseek7b 1`

### 生成脚本
- `tests/glm5/phase498_rmsnorm_decomposition.py`

### 原理
Phase 498的核心是对RMSNorm读出机制进行数学精确分解。关键公式:
- D_post = <h_pre, g⊙w_D> / rms(h_pre) = numerator / denominator
- 干预MLP后: δD ≈ δnumerator/rms - D·δrms/rms = 分子效应 + 分母效应
- 四种对照读出: normal/fixed_denom/no_gain/no_norm
- MLP范数通道: zero/0.5x/2x/ortho/aligned 五种干预

### Exp1: RMSNorm数学精确分解 (Qwen3)

{extract_exp1(qwen3)}

**关键发现**:
- gain_effect(增益权重效应)是D_post的主导贡献者: fruit +4.68, clothing +2.66, animal +4.88
- numerator_effect(方向/分子效应): 负向，fruit -1.68, animal -1.29 — MLP在pre-norm空间写入的方向对D贡献为负
- denominator_effect(范数/分母效应): 较小但一致为负 — 去掉MLP后RMS变大，D下降
- **RMSNorm weight (gain向量)才是D_post的主要来源，不是方向写入！**

### Exp2: 固定RMSNorm对照 (Qwen3)

{extract_exp2(qwen3)}

**关键发现**:
- norm_scale_effect(归一化缩放)极大: fruit -28.6, clothing -24.9, emotion -29.1, **action +11.9**, animal -26.8
- gain_weight_effect(增益权重)巨大: fruit +4.68, clothing +2.66
- rms_denom_effect(动态分母): 接近0!
- **action的norm_scale为正(+11.9)，其他类别为负 — 这是action符号翻转的根源！**
- D_no_norm ≈ D_pre (确认无归一化 = pre-norm空间)
- D_no_gain ≈ D_post/gain增益因子 (去掉gain后D大幅下降)

### Exp2: 固定RMSNorm对照 (GLM4, 无gain weight)

{extract_exp2(glm4)}

**关键发现**:
- GLM4的gain weight在meta device无法访问，但通过fixed_denom vs no_gain相等验证: GLM4没有可测量的gain_weight_effect (可能全部在safetensors中)
- D_normal >> D_fixed_denom: fruit 3.94 vs 1.56 — 说明norm缩放对GLM4也很重要
- action: D_no_norm=-4.11, D_normal=1.22 — 同样符号翻转!

### Exp3: MLP范数通道闭环 (Qwen3)

{extract_exp3(qwen3)}

**关键发现**:
- clothing: Δortho=+2.03 > Δzero=+0.60 — MLP方向本身在**抑制**clothing的D，正交方向反而释放!
- action: Δortho≈Δzero≈+0.57 — MLP对action的范数和方向效应相当
- Δ2x(加倍MLP): clothing=-1.30 — MLP加倍导致D下降，MLP在**抑制**释放
- fruit/animal: Δortho比Δzero大很多(3.34 vs 2.16, 3.67 vs 1.73) — MLP方向正在抑制这些类别的D
- **MLP的方向对大多数类别起抑制作用，去掉MLP方向反而提升D_pre。但RMSNorm重映射后，这种抑制被gain向量逆转！**

### Exp3: MLP范数通道闭环 (GLM4)

{extract_exp3(glm4)}

**关键发现**:
- GLM4中MLP也是抑制性的: Δzero为负
- emotion: Δortho=-3.64 >> Δzero=-1.25 — MLP方向对emotion有强抑制
- Δaligned ≈ Δzero — 对齐到residual方向的MLP与零化效果类似

### Exp4: Action类符号翻转专项 (Qwen3)

{extract_exp4(qwen3)}

**关键发现**:
- **Action 8/8全部符号翻转!** D_pre为负, D_post为正
- comp_logit从16-25降到1-3 (压缩93%), target_logit只从7-15降到3-5 (压缩66%)
- 翻转机制: RMSNorm对竞争token的压缩远大于对target token的压缩
- 不是gain向量翻转了action方向，而是RMSNorm**不等比例压缩**了target vs competitor

### Exp4: Action类符号翻转专项 (GLM4)

{extract_exp4(glm4)}

**关键发现**:
- **GLM4 Action同样8/8全部翻转!**
- comp_logit从5-7降到-2到-0.5 (压缩超过100%，变负!)
- target_logit从1-2降到0-1 (只压缩50%)
- 与Qwen3相同的机制: RMSNorm对comp的压缩 >> 对target的压缩

### DS7B结果
- DS7B因CPU offload导致大量NaN，所有实验数据不可用
- 与Phase 497一致，需纯GPU方式重测

### 核心客观结论

1. **RMSNorm gain向量是D_post的主导贡献者** (Qwen3: fruit gain_eff=+4.68 vs D_post=7.13)
2. **归一化缩放效应(norm_scale)在大多数类别为负(-25到-29)，但action为正(+11.9)** — 这是action符号翻转的数学根源
3. **动态RMS分母效应接近0** — 不是分母变化导致D变化
4. **MLP方向对大多数类别起抑制作用** (Δortho > Δzero) — MLP在pre-norm空间写入的方向实际上在压制D
5. **RMSNorm gain向量逆转了MLP的抑制效应** — pre-norm中MLP抑制D，但gain向量放大了被抑制方向的D贡献
6. **Action符号翻转机制**: RMSNorm对竞争token的压缩率远大于对target token的压缩率，导致D符号翻转
7. **GLM4与Qwen3共享相同的RMSNorm读出几何机制**，虽然gain weight无法直接访问

### 机制修正

之前理论: "MLP改变范数 → RMSNorm缩放 → D贡献变化"
修正: "MLP在pre-norm空间写入方向(实际上抑制D) → RMSNorm gain向量重映射 → 抑制效应被逆转 → D_post显现为正"

更精确: D_post = <h_pre, g⊙w_D> / rms(h_pre)
- <h_pre, w_D> (无gain) 对大多数类别为负或很小
- gain向量g把w_D重映射为g⊙w_D，使得<g⊙w_D, h_pre>变为正且大幅增大
- 这就是RMSNorm gain向量的核心作用: **语义读出门控**

### 问题与硬伤

1. GLM4的gain weight在meta device无法获取，无法做完整的gain向量分解
2. DS7B全部不可用，缺少第三个模型的验证
3. gain向量g为什么能逆转MLP的抑制效应? 这是学习到的结构还是数学必然?
4. MLP在pre-norm空间的方向为什么对大多数类别抑制D? 需要分析MLP的W_D投影
5. action的norm_scale为正而其他为负的深层原因未解释
6. ortho方向的D效应比zero还大，说明不是简单的范数通道

### 理论研究进展

Phase 498的核心突破是发现了**RMSNorm gain向量的语义门控作用**:
- 不是简单的缩放/归一化
- gain向量g与w_D的逐元素乘积g⊙w_D定义了**有效读出方向**
- 这个有效读出方向与pre-norm hidden的内积决定了D_post
- MLP写入的方向在原始w_D下为负/小，但在g⊙w_D下为正/大
- 这解释了为什么zeroMLP的大效应主要来自RMSNorm重映射

下一步关键问题: gain向量g的数学结构是什么? 它如何把"抑制性方向"变成"释放性方向"?
"""

# 追加到MEMO文件
with open(MEMO_FILE, 'a', encoding='utf-8') as f:
    f.write(memo_text)

print(f"MEMO appended at {timestamp}")
print(f"File: {MEMO_FILE}")
