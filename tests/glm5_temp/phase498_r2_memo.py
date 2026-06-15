"""
Phase 498 R2 MEMO: Gain向量分析结果追加
"""
import json
from pathlib import Path
from datetime import datetime

MEMO_FILE = Path("research/glm5/docs/AGI_GLM5_MEMO.md")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

memo_text = f"""

## Phase 498 R2: Gain向量深度分析 [{timestamp}]

### 本轮执行命令
- `python tests/glm5/phase498_gain_vector_analysis.py qwen3 2`
- `python tests/glm5/phase498_gain_vector_analysis.py glm4 2`

### 生成脚本
- `tests/glm5/phase498_gain_vector_analysis.py`

### 原理
分析RMSNorm gain向量g的数学结构:
- D_post = <h, g⊙w_D> / rms(h)
- 对比D_no_gain = <h/rms, w_D> vs D_with_gain = <h*g/rms, w_D>
- 分析g如何改变w_D的方向和范数

### 核心结果1: RMSNorm Gain向量统计

Qwen3: mean=2.7600, std=0.4547, 99.4%维度>1 (2545/2560)
GLM4: mean=3.4752, std=0.2162, 99.98%维度>1 (4095/4096)

**两个模型的gain向量几乎全部>1，这是放大性gain，不是抑制性gain。**

### 核心结果2: Gain向量不改变方向，只放大范数

Qwen3 cos(w_D, g⊙w_D):
- fruit: 0.993, clothing: 0.993, emotion: 0.991, action: 0.989, animal: 0.993

GLM4 cos(w_D, g⊙w_D):
- fruit: 0.999, clothing: 0.999, emotion: 0.999, action: 0.997, animal: 0.998

**cos全部>0.989！gain向量几乎不改变DCF读出方向，只是把||w_D||放大2.7-3.6倍。**

### 核心结果3: Gain向量的类别差异增益

Qwen3 gain_eff:
- fruit: +4.77, clothing: +2.69, emotion: +0.63, action: +2.61, animal: +4.79

GLM4 gain_eff:
- fruit: +2.25, clothing: +2.96, emotion: +1.64, action: +2.29, animal: +3.19

**emotion的gain_eff最小(Qwen3: 0.63, GLM4: 1.64)，这解释了为什么emotion的D_post最低！**
**gain向量在emotion类别上的"语义门控"最弱。**

### 核心结果4: Action的符号翻转完全由gain+norm缩放导致

Qwen3 action: D_no_gain=-0.18到-1.62, D_with_gain=+0.05到+3.74
GLM4 action: D_no_gain=-0.55到-1.44, D_with_gain=-0.02到+2.32

**无gain时action的D为负(刹车)，有gain时翻转为正(释放)！**
**gain向量把action从"抑制"状态翻转到"释放"状态。**

### 核心结果5: D_no_gain vs D_pre的比例

Qwen3: D_no_gain/D_pre ≈ 2.4/30 = 0.08 (压缩92%)
GLM4: D_no_gain/D_pre ≈ 1.5/5 = 0.30 (压缩70%)

**归一化压缩了大量D信息。gain向量把压缩后的D重新放大。**
**Qwen3的归一化压缩更严重(92%)，gain放大更多。**

### 机制公式精确化

D_post = <h_normed * g, w_D> = <h_normed, g⊙w_D>

由于cos(w_D, g⊙w_D) ≈ 0.99:
D_post ≈ ||g⊙w_D|| / ||w_D|| * D_no_gain

即: D_post ≈ gain_ratio * D_no_gain

其中:
- gain_ratio = ||g⊙w_D|| / ||w_D||
- Qwen3: gain_ratio ≈ 2.7-3.5
- GLM4: gain_ratio ≈ 3.2-3.6

但这个近似对emotion和action不完全成立:
- emotion: D_no_gain=2.17, D_with_gain=3.01, ratio=1.39 (远小于gain_ratio)
- action: D_no_gain=-0.73, D_with_gain=2.10, 符号翻转!

**这说明gain不是简单均匀放大，而是类别特异的门控机制。**

### 对用户Phase 497评价的修正

用户说"RMSNorm不是简单放大器，而是非线性重映射"——部分正确:
- 正确: RMSNorm确实重映射了pre-norm空间到post-norm空间
- 需要修正: gain向量g本身几乎不改变w_D方向(cos>0.99)，但它与归一化缩放结合后，产生了类别特异的D变化
- 关键洞察: 不是gain改变了方向，而是**归一化压缩+gain放大的组合**产生了非均匀的类别D效应

更准确的机制描述:
1. 归一化(h/rms)把D压缩到原来的8-30%
2. gain向量(g⊙w_D)把压缩后的有效读出方向范数放大2.7-3.6倍
3. 但这两个操作不是简单的"压缩再放大"，因为:
   - 归一化压缩了hidden state的所有方向
   - gain放大了w_D方向但不放大其他方向
   - 组合效果: 只有w_D方向的信号被放大回来，其他方向被压制
4. 对action: D_no_gain为负是因为归一化后竞争token的logit > target logit
5. gain放大后，target方向被选择性增强，使D翻转为正

### 问题与硬伤

1. 为什么gain向量>1的维度占99%+? 这意味着几乎所有维度都被放大
2. gain的类别差异来源: 是gain本身的结构差异，还是hidden state与gain的交互?
3. action的D_no_gain为负的根本原因: 归一化后竞争token优势更大
4. emotion的gain_eff最小，是否意味着emotion在模型中的语义门控最弱?
5. DS7B仍无法验证

### 理论研究进展

Phase 498的最大突破是发现了语言编码的**增益门控机制**:

语言编码不是简单的"方向+范数"系统，而是:
1. Residual stream携带潜在语义(pre-norm D很大: 25-35)
2. 归一化压缩所有语义到8-30%
3. Gain向量选择性放大DCF读出方向(放大2.7-3.6倍)
4. 组合效果: 类别语义通过gain门控被选择性释放

这可以类比为:
- 归一化 = 均匀噪声门(压制所有信号)
- Gain向量 = 语义选择器(只放行特定方向的信号)
- 组合 = 语义门控机制(只允许语义方向通过)

一句话: **语言概念的输出不是方向写入的结果，而是增益门控选择性放大的结果。**
"""

with open(MEMO_FILE, 'a', encoding='utf-8') as f:
    f.write(memo_text)

print(f"MEMO R2 appended at {timestamp}")
