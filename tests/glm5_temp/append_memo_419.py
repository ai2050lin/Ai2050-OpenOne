"""追加Phase 419结果到AGI_GLM5_MEMO.md"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import time

timestamp = time.strftime("%Y-%m-%d %H:%M")
memo_path = Path(r"d:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md")

entry = f"""

## Phase 419: 大规模随机Token轨道图 [{timestamp}]

### 目标
用大规模低频token(200个)测试3个属性(temperature/speed/size)的规则反转非对称性,
构建token→语义轨道映射, 验证Phase 416发现的架构差异.

### 实验设计

- **R1**: 每属性60个token(30 LOW定义+30 HIGH定义), 3属性, 180 tokens
- **R2确认**: 每属性100个token(50+50), 2属性(temperature/speed), 200 tokens, 不同seed
- **Token筛选**: 词表中段低频子词碎片, 3-6字符, 纯小写ASCII, 排除常见词/前后缀
- **条件**: L0(定义), L4(定义+反转), 计算asymmetry = up_mean - down_abs_mean
- **Bootstrap**: 2000次重采样, 95%置信区间
- **3模型**: Qwen3, GLM4, DS7B (BF16+device_map=auto)

### R1核心结果: 非对称性

| 属性 | Qwen3 (R1/R2) | GLM4 (R1/R2) | DS7B (R1/R2) |
|------|---------------|--------------|--------------|
| temperature | -0.423 / **-1.445** | +0.330 / **+0.425** | -0.286 / -0.193 |
| speed | -0.466 / **-0.824** | +0.590 / **+0.755** | -0.372 / -0.121 |
| size(R1) | **-1.177** | -0.218(不显著) | **-0.675** |

### R2 95% Bootstrap置信区间

| 属性 | Qwen3 CI | GLM4 CI | DS7B CI |
|------|----------|---------|---------|
| temperature | [-1.635, -1.250] | [+0.256, +0.607] | [-0.429, +0.038] |
| speed | [-1.033, -0.612] | [+0.551, +0.946] | [-0.344, +0.122] |

**asymmetry > 0 = up-reversal更容易; asymmetry < 0 = down-reversal更容易**

### R2反转成功率

| 属性 | Qwen3 up%/down% | GLM4 up%/down% | DS7B up%/down% |
|------|-----------------|----------------|----------------|
| temperature | 2%/76% | 38%/8% | 38%/50% |
| speed | 14%/66% | 92%/38% | 56%/72% |

### 关键发现

**1. 架构分叉: Qwen系和GLM4的结构性偏置方向完全相反**

- Qwen3: DOWN-reversal远更容易 (temperature asym=-1.445, down成功率76% vs up成功率2%)
- GLM4: UP-reversal远更容易 (speed asym=+0.755, up成功率92% vs down成功率38%)
- DS7B: 偏置弱且方向不稳定 (CI跨0)

**2. Size属性在Qwen系中DOWN偏置最强**

- Qwen3 size asym=-1.177 (所有属性中最强)
- DS7B size asym=-0.675
- GLM4 size asym=-0.218 (唯一不显著)

**3. R1→R2一致性: 架构分叉在不同token集上完全复现**

R1(seed=42)和R2(seed=123)使用完全不同的token集, 但:
- Qwen3方向一致: 温度从-0.423→-1.445, 速度从-0.466→-0.824
- GLM4方向一致: 温度从+0.330→+0.425, 速度从+0.590→+0.755
- DS7B偏置减弱但仍偏负

**4. 反转成功率揭示了更深层模式**

- Qwen3: temperature的up-reversal成功率仅2%! 几乎不可能把冷物体说成热
- GLM4: speed的up-reversal成功率92%! 把慢物体说成快非常容易
- DS7B: 更平衡, 但speed的down成功率(72%)仍高于up(56%)

**5. 定义效果跨模型一致**

所有模型的LOW定义使L0≈2.2-2.7, HIGH定义使L0≈3.3-4.7.
说明定义句子对随机token的属性锚定是有效的.

### 与Phase 416的对比

Phase 416-R2只测temperature, 发现:
- Qwen3 random asymmetry = -0.900
- GLM4 random asymmetry = +0.123
- DS7B random asymmetry = -0.546

Phase 419扩大到3属性+200 tokens后:
- Qwen3 temperature asymmetry = -1.445 (比R1更强!)
- GLM4 temperature asymmetry = +0.425 (比R1更强!)
- DS7B temperature asymmetry = -0.193 (减弱, CI接近0)

**方向完全一致, 且更大样本量使效应更显著.**

### 客观现象总结(不加理论)

1. 低频token的规则反转非对称性在Qwen3和GLM4中方向相反
2. Qwen3: 把HIGH对象反转为LOW更容易; GLM4: 把LOW对象反转为HIGH更容易
3. DS7B(Qwen2架构)偏置较弱, 方向更接近Qwen3但远不够显著
4. size属性的非对称性模式与temperature/speed不同
5. 反转成功率差异极大: Qwen3 temperature up仅2%, GLM4 speed up达92%
6. 不同随机token集(seed)产生相同方向的结果

### 问题与硬伤

1. **随机token仍有embedding偏见**: 即使200个token取平均, 仍不是零先验
   - 但不同seed一致的结果降低了个别token偏见的影响
   
2. **定义句子本身可能非对称**: "A X is cold" vs "A X is hot"
   - cold和hot在模型中的先验概率不同
   - 需要无定义的基线测试来分离定义效果

3. **L4规则格式可能影响结果**: QA格式在不同模型中的效果不同
   - GLM4(chat模型)可能对QA格式更敏感
   - 需要更多规则格式变体

4. **DS7B的偏置弱且不稳定**: CI跨0
   - 可能是DS7B(Qwen2架构+R1蒸馏)的混合特性
   - 需要更多Qwen2架构模型验证

5. **架构分叉的因果机制未明**: 是W_U? RMSNorm? MLP? 训练数据?
   - 只知道现象, 不知道原因

### 下一步任务

**Phase 420: 架构分叉机制定位**
- 直接对比Qwen3和GLM4的W_U logit基线
- 检查cold/hot, slow/fast, small/big候选词的无上下文logit
- 如果W_U基线就偏向cold → 解释了Qwen3的DOWN偏置
- 如果W_U基线偏向hot → 需要检查RMSNorm和残差流

**Phase 421: 无定义基线测试**
- 不加定义句子, 直接问"A X is", 测量随机token的默认level
- 这能分离"定义锚定效果"和"纯规则反转效果"

**Phase 422: 更多Qwen架构模型验证**
- 测试Qwen2-7B(非R1蒸馏)来验证DS7B的弱偏置是架构还是蒸馏的结果
- 测试Qwen3-8B来验证偏置是否随模型规模增长

### 测试脚本
`tests/glm5/phase419_token_trajectory_map.py`
`tests/glm5/phase419_r2_confirm.py`
### 结果文件
results/phase419_token_trajectory/qwen3_phase419.json (etc.)
results/phase419_token_trajectory/qwen3_phase419_r2.json (etc.)

"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(entry)

print(f"Phase 419 appended to MEMO at {timestamp}")
