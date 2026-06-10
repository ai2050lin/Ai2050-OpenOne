"""更新MEMO: Phase 437b R2 确认结果"""
import os
from datetime import datetime

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

content = f"""

## Phase 437b: 扩展属性-类别中介确认 (R2) [{timestamp}]

### 目标
用更多对象(每类4个)确认Phase 437的核心发现

### 关键结果 (8对象平均, alpha=2.0)

| 模型 | fruit->animal avg_med | tool->vehicle avg_med | 对象数 |
|------|----------------------|----------------------|--------|
| Qwen3 | **+4.24** | **+5.29** | 8(全正) |
| GLM4 | **-0.03** | **-0.04** | 8(近零) |

### R2详细结果

#### Qwen3 (全正mediation)
| 对象 | fruit->animal med(a2) |
|------|----------------------|
| apple | 4.68 |
| orange | 5.06 |
| lemon | 3.32 |
| grape | 3.92 |
| knife | 6.33 |
| hammer | 3.30 |
| spoon | 6.38 |
| axe | 5.16 |

#### GLM4 (近零mediation)
| 对象 | fruit->animal med(a2) |
|------|----------------------|
| apple | -0.06 |
| orange | -0.06 |
| lemon | -0.05 |
| grape | +0.06 |
| knife | -0.08 |
| hammer | +0.06 |
| spoon | -0.22 |
| axe | +0.11 |

### 结论确认
1. Qwen3的类别-属性中介效应稳健(8/8对象全部正，avg=4-5)
2. GLM4的属性-类别独立性稳健(8/8对象mediation近零，avg=-0.03)
3. **两模型差异>100倍，不是噪声**
4. 这不是8bit量化问题(bf16结果一致)

### 理论意义
语言编码的数学结构不是唯一的！
- Qwen3采用"类别→属性"层级编码
- GLM4采用"对象→属性"直接绑定

这挑战了"语言有统一数学结构"的假设。
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {timestamp}")
