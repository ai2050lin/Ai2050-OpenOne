"""更新MEMO: Phase 447 R2确认测试结果"""
import os
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M")

content = f"""

## Phase 447 R2: 确认测试结果 [2026-06-10 {now[11:]}]

### 确认1: 绑定态分解在不同alpha下的稳定性

**Qwen3: alpha越大，深层共享比越高（关键发现！）**

| alpha | fruit L0 shared | fruit Last shared | animal L0 | animal Last |
|---|---|---|---|---|
| 0.5 | 0.887 | 0.219 | 0.956 | 0.336 |
| 1.0 | 0.898 | 0.407 | 0.955 | 0.415 |
| 2.0 | 0.939 | 0.715 | 0.968 | 0.880 |

**解读:**
- 小alpha(0.5)时，深层shared_ratio降到0.22-0.34 → 对象差异明显
- 大alpha(2.0)时，深层shared_ratio保持0.72-0.88 → 大扰动强制走共享路径
- 这与Phase 440的"自然vs强制中介"一致：小alpha是自然机制，大alpha是强制机制

**GLM4: 共享比对alpha更鲁棒**

| alpha | fruit Last shared | animal Last shared | tool Last shared |
|---|---|---|---|
| 0.5 | 0.841 | 0.809 | 0.866 |
| 1.0 | 0.814 | 0.810 | 0.846 |
| 2.0 | 0.825 | 0.841 | 0.867 |

GLM4的shared_ratio在所有alpha下几乎不变(~0.81-0.87)。

### 确认2: SlotMediation深入分析

**Qwen3 SlotMediation:**
- apple: category SlotRange=5.72, color SlotRange=4.47, part SlotRange=3.33
- dog: category SlotRange=4.73, color SlotRange=4.37
- knife: category SlotRange=5.27

**GLM4 SlotMediation:**
- apple: category SlotRange=4.35, taste SlotRange=3.20
- knife: category SlotRange=5.96, part SlotRange=4.46

**关键发现:**
1. 所有模型中"category"属性组的SlotRange都最大(4.35-5.96)
   改变模板对"类别词logit"的影响远大于对"颜色/味道"的影响
2. "is_a"模板在所有模型中都产生最高的category logit
3. GLM4中所有logit值偏低（绝对值小），但相对模式与Qwen3类似

### 确认3: SwitchMediation方法对齐

**Qwen3 SwitchMediation (R2):**
- apple: alpha=0.5时cat_shift=-0.069, attr_med=+0.029 (弱)
- apple: alpha=2.0时cat_shift=-1.210, attr_med=+0.282 (中等)
- dog: alpha=2.0时cat_shift=-0.949, attr_med=-0.406 (负)

**GLM4 SwitchMediation (R2):**
- apple: alpha=0.5时cat_shift=-1.305, attr_med=+0.589 (中等正)
- apple: alpha=2.0时cat_shift=-0.564, attr_med=+0.837 (强正!)
- dog: alpha=0.5时cat_shift=-4.309, attr_med=-0.060 (弱负)
- dog: alpha=2.0时cat_shift=-2.073, attr_med=-0.841 (强负)

**关键修正:**
- GLM4在apple对象上有**强正SwitchMediation**(attr_med=+0.84)
- 但在dog对象上有**强负SwitchMediation**(attr_med=-0.84)
- 这说明**SwitchMediation在GLM4中是对象依赖的**，不是模型统一属性

**为什么与Phase 437/440矛盾?**
- Phase 437/440用"push类别到对立方向+测最后token的属性变化"
- Phase 447用"related vs unrelated属性差"测中介
- 两种方法的对象不同：Phase 437用8个对象平均，Phase 447只测3个
- **对象特异效应被平均掩盖了**

### 综合结论

1. **共享→私有化是所有模型的共性**，但速度不同(GLM4最慢，DS7B最快)
2. **Qwen3的共享比受alpha影响大** — 小alpha更对象私有，大alpha更共享
3. **GLM4的共享比对alpha鲁棒** — 始终保持高共享比
4. **SlotMediation(关系槽位)是所有模型中最强的中介** — 改变问题模板对属性的影响最大
5. **SwitchMediation在GLM4中是对象依赖的** — apple强正，dog强负
6. **之前"GLM4无SwitchMediation"的结论需要修正** — 不是没有，而是强对象依赖

### 更新的模型画像

**Qwen3:**
- 类别绑定态: 中等私有化(小alpha时深层shared~0.22-0.34)
- 类别中介: 弱SwitchMediation，弱BoostMediation
- SlotMediation: 强(category SlotRange~5.7)
- L0 attention: 强方向校准器(消融后dir_cos<0.15)
- 对象差异: 小 — 不同对象的中介行为较一致

**GLM4:**
- 类别绑定态: 高共享(深层shared~0.81-0.87，对alpha鲁棒)
- 类别中介: 强对象依赖(apple正,dog负)
- SlotMediation: 中等(category SlotRange~4.4-5.9)
- L0 attention: 熵控制器(消融后dir_cos>0.79但熵增大)
- 对象差异: 大 — 不同对象的中介行为截然不同

**DS7B:**
- 类别绑定态: 极端私有化(深层shared~0.04-0.35)
- 类别中介: 不稳定
- SlotMediation: 极强(SlotRange~5.1-6.9)
- L0 attention: 信号放大器(消融后norm增大但entropy下降)
- 对象差异: 不稳定
"""

memo_path = r"d:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md"
with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {now}")
