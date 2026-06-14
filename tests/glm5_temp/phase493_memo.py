"""Phase 493 MEMO update script"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d %H:%M")

memo_text = f"""
## Phase 493: 高维支撑子空间、跨层刹车释放传递与16类全局验证 [{now}]

### 核心实验

- Exp1: 高维SVD分解(k=8,16,32,64)解决Qwen3预测失败
- Exp2: 跨层因果追踪 — L(n-2)操作对L(n-1)shared投影的影响
- Exp3: 16类全局验证末端刹车-释放机制普遍性

---

### ★★★ Exp1关键发现: SVD维度不是问题, orth子空间方法本身有缺陷

**所有三模型k=8,16,32,64的net_release完全相同!**

```
Qwen3 fruit: k=8→net_release=-18.240, k=64→net_release=-18.240 (完全相同)
GLM4 fruit:  k=8→net_release=-6.253,  k=64→net_release=-6.253  (完全相同)
DS7B fruit:  k=8→net_release=+52.614, k=64→net_release=+52.614 (完全相同)
```

这意味着:
1. **SVD k=8已经捕获了orth子空间中所有有效方向**
2. **k=9到64的方向ablate效应<0.05, 全部被分类为neutral**
3. **Qwen3预测失败不是因为维度不够, 而是因为support信号不在orth子空间中**

Qwen3的support信号在哪里? 在shared_semantic方向本身! 这就是为什么ablate_shared正确预测8/8反转, 而orth子空间SVD预测1/8。

DS7B和GLM4的orth子空间能正确预测, 是因为它们的support/inhibit信号恰好集中在orth子空间中。

---

### ★★★ Exp2关键发现: 跨层shared投影传递有模型特异性

| 模型   | fruit slope | clothing slope | animal slope | food slope | 平均slope |
|--------|------------|---------------|-------------|-----------|----------|
| Qwen3  | 3.190      | 2.923         | 2.127       | 2.516     | 2.69     |
| GLM4   | 0.918      | 0.903         | 1.237       | 0.867     | 0.98     |
| DS7B   | 2.084      | 1.843         | 3.811       | 0.090     | 1.96     |

| 模型   | fruit corr | clothing corr | animal corr | food corr |
|--------|-----------|--------------|------------|----------|
| Qwen3  | 0.972     | 0.981        | 0.822      | 0.813    |
| GLM4   | 0.939     | 0.973        | 0.897      | 0.876    |
| DS7B   | 0.847     | 0.854        | 0.822      | 0.011    |

关键洞察:
1. **Qwen3 slope≈2-3**: L(n-1)的shared变化是L(n-2)的2-3倍 → 存在跨层放大
2. **GLM4 slope≈1**: L(n-1)的shared变化约等于L(n-2) → 无放大, 1:1传递
3. **DS7B slope高变**: food的corr=0.011, slope=0.09 → food跨层传递几乎断裂

这解释了为什么:
- Qwen3和DS7B末层反转(slope>1 → 放大导致释放)
- GLM4末层不反转(slope≈1 → 无放大, 刹车状态持续)
- DS7B food的跨层传递异常弱, 但末层仍反转 → DS7B末层释放不完全依赖L(n-2)

---

### ★★★ Exp3关键发现: 16类验证 — 末层反转是模型级策略但有例外

| 模型   | 反转率 | 前8类反转 | 后8类反转 | 例外类别 |
|--------|-------|----------|----------|---------|
| Qwen3  | 13/16 | 7/8      | 6/8      | vehicle, container, emotion |
| GLM4   | 0/16  | 0/8      | 0/8      | 无(全部不反转) |
| DS7B   | 15/16 | 8/8      | 7/8      | action |

z_delta统计:
```
Qwen3:  mean=-0.82, std=0.57, min=-2.17, max=+0.18
GLM4:   mean=+2.20, std=1.26, min=+0.37, max=+5.43
DS7B:   mean=-3.21, std=1.95, min=-8.05, max=+0.12
```

Qwen3的3个例外:
- **vehicle**: L34 ablate_shared = -6.60 (已经是负的! 不满足"L(n-2)刹车"前提)
- **container**: L35 ablate_shared = +0.95 (非常弱的正, 几乎中性)
- **emotion**: L35 ablate_shared = +4.07 (仍然正, 末层保持抑制)

DS7B的1个例外:
- **action**: L27 ablate_shared = +7.05 (末层微弱抑制, 接近中性)

这些例外很重要: **抽象概念(emotion)和动作概念(action)可能不需要末层释放**。

---

### 对用户Phase 492评价的验证

1. ✅ "inject失效是尺度问题" — 已在492确认
2. ✅ "DS7B极端数值来自残差范数放大" — 已在492确认
3. ✅ "8/8反转" — 493修正为13/16(Qwen3), 15/16(DS7B), 0/16(GLM4)
4. ⚠️ "8个SVD方向不够" — **493证明这个判断是错的!** k=64和k=8结果完全相同, 问题不是维度而是方法本身
5. ✅ "跨层因果还没有完成" — 493首次完成跨层传递测量

---

### 客观结论(不加理论总结)

1. orth子空间SVD分解的support/inhibit预测方法对Qwen3失效, 原因不是维度不足而是support信号不在orth子空间中
2. Qwen3和DS7B的跨层shared传递有放大(slope>1), GLM4无放大(slope≈1)
3. DS7B food的跨层传递几乎断裂(corr=0.011), 但末层仍反转
4. 16类验证: Qwen3 13/16, DS7B 15/16, GLM4 0/16
5. 例外类别(vehicle, container, emotion, action)揭示了刹车-释放机制的适用边界
6. Qwen3的vehicle在L(n-2)就是负的(没有刹车), emotion在L(n-1)仍为正(没有释放)

---

### 硬伤和瓶颈

1. **orth子空间方法根本不适合预测Qwen3末层反转** — support信号在shared方向中, 不在orth中
2. **跨层传递只是相关性, 不是因果性** — slope和corr只能说明L(n-2)和L(n-1)的shared投影有关联, 不能说明L(n-2)操作导致了L(n-1)的变化
3. **DS7B food跨层传递断裂是未解释的异常**
4. **例外的vehicle/emotion/action没有理论解释** — 为什么这些类别不遵循刹车-释放模式?
5. **GLM4所有z_delta>0, DS7B所有z_delta<0, Qwen3多数z_delta<0** — 但这已经是描述而非解释

---

### 下一步核心任务

1. **真正跨层因果干预**: 在L(n-2)修改hidden state, 让模型继续前向传播到L(n-1), 追踪L(n-1)的实际变化
2. **shared_semantic方向本身的support/inhibit分解**: 不用orth子空间, 而是分解shared方向的logit贡献
3. **例外类别的特殊机制**: 为什么vehicle没有L(n-2)刹车? 为什么emotion没有L(n-1)释放?
4. **Attention pattern分析**: L(n-2)→L(n-1)的attention head如何传递刹车信号
"""

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
with open(memo_path, "a", encoding="utf-8") as f:
    f.write(memo_text)

print(f"MEMO updated at {memo_path}")
print(f"Added {len(memo_text)} characters")
