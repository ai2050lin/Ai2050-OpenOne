"""Phase 492 MEMO更新脚本"""
import json, time

def load_results(model):
    with open(f"results/glm5/phase492_{model}_r1.json", encoding="utf-8") as f:
        return json.load(f)

models = ["qwen3", "glm4", "deepseek7b"]
data = {m: load_results(m) for m in models}

memo = f"""
## Phase 492: 末端刹车-释放机制的注入修复、尺度校准与预测验证 [2026-06-14 19:22]

### 核心目标
解决Phase 491的最大技术瓶颈(inject失效)，完成跨模型尺度校准，验证support/inhibit ratio预测能力。

### ★★★ Exp1: inject失效已完全解决 — matched-norm inject = double_shared ★★★

**根本原因**: inject_direction使用单位向量(scale=1.0)，但实际shared_semantic投影范数为632-1205！

**4种注入方法对比 (关键数据)**:

| 模型-层 | dir_inject | matched_norm | sample_wise | comp_replace | ablate | double |
|---------|-----------|-------------|-------------|--------------|--------|--------|
| Qwen3 fruit L34 | -0.037 | **-23.363** | -23.363 | +0.260 | +23.363 | -23.363 |
| Qwen3 fruit L35 | +0.024 | **+18.618** | +18.618 | +0.863 | -18.618 | +18.618 |
| GLM4 fruit L38 | -0.067 | **-16.622** | -16.622 | -0.319 | +16.622 | -16.622 |
| GLM4 fruit L39 | -0.048 | **-10.365** | -10.365 | -0.801 | +10.365 | -10.365 |
| DS7B fruit L26 | -0.064 | **-77.173** | -77.173 | -3.143 | +77.173 | -77.173 |
| DS7B fruit L27 | +0.081 | **+122.847** | +122.847 | +0.126 | -122.847 | +122.847 |

**结论**:
- matched_norm_inject和sample_wise_inject都与double_shared完全一致（精度范围内）
- inject失效原因确认：**单位向量注入在高维空间中范数不足632-1205倍**
- component_replacement效果不一致，不推荐使用
- scaled_inject(s5/s10)虽线性但远不够大

---

### ★★★ Exp2: 尺度校准 — DS7B极端数值来自残差范数放大 ★★★

**关键尺度对比**:

| 模型 | L(n-2)残差范数 | L(n-1)残差范数 | proj_shared占比 | norm_delta(Ln-2) | norm_delta(Ln-1) | z_delta(Ln-2) | z_delta(Ln-1) |
|------|-------------|-------------|--------------|----------------|----------------|-------------|-------------|
| Qwen3 fruit | 650.9 | 796.2 | 97-99% | +0.036 | -0.023 | +1.44 | -0.98 |
| GLM4 fruit | 248.5 | 228.2 | 94-99% | +0.067 | +0.045 | +2.84 | +2.30 |
| DS7B fruit | 1208.4 | 1592.4 | 95-99% | +0.064 | -0.077 | +0.58 | -3.49 |

**关键发现**:
1. **proj_shared占残差范数94-99%** — shared_semantic是残差流的绝对主导成分！
2. **norm_delta跨模型可比** (0.02-0.08)，DS7B并不异常 — 说明DS7B的极端raw delta来自残差范数放大
3. **z_delta揭示真实控制强度**: DS7B L27 z=-3.49(最强), Qwen3 z=-0.98(中等), GLM4 z=+2.30(仍是刹车)
4. **GLM4残差范数最小(228-275)**，但z_delta最大(2.3-2.8) — 说明GLM4的相对控制强度不弱

---

### ★★★ Exp3: support/inhibit ratio预测 — 跨模型差异极大 ★★★

| 模型 | 预测准确率 | 说明 |
|------|----------|------|
| Qwen3 | **1/8 = 12.5%** | 7/8类别预测失败！ |
| GLM4 | **8/8 = 100%** | 全部正确：无反转 |
| DS7B | **8/8 = 100%** | 全部正确：全部反转 |

**Qwen3预测失败原因分析**:
- Qwen3 fruit L35: n_support=3(s_sum=-4.634), n_inhibit=5(i_sum=+22.874)
- net_release = |4.634| - |22.874| = -18.24 → 预测不反转
- 但ablate_shared = -18.548 → 实际反转(支撑)

**根本原因**: 8个SVD正交方向不能捕获完整的shared_semantic效应。Qwen3末层support方向效应分散(单个方向-0.2到-2.7)，被inhibit方向掩盖。但shared_semantic作为整体仍起支撑作用。

**DS7B预测成功原因**: support方向效应高度集中(单个方向-30到-85)，8个方向足够捕获。

**GLM4预测成功原因**: 所有类别末层都是inhibit主导，support极弱(0-1个方向)，简单判断即可。

**8类末层反转数据**:

Qwen3 (6/8反转, 2/8不反转):
- fruit: ablate_shared=-18.5 反转 ✓
- animal: ablate_shared=-22.3 反转 ✓
- clothing: ablate_shared=-25.8 反转 ✓
- food: ablate_shared=-18.8 反转 ✓
- vehicle: ablate_shared=-40.9 反转 ✓
- plant: ablate_shared=-12.7 反转 ✓
- tool: ablate_shared=-5.7 反转 ✓
- furniture: ablate_shared=-24.8 反转 ✓

**注意**: Qwen3全部8类都反转！之前只测3类(fruit/clothing/food)已见反转，现在8类全部确认。

GLM4 (0/8反转):
- 全部8类ablate_shared为正(3.0-10.6)，shared_semantic全部为刹车

DS7B (8/8反转):
- 全部8类ablate_shared为负(-67到-208)，shared_semantic全部为支撑

---

### ★★★ Exp4: 竞争类别末层控制 — 因果闭环成立但效应弱于shared_semantic ★★★

| 模型 | 目标→竞争 | ablate_comp | double_comp | reverse_comp | ablate_shared |
|------|----------|------------|------------|-------------|-------------|
| Qwen3 | fruit→food | -0.851 | +0.851 | -1.703 | -18.548 |
| Qwen3 | cloth→tool | -0.127 | +0.127 | -0.254 | -25.843 |
| GLM4 | fruit→food | +0.036 | -0.036 | +0.072 | +10.322 |
| GLM4 | cloth→tool | -0.132 | +0.132 | -0.264 | +3.962 |
| DS7B | fruit→food | -3.145 | +3.145 | -6.289 | -125.745 |
| DS7B | cloth→tool | -11.188 | +11.188 | -22.376 | -88.038 |

**关键发现**:
1. 竞争方向完美满足ablate/double/reverse对称性 — 是真实因果方向
2. 竞争效应远小于shared_semantic: DS7B 3-11 vs 88-125, Qwen3 0.1-0.8 vs 18-26
3. DS7B竞争效应最强(3-11), Qwen3次之(0.1-0.9), GLM4最弱(0.03-0.13)
4. 竞争方向消融→目标边界下降(负delta) → 竞争方向是支撑(对target而言)

---

### 5个核心客观结论

1. **inject失效已解决**: 根本原因是单位向量注入在高维空间中范数不足。matched_norm inject(scale=||proj_shared||)与double_shared完全一致。
2. **DS7B极端数值来自残差范数放大**: norm_delta跨模型可比(0.02-0.08), 但z_delta显示DS7B末层控制确实最强(z=-3.49 vs Qwen3 z=-0.98 vs GLM4 z=+2.30)
3. **shared_semantic占残差范数94-99%**: 它是残差流的绝对主导成分
4. **8类全局验证**: Qwen3全部8类末层反转, GLM4全部8类不反转, DS7B全部8类反转 — 末层机制是模型级策略
5. **support/inhibit ratio预测**: DS7B/GLM4 100%准确, 但Qwen3 12.5% — 因为8个SVD方向不能捕获Qwen3末层分散的support效应

### 硬伤与瓶颈

1. **Qwen3的8个SVD方向不能捕获完整support效应**: 需要更多方向(16/32)或直接用shared_semantic方向
2. **竞争类别效应远弱于shared_semantic**: 竞争控制是否真的重要？还是只是shared_semantic的附带效应？
3. **component_replacement效果不一致**: 需要更精确的替换方案
4. **GLM4残差范数异常小(228)**: 与Qwen3(650-796)和DS7B(1208-1592)差距大，可能与40层结构有关
5. **尚未测试跨层因果关系**: L(n-2)操作后L(n-1)如何响应？

### 下一步方向

1. 扩展SVD方向数(8→32)解决Qwen3预测失败
2. 跨层因果追踪: 在L(n-2)做操作后追踪L(n-1)的变化
3. 分析L(n-2)→L(n-1)的信息传递机制(attention pattern, MLP transformation)
4. 扩展到16类验证末层机制的普遍性
"""

# 追加到MEMO文件
with open("research/glm5/docs/AGI_GLM5_MEMO.md", "a", encoding="utf-8") as f:
    f.write(memo)

print(f"MEMO updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
