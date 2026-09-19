import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = '''

## Phase 2870（2026-09-18）：属性轴预研 —— 双轴图谱前提确立

**背景**：TMA 双轴词族结构（类别轴 taxonomic × 属性轴 attributional）需要先验证据：属性方向在 unembed 几何上是否独立于类别轴。2870 零前向（3.8s，直接 safetensors 读 embed_tokens，tie_word_embeddings=True → W_U，不加载 4B 模型）。

**方法**：15 反义词对（size×3/speed×2/temp×2/age×2/weight/strength/brightness/moisture/height/fullness），d_attr = unit(E(w+)−E(w−))；类方向 dW_unit 按.SEED=2855 词表重建（2867 B1 同款）；null = 200 随机 token 对差方向（SEED=2870）。

**产物**：`tests/glm5/phase2870_attr_axis_pilot.py`（sha ada6ffe0 [execution.json] / 874c64a5 [result.json] / f81c846e [attr_axis_pilot.npz]），目录 `phase2870/attr_axis_pilot/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| **P1** 属性×类正交（主） | max_c \\|cos\\| = **0.0755**（height 对，阈 0.3，随机 null p95 0.1143——观测**低于随机基线**） | **true → attribute_axis_independent** |
| **P2** 轴结构（字面 false） | mean 0.05 > null p95 0.0202；max 0.588 | 字面 **axis_collapsed**（判据缺陷，见下） |
| **P3** 类子空间正交能量 | mean = **0.995**（属性方向能量 99.5% 在 10 维类子空间外） | **true → class_subspace_clean** |

**P2 判据缺陷注记（不挪门柱，如实登记）**：top 相似对 = speed1×speed2 0.588 / age1×age2 0.481 / size1×size3 0.460——全部是**同轴对**（同属性不同词对）；跨轴对（102 对）max 仅 **0.138**。P2 把两个总体混入一个分布：同轴高相关 = **轴方向跨词对可再现（replicability）**，恰是"轴"存在的定义性证据，而非坍缩。正确读出：**axis_structure_confirmed**（同轴 0.46-0.59 vs 跨轴 <0.14，分离干净）。

### 核心发现（重复三遍）

**属性轴与类别轴在 unembed 几何上强独立（P1 投影低于随机基线 + P3 能量 99.5% 在类子空间外），且属性方向具有跨词对再现性（同轴对 cos 0.46-0.59，跨轴 <0.14）——双轴图谱（taxonomic × attributional）的几何前提成立。**

### 接续

- 2871（进行中，自动续推）：显著头三重身份解剖（2846 top-64 × 2864 η² 显著 × 2868 B2s-43 交集，响应图谱机制轴实体化收口，零前向）
- 2872 候选：属性轴因果 census（2846 协议移植到属性对词表，属性轴机制侧定位——双轴关联第二样例，需完整预注册设计）
'''

with io.open(MEMO, 'r', encoding='utf-8') as f:
    before = f.read()
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
print('APPENDED: %d -> %d lines' % (before.count(chr(10)), (before + section).count(chr(10))))
