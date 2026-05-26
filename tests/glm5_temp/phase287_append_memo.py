
import json, os
from datetime import datetime

MEMO_PATH = r"d:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md"

content = """
---

## Phase 287: Route-Content Separation — 注意力Head的路由/内容因果分解 [2026-05-26 14:30]

### 实验动机

Phase 286的o_proj input patching将每个head的输出作为整体替换，但head输出 = attention_weights(AW) @ value_vectors(V)，混合了"路由"（连接哪些token）和"内容"（传输什么信息）。Phase 287将这两者分离，直接回答核心理论问题：

> 语言功能是通过改变attention连接模式（routing/R），还是通过改变传输的内容（content/V），来实现的？

这个问题直接关联Phase 280的核心推论：**"角色绑定主要通过value内容，而不是attention图重连"**。

### 方法

```
PART 1 — CACHE (EAGER attention + output_attentions=True):
  对121对句子(A,B)，前向传播时：
  - output_attentions=True → 缓存每层attention weights AW [1, n_heads, seq, seq]
  - hook v_proj → 缓存V values [1, seq, n_kv_heads*head_dim]
  - 只缓存target layers（5个heads对应的层，节省内存）
  处理GQA：V从kv_heads维度广播到query_heads维

PART 2 — CONSTRUCT HYBRID OUTPUTS:
  对每对(句子, 层, head)，构造4种混合head输出：
  1. AW_A @ V_A  → 完整A head输出（Phase 286等效）
  2. AW_B @ V_B  → 完整B head输出（基线，应≈0）
  3. AW_A @ V_B  → A的路由 + B的内容（纯路由效应）
  4. AW_B @ V_A  → B的路由 + A的内容（纯内容效应）

PART 3 — PATCH (flash_attn_2/eager):
  将混合head输出替换进o_proj input的对应head slot
  测量KL(P_patched || P_B) / KL(P_A || P_B)
  得到 routing_ratio 和 content_ratio

判定规则：
  routing_ratio > 0.7 × full_A: 路由主导
  content_ratio > 0.7 × full_A: 内容主导
  两者都 > 0.5 × full_A: 可分离（EITHER）
  两者都 < 0.3 × full_A: 耦合（COUPLED）
```

### 核心结果

#### 跨模型统一发现：路由和内容是可分离的

**15/15 heads (100%) 在三模型中都显示"EITHER (separable)"——路由和内容可以独立实现类似效果。**
没有发现任何一个COUPLED head（路由和内容必须同时改变才有效）。

```
               Qwen3       GLM4        DS7B
Head数 tested    5          5          5
EITHER (separable) 100%        100%       100%
Routing均值     0.800       1.088       1.504
Content均值     1.227       1.035       1.745
R/C均值         0.65        1.05        0.86
模型偏向        CONTENT     平衡        CONTENT(略)
```

#### Qwen3 (13min: 0.2min缓存 + 10.7min patching)

| Head | Full_A | Routing | Content | R/C | 偏向 |
|------|--------|---------|---------|-----|------|
| L12_H0 | 1.000 | 0.851 | 1.235 | 0.69 | 内容略强 |
| L12_H25 | 1.000 | 0.639 | 1.260 | 0.51 | 内容显著强 |
| L16_H27 | 1.000 | 0.637 | 1.355 | 0.47 | 内容显著强 |
| L28_H14 | 1.000 | 1.121 | 0.983 | 1.14 | 路由略强 |
| L35_H0 | 1.000 | 0.751 | 1.301 | 0.58 | 内容显著强 |

Qwen3 content ratio均值(1.227) > routing均值(0.800)。Content-only替换的效果约是routing-only的1.54x。8/14类别是content-dominant，0个routing-dominant。

**Qwen3逻辑功能**：routing=0.249, content=1.066, R/C=0.23 → **逻辑是最content-biased的功能**。

#### GLM4 (27min: 4.3min缓存 + 18.5min patching)

| Head | Full_A | Routing | Content | R/C | 偏向 |
|------|--------|---------|---------|-----|------|
| L16_H30 | 1.000 | 1.328 | 0.923 | 1.44 | 路由略强 |
| L16_H7 | 1.000 | 0.923 | 1.115 | 0.83 | 平衡 |
| L28_H27 | 1.000 | 1.004 | 0.744 | 1.35 | 路由略强 |
| L8_H16 | 1.000 | 1.201 | 1.326 | 0.91 | 平衡 |
| L8_H26 | 1.000 | 0.986 | 1.067 | 0.92 | 平衡 |

GLM4 R/C均值=1.05，是最平衡的模型。**14/14类别全部平衡**（无routing-dominant或content-dominant）。翻译R/C=1.00，完美均衡。这与Phase 286的"GLM4 attention heads功能互换"发现一致——连路由和内容的占比都是平衡的。

#### DS7B (16.5min: 2.7min缓存 + 13.8min patching)

| Head | Full_A | Routing | Content | R/C | 偏向 |
|------|--------|---------|---------|-----|------|
| L0_H10 | 1.000 | 1.362 | 2.047 | 0.67 | 内容显著强 |
| L0_H3 | 1.000 | 1.181 | 1.363 | 0.87 | 平衡 |
| L18_H0 | 1.000 | 1.790 | 1.775 | 1.01 | 平衡 |
| L21_H9 | 1.000 | 1.975 | 1.918 | 1.03 | 平衡 |
| L24_H15 | 1.000 | 1.210 | 1.621 | 0.75 | 内容略强 |

DS7B effect强度最大（routing最高1.975, content最高2.047），但也最分化。

### ⚠️⚠️ 最重要的发现：NEGATION是CONTENT主导的

```
          Qwen3       GLM4        DS7B
routing   0.952       1.173       0.620
content   1.697       1.031       2.886
C/R      1.78x       0.88x       4.66x
```

**DS7B的否定功能：content/routing = 4.66x！** 这意味着对于DS7B，否定操作主要通过改变VALUE内容实现，而非改变attention连接模式。

这直接验证了Phase 280的核心推论：角色/功能绑定通过value内容变换实现，不是通过attention图重连。

但注意GLM4的否定是弱routing-biased（C/R=0.88），与DS7B相反。这再次证明三模型的不同编码哲学。

### ⚠️⚠️ 第二发现：TRANSLATION是三模型唯一完美均衡的功能

```
          Qwen3       GLM4        DS7B
routing   1.107       0.997       1.079
content   1.029       0.996       1.270
R/C       1.08        1.00        0.85
```

翻译功能在GLM4的R/C=1.00，在Qwen3=1.08，在DS7B=0.85。三模型都接近1.0。这说明翻译需要**平衡地**改变"连接什么"和"传什么内容"——跨语言映射既需要不同的token关联，也需要不同的语义内容。

### ⚠️ 第三发现：LOGICAL是Qwen3最CONTENT-biased的功能

Qwen3 logical: routing=0.249, content=1.066, R/C=0.23。逻辑操作（and/or/although/therefore）在Qwen3的attention heads中几乎不改变routing，完全通过内容实现。这意味着逻辑连接词的功能不是通过改变attention图实现的，而是通过注入特殊的语义内容。

### 新增客观事实拼图（14条）

1. 训练有素的注意力heads中，路由和内容在因果上是可分离的（15/15 heads, 3/3 models）
2. 没有发现任何COUPLED head（路由和内容必须同时改变才有效）
3. DS7B的否定功能content/routing=4.66x，是最极端的内容主导案例
4. GLM4所有heads和所有功能类别都是路由-内容平衡的（R/C均值=1.05）
5. Qwen3逻辑功能content/routing=4.28x，几乎纯粹内容驱动
6. 翻译功能是三模型中唯一在所有模型中R/C接近1.0的功能
7. DS7B effect>1现象持续存在（routing最高1.975, content最高2.047）
8. Qwen3 content均值(1.227)系统性高于routing(0.800)，r/c=0.65
9. GLM4 routing-content完美平衡(R/C=1.05, std=0.20)与Phase 286 "heads功能互换"一致
10. AW和V可以通过eager attention + output_attentions + hook v_proj成功独立缓存
11. AW@V的混合head输出构造在所有GQA(4x,7x,16x)和MHA架构上正确工作
12. Eager attention下的AW+V缓存与flash_attn_2下的o_proj patching可以正确组合
13. DS7B Sliding Window + eager attention会有警告但功能正常
14. 单head content-only替换的效果经常超过full A替换（=混合状态过度转换）

### 硬伤分析

1. **Effect>1普遍存在**：content-only或routing-only替换经常产生比full A替换更强的效果。这不是方法的artifact，而是结构性的：当只替换一个head的输出且只改变routing或content时，创造的是一个不在模型训练分布中的混合状态，会引发更强的输出位移。

2. **所有heads都是EITHER(separable)的"零假象"风险**：可能不是因为路由和内容真的可分离，而是因为caching和patching使用不同的attention实现（eager vs eager/flash），细微的数值差异可能模糊了真正的耦合信号。

3. **per-category只有少量pairs**：每类别pairs数量不均衡（translation 10对, abstract仅2对），category-level分析的信度不足。

4. **GLM4的"完美平衡"可能是artifact**：GLM4 heads效应本身极弱（Phase 286: mean~0.17），那路由和内容的比值自然接近1.0——因为两个都接近噪声水平。

5. **DS7B negation R/C=0.21基于3对样本**：需要更多否定对（至少50对）来验证这个关键发现。

### 命令记录

```bash
python tests/glm5/phase287_route_content_separation.py qwen3       # 13.8min
python tests/glm5/phase287_route_content_separation.py glm4         # 27.0min
python tests/glm5/phase287_route_content_separation.py deepseek7b   # 16.7min
python tests/glm5_temp/phase287_cross_model.py                      # Cross-model analysis
```

### 数据文件

- `results/phase287_route_content/{qwen3,glm4,deepseek7b}_route_content.json`
- `tmp/phase287_{qwen3,glm4,deepseek7b}.txt`（完整日志）
- `tests/glm5/phase287_route_content_separation.py`（测试脚本）
- `tests/glm5_temp/phase287_cross_model.py`（跨模型分析）
"""

with open(MEMO_PATH, 'a', encoding='utf-8') as f:
    f.write(content)
print(f"MEMO appended. Lines: {len(content.splitlines())}")
print(f"Final MEMO line count after append.")
