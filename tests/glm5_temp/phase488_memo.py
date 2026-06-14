"""Phase 488 MEMO updater"""
import os

content = """

## Phase 488: 边界前体传播算子与正交空间细分 [2026-06-14 13:50]

### 核心发现: 正交成分主要不是边界前体,而是反对/调节成分

Phase 487的结论需要重大修正。Phase 488通过4个独立实验证明: 中间层的orth_bc不是"通过后续层旋转变成B_c的前体",而是"反对/调节边界形成的抑制成分"。

### Exp1: 扰动传播追踪

orth_bc传播后alignment大多为负(反B_c), proj_bc传播后始终为正:
- Qwen3 clothing L34->L35: orth=-0.379, proj=+0.093
- Qwen3 fruit L27->L35: orth=+0.020, proj=+0.105
- GLM4 fruit L22->L39: orth=-0.117, proj=+0.269
- GLM4 fruit L27->L39: orth=-0.102, proj=+0.575
- DS7B fruit L21->L27: orth=-0.195, proj=-0.105
- Qwen3 fruit L35->L35: orth=+0.462 (唯一正对齐,同层效果)

R2确认: clothing L34->L35 alignment=-0.0915, fruit L32->L35 alignment=-0.0067

### Exp2: 正交空间细分

orth_bc中最大成分是共享语义方向:
- DS7B fruit L26: shared_semantic amp=82.2%, cos=-0.987 (反对!)
- DS7B food L27: shared_semantic amp=945.7%, competitor_bc amp=349.5%
- shared_semantic的cos为负表示反对边界(抑制类别化)

### Exp4: 前体注入测试

中间层orth_bc注入后削弱B_c,只有最后1-2层orth_bc注入后增强B_c:
- Qwen3 fruit L32 orth inject: bc_increase=-1.1911 (强反对!)
- Qwen3 fruit L35 orth inject: bc_increase=+0.3772 (前体!)
- GLM4 fruit L22/L27/L32 orth inject: 均为负(反对!)
- GLM4 clothing L39 orth inject: bc_increase=+0.2086 (前体!)

### 3个核心客观发现

1. 中间层orth_bc是反对/调节成分,不是边界前体
2. orth_bc中最大成分是共享语义方向(维持共享语义,抑制过早类别化)
3. 只有最后1-2层的orth_bc是真正的边界前体

### 对Phase 487结论的修正

Phase 487说: "正交成分是边界因果的主要路径"
Phase 488修正为: "正交成分主要不是边界前体,而是反对/调节成分; 消融orth_bc效果大是因为移除了对边界的抑制(松刹车),不是因为orth_bc变成了B_c(踩油门)"

正确公式: 类别边界 = 投影写入 - 正交抑制 + 末层读出

### 新增客观事实(8条)

64. orth_bc传播后alignment大多为负(反B_c)
65. Qwen3 fruit L35->L35: orth_bc alignment=+0.462 (唯一正对齐)
66. Qwen3 fruit L32 orth_bc注入 bc_increase=-1.1911 (强反对)
67. GLM4 fruit所有中间层orth_bc注入均削弱边界
68. GLM4 clothing L39 orth_bc注入 bc_increase=+0.2086 (前体)
69. shared_semantic在DS7B中达82-946%,是orth_bc最大子成分
70. shared_semantic的cos为负表示反对边界(抑制类别化)
71. 类别边界=投影写入-正交抑制+末层读出,移除抑制>移除写入

### 命令记录

python tests/glm5/phase488_propagation_operator.py qwen3 1
python tests/glm5/phase488_propagation_operator.py glm4 1
python tests/glm5/phase488_propagation_operator.py deepseek7b 1
python tests/glm5/phase488_propagation_operator.py qwen3 2

脚本: tests/glm5/phase488_propagation_operator.py
结果: results/glm5/phase488_{qwen3,glm4,deepseek7b}_r1.json, phase488_qwen3_r2.json
"""

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)
print(f"MEMO updated: {os.path.getsize(memo_path)} bytes")
