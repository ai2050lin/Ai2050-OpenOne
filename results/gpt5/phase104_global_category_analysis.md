# Phase 104 Global Category Analysis

Generated: 2026-06-13 23:59:50

## Core Findings
- 全局图谱支持'类别=共享语义流形+类别边界残差+竞争释放'，但写入机制不是统一的单一模块。
- 跨三模型都为正的释放边包括: animal->clothing, clothing->furniture, tool->vehicle, fruit->animal, clothing->plant, vehicle->clothing, furniture->clothing, fruit->clothing, furniture->fruit。这些边更像稳定竞争骨架。
- Qwen3 的强释放边幅度最大，GLM4 幅度整体很小，DS7B 存在方向不干净和抑制性神经元问题。
- MLP 因果写入器只在局部类别中清晰出现: Qwen3 clothing、GLM4 fruit 最强；fruit/animal 等类别常表现为非 MLP 主导或反向。
- 类别最佳层位不是统一层: Qwen3 多在 L23-L34，GLM4 多在 L27-L39，DS7B 多在 L23-L27，说明边界有类别-模型特异发育时间。
- 关系槽位测试显示 B_c 读出跨 kind_of/used_for/found_in 基本不变，但 scale=1.0 可能过强，必须做小尺度复核。
- food->vehicle、animal->clothing 不是简单错误边，更可能由属性共享/压制释放产生；但 DS7B 的异常边不能作为干净证据。

## Universal Positive Competition Edges
- animal -> clothing: models=qwen3,glm4,deepseek7b, avg_delta=4.081, max_delta=9.294
- clothing -> furniture: models=qwen3,glm4,deepseek7b, avg_delta=3.347, max_delta=7.265
- tool -> vehicle: models=qwen3,glm4,deepseek7b, avg_delta=2.443, max_delta=6.806
- fruit -> animal: models=qwen3,glm4,deepseek7b, avg_delta=2.336, max_delta=4.36
- clothing -> plant: models=qwen3,glm4,deepseek7b, avg_delta=1.556, max_delta=2.204
- vehicle -> clothing: models=qwen3,glm4,deepseek7b, avg_delta=1.005, max_delta=2.525
- furniture -> clothing: models=qwen3,glm4,deepseek7b, avg_delta=0.986, max_delta=1.499
- fruit -> clothing: models=qwen3,glm4,deepseek7b, avg_delta=0.537, max_delta=0.805
- furniture -> fruit: models=qwen3,glm4,deepseek7b, avg_delta=0.404, max_delta=0.564

## Top Model Edges
### qwen3
- animal -> clothing: delta=9.294 (strong)
- clothing -> tool: delta=7.47 (strong)
- clothing -> furniture: delta=7.265 (strong)
- food -> vehicle: delta=6.19 (strong)
- food -> plant: delta=6.122 (strong)
- animal -> food: delta=6.019 (strong)
- fruit -> animal: delta=4.36 (medium)
- animal -> fruit: delta=2.895 (medium)
### glm4
- clothing -> plant: delta=1.184 (medium)
- food -> plant: delta=1.087 (medium)
- food -> fruit: delta=1.074 (medium)
- clothing -> furniture: delta=1.003 (medium)
- animal -> clothing: delta=0.745 (weak)
- plant -> clothing: delta=0.492 (weak)
- plant -> vehicle: delta=0.418 (weak)
- furniture -> vehicle: delta=0.34 (weak)
### deepseek7b
- tool -> vehicle: delta=6.806 (strong)
- fruit -> food: delta=5.619 (strong)
- animal -> food: delta=4.731 (medium)
- fruit -> vehicle: delta=4.457 (medium)
- tool -> furniture: delta=3.276 (medium)
- vehicle -> clothing: delta=2.525 (medium)
- fruit -> animal: delta=2.415 (medium)
- vehicle -> plant: delta=2.358 (medium)

## MLP Writer Map
### qwen3
- animal: non_mlp_or_opposed, L33, cos50=0.244, sig=980, k5_cos=-0.204
- clothing: mlp_causal_writer, L30, cos50=0.672, sig=39, k5_cos=0.962
- fruit: non_mlp_or_opposed, L32, cos50=0.287, sig=745, k5_cos=-0.294
### glm4
- animal: distributed_or_missing, L38, cos50=0.316, sig=3267, k5_cos=0.618
- clothing: non_mlp_or_opposed, L39, cos50=0.389, sig=651, k5_cos=-0.054
- fruit: mlp_causal_writer, L27, cos50=0.62, sig=13, k5_cos=0.924
### deepseek7b
- animal: non_mlp_or_opposed, L27, cos50=0.714, sig=243, k5_cos=-0.292
- clothing: mixed_or_unresolved, L23, cos50=0.468, sig=27, k5_cos=0.126
- fruit: mixed_or_unresolved, L26, cos50=0.381, sig=250, k5_cos=0.673

## Layer Map
- qwen3: fruit:L32, animal:L33, tool:L23, vehicle:L29, clothing:L30, furniture:L26, food:L34, plant:L28
- glm4: fruit:L27, animal:L38, tool:L27, vehicle:L29, clothing:L39, furniture:L34, food:L38, plant:L32
- deepseek7b: fruit:L26, animal:L27, tool:L26, vehicle:L26, clothing:L23, furniture:L25, food:L27, plant:L25

## Hard Limits
- 本轮没有重新运行模型，只整合 Phase 483/484 既有结果；结论是全局拼图，不是新因果实验。
- 类别只有 8 类，每类 8 个对象；足以看边界网络雏形，不足以证明完整语义大陆。
- DCF 词表仍可能造成候选集偏置，尤其 food->vehicle、animal->clothing 等异常边需要更宽属性词表复核。
- 关系不变性使用 scale=1.0 注入，可能覆盖关系模板差异，必须用更小 scale 做下一轮。
- MLP writer 只覆盖 fruit/animal/clothing 三类的 Phase 484 重构，其他五类仍缺少写入器级证据。

