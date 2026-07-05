# Phase 941 三模型颜色编码特征图谱

生成时间：2026-07-05 06:32:57

## 1. 本轮目标

本轮任务不是只验证模型能回答颜色，而是把颜色相关的内部编码特征压缩成可预测、可验证、可复用的图谱。测试对象是 qwen3、deepseek7b、glm4 三个本地模型。每个模型使用相同的颜色对象数据、相同的提示模板、相同的通道贡献公式和相同的 top 通道干预流程。

## 2. 核心机制公式

对一个颜色标签 \(c\)，先构造输出读出方向：

$$
d_c = W_U[t_c] - \frac{1}{|C|-1}\sum_{c' \ne c} W_U[t_{c'}]
$$

其中 \(W_U[t_c]\) 是颜色词 token 的输出权重行。再把 MLP down projection 的第 \(j\) 个通道投影到这个方向上：

$$
r_{\ell,j,c} = d_c^\top W^{down}_{\ell,:,j}
$$

样本 \(x\) 在该通道上的颜色贡献为：

$$
K_{\ell,j,c}(x)=a_{\ell,j}(x)\cdot r_{\ell,j,c}
$$

最终通道分数为：

$$
S_{\ell,j,c}=\overline{K}_{c}+(\overline{K}_{c}-\overline{K}_{\neg c})+0.05\overline{|K|}_{c}+0.02N_{obj}+0.01N_{tpl}
$$

通俗说：一个通道要成为颜色通道，必须同时满足四件事：它对目标颜色有正贡献；它比其他颜色更偏向目标颜色；贡献不是偶然的小数值；它能覆盖多个对象和多个模板。

## 3. 全量测试规模

| 模型 | 样本数 | 通道统计 | 干预记录 | Top1 | Top10 | 平均 margin | 主要层位 |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 184 | 20000 | 384 | 67 | 115 | 1.271 | L27:3, L35:8 |
| deepseek7b | 184 | 20000 | 384 | 17 | 66 | -1.130 | L27:11 |
| glm4 | 184 | 20000 | 384 | 98 | 143 | 1.242 | L30:7, L39:4 |

## 4. 三模型颜色图谱

表中 `0x` 表示把 top 通道组置零后的 margin 变化，通常越负说明该通道组越必要；`2x` 表示放大到 2 倍后的 margin 变化，通常越正说明该通道组越有推动作用。

| 颜色 | qwen3 | deepseek7b | glm4 |
|---|---|---|---|
| black | L35 C284 稳 score=2.425 0x=-0.055 2x=0.039 weak_or_mixed | L27 C15305 稳 score=6.870 0x=-0.156 2x=0.188 directional | L30 C9374 稳 score=1.045 0x=-0.266 2x=0.250 strong_directional |
| blue | L27 C774 稳 score=1.169 0x=-0.531 2x=0.062 directional | L27 C15305 稳 score=14.086 0x=0.016 2x=-0.102 weak_or_mixed | L30 C11128 稳 score=1.475 0x=-0.484 2x=0.141 directional |
| brown | L35 C188 稳 score=2.103 0x=-0.083 2x=0.052 weak_or_mixed | L27 C16317 稳 score=6.807 0x=-0.188 2x=-0.055 weak_or_mixed | L39 C5902 稳 score=0.579 0x=-0.109 2x=0.062 weak_or_mixed |
| gray | L35 C689 稳 score=1.737 0x=-0.133 2x=0.125 directional | L27 C8030 稳 score=6.869 0x=-0.141 2x=0.484 directional | L39 C200 稳 score=0.353 0x=-0.133 2x=0.188 directional |
| green | L27 C774 稳 score=1.522 0x=-0.195 2x=0.258 directional | L27 C15791 稳 score=9.307 0x=0.023 2x=0.109 weak_or_mixed | L30 C1775 稳 score=4.984 0x=-0.891 2x=0.461 strong_directional |
| orange | L35 C2290 稳 score=1.741 0x=-0.055 2x=-0.031 weak_or_mixed | L27 C1645 稳 score=4.789 0x=0.031 2x=0.094 weak_or_mixed | L30 C11128 稳 score=2.502 0x=-0.273 2x=0.070 directional |
| purple | L35 C501 稳 score=0.928 0x=-0.016 2x=0.203 weak_or_mixed | L27 C5703 稳 score=6.356 0x=0.102 2x=0.133 weak_or_mixed | L39 C5902 稳 score=0.557 0x=-0.312 2x=0.172 directional |
| red | L35 C284 稳 score=6.406 0x=-0.336 2x=0.344 strong_directional | L27 C15305 稳 score=27.489 0x=-0.083 2x=-0.208 weak_or_mixed | L30 C7088 稳 score=3.391 0x=-0.422 2x=0.375 strong_directional |
| silver | L35 C3004 稳 score=3.985 0x=-0.336 2x=0.297 strong_directional | L27 C5656 稳 score=8.488 0x=0.068 2x=0.141 weak_or_mixed | L39 C12772 稳 score=0.726 0x=-0.031 2x=0.035 weak_or_mixed |
| white | L35 C1158 稳 score=1.565 0x=-0.172 2x=0.156 directional | L27 C15305 稳 score=7.004 0x=0.105 2x=0.160 weak_or_mixed | L30 C9374 稳 score=1.428 0x=-0.391 2x=0.164 directional |
| yellow | L27 C7176 稳 score=1.165 0x=0.047 2x=0.031 weak_or_mixed | L27 C9763 稳 score=3.402 0x=-0.141 2x=0.125 directional | L30 C11128 稳 score=1.464 0x=-0.901 2x=0.646 strong_directional |

## 5. 共享通道和复用结构

### qwen3

- L27 C774：blue, green
- L35 C284：black, red

### deepseek7b

- L27 C15305：black, blue, red, white

### glm4

- L30 C11128：blue, orange, yellow
- L30 C9374：black, white
- L39 C5902：brown, purple

## 6. 关键结论

1. qwen3 的颜色编码主要落在后段 L27/L35，红色、银色、黑色最清楚。红色 L35 C284 同时也是黑色强候选，说明单通道可能不是纯颜色名，而是承载一组颜色/材质/对象联动方向。
2. deepseek7b 的候选分数很高，但颜色回答 baseline 较弱，很多颜色不是 top1。因此它的读出通道很强，因果干预却更混合，说明读出方向和实际生成决策之间还有竞争项没有被当前公式完全捕获。
3. glm4 的图谱最像清晰的可控机制：置零 top 通道通常降低 margin，放大 top 通道通常提高 margin。green、red、yellow 的干预信号尤其明显。
4. 三个模型不存在可直接比较的相同 channel id，因为架构和训练不同；真正可复用的是角色层级：颜色特征通常集中在中后层 MLP 通道，并通过输出词向量方向形成读出贡献。

## 7. 当前缺口

当前图谱已经能定位颜色编码候选，但还不是完整编码机制。还缺三块：

- 反事实对象控制：例如 red apple、green apple、yellow apple，区分对象知识和颜色属性。
- 跨 token 位置追踪：当前主要看最后 token，需要追踪颜色信息在前文对象 token、属性 token、答案 token 之间如何迁移。
- 竞争项建模：特别是 deepseek7b，读出贡献强但生成结果弱，说明还有 blocker/suppressor 或其他候选词竞争。

## 8. 下一步算法

下一步应进入 Phase 942：颜色反事实闭环。做法是固定对象、替换颜色、固定模板，比较同一对象在不同颜色属性下的通道变化，并对当前 top 通道做正负向干预。

目标公式：

$$
\Delta K_{\ell,j,c}(x_{object,color_1},x_{object,color_2}) = K_{\ell,j,c}(x_{object,color_1}) - K_{\ell,j,c}(x_{object,color_2})
$$

如果一个通道是真正的颜色编码通道，它应该随颜色改变而改变，而不是只随对象改变而改变。

