# 2026-04-09 定向瓶颈升级：P_personal 与 M_epistemic

## 1. 这轮新增了什么

为了把统一语言理论从“大状态桶”继续压缩成更自然的状态变量，本轮新增了三项工作：

- `stage578`：篇章级人称共指链实测
- `stage579`：认识论副词与不确定性实测
- `stage580`：定向瓶颈评估与状态方程升级

对应脚本与结果：

- [stage578_personal_coreference_discourse_empirical.py](/d:/develop/TransformerLens-main/tests/codex/stage578_personal_coreference_discourse_empirical.py)
- [stage579_epistemic_uncertainty_empirical.py](/d:/develop/TransformerLens-main/tests/codex/stage579_epistemic_uncertainty_empirical.py)
- [stage580_targeted_bottleneck_assessment.py](/d:/develop/TransformerLens-main/tests/codex/stage580_targeted_bottleneck_assessment.py)
- [stage578 summary.json](/d:/develop/TransformerLens-main/tests/codex_temp/stage578_personal_coreference_discourse_empirical_20260409/summary.json)
- [stage579 summary.json](/d:/develop/TransformerLens-main/tests/codex_temp/stage579_epistemic_uncertainty_empirical_20260409/summary.json)
- [stage580 summary.json](/d:/develop/TransformerLens-main/tests/codex_temp/stage580_targeted_bottleneck_assessment_20260409/summary.json)

## 2. 关键结果

### 2.1 P_personal 不再能当成单一状态

简单人称共指的平均准确率是：

```text
simple_personal_mean = 0.75
```

篇章级人称共指链的平均准确率是：

```text
discourse_personal_mean = 0.625
personal_gap_mean = -0.125
```

更关键的是四模型分化非常大：

- `Qwen3`：简单共指 `1.0`，篇章链 `0.0`
- `DeepSeek7B`：简单共指 `0.5`，篇章链 `1.0`
- `GLM4`：简单共指 `1.0`，篇章链 `0.5`
- `Gemma4`：简单共指 `0.5`，篇章链 `1.0`

这说明当前的 `P_personal` 混合了至少两种不同机制：

- 句内局部绑定
- 跨句链路跟踪

所以更合理的升级不是继续使用单个 `P_personal`，而是改成：

```text
P_local
P_discourse
```

### 2.2 M_epistemic 更像判断状态，不像普通修饰

前一轮“认识论副词”粗粒度显式标签任务平均准确率是：

```text
simple_epistemic_mean = 0.375
```

这轮“认识论副词 + 不确定性判断”的平均准确率提升到：

```text
targeted_epistemic_mean = 0.71875
epistemic_gap_mean = +0.34375
```

而且内部结构也很清楚：

- `epistemic_force` 平均大致中等
- `epistemic_entailment` 明显更强

这说明模型对“probably / might / certainly / definitely”最稳的能力，不是给它贴“修饰 statement”这种标签，而是：

- 判断是否表达了确定性
- 判断是否构成保证性推断
- 判断真假承诺的强弱

所以 `M_epistemic` 更像和确定性判断共同构成一个耦合结构：

```text
M_epistemic_scope + Q_certainty
```

而不是一个孤立的副词修饰通道。

## 3. 对统一状态方程的升级

上一轮的状态方程是：

```text
S_t,l = (O_t,l, A_t,l, R_t,l, P_personal,l, P_reflexive,l, P_demonstrative,l,
         M_manner,l, M_epistemic,l, M_degree,l, M_frequency,l,
         Q_t,l, G_t,l, C_t,l)
```

这一轮更合理的候选升级是：

```text
S_t,l = (O_t,l, A_t,l, R_t,l,
         P_local,l, P_discourse,l, P_reflexive,l, P_demonstrative,l,
         M_manner,l, M_epistemic_scope,l, M_degree,l, M_frequency,l,
         Q_certainty,l, Q_reasoning,l,
         G_t,l, C_t,l)
```

它背后的含义是：

- `P_local`：句内局部绑定
- `P_discourse`：跨句链路跟踪
- `M_epistemic_scope`：认识论副词的显式范围信息
- `Q_certainty`：命题确定性/真假承诺状态

## 4. 为什么这更接近第一性原理

第一性原理理论不是“能解释很多现象”就够了，而是要不断把大兜底项压成自然状态变量。

这一轮最重要的推进就在这里：

- `P_personal` 被迫拆成更自然的两层结构
- `M_epistemic` 从“副词修饰”被逼近为“确定性判断耦合状态”

这说明我们不是在给现象重新命名，而是在逼近一套更小、更自然、更可组合的生成变量。

## 5. 如何解释大脑编码机制

如果把这轮结果往脑侧翻译，最自然的解释不是“代词区”“副词区”。

更像是：

- 人称共指至少有两类回路
  - 局部语法绑定回路
  - 跨句工作记忆/篇章链路回路
- 认识论副词不是普通修饰，而是和确定性评估回路耦合

换句话说，大脑侧更可能存在：

- 负责快速句内绑定的局部回路
- 负责追踪对象链和指代链的延时状态回路
- 负责真假承诺与不确定性评估的判断回路

这和“单块区域对应单类词”的想法很不一样，更接近分布式群体状态更新。

## 6. 严格审视：硬伤与瓶颈

这轮结果虽然很有推进，但还有几个硬伤。

- `stage578` 的样本仍然偏少，只有 6 个篇章级样本，不足以下定律。
- `Qwen3` 在篇章链任务上从 `1.0` 直接掉到 `0.0`，说明任务构造很可能击中了它的偏置，也可能提示当前评分口径仍有脆弱性。
- `stage579` 虽然支持 `Q_certainty`，但还没有把它写成逐层闭式更新方程。
- 现在仍然主要是行为层和层间 margin 证据，离最小因果神经元回路还差一步。

## 7. 下一阶段大任务

如果目标是走向第一性原理，下一阶段不该继续平均铺开，而应该集中做三件事：

- `P_discourse` 大任务
  - 做更长的多句链路
  - 引入三实体、四实体和干扰句
  - 直接寻找跨句链路的最小因果回路
- `Q_certainty` 大任务
  - 做 certainty / probability / possibility / contradiction 的层间动力学
  - 把不确定性判断写成状态更新方程
- 神经元回路映射大任务
  - 把 `P_local / P_discourse / Q_certainty` 从状态变量真正投影回最小因果模块

如果这三条线能打通，当前理论就会更接近：

**“语言不是一堆词类规则，而是一套由对象、关系、绑定、链路和确定性判断共同构成的状态更新系统。”**
