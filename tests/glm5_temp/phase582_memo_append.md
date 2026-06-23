

## Phase 582: Relation Necessity, State-Bridge Failure Typing, and Parametric Category Judgment [2026-06-22 06:40]

### 本阶段目标

修复Phase 581的两个关键缺口 + 桥接到参数化知识：
1. 修复关系必要性审计（强制R1!=R2 => V1!=V2）
2. 组合失败分型（分类两跳失败的具体原因）
3. 参数化类别判断（水果/动物/天体归属判断）
4. 显式规则 vs 参数常识对比

### 生成脚本

```text
tests/glm5/phase582_relation_parametric.py
```

### 测试设计

```text
Part A: 关系必要性审计（修复版）
  - 强制同一对象不同关系对应不同值
  - 测量关系区分率（模型是否给不同关系不同答案）

Part B: 组合失败分型
  - 分解失败类型: success, C_wrong_cat, B_fail, V_copy_fail
  - C_wrong_cat: 两步都对但组合用了错误中间类别
  - B_fail: 第二步gold检索失败
  - V_copy_fail: 值复制失败

Part C: 参数化类别判断
  - 6类别 × 10对象 × 4语法模板 = 240样本
  - 正例: 苹果是不是水果？-> 是
  - 负例: 老虎是不是水果？-> 否
  - 模板: direct, formal, simple, negative

Part D: 显式规则 vs 参数常识
  - 对比有无显式规则的yes/no边际差异
```

### 执行命令

```text
python tests/glm5/phase582_relation_parametric.py qwen3 --smoke --hard-exit-after-model
python tests/glm5/phase582_relation_parametric.py glm4 --smoke --hard-exit-after-model
python tests/glm5/phase582_relation_parametric.py deepseek7b --smoke --hard-exit-after-model
python tests/glm5/phase582_relation_parametric.py qwen3 --hard-exit-after-model
python tests/glm5/phase582_relation_parametric.py glm4 --hard-exit-after-model
python tests/glm5/phase582_relation_parametric.py deepseek7b --hard-exit-after-model
```

### 运行时间

```text
qwen3:      0.76 min (主测试)
glm4:       15.24 min (主测试)
ds7b:       6.53 min (主测试)
```

### 客观结果

#### Part A: 关系必要性审计（修复版）

```text
模型      accuracy    discrimination    correct_discrim
Qwen3     0.875       0.750             0.750
GLM4      0.800       0.800             0.650
DS7B      0.750       0.550             0.500
```

核心发现1: 关系必要性审计修复成功！三模型都表现出关系区分能力。
- Qwen3最强(87.5%准确率, 75%区分率)
- DS7B最弱(75%准确率, 55%区分率)
- 但DS7B的correct_discrim只有50%，说明即使区分了关系，答案也不一定对

核心发现2: 关系变量确实被使用，但使用不稳定。
- 之前Phase 580的rel_only污染最弱是因为任务退化
- 现在强制R1!=R2=>V1!=V2后，模型必须使用关系才能答对
- DS7B只有55%区分率，说明它经常忽略关系

#### Part B: 组合失败分型

```text
模型      success    C_wrong_cat    B_fail    V_copy_fail
Qwen3     32/40(0.80)  8/40(0.20)     0/40(0)    0/40(0)
GLM4      34/40(0.85)  4/40(0.10)     2/40(0.05) 0/40(0)
DS7B      20/40(0.50) 12/40(0.30)     5/40(0.125) 3/40(0.075)
```

核心发现3（重大）: 组合失败的主要类型是C_wrong_cat——使用了错误的中间类别！
- Qwen3: 100%的失败都是C_wrong_cat
- GLM4: 66%的失败是C_wrong_cat，33%是B_fail
- DS7B: 60%是C_wrong_cat，25%是B_fail，15%是V_copy_fail

这说明：**组合推理的瓶颈不是状态传递失败，而是模型在两跳任务中使用了错误的中间类别。**

模型知道要找类别，但找错了类别——不是"找到了C但没传给第二步"，而是"找到了错误的C"。

#### Part C: 参数化类别判断

```text
模型      positive    negative    overall
Qwen3     1.000       0.692       0.846
GLM4      1.000       0.000       0.500
DS7B      1.000       0.008       0.504
```

核心发现4（重大）: 三模型对正例(是)判断100%正确，但对负例(否)判断差异巨大！
- Qwen3: 负例69.2%正确——能说"否"
- GLM4: 负例0%正确——总是说"是"（强yes-bias）
- DS7B: 负例0.8%正确——几乎总是说"是"（强yes-bias）

这说明GLM4和DS7B在参数化判断中有强烈的"是"偏置——无论问什么都倾向回答"是"。

按语法模板分（Qwen3）:
```text
direct:     0.900
formal:     0.933
simple:     0.917
negative:   0.633  (否定问法最差)
```

按类别分（Qwen3）:
```text
水果:     0.950
动物:     0.875
天体:     0.775
工具:     0.650  (工具判断最差)
家具:     0.900
交通工具: 0.925
```

核心发现5: Qwen3的否定问法("不是...吗？")准确率最低(63.3%)，说明否定极性处理是独立机制。

#### Part D: 显式规则 vs 参数常识

```text
模型      苹果/水果         老虎/水果         香蕉/水果         地球/水果
          param explicit    param explicit    param explicit    param explicit
Qwen3     是(T) 是(T)       否(T) 否(T)       是(T) 是(T)       否(T) 否(T)
GLM4      是(T) 是(T)       是(F) 是(F)       是(T) 是(T)       是(F) 是(F)
DS7B      是(T) 是(T)       是(F) 是(F)       是(T) 是(T)       是(F) 是(F)
```

核心发现6: GLM4和DS7B在参数化判断中把老虎和地球都判断为水果（说"是"），这是严重的yes-bias。
- Qwen3是唯一能正确说"否"的模型
- 显式规则对GLM4/DS7B也没帮助——即使给出"老虎不属于水果"的规则，它们仍然说"是"

margin_diff（显式-参数）:
- Qwen3: 全负，说明显式规则反而降低了yes-margin（更接近不确定）
- GLM4: 有正有负，显式规则对苹果/香蕉增强了yes-margin
- DS7B: 全负，显式规则也降低了margin

### 硬伤与问题

```text
1. GLM4/DS7B的yes-bias是参数化判断的最大问题
   - 这不是任务理解问题，而是输出偏置
   - 可能是训练数据中"是"的频率远高于"否"
   - 需要测试更复杂的回答格式（如"是的/不是"）

2. Part D样本量太小（4个）
   - 需要扩大到至少20个样本
   - 需要包含更多类别和对象

3. C_wrong_cat占失败主体说明瓶颈定位更精确了
   - 不是"状态没传递"，而是"传了错误的类别"
   - 需要分析：模型在两跳任务中预测的中间类别是什么？
   - 是不是第一步检索结果就没正确传递，还是第二步用了不同的类别？

4. flash_attention_2未安装，回退到eager
   - 需要安装flash-attn包以加速
   - 当前eager模式速度可接受
```

### 新增客观事实拼图（10条）

232. **关系必要性审计修复成功**：强制R1!=R2=>V1!=V2后，三模型都表现出关系区分能力（55-80%）。
233. **Qwen3关系使用最强(87.5%)**，DS7B最弱(75%)，DS7B经常忽略关系。
234. **组合失败的主要类型是C_wrong_cat**——使用了错误的中间类别，不是状态传递失败。
235. **Qwen3的100%失败都是C_wrong_cat**，说明它不是不能传状态，而是传了错误的类别。
236. **DS7B有4种失败类型**（C_wrong/B_fail/V_copy），说明其组合推理更不稳定。
237. **三模型参数化判断正例100%正确**，但负例差异巨大。
238. **GLM4/DS7B有强烈yes-bias**——负例准确率0%，总是说"是"。
239. **Qwen3是唯一能正确说"否"的模型**(69.2%)，但否定问法仍困难(63.3%)。
240. **显式规则对GLM4/DS7B的yes-bias无帮助**，即使给出否定规则仍说"是"。
241. **工具类别判断最差(65%)**，天体次之(77.5%)，说明参数化知识有类别特异性。

### 结果文件

```text
results/glm5_phase582_relation_parametric/
  phase582_qwen3_relation_parametric_smoke.json
  phase582_qwen3_relation_parametric.json
  phase582_glm4_relation_parametric_smoke.json
  phase582_glm4_relation_parametric.json
  phase582_deepseek7b_relation_parametric_smoke.json
  phase582_deepseek7b_relation_parametric.json
```
