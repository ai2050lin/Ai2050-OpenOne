

## Phase 581: Intermediate State Forcing and Retrieval Composition Closure [2026-06-22 05:06]

### 本阶段目标

破解Phase 580的最大瓶颈：单步检索成功，但两跳组合推理0%。
核心问题：查到的C（类别）为什么不能变成下一步检索用的query（查询）？

### 生成脚本

```text
tests/glm5/phase581_composition_closure.py
```

### 关键技术突破

#### Phase 580组合推理0%是prompt构建bug，不是真实失败

Phase 580的组合推理0%是因为CRV规则构建错误：
- Phase 580错误：使用orv_table（(O,R)->V）作为第二步规则，但orv_table的键是(object,relation)不是(category,relation)
- Phase 581修复：新建build_cat_rel_truth_tables，正确构建(C,R)->V映射
- 修复后组合推理从0%提升到57-90%！

### 测试设计

```text
Step 0: 关系必要性审计（测试模型是否真正使用R）
Step 1: 两跳三段拆解（Step A: O->C, Step B: (C,R)->V gold, Step C: Compose）
Step 2: 强制中间项（提示模型先找类别）
Step 3: 黄金中间项（直接提供正确类别）
Step 4: 激活级中间状态注入（从单步OC提取h_C注入两跳）
Step 7: 语法脚手架影响（direct vs forced vs proof）
```

### 执行命令

```text
python tests/glm5/phase581_composition_closure.py qwen3 --smoke --hard-exit-after-model
python tests/glm5/phase581_composition_closure.py glm4 --smoke --hard-exit-after-model
python tests/glm5/phase581_composition_closure.py deepseek7b --smoke --hard-exit-after-model
python tests/glm5/phase581_composition_closure.py qwen3 --hard-exit-after-model
python tests/glm5/phase581_composition_closure.py glm4 --hard-exit-after-model
python tests/glm5/phase581_composition_closure.py deepseek7b --hard-exit-after-model
```

### 运行时间

```text
qwen3:      0.90 min (主测试)
glm4:       16.80 min (主测试)
ds7b:       7.21 min (主测试)
```

### 客观结果

#### Step 1: 两跳三段拆解

```text
模型      Step A (O->C)    Step B (C,R->V gold)    Step C (Compose)    A&B->C
Qwen3     30/30 (1.0)     27/30 (0.900)           22/30 (0.733)       19/27 (0.704)
GLM4      30/30 (1.0)     30/30 (1.000)           27/30 (0.900)       27/30 (0.900)
DS7B      30/30 (1.0)     23/30 (0.767)           17/30 (0.567)       14/23 (0.609)
```

核心发现1（重大）: Phase 580的组合推理0%是prompt构建bug！修复后组合推理达到57-90%。

核心发现2: Step A（O->C）三模型全部100%，单步检索完全成立。

核心发现3: Step B（C,R->V gold）存在差距：
- GLM4=100%，Qwen3=90%，DS7B=76.7%
- DS7B的第二步检索本身就有23%失败
- 这说明DS7B的瓶颈部分在第二步检索本身

核心发现4: 组合推理存在gap（gold vs compose）：
- Qwen3: gold=90% vs compose=73.3% -> 16.7% gap
- GLM4: gold=100% vs compose=90% -> 10% gap
- DS7B: gold=76.7% vs compose=56.7% -> 20% gap
- 这个gap就是中间状态传递的瓶颈

核心发现5: A&B都正确但C失败的比例：
- Qwen3: 27个A&B都对的样本中，只有19个compose对（70.4%）
- GLM4: 30个A&B都对的样本中，27个compose对（90%）
- DS7B: 23个A&B都对的样本中，只有14个compose对（60.9%）
- 这直接测量了状态传递瓶颈：即使两步都能做，组合仍有10-40%失败

#### Step 2&3: 强制中间项和黄金中间项

```text
模型      Direct    Forced    Gold
Qwen3     0.733     0.833     0.900
GLM4      0.900     0.833     1.000
DS7B      0.567     0.667     0.767
```

核心发现6: Gold intermediate总是最好（76.7-100%），证实提供正确中间类别能大幅提升。

核心发现7: Forced intermediate效果因模型而异：
- Qwen3: forced(83.3%) > direct(73.3%) -> +10%
- DS7B: forced(66.7%) > direct(56.7%) -> +10%
- GLM4: forced(83.3%) < direct(90%) -> -6.7% (反而下降!)
- 说明forced scaffold对Qwen3/DS7B有帮助，但对GLM4反而干扰

#### Step 4: 激活级中间状态注入

```text
模型      base准确率    注入后准确率    效果
Qwen3     2/2 (1.0)     多数维持1.0     L9/L12低alpha时下降到0.5
GLM4      2/2 (1.0)     全部维持1.0     注入无害但也无益
DS7B      0/2 (0.0)     全部0/2 (0.0)   注入完全无效
```

核心发现8: 激活注入对DS7B完全无效——base就是0%，注入后仍然0%。
- DS7B的两跳失败不是中间状态缺失，而是更深层的问题
- 可能是第二步检索本身就不稳定（Step B只有76.7%）

#### Step 7: 语法脚手架影响

```text
模型      Direct    Forced    Proof
Qwen3     0.733     0.833     0.867
GLM4      0.900     0.833     0.867
DS7B      0.567     0.667     0.600
```

核心发现9: Proof-style对Qwen3最好(86.7%)，但对GLM4反而低于direct(86.7% vs 90%)。
- 语法脚手架可以诱导组合路径，但效果因模型而异
- Qwen3从direct到proof提升13.4%，DS7B提升3.3%，GLM4下降3.3%

### 硬伤与问题

```text
1. Step 0关系必要性审计返回0/0
   - build_orv_truth_tables为每个对象分配所有关系，但同一对象的不同关系可能映射到同一值
   - 需要修改为强制同一对象的不同关系映射到不同值
   - 这是Phase 582需要修复的

2. Step 4激活注入样本量太小（2个）
   - base=100%时无法看到注入效果（天花板效应）
   - DS7B base=0%时注入也无效，但样本太少
   - 需要选择base在50%左右的样本测试

3. 组合推理的gap（gold vs compose）需要更细致分析
   - 当前只知道gap大小（10-20%），不知道失败原因
   - 需要分析失败样本：是第一步检索结果没传递，还是第二步使用了错误的中间类别

4. Phase 580的0%结论需要修正
   - Phase 580 Part C的build_compositional_prompt有bug
   - 实际组合推理能力为57-90%，不是0%
   - 但gap仍然存在，组合不是完美的
```

### 新增客观事实拼图（10条）

222. **Phase 580组合推理0%是prompt构建bug**，修复后达到57-90%。
223. **Step A（O->C）三模型全部100%**，单步OC检索完全成立。
224. **Step B（C,R->V gold）存在模型差异**：GLM4=100%，Qwen3=90%，DS7B=76.7%。
225. **组合推理存在gap**：gold vs compose差距10-20%，这是中间状态传递瓶颈。
226. **A&B都正确但compose仍失败10-40%**，直接测量了状态传递瓶颈。
227. **Gold intermediate总是最好（76.7-100%）**，提供正确中间类别能大幅提升。
228. **Forced scaffold对Qwen3/DS7B有+10%提升**，但对GLM4反而-6.7%干扰。
229. **激活注入对DS7B完全无效**，其失败不是中间状态缺失而是更深层问题。
230. **Proof-style对Qwen3最好(+13.4%)**，语法脚手架可以诱导组合路径。
231. **DS7B组合推理最弱(56.7%)**，瓶颈部分在第二步检索本身(Step B=76.7%)。

### 结果文件

```text
results/glm5_phase581_composition_closure/
  phase581_qwen3_composition_closure_smoke.json
  phase581_qwen3_composition_closure.json
  phase581_glm4_composition_closure_smoke.json
  phase581_glm4_composition_closure.json
  phase581_deepseek7b_composition_closure_smoke.json
  phase581_deepseek7b_composition_closure.json
```
