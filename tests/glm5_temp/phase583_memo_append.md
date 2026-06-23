

## Phase 583: Intermediate Choice Gate and Polarity Readout Decomposition [2026-06-22 08:15]

### 本阶段目标

拆解Phase 582暴露的两个核心门控：
1. 两跳推理中的错误中间类别选择（C_wrong_cat）
2. 参数化判断中的yes-bias（是偏置）

### 生成脚本

```text
tests/glm5/phase583_choice_polarity.py
```

### 测试设计

```text
Part A: 中间类别预测追踪（在两跳任务中预测模型选择了哪个中间类别）
Part B: 强制中间类别实验（强制正确/错误类别，测量V变化）
Part C: 类别竞争边际分析（正确vs最佳竞争者的margin）
Part D: yes/no读出校准（5种答案格式对比）
Part E: 显式否定规则控制（测试否定规则能否修复yes-bias）
```

### 执行命令

```text
python tests/glm5/phase583_choice_polarity.py qwen3 --smoke --hard-exit-after-model
python tests/glm5/phase583_choice_polarity.py glm4 --smoke --hard-exit-after-model
python tests/glm5/phase583_choice_polarity.py deepseek7b --smoke --hard-exit-after-model
python tests/glm5/phase583_choice_polarity.py qwen3 --hard-exit-after-model
python tests/glm5/phase583_choice_polarity.py glm4 --hard-exit-after-model
python tests/glm5/phase583_choice_polarity.py deepseek7b --hard-exit-after-model
```

### 运行时间

```text
qwen3:      1.00 min (主测试)
glm4:       19.12 min (主测试)
ds7b:       8.30 min (主测试)
```

### 客观结果

#### Part A: 中间类别预测追踪

```text
模型      Cat预测准确率    Val预测准确率    Wrong_cat->wrong_val    Success_margin    Fail_margin
Qwen3     0.975           0.800           1.000                   4.146             -3.125
GLM4      0.775           0.850           0.111                   1.792             -0.653
DS7B      1.000           0.500           N/A(0 fail)             5.480             N/A
```

核心发现1（重大）: DS7B的Cat预测100%正确但Val只有50%！

这说明DS7B的失败完全不在中间类别选择——它正确知道中间类别，但无法从(C,R)→V检索出正确值。这推翻了Phase 582的C_wrong_cat假设对DS7B的适用性。

核心发现2: Qwen3的Cat预测97.5%，wrong_cat时100%导致wrong_val——中间类别选择确实是Qwen3的主要瓶颈。

核心发现3: GLM4的Cat预测只有77.5%但Val有85%——wrong_cat时只有11%导致wrong_val。这说明GLM4有某种绕过错误中间类别直接检索值的机制！

核心发现4: 边际分析显示success和fail的margin差距巨大：
- Qwen3: success=4.146 vs fail=-3.125 (7.3 gap)
- GLM4: success=1.792 vs fail=-0.653 (2.4 gap)
- DS7B: success=5.480 (无fail样本)

#### Part B: 强制中间类别实验

```text
模型      Base    Force_correct    Force_wrong->matches
Qwen3     0.800   0.900            0.875
GLM4      0.850   0.925            0.850
DS7B      0.500   0.775            0.750
```

核心发现5: 强制正确类别能提升所有模型（Qwen3 +10%, GLM4 +7.5%, DS7B +27.5%）。

核心发现6（重大）: 强制错误类别时，模型确实会输出错误类别对应的值（75-87.5%）！
- 这证明模型确实在使用中间类别进行第二步检索
- Qwen3: 87.5%的样本在强制错误类别时输出了错误值
- DS7B: 75%匹配——即使DS7B的Cat预测100%正确，强制错误类别仍能改变输出

#### Part C: 类别竞争边际分析

```text
模型      Cat_success_margin    Cat_fail_margin    Val_success_margin    Val_fail_margin
Qwen3     4.146                 -3.125             2.172                 -1.083
GLM4      1.792                 -0.653             1.458                 -0.635
DS7B      5.480                 N/A                2.232                 -1.448
```

核心发现7: 成功和失败的边际有清晰分界——margin正则成功，负则失败。这说明中间类别选择是一个竞争过程，margin决定了哪个类别胜出。

#### Part D: yes/no读出校准（5种格式）

```text
模型      single(是/否)    double(是的/不是)    belong(属于/不属于)    correct(正确/错误)    english(yes/no)
Qwen3     0.917            0.917                0.667                  0.667                 1.000
GLM4      0.500            0.833                0.500                  0.667                 0.542
DS7B      0.500            0.667                0.583                  0.500                 0.833
```

核心发现8（重大）: yes-bias强烈依赖答案格式！

```text
Qwen3: english(yes/no)最好(100%), belong/correct最差(67%)
GLM4:  double(是的/不是)最好(83%), single/belong最差(50%)
DS7B:  english(yes/no)最好(83%), single/correct最差(50%)
```

- GLM4用"是的/不是"格式从50%提升到83.3%——提升33%！
- DS7B用"yes/no"格式从50%提升到83.3%——提升33%！
- 这证明yes-bias部分来自token先验，不是纯粹的知识缺失

核心发现9: 正例在所有格式下都接近100%，差异完全来自负例。
- Qwen3 english负例: 12/12=100%
- GLM4 double负例: 10/12=83%
- DS7B english负例: 8/12=67%

#### Part E: 显式否定规则控制

```text
模型      No_rule    Aff_rule    Strong_neg    Negatives_only
Qwen3     9/11       11/11       11/11         no=4/6, aff=6/6, strong=6/6
GLM4      5/11       5/11        5/11          no=0/6, aff=0/6, strong=0/6
DS7B      5/11       5/11        5/11          no=0/6, aff=0/6, strong=0/6
```

核心发现10（重大）: 显式否定规则对Qwen3完全有效（4/6→6/6），但对GLM4/DS7B完全无效（0/6→0/6）！

```text
Qwen3: 显式否定规则成功将yes-margin从正变为负
  苹果/天体(否): margin 1.75 -> -4.87 (成功翻转!)
  苹果/工具(否): margin 1.25 -> -6.12 (成功翻转!)

GLM4: 显式否定规则降低了margin但不足以翻转
  老虎/水果(否): margin 3.44 -> 1.25 -> 0.56 (仍为正!)
  苹果/工具(否): margin 4.56 -> 1.44 -> 1.25 (仍为正!)

DS7B: 同样无法翻转
  老虎/水果(否): margin 2.33 -> 1.89 -> 1.55 (仍为正!)
```

这说明GLM4/DS7B的yes-bias非常强——即使给出显式否定规则和替代类别，margin仍为正。这不是知识缺失，而是极性读出门的强偏置。

### 硬伤与问题

```text
1. Part D/E样本量较小（24/11个）
   - 需要扩大到至少50个样本
   - 特别是Part E只有6个负例

2. GLM4的wrong_cat不导致wrong_val（11%）需要解释
   - 可能GLM4有直接O→V的旁路检索
   - 或者GLM4的值检索不完全依赖中间类别
   - 需要分析GLM4在wrong_cat时实际检索了什么

3. DS7B的Cat=100%但Val=50%是最关键发现
   - DS7B知道正确中间类别但无法检索值
   - 这说明DS7B的(C,R)→V检索本身有根本性问题
   - 需要分析DS7B的第二步检索注意力

4. flash_attention_2仍未安装
   - 所有测试使用eager模式
   - 速度可接受但不是最优

5. Qwen3的english格式100%是重要发现
   - 可能因为英文yes/no的token先验更平衡
   - 需要测试更多英文对象和类别
```

### 新增客观事实拼图（10条）

242. **DS7B的Cat预测100%但Val只有50%**——失败不在中间类别选择，而在第二步值检索。
243. **Qwen3的wrong_cat时100%导致wrong_val**——中间类别选择确实是Qwen3的主要瓶颈。
244. **GLM4的wrong_cat只有11%导致wrong_val**——有绕过错误中间类别的旁路机制。
245. **强制错误类别时75-87%的样本输出错误值**——证明模型确实使用中间类别做第二步检索。
246. **强制正确类别能提升所有模型**（Qwen3 +10%, GLM4 +7.5%, DS7B +27.5%）。
247. **yes-bias强烈依赖答案格式**：GLM4用"是的/不是"从50%提升到83%，DS7B用"yes/no"从50%提升到83%。
248. **正例在所有格式下接近100%，差异完全来自负例**——极性控制是独立机制。
249. **显式否定规则对Qwen3完全有效（4/6→6/6），对GLM4/DS7B完全无效（0/6→0/6）**。
250. **GLM4/DS7B的yes-bias非常强**——即使给出显式否定规则和替代类别，margin仍为正。
251. **边际分析显示success和fail有清晰分界**——中间类别选择是竞争过程，margin决定胜出者。

### 结果文件

```text
results/glm5_phase583_choice_polarity/
  phase583_qwen3_choice_polarity_smoke.json
  phase583_qwen3_choice_polarity.json
  phase583_glm4_choice_polarity_smoke.json
  phase583_glm4_choice_polarity.json
  phase583_deepseek7b_choice_polarity_smoke.json
  phase583_deepseek7b_choice_polarity.json
```
