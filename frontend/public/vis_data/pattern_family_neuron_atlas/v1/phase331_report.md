# Phase331 五条候选链扩展留出、双接口与补偿审计

## 客观分母

- 接口案例：720（正机制 360，匹配负对照 360）。
- 条件结果：9360；执行自然生成：5760。
- 全层路径事件：1288560；组件响应事件：195520。
- 留出对象固定为 19-22，覆盖三个模板、原始续写接口和模型对话模板接口。

## 结果边界

- 五条 Phase330 候选中，扩展后仍通过跨模型、跨接口集合读出门槛：0/5。
- 通过完整八项 Phase331 门槛：0/5。
- 可宣称行为机制闭合：0/5；全 72 机制仍为 0/72。
- 所有成员仍是注意力头或 MLP 乘积组候选，没有把组级效果改写成单神经元因果。

## 五条候选链

| 模式机制 | 集合读出扩展 | 成员定位 | 补偿受控 | 生成行为损失 | 负对照失败 | 证据等级 |
|---|---:|---:|---:|---:|---:|---|
| content_knowledge/negated_attribute | False | False | False | False | False | L3_candidate_not_expanded_cross_interface |
| language_action/summarize | False | False | False | False | True | L3_candidate_not_expanded_cross_interface |
| language_action/transform | False | False | False | False | True | L3_candidate_not_expanded_cross_interface |
| reasoning_constraint/missing_condition_control | False | False | True | False | False | L3_candidate_not_expanded_cross_interface |
| syntax_structure/singular_agreement | False | False | False | False | True | L3_candidate_not_expanded_cross_interface |

## 关键校准

1. 对话模板不是表面包装。模型会引入助手前缀或思考前缀，因此首词元读出必须和完整答案串概率、自然生成分开报告。
2. 累计残差状态不能当作本层写入量。本轮同时保存累计状态、注意力增量、MLP 增量和层间残差增量。
3. 组级联合干预可能被未干预头、其他 MLP 组或后层残差恢复补偿；没有补偿审计就不能把读出变化解释成行为必要性。
4. 小模型之间的路径位置和思考协议差异较大；跨模型失败既可能是否定共享机制，也可能反映 4B-9B 模型的粗糙结构，二者尚不能区分。

## 图谱进度向量

- 九族注册与观察分母：9/9（100%）。
- 72 机制三模型行为、读出和全层普查：72/72（100%）。
- Phase330 五条候选的扩展留出与双接口审计：5/5（100% 已测试，不等于通过）。
- 通过 Phase331 全门槛：0/5。
- 语言机制行为闭合：0/72。
- 单神经元因果闭合：0/72。

因此不提供一个会混淆工程覆盖率与科学证据率的单一总百分比。
