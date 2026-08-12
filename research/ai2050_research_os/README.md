# AI2050 语言编码机制研究操作系统

本目录把长期的顺序 `Phase` 叙事，重构为一套可检索、可校验、可汇总、可批量否决的研究工程框架。

它解决五个问题：

1. 当前到底已知什么，证据等级和适用域是什么；
2. 哪些候选机制仍然存活，哪些强版本已经在限定范围内关闭；
3. 破解语言编码机制还缺哪些关键拼图，各自卡在哪里；
4. 下一批测试为何运行、能够区分什么、失败后关闭什么；
5. 如何自动生成项目状态、拼图看板、假说记分牌和决策总账。

## 一、核心转变

研究主流程从：

```text
局部发现 → 增加控制 → 出现边界 → 再做相邻 Phase
```

切换为：

```text
全局结构扫描
→ 候选机制并行建模
→ 主动选择分歧最大的实验
→ 封存确认
→ 批量淘汰
→ 少数幸存机制完成因果与生成闭环
```

`Phase`（执行批次）仍然保留，但不再负责决定研究方向。方向由 `Campaign`（完整研究战役）、关键拼图图和候选机制竞赛共同决定。

## 二、快速使用

在本目录运行：

```bash
python3 scripts/researchctl.py validate
python3 scripts/researchctl.py build
python3 scripts/researchctl.py summary
```

- `validate`（校验）：检查编号、引用、依赖、状态和值域；
- `build`（生成）：根据 `registry/` 唯一事实源重建全部看板；
- `summary`（摘要）：在终端输出当前战役、阻塞项和下一决策；
- `freeze`（冻结）：为预注册合同计算摘要，不执行实验。

冻结示例：

```bash
python3 scripts/researchctl.py freeze templates/experiment_contract.json
```

## 三、目录结构

```text
ai2050_research_os/
├── README.md
├── docs/
│   ├── RESEARCH_FRAMEWORK.md
│   ├── TEST_FRAMEWORK.md
│   ├── OPERATING_RULES.md
│   ├── CURRENT_BASELINE.md
│   └── INTEGRATION_GUIDE.md
├── registry/                 # 唯一事实源，人工维护
│   ├── project.json
│   ├── campaigns.json
│   ├── hypotheses.json
│   ├── puzzles.json
│   ├── tests.json
│   ├── evidence.json
│   ├── phases.json
│   └── decisions.json
├── templates/                # 新实验、新 Phase、新拼图的冻结模板
├── schemas/                  # 状态和合同字段说明
├── scripts/researchctl.py    # 无第三方依赖的校验与汇总工具
└── generated/                # 自动生成；不要手工修改
```

## 四、四本核心账

| 账本 | 回答的问题 | 禁止混入的内容 |
|---|---|---|
| 证据账 | 实际观测了什么，在哪个适用域成立 | 未运行实验、理论愿望 |
| 候选机制账 | 哪些结构在竞争，各自独特预测是什么 | 可以解释任何结果的宽泛表述 |
| 关键拼图账 | 破解机制还缺哪些闭合环节 | 单个漂亮热点或孤立指标 |
| 决策账 | 为什么授权、停止、否决或重开 | 事后改门、口头例外 |

## 五、最重要的使用纪律

1. `registry/` 是唯一事实源，`generated/` 只是自动视图；
2. 新结果先登记证据，再改变假说或拼图状态；
3. “失败”必须写成带适用域的失败类型，不能直接写成“理论错误”；
4. 行为资格、内部预测、组件因果、完整生成、跨模型和训练形成分别记账；
5. 被关闭的机制只有在提出数学上不同、并能提前解释旧失败的新版本时才能重开；
6. 热力图用于生成结构候选，封存集只用于一次裁决；
7. 不允许通过换层、换模板、换指标或删除失败分区恢复同一个已否决强版本。

## 六、当前起点

初始账本以 Phase 1210–1235 的可见材料为基线。当前最重要的事实是：

- 已有若干已知真值测量相机和有限功能商校准；
- 自然化整状态运输、形成前摘要预测、答案边界单点补丁均出现明确适用域边界；
- Phase 1235 得到稳定的类型化行为响应，但严格短字符串合同不稳定；
- 尚未得到协议相对独立的内部功能状态、未来响应张量、最小自然因果联盟或完整自回归闭环；
- 下一步不是自动执行原 Phase 1236，而是启动全局结构辨识战役的合同冻结与发现集构建。

详见 [CURRENT_BASELINE.md](docs/CURRENT_BASELINE.md)。

