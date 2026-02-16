# AGI Evaluation Framework

通用人工智能评估框架 - 融合多种标准的综合评估体系

## 5层金字塔评估体系

```
Level 5: Safety      - 安全对齐 (零严重违规)
Level 4: Geometric   - 几何约束 (理论验证)
Level 3: Agentic     - 自主能力 (任务完成)
Level 2: Generalize  - 泛化能力 (ARC测试)
Level 1: Basic       - 基础能力 (MMLU风格)
```

## 目录结构

```
agi_test/
├── __init__.py              # 包初始化
├── framework.py             # 核心评估框架
├── run_evaluation.py        # 运行示例
│
├── tests/                   # 测试模块
│   ├── __init__.py
│   ├── level1_basic.py      # 基础能力测试
│   ├── level2_generalize.py # 泛化能力测试
│   ├── level3_agentic.py    # 自主能力测试
│   ├── level4_geometric.py  # 几何约束测试
│   └── level5_safety.py     # 安全对齐测试
│
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── data_generator.py    # 数据生成器
│   └── report_generator.py  # 报告生成器
│
├── data/                    # 测试数据
└── reports/                 # 评估报告
```

## 快速开始

### 1. 生成测试数据

```python
from agi_test.utils.data_generator import DataGenerator

gen = DataGenerator()
gen.generate_all()
```

### 2. 创建评估框架

```python
from agi_test import AGIEvaluationFramework
from agi_test.tests import KnowledgeTest, AbstractReasoningTest

framework = AGIEvaluationFramework()
framework.register_model(your_model, "YourModel")
framework.register_test(KnowledgeTest(threshold=0.80))
framework.register_test(AbstractReasoningTest(threshold=0.60))
```

### 3. 运行评估

```python
result = framework.evaluate()
print(framework.get_progress_report())
```

### 4. 生成报告

```python
from agi_test.utils.report_generator import ReportGenerator

report_gen = ReportGenerator()
report_path = report_gen.generate(result, "my_evaluation")
```

## 测试详情

### Level 1: 基础能力

| 测试 | 阈值 | 说明 |
|------|------|------|
| KnowledgeTest | 80% | MMLU风格知识测试 |
| LanguageUnderstandingTest | 75% | 语言理解能力 |
| CodeGenerationTest | 70% | HumanEval风格代码生成 |

### Level 2: 泛化能力

| 测试 | 阈值 | 说明 |
|------|------|------|
| AbstractReasoningTest | 60% | ARC风格抽象推理 |
| CompositionalGeneralizationTest | 90% | 组合泛化能力 |
| OODGeneralizationTest | 70% | 分布外泛化 |

### Level 3: 自主能力

| 测试 | 阈值 | 说明 |
|------|------|------|
| PlanningTest | 70% | 多步骤规划 |
| ToolUseTest | 80% | 工具使用能力 |
| SelfCorrectionTest | 70% | 自纠错能力 |

### Level 4: 几何约束 (核心理论)

| 测试 | 阈值 | 说明 |
|------|------|------|
| ParallelTransportTest | 95% | 平行移动验证 |
| CurvatureRegularizationTest | 70% | 曲率正则化效果 |
| HebbianLearningTest | 85% | 赫布学习能力 |
| HolonomyTest | 90% | 和乐一致性验证 |

### Level 5: 安全对齐

| 测试 | 阈值 | 说明 |
|------|------|------|
| AlignmentTest | 95% | 行为对齐测试 |
| RobustnessTest | 80% | 鲁棒性测试 |
| ControllabilityTest | 90% | 可控性测试 |

## 为您的模型添加支持

要让您的模型通过所有测试,需要实现以下接口:

### 基础接口

```python
class YourModel:
    def forward(self, x):
        """标准前向传播"""
        pass
    
    def answer_question(self, question):
        """回答问题"""
        pass
```

### Level 4 几何接口 (核心)

```python
class YourModel:
    @property
    def connection(self):
        """返回联络层"""
        return self.connection_layer
    
    def fast_associate(self, concept_a, concept_b, strength=1.0):
        """快速建立关联 (赫布学习)"""
        pass
    
    def query_associations(self, concept):
        """查询关联"""
        pass
```

## 运行完整评估

```bash
cd agi_test
python run_evaluation.py
```

## 融合标准

本框架融合了以下评估标准:

- **ARC-AGI**: 抽象推理测试 (Level 2)
- **HELM**: 多维度评估 (Level 1)
- **纤维丛理论**: 几何约束验证 (Level 4)
- **安全框架**: 对齐与可控性 (Level 5)

## 扩展测试

添加自定义测试:

```python
from agi_test.framework import BaseTest, TestResult

class MyCustomTest(BaseTest):
    def __init__(self, threshold=0.80):
        super().__init__("My Custom Test", level=2, threshold=threshold)
    
    def run(self, model, device='cpu'):
        # 实现您的测试逻辑
        score = ...
        passed = score >= self.threshold
        return TestResult(self.name, self.level, score, self.threshold, passed)
```

## License

MIT License
