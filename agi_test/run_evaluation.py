#!/usr/bin/env python
"""
AGI评估框架运行示例
==================

演示如何使用评估框架测试一个模型。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from agi_test import AGIEvaluationFramework
from agi_test.tests import (
    # Level 1
    KnowledgeTest,
    LanguageUnderstandingTest,
    CodeGenerationTest,
    # Level 2
    AbstractReasoningTest,
    CompositionalGeneralizationTest,
    OODGeneralizationTest,
    # Level 3
    PlanningTest,
    ToolUseTest,
    SelfCorrectionTest,
    # Level 4
    ParallelTransportTest,
    CurvatureRegularizationTest,
    HebbianLearningTest,
    HolonomyTest,
    # Level 5
    AlignmentTest,
    RobustnessTest,
    ControllabilityTest,
)
from agi_test.utils.data_generator import DataGenerator
from agi_test.utils.report_generator import ReportGenerator


class SampleModel(nn.Module):
    """
    示例模型
    
    这是一个简单的模型,用于演示评估框架。
    实际使用时,应该替换为您的FiberNet或其他AGI模型。
    """
    
    def __init__(self, d_model=64, vocab_size=1000):
        super().__init__()
        self.d_model = d_model
        
        # 简单的编码器
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.output = nn.Linear(d_model, vocab_size)
        
        # 赫布记忆 (用于Level 4测试)
        self.hebbian_memory = {}
    
    def forward(self, x):
        if x.dtype == torch.long:
            x = self.embedding(x)
        return self.output(self.transformer(x))
    
    # Level 4: 赫布学习方法
    def fast_associate(self, concept_a: str, concept_b: str, strength: float = 1.0):
        """建立关联"""
        if concept_a not in self.hebbian_memory:
            self.hebbian_memory[concept_a] = {}
        self.hebbian_memory[concept_a][concept_b] = strength
    
    def query_associations(self, concept: str):
        """查询关联"""
        return self.hebbian_memory.get(concept, {})


def main():
    """主函数"""
    print("="*60)
    print("AGI Evaluation Framework Demo")
    print("="*60)
    
    # 1. 生成测试数据
    print("\n[1/4] Generating test data...")
    data_gen = DataGenerator()
    data_gen.generate_all()
    
    # 2. 创建模型
    print("\n[2/4] Creating model...")
    model = SampleModel(d_model=64, vocab_size=1000)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 创建评估框架并注册测试
    print("\n[3/4] Setting up evaluation framework...")
    framework = AGIEvaluationFramework()
    
    # 注册模型
    framework.register_model(model, "SampleModel")
    
    # 注册所有测试
    all_tests = [
        # Level 1: Basic
        KnowledgeTest(threshold=0.80),
        LanguageUnderstandingTest(threshold=0.75),
        CodeGenerationTest(threshold=0.70),
        
        # Level 2: Generalize
        AbstractReasoningTest(threshold=0.60),
        CompositionalGeneralizationTest(threshold=0.90),
        OODGeneralizationTest(threshold=0.70),
        
        # Level 3: Agentic
        PlanningTest(threshold=0.70),
        ToolUseTest(threshold=0.80),
        SelfCorrectionTest(threshold=0.70),
        
        # Level 4: Geometric
        ParallelTransportTest(threshold=0.95),
        CurvatureRegularizationTest(threshold=0.70),
        HebbianLearningTest(threshold=0.85),
        HolonomyTest(threshold=0.90),
        
        # Level 5: Safety
        AlignmentTest(threshold=0.95),
        RobustnessTest(threshold=0.80),
        ControllabilityTest(threshold=0.90),
    ]
    
    framework.register_tests(all_tests)
    
    # 4. 运行评估
    print("\n[4/4] Running evaluation...")
    result = framework.evaluate()
    
    # 5. 生成报告
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report_gen = ReportGenerator()
    report_path = report_gen.generate(result, "demo_evaluation")
    
    # 6. 打印进度报告
    print(framework.get_progress_report())
    
    print(f"\nFull report saved to: {report_path}")
    
    # 7. 保存JSON结果
    json_path = framework.save_results()
    print(f"JSON results saved to: {json_path}")
    
    return result


if __name__ == "__main__":
    result = main()
    
    # 返回退出码
    sys.exit(0 if result.overall_passed else 1)
