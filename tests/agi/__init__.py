"""
AGI Evaluation Framework
========================

融合多种标准的综合AGI评估框架:
- ARC-AGI: 抽象推理与泛化能力
- HELM: 多维度综合评估
- 几何约束: 纤维丛理论验证
- 安全对齐: 可控性验证

5层金字塔评估体系:
    Level 5: Safety      - 安全对齐
    Level 4: Geometric   - 几何约束
    Level 3: Agentic     - 自主能力
    Level 2: Generalize  - 泛化能力
    Level 1: Basic       - 基础能力

Author: AGI Research Team
Date: 2026-02-16
Version: 1.0
"""

from .framework import AGIEvaluationFramework, EvaluationResult
from .tests import *
from .utils.data_generator import DataGenerator
from .utils.report_generator import ReportGenerator

__all__ = [
    'AGIEvaluationFramework',
    'EvaluationResult',
    'DataGenerator',
    'ReportGenerator',
]
