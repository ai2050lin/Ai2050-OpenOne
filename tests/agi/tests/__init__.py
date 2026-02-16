"""
AGI测试模块
==========

包含所有层级的测试实现。
"""

from .level1_basic import (
    KnowledgeTest,
    LanguageUnderstandingTest,
    CodeGenerationTest,
)
from .level2_generalize import (
    AbstractReasoningTest,
    CompositionalGeneralizationTest,
    OODGeneralizationTest,
)
from .level3_agentic import (
    PlanningTest,
    ToolUseTest,
    SelfCorrectionTest,
)
from .level4_geometric import (
    ParallelTransportTest,
    CurvatureRegularizationTest,
    HebbianLearningTest,
    HolonomyTest,
)
from .level5_safety import (
    AlignmentTest,
    RobustnessTest,
    ControllabilityTest,
)

__all__ = [
    # Level 1
    'KnowledgeTest',
    'LanguageUnderstandingTest',
    'CodeGenerationTest',
    # Level 2
    'AbstractReasoningTest',
    'CompositionalGeneralizationTest',
    'OODGeneralizationTest',
    # Level 3
    'PlanningTest',
    'ToolUseTest',
    'SelfCorrectionTest',
    # Level 4
    'ParallelTransportTest',
    'CurvatureRegularizationTest',
    'HebbianLearningTest',
    'HolonomyTest',
    # Level 5
    'AlignmentTest',
    'RobustnessTest',
    'ControllabilityTest',
]
