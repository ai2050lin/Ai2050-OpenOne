"""
AGI评估框架核心模块
==================

实现5层金字塔评估体系的统一入口。
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn


@dataclass
class TestResult:
    """单个测试结果"""
    test_name: str
    level: int
    score: float
    threshold: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class LevelResult:
    """单层级评估结果"""
    level: int
    level_name: str
    tests: List[TestResult]
    overall_score: float
    passed: bool
    pass_rate: float


@dataclass
class EvaluationResult:
    """完整评估结果"""
    model_name: str
    timestamp: str
    levels: List[LevelResult]
    overall_passed: bool
    total_score: float
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'overall_passed': self.overall_passed,
            'total_score': self.total_score,
            'summary': self.summary,
            'levels': [
                {
                    'level': lv.level,
                    'name': lv.level_name,
                    'overall_score': lv.overall_score,
                    'passed': lv.passed,
                    'pass_rate': lv.pass_rate,
                    'tests': [
                        {
                            'name': t.test_name,
                            'score': t.score,
                            'threshold': t.threshold,
                            'passed': t.passed,
                            'details': t.details,
                        }
                        for t in lv.tests
                    ]
                }
                for lv in self.levels
            ]
        }


class BaseTest:
    """测试基类"""
    
    def __init__(self, name: str, level: int, threshold: float):
        self.name = name
        self.level = level
        self.threshold = threshold
        self.result: Optional[TestResult] = None
    
    def setup(self):
        """测试前准备"""
        pass
    
    def run(self, model: nn.Module, device: str = 'cpu') -> TestResult:
        """运行测试 - 子类必须实现"""
        raise NotImplementedError
    
    def teardown(self):
        """测试后清理"""
        pass
    
    def evaluate(self, model: nn.Module, device: str = 'cpu') -> TestResult:
        """完整评估流程"""
        start_time = time.time()
        
        try:
            self.setup()
            self.result = self.run(model, device)
            self.result.duration = time.time() - start_time
            self.teardown()
        except Exception as e:
            self.result = TestResult(
                test_name=self.name,
                level=self.level,
                score=0.0,
                threshold=self.threshold,
                passed=False,
                error=str(e),
                duration=time.time() - start_time,
            )
        
        return self.result


class AGIEvaluationFramework:
    """
    AGI评估框架主类
    
    使用示例:
    ```python
    from agi_test import AGIEvaluationFramework
    
    # 创建评估框架
    framework = AGIEvaluationFramework()
    
    # 注册模型
    framework.register_model(my_model, "MyAGIModel")
    
    # 运行评估
    result = framework.evaluate()
    
    # 生成报告
    framework.generate_report(result, "reports/evaluation.html")
    ```
    """
    
    LEVELS = {
        1: {'name': 'Basic', 'description': '基础能力'},
        2: {'name': 'Generalize', 'description': '泛化能力'},
        3: {'name': 'Agentic', 'description': '自主能力'},
        4: {'name': 'Geometric', 'description': '几何约束'},
        5: {'name': 'Safety', 'description': '安全对齐'},
    }
    
    def __init__(self, output_dir: str = 'agi_test/reports'):
        """
        初始化评估框架
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[nn.Module] = None
        self.model_name: str = "Unknown"
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 各层级的测试列表
        self.tests: Dict[int, List[BaseTest]] = {i: [] for i in range(1, 6)}
        
        # 评估结果
        self.results: Optional[EvaluationResult] = None
    
    def register_model(self, model: nn.Module, name: str):
        """注册待评估的模型"""
        self.model = model
        self.model_name = name
        self.model.to(self.device)
    
    def register_test(self, test: BaseTest):
        """注册测试"""
        if test.level not in self.tests:
            raise ValueError(f"Invalid level: {test.level}")
        self.tests[test.level].append(test)
    
    def register_tests(self, tests: List[BaseTest]):
        """批量注册测试"""
        for test in tests:
            self.register_test(test)
    
    def evaluate_level(self, level: int) -> LevelResult:
        """评估单个层级"""
        if level not in self.LEVELS:
            raise ValueError(f"Invalid level: {level}")
        
        level_info = self.LEVELS[level]
        tests = self.tests[level]
        
        if not tests:
            return LevelResult(
                level=level,
                level_name=level_info['name'],
                tests=[],
                overall_score=0.0,
                passed=False,
                pass_rate=0.0,
            )
        
        # 运行所有测试
        results = []
        for test in tests:
            result = test.evaluate(self.model, self.device)
            results.append(result)
        
        # 计算层级得分
        scores = [r.score for r in results if r.error is None]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / len(results) if results else 0.0
        
        # 层级通过条件: 所有测试都通过
        level_passed = all(r.passed for r in results)
        
        return LevelResult(
            level=level,
            level_name=level_info['name'],
            tests=results,
            overall_score=overall_score,
            passed=level_passed,
            pass_rate=pass_rate,
        )
    
    def evaluate(self) -> EvaluationResult:
        """运行完整评估"""
        if self.model is None:
            raise ValueError("No model registered. Call register_model() first.")
        
        start_time = time.time()
        level_results = []
        
        # 从低到高逐层评估
        for level in range(1, 6):
            print(f"\n{'='*60}")
            print(f"Evaluating Level {level}: {self.LEVELS[level]['name']}")
            print(f"{'='*60}")
            
            level_result = self.evaluate_level(level)
            level_results.append(level_result)
            
            # 打印层级结果
            print(f"\nLevel {level} Result:")
            print(f"  Overall Score: {level_result.overall_score:.2%}")
            print(f"  Pass Rate: {level_result.pass_rate:.2%}")
            print(f"  Passed: {level_result.passed}")
            
            for test in level_result.tests:
                status = "[PASS]" if test.passed else "[FAIL]"
                print(f"  {status} {test.test_name}: {test.score:.2%} (threshold: {test.threshold:.2%})")
        
        # 计算总体结果
        total_score = sum(lv.overall_score for lv in level_results) / 5
        overall_passed = all(lv.passed for lv in level_results)
        
        # 生成摘要
        summary = {
            'total_duration': time.time() - start_time,
            'levels_passed': sum(1 for lv in level_results if lv.passed),
            'total_tests': sum(len(lv.tests) for lv in level_results),
            'tests_passed': sum(
                sum(1 for t in lv.tests if t.passed) 
                for lv in level_results
            ),
        }
        
        self.results = EvaluationResult(
            model_name=self.model_name,
            timestamp=datetime.now().isoformat(),
            levels=level_results,
            overall_passed=overall_passed,
            total_score=total_score,
            summary=summary,
        )
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None):
        """保存评估结果"""
        if self.results is None:
            raise ValueError("No results to save. Run evaluate() first.")
        
        if filename is None:
            filename = f"evaluation_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def get_progress_report(self) -> str:
        """生成进度报告"""
        if self.results is None:
            return "No evaluation results available."
        
        report = []
        report.append("\n" + "="*60)
        report.append("AGI EVALUATION PROGRESS REPORT")
        report.append("="*60)
        report.append(f"Model: {self.results.model_name}")
        report.append(f"Timestamp: {self.results.timestamp}")
        report.append(f"\nOverall Score: {self.results.total_score:.2%}")
        report.append(f"Overall Status: {'PASSED' if self.results.overall_passed else 'FAILED'}")
        report.append(f"\nLevels Passed: {self.results.summary['levels_passed']}/5")
        report.append(f"Tests Passed: {self.results.summary['tests_passed']}/{self.results.summary['total_tests']}")
        report.append(f"Duration: {self.results.summary['total_duration']:.2f}s")
        
        report.append("\n" + "-"*60)
        report.append("LEVEL DETAILS")
        report.append("-"*60)
        
        for lv in self.results.levels:
            status = "[PASS]" if lv.passed else "[FAIL]"
            report.append(f"\n{status} Level {lv.level} ({lv.level_name}): Score {lv.overall_score:.2%}")
            for t in lv.tests:
                t_status = "  [OK]" if t.passed else "  [X]"
                report.append(f"{t_status} {t.test_name}: {t.score:.2%}")
        
        return "\n".join(report)
