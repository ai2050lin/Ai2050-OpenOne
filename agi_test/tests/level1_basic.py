"""
Level 1: 基础能力测试
===================

测试模型的基础知识和语言能力。
"""

import random
from typing import Dict, List

import torch
import torch.nn as nn

from ..framework import BaseTest, TestResult


class KnowledgeTest(BaseTest):
    """
    知识测试
    
    评估模型的基础知识储备，类似MMLU风格。
    """
    
    def __init__(self, threshold: float = 0.80):
        super().__init__(
            name="Knowledge Test (MMLU-style)",
            level=1,
            threshold=threshold
        )
        self.questions = self._generate_questions()
    
    def _generate_questions(self) -> List[Dict]:
        """生成测试问题"""
        # 简化的知识测试问题
        questions = [
            {
                'question': '法国的首都是哪里?',
                'choices': ['伦敦', '巴黎', '柏林', '马德里'],
                'answer': 1,
                'category': '地理'
            },
            {
                'question': '光速约为多少公里/秒?',
                'choices': ['3000', '30000', '300000', '3000000'],
                'answer': 2,
                'category': '物理'
            },
            {
                'question': 'DNA的全称是什么?',
                'choices': [
                    '脱氧核糖核酸',
                    '核糖核酸',
                    '腺嘌呤核苷酸',
                    '胸腺嘧啶核苷酸'
                ],
                'answer': 0,
                'category': '生物'
            },
            {
                'question': '《红楼梦》的作者是谁?',
                'choices': ['罗贯中', '曹雪芹', '吴承恩', '施耐庵'],
                'answer': 1,
                'category': '文学'
            },
            {
                'question': '勾股定理中,直角三角形三边关系是?',
                'choices': [
                    'a + b = c',
                    'a² + b² = c²',
                    'a × b = c',
                    'a / b = c'
                ],
                'answer': 1,
                'category': '数学'
            },
        ]
        return questions
    
    def run(self, model: nn.Module, device: str = 'cpu') -> TestResult:
        """运行知识测试"""
        correct = 0
        total = len(self.questions)
        details = {'by_category': {}}
        
        for q in self.questions:
            # 简化评估: 假设模型输出正确答案的索引
            # 实际实现需要根据模型接口调整
            predicted = self._evaluate_question(model, q, device)
            is_correct = predicted == q['answer']
            
            if is_correct:
                correct += 1
            
            # 按类别统计
            cat = q['category']
            if cat not in details['by_category']:
                details['by_category'][cat] = {'correct': 0, 'total': 0}
            details['by_category'][cat]['total'] += 1
            if is_correct:
                details['by_category'][cat]['correct'] += 1
        
        score = correct / total
        
        return TestResult(
            test_name=self.name,
            level=self.level,
            score=score,
            threshold=self.threshold,
            passed=score >= self.threshold,
            details=details,
        )
    
    def _evaluate_question(self, model: nn.Module, question: Dict, device: str) -> int:
        """评估单个问题 - 子类可重写"""
        # 默认实现: 随机猜测 (需要根据实际模型实现)
        # 这里提供一个模拟接口
        if hasattr(model, 'answer_question'):
            return model.answer_question(question)
        else:
            # 模拟: 基于模型参数的随机选择
            return random.randint(0, 3)


class LanguageUnderstandingTest(BaseTest):
    """
    语言理解测试
    
    评估模型的语言理解和生成能力。
    """
    
    def __init__(self, threshold: float = 0.75):
        super().__init__(
            name="Language Understanding Test",
            level=1,
            threshold=threshold
        )
        self.tasks = self._generate_tasks()
    
    def _generate_tasks(self) -> List[Dict]:
        """生成语言理解任务"""
        return [
            {
                'type': 'sentiment',
                'text': '这部电影非常精彩,我非常喜欢!',
                'expected': 'positive'
            },
            {
                'type': 'sentiment',
                'text': '这个产品太差了,完全不推荐。',
                'expected': 'negative'
            },
            {
                'type': 'summary',
                'text': '人工智能是计算机科学的一个分支,它企图了解智能的实质,并生产出一种新的能以人类智能相似的方式做出反应的智能机器。',
                'expected_keywords': ['人工智能', '智能', '机器']
            },
            {
                'type': 'translation_check',
                'source': 'Hello, how are you?',
                'reference': '你好,你好吗?',
                'expected_acceptable': True
            },
        ]
    
    def run(self, model: nn.Module, device: str = 'cpu') -> TestResult:
        """运行语言理解测试"""
        correct = 0
        total = len(self.tasks)
        details = {'tasks': []}
        
        for task in self.tasks:
            result = self._evaluate_task(model, task, device)
            if result['correct']:
                correct += 1
            details['tasks'].append(result)
        
        score = correct / total
        
        return TestResult(
            test_name=self.name,
            level=self.level,
            score=score,
            threshold=self.threshold,
            passed=score >= self.threshold,
            details=details,
        )
    
    def _evaluate_task(self, model: nn.Module, task: Dict, device: str) -> Dict:
        """评估单个任务"""
        task_type = task['type']
        
        if task_type == 'sentiment':
            # 情感分析任务
            predicted = self._predict_sentiment(model, task['text'], device)
            correct = predicted == task['expected']
            return {'type': task_type, 'predicted': predicted, 'correct': correct}
        
        elif task_type == 'summary':
            # 摘要任务
            predicted = self._generate_summary(model, task['text'], device)
            keyword_hits = sum(1 for kw in task['expected_keywords'] if kw in predicted)
            correct = keyword_hits >= len(task['expected_keywords']) * 0.5
            return {'type': task_type, 'keyword_hits': keyword_hits, 'correct': correct}
        
        else:
            return {'type': task_type, 'correct': False}
    
    def _predict_sentiment(self, model: nn.Module, text: str, device: str) -> str:
        """预测情感"""
        if hasattr(model, 'predict_sentiment'):
            return model.predict_sentiment(text)
        else:
            # 模拟
            return random.choice(['positive', 'negative'])
    
    def _generate_summary(self, model: nn.Module, text: str, device: str) -> str:
        """生成摘要"""
        if hasattr(model, 'summarize'):
            return model.summarize(text)
        else:
            # 返回原文的一部分作为模拟
            return text[:50]


class CodeGenerationTest(BaseTest):
    """
    代码生成测试
    
    评估模型的编程能力,类似HumanEval风格。
    """
    
    def __init__(self, threshold: float = 0.70):
        super().__init__(
            name="Code Generation Test (HumanEval-style)",
            level=1,
            threshold=threshold
        )
        self.problems = self._generate_problems()
    
    def _generate_problems(self) -> List[Dict]:
        """生成编程问题"""
        return [
            {
                'prompt': 'def add(a, b):\n    """返回两个数的和"""\n',
                'test_cases': [(1, 2, 3), (0, 0, 0), (-1, 1, 0)],
                'function_name': 'add'
            },
            {
                'prompt': 'def is_even(n):\n    """判断一个数是否为偶数"""\n',
                'test_cases': [(2, True), (3, False), (0, True), (-2, True)],
                'function_name': 'is_even'
            },
            {
                'prompt': 'def reverse_string(s):\n    """反转字符串"""\n',
                'test_cases': [('abc', 'cba'), ('', ''), ('a', 'a')],
                'function_name': 'reverse_string'
            },
            {
                'prompt': 'def factorial(n):\n    """计算阶乘"""\n',
                'test_cases': [(0, 1), (1, 1), (5, 120)],
                'function_name': 'factorial'
            },
        ]
    
    def run(self, model: nn.Module, device: str = 'cpu') -> TestResult:
        """运行代码生成测试"""
        correct_problems = 0
        total = len(self.problems)
        details = {'problems': []}
        
        for problem in self.problems:
            result = self._evaluate_problem(model, problem, device)
            if result['passed']:
                correct_problems += 1
            details['problems'].append(result)
        
        score = correct_problems / total
        
        return TestResult(
            test_name=self.name,
            level=self.level,
            score=score,
            threshold=self.threshold,
            passed=score >= self.threshold,
            details=details,
        )
    
    def _evaluate_problem(self, model: nn.Module, problem: Dict, device: str) -> Dict:
        """评估单个编程问题"""
        # 生成代码
        generated_code = self._generate_code(model, problem['prompt'], device)
        
        # 测试生成的代码
        passed = 0
        total_tests = len(problem['test_cases'])
        
        try:
            # 创建执行环境
            local_vars = {}
            exec(generated_code, {}, local_vars)
            func = local_vars.get(problem['function_name'])
            
            if func is not None:
                for test_input, expected in problem['test_cases']:
                    try:
                        if isinstance(test_input, tuple):
                            result = func(*test_input)
                        else:
                            result = func(test_input)
                        
                        if result == expected:
                            passed += 1
                    except:
                        pass
        except:
            pass
        
        return {
            'prompt': problem['prompt'][:50] + '...',
            'passed_all': passed == total_tests,
            'passed': passed == total_tests,
            'test_results': f"{passed}/{total_tests}"
        }
    
    def _generate_code(self, model: nn.Module, prompt: str, device: str) -> str:
        """生成代码"""
        if hasattr(model, 'generate_code'):
            return model.generate_code(prompt)
        else:
            # 模拟: 返回简单的实现
            if 'add' in prompt:
                return prompt + '    return a + b\n'
            elif 'is_even' in prompt:
                return prompt + '    return n % 2 == 0\n'
            elif 'reverse_string' in prompt:
                return prompt + '    return s[::-1]\n'
            elif 'factorial' in prompt:
                return prompt + '    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n'
            else:
                return prompt + '    pass\n'
