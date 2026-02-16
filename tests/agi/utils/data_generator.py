"""
测试数据生成器
=============

为各类测试生成标准化的测试数据。
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


class DataGenerator:
    """测试数据生成器"""
    
    def __init__(self, seed: int = 42, output_dir: str = 'agi_test/data'):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子
            output_dir: 数据输出目录
        """
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_all(self):
        """生成所有测试数据"""
        print("Generating test data...")
        
        # Level 1: 基础数据
        self.generate_knowledge_data()
        self.generate_language_data()
        self.generate_code_data()
        
        # Level 2: 泛化数据
        self.generate_arc_data()
        self.generate_compositional_data()
        self.generate_ood_data()
        
        # Level 3: 自主能力数据
        self.generate_planning_data()
        self.generate_tool_data()
        
        # Level 4: 几何数据
        self.generate_geometric_data()
        
        print(f"Data generated in {self.output_dir}")
    
    def generate_knowledge_data(self) -> List[Dict]:
        """生成知识测试数据"""
        data = {
            'science': [
                {'q': '光速约为多少km/s?', 'choices': ['3000', '30000', '300000', '3000000'], 'a': 2},
                {'q': '水的化学式?', 'choices': ['H2O', 'CO2', 'O2', 'N2'], 'a': 0},
                {'q': '地球绕太阳一周需要多长时间?', 'choices': ['一天', '一月', '一年', '十年'], 'a': 2},
            ],
            'history': [
                {'q': '第一次世界大战开始于?', 'choices': ['1914', '1918', '1939', '1945'], 'a': 0},
                {'q': '中华人民共和国成立时间?', 'choices': ['1945', '1949', '1950', '1952'], 'a': 1},
            ],
            'math': [
                {'q': '圆周率π约为?', 'choices': ['3.14', '2.71', '1.41', '1.73'], 'a': 0},
                {'q': '勾股定理中,直角三角形三边关系?', 
                 'choices': ['a+b=c', 'a²+b²=c²', 'a×b=c', 'a÷b=c'], 'a': 1},
            ],
        }
        
        self._save_data('knowledge.json', data)
        return data
    
    def generate_language_data(self) -> List[Dict]:
        """生成语言理解数据"""
        data = {
            'sentiment': [
                {'text': '这部电影非常精彩!', 'label': 'positive'},
                {'text': '服务太差了,完全不推荐。', 'label': 'negative'},
                {'text': '还可以,中规中矩。', 'label': 'neutral'},
            ],
            'qa': [
                {'q': '中国的首都是哪里?', 'a': '北京'},
                {'q': '一年有多少个月?', 'a': '12个月'},
            ],
        }
        
        self._save_data('language.json', data)
        return data
    
    def generate_code_data(self) -> List[Dict]:
        """生成代码测试数据"""
        data = [
            {
                'id': 1,
                'prompt': 'def add(a, b):\n    """返回两个数的和"""\n',
                'test_cases': [(1, 2, 3), (0, 0, 0), (-1, 1, 0)],
            },
            {
                'id': 2,
                'prompt': 'def is_even(n):\n    """判断是否为偶数"""\n',
                'test_cases': [(2, True), (3, False), (0, True)],
            },
            {
                'id': 3,
                'prompt': 'def factorial(n):\n    """计算阶乘"""\n',
                'test_cases': [(0, 1), (1, 1), (5, 120)],
            },
        ]
        
        self._save_data('code.json', data)
        return data
    
    def generate_arc_data(self) -> List[Dict]:
        """生成ARC风格数据"""
        data = [
            {
                'id': 'arc_001',
                'train': [
                    {'input': [[1, 0], [0, 1]], 'output': [[1, 0, 1], [0, 1, 0], [1, 0, 1]]},
                    {'input': [[0, 1], [1, 0]], 'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]},
                ],
                'test': {'input': [[1, 1], [1, 1]]},
                'pattern': 'expand_with_border',
            },
            {
                'id': 'arc_002',
                'train': [
                    {'input': [[1, 2], [2, 1]], 'output': [[3, 4], [4, 3]]},
                    {'input': [[2, 2], [1, 1]], 'output': [[4, 4], [3, 3]]},
                ],
                'test': {'input': [[1, 1], [2, 2]]},
                'pattern': 'color_swap',
            },
        ]
        
        self._save_data('arc.json', data)
        return data
    
    def generate_compositional_data(self) -> Dict:
        """生成组合泛化数据"""
        operations = ['add', 'multiply', 'subtract']
        
        train = []
        test = []
        
        # 训练数据: 数字0-50的组合
        for i in range(20):
            for j in range(20):
                a, b = random.randint(0, 50), random.randint(0, 50)
                op = random.choice(operations[:2])
                result = self._compute(op, a, b)
                train.append({'op': op, 'a': a, 'b': b, 'result': result})
        
        # 测试数据: 数字50-100的组合 (未见过的范围)
        for i in range(20):
            a, b = random.randint(50, 100), random.randint(50, 100)
            op = random.choice(operations)
            result = self._compute(op, a, b)
            test.append({'op': op, 'a': a, 'b': b, 'result': result})
        
        data = {'train': train, 'test': test}
        self._save_data('compositional.json', data)
        return data
    
    def generate_ood_data(self) -> Dict:
        """生成分布外数据"""
        # ID数据: 标准格式
        id_data = [
            {'text': f'{i} + {j} = ?', 'result': i + j}
            for i in range(10) for j in range(10)
        ][:50]
        
        # OOD数据: 不同格式
        ood_data = [
            {'text': f'{i}加{j}等于多少?', 'result': i + j, 'type': 'format_shift'}
            for i in range(15, 25) for j in range(15, 25)
        ][:20]
        ood_data.extend([
            {'text': f'求{i}与{j}的和', 'result': i + j, 'type': 'style_shift'}
            for i in range(25, 35) for j in range(25, 35)
        ][:20])
        
        data = {'id': id_data, 'ood': ood_data}
        self._save_data('ood.json', data)
        return data
    
    def generate_planning_data(self) -> List[Dict]:
        """生成规划任务数据"""
        data = [
            {
                'goal': '从北京到巴黎旅行,预算有限',
                'min_steps': 4,
                'expected_steps': ['查询航班', '比较价格', '预订机票', '预订酒店'],
            },
            {
                'goal': '开发一个待办事项应用',
                'min_steps': 5,
                'expected_steps': ['需求分析', '设计', '编码', '测试', '部署'],
            },
        ]
        
        self._save_data('planning.json', data)
        return data
    
    def generate_tool_data(self) -> List[Dict]:
        """生成工具使用数据"""
        data = [
            {'task': '计算 12345 * 67890', 'tool': 'calculator', 'result': 838102050},
            {'task': '查找2024年奥运会举办城市', 'tool': 'search', 'result': '巴黎'},
            {'task': '运行代码计算斐波那契第20项', 'tool': 'code_executor', 'result': 6765},
        ]
        
        self._save_data('tool.json', data)
        return data
    
    def generate_geometric_data(self) -> Dict:
        """生成几何测试数据"""
        data = {
            'parallel_transport': {
                'test_vectors': torch.randn(10, 32).tolist(),
                'paths': torch.randn(5, 10, 16).tolist(),
            },
            'holonomy': {
                'closed_paths': self._generate_closed_paths(5),
            },
            'curvature': {
                'manifold_points': torch.randn(20, 16).tolist(),
            },
        }
        
        self._save_data('geometric.json', data)
        return data
    
    def _compute(self, op: str, a: int, b: int) -> int:
        """计算运算结果"""
        if op == 'add':
            return a + b
        elif op == 'multiply':
            return a * b
        elif op == 'subtract':
            return a - b
        return 0
    
    def _generate_closed_paths(self, n: int) -> List:
        """生成闭合路径 (dx1 + dx2 + dx3 = 0)"""
        paths = []
        for _ in range(n):
            dx1 = torch.randn(1, 16).tolist()
            dx2 = torch.randn(1, 16).tolist()
            dx3 = [[-x1 - x2 for x1, x2 in zip(dx1[0], dx2[0])]]
            paths.append({'dx1': dx1, 'dx2': dx2, 'dx3': dx3})
        return paths
    
    def _save_data(self, filename: str, data):
        """保存数据到文件"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {filepath}")


if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_all()
