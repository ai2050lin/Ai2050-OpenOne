# -*- coding: utf-8 -*-
"""题库数据集 ↔ 三类（知识网络 / 功能推理 / 语法规则）的显式映射表。

唯一真源：下载脚本生成题库的 `category` 字段用它，语言图谱「按学科重算」也用它。
改映射只能改这里——接口会把依据一起返回，便于审阅。

键是题库里的 `source`（数据集标识），值是该类对应的 `subject` 列表（题库里存的英文键）。
**不在表里的 subject 就是「不能映射三类」**，一律返回 None：
不给兜底类别（比如把剩下的都塞进「综合知识」），兜底等于编造口径，
后续所有「三类各多少题」都会带着这个假数字往下走。

MMLU 的「语法规则」是空的，这是事实而不是遗漏：MMLU 57 个科目里没有语法 / 语文类，
最近的 professional_law 考的是法律条文适用，不是语句结构。空着比硬凑一个科目诚实。
"""

from typing import Dict, List, Optional

ALL_CATS: List[str] = ['知识网络', '功能推理', '语法规则']

CATEGORY_SUBJECTS: Dict[str, Dict[str, List[str]]] = {
    '知识网络': {
        # C-Eval：7 个 val 科目里的三个记忆型学科
        'C-Eval val': ['high_school_biology', 'high_school_geography', 'high_school_history'],
        # MMLU：学科事实型（历史 / 地理 / 理化生 / 天文 / 宗教 / 史前）
        'MMLU test': ['high_school_biology', 'high_school_geography', 'high_school_chemistry',
                      'high_school_physics', 'astronomy', 'world_religions',
                      'high_school_world_history', 'high_school_european_history',
                      'high_school_us_history', 'global_facts', 'prehistory'],
        # M3Exam 中文：中学五科
        'M3Exam zh test': ['history', 'geography', 'biology', 'chemistry', 'physics'],
        # 旧题库（v2）保留键：v2 文件还在磁盘上，离线数据包可能仍引用
        'CMMLU test': ['high_school_biology', 'high_school_geography', 'world_history', 'chinese_history',
                       'global_facts', 'astronomy', 'world_religions', 'high_school_chemistry',
                       'high_school_physics'],
    },
    '功能推理': {
        'C-Eval val': ['civil_servant', 'logic'],
        'MMLU test': ['formal_logic', 'logical_fallacies'],
        'M3Exam zh test': ['math'],
        'CMMLU test': ['logical', 'chinese_civil_service_exam'],
    },
    '语法规则': {
        'C-Eval val': ['chinese_language_and_literature', 'high_school_chinese'],
        'MMLU test': [],                       # 无对应科目：如实为空
        'M3Exam zh test': ['chinese'],
        'CMMLU test': ['modern_chinese', 'ancient_chinese', 'elementary_chinese'],
    },
}


def category_of(source: str, subject: str) -> Optional[str]:
    """(数据集, 学科) → 三类之一；映射不到返回 None（不是「综合知识」）。"""
    for cat in ALL_CATS:
        if subject in (CATEGORY_SUBJECTS[cat].get(source) or []):
            return cat
    return None


def subjects_of(dataset: str) -> Dict[str, List[str]]:
    """某个数据集在三类下各自的学科清单（供接口返回映射依据）。"""
    return {cat: list(CATEGORY_SUBJECTS[cat].get(dataset) or []) for cat in ALL_CATS}
