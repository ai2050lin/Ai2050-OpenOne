# -*- coding: utf-8 -*-
"""P0.4 题库类别映射 v1（G5-DP01 P0.4）：CMMLU 科目 → 图谱三大类，冻结版。

纪律：
- 本文件是冻结的映射表（category_map.v1）。改动映射 = 新版本文件 + 灵敏度检查，不改本表。
- 多标签允许（一个科目可属多类）；C-Eval val 三类保留为外部锚，不进功效计算。
- 映射依据（AGENTS.md 行为门语义）：
  * 语法规则 = 汉语语言/语文运用：古代汉语、现代汉语、汉语言文学、小学语文（核心）
    + 新闻学、语文教师资格（运用扩展，灵敏度检查项）。
  * 功能推理 = 逻辑与规则运用：逻辑学、公务员（核心）
    + 法理学、专业法律、法律与道德基础、国际法、大学法律（法律规则运用家族，灵敏度检查项）。
  * 知识网络 = 生物/历史/地理系列：高中生物、中国历史、世界历史、高中地理（核心）
    + 解剖学、遗传学、病毒学（生物学家族，灵敏度检查项）。
"""
from __future__ import annotations

VERSION = 'category_map.v1'
FROZEN_AT = '2026-10-31'
SEED_NOTE = 'pool 构建为确定性映射（无抽样）；后续分层抽样种子在 P1 预注册，默认 20261031'

GRAMMAR = ['ancient_chinese', 'modern_chinese', 'chinese_literature', 'elementary_chinese',
           'journalism', 'chinese_teacher_qualification']
REASONING = ['logical', 'chinese_civil_service_exam', 'jurisprudence', 'professional_law',
             'legal_and_moral_basis', 'international_law', 'college_law']
KNOWLEDGE = ['high_school_biology', 'anatomy', 'genetics', 'virology',
             'chinese_history', 'world_history', 'high_school_geography']

# 多标签映射：subject → [类别]
MAP = {}
for _s in GRAMMAR:
    MAP.setdefault(_s, []).append('语法规则')
for _s in REASONING:
    MAP.setdefault(_s, []).append('功能推理')
for _s in KNOWLEDGE:
    MAP.setdefault(_s, []).append('知识网络')

# 灵敏度检查口径：只留各家族核心科目（去扩展项）后各类题数仍应报告
CORE_ONLY = {
    '语法规则': ['ancient_chinese', 'modern_chinese', 'chinese_literature', 'elementary_chinese'],
    '功能推理': ['logical', 'chinese_civil_service_exam'],
    '知识网络': ['high_school_biology', 'chinese_history', 'world_history', 'high_school_geography'],
}

CATEGORIES = ['知识网络', '功能推理', '语法规则']


def categories_of(subject: str) -> list:
    return list(MAP.get(subject, []))


def is_mapped(subject: str) -> bool:
    return subject in MAP
