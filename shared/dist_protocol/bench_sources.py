# -*- coding: utf-8 -*-
"""题库数据集的来源登记表（纯标准库）：源地址、拆分、许可、以及在本机怎么取到题。

为什么要有这张表：从 1.0.24 起，发布包里不再带题库正文（questions.v3.jsonl，3.3 MB，且
C-Eval / M3Exam 是 CC BY-NC-SA 4.0，随包分发等于再分发一份副本）。中心只保存「有哪些数据集、
各多少题、许可是什么、原始数据在哪」这些元信息，题目本身按源地址现取。

于是「数据集」这件事在三个地方各要一个答案，都从这里取，不各写一份：
  界面     显示数据集名、题量、许可、源地址（可点开原始数据页）
  中心     /api/lang/atlas/datasets、/api/lang/bench/overview 只用元信息作答，不需要正文
  本机     scripts/download_lang_bench_v3.py 按这里的 repo/split 下载、校验、构建 questions.v3.jsonl

字段口径：
  id          题库正文里的 source 原值（'C-Eval val'），与 questions.v3.jsonl 的 source 一一对应
  prefix      短名（'C-Eval'），结果表 results.v2 的键用它拼 `${prefix}|${model}`
  dataset_key 稳定 slug（'ceval'），跨表 join 用；改名不影响历史数据
  repo        上游仓库（Hugging Face dataset id），空串表示没有可直达的镜像
  repo_url    给人点开的数据集页
  homepage    数据集自己的主页（论文/榜单），没有就空
  split       用的拆分
  license     数据许可原样照抄上游，不推断
  license_url 许可证正文地址（CC 系列给 deeds 页）
  files       构建时实际读取的文件（相对 repo 的路径），给复核用
  note        一句话说明这个数据集在本平台里怎么用

不要在这里写题量：题量属于「题库正文统计」，由 bench_index 从正文或 meta 摘要得出。
把题量写进常量表，题库一升级就会和实际行数对不上。
"""
from __future__ import annotations

from typing import Any, Dict, List

SCHEMA_SOURCES = 'lang_bench_sources.v1'

CC_BY_NC_SA = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'

SOURCES: Dict[str, Dict[str, Any]] = {
    'C-Eval': {
        'id': 'C-Eval val',
        'dataset_key': 'ceval',
        'prefix': 'C-Eval',
        'label': 'C-Eval',
        'full_name': 'C-Eval 中文基础模型评估套件',
        'repo': 'ceval/ceval-exam',
        'repo_url': 'https://huggingface.co/datasets/ceval/ceval-exam',
        'homepage': 'https://cevalbenchmark.com/',
        'split': 'val',
        'license': 'CC BY-NC-SA 4.0',
        'license_url': CC_BY_NC_SA,
        'files': ['val/**/*.csv (7 科目)'],
        'note': '中文选择题；三类各取本数据集里映射得到的科目，作为「中文侧」的对照。',
    },
    'MMLU': {
        'id': 'MMLU test',
        'dataset_key': 'mmlu',
        'prefix': 'MMLU',
        'label': 'MMLU',
        'full_name': 'MMLU（Massive Multitask Language Understanding）',
        'repo': 'cais/mmlu',
        'repo_url': 'https://huggingface.co/datasets/cais/mmlu',
        'homepage': 'https://arxiv.org/abs/2009.03300',
        'split': 'test',
        'license': 'MIT',
        'license_url': 'https://opensource.org/license/mit',
        'files': ['all/test-00000-of-00001.parquet'],
        'note': '英文 4 选 1，57 科目；题量最大，是跨学科正确率的主盘。',
    },
    'M3Exam': {
        'id': 'M3Exam zh test',
        'dataset_key': 'm3exam',
        'prefix': 'M3Exam',
        'label': 'M3Exam',
        'full_name': 'M3Exam 多语言多模态多层级考试题（中文 test）',
        'repo': 'chiayewken/m3exam',
        'repo_url': 'https://huggingface.co/datasets/chiayewken/m3exam',
        'homepage': 'https://github.com/DamienLiang/M3Exam',
        'split': 'chinese/test',
        'license': 'CC BY-NC-SA 4.0',
        'license_url': CC_BY_NC_SA,
        'files': ['chinese/test-00000-of-00001-a9ef43568bebf5d0.parquet'],
        'note': '中文考试题，8 科目；许可为 NC-SA，只能用于非商业研究，转载须同许可。',
    },
}

# 展示顺序：中文小盘在前（几分钟能跑完），大盘在后。与界面方块顺序一致。
ORDER: List[str] = ['C-Eval', 'MMLU', 'M3Exam']


def get(prefix: str) -> Dict[str, Any]:
    """按短名取登记项；没有就返回空 dict（不编一个出来）。"""
    return dict(SOURCES.get(prefix or '') or {})


def for_dataset_key(key: str) -> Dict[str, Any]:
    for src in SOURCES.values():
        if src['dataset_key'] == key:
            return dict(src)
    return {}


def public_rows() -> List[Dict[str, Any]]:
    """给接口 / 界面用：只出对外字段，按 ORDER 排。"""
    keys = [k for k in ORDER if k in SOURCES] + [k for k in SOURCES if k not in ORDER]
    out = []
    for k in keys:
        s = SOURCES[k]
        out.append({
            'prefix': s['prefix'], 'id': s['id'], 'dataset_key': s['dataset_key'],
            'label': s['label'], 'full_name': s['full_name'],
            'repo': s['repo'], 'repo_url': s['repo_url'], 'homepage': s['homepage'],
            'split': s['split'], 'license': s['license'], 'license_url': s['license_url'],
            'note': s['note'],
        })
    return out


def fetch_hint(prefix: str = '') -> str:
    """在本机把题库取下来的命令（一句，可直接粘）。中心不执行这条命令，只展示。"""
    return ('cd <项目根目录> && python scripts/download_lang_bench_v3.py'
            if not prefix else
            f'cd <项目根目录> && python scripts/download_lang_bench_v3.py --only {get(prefix).get("dataset_key", "")}')


def notice() -> str:
    """一句话口径说明：为什么包里没有题目。中心与界面都要用同一句，避免两边说法不同。"""
    return ('题目正文不随发布包分发：C-Eval / M3Exam 为 CC BY-NC-SA 4.0，MMLU 为 MIT。'
            '包内只有题量、科目、三类分布与源地址；题目按源地址在本机取（scripts/download_lang_bench_v3.py）。')
