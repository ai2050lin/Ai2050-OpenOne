# -*- coding: utf-8 -*-
"""题库索引（���标准库）：题库正文 → 数据集清单 → 两个接口的响应体。

为什么单独一个模块：同一句话「有哪些数据集、各多少题、哪些题能映射三类」现在有三处要回答——
  本机 GPU 后端（server/lang_bench_service、server/lang_atlas_service）
  中心服务（deploy/platform_service 的 /api/lang/atlas/datasets、/api/lang/bench/overview）
  离线数据包（deploy/datapack 的 /api/lang/bench/overview）
三处各写一份，题库一换（v1 → v2 → v3）就会各漂一点，界面上出现「同一个数据集两处题量不同」。
这里按题库正文算一次，三方共用同一份输出。

为什么不能直接 import server/ 下的服务：它们顶层 `import torch`，中心服务器不装 GPU 依赖。
本模块只用标准库 + 同包的映射表，任何 Python 3 环境都能跑。

字段口径（也是「数据集相关的数据都要带上的字段」）：
  id          题库里的 source 原值（'C-Eval val' / 'MMLU test' / 'M3Exam zh test'），主键
  prefix      短名（'C-Eval'），结果表 results.v2 的键用它拼 `${prefix}|${model}`
  dataset_key 稳定 slug（'ceval' / 'mmlu' / 'm3exam'），给跨表 join 用，改名不影响历史数据
  label       显示名
  license     数据许可（从 meta 的 licenses 读；没有就空，不猜）
  n_questions 题库正文里的实际行数（不是 meta 里手写的数）
  by_category 三类各多少题（atlas 形状带 subjects / missing，bench 形状只给计数）
  n_unmapped  映射不到三类的题量：进总正确率，不进任何一类
  testable    是否有可映射三类的题：false 的数据集只能做总正确率基线
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:                                     # 包内相对导入（正常路径）
    from . import bench_sources as SRC
    from .bench_categories import ALL_CATS, CATEGORY_SUBJECTS
except ImportError:                       # 以 shared/dist_protocol 为根直接跑脚本时退回绝对导入
    from shared.dist_protocol import bench_sources as SRC
    from shared.dist_protocol.bench_categories import ALL_CATS, CATEGORY_SUBJECTS

SCHEMA_INDEX = 'lang_bench_index.v4'
STRICT_CATEGORIES = list(ALL_CATS)        # 知识网络 / 功能推理 / 语法规则

# 短名 ↔ slug：既用于把题库 source 归一，也写进结果表，历史结果可按 slug 反查数据集。
PREFIX_TO_KEY = {'C-Eval': 'ceval', 'MMLU': 'mmlu', 'M3Exam': 'm3exam',
                 'CMMLU': 'cmmlu', 'CMExam': 'cmexam', 'Xiezhi': 'xiezhi'}
KEY_TO_PREFIX = {v: k for k, v in PREFIX_TO_KEY.items()}

# 数据集来源识别顺序：长名在前，避免 'MMLU' 抢先匹配掉别的名字。
_PREFIX_ORDER = ('C-Eval', 'M3Exam', 'MMLU', 'CMMLU', 'CMExam', 'Xiezhi')

QUESTIONS_NAMES = ('questions.v3.jsonl', 'questions.v2.jsonl', 'ceval_val.v1.jsonl')
META_NAMES = ('meta.v3.json', 'meta.json')
RESULTS_V2 = 'results.v2.json'
RESULTS_V1 = 'results.json'

_CACHE: Dict[str, Any] = {'stamp': None, 'rows': None, 'meta': None, 'path': None}


def default_data_dir(root: Optional[Path] = None) -> Path:
    """题库目录：显式参数 > 环境变量 AI2050_LANG_BENCH > 仓库内 data/lang_bench。"""
    if root is not None:
        return Path(root)
    env = os.environ.get('AI2050_LANG_BENCH', '').strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / 'data' / 'lang_bench'


def prefix_of(source: str) -> str:
    """'MMLU test' → 'MMLU'。空值返回空串（不返回 '未知'，免得被当成一个数据集）。"""
    s = (source or '').strip()
    low = s.lower()
    for key in _PREFIX_ORDER:
        if low.startswith(key.lower()):
            return key
    return s.split()[0] if s else ''


def dataset_key(prefix: str) -> str:
    return PREFIX_TO_KEY.get(prefix or '', (prefix or '').lower().replace('-', '') or '')


def questions_path(data_dir: Optional[Path] = None) -> Path:
    d = default_data_dir(data_dir)
    for name in QUESTIONS_NAMES:
        p = d / name
        if p.is_file():
            return p
    return d / QUESTIONS_NAMES[-1]


def meta(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    d = default_data_dir(data_dir)
    for name in META_NAMES:
        p = d / name
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except ValueError:
                return {}
    return {}


def _license_map(meta_doc: Dict[str, Any]) -> Dict[str, str]:
    """licenses 按 source 记（'C-Eval val': 'CC BY-NC-SA 4.0'）；也接受按 prefix 记的旧写法。"""
    out: Dict[str, str] = {}
    for k, v in (meta_doc.get('licenses') or {}).items():
        out[str(k)] = str(v)
    return out


def _blank_row(src: str, prefix: str) -> Dict[str, Any]:
    """一行数据集统计的骨架。正文路径与 meta 路径共用，避免两条路字段不一致。"""
    return {'id': src, 'prefix': prefix, 'dataset_key': dataset_key(prefix), 'label': prefix,
            'n_questions': 0, 'by_source': {}, 'subjects': {}, 'cat_counts': {},
            'cat_subjects': {c: {} for c in STRICT_CATEGORIES}}


def _rows_from_questions(qp: Path) -> Dict[str, Dict[str, Any]]:
    """正文路径：逐行统计题量、科目、三类可映射数。"""
    rows: Dict[str, Dict[str, Any]] = {}
    with open(qp, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                q = json.loads(line)
            except ValueError:
                continue                      # 半行不炸整份清单：跳过并继续计数
            src = q.get('source') or ''
            prefix = prefix_of(src)
            if not prefix:
                continue
            row = rows.setdefault(prefix, _blank_row(src, prefix))
            row['n_questions'] += 1
            row['by_source'][src] = row['by_source'].get(src, 0) + 1
            subj = q.get('subject') or ''
            subj_cn = q.get('subject_cn') or subj
            if subj:
                cur = row['subjects'].setdefault(subj, {'key': subj, 'name': subj_cn, 'n': 0})
                cur['n'] += 1
            # 类别优先用题库正文里的 category（v3 起只存三类或 null）；
            # 旧题库（v2）的 category 混着数据集题材，不能用——按映射表重算一遍。
            cat = q.get('category')
            if cat not in STRICT_CATEGORIES:
                mapped = next((c for c in STRICT_CATEGORIES if subj in (CATEGORY_SUBJECTS[c].get(src) or [])), None)
                cat = mapped
            if cat:
                row['cat_counts'][cat] = row['cat_counts'].get(cat, 0) + 1
                if subj:
                    cs = row['cat_subjects'][cat]
                    cs[subj] = cs.get(subj, 0) + 1
    return rows


def _rows_from_meta(meta_doc: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """摘要路径（中心与数据包没有题库正文）：从 meta 的 by_dataset 恢复。

    返回 (rows, exact)。exact=True 表示这份 meta 带 by_dataset——题量、科目、三类分布都是
    建库时按正文数出来的；False 表示只有 by_source 的旧摘要，三类分布无从得知，
    graph_n 会算成 0、testable=False：宁可显示「不可测」，也不按比例编一个数出来。
    """
    rows: Dict[str, Dict[str, Any]] = {}
    by_ds = meta_doc.get('by_dataset') or {}
    for src, block in by_ds.items():
        if not isinstance(block, dict):
            continue
        prefix = prefix_of(src) or str(block.get('prefix') or '')
        if not prefix:
            continue
        row = rows.setdefault(prefix, _blank_row(src, prefix))
        n = int(block.get('n_questions') or 0)
        row['n_questions'] += n
        row['by_source'][src] = row['by_source'].get(src, 0) + n
        for s in block.get('subjects') or []:
            if not isinstance(s, dict) or not s.get('key'):
                continue
            row['subjects'][s['key']] = {'key': s['key'], 'name': s.get('name') or s['key'],
                                         'n': int(s.get('n') or 0)}
        for cat, cn in (block.get('by_category') or {}).items():
            if cat in STRICT_CATEGORIES:
                row['cat_counts'][cat] = row['cat_counts'].get(cat, 0) + int(cn or 0)
    exact = bool(by_ds)
    # by_dataset 缺某个来源时用 by_source 补题量（旧 meta 只有这一段）
    for src, n in (meta_doc.get('by_source') or {}).items():
        prefix = prefix_of(src)
        if not prefix:
            continue
        if src in (rows.get(prefix) or {}).get('by_source', {}):
            continue
        row = rows.setdefault(prefix, _blank_row(src, prefix))
        row['n_questions'] += int(n or 0)
        row['by_source'][src] = row['by_source'].get(src, 0) + int(n or 0)
    return rows, exact


def _rows(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """按数据集分组统计。缓存键带上题库文件与 meta 的路径 / mtime / 大小：换题库即重算。

    两条路：有题库正文就按正文数（exact=True）；没有就走 meta 摘要（中心与离线数据包）。
    从 1.0.24 起发布包不带正文，中心走的就是后者——两条路必须给出同样的题量、科目与三类
    分布，meta 的 by_dataset 由同一份正文统计而来（scripts/download_lang_bench_v3.py --meta-only）。
    """
    d = default_data_dir(data_dir)
    qp = questions_path(d)
    mp = next((d / n for n in META_NAMES if (d / n).is_file()), None)
    stamp = (str(qp), qp.stat().st_mtime_ns if qp.is_file() else 0,
             qp.stat().st_size if qp.is_file() else 0,
             str(mp) if mp else '', mp.stat().st_mtime_ns if mp else 0)
    if _CACHE.get('stamp') == stamp:
        return _CACHE['rows'] or []

    meta_doc = meta(d)
    lic = _license_map(meta_doc)
    body = qp.is_file()
    if body:
        rows = _rows_from_questions(qp)
        exact = True
    else:
        rows, exact = _rows_from_meta(meta_doc)

    out: List[Dict[str, Any]] = []
    for prefix, row in rows.items():
        by_cat: Dict[str, Dict[str, Any]] = {}
        for cat in STRICT_CATEGORIES:
            want = list((CATEGORY_SUBJECTS.get(cat) or {}).get(row['id']) or [])
            have = [s for s in want if s in row['subjects']]
            by_cat[cat] = {
                'n': row['cat_counts'].get(cat, 0),
                'subjects': [{'key': s, 'name': row['subjects'][s]['name'], 'n': row['subjects'][s]['n']} for s in have],
                'missing': [s for s in want if s not in row['subjects']],
            }
        graph_n = sum(v['n'] for v in by_cat.values())
        row['by_category'] = by_cat
        row['graph_n'] = graph_n
        row['n_subjects'] = len(row['subjects'])
        row['n_unmapped'] = row['n_questions'] - graph_n
        row['testable'] = graph_n > 0
        info = SRC.get(prefix)
        row['license'] = info.get('license') or lic.get(row['id']) or lic.get(prefix) or ''
        row['license_url'] = info.get('license_url') or ''
        row['source_url'] = info.get('repo_url') or ''
        row['source_repo'] = info.get('repo') or ''
        row['source_split'] = info.get('split') or ''
        row['source_homepage'] = info.get('homepage') or ''
        row['source_full_name'] = info.get('full_name') or prefix
        row['source_note'] = info.get('note') or ''
        # 这份统计是从哪来的：questions = 本机有题库正文，逐行数出来；meta = 只有摘要。
        # 界面据此标注「题量为建库时统计」，避免把摘要当成刚数过的数。
        row['stats_source'] = 'questions' if body else ('meta' if row['n_questions'] else 'unknown')
        row['exact'] = exact
        if graph_n:
            row['reason'] = ''
        elif not exact:
            row['reason'] = ('数据集清单来自 meta 摘要（包里不带题库正文），这份摘要没有按数据集的三类分布，'
                             '无法判定可测性。')
        else:
            row['reason'] = '该数据集没有可映射到三大类型的科目，只能做综合能力基线，不能做类型对比。'
        row.pop('cat_counts', None)
        row.pop('cat_subjects', None)
        out.append(row)

    out.sort(key=lambda r: -r['n_questions'])
    _CACHE.update(stamp=stamp, rows=out, meta=meta_doc, path=str(qp))
    return out


def datasets_rows(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """每个数据集一行，带三类映射依据（/api/lang/atlas/datasets 的 datasets 字段）。"""
    rows = []
    for r in _rows(data_dir):
        row = {k: v for k, v in r.items() if k != 'subjects'}
        row['subjects'] = sorted(r['subjects'].values(), key=lambda s: -s['n'])
        rows.append(row)
    return rows


def datasets_payload(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """GET /api/lang/atlas/datasets 的完整响应体（本机后端与中心同形）。

    mode 说明这份清单怎么来的：'questions' = 本机有题库正文、逐行数出来；'meta' = 包里只有
    meta 摘要（中心就是这种）。两种模式下 available 都为真——题量、科目、三类分布、
    许可与源地址都是完整信息，缺的只是题目正文本身，而正文按源地址现取。
    """
    d = default_data_dir(data_dir)
    qp = questions_path(d)
    try:
        rows = datasets_rows(d)
    except Exception as exc:                  # 题库损坏时给空清单，不 500
        return {'schema': 'lang_atlas_datasets.v2', 'available': False,
                'reason': f'{type(exc).__name__}: {exc}', 'datasets': [],
                'sources': SRC.public_rows(), 'notice': SRC.notice()}
    from .bench_categories import ALL_CATS as CATS  # 别名只在这里用，避免与上面的 STRICT 混
    local = qp.is_file()
    return {
        'schema': 'lang_atlas_datasets.v2',
        'available': bool(rows),
        'categories': [{'id': c, 'name': c, 'also': '分析推理' if c == '功能推理' else None} for c in CATS],
        'datasets': rows,
        'sources': SRC.public_rows(),
        'notice': SRC.notice(),
        'source': {'questions': qp.name if local else '', 'n_questions': sum(r['n_questions'] for r in rows),
                   'mode': 'questions' if local else 'meta',
                   'fetch': SRC.fetch_hint(), 'fetch_scope': '本机'},
        'note': '类型 ↔ 科目的映射是显式表（shared/dist_protocol/bench_categories.py），不是模型推断；'
                '某数据集若没有可映射科目，只做综合能力基线，不参与类型对比。',
    }


def bench_rows(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """数据集清单的 bench 形状：by_category 只给计数（int），供图谱测试题那一栏算「三类各多少题」。

    source_url / license_url 一并带上：界面在「图谱测试题」标题下直接给出原始数据地址，
    用户不用来这里翻代码就知道这些题从哪来、以什么许可使用。
    """
    out = []
    for r in _rows(data_dir):
        out.append({
            'id': r['id'], 'prefix': r['prefix'], 'dataset_key': r['dataset_key'], 'label': r['label'],
            'n_questions': r['n_questions'], 'n_subjects': r['n_subjects'],
            'by_source': dict(r['by_source']),
            'by_category': {c: v['n'] for c, v in r['by_category'].items() if v['n']},
            'graph_n': r['graph_n'], 'n_unmapped': r['n_unmapped'],
            'testable': r['testable'], 'license': r['license'],
            'license_url': r.get('license_url', ''), 'source_url': r.get('source_url', ''),
            'source_repo': r.get('source_repo', ''), 'source_split': r.get('source_split', ''),
            'source_full_name': r.get('source_full_name', ''), 'reason': r.get('reason', ''),
            'stats_source': r.get('stats_source', ''), 'exact': r.get('exact', False),
        })
    return out


def results_payload(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """结果表：by_key（按 `${prefix}|${model}` 分键）+ legacy（v1 混跑一份，只读）。
    by_key 里每条都补齐 dataset_key / dataset_id / prefix —— 只有 dataset 文案的结果没法 join。"""
    d = default_data_dir(data_dir)
    by_key: Dict[str, Any] = {}
    v2 = d / RESULTS_V2
    if v2.is_file():
        try:
            by_key = json.loads(v2.read_text(encoding='utf-8')).get('by_key') or {}
        except (ValueError, AttributeError):
            by_key = {}
    for key, rec in list(by_key.items()):
        if not isinstance(rec, dict):
            by_key.pop(key, None)
            continue
        prefix = key.split('|')[0]
        rec.setdefault('dataset', prefix)
        rec.setdefault('prefix', prefix)
        rec.setdefault('dataset_key', dataset_key(prefix))
        if not rec.get('dataset_id'):
            rid = next((r['id'] for r in _rows(d) if r['prefix'] == prefix), '')
            if rid:
                rec['dataset_id'] = rid
    legacy = None
    v1 = d / RESULTS_V1
    if v1.is_file():
        try:
            r = json.loads(v1.read_text(encoding='utf-8'))
            legacy = {k: r[k] for k in ('model_name', 'n_questions', 'correct', 'acc', 'elapsed_s',
                                        'finished_at', 'by_category', 'by_subject') if k in r}
        except (KeyError, ValueError):
            legacy = None
    return {'by_key': by_key, 'legacy': legacy}


def overview_payload(data_dir: Optional[Path] = None, *, model_status: str = 'unavailable',
                     loaded_model: str = '', run_state: str = 'idle',
                     running: Optional[Dict[str, str]] = None,
                     last_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET /api/lang/bench/overview 的响应体。中心与数据包没有 GPU：
    model_status 传 'unavailable'，客户端据此把「运行实测」换成「本机执行」的指引；
    历史结果照常可读——中心保存的是别人跑完上传的读数，自己不做任何前向计算。"""
    d = default_data_dir(data_dir)
    rows = bench_rows(d)
    res = results_payload(d)
    local = questions_path(d).is_file()
    return {
        'schema': 'lang_bench_overview.v4',
        'available': bool(rows) or bool(res['by_key']) or bool(res['legacy']),
        'meta': meta(d),
        'datasets': rows,
        'sources': SRC.public_rows(),
        'notice': SRC.notice(),
        'results': res['by_key'],
        'legacy': res['legacy'],
        'graph_categories': STRICT_CATEGORIES,
        'model_status': model_status,
        'model_dir': None,
        'loaded_model': loaded_model,
        'run_state': run_state,
        'running': running or {'dataset': '', 'model_name': ''},
        'last_result': last_result if last_result is not None else res['legacy'],
        'source': {'questions': questions_path(d).name if local else '',
                   'mode': 'questions' if local else 'meta',
                   'n_questions': sum(r['n_questions'] for r in rows),
                   'fetch': SRC.fetch_hint(), 'fetch_scope': '本机'},
    }


def clear_cache() -> None:
    """测试或运行中换题库时用（正常路径靠 mtime 自动失效）。"""
    _CACHE.update(stamp=None, rows=None, meta=None, path=None)
