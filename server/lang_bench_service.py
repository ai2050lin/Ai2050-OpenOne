# -*- coding: utf-8 -*-
"""语言图谱测试题（lang_bench.v1）：用公开数据集 C-Eval 的 val 集对本机模型做选择题实测，
给出「知识网络 / 功能推理 / 语法规则」三个图谱大类的真实正确率百分比。

数据真源：data/lang_bench/ceval_val.v1.jsonl（由 scripts/download_lang_bench.py 一次性从
HuggingFace Hub 拉取，169 题、7 个科目，本地数据包自带，运行时不再依赖外网）。
模型真源：复用 live_analysis_service 已加载的本机权重（不重复占显存），逐题一次前向，
在「答案：」位置比较 A/B/C/D 选项字母的 next-token 概率（与官方 loglikelihood 等价，同前缀）。

两个维度分开：**数据集**决定「考哪批题」，**模型**决定「谁去考」。
结果按 (数据集, 模型) 分键存 results.v2.json，不再互相覆盖——否则换数据集或换模型后
看到的仍是上一轮的百分比，读数会张冠李戴。

端点：
  GET  /api/lang/bench/overview   数据集清单（题数/三类可映射题数）+ 模型可用性 + 按 (数据集, 模型) 分键的结果表
  POST /api/lang/bench/run        启动后台实测；body {dataset, model}，只跑该数据集的题
  GET  /api/lang/bench/status     进度（done/total、各大类实时正确率）与最终结果
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, Body

try:
    # 必须与 server.server 导入的是同一份模块；裸名导入会造出第二份 _state，
    # 导致这里以为模型未加载而再加载一份权重（显存翻倍）。
    from server import live_analysis_service as las
    from tools.provenance import provenance as _prov
except ImportError:
    import live_analysis_service as las

# 数据集清单的解析放在 shared/dist_protocol/bench_index.py：中心服务与离线数据包读的是同一份，
# 三处各算一遍就会在题库换版后给出不同题量（v2 → v3 时真出现过）。
from shared.dist_protocol import bench_index as BI

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'lang_bench'
QUESTIONS_PATH_V3 = DATA / 'questions.v3.jsonl'      # 主口径（C-Eval + MMLU + M3Exam，~14.9k 题）
QUESTIONS_PATH_V2 = DATA / 'questions.v2.jsonl'      # 旧口径（C-Eval + CMMLU + CMExam，~42k 题），只读保留
QUESTIONS_PATH_V1 = DATA / 'ceval_val.v1.jsonl'      # 最小题库（仅 C-Eval val 169 题），向后兼容


def _questions_path() -> Path:
    """运行时动态选择题库：v3 下载完成后无需重启即生效；没有 v3 才退回旧题库。"""
    for p in (QUESTIONS_PATH_V3, QUESTIONS_PATH_V2, QUESTIONS_PATH_V1):
        if p.exists():
            return p
    return QUESTIONS_PATH_V1
META_PATH = DATA / 'meta.v3.json' if (DATA / 'meta.v3.json').exists() else DATA / 'meta.json'
RESULTS_PATH = DATA / 'results.json'                # 旧口径：全部题库混合一份（保留只读，不再写入）
RESULTS_V2_PATH = DATA / 'results.v2.json'          # 按 (数据集, 模型) 分键的结果表
PERQ_PATH = DATA / 'results_per_question.json'      # 逐题明细（仅本机留存，gitignore；每次运行覆盖）

router = APIRouter(prefix='/api/lang/bench', tags=['lang-bench'])

CATEGORIES = ['知识网络', '功能推理', '语法规则']   # 排序用；实际以题库 meta 为准
# 三个图谱大类：只有这些类别的题能回答「知识/推理/语法」的进展。
# v3 起 `category` 只存这三类，映射不到就是 null（旧口径把「综合知识 / 医学知识」题材塞进同一字段，
# 按 category 的任何统计都会被污染；v2 文件仍只读保留，不再进主口径）。
GRAPH_CATS = ['知识网络', '功能推理', '语法规则']


_run_state: Dict[str, Any] = {
    'state': 'idle',     # idle | loading | running | done | error
    'error': '', 'done': 0, 'total': 0, 'started_at': 0.0, 'elapsed_s': 0.0,
    'current_subject': '', 'by_category': {}, 'by_subject': {},
    'dataset': '', 'model_name': '', 'result': None,
}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def _dataset_prefix(source: str) -> str:
    """把题库里的 source（'C-Eval val' / 'MMLU test' / 'M3Exam zh test'）归到数据集短名。"""
    return BI.prefix_of(source)


def _dataset_index() -> List[Dict[str, Any]]:
    """数据集清单（bench 形状）：题量 / 三类可映射题量 / 来源拆分。

    解析在 shared/dist_protocol/bench_index.py——中心服务、离线数据包与这里读同一份题库、
    用同一张映射表算；以前三处各写一份，题库换到 v3 后题量与类别开始对不上。
    这里的 by_category 只给计数（int），atlas 端点给的是带 subjects / missing 的形状。
    """
    return BI.bench_rows(DATA)


def _result_key(dataset: str, model_name: str) -> str:
    return f'{_dataset_prefix(dataset)}|{model_name}'


def _load_results() -> Dict[str, dict]:
    """按 (数据集, 模型) 分键的结果表。旧 results.json（全部题库混跑一份）单独放 legacy，
    不混进新表——它的 dataset 字段只是当初的文案，不能反推出它考了哪个数据集。"""
    by_key: Dict[str, dict] = {}
    legacy = None
    if RESULTS_V2_PATH.exists():
        try:
            doc = json.loads(RESULTS_V2_PATH.read_text(encoding='utf-8'))
            by_key = {k: v for k, v in (doc.get('by_key') or {}).items() if isinstance(v, dict)}
        except ValueError:
            by_key = {}
    if RESULTS_PATH.exists():
        try:
            r = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))
            legacy = {k: r[k] for k in ('model_name', 'n_questions', 'correct', 'acc', 'elapsed_s',
                                        'finished_at', 'by_category', 'by_subject') if k in r}
        except (KeyError, ValueError):
            legacy = None
    return {'by_key': by_key, 'legacy': legacy}


def _save_result(dataset: str, result: dict) -> None:
    doc = {'schema': 'lang_bench_results.v2', 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'by_key': {}}
    if RESULTS_V2_PATH.exists():
        try:
            prev = json.loads(RESULTS_V2_PATH.read_text(encoding='utf-8'))
            doc['by_key'] = prev.get('by_key') or {}
        except ValueError:
            pass
    doc['by_key'][_result_key(dataset, result.get('model_name', ''))] = result
    RESULTS_V2_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')


def _loaded_model() -> str:
    d = las._state.get('dir') or las.model_dir()
    return Path(d).name if d else ''


def _model_matches(page_model: str, loaded: str) -> bool:
    """页面上的模型 id（qwen3-4b|cfg-…|bf16）与实载目录名（Qwen3-4B / glm4-9b-chat-hf）是否同一模型。
    只比发布名，不比精度/配置：bench 跑的就是当前载入权重，精度不同不能用同一份结果。"""
    a = (page_model or '').split('|')[0].strip().lower().replace('_', '-')
    b = (loaded or '').strip().lower().replace('_', '-')
    if not a or not b:
        return False
    return a == b or b.startswith(a)


def _load_questions(dataset: str = '') -> List[dict]:
    p = _dataset_prefix(dataset)
    with open(_questions_path(), encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if _dataset_prefix(r.get('source', '')) == p] if p else rows


def _build_prompt(q: dict) -> str:
    """按数据集选提示模板，不能一律用中文模板。

    MMLU 是英文题（14,042 题）：套中文「以下是中国关于…答案：」等于把语言切换也测进去了，
    正确率里混着「模型能不能跟着中文指令做英文题」，这个数不能当学科能力用。
    所以 MMLU 走它的官方英文模板（The following are multiple choice questions … Answer:），
    C-Eval 走官方中文模板（含「中国关于」，是它原版的措辞），M3Exam 是中国考试、但科目是通用名，
    去掉「中国」二字，避免把政治类题目前置成立场提示。支持 A–E 五选项。
    """
    src = q.get('source') or ''
    lines = []
    if src.startswith('MMLU'):
        subj = (q.get('subject') or '').replace('_', ' ')
        lines += [f'The following are multiple choice questions (with answers) about {subj}.', '']
    elif src.startswith('C-Eval'):
        lines += [f'以下是中国关于{q["subject_cn"]}的单项选择题，请选出其中的正确答案。', '']
    else:
        lines += [f'以下是关于{q["subject_cn"]}的单项选择题，请选出其中的正确答案。', '']
    lines.append(q['question'])
    for letter, text in zip('ABCDE', q['choices']):
        lines.append(f'{letter}. {text}')
    lines.append('Answer:' if src.startswith('MMLU') else '答案：')
    return '\n'.join(lines)


def _letter_token_ids(tok) -> Dict[str, List[int]]:
    """每个选项字母的候选 token id（带不带前导空格两种写法都收）。"""
    out: Dict[str, List[int]] = {}
    for letter in 'ABCDE':
        ids = set()
        for form in (letter, ' ' + letter, '　' + letter):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        out[letter] = sorted(ids)
    return out


def _score_one(model, tok, device, letter_ids: Dict[str, List[int]], q: dict) -> str:
    prompt = _build_prompt(q)
    enc = tok(prompt, return_tensors='pt', add_special_tokens=True)
    ids = enc['input_ids'][0][-2048:]
    with las._RUN_LOCK, torch.no_grad():
        logits = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False).logits[0, -1]
    best: Dict[str, float] = {}
    for letter, cand in letter_ids.items():
        if cand:
            best[letter] = float(max(logits[i] for i in cand))
    if not best:
        return '?'
    return max(best, key=best.get)


def _run_work(dataset: str = '') -> None:
    try:
        questions = _load_questions(dataset)
        if not questions:
            raise RuntimeError(f'数据集「{dataset or "全部"}」在题库里没有题。')
        # 等模型就绪（可能正在后台首次加载）
        deadline = time.time() + 600
        while time.time() < deadline:
            if las._state['status'] == 'ready':
                break
            if las._state['status'] == 'idle':
                las._load_async()
            if las._state['status'] == 'error':
                raise RuntimeError(las._state['error'] or '模型不可用')
            time.sleep(1.0)
        else:
            raise RuntimeError('等待模型加载超时（10 分钟）。')
        model, tok = las._state['model'], las._state['tokenizer']
        device = las._state['device']
        letter_ids = _letter_token_ids(tok)
        src_counter: Dict[str, int] = {}
        for q in questions:
            src_counter[q['source'].split()[0]] = src_counter.get(q['source'].split()[0], 0) + 1

        with _LOCK:
            _run_state.update(state='running', done=0, total=len(questions), started_at=time.perf_counter(),
                              error='', by_category={c: [0, 0] for c in CATEGORIES},
                              by_subject={}, current_subject='')
        per_question = []
        for i, q in enumerate(questions):
            t0 = time.perf_counter()
            pred = _score_one(model, tok, device, letter_ids, q)
            ok = int(pred == q['answer'])
            cat, sub = q.get('category'), q['subject_cn']
            with _LOCK:
                _run_state['done'] = i + 1
                _run_state['current_subject'] = sub
                _run_state['elapsed_s'] = round(time.perf_counter() - _run_state['started_at'], 1)
                if cat:                                    # 映射不到三类的题只计总正确率
                    pair = _run_state['by_category'].setdefault(cat, [0, 0])
                    pair[0] += ok
                    pair[1] += 1
                spair = _run_state['by_subject'].setdefault(sub, [0, 0])
                spair[0] += ok
                spair[1] += 1
            per_question.append({'id': q['id'], 'category': cat, 'subject': sub,
                                 'pred': pred, 'answer': q['answer'], 'correct': bool(ok),
                                 'ms': int((time.perf_counter() - t0) * 1000)})

        with _LOCK:
            by_cat = {c: {'n': v[1], 'correct': v[0], 'acc': round(v[0] / v[1], 4) if v[1] else 0.0}
                      for c, v in _run_state['by_category'].items() if v[1]}
            by_sub = {s: {'n': v[1], 'correct': v[0], 'acc': round(v[0] / v[1], 4)}
                      for s, v in _run_state['by_subject'].items()}
            n_total = len(per_question)
            n_ok = sum(1 for r in per_question if r['correct'])
            ds_prefix = _dataset_prefix(dataset) or '全部题库'
            ds_row = next((d for d in _dataset_index() if d['prefix'] == ds_prefix), {})
            result = {
                'schema': 'lang_bench_result.v2',
                'model_name': Path(las._state['dir']).name if las._state['dir'] else '',
                'device': las._state['device'],
                'dataset': ds_prefix,
                # 数据集四件套：短名 / slug / 题库 source 原值 / 许可。
                # 只有 'dataset' 文案的结果无法 join 回题库（'C-Eval val（HuggingFace…）' 那种写法人手打过就变），
                # 所以这里把可程序的都写下来。
                'dataset_key': BI.dataset_key(ds_prefix),
                'dataset_id': ds_row.get('id') or '',
                'dataset_license': ds_row.get('license') or '',
                'dataset_n_questions': ds_row.get('n_questions'),
                'dataset_sources': ' + '.join(f'{k} {v}' for k, v in sorted(src_counter.items())) or 'unknown',
                'n_questions': n_total, 'correct': n_ok,
                'acc': round(n_ok / n_total, 4) if n_total else 0.0,
                'elapsed_s': _run_state['elapsed_s'],
                'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'by_category': by_cat, 'by_subject': by_sub,
                'provenance': _prov(model_path=las._state['dir'] or None, seed=None,
                                    tool='server/lang_bench_service.py'),
            }
            summary = {k: v for k, v in result.items() if k != 'per_question'}
            _run_state['result'] = summary
            _save_result(dataset, summary)                       # 按 (数据集, 模型) 分键，不覆盖别的组合
            PERQ_PATH.write_text(json.dumps({'schema': 'lang_bench_per_question.v1', 'dataset': result['dataset'],
                                             'dataset_key': result['dataset_key'], 'dataset_id': result['dataset_id'],
                                             'model_name': result['model_name'],
                                             'n_questions': n_total, 'items': per_question},
                                            ensure_ascii=False), encoding='utf-8')
            _run_state.update(state='done', current_subject='')
            _run_state['dataset'] = result['dataset']
            _run_state['model_name'] = result['model_name']

    except Exception as exc:
        with _LOCK:
            _run_state.update(state='error', error=f'{exc}', current_subject='')
        traceback.print_exc()


@router.get('/overview')
def overview() -> dict:
    """一次返回两个维度的全部事实，前端按当前选择筛选：
    datasets（每个数据集考多少题、其中能映射到三类图谱的有多少）+ results（按 (数据集, 模型) 分键的结果表）。
    """
    meta = json.loads(META_PATH.read_text(encoding='utf-8')) if META_PATH.exists() else {}
    has_q = _questions_path().exists()
    las_status = las._state['status']
    if las_status == 'idle' and las.model_dir() is not None:
        las_status = 'ready-after-load'          # 访问时会触发后台加载
    res = _load_results()
    return {'schema': 'lang_bench_overview.v3', 'available': has_q, 'meta': meta,
            'datasets': _dataset_index() if has_q else [],
            'results': res['by_key'], 'legacy': res['legacy'],
            'graph_categories': GRAPH_CATS,
            'model_status': las_status, 'model_dir': las.model_dir() and str(las.model_dir()),
            'loaded_model': _loaded_model(),
            'run_state': _run_state['state'],
            'running': {'dataset': _run_state.get('dataset', ''), 'model_name': _run_state.get('model_name', '')},
            'last_result': res['legacy']}


@router.post('/run')
def run(payload: Optional[dict] = Body(default=None)) -> dict:
    """启动一轮实测：只跑 body.dataset 指的那个数据集；body.model 只用于核对本机加载的权重。"""
    global _THREAD
    ds = _dataset_prefix((payload or {}).get('dataset') or '')
    page_model = (payload or {}).get('model') or ''
    loaded = _loaded_model()
    if page_model and loaded and not _model_matches(page_model, loaded):
        return {'schema': 'lang_bench_run.v2', 'accepted': False,
                'reason': f'本机当前加载的权重是 {loaded}，不是 {page_model.split("|")[0]}。'
                          f'图谱测试题只测已加载的那个模型——先在「空间 · 实时分析」里切到它，或先把这一项的选择改成 {loaded}。',
                'state': _run_state['state']}
    with _LOCK:
        if _run_state['state'] in ('running', 'loading'):
            return {'schema': 'lang_bench_run.v2', 'accepted': False, 'reason': '已有一轮实测在跑',
                    'state': _run_state['state']}
    n = next((d['n_questions'] for d in _dataset_index() if d['prefix'] == ds), 0) if ds else 0
    if _questions_path().exists() and ds and n == 0:
        return {'schema': 'lang_bench_run.v2', 'accepted': False,
                'reason': f'数据集「{ds}」在题库里没有题（题库只含 C-Eval / CMMLU / CMExam）。',
                'state': _run_state['state']}
    with _LOCK:
        _run_state.update(state='loading', error='', done=0, total=n, dataset=ds,
                          model_name=loaded, by_category={}, by_subject={})
    if _questions_path().exists():
        _THREAD = threading.Thread(target=_run_work, args=(ds,), daemon=True, name='lang-bench-run')
        _THREAD.start()
        return {'schema': 'lang_bench_run.v2', 'accepted': True, 'state': 'loading', 'dataset': ds, 'total': n}
    with _LOCK:
        _run_state.update(state='error', error='本机数据包里没有图谱测试题：先运行 scripts/download_lang_bench.py 下载题库。')
    return {'schema': 'lang_bench_run.v2', 'accepted': False, 'reason': _run_state['error'], 'state': 'error'}


@router.get('/status')
def status() -> dict:
    with _LOCK:
        d = {k: _run_state[k] for k in ('state', 'error', 'done', 'total', 'elapsed_s', 'current_subject')}
        d['dataset'] = _run_state.get('dataset', '')
        d['model_name'] = _run_state.get('model_name', '')
        d['by_category'] = {c: {'n': v[1], 'correct': v[0],
                                'acc': round(v[0] / v[1], 4) if v[1] else 0.0}
                            for c, v in _run_state['by_category'].items() if v[1]}
        if _run_state['state'] == 'done' and _run_state.get('result'):
            d['result'] = _run_state['result']
    d['schema'] = 'lang_bench_status.v2'
    return d

