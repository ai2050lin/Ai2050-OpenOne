# -*- coding: utf-8 -*-
"""语言图谱测试题（lang_bench.v1）：用公开数据集 C-Eval 的 val 集对本机模型做选择题实测，
给出「知识网络 / 功能推理 / 语法规则」三个图谱大类的真实正确率百分比。

数据真源：data/lang_bench/ceval_val.v1.jsonl（由 scripts/download_lang_bench.py 一次性从
HuggingFace Hub 拉取，169 题、7 个科目，本地数据包自带，运行时不再依赖外网）。
模型真源：复用 live_analysis_service 已加载的本机权重（不重复占显存），逐题一次前向，
在「答案：」位置比较 A/B/C/D 选项字母的 next-token 概率（与官方 loglikelihood 等价，同前缀）。

端点：
  GET  /api/lang/bench/overview   数据集构成 + 模型可用性 + 最近一次结果
  POST /api/lang/bench/run        启动后台实测（逐题前向，几秒后可轮询）
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
from fastapi import APIRouter

import live_analysis_service as las

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'lang_bench'
QUESTIONS_PATH_V2 = DATA / 'questions.v2.jsonl'      # 大题库（C-Eval + CMMLU + CMExam，~42k 题）
QUESTIONS_PATH_V1 = DATA / 'ceval_val.v1.jsonl'      # 小题库（仅 C-Eval val 169 题），向后兼容
QUESTIONS_PATH = QUESTIONS_PATH_V2 if QUESTIONS_PATH_V2.exists() else QUESTIONS_PATH_V1


def _questions_path() -> Path:
    """运行时动态选择题库：大题库下载完成后无需重启即生效。"""
    return QUESTIONS_PATH_V2 if QUESTIONS_PATH_V2.exists() else QUESTIONS_PATH_V1
META_PATH = DATA / 'meta.json'
RESULTS_PATH = DATA / 'results.json'

router = APIRouter(prefix='/api/lang/bench', tags=['lang-bench'])

CATEGORIES = ['知识网络', '功能推理', '语法规则', '综合知识', '医学知识']   # 排序用；实际以题库 meta 为准

_run_state: Dict[str, Any] = {
    'state': 'idle',     # idle | loading | running | done | error
    'error': '', 'done': 0, 'total': 0, 'started_at': 0.0, 'elapsed_s': 0.0,
    'current_subject': '', 'by_category': {}, 'by_subject': {},
}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def _load_questions() -> List[dict]:
    with open(_questions_path(), encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_prompt(q: dict) -> str:
    """C-Eval 官方 base 提示格式（few-shot 数为 0 的同款模板），支持 A–E 五选项。"""
    lines = [f'以下是中国关于{q["subject_cn"]}的单项选择题，请选出其中的正确答案。', '', q['question']]
    for letter, text in zip('ABCDE', q['choices']):
        lines.append(f'{letter}. {text}')
    lines.append('答案：')
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


def _run_work() -> None:
    try:
        questions = _load_questions()
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

        with _LOCK:
            _run_state.update(state='running', done=0, total=len(questions), started_at=time.perf_counter(),
                              error='', by_category={c: [0, 0] for c in CATEGORIES},
                              by_subject={}, current_subject='')
        per_question = []
        for i, q in enumerate(questions):
            t0 = time.perf_counter()
            pred = _score_one(model, tok, device, letter_ids, q)
            ok = int(pred == q['answer'])
            cat, sub = q['category'], q['subject_cn']
            with _LOCK:
                _run_state['done'] = i + 1
                _run_state['current_subject'] = sub
                _run_state['elapsed_s'] = round(time.perf_counter() - _run_state['started_at'], 1)
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
            result = {
                'schema': 'lang_bench_result.v1',
                'model_name': Path(las._state['dir']).name if las._state['dir'] else '',
                'device': las._state['device'],
                'dataset': 'C-Eval val（ceval/ceval-exam，HuggingFace 公开数据集）',
                'n_questions': n_total, 'correct': n_ok,
                'acc': round(n_ok / n_total, 4) if n_total else 0.0,
                'elapsed_s': _run_state['elapsed_s'],
                'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'by_category': by_cat, 'by_subject': by_sub,
                'per_question': per_question,
            }
            RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
            _run_state.update(state='done', current_subject='')
    except Exception as exc:
        with _LOCK:
            _run_state.update(state='error', error=f'{exc}', current_subject='')
        traceback.print_exc()


@router.get('/overview')
def overview() -> dict:
    meta = json.loads(META_PATH.read_text(encoding='utf-8')) if META_PATH.exists() else {}
    has_q = _questions_path().exists()
    las_status = las._state['status']
    if las_status == 'idle' and las.model_dir() is not None:
        las_status = 'ready-after-load'          # 访问时会触发后台加载
    result = None
    if RESULTS_PATH.exists():
        try:
            r = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))
            result = {k: r[k] for k in ('model_name', 'n_questions', 'correct', 'acc',
                                        'elapsed_s', 'finished_at', 'by_category', 'by_subject')}
        except Exception:
            pass
    return {'schema': 'lang_bench_overview.v1', 'available': has_q, 'meta': meta,
            'model_status': las_status, 'model_dir': las.model_dir() and str(las.model_dir()),
            'run_state': _run_state['state'], 'last_result': result}


@router.post('/run')
def run() -> dict:
    global _THREAD
    with _LOCK:
        if _run_state['state'] in ('running', 'loading'):
            return {'schema': 'lang_bench_run.v1', 'accepted': False, 'reason': '已有一轮实测在跑',
                    'state': _run_state['state']}
        _run_state.update(state='loading', error='', done=0, total=0)
    if _questions_path().exists():
        _THREAD = threading.Thread(target=_run_work, daemon=True, name='lang-bench-run')
        _THREAD.start()
        return {'schema': 'lang_bench_run.v1', 'accepted': True, 'state': 'loading'}
    with _LOCK:
        _run_state.update(state='error', error='本机数据包里没有图谱测试题：先运行 scripts/download_lang_bench.py 下载 C-Eval val。')
    return {'schema': 'lang_bench_run.v1', 'accepted': False, 'reason': _run_state['error'], 'state': 'error'}


@router.get('/status')
def status() -> dict:
    with _LOCK:
        d = {k: _run_state[k] for k in ('state', 'error', 'done', 'total', 'elapsed_s', 'current_subject')}
        d['by_category'] = {c: {'n': v[1], 'correct': v[0],
                                'acc': round(v[0] / v[1], 4) if v[1] else 0.0}
                            for c, v in _run_state['by_category'].items() if v[1]}
    d['schema'] = 'lang_bench_status.v1'
    if d['state'] == 'done' and RESULTS_PATH.exists():
        try:
            r = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))
            d['result'] = {k: r[k] for k in ('model_name', 'n_questions', 'correct', 'acc',
                                             'elapsed_s', 'finished_at', 'by_category', 'by_subject')}
        except Exception:
            pass
    return d
