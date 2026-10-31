# -*- coding: utf-8 -*-
"""语言图谱 · 数据集实测（lang_atlas.v1）：先选数据集，再对「知识网络 / 功能推理 / 语法规则」
三个类型做注意力实测，最后跨类型归纳「哪些头普遍参与、哪些只属于某一类」。

与 lang_bench_service 的分工：
  bench  = 只测行为正确率（打多少分）
  atlas  = 测「分怎么来的」——答案位置的注意力落在哪个选项上，按层×头统计

三个量（都可在同一坐标系里比较）：
  hit    P(该头注意力质量最大的选项 == 正确选项)。随机基线 = 1/n_options。
  delta  mean(m_正确选项 − 干扰选项均值)。衡量「偏向正确答案」的力度，不是「注意选项」的力度。
  mass   该头落在全部选项上的总质量。用于判断这个头到底有没有在看选项。

普遍规律的判据（写死在 aggregate() 里，不接受人工挑选）：
  对每个 (层,头)，在每个类型上做二项检验得 z；multipletest 用 Bonferroni。
  同一头在 ≥2 个类型上 z 同号且过阈 → 通用；只在 1 个类型 → 类型特异。
  层的分布本身就是结论：若通用头集中在少数层，那就是「普遍规律」的可读形态。

端点：
  GET  /api/lang/atlas/datasets   数据集清单 + 每个类型在各数据集里的可选题量 + 映射依据
  POST /api/lang/atlas/run        启动实测（后台线程，逐题一次前向）
  GET  /api/lang/atlas/status     进度与结果
"""
from __future__ import annotations

import json
import math
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter

try:
    # 服务以 `server.server:app` 启动时，本模块与它共享同一个 live_analysis_service 实例；
    # 若写成裸名 `import live_analysis_service`，会加载出第二份模块副本（独立 _state），
    # 于是这里读到 status=idle 并试图再加载一份 4B 权重，直接把显存打满。
    from server import live_analysis_service as las
except ImportError:                      # 独立脚本 / 以 server 为根启动时退回裸名
    import live_analysis_service as las

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'lang_bench'
QUESTIONS = DATA / 'questions.v2.jsonl'
QUESTIONS_V1 = DATA / 'ceval_val.v1.jsonl'
RESULTS = DATA / 'atlas_last.json'


def _slug(dataset: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in (dataset or 'run'))[:40]


def _saved_runs() -> List[dict]:
    """已落盘的各数据集结果摘要，用于「同一批头是否在两个数据集上都过阈」的复现对比。"""
    out = []
    for p in sorted(DATA.glob('atlas_*.json')):
        if p.name == 'atlas_last.json':
            continue
        try:
            r = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        out.append({
            'dataset': r.get('dataset'), 'model_name': r.get('model_name'),
            'n_measured': r.get('n_measured'), 'elapsed_s': r.get('elapsed_s'),
            'finished_at': r.get('finished_at'),
            'n_common': r.get('n_common'), 'n_specific': r.get('n_specific'),
            'acc': {c: v.get('acc') for c, v in (r.get('by_category') or {}).items()},
            'heads': [f"L{h['block']}H{h['head']}" for h in (r.get('common') or [])],
        })
    return sorted(out, key=lambda x: x.get('finished_at') or '')

router = APIRouter(prefix='/api/lang/atlas', tags=['lang-atlas'])

# 三个类型 ↔ 各数据集的科目（键是题库里的 subject 英文键，不是显示名）。
# 这张表是「类型」的唯一真源：改映射必须改这里，接口会把依据一起返回，便于审阅。
CATEGORY_SUBJECTS: Dict[str, Dict[str, List[str]]] = {
    '知识网络': {
        'C-Eval val': ['high_school_biology', 'high_school_geography', 'high_school_history'],
        'CMMLU test': ['high_school_biology', 'high_school_geography', 'world_history', 'chinese_history',
                       'global_facts', 'astronomy', 'world_religions', 'high_school_chemistry',
                       'high_school_physics'],
    },
    '功能推理': {
        'C-Eval val': ['civil_servant', 'logic'],
        'CMMLU test': ['logical', 'chinese_civil_service_exam'],
    },
    '语法规则': {
        'C-Eval val': ['chinese_language_and_literature', 'high_school_chinese'],
        'CMMLU test': ['modern_chinese', 'ancient_chinese', 'elementary_chinese'],
    },
}
CATEGORY_ALIAS = {'功能推理': '分析推理'}          # 用户口语与分析推理并列，显示用
ALL_CATS = ['知识网络', '功能推理', '语法规则']

_state: Dict[str, Any] = {'state': 'idle', 'error': '', 'done': 0, 'total': 0,
                          'elapsed_s': 0.0, 'phase': '', 'by_category': {}}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


# ---------------------------------------------------------------- 题库与数据集
def _load_questions() -> List[dict]:
    p = QUESTIONS if QUESTIONS.exists() else QUESTIONS_V1
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _datasets_index() -> List[dict]:
    """按 source 分组，给出每个数据集在三个类型上的可选题量与科目依据。"""
    rows = _load_questions()
    by_src: Dict[str, List[dict]] = {}
    for r in rows:
        by_src.setdefault(r.get('source') or '未知', []).append(r)
    out = []
    for src, rs in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        subj_cnt: Dict[str, int] = {}
        for r in rs:
            subj_cnt[r['subject']] = subj_cnt.get(r['subject'], 0) + 1
        cats = {}
        for cat, per_src in CATEGORY_SUBJECTS.items():
            want = per_src.get(src) or []
            have = [s for s in want if s in subj_cnt]
            n = sum(subj_cnt[s] for s in have)
            cats[cat] = {
                'n': n,
                'subjects': [{'key': s, 'name': next((x['subject_cn'] for x in rs if x['subject'] == s), s),
                              'n': subj_cnt[s]} for s in have],
                'missing': [s for s in want if s not in subj_cnt],
            }
        out.append({'id': src, 'n_questions': len(rs), 'n_subjects': len(subj_cnt),
                    'by_category': cats,
                    'testable': sum(v['n'] for v in cats.values()) > 0,
                    'reason': '' if sum(v['n'] for v in cats.values()) else
                              '该数据集没有可映射到三大类型的科目，只能做综合能力基线，不能做类型对比。'})
    return out


# ---------------------------------------------------------------- prompt 与位置
def _build_prompt(q: dict) -> tuple[str, Dict[str, tuple], int]:
    """与 lang_bench 同款提示格式；返回 prompt、各段 (起, 止) token 偏移、逐段总长。"""
    lines = [('instr', f'以下是中国关于{q["subject_cn"]}的单项选择题，请选出其中的正确答案。'),
             ('q', q['question'])]
    for letter, text in zip('ABCDE', q['choices']):
        lines.append((f'opt{letter}', f'{letter}. {text}'))
    lines.append(('ans', '答案：'))
    prompt = '\n'.join(t for _, t in lines)
    return prompt, lines, len(lines)


def _spans(tok, lines: List[tuple]) -> tuple[Dict[str, tuple], int]:
    spans, off = {}, 0
    for name, text in lines:
        n = len(tok.encode(text + '\n', add_special_tokens=False))
        spans[name] = (off, off + n)
        off += n
    return spans, off


def _letter_token_ids(tok) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for letter in 'ABCDE':
        ids = set()
        for form in (letter, ' ' + letter, '　' + letter):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        out[letter] = sorted(ids)
    return out


# ---------------------------------------------------------------- 单题测量
def _measure_one(model, tok, device, q: dict, letter_ids, max_len: int = 640) -> Optional[dict]:
    """一题一次前向，同时收两个量：

    ① attn  —— 答案位置各层各头对每个选项文本的注意力质量。
               回答「这个头有没有在看选项」，但实测总质量中位数只有 0.05–0.2，先验偏弱。
    ② dla   —— 各层各头对该「答案字母」logit 的直接贡献（direct logit attribution）。
               头输出经 o_proj 分块 → 残差 → 按该位置 RMSNorm 线性化 → 与 unembed 方向点积。
               这是连续量、可加和，比「argmax 落在哪个选项」稳定得多，作为主指标。

    dla 的线性化说明：RMSNorm 严格非线性，这里用该位置的 rms 作常数，是一阶近似
    （机制可解释性里的标准做法）；因此 dla 只用来做头间相对比较，不作绝对 logit 值使用。
    """
    prompt, lines, _ = _build_prompt(q)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:max_len]
    T = int(ids.shape[0])
    spans, seg_total = _spans(tok, lines)
    shift = T - seg_total
    if abs(shift) > 4:                       # 分段编码与整体编码的边界不一致：跳过，不猜
        return None
    letters = list('ABCDE'[:len(q['choices'])])
    idx = {}
    for letter in letters:
        lo, hi = spans[f'opt{letter}']
        lo, hi = max(0, lo + shift), min(T, hi + shift - 1)
        if hi > lo:
            idx[letter] = (lo, hi)
    if len(idx) < len(letters) or q['answer'] not in idx:
        return None

    ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
    if ans_tok is None:
        return None

    head_last: Dict[int, torch.Tensor] = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        def mk(i):
            def hook(_m, args):
                head_last[i] = args[0].detach()[0, -1].float()       # [H*hd]，o_proj 的输入
            return hook
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(li)))

    try:
        with las._RUN_LOCK, torch.no_grad():
            out = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False, output_attentions=True)
            logits = out.logits[0, -1].detach().float()
            atts = out.attentions
            n_layers = len(model.model.layers)
            H = int(model.config.num_attention_heads)
            WU = model.lm_head.weight.detach()                       # [V, d]
            norm_g = model.model.norm.weight.detach().float()        # [d]
            cols = []
            for letter in letters:
                t = next(iter(letter_ids.get(letter) or []), None)
                if t is None:
                    cols = []
                    break
                cols.append(WU[t].float())
            if not cols:
                return None
            # u = 有效读出方向（各选项字母一个）。严格 DLA 还要除以该位置 rms，但那是同一常数，
            # 不影响头间比较与跨题符号，故省略；因此 dla 只作相对量，不解释为绝对 logit。
            uk = torch.stack(cols) * norm_g                          # [k, d]
            dla = torch.zeros(n_layers, H, len(letters))
            for li in range(n_layers):
                Wo = model.model.layers[li].self_attn.o_proj.weight.detach().float()   # [d, H*hd]
                hd = Wo.shape[1] // H
                v = head_last[li].view(H, hd)                        # [H, hd]
                c = torch.einsum('dhk,hk->dh', Wo.view(Wo.shape[0], H, hd), v)          # [d, H]
                dla[li] = (c.t() @ uk.t()).cpu()                     # [H, k]
            best = {ch: float(max(logits[j] for j in cand))
                    for ch, cand in letter_ids.items() if cand and ch in letters}
    finally:
        for h in handles:
            h.remove()
    pred = max(best, key=best.get) if best else '?'

    n_layers = len(atts)
    H = int(atts[0].shape[1])
    # ① 注意力质量
    mass = torch.zeros(n_layers, H, len(letters))
    for l in range(n_layers):
        row = atts[l][0][:, -1, :].float().cpu()          # [H, T]
        for k, letter in enumerate(letters):
            lo, hi = idx[letter]
            mass[l, :, k] = row[:, lo:hi].sum(dim=1)
    return {'pred': pred, 'correct': pred == q['answer'], 'answer': q['answer'],
            'letters': letters, 'mass': mass.numpy(), 'dla': dla.numpy(),
            'L': n_layers, 'H': H, 'T': T}


# ---------------------------------------------------------------- 归纳
def _aggregate(rows: List[dict], cats_present: List[str], n_options: int) -> dict:
    """跨题、跨类型归纳。主指标是 dla 的净推力（对答案字母 − 对干扰字母），attn 只作辅助。

    每题先按「该头平均贡献绝对值」归一化，否则各题残差尺度不同会把 t 值冲成噪声。
    这样 net 的单位是「相当于本层一个平均头贡献的多少倍」，跨题可比。
    """
    import numpy as np

    p0 = 1.0 / n_options
    per_cat: Dict[str, dict] = {}
    for cat in cats_present:
        sub = [r for r in rows if r['category'] == cat]
        if not sub:
            continue
        L, H = sub[0]['L'], sub[0]['H']
        nets_l, nets_g, hits, masssum = [], [], [], np.zeros((L, H))
        n_ok = 0
        for r in sub:
            d = r['dla']                                   # [L,H,k]
            m = r['mass']                                  # [L,H,k]
            ai = r['letters'].index(r['answer'])
            dis = [i for i in range(d.shape[-1]) if i != ai]
            raw = d[..., ai] - d[..., dis].mean(-1)        # [L,H]
            nets_l.append(raw / (np.abs(d).mean(axis=(1, 2))[:, None] + 1e-12))   # 层内相对
            nets_g.append(raw / (np.abs(d).mean() + 1e-12))                        # 全局相对
            masssum += m.mean(-1)
            hits.append((m.argmax(-1) == ai).astype(float))
            n_ok += int(r['correct'])
        # 判定用层内归一化：否则深层的绝对贡献天然更大，跨层比较会系统性偏向最后几层
        A = np.stack(nets_l)
        G = np.stack(nets_g)
        n = A.shape[0]
        mean = A.mean(0)
        sd = A.std(0, ddof=1) if n > 1 else np.ones_like(mean)
        tval = mean / (sd / math.sqrt(n) + 1e-12)
        gmean = G.mean(0)
        hitm = np.stack(hits).mean(0)
        heads = []
        for l in range(L):
            for h in range(H):
                heads.append({'block': l, 'head': h,
                              'net': round(float(mean[l, h]), 3),
                              'net_global': round(float(gmean[l, h]), 2),
                              't': round(float(tval[l, h]), 2),
                              'hit': round(float(hitm[l, h]), 3),
                              'attn': round(float(masssum[l, h] / n), 4),
                              'n': n})
        heads.sort(key=lambda x: (-abs(x['t']), -abs(x['net'])))
        per_cat[cat] = {'n': n, 'acc': round(n_ok / n, 4), 'heads': heads,
                        'baseline_hit': round(p0, 4)}

    keys = sorted({(h['block'], h['head']) for c in per_cat.values() for h in c['heads']})
    zmap = {(c, h['block'], h['head']): h for c, v in per_cat.items() for h in v['heads']}
    n_scan = len(keys) or 1
    # 多重比较：1152 个头，双侧 α=0.05 的 Bonferroni 阈值约 3.9；取 4.0 且要求同号
    t_thr = 4.0
    common, specific = [], []
    for (b, h) in keys:
        ts = {c: zmap[(c, b, h)]['t'] for c in per_cat}
        pos = [c for c, t in ts.items() if t >= t_thr]
        neg = [c for c, t in ts.items() if t <= -t_thr]
        rec = {'block': b, 'head': h, 't': ts,
               'net': {c: zmap[(c, b, h)]['net'] for c in per_cat},
               'net_global': {c: zmap[(c, b, h)]['net_global'] for c in per_cat},
               'hit': {c: zmap[(c, b, h)]['hit'] for c in per_cat}}
        strong, other = (pos, neg) if len(pos) >= len(neg) else (neg, pos)
        if len(strong) >= 2:
            rec['verdict'] = 'general'
            rec['cats'] = strong
            common.append(rec)
        elif len(strong) == 1 and not other:
            rec['verdict'] = 'specific'
            rec['cats'] = strong
            specific.append(rec)
    common.sort(key=lambda r: -max(abs(r['t'][c]) for c in r['t']))
    specific.sort(key=lambda r: -max(abs(r['t'][c]) for c in r['t']))

    def layer_hist(items):
        hist: Dict[int, int] = {}
        for r in items:
            hist[r['block']] = hist.get(r['block'], 0) + 1
        return dict(sorted(hist.items()))

    return {
        't_threshold': t_thr,
        'n_heads_scanned': n_scan,
        'n_options_baseline': round(p0, 4),
        'by_category': {c: {'n': v['n'], 'acc': v['acc'], 'baseline_hit': v['baseline_hit'],
                            'top_heads': v['heads'][:24]} for c, v in per_cat.items()},
        'n_common': len(common),
        'n_specific': len(specific),
        'common': common[:40],
        'specific': specific[:40],
        'layer_hist_common': layer_hist(common),
    }


# ---------------------------------------------------------------- 后台运行
def _run_work(dataset: str, cats: List[str], n_per_cat: int, seed: int) -> None:
    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            if las._state['status'] == 'ready':
                break
            if las._state['status'] == 'idle':
                las._load_async()
            if las._state['status'] == 'error':
                raise RuntimeError(las._state['error'] or '模型不可用')
            time.sleep(1.0)
        else:
            raise RuntimeError('等待模型加载超时（15 分钟）。')
        model, tok, device = las._state['model'], las._state['tokenizer'], las._state['device']
        letter_ids = _letter_token_ids(tok)

        pool = []
        for q in _load_questions():
            if (q.get('source') or '') != dataset:
                continue
            cat = next((c for c in ALL_CATS if q['subject'] in (CATEGORY_SUBJECTS[c].get(dataset) or [])), None)
            if cat in cats:
                pool.append((cat, q))
        rng = __import__('random').Random(seed)
        picked: List[dict] = []
        for c in cats:
            sub = [q for cat, q in pool if cat == c]
            rng.shuffle(sub)
            take = sub if n_per_cat <= 0 else sub[:n_per_cat]
            picked.extend({'q': q, 'category': c} for q in take)

        with _LOCK:
            _state.update(state='running', done=0, total=len(picked), elapsed_s=0.0, error='',
                          phase='逐题前向（答案位置注意力）', started_at=time.perf_counter(),
                          by_category={c: {'n': 0, 'correct': 0} for c in cats})
        rows, skipped = [], 0
        for i, item in enumerate(picked):
            r = _measure_one(model, tok, device, item['q'], letter_ids)
            if r is None:
                skipped += 1
            else:
                r['category'] = item['category']
                rows.append(r)
                with _LOCK:
                    d = _state['by_category'][item['category']]
                    d['n'] += 1
                    d['correct'] += int(r['correct'])
            with _LOCK:
                _state['done'] = i + 1
                _state['elapsed_s'] = round(time.perf_counter() - _state['started_at'], 1)
        if not rows:
            raise RuntimeError('没有一道题能对齐 token 边界，无法测量。')

        with _LOCK:
            _state['phase'] = '跨类型归纳'
        n_opt = max(len(r['letters']) for r in rows)
        agg = _aggregate(rows, cats, n_opt)
        result = {
            'schema': 'lang_atlas_result.v1',
            'dataset': dataset, 'categories': cats,
            'model_name': Path(las._state['dir']).name if las._state['dir'] else '',
            'seed': seed, 'n_per_category': n_per_cat,
            'n_measured': len(rows), 'n_skipped': skipped,
            'elapsed_s': _state['elapsed_s'],
            'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'subject_map': {c: CATEGORY_SUBJECTS[c].get(dataset, []) for c in cats},
            **agg,
        }
        blob = json.dumps(result, ensure_ascii=False, indent=1)
        RESULTS.write_text(blob, encoding='utf-8')
        (DATA / f'atlas_{_slug(dataset)}.json').write_text(blob, encoding='utf-8')
        with _LOCK:
            _state.update(state='done', phase='')
    except Exception as exc:
        with _LOCK:
            _state.update(state='error', error=f'{exc}', phase='')
        traceback.print_exc()


# ---------------------------------------------------------------- 端点
@router.get('/datasets')
def datasets() -> dict:
    try:
        idx = _datasets_index()
    except Exception as exc:
        return {'schema': 'lang_atlas_datasets.v1', 'available': False, 'reason': f'{exc}', 'datasets': []}
    return {'schema': 'lang_atlas_datasets.v1', 'available': bool(idx),
            'categories': [{'id': c, 'name': c, 'also': CATEGORY_ALIAS.get(c)} for c in ALL_CATS],
            'datasets': idx,
            'note': '类型 ↔ 科目的映射是显式表（CATEGORY_SUBJECTS），不是模型推断；'
                    '某数据集若没有可映射科目，只做综合能力基线，不参与类型对比。'}


@router.post('/run')
def run(dataset: str = '', categories: str = '', n_per_category: int = 40, seed: int = 20261030) -> dict:
    global _THREAD
    cats = [c for c in (categories.split(',') if categories else ALL_CATS) if c in ALL_CATS] or ALL_CATS
    ds = dataset or next((d['id'] for d in _datasets_index() if d['testable']), '')
    if not ds:
        return {'schema': 'lang_atlas_run.v1', 'accepted': False, 'reason': '没有可用于类型测试的数据集。'}
    with _LOCK:
        if _state['state'] in ('running', 'loading'):
            return {'schema': 'lang_atlas_run.v1', 'accepted': False, 'reason': '已有一轮实测在跑',
                    'state': _state['state']}
        _state.update(state='loading', error='', done=0, total=0, phase='准备')
    _THREAD = threading.Thread(target=_run_work, args=(ds, cats, int(n_per_category), int(seed)),
                               daemon=True, name='lang-atlas-run')
    _THREAD.start()
    return {'schema': 'lang_atlas_run.v1', 'accepted': True, 'dataset': ds, 'categories': cats,
            'n_per_category': n_per_category}


@router.get('/status')
def status() -> dict:
    with _LOCK:
        d = {k: _state.get(k) for k in ('state', 'error', 'done', 'total', 'elapsed_s', 'phase')}
        d['by_category'] = {c: {**v, 'acc': round(v['correct'] / v['n'], 4) if v['n'] else 0.0}
                            for c, v in (_state.get('by_category') or {}).items()}
    d['schema'] = 'lang_atlas_status.v1'
    try:
        d['runs'] = _saved_runs()
    except Exception:
        d['runs'] = []
    if RESULTS.exists():
        try:
            d['last'] = json.loads(RESULTS.read_text(encoding='utf-8'))
        except Exception:
            pass
    return d
