# -*- coding: utf-8 -*-
"""AI2050 模板 runner v3 —— 可合并充分统计量 + 行为门 + 金丝雀（dist_protocol.v3）

随模板包下发到 Worker 节点，自包含（只依赖 numpy；真实模式另需 torch/transformers）:
    python runner.py [--mode auto|smoke|real|synthetic] [--out outputs]

环境变量:
    AI2050_RUNNER_MODEL   HF 模型目录（auto 模式下存在即走真实模式）
    AI2050_SHARD          分片 "i/n"（缺省 0/1；金丝雀条目在每个分片都执行）
    AI2050_PRECISION      bf16 | fp16 | fp32（缺省 bf16）
    AI2050_RUNNER_MODE    --mode auto 时强制模式（synthetic 用于沙盒中心的流程验证）

产出（Agent 上传 outputs/ 下全部文件）:
    summary.json      必有：状态、计数、逐格行为准确率、token 长度、本分片 η² 摘要
    suffstats.npz     真实/合成模式：每 (split, 条件格, 层, 维) 的 n、Σx、Σx²（float64，可跨节点相加）
    behavior.json     含 probe 的条目：首 token 是否 top-1、排名、对数概率
    canary.json       金丝雀摘要（目标位置向量的范数、均值、前 16 维）
    items.npz         逐条目标位置向量（float16；超过体积预算时用固定种子的随机投影降维，
                      各分片同一投影矩阵），附条目元数据——供中心做置换零分布、留出探针、
                      白化方向、句式不变性、3D 响应空间（corpus.capture_items=false 时不输出）

模式:
    smoke      不加载模型，只校验语料与分片，status=smoke（不冒充实测）
    real       真实前向，status=real
    synthetic  无 GPU 的协议与分析方法验证：按「已知答案」生成向量——早层以词身份与句式为主、
               中层出现类别方向、末层类别与句式并存，另有 token 长度干扰和一个行为薄弱格；
               status=synthetic，中心节点只合并 kind=real 结果，合成结果永不进入研究统计。
"""
import argparse
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path

HEAD_DIMS = 16


def cond_key(cond):
    return '|'.join(f'{k}={cond[k]}' for k in sorted(cond))


def parse_shard(value):
    try:
        i, n = (int(x) for x in str(value or '0/1').split('/'))
        if n >= 1 and 0 <= i < n:
            return i, n
    except ValueError:
        pass
    raise SystemExit(f'invalid AI2050_SHARD={value!r}; expected "i/n"')


def resolve_layers(spec, n_hs):
    out = []
    for tag in spec:
        if tag == 'first':
            out.append(('first', 1))
        elif tag == 'mid':
            out.append(('mid', n_hs // 2))
        elif tag == 'last':
            out.append(('last', n_hs - 1))
        elif isinstance(tag, int) and 0 <= tag < n_hs:
            out.append((f'L{tag}', tag))
        else:
            raise ValueError(f'unknown layer tag: {tag}')
    return out


ITEM_BUDGET_BYTES = 48 * 1024 * 1024


class ItemStore:
    """逐条向量（float16）。超过预算时投影到 k 维：投影矩阵由语料指纹确定，所有分片相同。"""

    def __init__(self, corpus_seed):
        self.seed = corpus_seed
        self.meta, self.vecs = [], []

    def add(self, it, vectors, n_tok, correct):
        self.meta.append({'id': it['id'], 'split': it['split'], 'cond': it['cond'],
                          'frame': it.get('frame_id', ''), 'filler': it.get('filler_id', ''),
                          'n_tok': int(n_tok), 'correct': -1 if correct is None else int(bool(correct))})
        self.vecs.append([list(v) if not hasattr(v, 'astype') else v for v in vectors])

    def npz_bytes(self, layer_tags):
        import numpy as np
        L, N = len(layer_tags), len(self.meta)
        X = np.stack([np.stack([np.asarray(self.vecs[i][l], dtype=np.float32) for i in range(N)]) for l in range(L)])
        d = X.shape[2]
        space, k = 'full', d
        if L * N * d * 2 > ITEM_BUDGET_BYTES:
            k = int(max(128, min(d, ITEM_BUDGET_BYTES // (L * N * 2))))
            rng = np.random.default_rng(self.seed)
            R = rng.standard_normal((d, k)).astype(np.float32) / np.sqrt(k)
            X = X @ R
            space = f'jl{k}'
        buf = io.BytesIO()
        np.savez_compressed(
            buf, schema=np.array('items.v1'), layer_tags=np.array([str(t) for t in layer_tags]),
            space=np.array(space), dim_original=np.array(d), X=X.astype(np.float16),
            ids=np.array([m['id'] for m in self.meta]), split=np.array([m['split'] for m in self.meta]),
            cond=np.array([json.dumps(m['cond'], sort_keys=True, ensure_ascii=False) for m in self.meta]),
            frame=np.array([m['frame'] for m in self.meta]), filler=np.array([m['filler'] for m in self.meta]),
            n_tok=np.array([m['n_tok'] for m in self.meta], dtype=np.int32),
            correct=np.array([m['correct'] for m in self.meta], dtype=np.int8))
        return buf.getvalue(), space


class Acc:
    """与 shared/dist_protocol/suffstats.py 同格式的累加器（runner 自包含，不 import 中心库）。"""

    def __init__(self, layer_tags, dim, factors):
        import numpy as np
        self.np = np
        self.layer_tags, self.dim, self.factors = list(layer_tags), dim, list(factors)
        self.cells, self.index, self.rows = [], {}, []

    def add(self, split, cond, vectors):
        np = self.np
        key = (split, json.dumps(cond, sort_keys=True, ensure_ascii=False))
        if key not in self.index:
            self.index[key] = len(self.cells)
            self.cells.append(key)
            L = len(self.layer_tags)
            self.rows.append([0, np.zeros((L, self.dim)), np.zeros((L, self.dim))])
        row = self.rows[self.index[key]]
        row[0] += 1
        for l, v in enumerate(vectors):
            v = np.asarray(v, dtype=np.float64)
            row[1][l] += v
            row[2][l] += v * v

    def npz_bytes(self):
        np = self.np
        buf = io.BytesIO()
        L, C = len(self.layer_tags), len(self.cells)
        s = np.zeros((L, C, self.dim)); q = np.zeros((L, C, self.dim))
        for i, row in enumerate(self.rows):
            s[:, i] = row[1]; q[:, i] = row[2]
        np.savez_compressed(
            buf, schema=np.array('suffstats.v1'), layer_tags=np.array([str(t) for t in self.layer_tags]),
            factors=np.array(json.dumps(self.factors, ensure_ascii=False)),
            cell_split=np.array([c[0] for c in self.cells]), cell_cond=np.array([c[1] for c in self.cells]),
            n=np.array([r[0] for r in self.rows], dtype=np.int64), sum=s, sumsq=q)
        return buf.getvalue()

    def eta2_digest(self, split='discovery'):
        """本分片 discovery 的逐因子平均 η²（仅供节点自检；正式统计在中心合并后计算）。"""
        np = self.np
        idx = [i for i, c in enumerate(self.cells) if c[0] == split]
        if len(idx) < 2:
            return {}
        out = {}
        for l, tag in enumerate(self.layer_tags):
            n = np.array([self.rows[i][0] for i in idx], dtype=float)
            s = np.stack([self.rows[i][1][l] for i in idx]); q = np.stack([self.rows[i][2][l] for i in idx])
            N = n.sum(); g = s.sum(0) / N
            sst = np.maximum(q.sum(0) - N * g * g, 1e-12)
            conds = [json.loads(self.cells[i][1]) for i in idx]
            per = {}
            for f in self.factors:
                lv = {}
                for j, c in enumerate(conds):
                    lv.setdefault(str(c[f]), []).append(j)
                ssf = np.zeros_like(g)
                for js in lv.values():
                    nl = n[js].sum(); ml = s[js].sum(0) / nl
                    ssf += nl * (ml - g) ** 2
                per[f] = float((ssf / sst).mean())
            out[tag] = per
        return out


def summarize_vec(v):
    vals = [float(x) for x in v]
    return {'norm': math.sqrt(sum(x * x for x in vals)), 'mean': sum(vals) / max(len(vals), 1),
            'head': vals[:HEAD_DIMS], 'dim': len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='auto', choices=['auto', 'smoke', 'real', 'synthetic'])
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    corpus = json.loads((here / 'corpus.json').read_text(encoding='utf-8'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_i, shard_n = parse_shard(os.environ.get('AI2050_SHARD', '0/1'))
    precision = os.environ.get('AI2050_PRECISION', 'bf16').lower()

    all_items = corpus['items']
    canaries = [it for it in all_items if it.get('canary')]
    study = [it for it in all_items if not it.get('canary')]
    mine = [it for k, it in enumerate(study) if k % shard_n == shard_i]
    factors = list(corpus.get('factors', []))
    layer_spec = corpus.get('layers', ['first', 'mid', 'last'])
    fp = hashlib.sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]

    cells, splits = {}, {}
    for it in mine:
        cells[cond_key(it['cond'])] = cells.get(cond_key(it['cond']), 0) + 1
        splits[it['split']] = splits.get(it['split'], 0) + 1
    summary = {
        'protocol': 'dist_protocol.v3', 'analysis': 'suffstats', 'tm_id': corpus.get('tm_id'),
        'family_id': corpus.get('family_id'), 'corpus_fingerprint': fp,
        'shard': f'{shard_i}/{shard_n}', 'n_items': len(mine), 'n_items_total': len(study),
        'by_split': splits, 'conditions': dict(sorted(cells.items())), 'factors': factors,
        'layers': [str(t) for t in layer_spec], 'precision': precision,
        'python': platform.python_version(), 'status': 'smoke',
    }

    mode = args.mode
    if mode == 'auto' and os.environ.get('AI2050_RUNNER_MODE') in ('smoke', 'real', 'synthetic'):
        mode = os.environ['AI2050_RUNNER_MODE']
    if mode == 'auto':
        mode = 'real' if os.environ.get('AI2050_RUNNER_MODEL') else 'smoke'

    try:
        if mode == 'real':
            run_real(corpus, mine, canaries, layer_spec, factors, precision, out, summary)
        elif mode == 'synthetic':
            run_synthetic(corpus, mine, canaries, layer_spec, factors, out, summary)
    except Exception as exc:  # 真实模式失败 → 诚实降级为 smoke，记录原因
        summary['status'] = 'smoke'
        summary['real_error'] = f'{type(exc).__name__}: {exc}'
        for name in ('suffstats.npz', 'behavior.json', 'canary.json', 'items.npz'):
            p = out / name
            if p.exists():
                p.unlink()

    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                                      encoding='utf-8')
    print(f"[runner-v3] status={summary['status']} shard={summary['shard']} items={len(mine)} out={out}")
    return 0


def _target_pos(text, target, offsets, start=None):
    s = start if isinstance(start, int) and start >= 0 else text.find(target)
    if s < 0 or text[s:s + len(target)] != target:
        raise ValueError(f'target {target!r} not in text {text!r}')
    e = s + len(target)
    cands = [i for i, (cs, ce) in enumerate(offsets) if cs < e and ce > s]
    if not cands:
        raise ValueError(f'target {target!r} has no token span in {text!r}')
    return max(cands)


def run_real(corpus, items, canaries, layer_spec, factors, precision, out, summary):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ['AI2050_RUNNER_MODEL']
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(precision, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
    model.eval()

    def forward(text, target, start=None):
        enc = tok(text, return_tensors='pt', return_offsets_mapping=True)
        off = enc.pop('offset_mapping')[0].tolist()
        pos = _target_pos(text, target, off, start)
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True).hidden_states
        layers = resolve_layers(layer_spec, len(hs))
        return [hs[i][0, pos, :].float().cpu().numpy() for _, i in layers], [t for t, _ in layers], len(off)

    acc = None
    tok_len = {}
    behavior = []
    store = ItemStore(corpus_seed(corpus)) if corpus.get('capture_items', True) else None
    with torch.inference_mode():
        for it in items:
            vecs, tags, n_tok = forward(it['text'], it['target'], it.get('target_char'))
            if acc is None:
                acc = Acc(tags, int(vecs[0].shape[0]), factors)
            acc.add(it['split'], it['cond'], vecs)
            tok_len.setdefault(cond_key(it['cond']), []).append(n_tok)
            correct = None
            probe = it.get('probe')
            if probe:
                enc = tok(probe['prompt'], return_tensors='pt').to(device)
                logits = model(**enc).logits[0, -1].float()
                ans_ids = tok(probe['answer'], add_special_tokens=False)['input_ids']
                if ans_ids:
                    logp = torch.log_softmax(logits, dim=-1)
                    tid = int(ans_ids[0])
                    rank = int((logits > logits[tid]).sum().item()) + 1
                    behavior.append({'id': it['id'], 'split': it['split'], 'cell': cond_key(it['cond']),
                                     'correct': rank == 1, 'rank': rank, 'logprob': float(logp[tid].item())})
                    correct = rank == 1
            if store is not None:
                store.add(it, vecs, n_tok, correct)
        can = {}
        for c in canaries:
            vecs, tags, _ = forward(c['text'], c['target'])
            can[c['id']] = {t: summarize_vec(v) for t, v in zip(tags, vecs)}
    if acc is None:
        raise RuntimeError('本分片没有样本')
    _write_outputs(out, acc, behavior, can, tok_len, summary,
                   status='real', extra={'model_path': model_path, 'device': device}, store=store)


def run_synthetic(corpus, items, canaries, layer_spec, factors, out, summary):
    """方法验证用合成数据（已知答案）。对每个层位 l：
        x = a_l·词身份(filler) + b_l·Σ类因子方向 + c_l·句式方向 + e_l·交互(格) + 0.04·字数·干扰方向 + 噪声（总范数约 0.8）
    其中「类因子」= factors[0]，「句式因子」= 其余因子；系数随层变化：
        first: 词身份 2.0、类 0.3、句式 1.5   mid: 0.8、1.6、0.6   last: 0.5、1.1、1.2（另有交互 0.4）
    行为：factors[0] 的第 5 个水平（字典序）在第 2 个句式水平上正确率约 55%，其余约 95%。
    这些参数只用于检验分析方法能否把植入的规律找回来，不代表任何真实模型。"""
    import numpy as np
    dim = int(corpus.get('synthetic_dim', 96))
    tags = [t if isinstance(t, str) else f'L{t}' for t in layer_spec]
    salt = os.environ.get('AI2050_SYNTHETIC_SALT', '')
    coef = {'first': (2.0, 0.3, 1.5, 0.0), 'mid': (0.8, 1.6, 0.6, 0.0), 'last': (0.5, 1.1, 1.2, 0.4)}

    def unit(name):
        rng = np.random.default_rng(int(hashlib.sha256((salt + name).encode()).hexdigest()[:8], 16))
        v = rng.standard_normal(dim)
        return v / np.linalg.norm(v)

    main_f = factors[0] if factors else None
    other_f = factors[1:]
    concepts = corpus.get('concepts') or {}

    def level_dir(f, lv):
        # 文法预注册了概念的水平（如 frame=interrog → 疑问）用「概念」作种子：同一合成模型里，不同模式族中
        # 同概念的水平共享方向。这是 atlas.v1 的阳性对照（植入共享结构），不代表真实模型。
        tag = concepts.get(f'{f}={lv}')
        return unit('concept=' + tag) if tag else unit(f'{f}={lv}')
    all_items = [it for it in corpus['items'] if not it.get('canary')]
    lv_main = sorted({str(it['cond'][main_f]) for it in all_items}) if main_f else []
    lv_other = sorted({str(it['cond'][other_f[0]]) for it in all_items}) if other_f else []
    weak = (lv_main[4] if len(lv_main) > 4 else None, lv_other[1] if len(lv_other) > 1 else None)
    nuis = unit('__length__')
    acc = Acc(tags, dim, factors)
    store = ItemStore(corpus_seed(corpus)) if corpus.get('capture_items', True) else None
    behavior, tok_len = [], {}
    for it in items:
        rng = np.random.default_rng(int(hashlib.sha256((salt + it['id']).encode()).hexdigest()[:8], 16))
        cond = it['cond']
        vecs = []
        for l, t in enumerate(tags):
            a_, b_, c_, e_ = coef.get(t, (1.0, 1.0, 1.0, 0.0))
            v = a_ * unit('filler=' + str(it.get('filler_id', it['id'])))
            if main_f:
                v = v + b_ * level_dir(main_f, cond[main_f])
            for f in other_f:
                v = v + c_ * level_dir(f, cond[f])
            if e_:
                v = v + e_ * unit('cell=' + cond_key(cond))
            v = v + 0.04 * len(it['text']) * nuis + rng.standard_normal(dim) * (0.8 / np.sqrt(dim))
            vecs.append(v)
        acc.add(it['split'], cond, vecs)
        tok_len.setdefault(cond_key(cond), []).append(len(it['text']))
        correct = None
        if it.get('probe'):
            p_ok = 0.55 if (main_f and other_f and str(cond[main_f]) == weak[0] and str(cond[other_f[0]]) == weak[1]) else 0.95
            correct = bool(rng.random() < p_ok)
            behavior.append({'id': it['id'], 'split': it['split'], 'cell': cond_key(cond),
                             'correct': correct, 'rank': 1 if correct else 2, 'logprob': -0.1 if correct else -2.0})
        if store is not None:
            store.add(it, vecs, len(it['text']), correct)
    can = {c['id']: {t: summarize_vec(unit(c['id']) * (l + 1)) for l, t in enumerate(tags)} for c in canaries}
    _write_outputs(out, acc, behavior, can, tok_len, summary, status='synthetic', extra={}, store=store)


def corpus_seed(corpus):
    key = corpus.get('grammar_sha256') or json.dumps(corpus.get('items', [])[:3], sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:8], 16)


def _write_outputs(out, acc, behavior, can, tok_len, summary, status, extra, store=None):
    (out / 'suffstats.npz').write_bytes(acc.npz_bytes())
    if store is not None and store.meta:
        data, space = store.npz_bytes(acc.layer_tags)
        (out / 'items.npz').write_bytes(data)
        summary['item_space'] = space
    (out / 'behavior.json').write_text(json.dumps({'schema': 'behavior.v1', 'items': behavior},
                                                  ensure_ascii=False), encoding='utf-8')
    (out / 'canary.json').write_text(json.dumps({'schema': 'canary.v1', 'entries': can},
                                                ensure_ascii=False), encoding='utf-8')
    by_cell = {}
    for b in behavior:
        rec = by_cell.setdefault(b['cell'], {'n': 0, 'correct': 0})
        rec['n'] += 1
        rec['correct'] += int(b['correct'])
    summary.update(extra)
    summary.update({
        'status': status,
        'dim': acc.dim,
        'layer_tags': acc.layer_tags,
        'behavior_by_cell': by_cell,
        'token_len_by_cell': {k: {'min': min(v), 'max': max(v), 'mean': round(sum(v) / len(v), 2)}
                              for k, v in sorted(tok_len.items())},
        'eta2_shard_digest': acc.eta2_digest(),
        'canary_ids': sorted(can),
    })


if __name__ == '__main__':
    raise SystemExit(main())
