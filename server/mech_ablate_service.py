# -*- coding: utf-8 -*-
"""机械可解释性 v3 · C1 层：绕开几何，直接做因果。

A2/G' 的结论（见 `design/lang_code_joint_attribution_v3.md` 附录 A2/G'）是几何筛不可用：
  · 深层 down_proj 列的写方向在残差流里近似各向同性（组内 |cos| 0.017 ≈ 零模型 0.0198）；
  · 头的 K/V 子空间并不偏好 MLP 写方向（head_pref 中位 −0.2，p90 0.6）；
  · GQA 只有 8 个 KV 组，头级分辨率上限 8，n=8 无功效。
但候选空间本来就只有 24 组 × 8 KV 组 = 192 对/层。这个规模不需要筛，直接因果测。

本模块做 C1，两种干预 × 三个读数：

  干预 A `zero`：把该组单元的 down_proj 输入列置零（单元级消融）。
  干预 B `dir` ：在 L{b} 输出处把该组的平均写方向投影掉（方向性消融，h ← h − (h·û)û）。
                这是 v3 真正想做的操作：阻断一个方向，而不是删掉一堆单元。

  读数 ① Δmargin：答案字母 logit 差的变化（可解释但不敏感，实测整层 MLP 置零 margin 只动 0.15）。
       ② KL(base‖ablated)：整词表分布散度，是主指标——它积分全部 logit，比 margin 灵敏一个量级。
       ③ Δdla(z)：下游每个头对答案对比方向的贡献改变，以基线跨题标准差归一 → 「这组喂的是哪些头」。

对照（必需）：
  · sham：同规模的随机单元集 / 随机方向，2 次抽样，给出 |·| 的经验 95 分位；
  · full：整层消融，阳性对照。若 full 都不动 KL，说明钩子或读数有问题，本次结果作废。

零消融是分布外操作，sham 与 full 都是判读的前提，不是可选装饰。

端点：
  POST /api/mech/ablate/run?block=24&dataset=C-Eval%20val&n=0&sham=2
  GET  /api/mech/ablate/status
  GET  /api/mech/ablate
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter

try:
    from server import lang_atlas_service as latlas
    from server import live_analysis_service as las
    from server import mech_pair_service as mpair
    from server import mech_service as msvc
except ImportError:                       # 独立脚本
    import lang_atlas_service as latlas    # type: ignore
    import live_analysis_service as las    # type: ignore
    import mech_pair_service as mpair      # type: ignore
    import mech_service as msvc            # type: ignore

ROOT = Path(__file__).resolve().parents[1]
ME = ROOT / 'data' / 'mech'
OUT_JSON = ME / 'ablate_cluster.json'      # 兼容旧路径（= L24 那一份）


def out_path(block: int) -> Path:
    """按层落盘，扩展多列时才不会互相覆盖。L24 同时写回旧文件名。"""
    return ME / ('ablate_L%d.json' % block)


def _index_blocks() -> Dict[str, Any]:
    rows = []
    for p in sorted(ME.glob('ablate_L*.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        z = d.get('methods', {}).get('zero', {})
        cs = z.get('clusters', [])
        kl = [c['kl'] for c in cs]
        top = max(cs, key=lambda c: c['kl']) if cs else None
        rows.append({
            'block': d.get('block'), 'n_questions': d.get('n_questions'), 'k': d.get('k'),
            'file': p.name,
            'zero_max_kl': round(max(kl), 5) if kl else None,
            'zero_top_cluster': top['cluster'] if top else None,
            'zero_top_size': top['size'] if top else None,
            'zero_kl_sum': round(sum(kl), 5) if kl else None,
            'zero_full_kl': z.get('controls', {}).get('full', {}).get('kl'),
            'dir_max_kl': max([c['kl'] for c in d.get('methods', {}).get('dir', {}).get('clusters', [])] or [0]),
        })
    return {'schema': 'mech_ablate_index.v1', 'n_blocks': len(rows), 'blocks': rows}

router = APIRouter(prefix='/api/mech/ablate', tags=['mech-ablate'])

_STATE: Dict[str, Any] = {'job': '', 'state': 'idle', 'error': '', 'done': 0, 'total': 0,
                          'elapsed_s': 0.0, 'note': '', 'block': -1}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def _set(**kw) -> None:
    with _LOCK:
        _STATE.update(kw)


def _prep(tok, q: dict, letter_ids, max_len: int = 640):
    prompt, lines, _ = latlas._build_prompt(q)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:max_len]
    T = int(ids.shape[0])
    spans, seg_total = latlas._spans(tok, lines)
    if abs(T - seg_total) > 4:
        return None
    ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
    if ans_tok is None:
        return None
    return ids, T, ans_tok


def _forward(model, device, ids, b, band, op, contrast, ans_tok, other_toks):
    """一次前向。op ∈ {None, ('zero', mask), ('dir', unit_vec)}，返回 (dla, margin, kl)。"""
    caps: Dict[int, torch.Tensor] = {}
    handles = []
    for l in band:
        def mk(li):
            def hook(_m, args):
                caps[li] = args[0].detach()[0, -1].float()
            return hook
        handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(mk(l)))

    if op is not None and op[0] == 'zero':
        mask = torch.as_tensor(op[1], device=device)

        def abl(_m, args):
            a = args[0].clone()
            a[..., mask] = 0.0
            return (a,) + tuple(args[1:])
        handles.append(model.model.layers[b].mlp.down_proj.register_forward_pre_hook(abl))
    elif op is not None and op[0] == 'dir':
        u = op[1].to(device).float()

        def abl(_m, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            proj = (h.float() @ u).unsqueeze(-1) * u
            h2 = (h.float() - proj).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        handles.append(model.model.layers[b].register_forward_hook(abl))

    try:
        with torch.no_grad():
            out = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
        lg = out.logits[0, -1].float()
        margin = float(lg[ans_tok] - lg[list(other_toks)].max())
        lp = torch.log_softmax(lg, dim=-1)
        dla = {}
        for l in band:
            Wo = model.model.layers[l].self_attn.o_proj.weight.detach().float()   # [d, H*dh]
            Hn = int(model.config.num_attention_heads)
            dh = caps[l].numel() // Hn
            t = (contrast @ Wo).view(Hn, dh)
            dla[l] = (t * caps[l].view(Hn, dh)).sum(dim=1).cpu().numpy()
        return dla, margin, lp
    finally:
        for h in handles:
            h.remove()


def _kl(lp_base: torch.Tensor, lp: torch.Tensor) -> float:
    """KL(base ‖ abl)，自然 log。fp32 全词表。"""
    p = lp_base.exp()
    return float((p * (lp_base - lp)).sum())


# ------------------------------------------------------------------ 主流程
def _run(block: int, dataset: str, n: int, n_sham: int, seed: int = 20261030) -> None:
    t0 = time.time()
    try:
        model, tok, device, meta, model_name = mpair._wait_ready()
        L = int(meta['n_layers'])
        Hn = int(model.config.num_attention_heads)
        d = int(model.config.hidden_size)
        letter_ids = latlas._letter_token_ids(tok)
        ldirs = msvc._letter_dirs(model, tok)

        cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
        rec = next((r for r in cj['layers'] if r['block'] == block), None)
        if rec is None:
            raise RuntimeError('unit_clusters.json 里没有 L%d，先跑 cluster_build' % block)
        cz = np.load(mpair.CLUST_NPZ)
        lab = cz['lab_%d' % block].astype(np.int64)
        K = int(rec['k'])
        d_ff = int(lab.size)

        band = list(range(block + 1, L))
        if not band:
            raise RuntimeError('L%d 之后没有层可测' % block)

        # 组的平均写方向（down_proj 列的平均，单位化）
        Wd = model.model.layers[block].mlp.down_proj.weight.detach().float().to(device)  # [d, d_ff]
        U = (Wd.t() / Wd.t().norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
        u_mean = {}
        for c in range(K):
            v = U[lab == c].mean(axis=0)
            fr = float(np.linalg.norm(v))
            u_mean[c] = v / (fr + 1e-12)
        u_norm = {c: float(np.linalg.norm(U[lab == c].mean(axis=0))) for c in range(K)}
        del Wd
        torch.cuda.empty_cache()

        rows = [r for r in latlas._load_questions() if (r.get('source') or '') == dataset]
        rng = np.random.default_rng(seed)
        if n and n < len(rows):
            rows = list(rng.choice(rows, size=n, replace=False))
        _set(total=len(rows), done=0, block=block,
             note='L%d：%d 组 × 2 种干预 + sham + full，%d 题' % (block, K, len(rows)))

        sizes = [int((lab == c).sum()) for c in range(K)]
        mean_size = int(np.mean(sizes))
        shams = []
        for s in range(n_sham):
            m = np.zeros(d_ff, dtype=bool)
            m[rng.choice(d_ff, size=mean_size, replace=False)] = True
            shams.append(('sham%d' % s, m))
        # dir 的匹配对照：同规模随机单元子集的平均写方向。它和簇平均方向同属「一堆单元方向的均值」，
        # 所以两者的差别只可能来自「分组方式」，不会来自「是不是均值」这个操作本身。
        u_rand = []
        for nm, m in shams:
            v = U[m].mean(axis=0)
            u_rand.append((nm, v / (np.linalg.norm(v) + 1e-12)))
        # 各向同性高斯方向：更弱的对照，用来标出「随便一个方向」的底噪
        gauss = []
        for s in range(n_sham):
            v = rng.normal(size=d).astype(np.float64)
            gauss.append(('gauss%d' % s, v / np.linalg.norm(v)))

        modes = ['zero', 'dir']
        cfg_names = (['base'] + ['C%d' % c for c in range(K)]
                     + [s[0] for s in shams] + [g[0] for g in gauss] + ['full'])
        acc: Dict[str, Dict[str, Any]] = {}
        for md in modes:
            acc[md] = {'dla': {nm: np.zeros((len(band), Hn)) for nm in cfg_names},
                       'margin': {nm: 0.0 for nm in cfg_names},
                       'kl': {nm: 0.0 for nm in cfg_names},
                       'ok': {nm: 0 for nm in cfg_names},
                       'flip': {nm: 0 for nm in cfg_names},
                       'sq': np.zeros((len(band), Hn))}
        n_ok = 0
        base_lp = None

        for i, q in enumerate(rows):
            pr = _prep(tok, q, letter_ids)
            if pr is None:
                continue
            ids, T, ans_tok = pr
            others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
                      if x != q['answer'] and letter_ids.get(x)]
            others = [o for o in others if o is not None]
            if not others or not ldirs:
                continue
            contrast = ldirs[q['answer']] - torch.stack(
                [ldirs[x] for x in 'ABCDE' if x != q['answer'] and x in ldirs]).mean(0)
            contrast = contrast / (contrast.norm() + 1e-9)

            dla0, m0, lp0 = _forward(model, device, ids, block, band, None,
                                     contrast, ans_tok, others)
            n_ok += 1
            for md in modes:
                a = acc[md]
                for li, l in enumerate(band):
                    a['dla']['base'][li] += dla0[l]
                    a['sq'][li] += dla0[l] ** 2
                a['margin']['base'] += m0
                a['kl']['base'] += 0.0
                a['ok']['base'] += int(m0 > 0)

            ops = []
            for md in modes:
                if md == 'zero':
                    ops += [(md, 'C%d' % c, ('zero', lab == c)) for c in range(K)]
                    ops += [(md, nm, ('zero', m)) for nm, m in shams]
                    ops += [(md, 'full', ('zero', np.ones(d_ff, dtype=bool)))]
                else:
                    ops += [(md, 'C%d' % c, ('dir', torch.as_tensor(u_mean[c], dtype=torch.float32)))
                            for c in range(K)]
                    ops += [(md, nm, ('dir', torch.as_tensor(v, dtype=torch.float32)))
                            for nm, v in u_rand]
                    ops += [(md, nm, ('dir', torch.as_tensor(v, dtype=torch.float32)))
                            for nm, v in gauss]
                    # dir 的阳性对照：把读数所用的答案对比方向本身投影掉——钩子或 KL 一旦失效，这里必然不动
                    ops += [(md, 'full', ('dir', contrast.detach().float()))]

            for md, nm, op in ops:
                dla1, m1, lp1 = _forward(model, device, ids, block, band, op,
                                         contrast, ans_tok, others)
                a = acc[md]
                for li, l in enumerate(band):
                    a['dla'][nm][li] += dla1[l]
                a['margin'][nm] += m1
                a['kl'][nm] += _kl(lp0, lp1)
                a['ok'][nm] += int(m1 > 0)
                if (m1 > 0) != (m0 > 0):
                    a['flip'][nm] += 1
            if (i + 1) % 10 == 0:
                _set(done=i + 1, elapsed_s=round(time.time() - t0, 1),
                     note='%d/%d 题，%.0fs' % (i + 1, len(rows), time.time() - t0))

        if n_ok < 10:
            raise RuntimeError('有效题数太少（%d），不出结论' % n_ok)

        out: Dict[str, Any] = {
            'schema': 'mech_ablate_cluster.v2', 'model_name': model_name,
            'block': block, 'dataset': dataset, 'n_questions': int(n_ok),
            'n_layers_band': len(band), 'n_heads': Hn, 'k': K, 'band': band,
            'sham_size_mean': mean_size,
            'note': '两种干预：zero=把组内单元的 down_proj 输入列置零；dir=在 L%d 输出处投影掉该组平均写方向。'
                    'kl 为 KL(base‖ablated)（自然 log，全词表，fp32），是主行为指标；'
                    'margin 为答案字母 logit 差，实测不灵敏（整层置零只动 0.15）。'
                    'dla z 以基线跨题标准差归一。all_controls 里的 sham 给经验幅度，full 是阳性对照。'
                    % block,
            'methods': {},
        }
        for md in modes:
            a = acc[md]
            base_mean = a['dla']['base'] / n_ok
            base_std = np.sqrt(np.maximum(a['sq'] / n_ok - base_mean ** 2, 1e-12))

            def z(nm, a=a, base_mean=base_mean, base_std=base_std):
                return (a['dla'][nm] / n_ok - base_mean) / base_std

            sham_z = [np.abs(z(s[0])) for s in shams]
            sa = np.stack(sham_z) if sham_z else np.zeros((1, len(band), Hn))
            thr = float(np.percentile(sa, 95))

            def marg(nm):
                return {'kl': round(float(a['kl'][nm] / n_ok), 5),
                        'dmargin': round(float((a['margin'][nm] - a['margin']['base']) / n_ok), 3),
                        'acc': round(float(a['ok'][nm] / n_ok), 3),
                        'n_flip': int(a['flip'][nm])}

            m = {'intervention': md, 'sham_z_abs_p95': round(thr, 2),
                 'sham_z_abs_max': round(float(sa.max()), 2),
                 'baseline': {'margin': round(float(a['margin']['base'] / n_ok), 3),
                              'acc': round(float(a['ok']['base'] / n_ok), 3)},
                 'controls': {nm: {**marg(nm), 'z_abs_mean': round(float(np.abs(z(nm)).mean()), 3)}
                              for nm in ([s[0] for s in shams] + ['full']
                                         + ([g[0] for g in gauss] if md == 'dir' else []))},
                 'clusters': []}
            if md == 'dir':
                m['cluster_mean_write_norm'] = {str(c): round(u_norm[c], 4) for c in range(K)}
            sham_kl = float(np.mean([a['kl'][s[0]] / n_ok for s in shams])) if shams else 0.0
            for c in range(K):
                zz = z('C%d' % c)
                m['clusters'].append({
                    'cluster': c, 'size': sizes[c],
                    **marg('C%d' % c),
                    'kl_vs_sham': round(float(a['kl']['C%d' % c] / n_ok) / sham_kl, 3) if sham_kl > 0 else None,
                    'z_abs_mean': round(float(np.abs(zz).mean()), 3),
                    'n_exceed': int((np.abs(zz) > thr).sum()),
                    'top_heads': [[int(band[i]), int(h), round(float(zz[i, h]), 2)]
                                  for i, h in sorted([(i, h) for i in range(len(band))
                                                      for h in range(Hn)],
                                                     key=lambda t: -abs(zz[t[0], t[1]]))[:8]],
                })
            out['methods'][md] = m

        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(out, ensure_ascii=False, indent=1)
        out_path(block).write_text(txt, encoding='utf-8')
        if block == 24:
            OUT_JSON.write_text(txt, encoding='utf-8')
        (ME / 'ablate_by_layer.json').write_text(
            json.dumps(_index_blocks(), ensure_ascii=False, indent=1), encoding='utf-8')
        _set(state='done', done=n_ok, total=len(rows), elapsed_s=round(time.time() - t0, 1),
             note='L%d 完成' % block)
    except Exception as e:
        _set(state='error', error='%s: %s' % (type(e).__name__, e),
             note=traceback.format_exc()[-900:])


# ------------------------------------------------------------------ 端点
def _spawn(fn, *args) -> Dict[str, Any]:
    global _THREAD
    with _LOCK:
        if _STATE['state'] == 'running':
            return {'accepted': False, 'reason': '已有任务在跑'}
        _STATE.update({'job': 'ablate', 'state': 'running', 'error': '', 'done': 0,
                       'total': 0, 'elapsed_s': 0.0, 'note': '启动'})
    _THREAD = threading.Thread(target=fn, args=args, daemon=True)
    _THREAD.start()
    return {'accepted': True, 'job': 'ablate'}


@router.post('/run')
def run(block: int = 24, dataset: str = 'C-Eval val', n: int = 0, sham: int = 2) -> dict:
    return _spawn(_run, int(block), dataset, int(n), int(sham))


@router.get('/status')
def status() -> dict:
    with _LOCK:
        st = dict(_STATE)
    st['schema'] = 'mech_ablate_status.v1'
    st['files'] = {p.name: p.stat().st_size for p in sorted(ME.glob('ablate_L*.json'))}
    return st


@router.get('')
def result(block: int = 24) -> dict:
    p = out_path(block)
    if not p.exists():
        idx = _index_blocks()
        if block == 24 and OUT_JSON.exists():
            p = OUT_JSON
        else:
            return {'available': False, 'reason': 'L%d 还没跑；已跑 %s'
                    % (block, [b['block'] for b in idx['blocks']])}
    d = json.loads(p.read_text(encoding='utf-8'))
    d['available'] = True
    return d


@router.get('/index')
def index() -> dict:
    d = _index_blocks()
    d['available'] = d['n_blocks'] > 0
    return d
