# -*- coding: utf-8 -*-
"""MF-v5：MF-v4 的 32 个门控单元在自然的代码 / 散文切换中做什么？

材料（scripts/mf_materials_v1.mixed_corpus，只用 MF-v1 确认集的 8 个任务）：
  md   说明段落（中 / 英）+ ```python 代码块 ``` + 结语；
  cmt  代码在函数头之后、return 之前各插一行中 / 英文注释（取说明的首句与末句）。
门控得分 G_t = Σ_{i∈G} sign_i A_i(t)；关门 = 32 个单元在全部位置设为自然散文位置（MF-v1 确认集中 / 英 / 论文）上的均值；
对照 = 5 个幅度匹配集合（同层、自然文本标准差最接近的 10 个之一）做同样处理。
  H1  G 跟随文本内的切换：md 中代码块 vs 散文位置 AUC ≥ 0.8；cmt 中代码行 vs 注释行 AUC ≥ 0.7（两条都成立）。
  H2  教师强制：关门后代码位置的 log M_散文（中 ∪ 英 ∪ 论文专属词表的下一词质量）平均上升 ≥ 0.5 nat，且 ≥ 每个幅度匹配集合的 3 倍。
  H3  自由生成（代码前缀，贪心 24 个 token）：关门后生成 token 中散文 token（散文专属词表或含汉字）的比例上升 ≥ 0.02，且大于每个幅度匹配集合的上升。
  描述：边界附近 G 的平均曲线与切换潜伏期；注释位置上关门的作用；代码下一词损失的变化；生成样例。
用法：python scripts/mode_field_v5.py prereg | run --model M [--int8] | analyze --model M [--int8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mf_materials_v1 as MAT  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v4 as MF4  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/mode_field/MF-v5.prereg.json'
STY = F.STYLES
CJK = re.compile('[一-鿿]')


def prereg_doc():
    C = MAT.mixed_corpus()
    return {'id': 'MF-v5', 'schema': 'mode_field_prereg.v5', 'created': '2026-11-02',
            'basis': 'MF-v4（Phase 278）：每个模型 32 个后段 MLP 单元承担人工写入下的「代码压过散文」；本版检验自然切换',
            'models': {'qwen3-4b': 'bf16（主）', 'glm4-9b-chat-hf': 'int8（复现）'},
            'units': 'MF-v4 选出的 32 个单元与符号（mf_v4_units.npz），不重新选择',
            'materials': {'source': 'scripts/mf_materials_v1.mixed_corpus', 'n_texts': len(C),
                          'sha256_texts': hashlib.sha256(json.dumps([c['text'] for c in C], ensure_ascii=False).encode()).hexdigest(),
                          'generation_prefixes': 'MF-v1 确认集代码文本前 50% token（40）；md 文本截到代码块中部（80）'},
            'power_note': '连续量与比例阈值；160 段混合文本、120 个生成前缀',
            'doc': __doc__, 'criteria': F.CRITERIA_V5}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MF-v6）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path, int8):
    d = ROOT / 'research/mode_field' / f'MF-v5-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def seg_labels(tok, c):
    offs = tok(c['text'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    lab = []
    for a, _ in offs:
        lab.append(next((s for st, en, s in c['spans'] if st <= a < en), c['spans'][-1][2]))
    return np.array(lab)


def do_run(model_path, int8):
    import torch
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    v4 = json.loads(MF4.report_path(model_path, int8).read_text(encoding='utf-8'))
    TL = v4['layers']
    off = {int(k_): v for k_, v in v4['unit_offsets'].items()}
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of, rd = [int(x) for x in U['top']], U['layer_of'], U['rd_mean']
    sign = np.sign(rd[top])
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    tok, model = XM.load(model_path, int8)
    R = XM.Runner(tok, model)
    PRO = torch.as_tensor(sorted(set(excl['zh']) | set(excl['en']) | set(excl['paper'])), device='cuda')
    prose_set = set(PRO.tolist())
    t0 = time.time()
    st = {'capture': False, 'abl': None}
    cap = {}

    def dp_hook(l):
        def f(m, a):
            x = a[0]
            if st['capture']:
                cap[l] = x.detach()
            if st['abl'] is not None and l in st['abl']:
                x = x.clone()
                for row, (units, vals) in enumerate(st['abl'][l]):
                    if units is not None:
                        x[row][..., units] = vals.to(x.dtype)
                return (x,) + tuple(a[1:])
            return None
        return f
    hooks = [R.blocks[l].mlp.down_proj.register_forward_pre_hook(dp_hook(l)) for l in TL]
    # 0 自然文本上的散文均值与标准差（MF-v1 确认集）
    C1, ids1 = MF1.tokenize(tok)
    conf = [i for i, c in enumerate(C1) if c['split'] == 'conf']
    s0 = {l: 0 for l in TL}; s1 = {l: 0 for l in TL}; s2 = {l: 0 for l in TL}; n_all = 0; n_pro = 0
    for j in range(0, len(conf), 16):
        chunk = conf[j:j + 16]
        st['capture'] = True
        try:
            R.forward([ids1[i] for i in chunk])
        finally:
            st['capture'] = False
        for l in TL:
            a = torch.cat([cap[l][b, :len(ids1[i])].float() for b, i in enumerate(chunk)])
            pm = torch.cat([torch.full((len(ids1[i]),), C1[i]['style'] != 'code', device='cuda') for i in chunk])
            s0[l] = s0[l] + a[pm].sum(0)
            s1[l] = s1[l] + a.sum(0)
            s2[l] = s2[l] + (a ** 2).sum(0)
        n_all += sum(len(ids1[i]) for i in chunk)
        n_pro += sum(len(ids1[i]) for i in chunk if C1[i]['style'] != 'code')
    pmean = np.concatenate([(s0[l] / n_pro).cpu().numpy() for l in TL])
    sd = np.concatenate([torch.sqrt(torch.clamp(s2[l] / n_all - (s1[l] / n_all) ** 2, min=0)).cpu().numpy() for l in TL])
    rng = np.random.default_rng(crit['seed'])
    topset = set(top)
    msets = []
    for _ in range(crit['n_matched_sets']):
        s = []
        for i in top:
            pool = np.array([j for j in np.where(layer_of == layer_of[i])[0] if j not in topset and j not in s])
            s.append(int(rng.choice(pool[np.argsort(np.abs(sd[pool] - sd[i]))[:crit['near_k']]])))
        msets.append(s)

    def spec(units):
        o = {}
        for i in units:
            o.setdefault(int(layer_of[i]), []).append(i)
        return {l: (torch.as_tensor([i - off[l] for i in v], device='cuda'), torch.as_tensor(pmean[v], device='cuda')) for l, v in o.items()}
    variants = [None, spec(top)] + [spec(s) for s in msets]

    def set_abl(rows):
        st['abl'] = {l: [(v[l] if (v is not None and l in v) else (None, None)) for v in rows] for l in TL}
    tl_units = {}
    for k_, i in enumerate(top):
        tl_units.setdefault(int(layer_of[i]), []).append((i - off[int(layer_of[i])], sign[k_]))
    # 1 H1：门控得分
    C = MAT.mixed_corpus()
    ids = [tok(c['text'], add_special_tokens=False).input_ids for c in C]
    segs = [seg_labels(tok, c) for c in C]
    Gs = []
    for j in range(0, len(C), 16):
        chunk = list(range(j, min(j + 16, len(C))))
        st['capture'] = True
        try:
            R.forward([ids[i] for i in chunk])
        finally:
            st['capture'] = False
        for b, i in enumerate(chunk):
            n = len(ids[i])
            G = np.zeros(n)
            for l, us in tl_units.items():
                a = cap[l][b, :n][:, [u for u, _ in us]].float().cpu().numpy()
                G += a @ np.array([s for _, s in us])
            Gs.append(G)
    print(f'@@progress H1 collected {round(time.time() - t0)}s', flush=True)
    # 2 H2：教师强制
    T2 = {'code': [[] for _ in variants], 'comment': [[] for _ in variants], 'prose': [[] for _ in variants]}
    NLL = {'code': [[] for _ in variants]}
    for i, c in enumerate(C):
        seq = ids[i]
        set_abl(variants)
        try:
            o = R.forward([seq] * len(variants))
        finally:
            st['abl'] = None
        lp = torch.log_softmax(o.logits.float(), -1)
        mp = torch.logsumexp(lp[..., PRO], -1).cpu().numpy()                      # [V, T]
        nxt = torch.as_tensor(seq[1:], device='cuda')
        nll = (-lp[:, :-1].gather(-1, nxt[None, :, None].expand(len(variants), -1, 1))[..., 0]).cpu().numpy()
        s = segs[i]
        for seg in T2:
            m = (s == seg)
            m[-1] = False
            if m.any():
                for v in range(len(variants)):
                    T2[seg][v].append(mp[v][m])
        m = (s == 'code'); m[-1] = False
        for v in range(len(variants)):
            NLL['code'][v].append(nll[v][m[:-1]])
        del o, lp
    print(f'@@progress H2 done {round(time.time() - t0)}s', flush=True)
    t2 = {seg: [float(np.concatenate(x).mean()) for x in vv] for seg, vv in T2.items()}
    nl = [float(np.concatenate(x).mean()) for x in NLL['code']]
    # 3 H3：自由生成
    prefixes = [ids1[i][:max(4, len(ids1[i]) // 2)] for i in conf if C1[i]['style'] == 'code']
    for i, c in enumerate(C):
        if c['kind'] == 'md':
            cpos = np.where(segs[i] == 'code')[0]
            prefixes.append(ids[i][:int(cpos[len(cpos) // 2]) + 1])
    gen = [[] for _ in variants]
    samples = []
    for pi, p in enumerate(prefixes):
        set_abl(variants)
        try:
            out = model.generate(input_ids=torch.as_tensor([p] * len(variants), device='cuda'), max_new_tokens=crit['gen_tokens'], do_sample=False,
                                 pad_token_id=R.pad)
        finally:
            st['abl'] = None
        for v in range(len(variants)):
            g = out[v, len(p):].tolist()
            gen[v].append(g)
        if pi % 20 == 0:
            samples.append({'prefix_tail': tok.decode(p[-12:]), 'base': tok.decode(gen[0][-1]), 'gate_off': tok.decode(gen[1][-1]),
                            'matched0': tok.decode(gen[2][-1])})
    cache = {}

    def is_prose(t):
        if t not in cache:
            cache[t] = (t in prose_set) or bool(CJK.search(tok.decode([t])))
        return cache[t]
    rate = [float(np.mean([is_prose(t) for g in gv for t in g])) for gv in gen]
    for h in hooks:
        h.remove()
    np.savez(od / 'mf_v5_gate.npz', G=np.concatenate(Gs), seg=np.concatenate(segs),
             kind=np.concatenate([[c['kind']] * len(ids[i]) for i, c in enumerate(C)]), msets=np.array(msets), sd=sd, pmean=pmean)
    (od / 'mf_v5_gen.json').write_text(json.dumps({'gen': gen, 'n_prefix': len(prefixes)}), encoding='utf-8')
    save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                            'precision': 'int8' if int8 else 'bf16', 'criteria': crit, 'layers': TL,
                            'teacher_forced_log_mass': t2, 'code_nll': nl, 'gen_prose_rate': rate, 'gen_samples': samples,
                            'variants': ['base', 'gate_off'] + [f'matched{j}' for j in range(len(msets))],
                            'seconds': round(time.time() - t0, 1)})


def do_analyze(model_path, int8):
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    rep = json.loads(report_path(model_path, int8).read_text(encoding='utf-8'))
    z = np.load(od / 'mf_v5_gate.npz')
    G, seg, kind = z['G'], z['seg'], z['kind']
    md, cm = kind == 'md', kind == 'cmt'
    a_md = F.segment_auc(G[md], seg[md], 'code', 'prose')
    a_cm = F.segment_auc(G[cm], seg[cm], 'code', 'comment')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    C = MAT.mixed_corpus()
    segs = [seg_labels(tok, c) for c in C]
    lens = np.cumsum([0] + [len(s) for s in segs])
    Gs = [G[lens[i]:lens[i + 1]] for i in range(len(C))]
    W = crit['latency_window']
    prof = {}
    for k_, frm, to in (('md', 'fence', 'code'), ('md', 'code', 'fence'), ('cmt', 'code', 'comment'), ('cmt', 'comment', 'code')):
        sel = [i for i, c in enumerate(C) if c['kind'] == k_]
        prof[f'{k_}:{frm}->{to}'] = F.boundary_profile([Gs[i] for i in sel], [segs[i] for i in sel], frm, to, W)
    seg_means = {k_: {s: float(G[(kind == k_) & (seg == s)].mean()) for s in np.unique(seg[kind == k_])} for k_ in ('md', 'cmt')}
    h1 = {'auc_md_code_vs_prose': a_md, 'auc_cmt_code_vs_comment': a_cm, 'segment_means': seg_means, 'boundaries': prof,
          'pass': bool(a_md >= crit['H1_md_auc_min'] and a_cm >= crit['H1_cmt_auc_min'])}
    t2 = rep['teacher_forced_log_mass']
    rise = lambda seg_, v: t2[seg_][v] - t2[seg_][0]
    nm = len(rep['variants']) - 2
    h2 = {'code_rise_gate_off': rise('code', 1), 'code_rise_matched': [rise('code', 2 + j) for j in range(nm)],
          'comment_rise_gate_off': rise('comment', 1), 'prose_rise_gate_off': rise('prose', 1),
          'code_nll_change_gate_off': rep['code_nll'][1] - rep['code_nll'][0],
          'code_nll_change_matched': [rep['code_nll'][2 + j] - rep['code_nll'][0] for j in range(nm)]}
    h2['pass'] = bool(h2['code_rise_gate_off'] >= crit['H2_prose_rise_min']
                      and h2['code_rise_gate_off'] >= crit['H2_over_matched'] * max(max(h2['code_rise_matched']), 1e-9))
    r = rep['gen_prose_rate']
    h3 = {'rate_base': r[0], 'rate_gate_off': r[1], 'rate_matched': r[2:], 'rise': r[1] - r[0], 'rise_matched': [x - r[0] for x in r[2:]]}
    h3['pass'] = bool(h3['rise'] >= crit['H3_rate_rise_min'] and h3['rise'] > max(h3['rise_matched']))
    rep = save(model_path, int8, {'H1': h1, 'H2': h2, 'H3': h3, 'verdict': {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}})
    p = write_result(f'mf_v5_{Path(model_path).name}{"_int8" if int8 else ""}', rep, model_path=model_path, seed=crit['seed'],
                     tool='scripts/mode_field_v5.py')
    print(json.dumps(rep['verdict']), f'\nH1 md={a_md:.3f} cmt={a_cm:.3f} means={seg_means}',
          '\nlatency', {k_: v.get('latency') for k_, v in prof.items()},
          '\nH2', {k_: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k_, v in h2.items() if k_ != 'pass'},
          '\nH3', {k_: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k_, v in h3.items() if k_ != 'pass'},
          '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--int8', action='store_true')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model, a.int8), 'analyze': lambda: do_analyze(a.model, a.int8)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
