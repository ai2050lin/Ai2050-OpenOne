# -*- coding: utf-8 -*-
"""MF-v3：代码模式在哪里压掉散文？（组件修补）

MF-v2（Phase 276）：在中层同时写入「代码 + 中文 / 英文说明」时，两模型都是代码效应完整保留、说明文字效应被压掉（同样 4 组）。
本版在**发现集文本**（MF-v2 只用确认集，这里是新任务）上：
  H0  复现压制：4 组中至少 3 组，散文保留 ≤ 0.2 且代码保留 ≥ 0.7（保留 = 同时写入的效应 / 单独写入的效应，log M 相对不写入）。
  修补：在「同时写入」运行中，把写入层之后某一层的注意力或 MLP 输出（全部位置）换成「只写散文」运行中的值，
        恢复比例 = (M_p[修补] − M_p[同时]) / (M_p[只散文] − M_p[同时])。
  H1  按任务切 A / B：A 上恢复最大的单个组件，在 B 上恢复 ≥ 0.3（4 组中至少 3 组）。
  H2  B 上 MLP 组件的正恢复之和 > 注意力组件的正恢复之和（4 组中至少 3 组）——压制主要由 MLP 计算。
  方向与写入同 MF-v2：层 round(0.5L)，共用 token 配平的风格方向，6σ，前 60% 前缀的全部位置。
用法：python scripts/mode_field_v3.py prereg | run --model M [--int8] | analyze --model M [--int8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/mode_field/MF-v3.prereg.json'
STY = F.STYLES
CODE = STY.index('code')


def prereg_doc():
    return {'id': 'MF-v3', 'schema': 'mode_field_prereg.v3', 'created': '2026-11-02',
            'basis': 'MF-v2（Phase 276）：代码与中文 / 英文说明同时写入时，两模型都在同样 4 组中压掉说明文字',
            'models': {'qwen3-4b': 'bf16（主）', 'glm4-9b-chat-hf': 'int8（复现）'},
            'materials': 'MF-v1 发现集文本（16 任务 × 5 填充，每种来源 80 段）；按任务 sha256 切 A / B 各 8 任务。'
                         '注意：风格方向也由发现集共用 token 位置求得，压制现象本身此前只在确认集上观察',
            'patch': '写入层之后每层的注意力输出与 MLP 输出各为一个组件；全部位置替换为「只写散文」运行中的值',
            'power_note': '连续量阈值；每组 80 段（A / B 各 40）',
            'doc': __doc__, 'criteria': F.CRITERIA_V3}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MF-v4）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path, int8):
    d = ROOT / 'research/mode_field' / f'MF-v3-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def in_A(task):
    tasks = sorted(set(MAT_tasks()), key=lambda t: hashlib.sha256(f'MF-v3|{t}'.encode()).hexdigest())
    return task in set(tasks[:len(tasks) // 2])


def MAT_tasks():
    import mf_materials_v1 as MAT
    return MAT.split()['disc']


def do_run(model_path, int8):
    import gc
    import torch
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    z = MF1.load_fields(key)
    L = int(z['Hs'].shape[1]) - 1
    ls = round(crit['switch_rel'] * L)
    y, tok, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    Xs = z['Hs'][D[sh], ls].astype(np.float32)
    ys, ts = y[sh][D[sh]], tok[sh][D[sh]]
    V, SG = {}, {}
    for s in range(4):
        v = F.token_matched_direction(Xs, ys, ts, s)
        V[s] = v / np.linalg.norm(v)
        SG[s] = float((Xs @ V[s]).std())
    del Xs, z
    gc.collect()
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    tok_, model = XM.load(model_path, int8)
    R = XM.Runner(tok_, model)
    C, ids = MF1.tokenize(tok_)
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    comps = [(l, kind) for l in range(ls, L) for kind in ('attn', 'mlp')]
    names = [f'{kind}{l}' for l, kind in comps]
    st = {'inject': None, 'pos': None, 'capture': None, 'patch': {}}
    cache = {}

    def inj(mod, args, kwargs):
        if st['inject'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        h[:, :st['pos']] += st['inject'][:, None, :].to(h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def comp_hook(l, kind):
        def f(m, a, o):
            out = o[0] if kind == 'attn' else o
            if st['capture'] is not None:
                cache[l, kind] = out[st['capture']].detach().clone()
            rows = st['patch'].get((l, kind))
            if rows:
                out = out.clone()
                out[rows] = cache[l, kind].to(out.dtype)
                return (out,) + tuple(o[1:]) if kind == 'attn' else out
            return None
        return f
    hooks = [R.blocks[ls].register_forward_pre_hook(inj, with_kwargs=True)]
    for l, kind in comps:
        mod = R.blocks[l].self_attn if kind == 'attn' else R.blocks[l].mlp
        hooks.append(mod.register_forward_hook(comp_hook(l, kind)))

    def masses(seqs):
        i_, att = R._batch(seqs)
        hs = R.model.model(input_ids=i_, attention_mask=att).last_hidden_state
        last = torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
        lp = torch.log_softmax(R.model.lm_head(hs[torch.arange(len(seqs), device='cuda'), last]).float(), -1)
        return torch.stack([torch.logsumexp(lp[:, e], -1) for e in EX], -1).cpu().numpy()

    t0 = time.time()
    out = {}
    for src, prose in crit['triples']:
        c, p = STY.index(src), STY.index(prose)
        vp = crit['k'] * SG[p] * V[p]
        vc = crit['k'] * SG[CODE] * V[CODE]
        texts = [i for i, cc in enumerate(C) if cc['split'] == 'disc' and cc['style'] == src]
        M = np.zeros((len(texts), 4 + len(comps), 4), np.float32)
        A = np.array([in_A(C[i]['task']) for i in texts])
        for ti, i in enumerate(texts):
            n = max(2, round(0.6 * len(ids[i])))
            seq = ids[i][:n]
            st['pos'] = n
            st['inject'] = torch.as_tensor(np.stack([np.zeros_like(vp), vp, vc, vp + vc]), dtype=torch.float32, device='cuda')
            st['capture'] = 1                                         # 第 1 行 = 只写散文
            try:
                M[ti, :4] = masses([seq] * 4)
            finally:
                st['capture'] = None
            st['inject'] = torch.as_tensor(np.stack([vp + vc] * len(comps)), dtype=torch.float32, device='cuda')
            st['patch'] = {cp: [j] for j, cp in enumerate(comps)}
            try:
                M[ti, 4:] = masses([seq] * len(comps))
            finally:
                st['patch'] = {}
            cache.clear()
        st['inject'] = None
        np.save(od / f'mf_v3_{src}_{prose}.npy', M)
        np.save(od / f'mf_v3_{src}_{prose}_A.npy', A)
        jm, sm, bm, cm = M[:, 3], M[:, 1], M[:, 0], M[:, 2]
        out[f'{src}:{prose}'] = {'prose_retain': float((jm[:, p] - bm[:, p]).mean() / (sm[:, p] - bm[:, p]).mean()),
                                 'code_retain': float((jm[:, CODE] - bm[:, CODE]).mean() / (cm[:, CODE] - bm[:, CODE]).mean())}
        print(f'@@progress {src}:{prose} {out[f"{src}:{prose}"]} {round(time.time() - t0)}s', flush=True)
    for h in hooks:
        h.remove()
    save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                            'precision': 'int8' if int8 else 'bf16', 'criteria': crit, 'switch_layer': ls, 'n_blocks': L,
                            'components': names, 'H0_raw': out, 'seconds': round(time.time() - t0, 1)})


def do_analyze(model_path, int8):
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    rep = json.loads(report_path(model_path, int8).read_text(encoding='utf-8'))
    names = rep['components']
    h0, h1 = {}, {}
    for src, prose in crit['triples']:
        p = STY.index(prose)
        M = np.load(od / f'mf_v3_{src}_{prose}.npy')
        A = np.load(od / f'mf_v3_{src}_{prose}_A.npy')
        r = rep['H0_raw'][f'{src}:{prose}']
        h0[f'{src}:{prose}'] = {**r, 'pass': bool(r['prose_retain'] <= crit['H0_prose_retain_max'] and r['code_retain'] >= crit['H0_code_retain_min'])}
        sel = F.select_component(M[:, 4:, p], M[:, 3, p], M[:, 1, p], A, names, crit)
        rall = F.restoration(M[:, 4:, p], M[:, 3, p], M[:, 1, p])
        sel['profile_all'] = {n: float(v) for n, v in zip(names, rall)}
        h1[f'{src}:{prose}'] = sel
    n0 = sum(v['pass'] for v in h0.values())
    n1 = sum(v['pass'] for v in h1.values())
    n2 = sum(v['mlp_pos_B'] > v['attn_pos_B'] for v in h1.values())
    verdict = {'H0': bool(n0 >= crit['H0_min_triples']), 'H1': bool(n1 >= crit['H1_min_triples']), 'H2': bool(n2 >= crit['H2_min_triples'])}
    rep = save(model_path, int8, {'H0': h0, 'H1': h1, 'counts': {'H0': n0, 'H1': n1, 'H2': n2}, 'verdict': verdict})
    p = write_result(f'mf_v3_{Path(model_path).name}{"_int8" if int8 else ""}', rep, model_path=model_path, seed=crit['seed'],
                     tool='scripts/mode_field_v3.py')
    for k_, v in h1.items():
        print(k_, v['component'], round(v['restore_A'], 2), round(v['restore_B'], 2), 'mlp', round(v['mlp_pos_B'], 2), 'attn', round(v['attn_pos_B'], 2),
              [(round(a, 2), b) for a, b in v['top5_B']])
    print(json.dumps(verdict), rep['counts'], '\nprovenance →', p)


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
