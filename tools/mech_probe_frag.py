# -*- coding: utf-8 -*-
"""探针：qi=73 / 25 / 78 在随机单点下到底是「答错」还是「解析崩掉」。

H17 的 B 段把所有 pred is None 也算打坏（与 A 段 `damaged_incl_parsefail` 同口径），
所以「随机单点最多打坏 2 道」里的 2 可能全是在数解析失败。这决定了预注册的 O2
（rand_max ≤ 1）到底是因为「特异性不存在」还是因为「把一个格式崩塌题算进了计数」。

本探针只跑 3 个单元 × 3 道题（9 次前向，约 1 min），把**原文**和**解析结果**都记下来：

  767  : 已知在 qi=73 上是解析失败（core_one_L34.json 的 parse fail 列表里有 73）
  2098 : 随机单点，B 段记录打坏 [73]
  7964 : 随机单点，B 段记录打坏 [25, 73]

判读：若随机单点在 73 上也是 None ⇒ 73 是「任何扰动都吐不出字母」的格式脆性题，
它同时抬高 767 与随机组的计数，不该用来判特异性；若随机单点在 73 上给出**错字母**，
则「随机单点也能改答」成立，特异性确实弱。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_probe_frag.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import mech_pair_sign as msign
from server import mech_pair_service as mpair

B = int(os.environ.get('B', '34'))
MAXNEW = int(os.environ.get('MAXNEW', '260'))
QS = [73, 25, 78]
UNITS = [767, 2098, 7964]
OUT = Path('tempdata') / f'probe_frag_L{B}.json'


def load_rows():
    from server import lang_atlas_service as latlas
    rows_all = [r for r in latlas._load_questions() if r.get('source')]
    cm = [r for r in rows_all if r['source'] == 'CMMLU test']
    by_sub = {}
    for r in cm:
        by_sub.setdefault(r['subject'], []).append(r)
    rng = np.random.default_rng(20261101)
    spec = [('high_school_biology', 19), ('high_school_geography', 19),
            ('chinese_history', 10), ('world_history', 10),
            ('chinese_civil_service_exam', 47), ('logical', 22),
            ('modern_chinese', 23), ('ancient_chinese', 19)]
    pool = []
    for s, k in spec:
        p = by_sub[s]
        pool += [p[i] for i in rng.choice(len(p), size=k, replace=False)]
    return pool[:169]


def main() -> int:
    t0 = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()
    core = json.loads((Path('tempdata') / f'core_L{B}.json').read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    rows = load_rows()
    d_ff = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].size
    prompts = {q: msign.build(rows[q]) for q in QS}

    def gen(prompt, units):
        enc = tok(prompt, return_tensors='pt').to(device)
        hs = []
        if units:
            msk = np.isin(np.arange(d_ff), np.array(units))
            msk_t = torch.as_tensor(msk, device=device)

            def abl(_mod, args):
                v = args[0].clone()
                v[..., msk_t] = 0.0
                return (v,) + tuple(args[1:])
            hs.append(model.model.layers[B].mlp.down_proj.register_forward_pre_hook(abl))
        try:
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            return msign.parse(txt), txt
        finally:
            for h in hs:
                h.remove()

    recs = []
    for q in QS:
        basep = R[q]['out']['base']['pred']
        rec = {'qi': q, 'gold': rows[q]['answer'], 'base': basep,
               'subject': rows[q]['subject'], 'conds': {}}
        for u in [None] + UNITS:
            p, txt = gen(prompts[q], [u] if u is not None else None)
            rec['conds'][str(u)] = {'pred': p, 'tail': txt[-70:].replace('\n', ' ')}
            print('  qi=%-3d unit=%-5s pred=%-5s gold=%s  ...%s'
                  % (q, u, p, rows[q]['answer'], txt[-45:].replace('\n', ' ')), flush=True)
        recs.append(rec)

    out = {'schema': 'mech_probe_frag.v1', 'model_name': model_name, 'block': B,
           'questions': QS, 'units': UNITS, 'records': recs,
           'secs': round(time.time() - t0, 1),
           'note': 'base 行是当轮重跑的（不复用），用于确认探针与 H17 走同一条 gen 路径'}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
