# -*- coding: utf-8 -*-
"""H7：放大 vs 删除 —— 不依赖「删除」这个操作本身的方向性检验。

H6 用「删哪一侧」测方向性，但删是一种**对称且极端**的操作（×0），
本身可能引入副作用（H5 里删 top-97 反而让格式遵循率从 0.888 升到 0.994）。
更干净的做法是把同一侧单元的贡献**推向两个相反方向**：

    abl_pos  该侧 × 0     （H6 已测：应当掉分）
    amp_pos  该侧 × 2     （若该侧真的推高答案，应当提分）
    abl_neg  该侧 × 0     （H6 已测：应当提分或不掉）
    amp_neg  该侧 × 2     （若该侧真的压制答案，应当掉分）

如果「差分对」成立，四个格应当形成**单调的符号模式**：
    amp_neg  <  base  <  amp_pos        （正侧推高、负侧压低）
且两个操作对同一侧的效果方向相反（×0 与 ×2 分别落在 base 两侧）。
这与「删掉就没了、什么都测不出」的无向假设直接冲突。

预注册判据：
  A1 单侧单调     acc(amp_neg) < acc(base) < acc(amp_pos)，且两者差距 ≥ 0.05
  A2 同侧反向     net(amp_pos) 与 net(abl_pos) 符号相反，且 |net(amp_pos)| ≥ 3；
                  net(amp_neg) 与 net(abl_neg) 符号相反，且 |net(amp_neg)| ≥ 3
  A3 非饱和       acc(amp_pos) 未达到 1.0（否则是任务天花板，不是效应）

判读：
  A1+A2 → 单元确实是**有符号的功能载体**，「删除测不出」只是对称操作抹平了符号
  A1 过 A2 不过 → 单侧单调但对侧不对称：只有一端真正参与答案计算
  A1 不过 → 这些单元的符号不是功能性的（只是归因的数值符号）

用法：
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_ampl.py
  AMP=2.0 .venv/Scripts/python.exe -X utf8 tools/mech_ampl.py          # 169 题 3 条件（约 70 min）
  CONDS=base,amp_pos,amp_neg .venv/Scripts/python.exe -X utf8 tools/mech_ampl.py
"""
import json
import os
import re
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = int(os.environ.get('LAYER', 34))
SEED = 20261101
SMOKE = os.environ.get('SMOKE') == '1'
NQ = int(os.environ.get('NQ', 3 if SMOKE else 169))
AMP = float(os.environ.get('AMP', 2.0))
CONDS = [c for c in os.environ.get('CONDS', 'base,amp_pos,amp_neg').split(',') if c]
TAG = os.environ.get('TAG', '')
OUT = Path('tempdata/ampl_L%d%s%s.json' % (B, TAG, '_smoke' if SMOKE else ''))
REF_SIGN = Path('tempdata/pair_sign_L%d.json' % B)      # H6 的 pos / neg 单元表
MAXNEW = int(os.environ.get('MAXNEW', 260))
SUFFIX = '\n请先简短分析理由，最后单独一行写「答案：X」（X 为选项字母）。\n分析：'
PAT = re.compile(r'答案\s*[：:]\s*([A-E])')
PAT2 = re.compile(r'([A-E])\s*(?:项)?\s*(?:是正确的|正确|符合题意|为正确答案)')


def parse(txt):
    m = PAT.findall(txt)
    if m:
        return m[-1]
    m = PAT2.findall(txt)
    return m[-1] if m else None


def build(q):
    prompt, _l, _ = latlas._build_prompt(q)
    return prompt + SUFFIX


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n)


def main():
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()

    rows_all = [r for r in latlas._load_questions() if r.get('source')]
    cm = [r for r in rows_all if r['source'] == 'CMMLU test']
    by_sub = {}
    for r in cm:
        by_sub.setdefault(r['subject'], []).append(r)
    rng = np.random.default_rng(SEED)
    spec = [('high_school_biology', 19), ('high_school_geography', 19),
            ('chinese_history', 10), ('world_history', 10),
            ('chinese_civil_service_exam', 47), ('logical', 22),
            ('modern_chinese', 23), ('ancient_chinese', 19)]
    pool = []
    for s, k in spec:
        p = by_sub[s]
        pool += [p[i] for i in rng.choice(len(p), size=k, replace=False)]
    rows = pool[:NQ]

    # 单元表直接沿用 H6（保证同一批单元），避免重算出现两套定义
    if REF_SIGN.exists():
        s6 = json.loads(REF_SIGN.read_text(encoding='utf-8'))
        pos = np.asarray(s6['units']['pos'], dtype=np.int64)
        neg = np.asarray(s6['units']['neg'], dtype=np.int64)
        src = 'pair_sign_L%d.json' % B
    else:
        a = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].astype(np.float64)
        d0 = a.size
        top = np.argsort(-np.abs(a))[:d0 // 100]
        pa, na = top[a[top] > 0], top[a[top] < 0]
        m = int(min(pa.size, na.size))
        pos = pa[np.argsort(-np.abs(a[pa]))][:m]
        neg = na[np.argsort(-np.abs(a[na]))][:m]
        src = 'unit_attr_L%d.npz' % B
    d_ff = int(np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].size)
    masks = {'base': None, 'amp_pos': np.isin(np.arange(d_ff), pos),
             'amp_neg': np.isin(np.arange(d_ff), neg),
             'abl_pos': np.isin(np.arange(d_ff), pos),
             'abl_neg': np.isin(np.arange(d_ff), neg)}
    scales = {'amp_pos': AMP, 'amp_neg': AMP, 'abl_pos': 0.0, 'abl_neg': 0.0}
    sel = {k: masks[k] for k in CONDS}
    print('ready %.0fs  %d 题 | 单元表来自 %s | pos %d neg %d | AMP %.2f'
          % (time.time() - t_all, len(rows), src, pos.size, neg.size, AMP), flush=True)

    def gen(prompt, mask, scale):
        enc = tok(prompt, return_tensors='pt').to(device)
        hs = []
        if mask is not None:
            msk = torch.as_tensor(mask, device=device)
            s = float(scale)

            def amp(_mod, args):
                v = args[0].clone()
                v[..., msk] = v[..., msk] * s
                return (v,) + tuple(args[1:])
            hs.append(model.model.layers[B].mlp.down_proj.register_forward_pre_hook(amp))
        try:
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            for h in hs:
                h.remove()

    recs = []
    t0 = time.time()
    for qi, q in enumerate(rows):
        prompt = build(q)
        r = {'qi': qi, 'subject': q['subject'], 'gold': q['answer'], 'out': {}}
        for nm, msk in sel.items():
            sc = 1.0 if nm == 'base' else scales.get(nm, AMP)
            txt = gen(prompt, msk, sc)
            r['out'][nm] = {'pred': parse(txt), 'text': txt.strip()}
        recs.append(r)
        print('  %d/%d %.0fs  gold %s  %s'
              % (qi + 1, len(rows), time.time() - t0, q['answer'],
                 '  '.join('%s=%s' % (n, r['out'][n]['pred']) for n in sel)), flush=True)

    def acc(nm):
        return round(float(np.mean([r['out'][nm]['pred'] == r['gold'] for r in recs])), 3)

    def net(nm):
        w = sum(1 for r in recs if r['out']['base']['pred'] == r['gold'] and r['out'][nm]['pred'] != r['gold'])
        rt = sum(1 for r in recs if r['out']['base']['pred'] != r['gold'] and r['out'][nm]['pred'] == r['gold'])
        return rt - w, w, rt

    summ = {'n': len(recs), 'conds': CONDS, 'amp': AMP, 'unit_src': src,
            'acc': {nm: acc(nm) for nm in sel},
            'format': {nm: round(float(np.mean([r['out'][nm]['pred'] is not None for r in recs])), 3)
                       for nm in sel}}
    if 'base' in sel:
        summ['net'] = {nm: net(nm)[0] for nm in sel if nm != 'base'}
        summ['flip'] = {nm: {'wrong': net(nm)[1], 'right': net(nm)[2]} for nm in sel if nm != 'base'}
        summ['changed'] = {nm: int(sum(1 for r in recs if r['out'][nm]['pred'] != r['out']['base']['pred']))
                           for nm in sel if nm != 'base'}
        summ['mvp'] = {}
        for a1, b1 in [('amp_pos', 'abl_pos'), ('amp_neg', 'abl_neg')]:
            if a1 in sel and b1 in sel:
                p1 = [r['out'][a1]['pred'] != r['out']['base']['pred'] for r in recs]
                p2 = [r['out'][b1]['pred'] != r['out']['base']['pred'] for r in recs]
                b_ = sum(1 for x, y in zip(p1, p2) if x and not y)
                c_ = sum(1 for x, y in zip(p1, p2) if y and not x)
                summ['mvp']['%s_vs_%s' % (a1, b1)] = {'only_a': b_, 'only_b': c_,
                                                      'p': round(mcnemar(b_, c_), 6)}

    v = {}
    if 'base' in sel and 'amp_pos' in sel and 'amp_neg' in sel:
        ap, an, bs = summ['acc']['amp_pos'], summ['acc']['amp_neg'], summ['acc']['base']
        v['A1_onesided_monotone'] = bool(an < bs < ap and (ap - an) >= 0.05)
        v['A3_not_saturated'] = bool(ap < 0.995)
    else:
        v['A1_onesided_monotone'] = None
        v['A3_not_saturated'] = None
    if 'base' in sel and all(k in summ.get('net', {}) for k in ('amp_pos', 'abl_pos', 'amp_neg', 'abl_neg')):
        n = summ['net']
        v['A2_same_side_opposite'] = bool(
            n['amp_pos'] * n['abl_pos'] < 0 and abs(n['amp_pos']) >= 3
            and n['amp_neg'] * n['abl_neg'] < 0 and abs(n['amp_neg']) >= 3)
    else:
        v['A2_same_side_opposite'] = None

    if v.get('A1_onesided_monotone') and v.get('A2_same_side_opposite'):
        reading = '单元是有符号的功能载体：×0 与 ×AMP 落在 base 两侧，「删除测不出」只是对称操作抹平了符号。'
    elif v.get('A1_onesided_monotone'):
        reading = '单侧单调，但对侧不对称：只有一端真正参与答案计算。'
    elif v.get('A1_onesided_monotone') is False:
        reading = '这些单元的符号不是功能性的：放大单侧也给不出方向。'
    else:
        reading = '条件不全，未能判定。'

    out = {'schema': 'mech_ampl.v1', 'model_name': model_name, 'block': B,
           'max_new_tokens': MAXNEW, 'summary': summ, 'verdict': v, 'reading': reading,
           'units': {'pos': pos.tolist(), 'neg': neg.tolist()},
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'acc': summ['acc'], 'net': summ.get('net'), 'flip': summ.get('flip'),
                      'changed': summ.get('changed')}, ensure_ascii=False, indent=1))
    print('verdict', json.dumps(v, ensure_ascii=False), '|', reading)
    print('written %s  %.0fs' % (OUT, time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
