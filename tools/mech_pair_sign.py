# -*- coding: utf-8 -*-
"""H6：差分对判决实验 —— 删单侧 vs 删双侧。

H5 的结论是「top-1% 的杠杆双向对称，删掉后 14 错 / 11 对互相抵消」。
如果这批单元真是**成对正反编码**（差分对），那么拆开两侧单独删，方向应当显现：

    删正侧（Δ>0，对答案有正贡献） → 净效应为负（该对的「正端」被拿走）
    删负侧（Δ<0，压制答案的那端） → 净效应为正（「负端」被拿走 → 答案更易出现）
    删双侧（两端的拿走）           → 净效应回到 0 附近（彼此抵消）

这不是「更多/更少单元」的效应：pos / neg 数量**等量匹配**（各 45 个，Σ|Δ| 比 1.07），
both = pos ∪ neg（90 个）。所以三者之间唯一的差别是「删哪一侧」。

预注册判据（跑之前固定）：
  S1 基线可复现   本次 base 的逐题答案与 H5（mech_cot_L34_n169.json）一致率 ≥ 0.95
                  （不成立则说明 greedy 不可复现，整轮作废，不能拿旧 base 当分母）
  S2 单侧有方向   net(pos) ≤ -5 且 net(neg) ≥ +3 且 acc(neg) > acc(pos) + 0.03
                  其中 net = (由错变对) − (由对变错)
  S3 双侧抵消     |net(both)| ≤ max(2, 0.35 × max(|net(pos)|, |net(neg)|))
  S4 都在动行为   三个条件的改答率均 ≥ 2× rand97 的改答率（0.089 → ≥0.178），
                  且 McNemar(改答, rand97) p < 0.01（rand97 取自 H5 同一批题）

判读：
  S1+S2+S3 → 差分对成立：这批单元是成对正反的载体，删一侧才有方向
  S1+S2 但 S3 不过 → 单侧有方向但双侧不抵消：是「部分抵消」而非严格差分对
  S1 过 S2 不过 → 差分对被否证：单侧删除也没有方向性，杠杆只是无向抖动

用法：
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_pair_sign.py     # 3 题，验 hook 与 base 复现
  .venv/Scripts/python.exe -X utf8 tools/mech_pair_sign.py             # 169 题正式（约 70 min）
  CONDS=pos,neg .venv/Scripts/python.exe -X utf8 tools/mech_pair_sign.py   # 只跑部分条件
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
CONDS = [c for c in os.environ.get('CONDS', 'base,pos,neg,both').split(',') if c]
TAG = os.environ.get('TAG', '')
OUT = Path('tempdata/pair_sign_L%d%s%s.json'
           % (B, TAG, '_smoke' if SMOKE else ''))
REF = Path('tempdata/mech_cot_L%d_n169.json' % B)      # H5 的 base / rand97 来源
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
    prompt, _lines, _ = latlas._build_prompt(q)
    return prompt + SUFFIX


def mcnemar(b, c):
    """两点式 McNemar 精确检验（双侧）。b/c 是两条件里只有一方为真的题数。"""
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

    a = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].astype(np.float64)
    d_ff = a.size
    top = np.argsort(-np.abs(a))[:d_ff // 100]
    pos_all = top[a[top] > 0]
    neg_all = top[a[top] < 0]
    m = int(min(pos_all.size, neg_all.size))
    pos = pos_all[np.argsort(-np.abs(a[pos_all]))][:m]
    neg = neg_all[np.argsort(-np.abs(a[neg_all]))][:m]
    both = np.union1d(pos, neg)
    masks = {
        'base': None,
        'pos': np.isin(np.arange(d_ff), pos),
        'neg': np.isin(np.arange(d_ff), neg),
        'both': np.isin(np.arange(d_ff), both),
    }
    sel = {k: masks[k] for k in CONDS}
    print('ready %.0fs  %d 题 | top%d 正 %d 负 %d → 匹配各 %d，both %d'
          % (time.time() - t_all, len(rows), top.size, pos_all.size, neg_all.size,
             m, both.size), flush=True)
    print('  Σ|Δ|  pos %.2f  neg %.2f  both %.2f'
          % (np.abs(a[pos]).sum(), np.abs(a[neg]).sum(), np.abs(a[both]).sum()), flush=True)

    def gen(prompt, mask):
        enc = tok(prompt, return_tensors='pt').to(device)
        hs = []
        if mask is not None:
            msk = torch.as_tensor(mask, device=device)

            def abl(_mod, args):
                v = args[0].clone()
                v[..., msk] = 0.0
                return (v,) + tuple(args[1:])
            hs.append(model.model.layers[B].mlp.down_proj.register_forward_pre_hook(abl))
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
            txt = gen(prompt, msk)
            r['out'][nm] = {'pred': parse(txt), 'text': txt.strip()}
        recs.append(r)
        print('  %d/%d %.0fs  gold %s  %s'
              % (qi + 1, len(rows), time.time() - t0, q['answer'],
                 '  '.join('%s=%s' % (nm, r['out'][nm]['pred']) for nm in sel)), flush=True)

    # ---- 参照：H5 的 base / rand97（同题序） ----
    ref_base, ref_rand = {}, {}
    if REF.exists():
        d5 = json.loads(REF.read_text(encoding='utf-8'))
        for rr in d5['records']:
            if rr['qi'] < len(rows):
                ref_base[rr['qi']] = rr['out']['base']['pred']
                ref_rand[rr['qi']] = rr['out']['rand97']['pred']

    def acc_of(getter):
        v = [getter(r) for r in recs]
        v = [x for x in v if x is not None or True]
        return float(np.mean([x == r['gold'] for x, r in zip(v, recs)]))

    def changed_rate(getter, baseget):
        n = len(recs)
        ch = sum(1 for r in recs if getter(r) != baseget(r))
        return ch / max(n, 1)

    base_of = (lambda r: r['out']['base']['pred']) if 'base' in sel else (lambda r: ref_base.get(r['qi']))
    have_base = ('base' in sel) or (len(ref_base) >= len(rows) and len(ref_base) > 0)

    summ = {'n': len(recs), 'conds': CONDS, 'n_units': {'pos': int(pos.size), 'neg': int(neg.size),
                                                       'both': int(both.size)}}
    if have_base:
        summ['acc'] = {nm: round(acc_of(lambda r, _n=nm: r['out'][_n]['pred']), 3) for nm in [c for c in sel if c != 'base']}
        summ['format'] = {nm: round(float(np.mean([r['out'][nm]['pred'] is not None for r in recs])), 3)
                          for nm in sel}
        summ['net'] = {}
        summ['flip'] = {}
        summ['changed'] = {}
        for nm in [c for c in sel if c != 'base']:
            w = sum(1 for r in recs if base_of(r) == r['gold'] and r['out'][nm]['pred'] != r['gold'])
            rt = sum(1 for r in recs if base_of(r) != r['gold'] and r['out'][nm]['pred'] == r['gold'])
            summ['flip'][nm] = {'wrong': w, 'right': rt}
            summ['net'][nm] = rt - w
            summ['changed'][nm] = int(sum(1 for r in recs
                                          if r['out'][nm]['pred'] != base_of(r)))
        # 与 H5 rand97 的改答率对照
        if ref_rand:
            rb = sum(1 for r in recs if ref_rand.get(r['qi']) != base_of(r))
            summ['rand97_ref'] = {'changed': rb, 'rate': round(rb / len(recs), 3)}
            summ['mvp'] = {}
            for nm in [c for c in sel if c != 'base']:
                b = sum(1 for r in recs if r['out'][nm]['pred'] != base_of(r) and ref_rand.get(r['qi']) == base_of(r))
                c2 = sum(1 for r in recs if r['out'][nm]['pred'] == base_of(r) and ref_rand.get(r['qi']) != base_of(r))
                summ['mvp'][nm] = {'only_cond': b, 'only_rand': c2, 'p': round(mcnemar(b, c2), 6)}
    # ---- base 复现性 ----
    if 'base' in sel and ref_base:
        agree = sum(1 for r in recs if r['out']['base']['pred'] == ref_base.get(r['qi']))
        summ['base_repro'] = round(agree / len(recs), 3)

    # ---- 判据 ----
    v = {}
    if 'base' in sel and ref_base:
        v['S1_base_reproducible'] = bool(summ.get('base_repro', 0) >= 0.95)
    elif ref_base:
        v['S1_base_reproducible'] = None      # 本轮没跑 base，沿用 H5 的（需在别的轮次验过）
    else:
        v['S1_base_reproducible'] = None
    if have_base and 'pos' in sel and 'neg' in sel:
        np_, nn_ = summ['net']['pos'], summ['net']['neg']
        v['S2_onesided_directional'] = bool(np_ <= -5 and nn_ >= 3
                                            and summ['acc']['neg'] > summ['acc']['pos'] + 0.03)
    else:
        v['S2_onesided_directional'] = None
    if have_base and 'both' in sel and all(k in summ['net'] for k in ('pos', 'neg')):
        nb = abs(summ['net']['both'])
        thr = max(2.0, 0.35 * max(abs(summ['net']['pos']), abs(summ['net']['neg'])))
        v['S3_both_cancels'] = bool(nb <= thr)
        summ['S3_threshold'] = round(thr, 2)
    else:
        v['S3_both_cancels'] = None
    if 'rand97_ref' in summ and summ.get('mvp'):
        v['S4_moves_behaviour'] = bool(all(
            summ['changed'][nm] / len(recs) >= 2 * summ['rand97_ref']['rate']
            and summ['mvp'][nm]['p'] < 0.01 for nm in summ['mvp']))
    else:
        v['S4_moves_behaviour'] = None

    if v.get('S2_onesided_directional') and v.get('S3_both_cancels'):
        reading = '差分对成立：这批单元成对正反编码，删单侧才有方向、删双侧互相抵消。'
    elif v.get('S2_onesided_directional'):
        reading = '单侧删除有方向性，但双侧未回到零：是「部分抵消」，不是严格差分对。'
    elif v.get('S2_onesided_directional') is False:
        reading = '差分对被否证：单侧删除也没有方向性，杠杆是无向抖动。'
    else:
        reading = '条件不全，未能判定。'

    out = {'schema': 'mech_pair_sign.v1', 'model_name': model_name, 'block': B,
           'max_new_tokens': MAXNEW, 'summary': summ, 'verdict': v, 'reading': reading,
           'units': {'pos': pos.tolist(), 'neg': neg.tolist()},
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'acc': summ.get('acc'), 'net': summ.get('net'),
                      'flip': summ.get('flip'), 'changed': summ.get('changed'),
                      'base_repro': summ.get('base_repro')},
                     ensure_ascii=False, indent=1))
    print('verdict', json.dumps(v, ensure_ascii=False), '|', reading)
    print('written %s  %.0fs' % (OUT, time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
