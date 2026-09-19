"""Phase 2783: bsub-free carrier-head-only arm for natural text.

Phase 2782 decomposed natural-row collateral into a GENERIC floor (any
unit-norm bsub breaks ~3.5/13 at alpha=0.3, ~1/13 at alpha=0.1) plus a
direction-specific excess (~2-3/13).  Removing the bsub perturbation
entirely while keeping the carrier-head ablation isolates the
competition-removal lever from the generic direction-perturbation floor.

Preregistered (frozen before any forward):
  T1: three arms on the SAME 48 frozen 2780 items, alpha-independent
      head selection identical to 2780/2782 (top-4 rival-gain heads,
      M={31..35}, u = W_U[rival]-W_U[target]):
      A0 heads-only: o_proj head-slice ablation, NO bsub.
      A1 heads+bsub alpha in {0.05, 0.1} (bridging configs).
      A2 plain bsub alpha in {0.05, 0.1} (floor reference on wrong rows
         only; correct-row floor already measured in 2782 C-R1).
  T2: mitigated iff EXISTS arm with repair >= 7/35 AND collateral
      <= 2/13; else 'head_only_failed' + full table.
Gates: G1 native args reproduce (35 wrong / 13 correct); G2 heads-only
  must not touch the residual stream outside o_proj head slices.
Descriptive: which heads are selected per row (overlap with 2782).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2780_rdc_natural_pilot as p2780

BASE = cc.BASE
OUT = BASE / 'phase2783' / 'qwen4_head_only_arm'
M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
ITEMS = p2780.ITEMS

PREREG = {
    'T1': 'arms on same 48 items: A0 heads-only; A1 heads+bsub '
          'alpha {0.05,0.1}; A2 plain bsub alpha {0.05,0.1} wrong rows',
    'T2': 'mitigated iff exists arm with repair >= 7/35 AND collateral '
          '<= 2/13; else head_only_failed + full table',
    'verdict': 'recorded per T2',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(cc.ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    W_O = {l: model.model.layers[l].self_attn.o_proj.weight.detach().
           float().cpu().numpy() for l in M_LAYERS}

    state = {'bsub': None, 'abl': frozenset()}
    cap = {'on': False, 'oproj': {}}
    cap_hfin = {'h': None}
    handles = []

    def make_bsub_hook():
        def hook(module, args, output):
            if state['bsub'] is None:
                return None
            out = output[0] if isinstance(output, tuple) else output
            h = out[0, -1]
            out[0, -1] = h - state['bsub'][0] * h.norm() * state['bsub'][1]
            return None
        return hook

    def make_abl_pre(l):
        def pre(module, args):
            hs_ = [h for (ll, h) in state['abl'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre

    def make_cap_pre(l):
        def pre(module, args):
            if cap['on']:
                cap['oproj'][l] = args[0].detach()
            return None
        return pre

    def make_fin_hook():
        def hook(module, args, output):
            cap_hfin['h'] = output.detach()
            return None
        return hook

    handles.append(model.model.layers[35].register_forward_hook(
        make_bsub_hook()))
    handles.append(model.model.layers[35].register_forward_hook(
        make_fin_hook()))
    for l in M_LAYERS:
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_abl_pre(l)))
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_cap_pre(l)))

    def fwd(ids):
        with torch.inference_mode():
            return model(torch.tensor([ids], device=device))

    def arg_of(o):
        return int(o.logits[0, -1].float().argmax())

    def h_final(ids):
        cap_hfin['h'] = None
        with torch.inference_mode():
            model(torch.tensor([ids], device=device))
        return cap_hfin['h'][0, -1].float().cpu().numpy().copy()

    ids_full, ids_wo, tgt_n, multi = [], [], [], []
    for (pre, span, suf, ans) in ITEMS:
        ids_f = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_w = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_full.append(ids_f)
        ids_wo.append(ids_w)
        tt = tok(ans, add_special_tokens=False)['input_ids']
        multi.append(len(tt) > 1)
        tgt_n.append(tt[0])
    tgt_arr = np.array(tgt_n, dtype=np.int64)

    arg_nat = np.array([arg_of(fwd(ids_full[i]))
                        for i in range(len(ITEMS))])
    wrong_mask = (arg_nat != tgt_arr) & ~np.array(multi)
    wrong_idx = [int(i) for i in np.where(wrong_mask)[0]]
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    n_wrong, n_correct = len(wrong_idx), len(correct_idx)
    assert n_wrong == 35 and n_correct == 13, (n_wrong, n_correct)
    print('P2783 G1_OK wrong=%d correct=%d' % (n_wrong, n_correct),
          flush=True)

    v_rows = {}
    for i in wrong_idx + correct_idx:
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        nrm = float(np.linalg.norm(v))
        assert nrm > 0
        v_rows[i] = v / nrm
    print('P2783 G2_OK v_norms>0', flush=True)

    def top4_heads(ids, u):
        cap['on'] = True
        cap['oproj'].clear()
        try:
            fwd(ids)
        finally:
            cap['on'] = False
        gains = {}
        for l in M_LAYERS:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = W_O[l]
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        return [(l, h) for (l, h), g in
                sorted(gains.items(), key=lambda x: -x[1])
                if g > 0][:TOP_HEADS]

    prep = {}
    for i in wrong_idx + correct_idx:
        zi = fwd(ids_full[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_arr[i]] = -1e30
        riv = int(tmp.argmax())
        u_gain = (W_U[riv] - W_U[tgt_arr[i]]) / \
            np.linalg.norm(W_U[riv] - W_U[tgt_arr[i]])
        prep[i] = {'rival': riv,
                   'heads': top4_heads(ids_full[i], u_gain)}

    def run(tag, use_abl, bsub_alpha, rows, repair_mode):
        n_ok = 0
        detail = []
        for i in rows:
            state['abl'] = frozenset(prep[i]['heads']) if use_abl else \
                frozenset()
            state['bsub'] = (bsub_alpha, torch.tensor(
                v_rows[i].astype(np.float32), device=device)) \
                if bsub_alpha is not None else None
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['abl'] = frozenset()
                state['bsub'] = None
            if repair_mode:
                ok = bool(m == tgt_arr[i])
            else:
                ok = bool(m != arg_nat[i])
            n_ok += int(ok)
            detail.append({'row': i, 'ok': ok})
        print('P2783 %s n=%d ok=%d' % (tag, len(rows), n_ok), flush=True)
        return {'repair' if repair_mode else 'collateral': int(n_ok),
                'n': len(rows), 'detail': detail}

    table = {}
    # A0 heads-only (no bsub)
    table['A0_heads_only'] = {
        'repair': run('A0_heads_only_repair', True, None, wrong_idx,
                      True),
        'collateral': run('A0_heads_only_coll', True, None, correct_idx,
                          False)}
    # A1 heads+bsub bridging
    for a in (0.05, 0.1):
        table['A1_heads_bsub_%.2f' % a] = {
            'repair': run('A1_hb%.2f_repair' % a, True, a, wrong_idx,
                          True),
            'collateral': run('A1_hb%.2f_coll' % a, True, a,
                              correct_idx, False)}
    # A2 plain bsub (floor reference on wrong rows)
    for a in (0.05, 0.1):
        table['A2_bsub_%.2f' % a] = {
            'repair': run('A2_b%.2f_repair' % a, False, a, wrong_idx,
                          True)}

    mitigated = [(t, c['repair']['repair'], c['collateral']['collateral'])
                 for t, c in table.items() if 'collateral' in c
                 and c['repair']['repair'] >= 7
                 and c['collateral']['collateral'] <= 2]
    verdict = {'mitigated': bool(mitigated), 'mitigating_arms': mitigated,
               'table': {t: {'repair': c['repair']['repair'],
                             'collateral': c.get('collateral', {})
                             .get('collateral')}
                         for t, c in table.items()}}
    result = {'phase': 2783, 'prereg': PREREG, 'verdict': verdict,
              'per_row': {t: {'repair_detail': c['repair']['detail'],
                              'coll_detail':
                              c.get('collateral', {}).get('detail')}
                          for t, c in table.items()}}
    fc.save(OUT / 'result.json', result)
    for h in handles:
        h.remove()
    print('P2783 VERDICT %s' % json.dumps(verdict['table']),
          flush=True)
    print('P2783 MITIGATED %s %s' % (verdict['mitigated'],
                                     verdict['mitigating_arms']),
          flush=True)


if __name__ == '__main__':
    main()
