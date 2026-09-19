"""Phase 2840 (LPF Delta-II/III): carrying-capacity census.

2839 established: 16 layers x top-4 heads independently carry 27.3%
of the cls_spec alignment rate; 73% unaccounted.  This phase runs an
ablation ladder (matched protocol, in-run) to locate the remainder:

  S4   top-4 per layer, L20-35     (2839 reproduction baseline)
  S8   top-8 per layer, L20-35
  S16  top-16 per layer, L20-35
  S32  all 32 heads, L20-35        (attention removed in mid/late)
  S36  all 32 heads, L0-35         (all attention removed)

Prereg (frozen before any readout):
  P1  monotone_ladder iff drop(S8) > drop(S4)
  P2  half_heads_substantial iff drop(S16) > 0.20
  P3  attn_midlate_carries_majority iff drop(S32) > 0.40
  P4  attention_backbone iff drop(S36) > 0.60
  verdict: attention_backbone iff P3 and P4;
           distributed_readout iff (not P3) and P2;
           mixed_capacity otherwise
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2840' / 'carrying_capacity'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2838_RESULT = BASE / 'phase2838' / 'decisive_layer_neuron' / 'result.json'
SEED = 2840
SCAN_LAYERS = list(range(20, 36))
LAST = 35

PREREG = {
    'P1': 'monotone_ladder iff drop(S8) > drop(S4)',
    'P2': 'half_heads_substantial iff drop(S16) > 0.20',
    'P3': 'attn_midlate_carries_majority iff drop(S32) > 0.40',
    'P4': 'attention_backbone iff drop(S36) > 0.60',
    'verdict': 'attention_backbone iff P3 and P4; distributed_readout '
               'iff (not P3) and P2; mixed_capacity otherwise',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    res2838 = json.loads(SRC_2838_RESULT.read_text(encoding='utf-8'))
    top4_2838 = {int(k): v for k, v in
                 res2838['top_heads_per_layer'].items()}
    # top-8+ sets are recomputed from THIS run's share map (pass below);
    # 2838 registered top-4 is used only for the agreement cross-check.

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED}
    fc.save(OUT / 'execution.json', execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    # ---------- ablation control (any layer set, any head list) ----------
    abl = {'groups': None}

    def make_abl_hook(li):
        def pre_hook(module, args):
            g = abl['groups']
            if g is None or li not in g:
                return None
            a = args[0]
            a2 = a.clone()
            v = a2.view(*a2.shape[:-1], 32, a2.shape[-1] // 32)
            v[..., list(g[li]), :] = 0
            return (a2,) + tuple(args[1:])
        return pre_hook

    for li, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(make_abl_hook(li))

    cap = {'attn': {}, 'mlp': {}}

    def make_out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o[0].detach().float().cpu().numpy())
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            make_out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))

    def clear_cap():
        for d in ('attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]

    def forward_run(tokens, pos):
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        return hs, attn, mlp

    # ---------- targets ----------
    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    targets = {}
    for cat in CAT_WORDS:
        cs = [w for w in CATS[cat] if w in single_tok]
        targets[cat] = cs[:2]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])

    rng = np.random.default_rng(SEED)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = []
    while len(null_tids) < len(target_list):
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)
    func_tid = tid('the')

    def conds_for(i, cat, w):
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        return {'same': [tid(same_cat[0]), w_tid],
                'func': [func_tid, w_tid],
                'null': [null_tids[i], w_tid]}

    def measure_cls(i, cat, w):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        conds = conds_for(i, cat, w)
        hs_iso, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[0]
        ds = {}
        for cn, toks in conds.items():
            hs, attn, mlp = forward_run(toks, 1)
            raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
            ds[cn] = raw_fin - iso0
        d_spec = ds['same'] - 0.5 * (ds['func'] + ds['null'])
        n2 = float(np.linalg.norm(d_spec))
        return float(abs(d_spec @ cdir) / max(n2, 1e-30))

    cls_base_list = []
    for i, (cat, w) in enumerate(target_list):
        cls_base_list.append(measure_cls(i, cat, w))
        if (i + 1) % 5 == 0:
            print('P2840 base [%d/%d]' % (i + 1, len(target_list)),
                  flush=True)
    cls_base_m = float(np.mean(cls_base_list))

    # top-8 per layer: reuse 2838 top-4 + rank up via |share| from a
    # fresh share pass (cheap: reuse opj capture from base measurement
    # is not stored; do one capture pass per word for share map)
    W3 = {}
    for li in SCAN_LAYERS:
        Wo = model.model.layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        W3[li] = Wo.reshape(Wo.shape[0], 32, Wo.shape[1] // 32)

    # share map via dedicated capture hooks (o_proj inputs)
    cap['opj'] = {}
    cap['opj_on'] = False

    def make_opj_hook(li):
        def pre_hook(module, args):
            if cap['opj_on']:
                cap['opj'].setdefault(li, []).append(
                    args[0].detach()[0].float().cpu().numpy())
        return pre_hook

    opj_handles = []
    for li in SCAN_LAYERS:
        opj_handles.append(model.model.layers[li].self_attn.o_proj
                           .register_forward_pre_hook(make_opj_hook(li)))

    def forward_cap(tokens, pos):
        clear_cap()
        for li in cap.setdefault('opj', {}):
            del cap['opj'][li][:]
        cap['opj_on'] = True
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        cap['opj_on'] = False
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        opj = {li: cap['opj'][li][0][pos] for li in SCAN_LAYERS}
        return hs, attn, mlp, opj

    share_map = {li: [] for li in SCAN_LAYERS}
    for i, (cat, w) in enumerate(target_list):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        conds = conds_for(i, cat, w)
        hs_iso, _, _, opj_iso = forward_cap([w_tid], 0)
        iso0 = hs_iso[0]
        dh = {}
        for cn, toks in conds.items():
            hs, attn, mlp, opj = forward_cap(toks, 1)
            dh[cn] = {li: np.einsum('dkh,kh->dk', W3[li],
                                    opj[li].astype(np.float64)
                                    .reshape(32, 128))
                      for li in SCAN_LAYERS}
        dh_iso = {li: np.einsum('dkh,kh->dk', W3[li],
                                opj_iso[li].astype(np.float64)
                                .reshape(32, 128)) for li in SCAN_LAYERS}
        for li in SCAN_LAYERS:
            d_spec_h = ((dh['same'][li] - dh_iso[li]) - 0.5 * (
                (dh['func'][li] - dh_iso[li])
                + (dh['null'][li] - dh_iso[li])))
            n2 = float(np.linalg.norm(d_spec_h.sum(axis=1)))
            sh = d_spec_h.T @ cdir / max(n2, 1e-30)
            share_map[li].append(sh)
        if (i + 1) % 5 == 0:
            print('P2840 share [%d/%d]' % (i + 1, len(target_list)),
                  flush=True)

    share_mean = {li: np.mean(share_map[li], axis=0) for li in SCAN_LAYERS}
    order = {li: list(np.argsort(-np.abs(share_mean[li])))
             for li in SCAN_LAYERS}
    top_k_sets = {
        4: {li: order[li][:4] for li in SCAN_LAYERS},
        8: {li: order[li][:8] for li in SCAN_LAYERS},
        16: {li: order[li][:16] for li in SCAN_LAYERS},
        32: {li: list(range(32)) for li in SCAN_LAYERS},
        'all36': {li: list(range(32)) for li in range(36)},
    }
    # cross-check: top-4 from this run vs 2838 registered
    agree = {li: sorted(int(h) for h in top_k_sets[4][li])
             == sorted(int(h) for h in top4_2838.get(li, []))
             for li in SCAN_LAYERS}
    n_agree = sum(agree.values())

    def ladder_drop(groups):
        abl['groups'] = groups
        cls_vals = []
        for i, (cat, w) in enumerate(target_list):
            cls_vals.append(measure_cls(i, cat, w))
        abl['groups'] = None
        return 1.0 - float(np.mean(cls_vals)) / max(cls_base_m, 1e-30)

    drops = {}
    for key in [4, 8, 16, 32, 'all36']:
        drops[key] = ladder_drop(top_k_sets[key])
        print('P2840 ladder S%s drop=%.4f' % (key, drops[key]), flush=True)

    p1 = drops[8] > drops[4]
    p2 = drops[16] > 0.20
    p3 = drops[32] > 0.40
    p4 = drops['all36'] > 0.60
    if p3 and p4:
        verdict = 'attention_backbone'
    elif (not p3) and p2:
        verdict = 'distributed_readout'
    else:
        verdict = 'mixed_capacity'

    v = {
        'cls_base_mean': round(cls_base_m, 6),
        'top4_agree_with_2838': '%d/16' % n_agree,
        'drop_S4': round(float(drops[4]), 4),
        'drop_S8': round(float(drops[8]), 4),
        'drop_S16': round(float(drops[16]), 4),
        'drop_S32_midlate_allheads': round(float(drops[32]), 4),
        'drop_S36_all_attention': round(float(drops['all36']), 4),
        'P1_monotone_ladder': bool(p1),
        'P2_half_heads_substantial': bool(p2),
        'P3_attn_midlate_majority': bool(p3),
        'P4_attention_backbone': bool(p4),
        'final_verdict': verdict,
    }

    result = {'phase': 2840, 'prereg': PREREG, 'verdict': v,
              'top4_agreement': {str(li): bool(agree[li])
                                 for li in SCAN_LAYERS}}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'capacity_curve.npz',
           drop_curve=np.array([drops[4], drops[8], drops[16],
                                drops[32], drops['all36']],
                               dtype=np.float32),
           **{('share_L%d' % li): share_mean[li].astype(np.float32)
              for li in SCAN_LAYERS})

    elapsed = time.monotonic() - t0
    cc.ledger('phase2840', elapsed)
    print('P2840 VERDICT %s' % json.dumps(v), flush=True)
    print('P2840 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
