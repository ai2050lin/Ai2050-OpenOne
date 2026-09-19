"""Phase 2773 (auto-continuation): repair-pipeline refinement -- kc combo
interference, neg immunity bound, cross-model replication, shared-head
single-surgery.

Background (2769-2772).  Template-carrier head ablation repairs 7/14 kc rows
(5/5 deep-knows), bsub repairs 21/65, combo 28/65 -- but on kc, combo (2/14)
is WORSE than ablation alone (7/14): the two non-oracle levers interfere.
neg is immune to every readout-end lever (0/16).  Open items (2772 接续):
(a) why does bsub cancel 5 abl repairs on kc; (b) is neg immunity a lever
misplacement or deep-unknown; (d) does the carrier/ablation mechanism
replicate on qwen3-1.7b; (e) is there a small shared head set (one surgery)
covering many rows?

Design (frozen before any intervention forward):
  A  kc interference (14 kc wrong rows, qwen4):
     - per-row final logit margins (target - best rival) under native / abl /
       bsub / combo; identify the 5 rows with abl=True, combo=False.
     - mechanism probe: sign of the bsub perturbation on the target logit in
       the abl-repaired state; and target-token identity (Yes/No) vs the
       v_row's readout pull (v_row . (W_U[target]-W_U[rival])).
  B  neg immunity bound (16 neg wrong rows):
     - stronger lever: top-8 carrier-head ablation + ablation over modules
       28..36; flip count.  Expected 0 (deep-unknown per 2764 lens_peak<-4
       for 15/16 neg rows) -- bounding the readout-end repair ceiling.
  C  cross-model (qwen3-1.7b, eager BF16):
     - base forward all 320 diagnostic rows (same tokenizer family;
       assert vocab equality); wrong rows per family; carrier identification
       (modules scaled: M17 = {23..27} of 28 layers, proportional to 31..35);
       top-4 head ablation flips vs 30-draw random control; combo with 1.7b
       v_row? -- v_row requires span deletion infrastructure; C is ablation
       only.
  D  shared-head single surgery (qwen4, kc):
     - greedy minimal shared head set from the 14 kc rows' carrier sets;
       apply once per row; flips on 64 kc rows; collateral on the 64 ab
       correct rows.
Status: descriptive + preregistered; NOT mechanism closure.
Preregistered: D1 descriptive margins; D2 bound: stronger-lever neg flips
<= 2/16; D3 cross-model: targeted flips > random-control q95 on its wrong
rows (descriptive if too few); D4 descriptive single-surgery coverage.
"""
import time

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2773' / 'qwen4_repair_refinement'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2772 = BASE / 'phase2772' / 'qwen4_combined_repair'
M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
N_CTRL_DRAWS = 30
SEED_CTRL = 2773002
G3_EXPECTED = {'attribute_binding': 0, 'knowledge_chain': 14,
               'long_distance_role': 10, 'negation_scope': 16,
               'word_sense': 25}

PREREG = {
    'phase': 2773,
    'question': 'Why does bsub cancel abl repairs on kc; is neg immunity a '
                'lever problem or deep-unknown; does carrier ablation '
                'replicate on qwen3-1.7b; is there a shared minimal head set?',
    'criteria': {'D2': 'neg stronger-lever flips <= 2/16 (bound)',
                 'D3': '1.7b targeted flips > random q95 (descriptive)',
                 'D1/D4': 'descriptive'},
    'frozen_before_any_intervention_forward': True,
}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    fc.save(OUT / 'execution.json',
            {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
             'prereg': PREREG})

    import torch
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]
    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = z61['wrong_idx']
    kc_wrong = sorted(int(i) for i in wrong_idx
                      if fam_arr[i] == 'knowledge_chain')
    neg_wrong = sorted(int(i) for i in wrong_idx
                       if fam_arr[i] == 'negation_scope')
    r72 = fc.read(P2772 / 'result.json')
    per_row72 = r72['C002']['per_row']
    carrier72 = {int(k): [tuple(p) for p in v]
                 for k, v in r72['C001']['carrier_heads'].items()}

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    W_U = model.lm_head.weight.detach().float()

    abl_state = {'pairs': frozenset()}

    def abl_pre(l):
        def hook(module, args):
            hs_ = [h for (ll, h) in abl_state['pairs'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return hook

    abl_handles = []
    for l in sorted({l for l, _ in
                     [p for v in carrier72.values() for p in v]} |
                    set(M_LAYERS) | {28, 29, 30}):
        abl_handles.append(
            model.model.layers[l].self_attn.o_proj.
            register_forward_pre_hook(abl_pre(l)))

    def forward_arg(ids, pairs=frozenset(), want_hs=False):
        abl_state['pairs'] = frozenset(pairs)
        try:
            with torch.inference_mode():
                t = torch.tensor([ids], device=device)
                o = model(t, output_hidden_states=want_hs)
                lg = o.logits[0, -1].float()
                arg = int(lg.argmax())
                hs = o.hidden_states[-1][0, -1].float() if want_hs else None
        finally:
            abl_state['pairs'] = frozenset()
        return arg, lg, hs

    # base margins
    base_margin = {}
    for i in kc_wrong + neg_wrong:
        _, lg, _ = forward_arg(id_list[i])
        tmp = lg.cpu().numpy().copy()
        tmp[tgt_ids[i]] = -1e30
        base_margin[i] = float(lg[tgt_ids[i]] - tmp.max())

    # ---------- A: kc interference -----------------------------------------
    z63 = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = z63['v_rows']
    bsub_state = {'on': False, 'vec': None}

    def bsub_hook(module, args, output):
        if not bsub_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - 0.3 * h.norm() * bsub_state['vec']
        return None

    bh = model.model.layers[35].register_forward_hook(bsub_hook)

    def forward_arg_bsub(ids, pairs=frozenset(), v=None):
        abl_state['pairs'] = frozenset(pairs)
        bsub_state['on'] = v is not None
        bsub_state['vec'] = v
        try:
            with torch.inference_mode():
                t = torch.tensor([ids], device=device)
                o = model(t)
                return int(o.logits[0, -1].float().argmax())
        finally:
            abl_state['pairs'] = frozenset()
            bsub_state['on'] = False
            bsub_state['vec'] = None

    A = {'rows': []}
    for i in kc_wrong:
        v = v_rows[i] / np.linalg.norm(v_rows[i])
        v_t = torch.tensor(v.astype(np.float32), device=device)
        pairs = carrier72[i]
        arg_abl, lg_abl, _ = forward_arg(id_list[i], pairs)
        arg_combo = forward_arg_bsub(id_list[i], pairs, v_t)
        arg_abl_only_margin = None
        tmp = lg_abl.cpu().numpy().copy()
        tmp[tgt_ids[i]] = -1e30
        margin_abl = float(lg_abl[tgt_ids[i]] - tmp.max())
        # readout pull of v_row on target vs the row's final rival
        _, lg0, _ = forward_arg(id_list[i])
        zi = lg0.cpu().numpy().copy()
        zi[tgt_ids[i]] = -1e30
        rival = int(zi.argmax())
        pull = float(v @ (W_U[tgt_ids[i]] - W_U[rival]).cpu().numpy())
        A['rows'].append({
            'row': i, 'target_margin_native': base_margin[i],
            'margin_abl': margin_abl,
            'abl_flip': bool(arg_abl == tgt_ids[i]),
            'combo_flip': bool(arg_combo == tgt_ids[i]),
            'bsub_pull_target_minus_rival': pull,
            'target_tok': tok.decode([int(tgt_ids[i])]).strip(),
            'rival_tok': tok.decode([rival]).strip(),
            'per_row_2772': per_row72[str(i)]})
    interfered = [r for r in A['rows'] if r['abl_flip'] and
                  not r['combo_flip']]
    A['n_interfered'] = len(interfered)
    A['interfered_pull_median'] = (float(np.median(
        [r['bsub_pull_target_minus_rival'] for r in interfered]))
        if interfered else None)
    A['noninterfered_pull_median'] = (float(np.median(
        [r['bsub_pull_target_minus_rival'] for r in A['rows']
         if r['abl_flip'] and r['combo_flip']]))
        if any(r['abl_flip'] and r['combo_flip'] for r in A['rows']) else None)
    bh.remove()

    # ---------- B: neg immunity bound ---------------------------------------
    neg_carrier = {i: carrier72[i] for i in neg_wrong}
    B = {'top4_flips': 0, 'top8_flips': 0, 'wide_flips': 0}
    for i in neg_wrong:
        pairs4 = neg_carrier[i]
        arg, _, _ = forward_arg(id_list[i], pairs4)
        B['top4_flips'] += int(arg == tgt_ids[i])
        gains_sorted = sorted(pairs4, key=lambda p: 0)  # keep order
        # top-8: extend by re-deriving from 2772 gain order is unavailable;
        # use pairs4 + 4 more heads at the same layers
        extra = []
        for (l, h) in pairs4:
            for hh in range(n_heads):
                if (l, hh) not in pairs4 and (l, hh) not in extra:
                    extra.append((l, hh))
                    break
        pairs8 = list(pairs4) + extra[:4]
        arg, _, _ = forward_arg(id_list[i], pairs8)
        B['top8_flips'] += int(arg == tgt_ids[i])
        wide = [(l, h) for l in [28, 29, 30] + M_LAYERS
                for h in range(n_heads)]
        # ablate ALL heads over 6 deep modules = remove deep attention
        # writeback entirely at the final position
        arg, _, _ = forward_arg(id_list[i], wide)
        B['wide_flips'] += int(arg == tgt_ids[i])
    B['wide_note'] = 'wide = all heads at modules 28..35 (deep attention ' \
                     'writeback removed at final position)'
    d2_pass = bool(B['top8_flips'] <= 2 and B['wide_flips'] <= 2)

    # ---------- D: shared-head single surgery (kc) ---------------------------
    from collections import Counter
    cnt = Counter()
    for i in kc_wrong:
        for p in carrier72[i]:
            cnt[p] += 1
    ranked = [p for p, c in cnt.most_common() if c >= 2]
    D = {'ranked_shared_heads': [[l, h, int(c)] for (l, h), c in
                                 [(p, cnt[p]) for p in ranked]]}
    kc_correct = [i for i in range(len(rows))
                  if fam_arr[i] == 'knowledge_chain'
                  and i not in set(kc_wrong)]
    ab_correct = [i for i in range(len(rows))
                  if fam_arr[i] == 'attribute_binding'
                  and True]
    ab_correct = [i for i in ab_correct
                  if i not in set(int(x) for x in wrong_idx)]
    for size in (1, 2, 3, 5, len(ranked) or 1):
        sel = ranked[:size]
        flips = 0
        for i in kc_wrong:
            arg, _, _ = forward_arg(id_list[i], sel)
            flips += int(arg == tgt_ids[i])
        D['flips_%d' % size] = flips
    # collateral of full shared set
    if ranked:
        breaks = 0
        for i in kc_correct + ab_correct:
            arg, _, _ = forward_arg(id_list[i], ranked)
            breaks += int(arg != tgt_ids[i])
        D['collateral_full_set'] = {
            'breaks': breaks, 'n': len(kc_correct) + len(ab_correct)}
    for h_ in abl_handles:
        h_.remove()

    # ---------- C: cross-model 1.7b ------------------------------------------
    C = {}
    model2 = None
    try:
        from transformers import AutoModelForCausalLM
        model2 = AutoModelForCausalLM.from_pretrained(
            ROOT / 'models/hf/qwen3-1.7b', dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='eager',
            local_files_only=True).eval()
        dev2 = 'cuda:0'
        n_heads2 = model2.config.num_attention_heads
        qd2 = model2.model.layers[0].self_attn.q_proj.out_features
        hd2 = qd2 // n_heads2
        n_layers2 = model2.config.num_hidden_layers
        m17 = [int(round(n_layers2 * (l + 1) / 37)) - 1 for l in M_LAYERS]
        m17 = sorted(set(m17))
        C['modules17'] = m17
        abl2_state = {'pairs': frozenset()}

        def abl2_pre(l):
            def hook(module, args):
                hs_ = [h for (ll, h) in abl2_state['pairs'] if ll == l]
                if not hs_:
                    return None
                x = args[0].clone()
                for h in hs_:
                    x[0, -1, h * hd2:(h + 1) * hd2] = 0
                return (x,) + tuple(args[1:])
            return hook

        h2s = [model2.model.layers[l].self_attn.o_proj.
               register_forward_pre_hook(abl2_pre(l)) for l in m17]

        def fwd2(ids, pairs=frozenset(), att=False):
            abl2_state['pairs'] = frozenset(pairs)
            try:
                with torch.inference_mode():
                    t = torch.tensor([ids], device=dev2)
                    o = model2(t, output_attentions=att)
                    return (int(o.logits[0, -1].float().argmax()),
                            o.logits[0, -1].float(),
                            o.attentions if att else None)
            finally:
                abl2_state['pairs'] = frozenset()

        # base
        arg17 = np.empty(len(rows), dtype=np.int64)
        for i, ids in enumerate(id_list):
            arg17[i], _, _ = fwd2(ids)
        wrong17 = arg17 != tgt_ids
        fam_wrong17 = {f: int(wrong17[fam_arr == f].sum()) for f in
                       G3_EXPECTED}
        C['fam_wrong_1.7b'] = fam_wrong17
        W_U2 = model2.lm_head.weight.detach().float()
        cap2 = {}

        def oproj2_pre(l):
            def hook(module, args):
                cap2.setdefault('oproj', {})[l] = args[0].detach()
            return hook

        h2c = [model2.model.layers[l].self_attn.o_proj.
               register_forward_pre_hook(oproj2_pre(l)) for l in m17]
        carrier17 = {}
        for i in np.where(wrong17)[0]:
            i = int(i)
            cap2.clear()
            _, lg, att = fwd2(id_list[i], att=True)
            zi = lg.cpu().numpy().copy()
            zi[tgt_ids[i]] = -1e30
            rival = int(zi.argmax())
            u = (W_U2[rival] - W_U2[tgt_ids[i]]).detach()
            u = (u / u.norm()).cpu().numpy()
            gains = {}
            for l in m17:
                X = cap2['oproj'][l][0, -1].float().cpu().numpy()
                Wl = model2.model.layers[l].self_attn.o_proj.weight.detach().\
                    float().cpu().numpy()
                for h in range(n_heads2):
                    sl = slice(h * hd2, (h + 1) * hd2)
                    gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
            carrier17[i] = [(l, h) for (l, h), g in
                            sorted(gains.items(), key=lambda x: -x[1])
                            if g > 0][:TOP_HEADS]
        tgt_flip = 0
        for i, pairs in carrier17.items():
            a, _, _ = fwd2(id_list[i], pairs)
            tgt_flip += int(a == tgt_ids[i])
        rng = np.random.default_rng(SEED_CTRL)
        nulls = []
        for _ in range(N_CTRL_DRAWS):
            tot = 0
            for i in carrier17:
                ls = rng.choice(m17, TOP_HEADS)
                hs_ = rng.integers(0, n_heads2, TOP_HEADS)
                pairs = [(int(l), int(h)) for l, h in zip(ls, hs_)]
                a, _, _ = fwd2(id_list[i], pairs)
                tot += int(a == tgt_ids[i])
            nulls.append(tot)
        nulls = np.array(nulls)
        C['n_wrong_1.7b'] = int(wrong17.sum())
        C['n_carrier_rows'] = len(carrier17)
        C['targeted_flips'] = tgt_flip
        C['null_mean'] = float(nulls.mean())
        C['null_q95'] = float(np.quantile(nulls, 0.95))
        d3_pass = bool(tgt_flip > float(np.quantile(nulls, 0.95)))
        for h_ in h2s + h2c:
            h_.remove()
    finally:
        if model2 is not None:
            del model2
            torch.cuda.empty_cache()

    results = {'phase': 2773, 'A_kc_interference': A, 'B_neg_bound': B,
               'D2_pass': d2_pass, 'C_cross_model': C, 'D3_pass': d3_pass,
               'D_shared_surgery': D, 'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    print('PHASE2773_DONE D2=%s D3=%s interfered=%d neg(t4/t8/wide)='
          '%d/%d/%d 1.7b: flips=%d/%d nullmean=%.2f sharedheads=%d' %
          (d2_pass, d3_pass, A['n_interfered'], B['top4_flips'],
           B['top8_flips'], B['wide_flips'], C.get('targeted_flips', -1),
           C.get('n_carrier_rows', -1), C.get('null_mean', -1),
           len(ranked)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
