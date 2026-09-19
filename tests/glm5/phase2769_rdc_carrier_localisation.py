"""Phase 2769 (Alpha): localise the CARRIER of the kc non-bias readout
competition.  Companion: phase2769_rdc_attachment_audit.py (C001, zero GPU).

Background.  2763: bsub (question-fact-span bias suppression) repairs 21/65
native-wrong rows but kc is immune (2/14).  2764 (bug-corrected): kc answers
surface only at L33+ and are overtaken at the end.  Open question (V3/2764
plan a-b): WHAT carries the rival signal that overtakes the internally-known
answer in the last layers?

Design (frozen before any intervention forward):
  Rows: the 320 controlled_relation diagnostic rows (2761/2763 ordering).
  C002 observation (all 64 kc rows):
    - final-position rival token = argmax(logit) excluding target.
    - rival direction u = unit(W_U[rival] - W_U[target]).
    - per-head attention writeback at modules M = {31..35} (hidden 32-36):
      contrib_{l,h} = W_O_l[:, h_slice] @ oproj_in_l[-1, h_slice]
      rival_gain_{l,h} = u . contrib_{l,h}.
    - carrier heads per wrong row = top-4 (l,h) by rival_gain > 0.
    - per-source attribution for carrier heads:
      contrib_s = A_l[h,-1,s] * (W_O_l[:,h_slice] @ v_s,h),
      source classes: fact-span (2759 rule) / content / other.
  C003 causal:
    - targeted ablation: zero o_proj input at the final position for the 4
      carrier heads jointly, per wrong row -> flip?
    - random control: 100 draws of 4 random (l in M, h) per row (seeded).
    - collateral: each identified carrier (l,h) applied to the 50 kc-correct
      rows -> break rate.
  C004 layered bsub: v_row from 2763 bias_dirs.npz hooked at module layers
    29..34, alpha=0.3, all 65 wrong rows; layer-35 point from 2763 native.

Preregistered criteria (frozen):
  E1: median fact+content share of positive rival mass at carrier heads over
      wrong rows >= 0.6.
  E2: targeted flips >= 4/14 AND targeted total > q95 of the random null.
  E3: collateral break rate <= 0.25.
  E4: descriptive layered bsub curve.
  verdict: carrier_localised iff E2 and E3.
Integrity gates:
  G1 base forward determinism (first 8 rows, exact).
  G3 native wrong per family == 2761 record.
  G4 fresh lens recompute on kc rows: wrong set == 2761 kc wrong set and
     corrected knows count == 4.
Status: descriptive + preregistered predictions; NOT mechanism closure.
"""
import time

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747
import phase2759_rdc_question_span_deletion as p2759

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2769' / 'qwen4_carrier_localisation'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
N_CTRL_DRAWS = 100
SEED_CTRL = 2769002
ALPHA_BSUB = 0.3
BSUB_LAYERS = [29, 30, 31, 32, 33, 34]
G3_EXPECTED = {'attribute_binding': 0, 'knowledge_chain': 14,
               'long_distance_role': 10, 'negation_scope': 16,
               'word_sense': 25}

PREREG = {
    'phase': 2769,
    'question': 'What carries the kc non-bias readout competition that '
                'overtakes the internally-known answer in the last layers?',
    'carrier_definition': 'heads (l in 31..35, h) whose attention writeback '
                          'at the final position has positive projection on '
                          'u = unit(W_U[rival] - W_U[target])',
    'causal_lever': 'zero o_proj input slice of carrier heads at the final '
                    'query position (head-level writeback ablation)',
    'criteria': {
        'E1': 'median fact+content share of positive rival mass at carrier '
              'heads over wrong rows >= 0.6',
        'E2': 'targeted flips >= 4/14 AND targeted total > q95 of 100-draw '
              'random-head null',
        'E3': 'collateral break rate on 50 kc-correct rows <= 0.25',
        'E4': 'descriptive layered bsub curve'},
    'verdict': 'carrier_localised iff E2 and E3',
    'gates': {'G1': 'base determinism first 8 rows exact',
              'G3': 'family wrong counts == 2761 record',
              'G4': 'fresh lens recompute on kc rows reproduces corrected '
                    'knows == 4'},
    'frozen_before_any_intervention_forward': True,
}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    import torch
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx_61 = z61['wrong_idx']
    tgt61 = z61['target']
    assert (tgt61[wrong_idx_61] == tgt_ids[wrong_idx_61]).all()
    kc_wrong = sorted(int(i) for i in wrong_idx_61
                      if fam_arr[i] == 'knowledge_chain')
    assert len(kc_wrong) == 14
    kc_correct = [i for i in range(len(rows))
                  if fam_arr[i] == 'knowledge_chain'
                  and i not in set(kc_wrong)]
    assert len(kc_correct) == 50
    kc_all = kc_correct + kc_wrong

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    assert model.config.num_hidden_layers == 36
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    group = n_heads // n_kv
    W_U = model.lm_head.weight.detach().float()

    # ---------- hooks -----------------------------------------------------
    cap = {}
    capture_handles = []

    def oproj_pre(l):
        def hook(module, args):
            cap.setdefault('oproj', {})[l] = args[0].detach()
        return hook

    def vproj_post(l):
        def hook(module, args, output):
            cap.setdefault('v', {})[l] = output.detach()
        return hook

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

    for l in M_LAYERS:
        capture_handles.append(
            model.model.layers[l].self_attn.o_proj.
            register_forward_pre_hook(oproj_pre(l)))
        capture_handles.append(
            model.model.layers[l].self_attn.v_proj.
            register_forward_hook(vproj_post(l)))
        capture_handles.append(
            model.model.layers[l].self_attn.o_proj.
            register_forward_pre_hook(abl_pre(l)))

    def base_forward(ids, want_att=False):
        with torch.inference_mode():
            t = torch.tensor([ids], device=device)
            o = model(t, output_attentions=want_att,
                      output_hidden_states=True)
            logits = o.logits[0, -1].float()
            h_last = o.hidden_states[-1][0, -1].float()
            att = o.attentions if want_att else None
        return logits, h_last, att

    # ---------- Pass A: base + gates --------------------------------------
    arg_native = np.empty(len(rows), dtype=np.int64)
    h_last_store = np.empty((len(rows), model.config.hidden_size),
                            dtype=np.float32)
    for i, ids in enumerate(id_list):
        logits, h_last, _ = base_forward(ids)
        arg_native[i] = int(logits.argmax())
        h_last_store[i] = h_last.cpu().numpy()
    g1_diff = 0.0
    for i in range(8):
        logits, h_last, _ = base_forward(id_list[i])
        g1_diff = max(g1_diff, float(np.abs(
            h_last.cpu().numpy() - h_last_store[i]).max()))
    assert g1_diff == 0.0, ('G1', g1_diff)

    wrong_native = arg_native != tgt_ids
    fam_wrong = {f: int(wrong_native[fam_arr == f].sum()) for f in G3_EXPECTED}
    assert fam_wrong == G3_EXPECTED, ('G3', fam_wrong)

    # G4: fresh lens recompute on kc rows (2761 recipe: final_norm(h) @ W.T,
    # W = lm_head.weight float)
    nrm = model.model.norm
    W_lm = model.lm_head.weight.detach().float()
    kc_lens_margin = np.empty((64, 37), dtype=np.float64)
    for k, i in enumerate(kc_all):
        with torch.inference_mode():
            t = torch.tensor([id_list[i]], device=device)
            o = model(t, output_hidden_states=True)
            for l in range(37):
                n = nrm(o.hidden_states[l][0, -1].float())
                zi = (n[None, :] @ W_lm.T)[0].float().cpu().numpy()
                tmp = zi.copy()
                tmp[tgt_ids[i]] = -1e30
                kc_lens_margin[k, l] = zi[tgt_ids[i]] - tmp.max()
    wrong_set_fresh = set(np.array(kc_all)[np.where(
        arg_native[np.array(kc_all)] != tgt_ids[np.array(kc_all)])[0]])
    assert wrong_set_fresh == set(kc_wrong), 'G4 wrong set mismatch'
    knows_corrected = int(sum(
        1 for k in range(64) if kc_all[k] in set(kc_wrong)
        and (kc_lens_margin[k, 8:36] >= 0).any()))
    assert knows_corrected == 4, ('G4 knows', knows_corrected)
    print('P2769 GATES_OK wrong=%s knows=%d' % (fam_wrong, knows_corrected),
          flush=True)

    # ---------- C002 observation ------------------------------------------
    rows_obs = []
    carrier = {}
    src_stats = {}
    for i in kc_all:
        cap.clear()
        logits, _, att = base_forward(id_list[i], want_att=True)
        zi = logits.cpu().numpy()
        tmp = zi.copy()
        tmp[tgt_ids[i]] = -1e30
        rival = int(tmp.argmax())
        u = (W_U[rival] - W_U[tgt_ids[i]]).detach()
        u = (u / u.norm()).cpu().numpy()
        is_wrong = i in set(kc_wrong)
        gains = {}
        for l in M_LAYERS:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = model.model.layers[l].self_attn.o_proj.weight.detach().\
                float().cpu().numpy()
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        entry = {'row': i, 'rival_id': rival,
                 'rival_tok': tok.decode([rival]).strip(),
                 'target_id': int(tgt_ids[i]),
                 'rival_gain_top': sorted(
                     [(l, h, g) for (l, h), g in gains.items()],
                     key=lambda x: -x[2])[:8]}
        ch = [(l, h) for (l, h), g in
              sorted(gains.items(), key=lambda x: -x[1]) if g > 0][:TOP_HEADS]
        if is_wrong:
            carrier[i] = ch
            _, fact, _, _ = p2759.fact_tokens(rows[i], tok)
            fact_set = set(int(x) for x in fact)
            ids = id_list[i]
            per_head_src = []
            for (l, h) in ch:
                A = att[l][0, h, -1].float().cpu().numpy()
                V = cap['v'][l][0].float().cpu().numpy()   # (seq, n_kv*hd)
                kv = h // group
                v_s = V[:, kv * hd:(kv + 1) * hd]          # (seq, hd)
                Wl = model.model.layers[l].self_attn.o_proj.weight.\
                    detach().float().cpu().numpy()
                o_s = (Wl[:, h * hd:(h + 1) * hd] @ v_s.T).T
                g_s = (o_s @ u) * A
                pos_mass = float(np.maximum(g_s, 0).sum())
                cls_mass = {'fact': 0.0, 'content': 0.0, 'other': 0.0}
                for s in range(len(ids)):
                    if g_s[s] <= 0:
                        continue
                    if s in fact_set:
                        c_ = 'fact'
                    else:
                        txt = tok.decode([ids[s]])
                        n_al = sum(ch2.isalnum() for ch2 in txt)
                        c_ = 'content' if n_al >= 2 else 'other'
                    cls_mass[c_] += float(g_s[s])
                per_head_src.append({
                    'layer': l, 'head': h, 'gain': gains[(l, h)],
                    'pos_mass': pos_mass,
                    'class_share': {k: (v / pos_mass if pos_mass > 0 else 0.0)
                                    for k, v in cls_mass.items()},
                    'top_src': sorted(
                        [dict(s=s, tok=tok.decode([ids[s]]).strip(),
                              gain=float(g_s[s]))
                         for s in range(len(ids))],
                        key=lambda x: -x['gain'])[:3]})
            src_stats[i] = per_head_src
        rows_obs.append(entry)
    fc_m = [float(np.mean([hh['class_share']['fact'] +
                           hh['class_share']['content']
                           for hh in src_stats[i]])) for i in kc_wrong]
    e1_median = float(np.median(fc_m))

    # ---------- C003 causal ------------------------------------------------
    def ablate_forward(i, pairs):
        abl_state['pairs'] = frozenset(pairs)
        try:
            with torch.inference_mode():
                t = torch.tensor([id_list[i]], device=device)
                o = model(t)
                a = int(o.logits[0, -1].float().argmax())
        finally:
            abl_state['pairs'] = frozenset()
        return a

    targeted = {}
    for i in kc_wrong:
        targeted[i] = bool(ablate_forward(i, carrier[i]) == tgt_ids[i])
    n_target = int(sum(targeted.values()))

    rng = np.random.default_rng(SEED_CTRL)
    null_totals = []
    for _ in range(N_CTRL_DRAWS):
        tot = 0
        for i in kc_wrong:
            ls = rng.choice(M_LAYERS, TOP_HEADS)
            hs_ = rng.integers(0, n_heads, TOP_HEADS)
            pairs = [(int(l), int(h)) for l, h in zip(ls, hs_)]
            tot += int(ablate_forward(i, pairs) == tgt_ids[i])
        null_totals.append(tot)
    null_totals = np.array(null_totals)
    q95 = float(np.quantile(null_totals, 0.95))
    e2_pass = bool(n_target >= 4 and n_target > q95)

    union_pairs = sorted({p for i in kc_wrong for p in carrier[i]})
    collateral_breaks = 0
    for (l, h) in union_pairs:
        for i in kc_correct:
            collateral_breaks += int(
                ablate_forward(i, [(l, h)]) != tgt_ids[i])
    n_collat = len(union_pairs) * len(kc_correct)
    e3_rate = collateral_breaks / max(n_collat, 1)
    e3_pass = bool(e3_rate <= 0.25)

    # ---------- C004 layered bsub ------------------------------------------
    for h_ in capture_handles:
        h_.remove()
    capture_handles = []
    z63 = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = z63['v_rows']
    hook_state = {'on': False, 'vec': None}

    def bsub_hook(module, args, output):
        if not hook_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - ALPHA_BSUB * h.norm() * hook_state['vec']
        return None

    bsub_handles = [model.model.layers[l].register_forward_hook(bsub_hook)
                    for l in BSUB_LAYERS]
    bsub_curve = {}
    for l in BSUB_LAYERS:
        flips = 0
        for i in wrong_idx_61:
            i = int(i)
            v = v_rows[i] / np.linalg.norm(v_rows[i])
            hook_state['on'] = True
            hook_state['vec'] = torch.tensor(v.astype(np.float32),
                                             device=device)
            try:
                with torch.inference_mode():
                    t = torch.tensor([id_list[i]], device=device)
                    o = model(t)
                    a = int(o.logits[0, -1].float().argmax())
            finally:
                hook_state['on'] = False
            flips += int(a == tgt_ids[i])
        bsub_curve[l] = flips
    for hd_ in bsub_handles:
        hd_.remove()
    z63b = np.load(P2763 / 'behaviour_scores.npz', allow_pickle=False)
    bsub_curve[35] = int(sum(int(z63b['native__bsub_a0.3_%d' % i]) ==
                             int(tgt_ids[i]) for i in wrong_idx_61))

    verdict = 'carrier_localised' if (e2_pass and e3_pass) else \
        'not_localised'
    deepknows_rows = [i for i in kc_wrong
                      if (kc_lens_margin[kc_all.index(i), 8:37].max() >= 0)]
    results = {
        'phase': 2769, 'G3_fam_wrong': fam_wrong,
        'G4_knows_corrected': knows_corrected,
        'C002': {'rows': rows_obs,
                 'carrier_heads': {str(i): [list(p) for p in carrier[i]]
                                   for i in kc_wrong},
                 'source_stats': {str(i): src_stats[i] for i in kc_wrong},
                 'E1_fact_content_share_median': e1_median,
                 'E1_pass': bool(e1_median >= 0.6)},
        'C003': {'targeted_flips': n_target,
                 'targeted_detail': {str(i): targeted[i] for i in kc_wrong},
                 'null_total_flips_mean': float(null_totals.mean()),
                 'null_q95': q95, 'E2_pass': e2_pass,
                 'collateral_breaks': collateral_breaks,
                 'collateral_n': n_collat, 'E3_rate': e3_rate,
                 'E3_pass': e3_pass,
                 'deepknows_rows': deepknows_rows,
                 'union_pairs': [[l, h] for (l, h) in union_pairs]},
        'C004': {'bsub_curve_module_layer': bsub_curve,
                 'alpha': ALPHA_BSUB},
        'verdict': verdict, 'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    fc.npz(OUT / 'carrier_stats.npz',
           kc_wrong=np.array(kc_wrong), kc_correct=np.array(kc_correct),
           lens_margin_kc=kc_lens_margin, null_totals=null_totals,
           targeted=np.array([targeted[i] for i in kc_wrong]))
    print('PHASE2769_DONE verdict=%s E1=%s E2=%s E3=%s targeted=%d/14 '
          'null_q95=%.0f collateral=%.3f bsub_curve=%s' %
          (verdict, results['C002']['E1_pass'], e2_pass, e3_pass, n_target,
           q95, e3_rate, bsub_curve), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
