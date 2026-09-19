"""Phase 2836 (LPF main-line, 2836 candidates a+b).

(a) Dual-channel layerwise GAIN decomposition: the separated specific
channel  inc_spec(l) = inc_same(l) - 0.5*(inc_func(l)+inc_null(l))  is
decomposed per layer into the attention increment and the MLP increment
(exact residual identity  h(l+1)-h(l) = attn(l) + mlp(l) ).

Prereg (frozen before any readout; Gen2 — Gen1 diagnosed that
hidden_states[36] is post-final-RMS-norm, so dsf is anchored on the
raw final state hs[35]+attn[35]+mlp[35]; layer profile uses the
direction-resolved cls share because raw norms are inflated by
massive-activation increments of random null tokens at L6):
  A1  gain_decomposition_closes iff mean over words of
      ||recon - delta_spec_raw|| / ||delta_spec_raw|| < 0.02
  A2  specific_gain_layer_resolved iff argmax_l mean cls_spec_inc(l)
      >= 20  (cls_spec_inc(l) = |inc_spec(l) @ cdir| / ||inc_spec(l)||)
  A3  carrier_split_reported (descriptor): f_attn = sum_l ||attn_spec(l)||
      / (sum_l ||attn_spec(l)|| + sum_l ||mlp_spec(l)||)
  verdict: dual_channel_gain_resolved iff A1 AND A2

(b) Induction-copy step vs attribute-write heads: greedy decode of
"The apple is a fruit. The" (entity-repetition probe). At the forward
that emits the copied entity token, decompose the final-layer (L35)
per-head outputs onto the ' apple' unembed direction and test whether
the 2824/2833 attribute-write head cluster {h20,h22,h23,h26} is engaged.

  B1  induction_copy_present iff some generated token decodes to 'apple'
      within the first 3 steps
  B2  write_heads_engaged iff >= 3 of {20,22,23,26} rank in the top 16
      of 32 heads by |c_apple head share| at the induction step
  verdict: induction_via_write_heads iff B1 AND B2
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
OUT = BASE / 'phase2836' / 'dual_gain_induction'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2836
L_FINAL = 35
WRITE_HEADS = [20, 22, 23, 26]

PREREG = {
    'A1': 'gain_decomposition_closes iff mean over words of '
          '||recon-delta_spec_raw|| / ||delta_spec_raw|| < 0.02 '
          '(delta_spec_raw anchored on raw final state '
          'hs[35]+attn[35]+mlp[35]; hidden_states[36] is post-final-norm)',
    'A2': 'specific_gain_layer_resolved iff argmax_l mean cls_spec_inc(l) '
          '>= 20, cls_spec_inc(l) = |inc_spec(l) @ cdir| / ||inc_spec(l)||',
    'A3': 'carrier_split_reported: f_attn = sum||attn_spec|| / '
          '(sum||attn_spec||+sum||mlp_spec||)',
    'verdict_a': 'dual_channel_gain_resolved iff A1 AND A2',
    'B1': 'induction_copy_present iff some generated token decodes to '
          "'apple' within first 3 steps",
    'B2': 'write_heads_engaged iff >=3 of {20,22,23,26} in top16 of 32 '
          'heads by |c_apple share| at the forward that emits the copy '
          'token (input prefix = cur[:len(ids)+induction_step])',
    'verdict_b': 'induction_via_write_heads iff B1 AND B2',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    DOMAIN_OF = exec2806['domain_of']

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
    print('P2836 targets: %d words single_tok %d'
          % (len(target_list), len(single_tok)), flush=True)

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    nat_idx = [CAT_WORDS.index(c) for c in exec2806['nat_dom']]
    art_idx = [CAT_WORDS.index(c) for c in exec2806['art_dom']]
    dom_dir = Cm[nat_idx].mean(0) - Cm[art_idx].mean(0)
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])
    dom_unit = unit(dom_dir)

    rng = np.random.default_rng(SEED)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = []
    while len(null_tids) < len(target_list):
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)

    func_tid = None
    for fw in ['the', 'a', 'an', 'this', 'that']:
        fids = tok(' ' + fw, add_special_tokens=False)['input_ids']
        if len(fids) == 1:
            func_tid = int(fids[0])
            break
    assert func_tid is not None

    # ---------- residual hooks: per-layer attn / mlp outputs ----------
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

    def get_cond(tokens, pos):
        for li in cap['attn']:
            del cap['attn'][li][:]
        for li in cap['mlp']:
            del cap['mlp'][li][:]
        with torch.no_grad():
            ids = torch.tensor([tokens], device='cuda')
            out = model(ids, output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        return hs, attn, mlp

    all_hs = {}
    n_layers = None
    for i, (cat, w) in enumerate(target_list):
        w_tid = tid(w)
        hs_iso, _, _ = get_cond([w_tid], 0)
        if n_layers is None:
            n_layers = hs_iso.shape[0]
            print('P2836 n_layers=%d dim=%d' % (n_layers, hs_iso.shape[1]),
                  flush=True)
        all_hs[w] = {'cat': cat, 'iso': hs_iso, 'conds': {}}

        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        m = same_cat[0]
        all_hs[w]['conds']['same'] = {
            'mod': m, **dict(zip(['hs', 'attn', 'mlp'],
                                 get_cond([tid(m), w_tid], 1)))}

        dom = DOMAIN_OF[cat]
        diff_cats = [c for c in CAT_WORDS if DOMAIN_OF[c] != dom]
        diff_words = [x for x in CATS[diff_cats[0]] if x in single_tok]
        m = diff_words[0]
        all_hs[w]['conds']['diff'] = {
            'mod': m, **dict(zip(['hs', 'attn', 'mlp'],
                                 get_cond([tid(m), w_tid], 1)))}

        all_hs[w]['conds']['func'] = {
            'mod': 'the', **dict(zip(['hs', 'attn', 'mlp'],
                                     get_cond([func_tid, w_tid], 1)))}

        nt = null_tids[i]
        all_hs[w]['conds']['null'] = {
            'mod_tid': nt, **dict(zip(['hs', 'attn', 'mlp'],
                                      get_cond([nt, w_tid], 1)))}
        print('P2836 arm-a [%d/%d] %s (%s)'
              % (i + 1, len(target_list), w, cat), flush=True)

    for h in handles:
        h.remove()

    # ---------- arm (a): layerwise gain decomposition ----------
    F = n_layers - 1
    LAST = n_layers - 2  # last raw layer index; hs[F] is post-final-norm
    a1_rels = []
    spec_gain_words = []
    gen_gain_words = []
    attn_spec_words = []
    mlp_spec_words = []
    cls_spec_inc_words = []
    proj_spec_words = []
    for cat, w in target_list:
        c = all_hs[w]['conds']
        cdir = dW_unit[CAT_WORDS.index(cat)]
        inc_spec_attn = (c['same']['attn']
                         - 0.5 * (c['func']['attn'] + c['null']['attn']))
        inc_spec_mlp = (c['same']['mlp']
                        - 0.5 * (c['func']['mlp'] + c['null']['mlp']))
        inc_spec = inc_spec_attn + inc_spec_mlp
        recon = inc_spec.sum(0)
        raw_fin = {k: c[k]['hs'][LAST] + c[k]['attn'][LAST]
                   + c[k]['mlp'][LAST] for k in ('same', 'func', 'null')}
        dsf = ((raw_fin['same'] - c['same']['hs'][0])
               - 0.5 * ((raw_fin['func'] - c['func']['hs'][0])
                        + (raw_fin['null'] - c['null']['hs'][0])))
        a1_rels.append(float(np.linalg.norm(recon - dsf)
                             / max(np.linalg.norm(dsf), 1e-30)))
        spec_gain_words.append(np.linalg.norm(inc_spec, axis=1))
        gen_gain_words.append(np.linalg.norm(
            0.5 * ((c['func']['attn'] + c['func']['mlp'])
                   + (c['null']['attn'] + c['null']['mlp'])), axis=1))
        attn_spec_words.append(np.linalg.norm(inc_spec_attn, axis=1))
        mlp_spec_words.append(np.linalg.norm(inc_spec_mlp, axis=1))
        proj_spec_words.append(np.abs(inc_spec @ cdir))
        cls_spec_inc_words.append(
            np.abs(inc_spec @ cdir)
            / np.maximum(np.linalg.norm(inc_spec, axis=1), 1e-30))

    spec_gain = np.mean(spec_gain_words, axis=0)
    gen_gain = np.mean(gen_gain_words, axis=0)
    attn_spec = np.mean(attn_spec_words, axis=0)
    mlp_spec = np.mean(mlp_spec_words, axis=0)
    cls_spec_inc = np.mean(cls_spec_inc_words, axis=0)
    proj_spec = np.mean(proj_spec_words, axis=0)
    a1 = float(np.mean(a1_rels))
    a2_peak = int(np.argmax(cls_spec_inc))
    f_attn = float(attn_spec.sum()
                   / max(attn_spec.sum() + mlp_spec.sum(), 1e-30))

    v = {
        'A1_rel_closure_mean': round(a1, 6),
        'A2_peak_layer_cls_spec_inc': a2_peak,
        'A2_peak_cls_val': round(float(cls_spec_inc.max()), 4),
        'A3_f_attn': round(f_attn, 4),
        'A1_gain_decomposition_closes': bool(a1 < 0.02),
        'A2_specific_gain_layer_resolved': bool(a2_peak >= 20),
        'spec_gain_peak_layer': int(np.argmax(spec_gain)),
        'spec_gain_peak_val': round(float(spec_gain.max()), 4),
        'gen_gain_peak_layer': int(np.argmax(gen_gain)),
        'proj_spec_peak_layer': int(np.argmax(proj_spec)),
        'corr_spec_gen_gain': round(float(np.corrcoef(spec_gain,
                                                      gen_gain)[0, 1]), 4),
    }
    v['final_verdict_a'] = ('dual_channel_gain_resolved'
                            if v['A1_gain_decomposition_closes']
                            and v['A2_specific_gain_layer_resolved']
                            else 'not_resolved')

    # ---------- arm (b): induction copy vs attribute-write heads ----------
    W_o = model.model.layers[L_FINAL].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    d_model = W_o.shape[0]
    n_heads, hd = 32, W_o.shape[1] // 32
    W3 = W_o.reshape(d_model, n_heads, hd)
    v_apple = W_U[tid('apple')].astype(np.float64)
    v_fruit = W_U[tid('fruit')].astype(np.float64)
    v_food = W_U[tid('food')].astype(np.float64)

    o_store = {}

    def oproj_hook(module, args):
        o_store['a'] = args[0].detach()[0, -1, :].float().cpu().numpy()

    hnd = model.model.layers[L_FINAL].self_attn.o_proj \
        .register_forward_pre_hook(oproj_hook)

    def head_shares(a, vdir):
        a3 = a.astype(np.float64).reshape(n_heads, hd)
        dh = np.einsum('dkh,kh->dk', W3, a3)
        return dh.T @ vdir / max(float(np.linalg.norm(dh.sum(1))), 1e-30)

    prompt = 'The apple is a fruit. The'
    ids = tok(prompt, add_special_tokens=False)['input_ids']
    steps = []
    cur = list(ids)
    induction_step = None
    for s in range(8):
        o_store['a'] = None
        with torch.no_grad():
            t = torch.tensor([cur], device='cuda')
            logits = model(t).logits[0, -1].float().cpu().numpy()
        a = o_store['a']
        nid = int(np.argmax(logits))
        txt = tok.decode([nid]).strip()
        steps.append({'step': s, 'token': txt,
                      'c_apple_top10': [
                          {'head': int(i), 'share': round(float(x), 4)}
                          for i, x in sorted(
                              enumerate(head_shares(a, v_apple)),
                              key=lambda kv: -abs(kv[1]))[:10]]})
        if txt == 'apple' and induction_step is None:
            induction_step = s
        cur.append(nid)
    hnd.remove()

    b1 = induction_step is not None and induction_step <= 2
    b2 = False
    b2_detail = {}
    # re-run the exact induction-step forward to re-capture the head
    # input (o_store only retains the final greedy forward otherwise).
    if induction_step is not None:
        hnd = model.model.layers[L_FINAL].self_attn.o_proj \
            .register_forward_pre_hook(oproj_hook)
        prefix = cur[:len(ids) + induction_step]
        o_store['a'] = None
        with torch.no_grad():
            model(torch.tensor([prefix], device='cuda'))
        hnd.remove()
        sh_apple = head_shares(o_store['a'], v_apple)
        sh_fruit = head_shares(o_store['a'], v_fruit)
        sh_food = head_shares(o_store['a'], v_food)
        order = np.argsort(-np.abs(sh_apple))
        rank = {int(h): int(np.where(order == h)[0][0]) + 1
                for h in WRITE_HEADS}
        n_in_top16 = sum(1 for r in rank.values() if r <= 16)
        b2 = n_in_top16 >= 3
        b2_detail = {'write_head_ranks': rank,
                     'n_in_top16': n_in_top16,
                     'top10_heads': [
                         {'head': 'L%d h%d' % (L_FINAL, int(i)),
                          'c_apple': round(float(sh_apple[i]), 4)}
                         for i in order[:10]],
                     'c_fruit_top5': [
                         {'head': 'L%d h%d' % (L_FINAL, int(i)),
                          'share': round(float(sh_fruit[i]), 4)}
                         for i in np.argsort(-np.abs(sh_fruit))[:5]],
                     'c_food_top5': [
                         {'head': 'L%d h%d' % (L_FINAL, int(i)),
                          'share': round(float(sh_food[i]), 4)}
                         for i in np.argsort(-np.abs(sh_food))[:5]]}

    v['B1_induction_copy_present'] = bool(b1)
    v['B2_write_heads_engaged'] = bool(b2)
    v['induction_step'] = induction_step
    v['generated'] = [s['token'] for s in steps]
    v['final_verdict_b'] = ('induction_via_write_heads'
                            if b1 and b2 else 'not_confirmed')

    result = {'phase': 2836, 'prereg': PREREG, 'verdict': v,
              'decode_steps': steps, 'b2_detail': b2_detail}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'gain_layers.npz',
           spec_gain_layer_mean=spec_gain.astype(np.float32),
           gen_gain_layer_mean=gen_gain.astype(np.float32),
           attn_spec_layer_mean=attn_spec.astype(np.float32),
           mlp_spec_layer_mean=mlp_spec.astype(np.float32),
           cls_spec_inc_layer_mean=cls_spec_inc.astype(np.float32),
           proj_spec_layer_mean=proj_spec.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2836', elapsed)
    print('P2836 VERDICT %s' % json.dumps(v), flush=True)
    print('P2836 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
