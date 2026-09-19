"""Phase 2833 (LPF main-line continuation of 2810): Delta-h channel
separation.

2810 found that the RAW hidden-state delta (ctx - iso) is dominated by a
generic context channel ("the" moved it most; P-K2/P-K4 both false).
The 2810 continuation plan (a): separate the semantic channel as

    delta_specific(cond) = delta_cond - 0.5*(delta_func + delta_null)

per layer per word, then re-test the dose-response at the separated level.
Plan (d), exploratory: decompose the final-layer separated delta onto the
32 per-head output directions delta_h = W_O^h @ a_h to locate carrier
heads.

Prereg (frozen before any readout):
  S1  semantic_channel_selective iff
      class_share(specific_same) > class_share(specific_diff) at final layer
  S2  separation_gain iff
      class_share(specific_same) > class_share(raw_same) at final layer
  S3  dose_response_restored iff
      ||specific_same|| > ||specific_diff|| at final layer
  S4  layer_resolved iff range across layers of class_share(specific_same)
      > 0.05
  verdict: channel_separated_substantive iff S1 AND S2
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
OUT = BASE / 'phase2833' / 'channel_separation'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2833

PREREG = {
    'S1': 'semantic_channel_selective iff class_share(specific_same) > '
          'class_share(specific_diff) at final layer',
    'S2': 'separation_gain iff class_share(specific_same) > '
          'class_share(raw_same) at final layer',
    'S3': 'dose_response_restored iff ||specific_same|| > '
          '||specific_diff|| at final layer',
    'S4': 'layer_resolved iff range across layers of '
          'class_share(specific_same) > 0.05',
    'verdict': 'channel_separated_substantive iff S1 AND S2',
}


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
    print('P2833 targets: %d single_tok words: %d'
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

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-30)

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

    # final-layer o_proj input capture at target position (exploratory arm)
    oproj_store = {}
    hook_handle = []

    def make_hook(key):
        def hook(module, args):
            a = args[0].detach()[0, -1, :].float().cpu().numpy()
            oproj_store.setdefault(key, []).append(a)
        return hook

    L_FINAL = 35
    hnd = model.model.layers[L_FINAL].self_attn.o_proj.register_forward_pre_hook(
        make_hook('x'))
    hook_handle.append(hnd)

    def get_hs(tokens, target_pos, key=None):
        oproj_store['x'] = []
        with torch.no_grad():
            ids = torch.tensor([tokens], device='cuda')
            out = model(ids, output_hidden_states=True)
            hs = np.stack([h[0, target_pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        if key is not None:
            oproj_store[key] = list(oproj_store['x'])
        return hs

    all_hs = {}
    n_layers = None
    for i, (cat, w) in enumerate(target_list):
        w_tid = tid(w)
        hs_iso = get_hs([w_tid], 0)
        if n_layers is None:
            n_layers = hs_iso.shape[0]
            print('P2833 n_layers=%d dim=%d' % (n_layers, hs_iso.shape[1]),
                  flush=True)
        all_hs[w] = {'iso': hs_iso, 'cat': cat, 'dom': DOMAIN_OF[cat],
                     'conds': {}}

        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        m = same_cat[0]
        all_hs[w]['conds']['same'] = {
            'mod': m,
            'hs': get_hs([tid(m), w_tid], 1, key=('same', w))}

        dom = DOMAIN_OF[cat]
        diff_cats = [c for c in CAT_WORDS
                     if DOMAIN_OF[c] != dom]
        diff_words = [x for x in CATS[diff_cats[0]] if x in single_tok]
        m = diff_words[0]
        all_hs[w]['conds']['diff'] = {
            'mod': m,
            'hs': get_hs([tid(m), w_tid], 1, key=('diff', w))}

        all_hs[w]['conds']['func'] = {
            'mod': 'the',
            'hs': get_hs([func_tid, w_tid], 1, key=('func', w))}

        nt = null_tids[i]
        all_hs[w]['conds']['null'] = {
            'mod_tid': nt,
            'hs': get_hs([nt, w_tid], 1, key=('null', w))}
        print('P2833 [%d/%d] %s (%s)' % (i + 1, len(target_list), w, cat),
              flush=True)

    for h in hook_handle:
        h.remove()

    # ---------- channel separation ----------
    per_word = []
    for cat, w in target_list:
        hs_iso = all_hs[w]['iso']
        cat_idx = CAT_WORDS.index(cat)
        cdir = dW_unit[cat_idx]
        wr = {'word': w, 'category': cat, 'domain': DOMAIN_OF[cat],
              'conditions': {}}
        d_ctrl = 0.5 * ((all_hs[w]['conds']['func']['hs'] - hs_iso)
                        + (all_hs[w]['conds']['null']['hs'] - hs_iso))
        for cn in ['same', 'diff', 'func', 'null']:
            hs_ctx = all_hs[w]['conds'][cn]['hs']
            delta_raw = hs_ctx - hs_iso
            delta_spec = delta_raw - d_ctrl
            tot_raw = np.linalg.norm(delta_raw, axis=1)
            tot_spec = np.linalg.norm(delta_spec, axis=1)
            cls_raw = np.abs(delta_raw @ cdir) / np.maximum(tot_raw, 1e-30)
            cls_spec = np.abs(delta_spec @ cdir) / np.maximum(tot_spec, 1e-30)
            dom_spec = np.abs(delta_spec @ dom_unit) / np.maximum(
                tot_spec, 1e-30)
            wr['conditions'][cn] = {
                'mag_raw': [round(float(x), 6) for x in tot_raw],
                'mag_spec': [round(float(x), 6) for x in tot_spec],
                'cls_raw': [round(float(x), 6) for x in cls_raw],
                'cls_spec': [round(float(x), 6) for x in cls_spec],
                'dom_spec': [round(float(x), 6) for x in dom_spec],
            }
        per_word.append(wr)

    F = n_layers - 1

    def mean_over(cond, field):
        vals = [w['conditions'][cond][field][F] for w in per_word
                if cond in w['conditions']]
        return float(np.mean(vals))

    s1_same = float(np.mean([w['conditions']['same']['cls_spec'][F]
                             for w in per_word]))
    s1_diff = float(np.mean([w['conditions']['diff']['cls_spec'][F]
                             for w in per_word]))
    s2_raw = float(np.mean([w['conditions']['same']['cls_raw'][F]
                            for w in per_word]))
    s3_same = float(np.mean([w['conditions']['same']['mag_spec'][F]
                             for w in per_word]))
    s3_diff = float(np.mean([w['conditions']['diff']['mag_spec'][F]
                             for w in per_word]))
    cls_spec_layer = np.mean(
        [[c for c in w['conditions']['same']['cls_spec']]
         for w in per_word], axis=0)
    s4_range = float(cls_spec_layer.max() - cls_spec_layer.min())

    v = {
        'n_targets': len(target_list),
        'n_layers': n_layers,
        'S1_same_cls_spec_final': round(s1_same, 6),
        'S1_diff_cls_spec_final': round(s1_diff, 6),
        'S2_raw_same_cls_final': round(s2_raw, 6),
        'S3_same_mag_spec_final': round(s3_same, 6),
        'S3_diff_mag_spec_final': round(s3_diff, 6),
        'S4_cls_spec_range': round(s4_range, 6),
        'S1_semantic_channel_selective': bool(s1_same > s1_diff),
        'S2_separation_gain': bool(s1_same > s2_raw),
        'S3_dose_response_restored': bool(s3_same > s3_diff),
        'S4_layer_resolved': bool(s4_range > 0.05),
        'peak_layer_cls_spec': int(np.argmax(cls_spec_layer)),
    }
    v['final_verdict'] = ('channel_separated_substantive'
                          if v['S1_semantic_channel_selective']
                          and v['S2_separation_gain'] else 'weak_or_absent')

    # ---------- exploratory head decomposition (final layer) ----------
    W_o = model.model.layers[L_FINAL].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)  # (2560, 4096)
    d_model = W_o.shape[0]
    n_heads, hd = 32, W_o.shape[1] // 32  # concat dim 4096 = 32 x 128
    W3 = W_o.reshape(d_model, n_heads, hd)
    head_rows = {}
    for cat, w in target_list:
        a_same = oproj_store[('same', w)][0].astype(np.float64)
        a_func = oproj_store[('func', w)][0].astype(np.float64)
        a_null = oproj_store[('null', w)][0].astype(np.float64)
        a_ctrl = 0.5 * (a_func + a_null)
        cdir = dW_unit[CAT_WORDS.index(cat)]
        d_h = {}
        for name, a in [('same', a_same), ('ctrl', a_ctrl)]:
            a3 = a.reshape(n_heads, hd)
            delta_h = np.einsum('dkh,kh->dk', W3, a3)  # (2560, n_heads)
            d_h[name] = delta_h
        delta_spec_h = d_h['same'] - d_h['ctrl']  # (2560, n_heads)
        n2 = float(np.linalg.norm(delta_spec_h.sum(axis=1)))
        c_h = delta_spec_h.T @ cdir / max(n2, 1e-30)  # (n_heads,)
        head_rows[w] = c_h
    head_mean = np.mean([head_rows[w] for cat, w in target_list], axis=0)
    order = np.argsort(-np.abs(head_mean))[:10]
    head_top = [{'head': 'L%d h%d' % (L_FINAL, int(i)),
                 'c_share': round(float(head_mean[i]), 4)}
                for i in order]
    v['head_decomposition_top10'] = head_top
    v['head_cls_share_mean_abs'] = round(float(np.abs(head_mean).mean()), 6)

    result = {'phase': 2833, 'prereg': PREREG, 'verdict': v,
              'per_word': per_word}
    fc.save(OUT / 'result.json', result)

    npz = {'cls_spec_layer_mean': cls_spec_layer.astype(np.float32)}
    for cat, w in target_list:
        d_raw = all_hs[w]['conds']['same']['hs'] - all_hs[w]['iso']
        d_ctrl = 0.5 * ((all_hs[w]['conds']['func']['hs'] - all_hs[w]['iso'])
                        + (all_hs[w]['conds']['null']['hs']
                           - all_hs[w]['iso']))
        npz[w + '_spec'] = (d_raw - d_ctrl).astype(np.float32)
    fc.npz(OUT / 'delta_specific.npz', **npz)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2833', elapsed)
    print('P2833 VERDICT %s' % json.dumps(v), flush=True)
    print('P2833 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
