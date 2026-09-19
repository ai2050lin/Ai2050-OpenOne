"""Phase 2810 (LPF-23): Phrase-level archive modulation — dose-response
curve and direction decomposition.

FIRST NON-ZERO-FORWARD PHASE.  All prior phases (2806-2809) used the
static embedding / W_U side.  This phase runs actual forward passes to
extract context-dependent hidden states and measure how a target noun's
"archive" shifts when preceded by a modifier word.

Design:
  20 target words (2 per category × 10 categories) × 5 conditions
  Conditions:
    iso    target alone (baseline)
    same   [same-category word] [target]   (e.g. "banana apple")
    diff   [opposite-domain word] [target] (e.g. "hammer apple")
    func   "the" [target]                  (function word control)
    null   [random vocab word] [target]    (null control)

  37 layers (0 = embedding, 1-36 = transformer) extracted per condition.

Metrics per layer:
  cos        cos(h_ctx, h_iso)            direction preservation
  rel_shift  ||h_ctx - h_iso|| / ||h_iso||  magnitude shift
  class_share  |<delta, class_dir>| / ||delta||  fraction along class axis
  dom_share    |<delta, dom_dir>|   / ||delta||  fraction along domain axis

Prereg (frozen before any readout):
  P-K1  modulation_exists iff cos(h_same, h_iso) < 0.999 at >=1 layer
  P-K2  selective_modulation iff mean ||delta_same|| > mean ||delta_diff||
        at final layer
  P-K3  layer_dependent iff cos range across layers > 0.05 for any
        content condition (same or diff)
  P-K4  function_word_minimal iff mean ||delta_func|| < mean ||delta_content||
        at final layer
  verdict: phrase_modulation_substantive iff P-K1 AND P-K2;
           layer_resolved iff P-K3; function_word_baseline iff P-K4
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
OUT = BASE / 'phase2810' / 'phrase_modulation'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2810

PREREG = {
    'P-K1': 'modulation_exists iff cos(h_same, h_iso) < 0.999 at >=1 layer',
    'P-K2': 'selective_modulation iff mean ||delta_same|| > mean ||delta_diff|| at final layer',
    'P-K3': 'layer_dependent iff cos range across layers > 0.05 for any content condition',
    'P-K4': 'function_word_minimal iff mean ||delta_func|| < mean ||delta_content|| at final layer',
    'verdict': 'phrase_modulation_substantive iff P-K1 AND P-K2; '
               'layer_resolved iff P-K3; function_word_baseline iff P-K4',
}


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    NAT_DOM = exec2806['nat_dom']
    ART_DOM = exec2806['art_dom']
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

    print('P2810 single-token words: %d/%d' % (len(single_tok), len(all_words)),
          flush=True)

    targets = {}
    for cat in CAT_WORDS:
        cs = [w for w in CATS[cat] if w in single_tok]
        targets[cat] = cs[:2]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]
    print('P2810 targets: %d' % len(target_list), flush=True)

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    nat_idx = [CAT_WORDS.index(c) for c in NAT_DOM]
    art_idx = [CAT_WORDS.index(c) for c in ART_DOM]
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
    assert func_tid is not None, 'No single-token function word found'

    def get_hs(tokens, target_pos):
        with torch.no_grad():
            ids = torch.tensor([tokens], device='cuda')
            out = model(ids, output_hidden_states=True)
            hs = np.stack([h[0, target_pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        return hs

    all_hs = {}
    n_layers = None

    for i, (cat, w) in enumerate(target_list):
        w_tid = tid(w)
        hs_iso = get_hs([w_tid], 0)
        if n_layers is None:
            n_layers = hs_iso.shape[0]
            print('P2810 n_layers = %d, hidden_dim = %d'
                  % (n_layers, hs_iso.shape[1]), flush=True)

        all_hs[w] = {'iso': hs_iso, 'cat': cat, 'dom': DOMAIN_OF[cat],
                     'conds': {}}

        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w and x in single_tok][:1]
        if same_cat:
            m_tid = tid(same_cat[0])
            all_hs[w]['conds']['same'] = {
                'mod': same_cat[0], 'hs': get_hs([m_tid, w_tid], 1)}

        dom = DOMAIN_OF[cat]
        if dom == 'natural':
            diff_cats = [c for c in CAT_WORDS if DOMAIN_OF[c] == 'artifact']
        elif dom == 'artifact':
            diff_cats = [c for c in CAT_WORDS if DOMAIN_OF[c] == 'natural']
        else:
            diff_cats = [c for c in CAT_WORDS if DOMAIN_OF[c] == 'natural']
        diff_words = [x for x in CATS[diff_cats[0]] if x in single_tok]
        if diff_words:
            m_tid = tid(diff_words[0])
            all_hs[w]['conds']['diff'] = {
                'mod': diff_words[0], 'hs': get_hs([m_tid, w_tid], 1)}

        all_hs[w]['conds']['func'] = {
            'mod': 'the', 'hs': get_hs([func_tid, w_tid], 1)}

        nt = null_tids[i]
        all_hs[w]['conds']['null'] = {
            'mod_tid': nt, 'hs': get_hs([nt, w_tid], 1)}

        print('P2810 [%d/%d] %s (%s) done' % (i + 1, len(target_list), w, cat),
              flush=True)

    per_word = []
    for cat, w in target_list:
        hs_iso = all_hs[w]['iso']
        iso_norm = np.linalg.norm(hs_iso, axis=1)
        wr = {'word': w, 'category': cat, 'domain': DOMAIN_OF[cat],
              'conditions': {}}
        cat_idx = CAT_WORDS.index(cat)
        cdir = dW_unit[cat_idx]

        for cn in ['same', 'diff', 'func', 'null']:
            if cn not in all_hs[w]['conds']:
                continue
            hs_ctx = all_hs[w]['conds'][cn]['hs']
            cos = np.array([
                float(np.dot(hs_iso[l], hs_ctx[l]) /
                      (max(np.linalg.norm(hs_iso[l]), 1e-30) *
                       max(np.linalg.norm(hs_ctx[l]), 1e-30)))
                for l in range(n_layers)])
            delta = hs_ctx - hs_iso
            rel_shift = np.linalg.norm(delta, axis=1) / np.maximum(iso_norm, 1e-30)
            total = np.linalg.norm(delta, axis=1)
            cls_p = np.abs(delta @ cdir)
            dom_p = np.abs(delta @ dom_unit)
            wr['conditions'][cn] = {
                'cos': [round(float(x), 6) for x in cos],
                'rel_shift': [round(float(x), 6) for x in rel_shift],
                'class_share': [round(float(x), 6)
                                for x in cls_p / np.maximum(total, 1e-30)],
                'dom_share': [round(float(x), 6)
                             for x in dom_p / np.maximum(total, 1e-30)],
            }
        per_word.append(wr)

    same_cos_f = np.mean([w['conditions']['same']['cos'][-1]
                          for w in per_word if 'same' in w['conditions']])
    diff_cos_f = np.mean([w['conditions']['diff']['cos'][-1]
                          for w in per_word if 'diff' in w['conditions']])
    same_shift_f = np.mean([w['conditions']['same']['rel_shift'][-1]
                            for w in per_word if 'same' in w['conditions']])
    diff_shift_f = np.mean([w['conditions']['diff']['rel_shift'][-1]
                            for w in per_word if 'diff' in w['conditions']])
    func_shift_f = np.mean([w['conditions']['func']['rel_shift'][-1]
                            for w in per_word if 'func' in w['conditions']])
    null_shift_f = np.mean([w['conditions']['null']['rel_shift'][-1]
                            for w in per_word if 'null' in w['conditions']])
    content_shift_f = np.mean([same_shift_f, diff_shift_f])

    max_cos_range = 0.0
    for w in per_word:
        for cn in ['same', 'diff']:
            if cn in w['conditions']:
                c = np.array(w['conditions'][cn]['cos'])
                max_cos_range = max(max_cos_range, float(c.max() - c.min()))

    p_k1 = bool(same_cos_f < 0.999)
    p_k2 = bool(same_shift_f > diff_shift_f)
    p_k3 = bool(max_cos_range > 0.05)
    p_k4 = bool(func_shift_f < content_shift_f)

    verdict = {
        'n_targets': len(target_list),
        'n_layers': n_layers,
        'conditions': ['same', 'diff', 'func', 'null'],
        'same_cos_final': round(float(same_cos_f), 6),
        'diff_cos_final': round(float(diff_cos_f), 6),
        'same_shift_final': round(float(same_shift_f), 6),
        'diff_shift_final': round(float(diff_shift_f), 6),
        'func_shift_final': round(float(func_shift_f), 6),
        'null_shift_final': round(float(null_shift_f), 6),
        'content_shift_final': round(float(content_shift_f), 6),
        'max_cos_range': round(float(max_cos_range), 6),
        'modulation_exists': p_k1,
        'selective_modulation': p_k2,
        'layer_dependent': p_k3,
        'function_word_minimal': p_k4,
        'final_verdict': ('phrase_modulation_substantive'
                          if p_k1 and p_k2 else 'weak_or_absent'),
    }
    result = {'phase': 2810, 'prereg': PREREG, 'verdict': verdict,
              'per_word': per_word}
    fc.save(OUT / 'result.json', result)

    npz_data = {}
    for cat, w in target_list:
        npz_data[w + '_iso'] = all_hs[w]['iso']
        for cn in ['same', 'diff', 'func', 'null']:
            if cn in all_hs[w]['conds']:
                npz_data[w + '_' + cn] = all_hs[w]['conds'][cn]['hs']
    fc.npz(OUT / 'hidden_states.npz', **npz_data)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2810', elapsed)
    print('P2810 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2810 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
