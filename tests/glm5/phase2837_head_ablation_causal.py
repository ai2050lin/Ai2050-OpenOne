"""Phase 2837 (LPF Delta-III / III2): causal double test of the
attribute-write head cluster {L35 h20,h22,h23,h26}.

2836 showed (correlational) that the induction-copy step engages the
same heads that carry semantic (attribute) modulation.  This phase
ablates the cluster at L35 (zero the per-head o_proj input slices) and
measures, in the SAME run:

  (A) attribute channel damage:
      cls_spec_final(raw) per 2833/2836 protocol, 20 words x
      {iso, same, diff, func, null}, three states:
      base / ablate_write / ablate_random (4 matched random heads,
      seed 2837, disjoint from the cluster)
  (B) induction copy survival:
      20 probes "The {X} is a {Y}. The" -> greedy 1 step; success iff
      the generated token equals the entity X.

Prereg (frozen before any readout):
  C1  attribute_channel_damaged iff
      spec_drop_write = 1 - cls_spec(ablate_write)/cls_spec(base) > 0.30
  C2  damage_head_specific iff spec_drop_write > 2 * spec_drop_random
  C3  induction_damaged iff
      (base_success - write_success) / max(base_success, 1) >= 0.5
  verdict:
    hardware_unified   iff C1 and C2 and C3
    attribute_only     iff C1 and C2 and not C3
    induction_only     iff (not C1 or not C2) and C3
    decoupled          otherwise
  Descriptors: spec_drop_random, random-state induction survival,
  per-word cls_spec table, per-probe copy logits.
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
OUT = BASE / 'phase2837' / 'head_ablation_causal'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2837
L_FINAL = 35
WRITE_HEADS = [20, 22, 23, 26]

PREREG = {
    'C1': 'attribute_channel_damaged iff spec_drop_write = '
          '1 - cls_spec(ablate_write)/cls_spec(base) > 0.30',
    'C2': 'damage_head_specific iff spec_drop_write > 2 * '
          'spec_drop_random (4 random heads, seed 2837, disjoint)',
    'C3': 'induction_damaged iff (base_success - write_success) / '
          'max(base_success,1) >= 0.5 over 20 entity probes',
    'verdict': 'hardware_unified iff C1 and C2 and C3; attribute_only '
               'iff C1 and C2 and not C3; induction_only iff '
               '(not C1 or not C2) and C3; decoupled otherwise',
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
                 'prereg': PREREG, 'seed': SEED,
                 'write_heads': WRITE_HEADS}
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

    # ---------- ablation hooks ----------
    active_heads = {'heads': None}

    def ablate_pre_hook(module, args):
        hs = active_heads['heads']
        if hs is None:
            return None
        a = args[0]
        a2 = a.clone()
        v = a2.view(*a2.shape[:-1], 32, a2.shape[-1] // 32)
        v[..., list(hs), :] = 0
        return (a2,) + tuple(args[1:])

    abl_hnd = model.model.layers[L_FINAL].self_attn.o_proj \
        .register_forward_pre_hook(ablate_pre_hook)

    def set_state(state, rng):
        if state == 'base':
            active_heads['heads'] = None
        elif state == 'write':
            active_heads['heads'] = list(WRITE_HEADS)
        else:  # random matched control, fixed by seed, disjoint
            pool = [h for h in range(32) if h not in WRITE_HEADS]
            active_heads['heads'] = sorted(
                rng.choice(pool, size=4, replace=False).tolist())

    # residual capture of attn/mlp per layer (for raw final state)
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
        for d in ('attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        return hs, attn, mlp

    # ---------- arm A: attribute channel damage ----------
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

    LAST = 35
    cls_table = {}
    for i, (cat, w) in enumerate(target_list):
        w_tid = tid(w)
        cdir = dW_unit[CAT_WORDS.index(cat)]
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        mod_same = same_cat[0]
        diff_cats = [c for c in CAT_WORDS if DOMAIN_OF[c] != DOMAIN_OF[cat]]
        mod_diff = [x for x in CATS[diff_cats[0]] if x in single_tok][0]
        conds = {'same': [tid(mod_same), w_tid],
                 'diff': [tid(mod_diff), w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        row = {}
        for state in ['base', 'write', 'random']:
            set_state(state, rng)
            hs_iso, _, _ = get_cond([w_tid], 0)
            hs_iso0 = hs_iso[0]
            ds = {}
            for cn, toks in conds.items():
                hs, attn, mlp = get_cond(toks, 1)
                raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
                ds[cn] = raw_fin - hs_iso0
            d_ctrl = 0.5 * (ds['func'] + ds['null'])
            d_spec = ds['same'] - d_ctrl
            n2 = float(np.linalg.norm(d_spec))
            row[state] = {'cls_spec': float(abs(d_spec @ cdir)
                                            / max(n2, 1e-30)),
                          'mag_spec': n2}
        cls_table[w] = {'cat': cat, **{s: row[s] for s in row}}
        print('P2837 arm-a [%d/%d] %s base=%.4f write=%.4f random=%.4f'
              % (i + 1, len(target_list), w, row['base']['cls_spec'],
                 row['write']['cls_spec'], row['random']['cls_spec']),
              flush=True)

    cls_base = float(np.mean([cls_table[w]['base']['cls_spec']
                              for _, w in target_list]))
    cls_write = float(np.mean([cls_table[w]['write']['cls_spec']
                               for _, w in target_list]))
    cls_random = float(np.mean([cls_table[w]['random']['cls_spec']
                                for _, w in target_list]))
    spec_drop_write = 1.0 - cls_write / max(cls_base, 1e-30)
    spec_drop_random = 1.0 - cls_random / max(cls_base, 1e-30)
    c1 = spec_drop_write > 0.30
    c2 = spec_drop_write > 2 * max(spec_drop_random, 0.0)

    # ---------- arm B: induction copy survival ----------
    entity_probes = []
    for cat in CAT_WORDS:
        for w in targets[cat][:2]:
            entity_probes.append((cat, w))
    vowel = 'aeiou'
    probes = []
    for state in ['base', 'write', 'random']:
        set_state(state, rng)
        succ = 0
        rows = []
        for cat, w in entity_probes:
            art = 'an' if w[0].lower() in vowel else 'a'
            prompt = 'The %s is %s %s. The' % (w, art, cat)
            ids = tok(prompt, add_special_tokens=False)['input_ids']
            with torch.no_grad():
                logits = model(torch.tensor([ids], device='cuda')) \
                    .logits[0, -1].float().cpu().numpy()
            nid = int(np.argmax(logits))
            txt = tok.decode([nid]).strip()
            ok = (txt == w)
            succ += int(ok)
            rows.append({'entity': w, 'cat': cat, 'gen': txt,
                         'copy': bool(ok),
                         'copy_logit': round(float(logits[tid(w)]), 3),
                         'top1_logit': round(float(logits[nid]), 3)})
        probes.append({'state': state, 'success': succ,
                       'rate': succ / len(entity_probes),
                       'rows': rows})
        print('P2837 arm-b %s success %d/%d'
              % (state, succ, len(entity_probes)), flush=True)

    base_succ = probes[0]['success']
    write_succ = probes[1]['success']
    random_succ = probes[2]['success']
    c3 = (base_succ - write_succ) / max(base_succ, 1) >= 0.5

    if c1 and c2 and c3:
        verdict = 'hardware_unified'
    elif c1 and c2 and not c3:
        verdict = 'attribute_only'
    elif (not c1 or not c2) and c3:
        verdict = 'induction_only'
    else:
        verdict = 'decoupled'

    v = {
        'cls_spec_base': round(cls_base, 6),
        'cls_spec_write': round(cls_write, 6),
        'cls_spec_random': round(cls_random, 6),
        'spec_drop_write': round(spec_drop_write, 4),
        'spec_drop_random': round(spec_drop_random, 4),
        'C1_attribute_channel_damaged': bool(c1),
        'C2_damage_head_specific': bool(c2),
        'induction_base_success': base_succ,
        'induction_write_success': write_succ,
        'induction_random_success': random_succ,
        'induction_drop_rate': round((base_succ - write_succ)
                                     / max(base_succ, 1), 4),
        'C3_induction_damaged': bool(c3),
        'final_verdict': verdict,
    }

    result = {'phase': 2837, 'prereg': PREREG, 'verdict': v,
              'cls_table': cls_table, 'probes': probes}
    fc.save(OUT / 'result.json', result)

    npz = {w + '_cls': np.array([cls_table[w][s]['cls_spec']
                                 for s in ('base', 'write', 'random')],
                                dtype=np.float32)
           for _, w in target_list}
    npz['copy_logit_base'] = np.array(
        [r['copy_logit'] for r in probes[0]['rows']], dtype=np.float32)
    npz['copy_logit_write'] = np.array(
        [r['copy_logit'] for r in probes[1]['rows']], dtype=np.float32)
    fc.npz(OUT / 'ablation_spec.npz', **npz)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2837', elapsed)
    print('P2837 VERDICT %s' % json.dumps(v), flush=True)
    print('P2837 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
