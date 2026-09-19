"""Phase 2834 (LPF): Generative-chain spectrum transfer (decode phase).

2828/2830 tested knowledge chains under teacher forcing (fixed sentence
pairs) and found spectral correlation WITHOUT causal transfer.  This
phase lets the MODEL generate the second hop itself (greedy decode) and
asks whether the generated tokens carry the chain spectrum, and whether
that spectrum depends on the first-hop entity.

Design:
  prompt_true   "The apple is a fruit."
  prompt_broken "The rock is a fruit."
  greedy decode 8 tokens from each (do_sample=False).
  Then one forward over prompt+generated with a final-layer o_proj
  input capture at ALL positions; per-position per-head write spectrum
  c = (W_O^h @ a_h(pos)) . dW_dir for dir in {fruit, food}.

Prereg (frozen before any readout):
  G1  spontaneous_chain iff greedy decode of prompt_true contains a
      fruit-class token AND then a food-class token within 8 tokens,
      and prompt_broken does NOT produce a food-class token
  G2  decode_transfer iff top10-mean c_food at the generated segment is
      higher for true than broken (ratio > 1.5)
  G3  activation_persistence iff top10-mean c_fruit over generated
      segment true > broken
  verdict: generative_chain_confirmed iff G1 AND G2
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
OUT = BASE / 'phase2834' / 'generative_chain'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2834
N_GEN = 8

PREREG = {
    'G1': 'spontaneous_chain iff greedy decode of prompt_true contains '
          'a fruit-class token AND then a food-class token within 8 '
          'tokens, and prompt_broken does NOT produce a food-class token',
    'G2': 'decode_transfer iff top10-mean c_food at generated segment '
          'higher for true than broken (ratio > 1.5)',
    'G3': 'activation_persistence iff top10-mean c_fruit over generated '
          'segment true > broken',
    'verdict': 'generative_chain_confirmed iff G1 AND G2',
}


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED, 'n_gen': N_GEN}
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

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-30)

    dir_fruit = unit(dW[CAT_WORDS.index('fruit')])
    dir_food = unit(dW[CAT_WORDS.index('food')])
    DMAT = np.stack([dir_fruit, dir_food], axis=1)  # (2560, 2)

    fruit_tids = set(tid(w) for w in CATS['fruit'] if w in single_tok)
    food_tids = set(tid(w) for w in CATS['food'] if w in single_tok)
    # 'fruit' itself is in the fruit category list; food-class words for
    # detection
    print('P2834 fruit_tids=%d food_tids=%d' % (len(fruit_tids),
                                                len(food_tids)), flush=True)

    # ---------- o_proj input capture (final layer, all positions) ----------
    L_FINAL = 35
    cap = {}

    def hook(module, args):
        a = args[0].detach()[0].float().cpu().numpy()  # (seq, 4096)
        cap['x'] = a

    hnd = model.model.layers[L_FINAL].self_attn.o_proj \
        .register_forward_pre_hook(hook)

    def spectra(text):
        ids = tok(text, add_special_tokens=False)['input_ids']
        cap['x'] = None
        with torch.no_grad():
            out = model(torch.tensor([ids], device='cuda'),
                        output_hidden_states=False)
        a = cap['x'].astype(np.float64)  # (seq, 4096)
        W_o = model.model.layers[L_FINAL].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)  # (2560, 4096)
        d_model = W_o.shape[0]
        n_heads, hd = 32, W_o.shape[1] // 32
        W3 = W_o.reshape(d_model, n_heads, hd)
        seq = a.shape[0]
        A3 = a.reshape(seq, n_heads, hd)
        delta = np.einsum('dkh,skh->dsk', W3, A3)  # (2560, seq, n_heads)
        # per position: head contributions summed then projected
        c = np.einsum('dsk,dr->srk', delta, DMAT)  # (seq, n_heads, 2)
        return ids, c

    def top10_mean(v):
        return float(np.sort(np.abs(v))[::-1][:10].mean())

    results = {}
    spec_store = {}
    for name, prompt in [('true', 'The apple is a fruit.'),
                         ('broken', 'The rock is a fruit.')]:
        pin = tok(prompt, add_special_tokens=False)['input_ids']
        with torch.no_grad():
            gen = model.generate(
                torch.tensor([pin], device='cuda'),
                max_new_tokens=N_GEN, do_sample=False,
                pad_token_id=tok.eos_token_id)
        gen_ids = gen[0][len(pin):].tolist()
        gen_toks = [tok.decode([i]) for i in gen_ids]
        full = tok.decode(gen[0])
        ids, c = spectra(full)
        spec_store[name] = c.astype(np.float32)
        n_prompt = len(pin)
        rec = {'prompt': prompt, 'generated_ids': gen_ids,
               'generated_tokens': gen_toks, 'full_text': full,
               'n_prompt': n_prompt, 'n_total': len(ids)}
        # spectra at generated positions
        seg = c[n_prompt:, :, :]
        rec['c_fruit_gen_top10_mean'] = round(
            float(np.mean([top10_mean(seg[p, :, 0])
                           for p in range(seg.shape[0])])), 6)
        rec['c_food_gen_top10_mean'] = round(
            float(np.mean([top10_mean(seg[p, :, 1])
                           for p in range(seg.shape[0])])), 6)
        # per-position detail
        rec['per_pos'] = [{'pos': n_prompt + p, 'tok': gen_toks[p],
                           'c_fruit': round(top10_mean(seg[p, :, 0]), 4),
                           'c_food': round(top10_mean(seg[p, :, 1]), 4)}
                          for p in range(seg.shape[0])]
        results[name] = rec
        print('P2834 %s gen: %s' % (name, gen_toks), flush=True)
        print('P2834 %s c_fruit_gen=%.4f c_food_gen=%.4f'
              % (name, rec['c_fruit_gen_top10_mean'],
                 rec['c_food_gen_top10_mean']), flush=True)

    hnd.remove()

    # ---------- verdicts ----------
    gt = results['true']['generated_tokens']
    gb = results['broken']['generated_tokens']

    def has_class_after(toks, first_set, then_set):
        seen_first = False
        for t in toks:
            tt = t.strip()
            tids = tok(' ' + tt if tt else tt,
                       add_special_tokens=False)['input_ids']
            if len(tids) != 1:
                continue
            i = tids[0]
            if not seen_first and i in first_set:
                seen_first = True
            elif seen_first and i in then_set:
                return True
        return False

    has_food_broken = False
    for t in gb:
        tt = t.strip()
        tids = tok(' ' + tt if tt else tt,
                   add_special_tokens=False)['input_ids']
        if len(tids) == 1 and tids[0] in food_tids:
            has_food_broken = True
    g1 = bool(has_class_after(gt, fruit_tids, food_tids)
              and not has_food_broken)
    cf_t = results['true']['c_food_gen_top10_mean']
    cf_b = results['broken']['c_food_gen_top10_mean']
    g2 = bool(cf_t > 1.5 * cf_b) if cf_b > 1e-9 else bool(cf_t > 0)
    g3 = bool(results['true']['c_fruit_gen_top10_mean']
              > results['broken']['c_fruit_gen_top10_mean'])

    verdict = {
        'G1_spontaneous_chain': g1,
        'G2_decode_transfer': g2,
        'G3_activation_persistence': g3,
        'c_food_true': cf_t,
        'c_food_broken': cf_b,
        'c_food_ratio': round(cf_t / max(cf_b, 1e-9), 3),
        'c_fruit_true': results['true']['c_fruit_gen_top10_mean'],
        'c_fruit_broken': results['broken']['c_fruit_gen_top10_mean'],
        'final_verdict': ('generative_chain_confirmed'
                          if g1 and g2 else 'not_confirmed'),
    }

    result = {'phase': 2834, 'prereg': PREREG, 'verdict': verdict,
              'arms': results}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'gen_spec.npz',
           c_true=spec_store['true'], c_broken=spec_store['broken'])

    elapsed = time.monotonic() - t0
    cc.ledger('phase2834', elapsed)
    print('P2834 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2834 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
