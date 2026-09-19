"""Phase 2860 R1-failure probe: hook path vs rebuild path, one word.

Layers L26-35, first target word of the SEED=2855 vocab, 'same' cond.
Captures, in ONE real forward:
  pre_ln2_in   (pre-hook on post_attention_layernorm)  = true LN2 input
  mlp_in       (pre-hook on mlp)                        = true mlp input
  mlp_out      (hook on mlp)                            = true mlp output
Then rebuilds off the captured sain/attn values:
  x_in_f   = sain[l][1] + attn[l][1]
  ln_x_f   = LN2(x_in_f)
  mlp_f    = mlp(ln_x_f)
Reports per-layer max-abs diffs at each stage.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2855
MAX_WORDS = 8
LAST = 35
NL = 36
WIN_LO, WIN_HI = 26, 36
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2860_probe_report.txt'

lines = []


def w(s):
    lines.append(s)


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

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

    all_words = [x for v in CATS.values() for x in v]
    single_tok = []
    for x in all_words:
        try:
            tid(x)
            single_tok.append(x)
        except AssertionError:
            pass
    targets = {c: [x for x in CATS[c] if x in single_tok][:MAX_WORDS]
               for c in CAT_WORDS}
    target_list = [(c, x) for c in CAT_WORDS for x in targets[c]]

    Erows = {x: W_U[tid(x)].astype(np.float64) for x in single_tok}
    cents = [np.stack([Erows[x] for x in CATS[c] if x in single_tok]).mean(0)
             for c in CAT_WORDS]
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])

    rng = np.random.default_rng(SEED)
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < len(target_list):
        r = int(rng.integers(0, W_U.shape[0]))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = tid('the')

    layers = model.model.layers
    vdt = next(layers[0].mlp.parameters()).dtype

    cap = {'sain': [], 'attn': [], 'ln2in': [], 'mlpin': [], 'mlpout': []}

    def ohook(kind, pick0):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].append(o[0].detach().float().cpu().numpy())
        return hook

    def ihook(kind):
        def pre(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            cap[kind].append(x.detach()[0].float().cpu().numpy())
        return pre

    hs_ = []
    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            ohook('attn', True)))
        handles.append(layer.self_attn.register_forward_pre_hook(
            ihook('sain'), with_kwargs=True))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            ihook('ln2in'), with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(
            ihook('mlpin'), with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(ohook('mlpout', True)))

    cat0, w0 = target_list[0]
    ci = CAT_WORDS.index(cat0)
    cdir = dW_unit[ci]
    same_cat = [x for x in targets[cat0] if x != w0][:1]
    toks = [tid(same_cat[0]), tid(w0)]

    with torch.no_grad():
        model(torch.tensor([toks], device='cuda'))

    sain = {li: cap['sain'][li] for li in range(NL)}
    attn = np.stack([cap['attn'][li] for li in range(NL)])
    ln2in = {li: cap['ln2in'][li] for li in range(NL)}
    mlpin = {li: cap['mlpin'][li] for li in range(NL)}
    mlpout = np.stack([cap['mlpout'][li] for li in range(NL)])

    w('word (%s, %s) toks %s  layers L%d..%d'
      % (cat0, w0, toks, WIN_LO, WIN_HI - 1))
    w('q  L  |x_in_f - ln2in_true|   |ln_x_f - mlpin_true|   '
      '|mlp_f - mlpout_true|   proj_diff(cdir)')
    for q, l in enumerate(range(WIN_LO, WIN_HI)):
        x_in_f = sain[l][1].astype(np.float64) + attn[l][1].astype(np.float64)
        d_in = np.abs(x_in_f - ln2in[l][1].astype(np.float64)).max()
        t = torch.tensor(x_in_f[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            ln_x_f = layers[l].post_attention_layernorm(
                t)[0].detach().float().cpu().numpy().astype(np.float64)
        d_ln = np.abs(ln_x_f - mlpin[l][1].astype(np.float64)).max()
        t2 = torch.tensor(ln_x_f[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            mlp_f = layers[l].mlp(t2)[0].detach().float() \
                .cpu().numpy().astype(np.float64)
        d_out = np.abs(mlp_f - mlpout[l][1].astype(np.float64)).max()
        pd = float((mlp_f - mlpout[l][1].astype(np.float64)) @ cdir)
        w('%d %02d  %.6e  %.6e  %.6e  %+.6f' % (q, l, d_in, d_ln, d_out, pd))

    for h in handles:
        h.remove()
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('WROTE', OUT)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
