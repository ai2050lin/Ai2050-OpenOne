# -*- coding: utf-8 -*-
"""Phase 2995 probe: GLM4-9B load + hook structure + tokenizer check."""
import io
import json
import os
import traceback

import torch as _t

MODEL = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
BASE72 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
          r'\rdc_query_construction_20260913\phase2972')
OUT = r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05\.workbuddy\tmp_probe2995.txt'

o = []


def log(s):
    o.append(str(s))


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    log('tokenizer ok vocab=%d' % tok.vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=_t.bfloat16).cuda().eval()
    cfg = model.config
    log('model ok type=%s h=%d L=%d H=%d HD=%s kv=%d' % (
        cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers,
        cfg.num_attention_heads,
        getattr(cfg, 'head_dim', None), cfg.num_key_value_heads))

    layers = model.model.layers
    NL = len(layers)
    log('nlayers=%d' % NL)

    # hook structure probe
    cap = {}

    def hk_selfattn(mod, args, kwargs):
        x = kwargs['hidden_states']
        cap['attn_in'] = x.detach().float().cpu().numpy().copy()
        return None

    def hk_downproj(mod, args):
        cap['mlp_mid'] = args[0].detach().float().cpu().numpy().copy()
        return None

    def hk_oproj(mod, args):
        cap['op_in'] = args[0].detach().float().cpu().numpy().copy()
        return None

    h1 = layers[10].self_attn.register_forward_pre_hook(
        hk_selfattn, with_kwargs=True)
    h2 = layers[10].mlp.down_proj.register_forward_pre_hook(hk_downproj)
    h3 = layers[10].self_attn.o_proj.register_forward_pre_hook(hk_oproj)

    ids = tok('the', add_special_tokens=False)['input_ids']
    log('the ids=%s' % ids)
    w_ids = tok('apple', add_special_tokens=False)['input_ids']
    log('apple ids=%s' % w_ids)
    seq = ids + w_ids
    with _t.no_grad():
        out = model(_t.tensor([seq], device='cuda'))
    log('forward ok logits=%s' % (tuple(out.logits.shape),))
    h1.remove(); h2.remove(); h3.remove()
    log('attn_in %s mlp_mid %s op_in %s' % (
        cap['attn_in'].shape, cap['mlp_mid'].shape,
        cap['op_in'].shape))
    log('attn_in[0,-1,:4]=%s' % cap['attn_in'][0, -1, :4])

    # weights
    Wo = layers[39].self_attn.o_proj.weight
    log('o_proj w %s' % (tuple(Wo.shape),))
    Wup = layers[10].mlp.up_proj.weight
    log('up_proj w %s' % (tuple(Wup.shape),))
    Wd = layers[10].mlp.down_proj.weight
    log('down_proj w %s' % (tuple(Wd.shape),))

    # tokenizer: single-token filter over all source words
    e72 = None
    for s in os.listdir(BASE72):
        p = os.path.join(BASE72, s, 'execution.json')
        if os.path.isfile(p):
            e72 = json.load(io.open(p, encoding='utf-8'))
            break
    cells = e72['cells']
    L_CAND = ["because", "therefore", "although", "unless",
              "however", "thus", "moreover", "since", "whereas",
              "despite", "hence", "nevertheless", "consequently",
              "furthermore", "otherwise", "instead", "while",
              "accordingly", "likewise", "meanwhile", "nonetheless",
              "thereafter", "whereby", "albeit"]
    allw = {}
    for k in ['F_en', 'F_fr', 'C_en', 'C_fr']:
        allw[k] = cells[k]
    allw['L_cand'] = L_CAND
    for k, ws in allw.items():
        single, multi = [], []
        for w in ws:
            ii = tok(w, add_special_tokens=False)['input_ids']
            (single if len(ii) == 1 else multi).append(w)
        log('%s n=%d single=%d multi=%s' % (k, len(ws), len(single), multi))

    # determinism probe: same forward twice, compare last hidden
    def fwd_last():
        with _t.no_grad():
            oo = model(_t.tensor([seq], device='cuda'),
                       output_hidden_states=True)
        return oo.hidden_states[-1][0, -1].detach().float().cpu().numpy()

    a = fwd_last()
    b = fwd_last()
    log('determinism max|a-b|=%.3e rel=%.3e' % (
        abs(a - b).max(),
        (abs(a - b) / max(abs(a).max(), 1e-30)).max()))

    # GPU mem
    log('mem=%.1f GB' % (_t.cuda.memory_allocated() / 2**30))
except Exception:
    log(traceback.format_exc())

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('probe done')
