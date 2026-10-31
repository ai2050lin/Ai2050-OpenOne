# -*- coding: utf-8 -*-
"""怀疑检验：top-1% 消融的 KL 那么大，是因为「动到了关键单元」，还是因为「模型被打崩了」？

对每题比较 base 与 top1-ablated：
  entropy(base) / entropy(ablated)、top-1 token 是否还是字母、答案 margin 变化、
  以及「答案字母里的概率质量」。
若 ablated 分布熵爆炸、top-1 变成乱码 token、字母质量塌到 0，那 KL 只是「坏掉」的度量，
本轮结论作废。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair
from server import mech_service as msvc

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def main():
    model, tok, device, meta, model_name = mpair._wait_ready()
    letter_ids = latlas._letter_token_ids(tok)
    ldirs = msvc._letter_dirs(model, tok)
    attr = np.load(Path('tempdata') / ('unit_attr_L%d.npz' % BLOCK))['attr_sum']
    d_ff = attr.size
    top1 = np.argsort(-np.abs(attr))[:max(1, d_ff // 100)]
    m_top = np.zeros(d_ff, dtype=bool)
    m_top[top1] = True

    rows = [r for r in latlas._load_questions() if (r.get('source') or '') == 'C-Eval val']
    rng = np.random.default_rng(5)
    if NQ < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=NQ, replace=False)]

    out = []
    for q in rows:
        prompt, lines, _ = latlas._build_prompt(q)
        ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:640]
        T = int(ids.shape[0])
        spans, tot = latlas._spans(tok, lines)
        if abs(T - tot) > 4:
            continue
        ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
        others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
                  if x != q['answer'] and letter_ids.get(x)]
        if ans_tok is None or not others:
            continue
        other_ids = [i for x in 'ABCDE' if x != q['answer'] for i in (letter_ids.get(x) or [])]

        def run(mask=None):
            hs = []
            if mask is not None:
                msk = torch.as_tensor(mask, device=device)

                def abl(_m, args):
                    a = args[0].clone()
                    a[..., msk] = 0.0
                    return (a,) + tuple(args[1:])
                hs.append(model.model.layers[BLOCK].mlp.down_proj.register_forward_pre_hook(abl))
            try:
                with torch.no_grad():
                    o = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
                lg = o.logits[0, -1].float()
                p = torch.softmax(lg, dim=-1)
                ent = float(-(p * torch.log(p + 1e-12)).sum())
                top1_tok = int(lg.argmax())
                letter_mass = float(p[other_ids].sum()) if other_ids else 0.0
                return ent, top1_tok, float(lg[ans_tok] - lg[list(others)].max()), letter_mass
            finally:
                for h in hs:
                    h.remove()

        e0, t0, m0, lm0 = run(None)
        e1, t1, m1, lm1 = run(m_top)
        out.append({'ent_base': e0, 'ent_abl': e1, 'top1_base': tok.decode([t0]),
                    'top1_abl': tok.decode([t1]), 'marg_base': m0, 'marg_abl': m1,
                    'letter_mass_base': lm0, 'letter_mass_abl': lm1,
                    'ent_ratio': e1 / max(e0, 1e-9)})

    o = {k: round(float(np.mean([r[k] for r in out])), 3) for k in
         ('ent_base', 'ent_abl', 'ent_ratio', 'marg_base', 'marg_abl',
          'letter_mass_base', 'letter_mass_abl')}
    o['n'] = len(out)
    o['frac_top1_still_letter'] = round(float(np.mean([
        1.0 if r['top1_abl'].strip() in list('ABCDE') else 0.0 for r in out])), 3)
    o['frac_base_top1_letter'] = round(float(np.mean([
        1.0 if r['top1_base'].strip() in list('ABCDE') else 0.0 for r in out])), 3)
    o['examples'] = [{'base': r['top1_base'], 'abl': r['top1_abl']} for r in out[:8]]
    p = Path('tempdata') / ('degrade_check_L%d.json' % BLOCK)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(o, ensure_ascii=False, indent=1))
    print('written', p)


if __name__ == '__main__':
    main()
