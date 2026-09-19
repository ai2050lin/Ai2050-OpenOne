"""Probe 2790-1: structure check + full-layer lens curve for 'apple'.

Before freezing the Phase 2790 prereg, verify:
  1. Qwen3-4B layer module names for MLP hooking.
  2. len(output.hidden_states) == n_layers + 1.
  3. Full-layer (0..36) lens rank/logit of apple's property tokens
     ('fruit','red','plant') and a control pair ('animal','green')
     at the word position of "The apple" and one in-situ sentence.
Writes report to probe2790_1.txt (stdout unreliable in this env).
"""
import sys
from pathlib import Path

REPORT = Path('C:/Users/Admin/WorkBuddy/2026-09-15-08-09-16/probe2790_1.txt')
lines = []


def out(s=''):
    lines.append(str(s))


def main():
    sys.path.insert(0, 'D:/AI2050/Ai2050-OpenOne/tests/glm5')
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from phase2662_symmetric_mapping_contract import load_native
    import rdc_construction_common as cc

    ROOT = cc.ROOT
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    model, _ = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm

    n_layers = len(model.model.layers)
    out('n_layers=%d' % n_layers)
    out('layers[20] children: %s'
        % [n for n, _ in model.model.layers[20].named_children()])
    out('mlp children: %s'
        % [n for n, _ in model.model.layers[20].mlp.named_children()])

    def lens_all_layers(ids, pos):
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
            hs = o.hidden_states
            out('n_hidden_states=%d' % len(hs))
            res = []
            for l in range(len(hs)):
                h = hs[l][0, pos].float().cpu().numpy()
                hn = final_norm(torch.tensor(h, device=device).unsqueeze(0))
                z = W_U @ hn[0].float().cpu().numpy()
                res.append(z)
        return res

    tok_ids = tok('The apple', add_special_tokens=False)['input_ids']
    p_apple = 1
    out('probe sentence "The apple" ids=%s toks=%s'
        % (tok_ids, [tok.decode([t]) for t in tok_ids]))

    targets = ['fruit', 'red', 'plant', 'apple', 'animal', 'green', 'dog']
    zseq = lens_all_layers(tok_ids, p_apple)
    out('=== "The apple" @pos1: rank/logit per layer ===')
    for l in range(0, len(zseq), 2):
        z = zseq[l]
        row = []
        for t in targets:
            tid = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(tid) != 1:
                tid = tok(t, add_special_tokens=False)['input_ids']
            zz = float(z[tid[0]])
            rank = int((z > zz).sum())
            row.append('%s:r%d/%.1f' % (t, rank, zz))
        out('L%02d %s' % (l, ' '.join(row)))

    # one in-situ sentence from the 2788/2789 panel
    s = 'Unlike the car, the apple was seen at the market yesterday'
    ids = tok(s, add_special_tokens=False)['input_ids']
    # find 'apple' token position (first subtoken of ' apple')
    ap = tok(' apple', add_special_tokens=False)['input_ids']
    pos = None
    for i in range(len(ids) - len(ap) + 1):
        if ids[i:i + len(ap)] == ap:
            pos = i
            break
    out('in-situ sentence len=%d apple pos=%s' % (len(ids), pos))
    zseq2 = lens_all_layers(ids, pos)
    out('=== in-situ apple @pos%d: rank/logit per layer ===' % pos)
    for l in range(0, len(zseq2), 2):
        z = zseq2[l]
        row = []
        for t in targets:
            tid = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(tid) != 1:
                tid = tok(t, add_special_tokens=False)['input_ids']
            zz = float(z[tid[0]])
            rank = int((z > zz).sum())
            row.append('%s:r%d/%.1f' % (t, rank, zz))
        out('L%02d %s' % (l, ' '.join(row)))

    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print('OK')


if __name__ == '__main__':
    main()
