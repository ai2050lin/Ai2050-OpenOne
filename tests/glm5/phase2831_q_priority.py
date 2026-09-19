"""Phase 2831 = QUESTION-MODE PRIORITY (LPF-43).

User question 1: after 'I like to eat apples', different questions
('Who likes to eat apples?' 'Do I like to eat apples?' 'What kind of
apples do I like to eat?' 'What do I like to eat?') shift the answer
mode.  What mechanism picks the mode?  Is there a question-form
priority table?

Design: 5 conditions (none / who / whether / what-kind / what).
Prefix 'I like to eat apples' is token-aligned (3 tokens, apple@2).
All-Chinese direction set built from lm_head rows (leave-one-out for
color; antonym pairs for 7 scalar domains):
  red, big, heavy, sweet, hard, hot, fast, round.
Per condition: greedy generation (behavioural answer) + full-token
spectrum c[L, pos, head, 8 dirs].

Prereg (frozen):
  Q1_mode_switch: behavioural answers differ across the 4 question
      forms (answer token sets overlap < 50%)
  Q2_reprogram: apple@2 spectrum changes with question form: for at
      least 3 of 8 directions, |c(q) - c(none)| mean-top10 > null q95
  Q3_mode_separation: pairwise cosine (over 36x32 heads summed per
      direction, normalized) between the 4 question-form '?'-position
      spectra mean < 0.5
  Q4_priority_match: 'what-kind' condition has higher
      color+round+sweet spectrum at '?' position than 'who' and
      'whether' conditions (mode priority follows the question's
      interrogative slot)
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
OUT = BASE / 'phase2831' / 'q_priority'
SEED = 2831
N_NULL = 100
PREFIX = '我喜欢吃苹果'
CONDS = ['none', 'who', 'whether', 'whatkind', 'what']
QUESTIONS = {
    'who': '谁喜欢吃苹果？',
    'whether': '我是否喜欢吃苹果？',
    'whatkind': '我喜欢吃什么样的苹果？',
    'what': '我喜欢吃什么？',
}
PREFIX_LEN = 3  # '我喜欢' '吃' '苹果' -> apple at index 2

COLOR7 = ['红色', '黑色', '绿色', '蓝色', '黄色', '白色', '紫色']
PAIRS = {
    'big': ('大', '小'),
    'heavy': ('重', '轻'),
    'sweet': ('甜', '苦'),
    'hard': ('硬', '软'),
    'hot': ('热', '冷'),
    'fast': ('快', '慢'),
    'round': ('圆', '方'),
}
DIR_ORDER = ['red'] + list(PAIRS.keys())

PREREG = {
    'Q1': 'mode_switch: behavioural answers differ across 4 question '
          'forms (answer token overlap < 50%)',
    'Q2': 'reprogram: apple@2 spectrum changes with question form; '
          '>= 3/8 directions with |c(q)-c(none)| top10 > null q95',
    'Q3': 'mode_separation: mean pairwise cosine of 4 question-form '
          "'?'-position head-sum spectra < 0.5",
    'Q4': 'priority_match: what-kind > who and whether on '
          'color+round+sweet spectrum at ? position',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'prereg': PREREG, 'seed': SEED, 'prefix': PREFIX,
        'conds': CONDS, 'questions': QUESTIONS,
        'dir_order': DIR_ORDER, 'prefix_len': PREFIX_LEN,
        'note': 'question-form mode priority: all-Chinese direction '
                'set, behavioural generation + spectrum capture'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors ----------
    from safetensors import safe_open
    mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
    index = json.loads((mdir / 'model.safetensors.index.json')
                       .read_text(encoding='utf-8'))['weight_map']

    def read_tensor(name):
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            return f.get_tensor(name).float().numpy()

    try:
        Wu = read_tensor('lm_head.weight')
    except KeyError:
        Wu = read_tensor('model.embed_tokens.weight')
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    n_layers = int(cfg.get('num_hidden_layers', 36))
    n_heads = int(cfg.get('num_attention_heads', 32))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(mdir), local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, (t, len(ids))
            tc[t] = int(ids[0])
        return tc[t]

    cent_red = np.stack([Wu[tid(w)].astype(np.float64)
                         for w in COLOR7])
    dW_red = unit(cent_red[0] - cent_red[1:].mean(0))
    dirs = {'red': dW_red}
    for d, (w1, w2) in PAIRS.items():
        dirs[d] = unit(Wu[tid(w1)].astype(np.float64)
                       - Wu[tid(w2)].astype(np.float64))
    DMAT = np.stack([dirs[d] for d in DIR_ORDER], axis=1)  # (2560, 8)

    # ---------- model ----------
    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    sents = {c: PREFIX + QUESTIONS.get(c, '') for c in CONDS}
    enc = {c: tok(sents[c], add_special_tokens=False)['input_ids']
           for c in CONDS}
    for c in CONDS:
        assert enc[c][:PREFIX_LEN] == enc['none'][:PREFIX_LEN], c
    assert enc['none'][2] == tid('苹果')

    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0

    store = {}

    def make_cap(cid, Lr):
        def cap(mod, args):
            store[(cid, Lr)] = args[0][0].detach().float() \
                .cpu().numpy()
            return None
        return cap

    def run_cond(cid, gen=False):
        hs = [model.model.layers[i].self_attn.o_proj
              .register_forward_pre_hook(make_cap(cid, i))
              for i in range(n_layers)]
        ids = torch.tensor([enc[cid]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=ids.to(dev))
        for h in hs:
            h.remove()
        lg = out.logits[0].float().cpu().numpy()
        gen_toks = []
        if gen:
            cur = ids.to(dev)
            with torch.no_grad():
                for _ in range(8):
                    o = model(input_ids=cur)
                    nxt = int(o.logits[0, -1].argmax())
                    if nxt == tok.eos_token_id:
                        break
                    gen_toks.append(nxt)
                    cur = torch.cat([cur,
                                     torch.tensor([[nxt]], device=dev)],
                                    dim=1)
        return lg, gen_toks

    logits, gen_ids = {}, {}
    for c in CONDS:
        logits[c], gen_ids[c] = run_cond(c, gen=(c != 'none'))
        print('P2831 cond %s len %d gen %s' % (
            c, len(enc[c]),
            json.dumps([tok.decode([i]) for i in gen_ids[c]])),
            flush=True)

    # ---------- spectra ----------
    spec = {}
    for c in CONDS:
        sp = np.zeros((n_layers, len(enc[c]), n_heads, 8),
                      dtype=np.float32)
        for Lr in range(n_layers):
            W = read_tensor(
                'model.layers.%d.self_attn.o_proj.weight' % Lr)
            W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
            K3 = store[(c, Lr)].reshape(len(enc[c]), n_heads, hd)
            delta = np.einsum('dhk,phk->pdh', W3, K3)
            sp[Lr] = np.einsum('pdh,dq->phq', delta, DMAT) \
                .astype(np.float32)
            del W, W3, delta
        spec[c] = sp
    print('P2831 spectra done', flush=True)

    # ---------- random null ----------
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_NULL, d_model))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    null_parts = []
    for Lr in range(0, n_layers, 3):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % Lr)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[('none', Lr)].reshape(len(enc['none']), n_heads, hd)
        delta = np.einsum('dhk,phk->pdh', W3, K3)
        c = np.einsum('pdh,dq->phq', delta, U.T.astype(np.float64))
        null_parts.append(c.ravel())
        del W, W3, delta
    null_q95 = float(np.quantile(np.concatenate(null_parts), 0.95))
    print('P2831 null q95 %.4f' % null_q95, flush=True)

    def head_sum_dir(c, pos, di):
        return spec[c][:, pos, :, di].astype(np.float64).sum(axis=0)

    def top10_sum(c, pos, di):
        return float(np.sort(head_sum_dir(c, pos, di))[::-1][:10]
                     .mean())

    # Q2: apple@2 reprogramming
    q2_hits = []
    apple_diff = {}
    for di, d in enumerate(DIR_ORDER):
        base = head_sum_dir('none', 2, di)
        diffs = []
        for c in CONDS[1:]:
            diffs.append(np.abs(head_sum_dir(c, 2, di) - base).max())
        apple_diff[d] = round(float(np.mean(diffs)), 3)
        if np.mean(diffs) > null_q95:
            q2_hits.append(d)
    q2 = bool(len(q2_hits) >= 3)

    # Q3: '?'-position separation
    qmarks = {}
    for c in CONDS[1:]:
        qmarks[c] = len(enc[c]) - 1
        assert tok.decode([enc[c][qmarks[c]]]) == '？', c
    vecs = {}
    for c in CONDS[1:]:
        v = np.concatenate([head_sum_dir(c, qmarks[c], di)
                            for di in range(8)])
        vecs[c] = v / max(np.linalg.norm(v), 1e-30)
    cos_pairs = []
    for i, c1 in enumerate(CONDS[1:]):
        for c2 in CONDS[2 + i:]:
            cos_pairs.append((c1, c2,
                              round(float(vecs[c1] @ vecs[c2]), 3)))
    q3 = bool(np.mean([x[2] for x in cos_pairs]) < 0.5)

    # Q4: what-kind vs who/whether on color+round+sweet at '?'
    attr_idx = [DIR_ORDER.index(d) for d in ['red', 'round', 'sweet']]

    def attr_amp(c):
        return float(np.mean([top10_sum(c, qmarks[c], di)
                              for di in attr_idx]))

    a_whatkind = attr_amp('whatkind')
    a_who = attr_amp('who')
    a_whether = attr_amp('whether')
    q4 = bool(a_whatkind > a_who and a_whatkind > a_whether)

    # Q1: behavioural answer overlap
    sets = {c: set(gen_ids[c]) for c in CONDS[1:]}
    overlaps = []
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sa, sb = sets[names[i]], sets[names[j]]
            ov = len(sa & sb) / max(min(len(sa), len(sb)), 1)
            overlaps.append(round(ov, 3))
    q1 = bool(np.mean(overlaps) < 0.5)

    verdict = {
        'null_q95': round(null_q95, 4),
        'Q1_mode_switch': q1,
        'Q2_reprogram': q2,
        'Q3_mode_separation': q3,
        'Q4_priority_match': q4,
        'gen_answers': {c: [tok.decode([i]) for i in gen_ids[c]]
                        for c in CONDS[1:]},
        'overlaps': overlaps,
        'apple_diff_by_dir': apple_diff,
        'q2_hit_dirs': q2_hits,
        'cos_pairs': ['%s-%s %.3f' % p for p in cos_pairs],
        'attr_amp': {'whatkind': round(a_whatkind, 3),
                     'who': round(a_who, 3),
                     'whether': round(a_whether, 3)},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    fc.save(OUT / 'result.json', verdict)
    np.savez(OUT / 'q_spec.npz',
             **{'spec_%s' % c: spec[c] for c in CONDS},
             enc=np.array([enc[c] + [-1] * (8 - len(enc[c]))
                           for c in CONDS]))
    print('P2830 verdict %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k.startswith('Q') and isinstance(v, bool)}), flush=True)
    print('P2831 seconds %.1f' % (time.monotonic() - t0), flush=True)


if __name__ == '__main__':
    main()
