"""Phase 3087 A1: Omega-P84 GLM4-9B layer scan
=========================================================
Menu A of 3086 (continuum confirmed): the fourth
spectral point is GLM4-9B (Zhipu, non-Qwen arch:
GlmForCausalLM, 40L / 4096H / 32 heads / 2 kv,
vocab 151552, head_dim 128, untied), the only
non-Qwen-family checkpoint on disk.  Before the
full four-way arbitration at one layer (A2), the
3083/3084 lesson (family-C degeneration that
turned out to be a LAYER-POSITION effect,
rescued at L28/L34) requires a layer-position
scan on GLM4 first.

Question: over 4 candidate injection layers,
where (if anywhere) is the GLM4 focus machinery
healthy on all three frozen families?

Design (3084-identical pipeline, GLM4-tuned):
- bf16 eager full load on the 16GB card (18.84GB
  allocated, driver sysmem fallback for ~1.75GB,
  measured viable in p3087_bf16_timing: no OOM
  at seq<=74, 0.16-0.59s/forward).
- L_INJ in (31, 34, 37, 38) - depth 0.78 / 0.85
  / 0.93 / 0.95 of 40 layers (proportional
  mapping of the 3B candidates 28/31/33/34 of
  36); L_POST = L_INJ + 1.
- E1 repV ladder (24 pairs, med_c reference,
  b4/b8 per layer), 32-head single swap scan
  (24 pairs) -> r1 = median cos - med_c;
  n_neg from the 3071 criterion; focal top8;
  R_ALL full 32-head swap.
- b7a identity attn self-swap anchor moved to
  the neutral layer 36 (GLM4 has no 3083-style
  degenerate layer; any healthy layer verifies
  the swap mechanism bit-free).

Verdict (preregistered, 3084-identical):
- setup/anchor fail -> setup_failed
- some L in CAND with n_neg >= 8 on ALL three
  families -> layer_rescue (best = argmax_L
  min-family n_neg; ties -> lower L, matching
  the 3084 min-family tie-break)
- else family C n_neg >= 8 at some L ->
  layer_partial
- else -> layer_absent

Output: tests/glm5/result/
rdc_query_construction_20260913/phase3087/
omega_p84_glm4_layer_scan/
"""
import gc
import hashlib
import io
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3087
NAME = 'omega_p84_glm4_layer_scan'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3087', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')

MDIR = os.path.join(ROOT, 'models', 'hf',
                    'glm4-9b-chat-hf')
NL, HID, KV_HEAD, NQ = 40, 4096, 2, 32
HDIM = 128
KVW = KV_HEAD * HDIM
NQW = NQ * HDIM
FRONT = 4
SEED_MAIN = 3087
SEED = 3087
L_CAND = (31, 34, 37, 38)
if SMOKE:
    L_CAND = (38,)
NH = NQ
FKEYS = ('A', 'B', 'C')
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
FAM = {
    'A': {
        'domain': 'everyday-causal (3074 texts '
                  'rerun, bit-anchor family)',
        'bodies': (
            'The weather was cold, so',
            'He studied every night because',
            'The experiment failed, therefore',
            'He missed the train, however',
            'The garden grows quickly while',
            'The price was high, yet',
            'She speaks French, although',
            'The road was closed, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the weather,'),
    },
    'B': {
        'domain': 'science-causal',
        'bodies': (
            'The solution turned acidic, so',
            'The sample was heated because',
            'The catalyst degraded, therefore',
            'The vacuum leaked, however',
            'The crystals formed while',
            'The pressure dropped, yet',
            'The alloy expanded, although',
            'The circuit overheated, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the experiment,'),
    },
    'C': {
        'domain': 'social-emotional-causal',
        'bodies': (
            'She felt deeply betrayed, so',
            'He apologized to her because',
            'They reconciled after the '
            'quarrel, therefore',
            'She stormed out of the room, '
            'however',
            'He listened quietly to every '
            'word while',
            'The gift was cheap and hasty, '
            'yet',
            'She forgave him in the end, '
            'although',
            'The friendship ended without '
            'warning, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the '
                     'conversation,'),
    },
}

PREREG = {
    'mode': 'single model glm4-9b-chat-hf '
            'bf16 eager seed 3087; three frozen '
            '3076 families; banks collected '
            'once per family (layer-'
            'independent), then per candidate '
            'layer L_INJ in (31, 34, 37, 38) '
            'with L_POST=L_INJ+1: E1 repV '
            'ladder (24 pairs, med_c '
            'reference, b4/b8 at that layer), '
            '32-head single swap scan (24 '
            'pairs), focal top8 by the 3071 '
            'criterion, R_ALL full 32-head '
            'swap; E4/cross statistics '
            'omitted (descriptive scan)',
    'question': '3087 A (menu of 3086): does '
                'the fourth spectral point '
                'GLM4-9B (non-Qwen arch, 40L/'
                '4096H/32H/2kv) need a layer '
                'rescue scan before its full '
                'arbitration, and which layer '
                'is its focus-optimal L_INJ?  '
                'Scan the focus machinery over '
                '4 candidate layers and record '
                'per-(layer, family) n_neg / r1 '
                'magnitudes / capture8.',
    'independence': 'the verdict tree '
                    '(layer_rescue / '
                    'layer_partial / '
                    'layer_absent with n_neg>=8 '
                    'thresholds) was frozen '
                    'before any GLM4 observation; '
                    '3B layer-scan results (3084) '
                    'are prior data but concern '
                    'qwen2.5-3b only; GLM4 has no '
                    'prior injections',
    'families': {
        fk: {'domain': FAM[fk]['domain'],
             'bodies': list(FAM[fk]['bodies']),
             'targets': list(TARGETS),
             'prefixes': list(FAM[fk]
                              ['prefixes'])}
        for fk in FKEYS},
    'layer_candidates': 'L_INJ in (31, 34, 37, '
                        '38): depth 0.78/0.85/'
                        '0.93/0.95 of 40 layers '
                        '(proportional mapping of '
                        'the 3B candidates 28/31/'
                        '33/34 of 36); 37/38 are '
                        'the 3rd/2nd from last '
                        '(the 3080/3084 rescue '
                        'band), 31/34 probe the '
                        'mid-deep band; b7a anchor '
                        'at the neutral layer 36; '
                        'L_POST = L_INJ + 1',
    'top8_criterion': 'per (layer, family) '
                      '(3071-identical): order = '
                      'np.argsort(r1) ASCENDING, '
                      'topk = min(8, n_neg), '
                      'top8 = order[:topk]; '
                      'top8_sel_ok = (topk == 8 '
                      'and all r1[top8] < 0)',
    'anchors': {
        'b0': 'bank recapture (4 prompts) bit '
              '0.0 per family',
        'b1': 'sham self-V-replacement bit '
              '0.0 per family',
        'b3': 'all banks finite',
        'b4': 'delta-x at L_INJ bit 0.0 (per '
              'layer)',
        'b6': '(x+a)+m=h2 bf16 identity per '
              'family',
        'b7a': 'identity attn self-swap (all '
               'heads, neutral layer 36) bit '
               '0.0 per family (bank check '
               'only)',
        'b8': 'block-output continuity '
              'zX[L_INJ+1] == zP[L_INJ] bit '
              '0.0 (per layer)',
    },
    'verdict': 'setup/anchor fail -> '
               'setup_failed; some L with '
               'n_neg>=8 on ALL families -> '
               'layer_rescue (best layer = '
               'argmax_L min_family n_neg); '
               'else family C n_neg>=8 at some '
               'L -> layer_partial; else -> '
               'layer_absent',
    'statistics_discipline': 'descriptive '
        'scan: no permutation tests, no gate '
        'on magnitudes; n_neg >= 8 threshold '
        'is the 3071 focal-frame requirement '
        '(frozen), reused verbatim; no '
        'post-hoc model changes',
    'limitations': 'coarse 4-layer grid - '
        'the true loading layer may lie '
        'between candidates; E4 spectra not '
        'collected per layer; descriptive, '
        'no hypothesis tests; same 3076 '
        'texts; tied embeddings model '
        '(recorded)',
    'memory_discipline': 'single model; '
        'families sequential with banks '
        'freed between families; per-layer '
        'arrays small; del + gc + '
        'empty_cache at family end and run '
        'end',
}

os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'run_log.txt',
           'execution.json', 'result.json',
           'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG,
             'smoke': SMOKE}
with io.open(os.path.join(OUT, 'execution.json'),
             'w', encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False,
              indent=1)

lines = []


def log(msg):
    lines.append(str(msg))
    with io.open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


LOG = os.path.join(OUT, 'run_log.txt')
log('execution.json written (prereg frozen) %s '
    'smoke=%s' % (created, SMOKE))
torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


# ==== load model ====
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda') \
    .eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.num_key_value_heads) \
    == KV_HEAD
assert int(model.config.num_attention_heads) \
    == NQ
assert int(model.config.hidden_size) == HID
TIED = bool(getattr(
    model.config, 'tie_word_embeddings',
    False))
log('tie_word_embeddings=%s (tied '
    'embeddings: forwards/logits-only '
    'protocol unaffected, recorded as an '
    'adaptation)' % TIED)
INTER = int(model.config.intermediate_size)
NVOC = int(model.config.vocab_size)
assert NVOC == 151552
log('glm4-9b (glm4-9b-chat-hf) loaded '
    'bf16 (vocab=%d inter=%d layers=%d) '
    'gpu=%.2f GB / total %.2f GB'
    % (NVOC, INTER, NL,
       torch.cuda.memory_allocated() / 1e9,
       torch.cuda.get_device_properties(0)
       .total_memory / 1e9))

FW = [0]
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
stateATN = {li: {'repl': None, 'mask': None}
            for li in range(NL)}
capV = {li: {'rec': False, 'orig': None}
        for li in range(NL)}
capP = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capX = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capA = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capM = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capH = {li: {'rec': False, 'v': None}
        for li in range(NL)}


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def hook_post(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = out[0].detach().clone()
        return out
    return h


def hook_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] \
                if isinstance(out, tuple) else out
            cp['v'] = t[0, -1].detach().clone()
        return out
    return h


def hook_in_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = inp[0][0, -1] \
                .detach().clone()
        return out
    return h


def hook_pre_last(st):
    def h(module, args):
        if st['repl'] is None:
            return None
        a = args[0].clone()
        a[0, -1, st['mask']] = st['repl']
        return (a,)
    return h


for li in range(NL):
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].register_forward_hook(hook_post(
        capP[li]))
    layers[li].register_forward_hook(hook_in_last(
        capX[li]))
    layers[li].self_attn \
        .register_forward_hook(hook_last(capA[li]))
    layers[li].mlp \
        .register_forward_hook(hook_last(capM[li]))
    layers[li].self_attn.o_proj \
        .register_forward_hook(hook_in_last(
            capH[li]))
    layers[li].self_attn.o_proj \
        .register_forward_pre_hook(hook_pre_last(
            stateATN[li]))


def reset_all():
    for li in range(NL):
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capV[li]['rec'] = False
        capV[li]['orig'] = None
        capP[li]['rec'] = False
        capP[li]['v'] = None
        capX[li]['rec'] = False
        capX[li]['v'] = None
        capA[li]['rec'] = False
        capA[li]['v'] = None
        capM[li]['rec'] = False
        capM[li]['v'] = None
        capH[li]['rec'] = False
        capH[li]['v'] = None
        stateATN[li]['repl'] = None
        stateATN[li]['mask'] = None


def forward_gen(ids, repl=None, attn_swaps=None):
    reset_all()
    FW[0] += 1
    m_all = torch.ones(
        len(ids), dtype=torch.bool,
        device='cuda')
    if repl is not None:
        rt = torch.tensor(
            np.ascontiguousarray(repl),
            dtype=torch.bfloat16,
            device='cuda')
        for li in range(NL):
            stateV[li]['repl'] = rt[li]
            stateV[li]['mask'] = m_all
    if attn_swaps is not None:
        for li_, midx, mval in attn_swaps:
            stateATN[li_]['mask'] = torch.tensor(
                np.ascontiguousarray(midx),
                dtype=torch.long,
                device='cuda')
            stateATN[li_]['repl'] = torch.tensor(
                np.ascontiguousarray(mval),
                dtype=torch.bfloat16,
                device='cuda')
    for li in range(NL):
        capV[li]['rec'] = True
        capP[li]['rec'] = True
        capX[li]['rec'] = True
        capA[li]['rec'] = True
        capM[li]['rec'] = True
        capH[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor(
            [ids], device='cuda'),
            use_cache=False)
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    vb = np.stack([capV[li]['orig'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    po = np.stack([capP[li]['v'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    zX = torch.stack([capX[li]['v']
                      for li in range(NL)])
    zA = torch.stack([capA[li]['v']
                      for li in range(NL)])
    zM = torch.stack([capM[li]['v']
                      for li in range(NL)])
    zH = torch.stack([capH[li]['v']
                      for li in range(NL)])
    reset_all()
    return lg, vb, po, zX.double().cpu() \
        .numpy(), zA.double().cpu().numpy(), \
        zM.double().cpu().numpy(), \
        zH.double().cpu().numpy()


ridx = np.arange(FRONT)
ALLH = np.arange(NQW)
HEAD_IDX = [np.arange(h * HDIM, (h + 1) * HDIM)
            for h in range(NH)]
NP_ = 24
NP_USE = 8 if SMOKE else NP_
K3 = 4 if SMOKE else NP_

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(8):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)


def run_family_banks(fkey):
    fam = FAM[fkey]
    bodies = fam['bodies']
    prefixes = fam['prefixes']
    log('[%s] ==== family begin (%s) ===='
        % (fkey, fam['domain']))
    word_tok = {}
    for w in TARGETS:
        wi = tok(' ' + w,
                 add_special_tokens=False)[
            'input_ids']
        assert len(wi) == 1, (fkey, w, wi)
        word_tok[w] = int(wi[0])
    assembled = []
    for bi in range(len(bodies)):
        for ci in range(len(prefixes)):
            s = (prefixes[ci] + ' '
                 + bodies[bi]) if prefixes[ci] \
                else bodies[bi]
            ids = [int(x) for x in tok(
                s, add_special_tokens=False)[
                'input_ids']]
            t = word_tok[TARGETS[bi]]
            assert ids.count(t) == 1, \
                (fkey, bi, ci)
            assembled.append(
                {'ids': ids, 'cond': ci,
                 'body': bi})
    n_pr = len(assembled)
    assert n_pr == 32
    idx_of = {}
    for i in range(n_pr):
        idx_of[(assembled[i]['cond'],
                assembled[i]['body'])] = i
    for i in range(n_pr):
        ci = assembled[i]['cond']
        if ci == 0:
            assembled[i]['off'] = 0
        else:
            bid = assembled[idx_of[(0,
                assembled[i]['body'])]]['ids']
            pid = assembled[i]['ids']
            off = len(pid) - len(bid)
            assert off > 0, (fkey, i)
            assert list(pid[off + 1:]) \
                == list(bid[1:]), (fkey, i)
            w0b = tok.decode([bid[0]]).strip()
            w0p = tok.decode(
                [pid[off]]).strip()
            assert w0b == w0p, (fkey, i)
            assembled[i]['off'] = off
    LENS = np.array([len(assembled[i]['ids'])
                     for i in range(n_pr)])
    NMAX = int(LENS.max())
    log('[%s] assembled 32 prompts (lens %d-%d)'
        % (fkey, int(LENS.min()), NMAX))

    def pair_idx(k):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        pref_i = idx_of[(c, b)]
        off = assembled[pref_i]['off']
        return b, base_i, pref_i, off

    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        assert int(LENS[base_i]) >= FRONT, \
            (fkey, k, int(LENS[base_i]))

    LG = np.zeros((n_pr, NVOC))
    VB = np.zeros((n_pr, NL, NMAX, KVW))
    PB = np.zeros((n_pr, NL, NMAX, HID))
    BX = np.zeros((n_pr, NL, HID))
    BA = np.zeros((n_pr, NL, HID))
    BM = np.zeros((n_pr, NL, HID))
    BH = np.zeros((n_pr, NL, NQW))
    for i in range(n_pr):
        lg, vb, po, zX, zA, zM, zH = \
            forward_gen(assembled[i]['ids'])
        n = int(LENS[i])
        if i == 0:
            assert vb.shape == (NL, n, KVW), \
                vb.shape
            assert po.shape == (NL, n, HID), \
                po.shape
        LG[i] = lg
        VB[i, :, :n, :] = vb
        PB[i, :, :n, :] = po
        BX[i] = zX
        BA[i] = zA
        BM[i] = zM
        BH[i] = zH
    log('[%s] banks: LG%s VB%s PB%s BH%s '
        '(forwards=%d)'
        % (fkey, LG.shape, VB.shape, PB.shape,
           BH.shape, FW[0]))

    b0_diff = 0.0
    b0_samples = (0,) if SMOKE \
        else (0, 9, 17, 31)
    for si in b0_samples:
        lg, vb, po, zX, zA, zM, zH = \
            forward_gen(assembled[si]['ids'])
        n2 = int(LENS[si])
        b0_diff = max(b0_diff, float(np.max(
            np.abs(LG[si] - lg))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(VB[si, :, :n2, :] - vb))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(PB[si, :, :n2, :] - po))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BX[si] - zX))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BA[si] - zA))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BM[si] - zM))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BH[si] - zH))))
    b0_diff = float(b0_diff)
    b0_ok = bool(b0_diff == 0.0)
    log('[%s] b0 recapture diff=%.3e ok=%s'
        % (fkey, b0_diff, b0_ok))

    TT = np.stack([
        LG[idx_of[(int(cidx[k]),
                   int(bidx[k]))]]
        - LG[idx_of[(0, int(bidx[k]))]]
        for k in range(NP_)])
    TT32 = TT.astype(np.float32)

    base0 = idx_of[(0, int(bidx[0]))]
    n0 = int(LENS[base0])
    selfV = VB[base0, :, :n0, :].copy()
    lg_s = forward_gen(
        assembled[base0]['ids'], repl=selfV)[0]
    b1_diff = float(np.max(np.abs(lg_s
                                  - LG[base0])))
    b1_ok = bool(b1_diff == 0.0)
    log('[%s] b1 sham self-replacement diff='
        '%.3e ok=%s' % (fkey, b1_diff, b1_ok))

    b3_ok = bool(np.isfinite(LG).all()
                 and np.isfinite(VB).all()
                 and np.isfinite(PB).all()
                 and np.isfinite(TT).all()
                 and np.isfinite(BX).all()
                 and np.isfinite(BA).all()
                 and np.isfinite(BM).all()
                 and np.isfinite(BH).all())
    log('[%s] b3 finite=%s' % (fkey, b3_ok))

    b6_diff = 0.0
    for i in range(n_pr):
        n = int(LENS[i])
        x_ = torch.tensor(BX[i],
                          device='cuda') \
            .to(torch.bfloat16)
        a_ = torch.tensor(BA[i],
                          device='cuda') \
            .to(torch.bfloat16)
        m_ = torch.tensor(BM[i],
                          device='cuda') \
            .to(torch.bfloat16)
        p_ = torch.tensor(PB[i, :, n - 1, :],
                          device='cuda') \
            .to(torch.bfloat16)
        h2 = (x_ + a_) + m_
        b6_diff = max(b6_diff, float(
            (h2 - p_).abs().max()))
    b6_diff = float(b6_diff)
    b6_ok = bool(b6_diff == 0.0)
    log('[%s] b6 (x+a)+m=h2 bf16 diff=%.3e '
        'ok=%s' % (fkey, b6_diff, b6_ok))

    # b7a identity attn self-swap at neutral layer 36
    # (bank sanity; layer-independent in
    # nature)
    b, base_i, pref_i, off = pair_idx(0)
    lg7a = forward_gen(
        assembled[base_i]['ids'],
        attn_swaps=[(36, ALLH,
                     BH[base_i, 36])])[0]
    b7a_diff = float(np.max(np.abs(
        lg7a - LG[base_i])))
    b7a_ok = bool(b7a_diff == 0.0)
    log('[%s] b7a identity attn self-swap '
        'diff=%.3e ok=%s'
        % (fkey, b7a_diff, b7a_ok))

    return {
        'fkey': fkey,
        'domain': fam['domain'],
        'assembled': assembled,
        'idx_of': idx_of,
        'LENS': LENS,
        'LG': LG, 'VB': VB, 'PB': PB,
        'BX': BX, 'BA': BA, 'BM': BM,
        'BH': BH,
        'TT32': TT32,
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'b1_diff': b1_diff, 'b1_ok': b1_ok,
        'b3_ok': b3_ok,
        'b6_diff': b6_diff, 'b6_ok': b6_ok,
        'b7a_diff': b7a_diff,
        'b7a_ok': b7a_ok,
    }


def run_layer(B, L_INJ):
    """E1 ladder + b4/b8 + E3 head scan +
    top8 + R_ALL at one injection layer."""
    fkey = B['fkey']
    assembled = B['assembled']
    idx_of = B['idx_of']
    LENS = B['LENS']
    LG = B['LG']
    VB = B['VB']
    BX = B['BX']
    BH = B['BH']
    TT = B['TT32'].astype(np.float64)
    L_POST = L_INJ + 1

    def pair_idx(k):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        pref_i = idx_of[(c, b)]
        off = assembled[pref_i]['off']
        return b, base_i, pref_i, off

    def repv_of(k):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[L_INJ, ridx, :] = VB[pref_i][
            L_INJ, off + ridx, :]
        return base_i, repV

    def run_cond(mk_kwargs):
        cs = np.full(K3, np.nan)
        for k in range(K3):
            base_i, repV = repv_of(k)
            lg = forward_gen(
                assembled[base_i]['ids'],
                repl=repV,
                **mk_kwargs(base_i))[0]
            cs[k] = cosv(lg - LG[base_i],
                         TT[k])
        return cs

    COS_LAD = np.full(NP_USE, np.nan)
    b4_diff = 0.0
    b8_diff = 0.0
    for k in range(NP_USE):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        base_i, repV = repv_of(k)
        lg, _, po_i, zX, _, _, _ = \
            forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        dlg = lg - LG[base_i]
        COS_LAD[k] = cosv(dlg, TT[k])
        b4_diff = max(b4_diff, float(np.max(
            np.abs(zX[L_INJ] - BX[base_i,
                                   L_INJ]))))
        b8_diff = max(b8_diff, float(np.max(
            np.abs(zX[L_POST]
                   - po_i[L_INJ, nb - 1]))))
    b4_ok = bool(b4_diff == 0.0)
    b8_ok = bool(b8_diff == 0.0)
    med_c = float(np.median(COS_LAD))
    log('[%s] L%d: E1 med_c=%.4f b4 ok=%s '
        'b8 ok=%s (forwards=%d)'
        % (fkey, L_INJ, med_c, b4_ok, b8_ok,
           FW[0]))

    CS1H = np.full((NH, K3), np.nan)
    for h in range(NH):
        idx = HEAD_IDX[h]
        CS1H[h] = run_cond(
            lambda bi, idx=idx: {
                'attn_swaps': [
                    (L_INJ, idx,
                     BH[bi, L_INJ][idx])]})
    r1_nh = np.median(CS1H, axis=1) - med_c
    n_neg = int((r1_nh < 0).sum())
    order = np.argsort(r1_nh)
    topk = min(8, n_neg)
    top8 = [int(h) for h in order[:topk]]
    top8_sel_ok = bool(
        topk == 8
        and bool((r1_nh[top8] < 0).all()))
    neg_sum = float(r1_nh[r1_nh < 0].sum())
    capture8 = abs(float(
        r1_nh[top8].sum())) \
        / abs(neg_sum) if abs(neg_sum) > 1e-9 \
        else float('nan')
    med_neg_r1 = float(np.median(
        r1_nh[r1_nh < 0])) if n_neg > 0 \
        else float('nan')
    log('[%s] L%d: E3 r1 min=%.4f n_neg=%d '
        'top8=%s capture8=%.4f sel_ok=%s '
        'med_neg=%.4f'
        % (fkey, L_INJ, float(r1_nh.min()),
           n_neg, top8, capture8,
           top8_sel_ok, med_neg_r1))

    lg_all = run_cond(lambda bi: {
        'attn_swaps': [(L_INJ, ALLH,
                        BH[bi, L_INJ])]})
    R_ALL = float(np.median(lg_all)
                  - med_c)
    log('[%s] L%d: full 32-head swap '
        'recov=%+.4f (forwards=%d)'
        % (fkey, L_INJ, R_ALL, FW[0]))

    return {
        'L_INJ': L_INJ,
        'med_c': med_c,
        'COS_LAD': COS_LAD,
        'CS1H': CS1H,
        'r1_nh': r1_nh,
        'n_neg': n_neg, 'topk': topk,
        'top8': top8,
        'top8_sel_ok': top8_sel_ok,
        'capture8': capture8,
        'med_neg_r1': med_neg_r1,
        'R_ALL': R_ALL,
        'b4_diff': b4_diff, 'b4_ok': b4_ok,
        'b8_diff': b8_diff, 'b8_ok': b8_ok,
    }


RES = {}
FAM_ANCH = {}
for fk in FKEYS:
    B = run_family_banks(fk)
    FAM_ANCH[fk] = {
        'b0_diff': B['b0_diff'],
        'b0_ok': B['b0_ok'],
        'b1_diff': B['b1_diff'],
        'b1_ok': B['b1_ok'],
        'b3_ok': B['b3_ok'],
        'b6_diff': B['b6_diff'],
        'b6_ok': B['b6_ok'],
        'b7a_diff': B['b7a_diff'],
        'b7a_ok': B['b7a_ok'],
    }
    for L in L_CAND:
        RES[(L, fk)] = run_layer(B, L)
    del B
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] banks freed (forwards=%d)'
        % (fk, FW[0]))

# ==== setup ok & verdict ====
setup_ok_all = bool(
    all(v[k]
        for v in FAM_ANCH.values()
        for k in v if k.endswith('_ok'))
    and all(RES[k]['b4_ok']
            and RES[k]['b8_ok']
            for k in RES))
log('setup_ok_all=%s (family anchors + '
    'per-layer b4/b8)' % setup_ok_all)

if not setup_ok_all:
    verdict = 'setup_failed'
    rescue = []
    best = None
    log('VERDICT: %s' % verdict)
elif SMOKE:
    verdict = 'smoke_pending'
    rescue = []
    best = None
    log('verdict skipped (smoke)')
    log('VERDICT: smoke_pending')
else:
    NN = {(L, f): RES[(L, f)]['n_neg']
          for L in L_CAND for f in FKEYS}
    for L in L_CAND:
        log('L%d n_neg: A=%d B=%d C=%d | '
            'med_c %.4f/%.4f/%.4f | R_ALL '
            '%+.4f/%+.4f/%+.4f'
            % (L, NN[(L, 'A')], NN[(L, 'B')],
               NN[(L, 'C')],
               RES[(L, 'A')]['med_c'],
               RES[(L, 'B')]['med_c'],
               RES[(L, 'C')]['med_c'],
               RES[(L, 'A')]['R_ALL'],
               RES[(L, 'B')]['R_ALL'],
               RES[(L, 'C')]['R_ALL']))
    rescue = [L for L in L_CAND
              if all(NN[(L, f)] >= 8
                     for f in FKEYS)]
    if rescue:
        verdict = 'layer_rescue'
        best = max(rescue,
                   key=lambda L: min(
                       NN[(L, f)]
                       for f in FKEYS))
        log('RESCUE layers: %s (best=%d '
            'min_family_n_neg=%d)'
            % (rescue, best,
               min(NN[(best, f)]
                   for f in FKEYS)))
    elif any(NN[(L, 'C')] >= 8
             for L in L_CAND):
        verdict = 'layer_partial'
        best = None
        log('PARTIAL: family C recovers at '
            'some layer but not all three '
            'families anywhere')
    else:
        verdict = 'layer_absent'
        best = None
        log('ABSENT: no candidate layer '
            'gives family C n_neg >= 8')
    log('VERDICT: %s' % verdict)

# ==== npz ====
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'FORWARDS': np.int64(FW[0]),
    'TIED': np.bool_(TIED),
    'L_CAND': np.array(L_CAND,
                       dtype=np.int64),
    'B7A_LAYER': np.int64(36),
    'SETUP_OK': np.bool_(setup_ok_all),
}
for fk in FKEYS:
    a = FAM_ANCH[fk]
    save['B0_OK_' + fk] = np.bool_(a['b0_ok'])
    save['B0_DIFF_' + fk] = np.float64(
        a['b0_diff'])
    save['B1_OK_' + fk] = np.bool_(a['b1_ok'])
    save['B1_DIFF_' + fk] = np.float64(
        a['b1_diff'])
    save['B3_OK_' + fk] = np.bool_(a['b3_ok'])
    save['B6_OK_' + fk] = np.bool_(a['b6_ok'])
    save['B6_DIFF_' + fk] = np.float64(
        a['b6_diff'])
    save['B7A_OK_' + fk] = np.bool_(
        a['b7a_ok'])
    save['B7A_DIFF_' + fk] = np.float64(
        a['b7a_diff'])
for (L, fk), R in sorted(RES.items()):
    pfx = 'L%d_%s' % (L, fk)
    save['MED_C_' + pfx] = np.float64(
        R['med_c'])
    save['COS_LAD_' + pfx] = R['COS_LAD']
    save['CS1H_' + pfx] = R['CS1H']
    save['R1_NH_' + pfx] = R['r1_nh']
    save['TOP8_' + pfx] = np.array(
        R['top8'], dtype=np.int64)
    save['N_NEG_' + pfx] = np.int64(
        R['n_neg'])
    save['CAPTURE8_' + pfx] = np.float64(
        R['capture8'])
    save['TOP8_SEL_OK_' + pfx] = np.bool_(
        R['top8_sel_ok'])
    save['MED_NEG_R1_' + pfx] = np.float64(
        R['med_neg_r1'])
    save['R_ALL_' + pfx] = np.float64(
        R['R_ALL'])
    save['B4_DIFF_' + pfx] = np.float64(
        R['b4_diff'])
    save['B4_OK_' + pfx] = np.bool_(
        R['b4_ok'])
    save['B8_DIFF_' + pfx] = np.float64(
        R['b8_diff'])
    save['B8_OK_' + pfx] = np.bool_(
        R['b8_ok'])
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ==== result.json ====
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


fam_layers = {}
for fk in FKEYS:
    fam_layers[fk] = {}
    for L in L_CAND:
        R = RES[(L, fk)]
        fam_layers[fk][str(L)] = {
            'med_c': f64(R['med_c']),
            'n_neg': int(R['n_neg']),
            'top8': R['top8'],
            'top8_sel_ok': bool(
                R['top8_sel_ok']),
            'capture8': f64(R['capture8']),
            'med_neg_r1': f64(
                R['med_neg_r1']),
            'r_all': f64(R['R_ALL']),
            'b4_ok': bool(R['b4_ok']),
            'b8_ok': bool(R['b8_ok']),
        }
rescue_best = None
if rescue:
    rescue_best = int(max(rescue,
                          key=lambda L: min(
                              RES[(L, f)]
                              ['n_neg']
                              for f in FKEYS)))
result = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'elapsed': time.time() - t0,
    'forwards': int(FW[0]),
    'run': 'run1 authoritative (qwen2.5-3b '
           'bf16 single model, layer scan '
           'over %s)' % (list(L_CAND),)
    if not SMOKE else 'smoke',
    'prereg': PREREG,
    'stats': {
        'families_layers': fam_layers,
        'fam_anchors': {
            fk: {k: (bool(v) if k.endswith(
                '_ok') else f64(v))
                for k, v
                in FAM_ANCH[fk].items()}
            for fk in FKEYS},
        'setup_ok_all': setup_ok_all,
        'rescue_layers': [int(L)
                          for L in rescue],
        'rescue_best': rescue_best,
    },
    'verdict': verdict,
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)

seal = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(
        os.path.join(OUT, 'result.json')),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(
        __file__)),
    'verdict': verdict,
    'setup_ok': setup_ok_all,
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'],
       time.time() - t0))
log('sealed')
print('RUN_COMPLETE %s' % verdict)

del model, layers, tok
gc.collect()
torch.cuda.empty_cache()
