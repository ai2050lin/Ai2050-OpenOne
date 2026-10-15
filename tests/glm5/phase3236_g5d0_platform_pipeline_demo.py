# -*- coding: utf-8 -*-
"""Phase 3236 G5-D0 platform-pipeline DEMO arm (user-directed).

End-to-end worked example of the distributed reverse-analysis pipeline:
task-list entry -> language templates (ZH/EN x statement/question/context)
-> terminal test on qwen3-4b -> stable-feature extraction -> atlas assembly
(link to sealed dR / u* / k* layer curve) -> mechanism readout.

Modes:  P3236_SMOKE=1  -> SMOKE (T1 only, 2 entities)
        default        -> FULL (5 templates, 4 entities x 4 classes)
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913')
NAME = 'g5d0_platform_pipeline_demo'
SMOKE = os.environ.get('P3236_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
OUTDIR = os.path.join(SRC, 'phase3236', NAME)
os.makedirs(OUTDIR, exist_ok=True)

MODEL_DIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
NL, D = 36, 2560
NH = NL + 1
FIN = NH - 1

# frozen sources (sha8)
A3230 = os.path.join(SRC, 'phase3230', 'g5g27_relcontrast_efficacy',
                     'e3_relcontrast_flip.npz')
A3225 = os.path.join(SRC, 'phase3225', 'g5g25_a1_behavioral_band_compute',
                     'a1_behavioral_band.npz')
SHA_3230, SHA_3225 = 'd430d29e', 'ac5d4d74'
LED_N, LED_POST = 387, '26a9fa05'


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(d):
    return hashlib.sha256(json.dumps(
        d, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
        ).hexdigest()[:8]


def log(msg):
    with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'a',
              encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg)


# ---------------- design (frozen before any observation) ----------------
DESIGN = {
    'phase': '3236', 'name': NAME, 'arm': 'platform pipeline demo '
    '(user-directed; landing arm prereg 3236 -> shifted to 3237)',
    'task_list_entry': {
        'fid': 'F-DEMO-ISA-001',
        'feature': '类属关系绑定强度 (isa class-membership binding)',
        'atlas_links': ['dR (3230 relation-contrast axis, 4b)',
                        'u* (3225 collapse carrier, 4b)',
                        'k*=3 class-subspace layer curve (3152)'],
    },
    'entities': {
        'zh': [('苹果', '水果'), ('玫瑰', '花'), ('金鱼', '鱼'), ('锤子', '工具')],
        'en': [('apple', 'fruit'), ('rose', 'flower'),
               ('goldfish', 'fish'), ('hammer', 'tool')],
    },
    'templates': {
        'T1_zh_stmt': dict(kind='class', fmt='{e}是一种'),
        'T2_zh_quest': dict(kind='judge', fmt='{e}是一种{c}吗？答案是'),
        'T3_en_stmt': dict(kind='class', fmt='The {e} is a kind of'),
        'T4_en_quest': dict(kind='judge', fmt='Is the {e} a kind of {c}? '
                                              'Answer:'),
        'T5_zh_ctx': dict(kind='class', fmt='我喜欢{e}。{e}是一种'),
    },
    'margin_def': {
        'class': 'logit(cls_tok) - mean(logit(other 3 cls toks)) at last '
                 'position, final slot',
        'judge': 'logit(yes_tok) - logit(no_tok) at last position',
    },
    'features': ['F1 separation/AUC per template', 'F2 cross-template '
                 'spearman (template robustness = atlas stitch)',
                 'F3 ZH-EN behavior corr + geometry cos (translation axis '
                 'check, 2881 expectation: behavior yes / geometry weak)',
                 'F4 atlas link: cos(v_T, dR), cos(v_T, u*), amplitude vs '
                 'token RMS', 'F5 layer emergence curve (class readout per '
                 'layer, k*=3 expectation)'],
    'demo_gates': {'G1_sep_pos': 'mean margin separation > 0 in all 5 '
                   'templates', 'G2_stitch': 'mean cross-template spearman '
                   '>= 0.6', 'G3_xlang_behavior': 'mean ZH-EN spearman '
                   '>= 0.5', 'G4_layer_report': 'layer curve recorded, '
                   'report-only'},
    'verdict_rule': 'DEMO_PIPELINE_COMPLETE if G1-G3 all pass else '
                    'DEMO_PIPELINE_PARTIAL (demo-level gates, no E-level '
                    'claim)',
    'anchors': {'p3230': SHA_3230, 'p3225': SHA_3225,
                'ledger': [LED_N, LED_POST]},
    'smoke_note': 'SMOKE: T1 only, 2 entities (apple, rose) x 4 classes',
}
assert sha8_file(A3230) == SHA_3230 and sha8_file(A3225) == SHA_3225

fz = dict(DESIGN)
fz['mode'] = 'SMOKE' if SMOKE else 'FULL'
# JSON round-trip normalization (tuples -> lists) so the sha is stable
# across dump/load
fz = json.loads(json.dumps(fz, ensure_ascii=False))
fz['sha8'] = csha8(fz)
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3236 %s %s == freeze %s (before any observation)'
    % (NAME, 'SMOKE' if SMOKE else 'FULL', fz['sha8']))

led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
assert len(led['measurements']) == LED_N and \
    led['ledger_sha256_8'] == LED_POST, 'ledger drift'

# ---------------- tokenizer + data ----------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tk = AutoTokenizer.from_pretrained(MODEL_DIR)

CLS_ZH = ['水果', '花', '鱼', '工具']
CLS_EN = ['fruit', 'flower', 'fish', 'tool']
CLS_TOK = {}
for w in CLS_ZH + CLS_EN:
    ids = tk.encode(w, add_special_tokens=False)
    assert len(ids) == 1, ('class word not single token', w, ids)
    CLS_TOK[w] = ids[0]
JUDG = {'是': None, '不': None, 'Yes': None, 'No': None}
for w in JUDG:
    ids = tk.encode(w, add_special_tokens=False)
    assert len(ids) == 1, ('judge word not single token', w, ids)
    JUDG[w] = ids[0]

if SMOKE:
    ENT = {'zh': [('苹果', '水果'), ('玫瑰', '花')],
           'en': [('apple', 'fruit'), ('rose', 'flower')]}
    TTPL = ['T1_zh_stmt']
else:
    ENT = DESIGN['entities']
    TTPL = sorted(DESIGN['templates'])

pairs = []
for lang in ('zh', 'en'):
    for e, true_c in ENT[lang]:
        cwords = CLS_ZH if lang == 'zh' else CLS_EN
        for c in cwords:
            for tid in TTPL:
                pairs.append((lang, e, c, true_c, tid))
rows = []
for (lang, e, c, true_c, tid) in pairs:
    t = DESIGN['templates'][tid]
    if t['kind'] == 'class':
        prompt = t['fmt'].format(e=e)
        cont = (' ' + c) if lang == 'en' else c
    else:
        prompt = t['fmt'].format(e=e, c=c)
        cont = ('是' if lang == 'zh' else ' Yes')
    ids_p = tk.encode(prompt, add_special_tokens=False)
    ids_pc = tk.encode(prompt + cont, add_special_tokens=False)
    assert ids_pc[:len(ids_p)] == ids_p, ('prefix broken', prompt, cont)
    rows.append(dict(lang=lang, e=e, c=c, true_c=true_c, tpl=tid,
                     kind=t['kind'], ids=ids_pc,
                     n_prompt=len(ids_p)))
log('rows: %d (templates %s, entities %d/lang)'
    % (len(rows), TTPL, len(ENT['zh'])))

# ---------------- sealed atlas directions (4b) ----------------
Z30 = np.load(A3230)
dR = Z30['qwen3-4b_dR_unit'].astype(np.float32)
Z25 = np.load(A3225)
ustar = Z25['qwen3-4b_ustar'].astype(np.float32)
assert dR.shape == (D,) and ustar.shape == (D,)
log('atlas dirs loaded: dR, u* (4b)')

# ---------------- model + forward ----------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
assert model.config.num_hidden_layers == NL

from safetensors import safe_open
with safe_open(os.path.join(MODEL_DIR, 'model-00001-of-00003.safetensors'),
               framework='pt') as f:
    keys = list(f.keys())
    emb_key = 'model.embed_tokens.weight'
    assert emb_key in keys
    need = [CLS_TOK[w] for w in CLS_ZH + CLS_EN] + list(JUDG.values())
    WU = f.get_slice(emb_key)[need].float().numpy()
WU_CLS_ZH = WU[:4]
WU_CLS_EN = WU[4:8]
WU_J = WU[8:]
tok2wu = {}
for i, w in enumerate(CLS_ZH + CLS_EN):
    tok2wu[CLS_TOK[w]] = i
for i, w in enumerate(('是', '不', 'Yes', 'No')):
    tok2wu[JUDG[w]] = 8 + i

hs_all = np.zeros((len(rows), NH, D), np.float16)
fin_logits = np.zeros((len(rows), 151936), np.float32)
with torch.no_grad():
    for i, r in enumerate(rows):
        h = torch.tensor([r['ids']], device='cuda')
        o = model(h, output_hidden_states=True)
        hs_all[i] = np.stack(
            [hh[0, -1, :].float().cpu().numpy() for hh in o.hidden_states]
            ).astype(np.float16)
        fin_logits[i] = o.logits[0, -1].float().cpu().numpy()
        del o
log('forward done: %d rows x %d slots' % (len(rows), NH))

# ---------------- margins ----------------
margins = np.zeros(len(rows), np.float64)
for i, r in enumerate(rows):
    if r['kind'] == 'class':
        cwords = CLS_ZH if r['lang'] == 'zh' else CLS_EN
        toks = [CLS_TOK[w] for w in cwords]
        lg = np.array([fin_logits[i, t] for t in toks])
        j = cwords.index(r['c'])
        margins[i] = lg[j] - lg[np.arange(4) != j].mean()
    else:
        yes = JUDG['是'] if r['lang'] == 'zh' else JUDG['Yes']
        no = JUDG['不'] if r['lang'] == 'zh' else JUDG['No']
        margins[i] = fin_logits[i, yes] - fin_logits[i, no]

npz_payload = dict(rows_lang=np.array([r['lang'] for r in rows]),
                   rows_e=np.array([r['e'] for r in rows]),
                   rows_c=np.array([r['c'] for r in rows]),
                   rows_tc=np.array([r['true_c'] for r in rows]),
                   rows_tpl=np.array([r['tpl'] for r in rows]),
                   rows_kind=np.array([r['kind'] for r in rows]),
                   margins=margins, hs_final=hs_all[:, FIN, :])

# ---------------- F1 separation / AUC ----------------
def auc(pos, neg):
    import itertools
    wins = sum(1 for p in pos for n in neg if p > n) + \
        0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


f1 = {}
for tid in TTPL:
    sel = [i for i, r in enumerate(rows) if r['tpl'] == tid]
    pos = [margins[i] for i in sel if rows[i]['c'] == rows[i]['true_c']]
    neg = [margins[i] for i in sel if rows[i]['c'] != rows[i]['true_c']]
    f1[tid] = dict(n_pos=len(pos), n_neg=len(neg),
                   sep=float(np.mean(pos) - np.mean(neg)),
                   auc=float(auc(pos, neg)),
                   m_pos=float(np.mean(pos)), m_neg=float(np.mean(neg)))
    log('  F1 %s: sep=%.3f auc=%.3f (pos %.2f vs neg %.2f)'
        % (tid, f1[tid]['sep'], f1[tid]['auc'], f1[tid]['m_pos'],
           f1[tid]['m_neg']))

# ---------------- F2 cross-template stitch ----------------
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


vec = {tid: np.array([margins[i] for i, r in enumerate(rows)
                      if r['tpl'] == tid]) for tid in TTPL}
f2 = {}
for a_i in range(len(TTPL)):
    for b_i in range(a_i + 1, len(TTPL)):
        f2['%s|%s' % (TTPL[a_i], TTPL[b_i])] = spearman(
            vec[TTPL[a_i]], vec[TTPL[b_i]])
g2 = float(np.mean(list(f2.values()))) if f2 else float('nan')
for k, v in f2.items():
    log('  F2 %s: rho=%.3f' % (k, v))
log('  F2 mean cross-template spearman = %.3f (gate >= 0.6)' % g2)

# ---------------- F3 ZH-EN ----------------
zh_t = [t for t in TTPL if t.endswith('_stmt') or t.endswith('_quest')
        or t.endswith('_ctx')]
zh_t = [t for t in TTPL if '_zh_' in t]
en_t = [t for t in TTPL if '_en_' in t]
f3 = dict(pairs={})
xrl = []
for a in zh_t:
    for b in en_t:
        v = spearman(vec[a], vec[b])
        f3['pairs']['%s|%s' % (a, b)] = v
        xrl.append(v)
g3 = float(np.mean(xrl)) if xrl else float('nan')
log('  F3 ZH-EN behavior spearman mean = %.3f (gate >= 0.5)' % g3)

# ---------------- F4 atlas links ----------------
f4 = {}
npz_payload['v_dirs'] = {}
for tid in TTPL:
    sel = [i for i, r in enumerate(rows) if r['tpl'] == tid]
    diffs = []
    for lang in ('zh', 'en'):
        ents = ENT[lang]
        for e, tc in ents:
            h_pos = [hs_all[i, FIN, :].astype(np.float32)
                     for i in sel if rows[i]['e'] == e
                     and rows[i]['c'] == tc]
            h_neg = [hs_all[i, FIN, :].astype(np.float32)
                     for i in sel if rows[i]['e'] == e
                     and rows[i]['c'] != tc]
            if h_pos and h_neg:
                diffs.append(np.mean(h_pos, 0) - np.mean(h_neg, 0))
    v = np.mean(diffs, 0)
    v = v / (np.linalg.norm(v) + 1e-12)
    npz_payload['v_dirs'][tid] = v.astype(np.float32)
    rms = float(np.sqrt((hs_all[sel][:, FIN, :].astype(np.float32) ** 2)
                        .mean()))
    f4[tid] = dict(cos_dR=float(v @ dR), cos_ustar=float(v @ ustar),
                   rms=rms)
    log('  F4 %s: cos(v,dR)=%.4f cos(v,u*)=%.4f rms=%.2f'
        % (tid, f4[tid]['cos_dR'], f4[tid]['cos_ustar'], rms))

# ---------------- F5 layer curve (class templates) ----------------
LAYERS = [0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 32, 35]
LAYERS = [L for L in LAYERS if L < NH]
f5 = {}
for tid in TTPL:
    if DESIGN['templates'][tid]['kind'] != 'class':
        continue
    sel = [i for i, r in enumerate(rows) if r['tpl'] == tid]
    curve = {}
    for L in LAYERS:
        pos_m, neg_m = [], []
        for i in sel:
            r = rows[i]
            cwords = CLS_ZH if r['lang'] == 'zh' else CLS_EN
            W = WU_CLS_ZH if r['lang'] == 'zh' else WU_CLS_EN
            hL = hs_all[i, L, :].astype(np.float32)
            lg = W @ hL
            j = cwords.index(r['c'])
            m = lg[j] - lg[np.arange(4) != j].mean()
            (pos_m if r['c'] == r['true_c'] else neg_m).append(m)
        curve[str(L)] = float(np.mean(pos_m) - np.mean(neg_m))
    f5[tid] = curve
    log('  F5 %s: %s' % (tid, {k: round(v, 2) for k, v in curve.items()}))

# ---------------- gates + verdict ----------------
gates = dict(
    G1_sep_pos=all(f1[t]['sep'] > 0 for t in TTPL),
    G2_stitch=bool(g2 >= 0.6),
    G3_xlang_behavior=bool(g3 >= 0.5),
    G4_layer_report=True)
verdict = 'DEMO_PIPELINE_COMPLETE' if (gates['G1_sep_pos'] and
                                       gates['G2_stitch'] and
                                       gates['G3_xlang_behavior']) \
    else 'DEMO_PIPELINE_PARTIAL'
log('gates: %s' % gates)
log('verdict: %s' % verdict)

res = dict(
    phase=3236, name=NAME, mode='SMOKE' if SMOKE else 'FULL',
    freeze_sha8=fz['sha8'],
    task_list_entry=DESIGN['task_list_entry'],
    n_rows=len(rows), templates=TTPL,
    f1=f1, f2=f2, f2_mean=g2, f3=f3, f3_mean=g3,
    f4=f4, f5=f5, gates=gates, verdict=verdict,
    contradictions=[],
    atlas_links_used=dict(dR=SHA_3230, ustar=SHA_3225))
res['res_sha8'] = csha8(dict(res, res_sha8=None))
with open(os.path.join(OUTDIR, 'result.json'), 'w', encoding='utf-8',
          newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
if not SMOKE:
    np.savez_compressed(os.path.join(OUTDIR, 'pipeline_demo.npz'),
                        **npz_payload)
    log('npz sealed: %d rows, v_dirs x%d' % (len(rows), len(npz_payload[
        'v_dirs'])))
log('res %s' % res['res_sha8'])
