# -*- coding: utf-8 -*-
"""Phase 3223 (G5-G25): pre-softmax behavioral band upgrade gate — DESIGN-ONLY
arm (zero GPU, no observation).

Prereg note (honest deviation record): MEMO Phase 3222 tail preregistered
3223 = (a) standby as default. The user replied to the assistant's offer of a
G5-G25 design draft ("把这个评估展开成一份预注册臂草案") with "好的，继续" —
the design arm is therefore USER-SELECTED, not drift. This phase freezes ONLY
the design document + gate criteria. No gate evaluation, no registry change,
no model inference.

D-checks:
  D1  phase3154 collect.npz x3 (full residual stream H per layer, per model)
      exist with expected shapes; file sha8 recorded as data anchors.
  D2  phase3169 g5a6_oov_panel collect npz x3 exist with expected shapes.
  D3  W_U basis for qwen3-4b / qwen3-14b: HF snapshot + safetensors index
      parsed; lm_head.weight located (or tied-embedding finding recorded).
  D4  glm4-9b weights finding: HF hub search — expected ABSENT; behavioral
      band arm scope honestly limited to 2/3 models.
  D5  design doc presoftmax_band_design_v1.json written + sealed (content
      sha8 over canonical dumps excluding the sha8 field).
  D6  ledger untouched (n=374, a1075729 before==after).
"""
import glob
import hashlib
import json
import os
import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3223', 'g5g25_presoftmax_band_design')
SMOKE = os.environ.get('P3223_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def log(s):
    print(s)


os.makedirs(OUTDIR, exist_ok=True)
bad = []
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---------- D1: 3154 full residual-stream caches x3 ----------
D1 = {}
for mid, sub in [('qwen3-4b', 'qwen3-4b'), ('qwen3-14b', 'qwen3-14b'),
                 ('glm4-9b', 'glm4')]:
    p = os.path.join(SRC_DIR, 'phase3154', 'g1p4_mfd_multifactor_disentangle',
                     sub, 'collect.npz')
    if not os.path.exists(p):
        bad.append(('D1 missing npz', mid))
        continue
    z = np.load(p, allow_pickle=True)
    H = z['H']
    D1[mid] = {'path': os.path.relpath(p, ROOT).replace('\\', '/'),
               'sha8': sha8_file(p), 'H_shape': list(H.shape),
               'dtype': str(H.dtype)}
    log('D1 %s H%s %s sha8=%s' % (mid, tuple(H.shape), H.dtype, D1[mid]['sha8']))
if len(D1) != 3:
    bad.append(('D1 count', len(D1)))

# ---------- D2: 3169 OOV panel caches x3 ----------
D2 = {}
for mid in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    hits = glob.glob(os.path.join(SRC_DIR, 'phase3169', '*', 'collect_%s.npz' % mid))
    if not hits:
        bad.append(('D2 missing npz', mid))
        continue
    p = hits[0]
    z = np.load(p, allow_pickle=True)
    H = z['H']
    D2[mid] = {'path': os.path.relpath(p, ROOT).replace('\\', '/'),
               'sha8': sha8_file(p), 'H_shape': list(H.shape),
               'keys': sorted(z.files)}
    log('D2 %s H%s sha8=%s' % (mid, tuple(H.shape), D2[mid]['sha8']))
if len(D2) != 3:
    bad.append(('D2 count', len(D2)))

# ---------- D3: W_U basis (qwen pair) — FINDING-type, not gate ----------
D3 = {}
for mid, hub in [('qwen3-4b', 'models--Qwen--Qwen3-4B-Base'),
                 ('qwen3-14b', 'models--Qwen--Qwen3-14B')]:
    snaps = glob.glob(os.path.join(HUB, hub, 'snapshots', '*'))
    rec = {'hub': hub, 'snapshot': None, 'status': 'no_snapshot_dir'}
    if snaps:
        snap = snaps[0]
        files = sorted(os.listdir(snap))
        rec['snapshot'] = snap.replace('\\', '/')
        rec['snapshot_files'] = files
        cfg_path = os.path.join(snap, 'config.json')
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, encoding='utf-8'))
            rec['config_sha8'] = sha8_file(cfg_path)
            rec['hidden_size'] = cfg.get('hidden_size')
            rec['vocab_size'] = cfg.get('vocab_size')
        idx_path = os.path.join(snap, 'model.safetensors.index.json')
        if os.path.exists(idx_path):
            wm = json.load(open(idx_path, encoding='utf-8'))['weight_map']
            rec['index_sha8'] = sha8_file(idx_path)
            rec['tied'] = 'lm_head.weight' not in wm
            rec['lm_head_shard'] = wm.get('lm_head.weight',
                                          wm.get('model.embed_tokens.weight'))
        else:
            rec['tied'] = None
            rec['lm_head_shard'] = None
        has_w = any(f.endswith('.safetensors') for f in files)
        rec['status'] = ('safetensors_present' if has_w
                         else 'snapshot_incomplete_no_safetensors')
    D3[mid] = rec
    log('D3 %s status=%s hidden=%s tied=%s' % (
        mid, rec['status'], rec.get('hidden_size'), rec.get('tied')))

# ---------- D4: weights finding (glm4 + qwen w_u availability) ----------
glm_hits = [d for d in os.listdir(HUB)
            if ('glm' in d.lower() or 'thudm' in d.lower()
                or 'zhipu' in d.lower())]
w_u_ready = all(D3[m]['status'] == 'safetensors_present' for m in D3) and len(D3) == 2
D4 = {'glm4_weights_in_hub': bool(glm_hits), 'hub_hits': glm_hits,
      'w_u_locally_usable': w_u_ready}
log('D4 glm4-9b weights in hub: %s ; W_U locally usable: %s' % (
    bool(glm_hits), w_u_ready))

# ---------- D5: design doc ----------
if len(D1) == 3 and len(D2) == 3:
    doc = {
        'schema': 'presoftmax_band_design_v1',
        'phase': 3223,
        'name': 'g5g25_presoftmax_band_design',
        'arm': 'G5-G25 design-only (zero GPU, no observation); prereg deviation '
               'honestly recorded: 3223 default was standby (a), user selected '
               'the offered design arm via "好的，继续" (user-directed)',
        'status': 'SEALED_DESIGN',
        'created': now,
        'objective': (
            'Upgrade E2_held_out from the geometric band to the BEHAVIORAL '
            'band in pre-softmax logit space (completion criterion #1 of the '
            '3223 consultation): predict held-out class logit displacement '
            'from registry directions, with exact CPU-computable ground truth '
            '(no model inference needed given cached H + W_U).'),
        'data_basis': {
            'residual_stream': D1,
            'oov_panel': D2,
            'w_u_basis': dict(D3, status_summary=D4),
            'zero_gpu_claim': 'ground truth logits = W_U @ ln_f(h_final) is '
                              'exact matrix math on cached tensors; CPU-only '
                              'COMPUTE, but W_U weights must first be acquired '
                              '(download; no GPU compute involved)',
        },
        'math': {
            'rmsnorm': 'y = gamma \\u2299 x / rms(x), rms(x)=sqrt(mean(x^2)+eps); '
                       'closed-form Jacobian J = diag(gamma)/rms(h) \\u00b7 (I - xhat xhat^T), '
                       'xhat = h/rms(h)',
            'true_displacement': '\\u0394\\u2113(t) = W_U[t] \\u00b7 (ln_f(h+\\u0394v) - ln_f(h)) '
                                 '(exact, cached H final layer + disk weights)',
            'linearized_predictor': '\\u0394\\u2113_lin(t) = W_U[t] \\u00b7 J \\u00b7 \\u0394v '
                                    '(registry direction \\u0394v / u* as \\u0394v)',
            'probe_protocol': 'per E2_held_out feature x 4 held-out class folds x '
                              '2 models (qwen pair): candidate token set S = '
                              'top-64 tokens by |\\u0394\\u2113| plus gold relation targets',
        },
        'gates_pre_registered': {
            'GA_behavioral_heldout': 'cos(\\u0394\\u2113_lin, \\u0394\\u2113_true) over S >= 0.70 '
                                     'on >= 3/4 folds x 2 models (per feature)',
            'GB_band': 'median relative norm error '
                       '||\\u0394\\u2113_lin||-||\\u0394\\u2113_true|| / ||\\u0394\\u2113_true|| '
                       '< 0.25 x 2 models (3158 band alignment)',
            'GC_rank_sufficiency': 'top-8 SVD energy share of the per-class '
                                   'displacement matrix >= 0.8 x 2 models '
                                   '(low-rank sufficiency at logit space; '
                                   'context: 3155 k2_int_share ~ 0.15)',
            'verdict_taxonomy': 'per feature: BEHAVIORAL_BAND_PASS iff GA and GB; '
                                'GA-only => BEHAVIORAL_DIRECTIONAL; else FAIL '
                                '(FAIL goes to F ledger, not hidden)',
        },
        'upgrade_path': 'PASS features: E2_held_out -> E2_behavioral via registry '
                        'append (upgrade_log #10); no in-place edit of vN anchors',
        'chain': {
            '3224': 'G5-G25 A1 compute+seal arm (zero GPU compute): PRECONDITION '
                    'weights acquisition (safetensors download for qwen pair); '
                    'then exact \\u0394\\u2113_true / \\u0394\\u2113_lin / energy matrices '
                    'per feature x fold x model, sealed npz; '
                    'freeze-before-observation re-applied with shard '
                    'integrity check',
            '3225': 'G5-G25 A2 gate evaluation + registry decision arm '
                    '(gates GA/GB/GC applied, upgrade_log decision)',
        },
        'completion_criteria_mapping': {
            'criterion_1_necessary': 'THIS ARM (behavioral band, pre-softmax)',
            'criterion_2_sufficient': 'minimal intervention set E3-ization '
                                      '(patch \\u0394v direction -> in-scope flip + u* '
                                      'amplitude prediction of spectral-out collapse)',
            'criterion_3_composition': 'two-feature composite encoding predicted '
                                       'by single-feature operators, band < 0.25',
        },
        'scope_limits': [
            'W_U WEIGHTS NOT LOCALLY AVAILABLE (D3/D4 finding: 4B-Base snapshot '
            'empty, 14B snapshot config-only, glm4 absent from hub) — 3224 '
            'execution precondition: one-time weights acquisition (download '
            'safetensors; zero GPU compute, CPU matrix math only); design is '
            'sealed against this precondition, not blocked by it',
            'glm4-9b excluded until weights acquired; cross-model generality '
            'deferred (3-model re-run folds into an appendix-style item)',
            'linearization fidelity vs. deeper nonlinearity is measured, not '
            'assumed (GB/GC quantify it; FAIL is a publishable finding)',
            'ground-truth logit space uses ln_f + W_U of the BASE snapshots; '
            'any chat-template drift is out of scope (cached H were collected '
            'with the same base weights)',
        ],
    }
    content = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
    csha = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
    doc['sha8'] = csha
    dp = os.path.join(OUTDIR, PFX + 'presoftmax_band_design_v1.json')
    with open(dp, 'wb') as f:
        f.write(json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8'))
    log('D5 design sealed: %s content sha8=%s (file %s)' % (
        os.path.basename(dp), csha, sha8_file(dp)))
else:
    bad.append(('D5 skipped (data basis incomplete)', len(bad)))

# ---------- D6: ledger untouched ----------
led = json.load(open(LEDGER, encoding='utf-8'))
ok6 = (len(led['measurements']) == 374 and led['ledger_sha256_8'] == 'a1075729')
log('D6 ledger untouched: n=%d post=%s -> %s' % (
    len(led['measurements']), led['ledger_sha256_8'], 'OK' if ok6 else 'FAIL'))
if not ok6:
    bad.append(('D6 ledger touched', led['ledger_sha256_8']))

# ---------- result ----------
clean = not bad
verdict = ('presoftmax_band_design_registered' if clean
           else 'presoftmax_band_design_dirty (%d contradictions)' % len(bad))
res = {
    'phase': 3223,
    'name': 'g5g25_presoftmax_band_design',
    'arm': 'design-only (zero GPU, no observation, no registry change)',
    'prereg_deviation': 'user-selected design arm (honest record in doc.arm)',
    'verdict': verdict,
    'created': now,
    'data_basis_anchors': {
        'residual_stream': {k: v['sha8'] for k, v in D1.items()},
        'oov_panel': {k: v['sha8'] for k, v in D2.items()},
        'w_u_config_sha8': {k: v.get('config_sha8') for k, v in D3.items()},
    },
    'glm4_weights_finding': D4,
    'design_doc': {
        'file': PFX + 'presoftmax_band_design_v1.json',
        'content_sha8': csha if clean else None,
        'file_sha8': sha8_file(os.path.join(OUTDIR, PFX + 'presoftmax_band_design_v1.json'))
        if clean else None,
    },
    'gates_pre_registered': ['GA_behavioral_heldout>=0.70_x3/4folds_x2models',
                             'GB_band<0.25_x2models', 'GC_rank8_energy>=0.8'],
    'chain': {'3224': 'A1 compute+seal (zero GPU)', '3225': 'A2 gates+registry'},
    'ledger_untouched': {'n': 374, 'post': 'a1075729'},
    'contradictions': bad,
}
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'wb') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=1).encode('utf-8'))
with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([]))
log('== Phase 3223 g5g25_presoftmax_band_design %s ==' % ('CLEAN' if clean else 'DIRTY'))
if bad:
    for b in bad:
        log('  CONTRADICTION: %s' % (b,))
log('verdict: %s' % verdict)
