# -*- coding: utf-8 -*-
"""Phase 3235: G5-G30 gate evaluation + registry/gap decision arm.
Zero GPU — pure arithmetic on the sealed 3234 npz + result.json.
Gates frozen in 3233 design doc (g5g30_upstream_lever_design_v1.json).
Chain: FOUND -> registry upgrade_log; PARTIAL -> F-ledger note;
ABSENT -> F-ledger append + GAP-5 mechanism_note (gap v1.10 -> v1.11).
"""
import os
import sys
import json
import glob
import time
import hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
NAME = 'g5g31_gap5_gate_decision'
OUTDIR = os.path.join(SRC_DIR, 'phase3235', NAME)
SMOKE = os.environ.get('P3235_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
MODELS = ['qwen3-4b', 'qwen3-14b']
D34 = os.path.join(SRC_DIR, 'phase3234', 'g5g30_gap5_upstream_lever_scan')

LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


os.makedirs(OUTDIR, exist_ok=True)

# ---------- freeze (before any gate arithmetic) ----------
res34 = json.load(open(os.path.join(D34, 'result.json'), encoding='utf-8'))
npz34_path = os.path.join(D34, res34['sealed_npz']['file'])
led0 = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                   'atlas_ledger.json'), encoding='utf-8'))
assert any(e.get('phase') == 3234 for e in led0['measurements']), \
    '3234 not in ledger yet — run 3234 closeout first'
ANCHORS = {
    'res34_sha8': res34['res_sha8'],
    'npz34_file_sha8': sha8_file(npz34_path),
    'res34_freeze': res34['freeze_sha8'],
    'ledger_n': len(led0['measurements']),
    'ledger_post': led0['ledger_sha256_8'],
    'registry_v1_31': sha8_file(os.path.join(
        SRC_DIR, r'phase3231\g5g28_registry_decision'
                 r'\atlas_registry_v1_31.json')),
    'gap_v1_10': sha8_file(os.path.join(
        SRC_DIR, r'phase3232\g5g29_gd_amp_class_robustness'
                 r'\gap_ledger_v1_10.json')),
    'design_3233': 'd74b6484',
    'UL_flip': 0.15, 'UL_spec': 1.0 / 3.0, 'UL_amp': 0.6,
    'kstar_band': [1, 5],
    'null_seeds': [20260400 + i for i in range(10)],
}
fz = {
    'phase': 3235, 'name': NAME, 'created': time.strftime('%Y-%m-%d %H:%M'),
    'design_basis': '3233 g5g30_upstream_lever_design_v1 (d74b6484) '
                    'gates_pre_registered verbatim',
    'gate_operationalization': {
        'UL_flip': 'per direction d in {u_isa,u_hasa} per model: for each '
                   'fold f (4 ci-folds of 3154 seen rows): '
                   'maxL flip_rate(f,L) - max over null dirs (10 rand '
                   'norm-matched + orth) flip_rate(f) at the SAME argmax '
                   'layer >= 0.15; pass iff >= 3/4 folds x 2 models. '
                   'argmax layer = pooled (sealed 3234 pooled[d].Lstar). '
                   'sensitivity variant (recorded, not gating): per-fold '
                   'rate at pooled L* instead of per-fold max over L',
        'UL_dose': 'at pooled L* (band-clipped): fine flip rate monotone '
                   'non-decreasing over alpha {0.5,1,2} (3169 t=0 730 rows) '
                   'x 2 models',
        'UL_spec': 'at pooled L*: cross-patch flip rate (patch u_isa -> '
                   'count J0=1->J1=0; patch u_hasa -> count J0=0->J1=1) '
                   '<= (1/3) x matched flip rate, from the SAME coarse J1 '
                   'vector x 2 models',
        'UL_amp': 'max over sealed u* band layers of spearman_4oov >= 0.6 '
                  'x 2 models (n=4 small-n scope)',
        'UL_loc': 'REPORT-ONLY: pooled L* per direction per model vs '
                  '[kstar-2, kstar+2] = [1, 5] (3152 k*=3)',
    },
    'verdict_taxonomy': 'UPSTREAM_LEVER_FOUND iff UL_flip and UL_dose and '
                        'UL_spec (>= 2 models); UPSTREAM_PARTIAL if UL_flip '
                        'passes but dose/spec fail; UPSTREAM_ABSENT if '
                        'UL_flip fails (=> F-ledger F20-family extension + '
                        'GAP-5 mechanism_note append)',
    'registry_gap_plan': {
        'FOUND': 'registry v1.31 -> v1.32 (new FTR-27 upstream lever, '
                 'upgrade_log #12)',
        'PARTIAL': 'F-ledger note append (registry v1.32, no E-level '
                   'change) + GAP-5 note',
        'ABSENT': 'F-ledger append (F21) + GAP-5 mechanism_note append '
                  '(gap v1.10 -> v1.11); registry v1.31 UNTOUCHED',
    },
    'clarifications': {
        'C1': 'zero GPU: all gate inputs are sealed 3234 matrices '
              '(coarse J1, fine J1+marg, u* marg, nulls J1); cross-patch '
              'is computable from the same J1 (no extra forwards)',
        'C2': 'matched flip for u_isa = (J0==0)&(J1==1); cross = '
              '(J0==1)&(J1==0); mirrored for u_hasa (J1 = judged isa)',
        'C3': 'nulls exist only at peak layers (sealed); null comparison '
              'restricted to those layers (recorded scope)',
        'C4': 'glm4 out of scope (3234 sealed qwen pair only)',
        'C5': 'UL_flip primary = per-fold max over L (design text verbatim); '
              'pooled-L* variant recorded as sensitivity',
    },
    'anchors': ANCHORS,
    'smoke_note': 'SMOKE: qwen3-4b only, gates computed, no registry/gap '
                  'writes' if SMOKE else 'FULL',
}
fz['sha8'] = csha8(dict(fz))
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3235 %s %s == freeze %s (before any gate arithmetic)'
    % (NAME, 'SMOKE' if SMOKE else 'FULL', fz['sha8']))
assert sha8_file(npz34_path) == res34['sealed_npz']['file_sha8']
log('3234 npz seal OK (%s)' % ANCHORS['npz34_file_sha8'])

# ---------- load sealed 3234 data ----------
NPZ = np.load(npz34_path, allow_pickle=True)
Z30 = np.load(os.path.join(SRC_DIR, r'phase3230\g5g27_relcontrast_efficacy'
                           r'\e3_relcontrast_flip.npz'))
MODELS_RUN = ['qwen3-4b'] if SMOKE else MODELS
NLL = {'qwen3-4b': 36, 'qwen3-14b': 40}
out = {}
bad = []

for m in MODELS_RUN:
    log('=== %s ===' % m)
    NL = NLL[m]
    z54 = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                               m, 'collect.npz'))
    tags54 = {k: z54[k].astype(int) for k in ('ci', 'ho')}
    del z54
    folds = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0]
             for c in range(4)]
    J0_54 = NPZ[m + '_coarse_J0'].astype(np.int8)

    rec = {}
    # pooled argmax layers (sealed)
    Lstar = {d: int(res34['models'][m]['pooled'][d]['Lstar'])
             for d in ('u_isa', 'u_hasa')}
    rec['Lstar'] = Lstar

    # nulls at peak layers (sealed J1)
    def flip_rate(J1, J0, sl, mode):
        # mode 'isa': (J0==0)&(J1==1); 'hasa': (J0==1)&(J1==0)
        if mode == 'isa':
            fl = (J0[sl] == 0) & (J1[sl] == 1)
        else:
            fl = (J0[sl] == 1) & (J1[sl] == 0)
        return float(fl.sum() / len(sl))

    null_fr = {}
    for dname in ('u_isa', 'u_hasa'):
        L = Lstar[dname]
        mode = 'isa' if dname == 'u_isa' else 'hasa'
        frs = []
        for nd in (['orth'] + ['rand_%d' % s for s in ANCHORS['null_seeds']]):
            key = '%s_null_%s_L%d_J1' % (m, nd, L)
            if key not in NPZ:
                bad.append('missing null key %s' % key)
                continue
            J1n = NPZ[key].astype(np.int8)
            frs.append([flip_rate(J1n, J0_54, sl, mode) for sl in folds])
        null_fr[dname] = np.array(frs) if frs else None  # (n_null, 4)
    rec['null_max_per_fold'] = {d: (None if null_fr[d] is None else
                                    [float(x) for x in null_fr[d].max(axis=0)])
                                for d in null_fr}

    # UL_flip: per-fold max over L
    LGRID = sorted(int(k.split('|L')[1]) for k in res34['models'][m]['coarse']
                   if k.startswith('u_isa|L'))
    ul_flip = {}
    for dname in ('u_isa', 'u_hasa'):
        mode = 'isa' if dname == 'u_isa' else 'hasa'
        per_fold_max = []
        per_fold_at_L = []
        for sl in folds:
            rates = []
            for L in LGRID:
                J1 = NPZ['%s_%s_J1_L%d' % (m, dname, L)].astype(np.int8)
                rates.append(flip_rate(J1, J0_54, sl, mode))
            per_fold_max.append(max(rates))
            J1L = NPZ['%s_%s_J1_L%d' % (m, dname, Lstar[dname])].astype(np.int8)
            per_fold_at_L.append(flip_rate(J1L, J0_54, sl, mode))
        nmax = null_fr[dname].max(axis=0)
        d_f = [per_fold_max[f] - float(nmax[f]) for f in range(4)]
        d_L = [per_fold_at_L[f] - float(nmax[f]) for f in range(4)]
        n_pass_primary = sum(1 for x in d_f if x >= ANCHORS['UL_flip'])
        n_pass_sens = sum(1 for x in d_L if x >= ANCHORS['UL_flip'])
        ul_flip[dname] = dict(
            per_fold_max_L=[float(x) for x in per_fold_max],
            per_fold_at_Lstar=[float(x) for x in per_fold_at_L],
            null_max=[float(x) for x in nmax],
            delta_primary=[float(x) for x in d_f],
            delta_sensitivity=[float(x) for x in d_L],
            n_folds_pass_primary=n_pass_primary,
            n_folds_pass_sensitivity=n_pass_sens,
            pass_primary=bool(n_pass_primary >= 3),
            pass_sensitivity=bool(n_pass_sens >= 3))
        log('  UL_flip %s: primary d=%s pass=%d/4 | sensitivity d=%s'
            % (dname, ['%.4f' % x for x in d_f], n_pass_primary,
               ['%.4f' % x for x in d_L]))
    rec['ul_flip'] = ul_flip

    # UL_dose: fine flip monotone in alpha at L* (band-clipped)
    ul_dose = {}
    for dname in ('u_isa', 'u_hasa'):
        L = Lstar[dname]
        key0 = '%s|L%d|a%s' % (dname, L, '0.5')
        if key0 not in res34['models'][m]['fine']:
            # L* not in fine band (should not happen: band = {L*-1,L*,L*+1})
            ul_dose[dname] = dict(status='band_missing', L=L)
            bad.append('fine band missing L* %s %s' % (m, dname))
            continue
        rates = []
        for a in ('0.5', '1.0', '2.0'):
            kk = '%s|L%d|a%s' % (dname, L, a)
            fr = res34['models'][m]['fine'][kk]['flip']['tpl0']['rate']
            rates.append(float(fr))
        mono = all(rates[i + 1] >= rates[i] - 1e-12 for i in range(2))
        # recompute from sealed J1 + fine_J0 (independent of result copy)
        J0_69 = NPZ[m + '_fine_J0'].astype(np.int8)
        all_rows = np.arange(len(J0_69))
        rates_j1 = []
        for a in ('0.5', '1.0', '2.0'):
            J1 = NPZ['%s_fine_%s_L%d_a%s_J1' % (m, dname, L, a)].astype(np.int8)
            rates_j1.append(flip_rate(J1, J0_69, all_rows, mode))
        if max(abs(rates[i] - rates_j1[i]) for i in range(3)) > 1e-12:
            bad.append('dose rate mismatch %s %s' % (m, dname))
        ul_dose[dname] = dict(L=L, rates=rates, rates_j1=rates_j1,
                              monotone=bool(mono))
        log('  UL_dose %s L%d: %s mono=%s' % (dname, L, rates, mono))
    rec['ul_dose'] = ul_dose

    # UL_spec: cross-patch from same coarse J1 at L*
    ul_spec = {}
    for dname in ('u_isa', 'u_hasa'):
        L = Lstar[dname]
        mode = 'isa' if dname == 'u_isa' else 'hasa'
        cross_mode = 'hasa' if mode == 'isa' else 'isa'
        J1 = NPZ['%s_%s_J1_L%d' % (m, dname, L)].astype(np.int8)
        all_rows = np.arange(len(J0_54))
        matched = flip_rate(J1, J0_54, all_rows, mode)
        cross = flip_rate(J1, J0_54, all_rows, cross_mode)
        ok = bool(cross <= ANCHORS['UL_spec'] * max(matched, 1e-12)
                  if matched > 0 else cross <= ANCHORS['UL_spec'] / 208.0)
        ul_spec[dname] = dict(L=L, matched=matched, cross=cross,
                              ratio=(cross / matched if matched > 0 else None),
                              threshold=ANCHORS['UL_spec'],
                              pass_gate=bool(ok))
        log('  UL_spec %s L%d: matched=%.4f cross=%.4f ratio=%s pass=%s'
            % (dname, L, matched, cross,
               ('%.3f' % (cross / matched)) if matched > 0 else 'n/a', ok))
    rec['ul_spec'] = ul_spec

    # UL_amp: max over sealed band layers
    ul_amp_all = res34['models'][m]['ul_amp']
    best_L = max(ul_amp_all, key=lambda L: ul_amp_all[L]['spearman_4oov'])
    rho = ul_amp_all[best_L]['spearman_4oov']
    rec['ul_amp'] = dict(
        per_layer={L: dict(spearman_4oov=ul_amp_all[L]['spearman_4oov'],
                           spearman_10cls_report_only=
                           ul_amp_all[L]['spearman_10cls_report_only'])
                   for L in ul_amp_all},
        best_L=best_L, spearman_best=float(rho),
        pass_gate=bool(rho >= ANCHORS['UL_amp']))
    log('  UL_amp: best L%s rho=%.4f pass=%s'
        % (best_L, rho, rho >= ANCHORS['UL_amp']))

    # UL_loc report-only
    rec['ul_loc'] = dict(
        band=[1, 5],
        overlap={d: bool(1 <= Lstar[d] <= 5) for d in Lstar},
        note='report-only per 3233 design; recorded verbatim, no gate')
    log('  UL_loc: %s (band [1,5])' % rec['ul_loc']['overlap'])

    out[m] = rec

# ---------- verdict ----------
gates_m = {}
for m in MODELS_RUN:
    r = out[m]
    flip_ok = all(r['ul_flip'][d]['pass_primary']
                  for d in ('u_isa', 'u_hasa'))
    dose_ok = all(r['ul_dose'][d].get('monotone', False)
                  for d in ('u_isa', 'u_hasa'))
    spec_ok = all(r['ul_spec'][d]['pass_gate'] for d in ('u_isa', 'u_hasa'))
    amp_ok = r['ul_amp']['pass_gate']
    gates_m[m] = dict(UL_flip=flip_ok, UL_dose=dose_ok,
                      UL_spec=spec_ok, UL_amp=amp_ok)
    log('%s gates: %s' % (m, gates_m[m]))
n_models = len(MODELS_RUN)
found = sum(1 for m in MODELS_RUN
            if gates_m[m]['UL_flip'] and gates_m[m]['UL_dose']
            and gates_m[m]['UL_spec']) >= max(2, n_models) and n_models == 2
flip_any = sum(1 for m in MODELS_RUN if gates_m[m]['UL_flip'])
if found:
    verdict = 'UPSTREAM_LEVER_FOUND'
elif flip_any >= 2:
    verdict = 'UPSTREAM_PARTIAL'
elif n_models == 2 and flip_any == 0:
    verdict = 'UPSTREAM_ABSENT'
else:
    verdict = 'UPSTREAM_INCONCLUSIVE(single-model:%s)' % flip_any
log('verdict: %s' % verdict)

# ---------- assemble result (zero new arrays -> no npz; all gate data
# already sealed in 3234 npz; decisions recorded in result.json) ----------
res = {
    'phase': 3235, 'name': NAME,
    'arm': 'gate evaluation + registry/gap decision (zero GPU, default (a))',
    'prereg_authorization_record': '"好的，继续" = preregistered-default-arm trigger',
    'freeze_sha8': fz['sha8'],
    'design_basis': fz['design_basis'],
    'anchors': ANCHORS,
    'models': out,
    'gates': gates_m,
    'contradictions': bad,
    'verdict': verdict,
    'res_sha8': None,
}
res['res_sha8'] = csha8(dict(res))
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=1))
with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')
log('verdict: %s | res %s | (%.1fs)'
    % (verdict, res['res_sha8'], time.time() - T0))
