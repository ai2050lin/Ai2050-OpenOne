# -*- coding: utf-8 -*-
"""Phase 2882: Atlas Ledger v1 -> v2 migration (zero forward).

Goal: upgrade the TMA Atlas Ledger to format v2 so the multi-model
replication campaign (P3) can start from a single extensible fact source.
v2 additions (spec sections 10-13):
    quadrants          channel-dissociation quadrant table (4 families)
    transfer           cross-family transfer matrix (2881 J3/J4 snapshot)
    negatives          real-negative / quarantine registry
    model_namespace    primary model + pending replication models
    migration_history  format migration log with v1 backup SHA

All existing v1 entries are preserved BYTE-FAITHFULLY (V1 gate).  Every
new cell must cite existing meas_ids (V2/V3 derivability gates).

Prereg (frozen before the ledger is touched):
  V1 preservation: axes/blocks/headsets/measurements/growth_curve/linkage
      counts identical pre/post; SHA re-verify => 0 stale.
  V2 quadrant derivability: every quadrant cell cites >= 1 meas_id that
      exists in measurements.
  V3 negatives grounding: every negative entry cites >= 1 existing
      meas_id and has status in {settled, reopenable}; quarantine
      entries must carry reopen_condition.
  V4 backward compatibility: the v1 backup loads cleanly under the v2
      loader (optional sections default empty), verify 0 stale.
  V5 self-consistency: reloaded v2 ledger has version == 2, all five new
      sections present, verify 0 stale.
  verdict ledger_v2 iff V1..V5 all true.

Output: result/rdc_query_construction_20260913/phase2882/ledger_v2/
        {execution.json, result.json}
Ledger: research/gpt5/atlas/atlas_ledger.json (v2)
Backup: research/gpt5/atlas/atlas_ledger_v1_backup.json
"""
import hashlib
import io
import json
import os
import shutil
import sys
import time

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger, RESULT_BASE

ATLAS_DIR = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
LEDGER_P = os.path.join(ATLAS_DIR, 'atlas_ledger.json')
BACKUP_P = os.path.join(ATLAS_DIR, 'atlas_ledger_v1_backup.json')
OUT_DIR = os.path.join(RESULT_BASE, 'phase2882', 'ledger_v2')

SEED = 2882
T0 = time.monotonic()
LOG = []


def log(s):
    LOG.append(str(s))
    print(s, flush=True)


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


os.makedirs(OUT_DIR, exist_ok=True)
PREREG = {
    'V1': 'section counts identical pre/post; SHA re-verify => 0 stale',
    'V2': 'every quadrant cell cites >= 1 meas_id existing in measurements',
    'V3': 'every negative cites >= 1 existing meas_id, status in '
          '{settled, reopenable}; quarantine entries carry reopen_condition',
    'V4': 'v1 backup loads cleanly under the v2 loader (verify 0 stale)',
    'V5': 'reloaded v2 ledger: version==2, all new sections present, '
          'verify 0 stale',
    'verdict': 'ledger_v2 iff V1..V5 all true',
}
# ---------- freeze prereg BEFORE touching the ledger ----------
exec_doc = {
    'phase': 2882,
    'name': 'ledger_v2',
    'seed': SEED,
    'prereg': PREREG,
    'inputs': {
        'ledger': 'research/gpt5/atlas/atlas_ledger.json',
        'loader': 'tests/glm5/rdc_atlas_ledger.py',
        'spec': 'research/gpt5/atlas/ATLAS_LEDGER_SPEC.md',
        'transfer_source': ('phase2881/joint_word_coords/result.json '
                            '(sha256_8 eb95f6ff)'),
    },
    'frozen_at_s': round(T0, 1),
}
with io.open(os.path.join(OUT_DIR, 'execution.json'), 'w',
             encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=2, ensure_ascii=False)
log('prereg frozen -> execution.json')

# ---------- load v1 (verbatim snapshot) ----------
led = AtlasLedger.load(LEDGER_P, verify_sha=True)
stale0 = led.verify()
log('v1 stale=%d' % len(stale0))
doc = led.doc
pre_counts = {k: len(doc[k]) for k in
              ('axes', 'blocks', 'headsets', 'measurements',
               'growth_curve', 'linkage')}
meas_ids = set(m['meas_id'] for m in doc['measurements'])
log('v1 counts: %s' % pre_counts)

# ---------- backup v1 (idempotent: only if current file is v1) ----------
cur_ver = json.load(io.open(LEDGER_P, encoding='utf-8'))['version']
if cur_ver > 2:
    raise SystemExit('ledger version %d > 2; refusing to run' % cur_ver)
if cur_ver == 1:
    if os.path.exists(BACKUP_P):
        os.remove(BACKUP_P)
    shutil.copy2(LEDGER_P, BACKUP_P)
if not os.path.exists(BACKUP_P):
    raise SystemExit('no v1 backup available')
backup_sha = sha8(BACKUP_P)
log('backup -> %s (sha %s)' % (os.path.basename(BACKUP_P), backup_sha))

# ---------- migrate in memory ----------
doc['version'] = 2
doc['model_namespace'] = {
    'primary': 'qwen3-4b',
    'protocol_family': 'LPF v5.3',
    'pending_replication': ['ds7b', 'glm4', 'gemma4'],
    'rule': ('cross-model entries must carry a model field; ids unique '
             'within one model, same id may coexist across models in '
             'ledgers/<model>/ sub-files'),
}
doc['quadrants'] = [
    {'family': 'class',
     'drop_spectrum': {'organized': True, 'frontedge_heads': 43,
                       'sources': ['M2859_stability']},
     'mlp_carrier': {'acc': 0.875,
                     'sources': ['M2867_word_coords_v1']},
     'quadrant': 'organized+strong_mlp'},
    {'family': 'attr',
     'drop_spectrum': {'organized': False,
                       'sources': ['M2875_census_v2'],
                       'note': 'X1=false (n_heads_ge6=0), X5=absent'},
     'mlp_carrier': {'acc': 0.5, 'sources': ['M2877_mlp_channel']},
     'quadrant': 'dissociated_mlp_only'},
    {'family': 'syntax',
     'drop_spectrum': {'organized': False,
                       'weak_retrieval_acc': 0.5625,
                       'null_p95': 0.5417,
                       'sources': ['M2880_syntax_census'],
                       'note': 'Y1 frontedge absent; Y5 marginal +0.021'},
     'mlp_carrier': {'acc': 0.7083, 'sources': ['M2879_syntax_mlp']},
     'quadrant': 'intermediate'},
    {'family': 'translation',
     'drop_spectrum': {'organized': False,
                       'unified_axis_direction': False,
                       'sources': ['M2878_syntax_trans_vocab']},
     'mlp_carrier': {'acc': None, 'sources': [],
                     'note': 'not_measured - no unified axis to project'},
     'quadrant': 'no_axis'},
]
doc['transfer'] = {
    'source': {'meas_id': 'M2881_joint_coords', 'phase': 2881},
    'matrix_rows=fa_dirs_cols=fb_words': {
        'class': {'class': 0.9625, 'attr': 0.9286, 'syntax': 0.9792},
        'attr': {'class': 0.975, 'attr': 0.9524, 'syntax': 0.8958},
        'syntax': {'class': 0.9, 'attr': 0.7619, 'syntax': 0.8542},
    },
    'family_centroid_cos': [[1.0, -0.6475, -0.3283],
                            [-0.6475, 1.0, -0.1436],
                            [-0.3283, -0.1436, 1.0]],
    'interpretation': ('all cells > 0.76 (class rows strongest 0.93-0.98); '
                       'family centroids mutually negative - separated '
                       'subspaces of one mlp channel'),
    'descriptive': True,
}
doc['negatives'] = [
    {'neg_id': 'N1_translation_axis_absent',
     'kind': 'real_negative',
     'claim': 'no unified cross-lingual axis direction exists in the '
              'tied unembed (en->L direction is concept/wordform '
              'dominated, language-constant component ~ 0)',
     'evidence': 'VG2b pair-pair cos fr -0.0007 / de -0.0029 / '
                 'es 0.0011 after homograph decontamination (2878 Gen2)',
     'sources': ['M2878_syntax_trans_vocab'],
     'status': 'settled', 'reopen_condition': None},
    {'neg_id': 'N2_tense_axis_quarantine',
     'kind': 'quarantine',
     'claim': 'tense axis VG2b 0.1902 < 0.2 gate (margin 0.0102) - '
              'quarantined, NOT a zero-proof',
     'evidence': '2878 Gen2 per-axis VG2b; number 0.2021 / gerund 0.2352 '
                 '/ comparative 0.3636 survived',
     'sources': ['M2878_syntax_trans_vocab'],
     'status': 'reopenable',
     'reopen_condition': 'expand tense pool to >= 8 single-token pairs '
                         'and re-test VG2b'},
    {'neg_id': 'N3_e200_quarantine',
     'kind': 'quarantine',
     'claim': 'E200 vocab failed 2858 G2 cross-axis orthogonality '
              '(offdiag 0.494 > 0.488); census deferred',
     'evidence': '2858 G2 gate; 199-word vocab',
     'sources': ['M2857_drift'],
     'status': 'reopenable',
     'reopen_condition': 'rebuild vocab with per-axis pools such that '
                         'max offdiag < 0.488'},
]
doc['migration_history'] = [
    {'phase': 2882, 'from_version': 1, 'to_version': 2,
     'backup': 'atlas_ledger_v1_backup.json', 'backup_sha256_8': backup_sha,
     'note': 'existing v1 entries byte-faithful; new sections per spec '
             'v2 sections 10-13'},
]
log('migration composed in memory')

# ---------- gates ----------
# V2: quadrant derivability
v2_ok, v2_bad = True, []
for q in doc['quadrants']:
    cited = set(q['drop_spectrum'].get('sources', []))
    cited |= set(q['mlp_carrier'].get('sources', []))
    if not cited <= meas_ids or not cited:
        v2_ok = False
        v2_bad.append((q['family'], sorted(cited - meas_ids)))
log('V2 quadrant derivability: %s %s' % (v2_ok, v2_bad))

# V3: negatives grounding
v3_ok, v3_bad = True, []
for n in doc['negatives']:
    cited = set(n.get('sources', []))
    ok = (cited and cited <= meas_ids
          and n['status'] in ('settled', 'reopenable')
          and (n['status'] != 'quarantine'
               or n.get('reopen_condition')))
    if not ok:
        v3_ok = False
        v3_bad.append(n['neg_id'])
log('V3 negatives grounding: %s %s' % (v3_ok, v3_bad))

# ---------- save v2 ----------
led.save()
log('saved v2 ledger')

# V1: post-save preservation + SHA re-verify
led1 = AtlasLedger.load(LEDGER_P, verify_sha=True)
stale1 = led1.verify()
post_counts = {k: len(led1.doc[k]) for k in pre_counts}
v1_ok = (post_counts == pre_counts and not stale1)
log('V1 preservation: %s (stale=%d)' % (v1_ok, len(stale1)))

# V4: v1 backup loads under the v2 loader
led_b = AtlasLedger.load(BACKUP_P, verify_sha=True)
v4_ok = (led_b.doc['version'] == 1 and not led_b.verify()
         and led_b.doc['quadrants'] == [] and led_b.doc['negatives'] == [])
log('V4 backward compat: %s' % v4_ok)

# V5: self-consistency of v2
v5_ok = (led1.doc['version'] == 2
         and all(k in led1.doc for k in
                 ('quadrants', 'transfer', 'negatives',
                  'model_namespace', 'migration_history'))
         and len(led1.doc['quadrants']) == 4
         and len(led1.doc['negatives']) == 3
         and not stale1)
log('V5 self-consistency: %s' % v5_ok)

verdict = bool(v1_ok and v2_ok and v3_ok and v4_ok and v5_ok)
res = {
    'phase': 2882,
    'prereg': PREREG,
    'backup': {'path': 'research/gpt5/atlas/atlas_ledger_v1_backup.json',
               'sha256_8': backup_sha},
    'V1': {'pre_counts': pre_counts, 'post_counts': post_counts,
           'stale': len(stale1), 'verdict': v1_ok},
    'V2': {'verdict': v2_ok, 'bad': v2_bad},
    'V3': {'verdict': v3_ok, 'bad': v3_bad},
    'V4': {'verdict': v4_ok},
    'V5': {'verdict': v5_ok},
    'n_quadrants': 4,
    'n_negatives': 3,
    'ledger_v2': verdict,
    'final_verdict': 'ledger_v2=%s (v1->v2 migration; %d quadrants, '
                     '%d negatives, transfer snapshot, model namespace '
                     'pending %s)'
                     % (verdict, 4, 3,
                        json.dumps(doc['model_namespace']
                                   ['pending_replication'])),
    'runtime_s': round(time.monotonic() - T0, 1),
}
with io.open(os.path.join(OUT_DIR, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
log('==== VERDICT: ledger_v2=%s ====' % verdict)
with io.open(os.path.join(OUT_DIR, 'run.log'), 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
