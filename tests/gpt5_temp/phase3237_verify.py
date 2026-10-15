# -*- coding: utf-8 -*-
"""Phase 3237 independent verify: recheck sealed artifacts from disk.
FAIL -> fix this verify only, NEVER the sealed artifacts."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase3237', 'g5g31_registry_landing')

P = N = 0
FAILS = []


def chk(name, cond):
    global P, N
    N += 1
    if cond:
        P += 1
        print('PASS %s' % name)
    else:
        FAILS.append(name)
        print('FAIL %s' % name)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def jd(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True)


reg32 = json.load(open(os.path.join(OUT, 'atlas_registry_v1_32.json'),
                       encoding='utf-8'))
gap11 = json.load(open(os.path.join(OUT, 'gap_ledger_v1_11.json'),
                       encoding='utf-8'))
res = json.load(open(os.path.join(OUT, 'result.json'), encoding='utf-8'))
fz = json.load(open(os.path.join(OUT, 'execution.json'), encoding='utf-8'))
reg31 = json.load(open(os.path.join(BASE, 'phase3231',
                                    'g5g28_registry_decision',
                                    'atlas_registry_v1_31.json'),
                       encoding='utf-8'))
gap10 = json.load(open(os.path.join(BASE, 'phase3232',
                                    'g5g29_gd_amp_class_robustness',
                                    'gap_ledger_v1_10.json'),
                       encoding='utf-8'))
r35 = json.load(open(os.path.join(BASE, 'phase3235',
                                  'g5g31_gap5_gate_decision',
                                  'result.json'), encoding='utf-8'))

# V1 file shas match sealed record
chk('V1a registry v1.32 file sha', fsha(os.path.join(
    OUT, 'atlas_registry_v1_32.json')) == res['wrote']['registry_v132']['file_sha8'])
chk('V1b gap v1.11 file sha', fsha(os.path.join(
    OUT, 'gap_ledger_v1_11.json')) == res['wrote']['gap_v111']['file_sha8'])

# V2 res_sha8 recompute
rc = dict(res)
rc['res_sha8'] = None
rec = hashlib.sha256(json.dumps(
    rc, ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]
chk('V2 res_sha8 recompute', rec == res['res_sha8'])

# V3 freeze recompute (self-key excluded)
fc = dict(fz)
fc.pop('sha8', None)
frec = hashlib.sha256(json.dumps(
    fc, ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]
chk('V3 freeze recompute', frec == fz['sha8'] == res['freeze_sha8'])

# V4 registry v1.32 structure + provenance
chk('V4a version 1.32', reg32['version'] == '1.32')
chk('V4b supersedes v1.31', 'atlas_registry_v1_31.json' in reg32['supersedes']
    and 'ef5189e7' in reg32['supersedes'])
chk('V4c features 26', len(reg32['features']) == 26)
chk('V4d failures 21 last F21',
    len(reg32['failures']) == 21 and reg32['failures'][-1]['id'] == 'F21')
chk('V4e upgrade_log untouched 11', jd(reg31['upgrade_log'])
    == jd(reg32['upgrade_log']))
chk('V4f provenance anchor npz', reg32['provenance']['anchor_files']
    .get('upstream_lever_scan', {}).get('sha8') == '48034bd9')

# V5 F21 evidence numbers vs sealed 3235 result
f21 = reg32['failures'][-1]
m4 = r35['models']['qwen3-4b']
m14 = r35['models']['qwen3-14b']
chk('V5a F21 kind/phase', f21['kind'] == 'upstream_lever_absent'
    and f21['phase'] == 3237)
chk('V5b F21 spearman 4b -0.20',
    '%+.2f' % m4['ul_amp']['spearman_best'] in f21['text'])
chk('V5c F21 spearman 14b +0.40',
    '%+.2f' % m14['ul_amp']['spearman_best'] in f21['text'])
chk('V5d F21 Lstar zeros',
    ('u_isa:%d' % m4['Lstar']['u_isa']) in f21['text']
    and ('u_hasa:%d' % m4['Lstar']['u_hasa']) in f21['text'])
chk('V5e F21 anchors res/npz', '5bab6419' in ' '.join(f21['evidence'])
    and '48034bd9' in ' '.join(f21['evidence']))
chk('V5f F21 mode-leak boundary recorded', 'V7' in f21['text']
    and 'mode' in f21['text'])
# all flips really zero in sealed 3235
allzero = all(x == 0.0 for m in (m4, m14) for d in ('u_isa', 'u_hasa')
              for x in m['ul_flip'][d]['per_fold_max_L'])
chk('V5g 3235 per_fold_max_L all zero (basis of F21)', allzero)
gates_ok = all(r35['gates'][m] == {'UL_flip': False, 'UL_dose': False,
                                   'UL_spec': True, 'UL_amp': False}
               for m in ('qwen3-4b', 'qwen3-14b'))
chk('V5h 3235 gates F/F/T/F x2', gates_ok)

# V6 byte-preservation: failures[:20], features
chk('V6a failures[:20] byte-exact', jd(reg31['failures'])
    == jd(reg32['failures'][:20]))
ok_feat = True
for a, b in zip(reg31['features'], reg32['features']):
    if a['id'] in ('FTR-25', 'FTR-26'):
        if (jd(a['statement']) != jd(b['statement'])
                or jd(a['evidence_level']) != jd(b['evidence_level'])
                or len(b.get('scope_limits') or '')
                <= len(a.get('scope_limits') or '')):
            ok_feat = False
    elif jd(a) != jd(b):
        ok_feat = False
chk('V6b features byte-preservation (24 exact + 2 scope appends)', ok_feat)
top_ok = all(jd(reg31[k]) == jd(reg32[k]) for k in reg31
             if k not in ('version', 'supersedes', 'failures', 'provenance',
                          'features'))
chk('V6c top-level byte-preservation', top_ok)

# V7 gap v1.11
chk('V7a version/updated', gap11['version'] == '1.11'
    and gap11['updated'] == '2026-10-14')
chk('V7b gaps count 5', len(gap11['gaps']) == 5)
ok_gaps = all(jd(a) == jd(b) for a, b in zip(gap10['gaps'], gap11['gaps'])
              if a['id'] != 'GAP-5')
chk('V7c legacy gaps byte-exact x4', ok_gaps)
g5o = [x for x in gap10['gaps'] if x['id'] == 'GAP-5'][0]
g5n = [x for x in gap11['gaps'] if x['id'] == 'GAP-5'][0]
chk('V7d GAP-5 legacy fields byte-exact',
    all(jd(g5o[k]) == jd(g5n[k]) for k in g5o if k != 'mechanism_note'))
chk('V7e mechanism_note appended longer',
    len(g5n['mechanism_note']) > len(g5o['mechanism_note'])
    and g5n['mechanism_note'].startswith(g5o['mechanism_note']))
chk('V7f final_verdict new key UPSTREAM_ABSENT',
    'final_verdict' not in g5o
    and g5n['final_verdict']['verdict'] == 'UPSTREAM_ABSENT'
    and g5n['final_verdict']['phase'] == 3237)
chk('V7g gap top byte-preservation',
    all(jd(gap10[k]) == jd(gap11[k]) for k in gap10
        if k not in ('version', 'updated', 'gaps')))

# V8 ledger state (post-closeout: #389 appended, chain recompute OK)
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
lc = dict(led)
stored_post = lc['ledger_sha256_8']
lc['ledger_sha256_8'] = '05a31fc4'  # pre-append value for chain recompute
rec_post = hashlib.sha256(json.dumps(
    lc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
chk('V8 ledger n=389 post chain-verified (#389 3237 entry)',
    len(led['measurements']) == 389
    and stored_post == rec_post
    and sum(1 for e in led['measurements']
            if e.get('phase') == 3237) == 1)

# V9 verdict consistency
chk('V9 verdict', res['verdict'] == 'f21_upstream_absent_registered'
    and res['recheck']['contradictions'] == [])

with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3237_verify_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('n_pass=%d n_fails=%d\n' % (P, N - P))
    f.write('fails: %s\n' % (FAILS or 'none'))
print('VERIFY %d/%d PASS' % (P, N))
