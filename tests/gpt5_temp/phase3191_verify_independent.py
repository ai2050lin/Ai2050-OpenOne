# -*- coding: utf-8 -*-
"""Phase 3191 independent verification (17 checks, disk-read口径)."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3191\g5c4_ftr25_heldout_upgrade')
OUT90 = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3190\g5c3_shared_direction_robustness')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


checks = []


def chk(i, name, cond, detail=''):
    checks.append((i, name, bool(cond), detail))


# 1 freeze
ex = load(os.path.join(OUT, 'execution.json'))
chk(1, 'freeze execution.json design_sha8', ex.get('design_sha8') == 'c4673cc3', str(ex.get('design_sha8')))

# 2 arm embedded res/seal anchors (content sha != file sha)
rp = os.path.join(OUT, 'result.json')
_tmp = load(rp)
chk(2, 'arm embedded res/seal anchors', _tmp.get('res_sha8') == '836fe933' and _tmp.get('seal_sha8') == 'a66c0b3d',
    '%s/%s' % (_tmp.get('res_sha8'), _tmp.get('seal_sha8')))

# 3 gate 3/3 + all drift anchors 0
res = _tmp['per_model']
MS = ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
chk(3, 'held-out gate 3/3 + drift69/89/90 == 0 x3',
    all(res[m]['gate_model'] for m in MS)
    and all(res[m]['drift_3169'] == 0.0 and res[m]['drift_3189'] == 0.0
            and res[m]['drift_3190'] == 0.0 for m in MS))

# 4 med_abs_cos values (4dp) + gate semantics (min >= 0.3 all folds)
d4 = {m: res[m]['min_med_abs_cos'] for m in MS}
chk(4, 'min_med_abs_cos (4dp)', all(round(v, 4) == e for v, e in
    zip([d4[m] for m in MS], [0.3906, 0.4174, 0.3717])), str({m: round(v, 4) for m, v in d4.items()}))
chk(5, 'gate semantics min>=0.3 all folds', all(
    min(res[m]['med_abs_cos_held']) >= 0.3 and res[m]['gate_model'] for m in MS))

# 6 3190 sealed scalars bitwise (device anchor, vs sealed 3190 arm res)
r90 = load(os.path.join(OUT90, 'result.json'))
pm90 = r90['per_model']
ok = all(pm90[m]['top1'] == res[m]['top1'] and pm90[m]['cross_med'] == res[m]['cross_med']
         and pm90[m]['loo_max_delta'] == res[m].get('loo_max_delta', pm90[m]['loo_max_delta'])
         for m in MS)
chk(6, '3190 sealed top1/cross_med bitwise == 3191 arm', ok)

# 7 final
fin = load(os.path.join(OUT, 'final_result.json'))
chk(7, 'final res/seal field', fin.get('res_sha8') == '28a852b0' and fin.get('seal_sha8') == '6eb89727',
    '%s/%s' % (fin.get('res_sha8'), fin.get('seal_sha8')))
chk(8, 'final arm per_model == arm result per_model (bitwise)',
    fin['arm']['per_model'] == _tmp['per_model'])

# 9 registry v1.18 file sha + structure
reg18p = os.path.join(OUT, 'atlas_registry_v1_18.json')
reg17p = os.path.join(OUT90, 'atlas_registry_v1_17.json')
chk(9, 'registry v1.18 file sha c08cbcc2', sha8(reg18p) == 'c08cbcc2', sha8(reg18p))
r18, r17 = load(reg18p), load(reg17p)
chk(10, 'version/supersedes', r18['version'] == '1.18' and '73923c83' in r18.get('supersedes', ''))
f17 = [f['id'] for f in r17['features']]
f18 = [f['id'] for f in r18['features']]
chk(11, 'features ids preserved x25', f18 == f17 and len(f18) == 25)
chk(12, 'failures byte-preserved', json.dumps(r17['failures'], ensure_ascii=False, sort_keys=True)
    == json.dumps(r18['failures'], ensure_ascii=False, sort_keys=True))
f25, f25_17 = [f for f in r18['features'] if f['id'] == 'FTR-25'][0], \
    [f for f in r17['features'] if f['id'] == 'FTR-25'][0]
chk(13, 'FTR-25 E1->E2_held_out + statement/anchors prefix',
    f25['evidence_level'] == 'E2_held_out' and f25_17['evidence_level'] == 'E1_repeatable'
    and f25['statement'].startswith(f25_17['statement'])
    and [a.get('role') for a in f25['anchors']] == [a.get('role') for a in f25_17['anchors']] + ['held_out'])
chk(14, 'upgrade_log append-only #7 FTR-25',
    len(r18['upgrade_log']) == 7 and r18['upgrade_log'][-1]['node'] == 'FTR-25'
    and 'E2_held_out' in r18['upgrade_log'][-1]['change']
    and json.dumps(r18['upgrade_log'][:-1], ensure_ascii=False, sort_keys=True)
    == json.dumps(r17['upgrade_log'], ensure_ascii=False, sort_keys=True))
chk(15, 'other 24 features byte-preserved', all(
    json.dumps(f17f, ensure_ascii=False, sort_keys=True)
    == json.dumps([x for x in r18['features'] if x['id'] == f17f['id']][0],
                  ensure_ascii=False, sort_keys=True)
    for f17f in r17['features'] if f17f['id'] != 'FTR-25'))

# 16 ledger #343
led = load(os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json'))
ms = led['measurements']
chk(16, 'ledger n=343 last=3191 chain ok', len(ms) == 343 and ms[-1]['phase'] == 3191
    and ms[-1]['chain']['registry_v118'] == 'c08cbcc2' and ms[-2]['phase'] == 3190)

# 17 atlas html file sha + field count + SMOKE/FULL payload identity
hp = os.path.join(OUT, 'atlas_v1_19.html')
html = io.open(hp, encoding='utf-8').read()
chk(17, 'atlas_v1_19.html file sha d5157665 + data-k 754', sha8(hp) == 'd5157665'
    and html.count('data-k=') == 754, '%s/%d' % (sha8(hp), html.count('data-k=')))
sm = load(os.path.join(OUT, 'smoke_result.json'))
strip = lambda d: {k: v for k, v in d.items() if k not in ('mode', 'res_sha8', 'seal_sha8')}
chk(18, 'SMOKE/FULL arm payload bitwise identical', strip(sm) == strip(_tmp))

# MEMO + honest log
memo = io.open(os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md'), encoding='utf-8').read()
chk(19, 'MEMO 3191 block + prereg 3192 + honest log', 'Phase 3191' in memo
    and 'G5-C4' in memo and '3192 预注册' in memo
    and 'E2_held_out' in memo and 'evidence_level' in memo and 'CONTENT_3190' in memo)

n_pass = sum(1 for c in checks if c[2])
for i, name, okk, det in checks:
    print('%s #%02d %s %s' % ('PASS' if okk else 'FAIL', i, name, det))
print('== %d/%d ==' % (n_pass, len(checks)))
