# -*- coding: utf-8 -*-
"""Phase 3190 independent verification (15 checks, disk-read口径)."""
import hashlib
import io
import json
import os

R = '8'  # unused marker
BASE = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3190\g5c3_shared_direction_robustness')
OUT89 = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3189\g5c2_residual_structure')

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def load(p):
    return json.load(io.open(p, encoding='utf-8'))

checks = []
def chk(i, name, cond, detail=''):
    checks.append((i, name, bool(cond), detail))

# 1 freeze
ex = load(os.path.join(OUT, 'execution.json'))
chk(1, 'freeze execution.json design_sha8', ex.get('design_sha8') == 'bfe79091', str(ex.get('design_sha8')))

# 2 arm result on disk + embedded seal
rp, sp = os.path.join(OUT, 'result.json'), None
_tmp = load(rp)
chk(2, 'arm embedded res/seal anchors (content sha != file sha)', _tmp.get('res_sha8') == '24461856' and _tmp.get('seal_sha8') == '0c975f5b',
    '%s/%s' % (_tmp.get('res_sha8'), _tmp.get('seal_sha8')))

# 3 arm payload: gate 3/3, drift anchors 0
arm = load(rp)
res = arm['per_model']
gates = [res[m]['gate_model'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')]
drifts = [res[m]['drift_3169'] for m in res]
chk(3, 'LOO gate 3/3 + drift69==0 x3',
    all(gates) and all(res[m]['drift_3169'] == 0.0 and res[m]['drift_3189'] == 0 for m in res),
    str(gates))

# 4 LOO deltas match log values bitwise
d4 = [res[m]['loo_max_delta'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')]
chk(4, 'loo_max_delta values (4dp)', [round(x, 4) for x in d4] == [0.0342, 0.0214, 0.0332], str([round(x,4) for x in d4]))

# 5 3189 sealed scalars match (device anchor, from sealed 3189 arm res)
r89 = load(os.path.join(OUT89, 'result.json'))
res89 = r89['per_model']
ok = True
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    for k in ('top1', 'cross_med', 'within_class', 'loo_top1' if False else 'within_class'):
        pass
    if abs(res89[m]['top1'] - res[m]['top1']) > 0 or res89[m]['cross_med'] != res[m]['cross_med']:
        ok = False
chk(5, '3189 sealed top1/cross_med bitwise == 3190 arm', ok)

# 6 final chain with arm anchors substituted
fin = load(os.path.join(OUT, 'final_result.json'))
chk(6, 'final res/seal field', fin.get('res_sha8') == 'fc8be7db' and fin.get('seal_sha8') == 'f7d8571a',
    str(fin.get('res_sha8')) + '/' + str(fin.get('seal_sha8')))
chk(7, 'final arm per_model == arm result per_model (bitwise)',
    fin['arm']['per_model'] == arm['per_model'])

# 8 registry v1.17 file sha + append-only vs v1.16
reg17p = os.path.join(OUT, 'atlas_registry_v1_17.json')
reg16p = os.path.join(OUT89, 'atlas_registry_v1_16.json')
chk(8, 'registry v1.17 file sha 73923c83', sha8(reg17p) == '73923c83', sha8(reg17p))
r17, r16 = load(reg17p), load(reg16p)
f16 = [f['id'] for f in r16['features']]
f17 = [f['id'] for f in r17['features']]
chk(9, 'features append-only 24->25', f17[:len(f16)] == f16 and len(f17) == 25, '%d->%d' % (len(f16), len(f17)))
chk(10, 'failures byte-preserved', json.dumps(r16.get('failures', []), ensure_ascii=False, sort_keys=True)
    == json.dumps(r17.get('failures', []), ensure_ascii=False, sort_keys=True))
f25 = [f for f in r17['features'] if f['id'] == 'FTR-25'][0]
f25_16 = [f for f in r16['features'] if f['id'] == 'FTR-25'][0]
chk(11, 'FTR-25 statement prefix + anchors append',
    f25['statement'].startswith(f25_16['statement'])
    and [a.get('role') for a in f25.get('anchors', [])] == ['shared_oov_collapse_direction', 'loo_robustness']
    and f25.get('e_level') == f25_16.get('e_level'))

# 12 ledger #342
led = load(os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json'))
ms = led['measurements']
chk(12, 'ledger n=342 last=3190 chain ok', len(ms) == 342 and ms[-1]['phase'] == 3190
    and ms[-1]['chain']['registry_v117'] == '73923c83' and ms[-2]['phase'] == 3189)

# 13 SMOKE vs FULL arm payload identical (exclude mode/sha fields)
sm = load(os.path.join(OUT, 'smoke_result.json'))
smr, fur = sm['per_model'], res
sd = {k: v for k, v in smr.items() if k not in ('mode', 'res_sha8', 'seal_sha8')}
fd = {k: v for k, v in fur.items() if k not in ('mode', 'res_sha8', 'seal_sha8')}
chk(13, 'SMOKE/FULL arm payload bitwise identical', sd == fd)

# 14 atlas html file sha + field count
hp = os.path.join(OUT, 'atlas_v1_18.html')
html = io.open(hp, encoding='utf-8').read()
chk(14, 'atlas_v1_18.html file sha 580af247', sha8(hp) == '580af247', sha8(hp))
nk = html.count('data-k=')
chk(15, 'html data-k fields 746', nk == 746, str(nk))

# 16 MEMO block present
memo = io.open(os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md'), encoding='utf-8').read()
chk(16, 'MEMO 3190 block + prereg 3191', 'Phase 3190' in memo and 'G5-C3' in memo
    and '3191 预注册' in memo and 'LOO 稳健 3/3' in memo and 'E1_repeatable' in memo)

# 17 SMOKE-1 honest log present
chk(17, 'SMOKE-1 U/Vt fix logged', 'Vt[0]' in memo and 'SMOKE-1' in memo)

n_pass = sum(1 for c in checks if c[2])
for i, name, okk, det in checks:
    print('%s #%02d %s %s' % ('PASS' if okk else 'FAIL', i, name, det))
print('== %d/%d ==' % (n_pass, len(checks)))
