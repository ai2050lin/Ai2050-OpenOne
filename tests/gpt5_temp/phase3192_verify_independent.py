# -*- coding: utf-8 -*-
"""Phase 3192 independent verification (19 checks, disk-read口径)."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3192\g5c5_gap4_amplitude_attribution')
OUT91 = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3191\g5c4_ftr25_heldout_upgrade')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


checks = []


def chk(i, name, cond, detail=''):
    checks.append((i, name, bool(cond), detail))


# 1 freeze
ex = load(os.path.join(OUT, 'execution.json'))
chk(1, 'freeze execution.json design_sha8', ex.get('design_sha8') == 'efc062be', str(ex.get('design_sha8')))

# 2 arm embedded res/seal anchors
rp = os.path.join(OUT, 'result.json')
_tmp = load(rp)
chk(2, 'arm embedded res/seal anchors', _tmp.get('res_sha8') == '6221fe00' and _tmp.get('seal_sha8') == '58c4f183',
    '%s/%s' % (_tmp.get('res_sha8'), _tmp.get('seal_sha8')))

# 3 gate 3/3 + all drift anchors 0
res = _tmp['per_model']
MS = ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
chk(3, 'carrier gate 3/3 + drift69/89/90/91 == 0 x3',
    all(res[m]['gate_model'] for m in MS)
    and all(res[m]['drift_3169'] == 0.0 and res[m]['drift_3189'] == 0.0
            and res[m]['drift_3190'] == 0.0 and res[m]['drift_3191'] == 0.0 for m in MS))

# 4 amp_ratio values (4dp) + gate semantics
chk(4, 'amp_ratio (4dp)', [round(res[m]['amp_ratio'], 4) for m in MS] == [10.5984, 7.1100, 8.8909],
    str([round(res[m]['amp_ratio'], 4) for m in MS]))
chk(5, 'gate semantics amp_ratio>=1.3 x3', all(res[m]['amp_ratio'] >= 1.3 for m in MS))
chk(6, 'amp_ratio super-proportional vs ratio_B 2.5388 x3',
    all(res[m]['amp_ratio'] > 2.5388114997805062 for m in MS))

# 7 3191 sealed scalars bitwise (device anchor)
r91 = load(os.path.join(OUT91, 'result.json'))
pm91 = r91['per_model']
ok = all(res[m]['top1'] == pm91[m]['top1'] and res[m]['cross_med'] == pm91[m]['cross_med']
         for m in MS)
chk(7, '3191 sealed top1/cross_med bitwise == 3192 arm', ok)

# 8 final
fin = load(os.path.join(OUT, 'final_result.json'))
chk(8, 'final res/seal field', fin.get('res_sha8') == '7f35fca4' and fin.get('seal_sha8') == 'b147d3d2',
    '%s/%s' % (fin.get('res_sha8'), fin.get('seal_sha8')))
chk(9, 'final arm per_model == arm result per_model (bitwise)',
    fin['arm']['per_model'] == _tmp['per_model'])
chk(10, 'final disposition + render refs', fin.get('registry_disposition') == 'v1_5_to_v1_6_gap4_amplitude_carrier'
    and fin.get('registry_render_sha8') == 'c08cbcc2' and fin.get('gap_render_sha8') == '0f6423e0')

# 11 gap ledger v1.6 file sha + append-only vs v1.5
gl6p = os.path.join(OUT, 'gap_ledger_v1_6.json')
gl5p = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json')
chk(11, 'gap ledger v1.6 file sha 0f6423e0', sha8(gl6p) == '0f6423e0', sha8(gl6p))
gl6, gl5 = load(gl6p), load(gl5p)
chk(12, 'version/supersedes', gl6['version'] == '1.6' and '4fb3f37d' in gl6.get('supersedes', ''))
g5ids = [g['id'] for g in gl5['gaps']]
g6ids = [g['id'] for g in gl6['gaps']]
chk(13, 'gap ids preserved', g6ids == g5ids)
g4, g4_5 = [g for g in gl6['gaps'] if g['id'] == 'GAP-4'][0], [g for g in gl5['gaps'] if g['id'] == 'GAP-4'][0]
chk(14, 'GAP-4 mechanism_note/evidence append + rest byte-preserved',
    g4['mechanism_note'].startswith(g4_5['mechanism_note'])
    and g4['evidence'][:-1] == g4_5['evidence'] and len(g4['evidence']) == 9
    and g4['statement'] == g4_5['statement'] and g4['prereg'] == g4_5['prereg']
    and g4['anchor_sha8'] == g4_5['anchor_sha8']
    and 'u*_amplitude_carrier' in g4['mechanism_note'])
chk(15, 'other gaps + appendix byte-preserved', all(
    json.dumps(g5g, ensure_ascii=False, sort_keys=True)
    == json.dumps([x for x in gl6['gaps'] if x['id'] == g5g['id']][0], ensure_ascii=False, sort_keys=True)
    for g5g in gl5['gaps'] if g5g['id'] != 'GAP-4')
    and gl6['appendix'] == gl5['appendix'])

# 16 registry v1.18 UNCHANGED (file on disk = sealed 3191 artifact sha)
chk(16, 'registry v1.18 unchanged (sha c08cbcc2)',
    sha8(os.path.join(OUT91, 'atlas_registry_v1_18.json')) == 'c08cbcc2')

# 17 ledger #344
led = load(os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json'))
ms = led['measurements']
chk(17, 'ledger n=344 last=3192 chain ok', len(ms) == 344 and ms[-1]['phase'] == 3192
    and ms[-1]['chain']['gap_v16'] == '0f6423e0' and ms[-2]['phase'] == 3191)

# 18 atlas html + SMOKE/FULL payload identity
hp = os.path.join(OUT, 'atlas_v1_20.html')
html = io.open(hp, encoding='utf-8').read()
chk(18, 'atlas_v1_20.html file sha 88300285 + data-k 754', sha8(hp) == '88300285'
    and html.count('data-k=') == 754, '%s/%d' % (sha8(hp), html.count('data-k=')))
sm = load(os.path.join(OUT, 'smoke_result.json'))
strip = lambda d: {k: v for k, v in d.items() if k not in ('mode', 'res_sha8', 'seal_sha8')}
chk(19, 'SMOKE/FULL arm payload bitwise identical', strip(sm) == strip(_tmp))

# MEMO + C-line close + prereg 3193
memo = io.open(os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md'), encoding='utf-8').read()
chk(20, 'MEMO 3192 block + C-line close + prereg 3193', 'Phase 3192' in memo
    and 'G5-C5' in memo and '3193 预注册' in memo
    and 'u*_amplitude_carrier' in memo and '五连' in memo)

n_pass = sum(1 for c in checks if c[2])
for i, name, okk, det in checks:
    print('%s #%02d %s %s' % ('PASS' if okk else 'FAIL', i, name, det))
print('== %d/%d ==' % (n_pass, len(checks)))
