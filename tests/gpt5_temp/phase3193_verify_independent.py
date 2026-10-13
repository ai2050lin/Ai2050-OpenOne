# -*- coding: utf-8 -*-
"""Phase 3193 independent verification (20 checks, disk-read口径)."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3193\g5d1_ustar_decomposition')
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
chk(1, 'freeze execution.json design_sha8', ex.get('design_sha8') == '7af34fe6', str(ex.get('design_sha8')))

# 2 arm embedded res/seal anchors
rp = os.path.join(OUT, 'result.json')
_tmp = load(rp)
chk(2, 'arm embedded res/seal anchors', _tmp.get('res_sha8') == '5465dd47' and _tmp.get('seal_sha8') == 'ce1d8c8d',
    '%s/%s' % (_tmp.get('res_sha8'), _tmp.get('seal_sha8')))

# 3 NOVEL branch + all drift anchors 0
res = _tmp['per_model']
MS = ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
chk(3, 'NOVEL branch + drift69/89/90/91 == 0 x3',
    _tmp['gates']['branch'] == 'NOVEL'
    and all(res[m]['drift_3169'] == 0.0 and res[m]['drift_3189'] == 0.0
            and res[m]['drift_3190'] == 0.0 and res[m]['drift_3191'] == 0.0
            for m in MS))

# 4 gate values (4dp)
chk(4, 'gate_max (4dp)', [round(res[m]['gate_max'], 4) for m in MS] == [0.2130, 0.3303, 0.2457],
    str([round(res[m]['gate_max'], 4) for m in MS]))
chk(5, 'novel semantics: gate_max<0.3 in 2/3', sum(1 for m in MS if res[m]['gate_max'] < 0.3) == 2)
chk(6, 'cos_dvu orthogonal (<=0.11 x3)', all(abs(res[m]['cos_dvu']) <= 0.11 for m in MS),
    str([round(abs(res[m]['cos_dvu']), 4) for m in MS]))

# 7 3191 sealed scalars bitwise (device anchor)
r91 = load(os.path.join(OUT91, 'result.json'))
pm91 = r91['per_model']
chk(7, '3191 sealed top1/cross_med bitwise == 3193 arm', all(
    res[m]['top1'] == pm91[m]['top1'] and res[m]['cross_med'] == pm91[m]['cross_med'] for m in MS))

# 8 report-only cos_vc bitwise == 3190 (self-consistency)
r90 = load(os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913\phase3190\g5c3_shared_direction_robustness\result.json'))
pm90 = r90['per_model']
chk(8, 'report-only |cos(u*,v_c)| bitwise == 3190', all(
    res[m]['cos_vc_report'] == [abs(x) for x in pm90[m]['cos_ustar_vc']] for m in MS))

# 9 final
fin = load(os.path.join(OUT, 'final_result.json'))
chk(9, 'final res/seal field', fin.get('res_sha8') == 'e75c6e34' and fin.get('seal_sha8') == '9164b1ec',
    '%s/%s' % (fin.get('res_sha8'), fin.get('seal_sha8')))
chk(10, 'final arm per_model == arm result per_model (bitwise)',
    fin['arm']['per_model'] == _tmp['per_model'])
chk(11, 'final disposition NOVEL', fin.get('registry_disposition') == 'v1_18_to_v1_19_ftr26_novel_candidate'
    and fin.get('registry_render_sha8') == '4d64a153')

# 12 registry v1.19 file sha + append-only
reg19p = os.path.join(OUT, 'atlas_registry_v1_19.json')
reg18p = os.path.join(OUT91, 'atlas_registry_v1_18.json')
chk(12, 'registry v1.19 file sha 4d64a153', sha8(reg19p) == '4d64a153', sha8(reg19p))
r19, r18 = load(reg19p), load(reg18p)
chk(13, 'version/supersedes + features 25->26 append-only',
    r19['version'] == '1.19' and 'c08cbcc2' in r19.get('supersedes', '')
    and [f['id'] for f in r19['features']] == [f['id'] for f in r18['features']] + ['FTR-26'])
chk(14, '25 legacy features byte-preserved', all(
    json.dumps(f, ensure_ascii=False, sort_keys=True)
    == json.dumps([x for x in r19['features'] if x['id'] == f['id']][0], ensure_ascii=False, sort_keys=True)
    for f in r18['features']))
f26 = [f for f in r19['features'] if f['id'] == 'FTR-26'][0]
chk(15, 'FTR-26 candidate fields', f26['evidence_level'] == 'E1_repeatable'
    and f26['family'] == 'context' and [a['role'] for a in f26['anchors']] == ['novel_direction_gate']
    and f26['counter_evidence'] == [])
chk(16, 'failures/upgrade_log unchanged', json.dumps(r18['failures'], ensure_ascii=False, sort_keys=True)
    == json.dumps(r19['failures'], ensure_ascii=False, sort_keys=True)
    and json.dumps(r18['upgrade_log'], ensure_ascii=False, sort_keys=True)
    == json.dumps(r19['upgrade_log'], ensure_ascii=False, sort_keys=True))

# 17 ledger #345
led = load(os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json'))
ms = led['measurements']
chk(17, 'ledger n=345 last=3193 chain ok', len(ms) == 345 and ms[-1]['phase'] == 3193
    and ms[-1]['chain']['registry_v119'] == '4d64a153'
    and ms[-1]['chain']['design_sha8'] == '7af34fe6' and ms[-2]['phase'] == 3192)

# 18 atlas html + SMOKE/FULL payload identity
hp = os.path.join(OUT, 'atlas_v1_21.html')
html = io.open(hp, encoding='utf-8').read()
chk(18, 'atlas_v1_21.html file sha 5a6d219f + data-k 764', sha8(hp) == '5a6d219f'
    and html.count('data-k=') == 764, '%s/%d' % (sha8(hp), html.count('data-k=')))
sm = load(os.path.join(OUT, 'smoke_result.json'))
chk(19, 'SMOKE/FULL per_model payload bitwise identical', sm['per_model'] == _tmp['per_model'])

# MEMO + downgrade + refreeze honest log + prereg 3194
memo = io.open(os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md'), encoding='utf-8').read()
chk(20, 'MEMO 3193 block + downgrade + refreeze + prereg 3194', 'Phase 3193' in memo
    and 'G5-D1' in memo and '3194 预注册' in memo
    and 'FTR-26' in memo and '降级' in memo and '7af34fe6' in memo)

n_pass = sum(1 for c in checks if c[2])
for i, name, okk, det in checks:
    print('%s #%02d %s %s' % ('PASS' if okk else 'FAIL', i, name, det))
print('== %d/%d ==' % (n_pass, len(checks)))
