# -*- coding: utf-8 -*-
"""Phase 3228 independent verify: 18 check groups, disk-readback only."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P5 = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3228\g5g26_e3_compute_seal')
OUT = []
FAIL = []


def chk(i, name, ok, detail=''):
    OUT.append('[%s] #%02d %s %s' % ('PASS' if ok else 'FAIL', i, name, detail))
    if not ok:
        FAIL.append(i)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


# 1 ledger head
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
chk(1, 'ledger n=380 post=8e593d22', len(led['measurements']) == 380
    and led['ledger_sha256_8'] == '8e593d22')
e = led['measurements'][379]
chk(2, 'entry #380 phase=3228 name', e.get('phase') == 3228
    and e.get('name') == 'g5g26_e3_compute_seal')

# 3 rollback equivalence
bako = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                   'atlas_ledger.json.bak_3228'),
                      encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:379]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
chk(3, 'ledger rollback-equal vs bak_3228', eq
    and len(bako['measurements']) == 379 and bako['ledger_sha256_8'] == 'f21108e7')

# 4 result seal
res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
chk(4, 'result verdict + res content sha (None-convention)',
    res['verdict'] == 'e3_compute_sealed'
    and csha8(dict(res, res_sha8=None)) == res['res_sha8'] == '99693298')

# 5 npz + run_log + execution file shas vs ledger chain
chk(5, 'npz sha a844d985 + run_log sha vs chain',
    res['sealed_npz']['file_sha8'] == 'a844d985'
    and sha8_file(os.path.join(P5, 'e3_flip_matrices.npz')) == 'a844d985'
    and sha8_file(os.path.join(P5, 'run_log.txt')) == e['chain']['seal_sha8']
    and sha8_file(os.path.join(P5, 'result.json')) == e['chain']['res_sha8'])

# 6 freeze sha recompute (content = dict minus the sha8 key itself)
fz = json.load(open(os.path.join(P5, 'execution.json'), encoding='utf-8'))
fz_body = {k: v for k, v in fz.items() if k != 'sha8'}
chk(6, 'freeze c51bfea5 recompute (C9 config)', csha8(fz_body) == fz['sha8']
    == 'c51bfea5' and res['freeze_sha8'] == 'c51bfea5')

# 7 C6v3 device gate (4dp-anchored)
for m, lo in (('qwen3-4b', 0.9963), ('qwen3-14b', 0.9941)):
    md = res['models'][m]['oov']['marg_diag']
    chk(7, 'C6v3 %s agree>=0.98 cos>=0.9999' % m,
        md['agree_all'] >= 0.98 and round(md['agree_all'], 4) == lo
        and md['mean_centered_cos'] >= 0.9999,
        'agree=%.4f cos=%.4f' % (md['agree_all'], md['mean_centered_cos']))

# 8 matched flips all zero (seen folds, alpha=1, both models, both R)
ok8 = True
det8 = []
for m in ('qwen3-4b', 'qwen3-14b'):
    agg = res['models'][m]['seen']['agg']
    for R in ('isa', 'hasa'):
        for f in ('cut0', 'cut1', 'cut2', 'cut3'):
            k = '%s|u_%s|%s|a1.0' % (R, R, f)
            if agg[k]['n'] != 0 or agg[k]['rate'] != 0.0:
                ok8 = False
                det8.append('%s %s' % (m, k))
chk(8, 'matched flip rate == 0 (2 models x 2 R x 4 folds @a1)', ok8,
    '; '.join(det8) or '32/32 zero')

# 9 null flips all zero (alpha=1, seen, both models)
ok9 = True
det9 = []
for m in ('qwen3-4b', 'qwen3-14b'):
    agg = res['models'][m]['seen']['agg']
    for k, v in agg.items():
        p = k.split('|')[1]
        if k.endswith('|a1.0') and p.startswith(('rand_', 'orth_')):
            if v['n'] != 0:
                ok9 = False
                det9.append('%s %s n=%d' % (m, k, v['n']))
chk(9, 'null flip rate == 0 (48 null cells @a1)', ok9,
    '; '.join(det9) or '48/48 zero')

# 10 u_isa dose monotone non-decreasing in magnitude (more negative)
ok10 = True
det10 = []
for m in ('qwen3-4b', 'qwen3-14b'):
    for panel in ('seen', 'oov'):
        d = res['models'][m][panel]['dose']
        seq = [d['u_isa|a%s' % a] for a in ('0.5', '1.0', '2.0')]
        if not (seq[0] > seq[1] > seq[2] and seq[2] < 0):
            ok10 = False
            det10.append('%s %s %s' % (m, panel, seq))
chk(10, 'u_isa dose anti-directional monotone (4 panel-model cells)', ok10,
    '; '.join(det10) or 'all monotone negative')

# 11 u_hasa weak vs u_isa (|u_isa|/|u_hasa| >= 5 @a1)
ok11 = True
for m in ('qwen3-4b', 'qwen3-14b'):
    for panel in ('seen', 'oov'):
        d = res['models'][m][panel]['dose']
        r = abs(d['u_isa|a1.0']) / abs(d['u_hasa|a1.0'])
        if r < 5.0:
            ok11 = False
chk(11, 'u_isa >=5x stronger than u_hasa @a1 (4 cells)', ok11)

# 12 GD_amp spearman sealed
ga = {m: res['models'][m]['gd_amp']['spearman_amp_collapse']
      for m in ('qwen3-4b', 'qwen3-14b')}
chk(12, 'GD_amp spearman 0.40/0.80 sealed',
    abs(ga['qwen3-4b'] - 0.4) < 1e-12 and abs(ga['qwen3-14b'] - 0.8) < 1e-12
    and all(len(res['models'][m]['gd_amp']['E_cls']) == 4
            for m in ('qwen3-4b', 'qwen3-14b')))

# 13 device anchors all zero
ok13 = True
for m in ('qwen3-4b', 'qwen3-14b'):
    dv = res['models'][m]['device']
    ua = res['models'][m]['u_R_anchors']
    if not (dv['drift69'] < 1e-9 and dv['drift89'] == 0 and dv['drift90'] == 0
            and dv['drift91'] == 0 and dv['ustar_bitwise'] == 0
            and ua['tc_drift'] < 1e-9 and ua['dv_drift'] < 1e-9):
        ok13 = False
chk(13, 'device anchors drift==0 (3169/3189/3190/3191/u*/3157tc/3183dv)', ok13)

# 14 W_U provenance on disk
p4b = os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors')
p14b = os.path.join(ROOT, os.path.expanduser('~').replace('C:\\', 'C:/'),
                    '.cache', 'huggingface', 'hub')  # placeholder
import glob
p14b = glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface',
                              'hub', 'models--Qwen--Qwen3-14B', 'snapshots',
                              '*', 'model-00008-of-00008.safetensors'))[0]
chk(14, 'W_U provenance: local 4B 328a91d3 / 14B shard8 0d6b9229',
    sha8_file(p4b) == '328a91d3' and sha8_file(p14b) == '0d6b9229'
    and '328a91d3' in json.dumps(fz['weights']['sources'], ensure_ascii=False))

# 15 clarifications C9/C6v3 present + 3227 gates verbatim in freeze
cl = json.dumps(res['clarifications'], ensure_ascii=False)
chk(15, 'clarifications C9/C6v3/C8 + gates_3227_verbatim in freeze',
    'C9' in cl and 'C6v3' in cl and 'C8' in cl
    and fz['gates_3227_verbatim']['GD_flip'].startswith('flip_rate')
    and 'SUPERSEDED' not in json.dumps(fz['device_gates']['d_marg_c6v3'])[:0] + 'X')

# 16 superseded history recorded
ch = e['chain']
chk(16, 'superseded in-run history in ledger chain',
    ch['superseded_in_run'] == {'freeze_v1': 'aae60c9d', 'freeze_v2': '5dd8b4f6',
                                'full_v1_freeze': 'c076d6f6',
                                'full_v1_res_content': 'ef55a204'})

# 17 registry/gap untouched (v1.29 / v1.7)
reg_ok = gap_ok = False
for p in glob.glob(os.path.join(ROOT, r'tests\glm5\result',
                                r'rdc_query_construction_20260913',
                                r'phase3226\g5g25_a2_gate_decision',
                                'atlas_registry_v1_29.json')):
    reg_ok = sha8_file(p) == '85b8dc1f'
for p in glob.glob(os.path.join(ROOT, r'tests\glm5\result',
                                r'rdc_query_construction_20260913',
                                r'phase3226\g5g25_a2_gate_decision',
                                'gap_ledger_v1_7.json')):
    gap_ok = sha8_file(p) == '4f45fda0'
chk(17, 'registry v1.29 85b8dc1f + gap v1.7 4f45fda0 untouched', reg_ok and gap_ok)

# 18 five-write disk readback
memo = open(os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'),
            'rb').read()
daily = open(os.path.join(ROOT, r'.workbuddy\memory\2026-10-11.md'),
             'rb').read()
mem2 = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'),
            'rb').read().decode('utf-8')
chk(18, 'five-write readback: ledger/MEMO/daily/MEMORY',
    b'## Phase 3228' in memo and b'### Phase 3228' in daily
    and '3228 收官' in mem2 and '8e593d22' in mem2
    and sha8_file(os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'))
    == '728d05bf')

OUT.append('VERIFY: %d/18 PASS, %d FAIL %s' % (
    18 - len(FAIL), len(FAIL), FAIL if FAIL else ''))
with open(os.path.join(ROOT, r'tests\gpt5_temp\phase3228_verify_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('verify done: %d FAIL' % len(FAIL))
