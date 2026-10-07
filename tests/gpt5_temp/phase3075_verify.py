# -*- coding: utf-8 -*-
"""Phase 3075 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3075'
     r'\omega_p72_supermodular_structure')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3075_omega_'
                 r'p72_supermodular_structure.py')

ok = []


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
exe = json.load(io.open(
    R + r'\execution.json', encoding='utf-8'))

checks = [
    ('npz8', 'ce51f6f6',
     sha8(R + r'\omega_p72_supermodular_'
           r'structure.npz')),
    ('result8', '200461ce',
     sha8(R + r'\result.json')),
    ('exec8', 'd0b2068e',
     sha8(R + r'\execution.json')),
    ('script8', '1a70c74f', sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))
ok.append(('seal_npz8', seal['npz_sha256_8']
           == 'ce51f6f6',
           seal['npz_sha256_8'], ''))
ok.append(('seal_result8', seal['result_sha256_8']
           == '200461ce', '', ''))
ok.append(('seal_exec8', seal['exec_sha256_8']
           == 'd0b2068e', '', ''))
ok.append(('seal_script8', seal['script_sha256_8']
           == '1a70c74f', '', ''))

V = 'supermodular_diffuse'
ok.append(('verdict', res['verdict'] == V,
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
an = res['anchors']
ok.append(('anchor_a1', an['a1_diff'] == 0.0
           and an['a1_ok'] is True, '', ''))
ok.append(('anchor_a2', an['a2_ok'] is True,
           '', ''))
ok.append(('anchor_a3', an['a3_ok'] is True,
           '', ''))
ok.append(('anchor_a4', an['a4_ok'] is True,
           '', ''))
ok.append(('anchor_a5', an['a5_ok'] is True,
           '', ''))
ok.append(('anchor_a6', an['a6_diff'] == 0.0
           and an['a6_ok'] is True, '', ''))
ok.append(('anchor_b0', an['b0_diff']
           == 3.3306690738754696e-16
           and an['b0_ok'] is True, '', ''))
st = res['stats']
ok.append(('n_viol', st['n_viol_re'] == 4359,
           '', ''))
ok.append(('viol_rate', st['viol_rate_re']
           == 0.26463088878096164, '', ''))
ok.append(('n_pos_mu', st['n_pos_mu'] == 88,
           '', ''))
ok.append(('pos_sum', abs(st['pos_sum']
           - 7.367974765012477) < 1e-12, '', ''))
ok.append(('c10', st['c10']
           == 0.3292310776105696, '', ''))
ok.append(('max_mu', st['max_mu_hi']
           == 0.34525139007733663, '', ''))
ok.append(('mu2_min', st['mu2_min']
           == -0.10387255949605928, '', ''))
ok.append(('mu2_max', st['mu2_max']
           == 0.005628104014067381, '', ''))
ok.append(('sp_m_viol', st['sp_m_viol']
           == 0.2142857142857143, '', ''))
ok.append(('sp_size_sm', st['sp_size_sm']
           == 0.5055599285568148, '', ''))
ok.append(('sm_count', st['sm_count'] == 467,
           '', ''))
ok.append(('sp_dah_m', st['sp_dah_m']
           == -0.7380952380952381, '', ''))
ok.append(('sp_dah_v', st['sp_dah_v']
           == -0.5952380952380953, '', ''))
ok.append(('w_census', st['w_pair_total']
           == 774 and st['w_gqa'] == 106
           and st['w_qtr'] == 188
           and st['base_gqa'] == 4
           and st['base_qtr'] == 7, '', ''))
spec = st['spectrum']
ok.append(('spec2', spec['2']['n_pos'] == 0
           and spec['2']['n'] == 28, '', ''))
ok.append(('spec3', spec['3']['n_pos'] == 5
           and spec['3']['max']
           == 0.05193340946245323, '', ''))
ok.append(('spec4', spec['4']['n_pos'] == 46
           and spec['4']['max']
           == 0.1570253024979793, '', ''))
ok.append(('spec6', spec['6']['n_pos'] == 19
           and spec['6']['max']
           == 0.34525139007733663, '', ''))
ok.append(('spec7', spec['7']['n_pos'] == 0
           and spec['7']['min']
           == -0.4062965144206656, '', ''))
ok.append(('spec8', spec['8']['max']
           == 0.30888883323097094, '', ''))
ok.append(('top1', st['top12_pos_mu'][0]
           == {'mask': 123, 'order': 6,
               'mu': 0.34525139007733663,
               'heads': [20, 7, 14, 26, 0, 2]},
           '', ''))
ok.append(('worst1', st['worst10'][0] == {
    'delta': 0.33719686075775934,
    'S_mask': 251, 'x': 2, 'T_mask': 186},
    '', ''))
ok.append(('rate_s7', st['rate_by_size']['7']
           == {'n': 1016, 'viol': 762,
               'rate': 0.75}, '', ''))
ok.append(('rate_b3', abs(
    st['rate_by_budget']['3']['rate']
    - 0.4733152096151095) < 1e-12, '', ''))
ok.append(('forwards', res['forwards'] == 0,
           '', ''))

# ---------- ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ok.append(('ledger_n', len(led['measurements'])
           == 214, len(led['measurements']), ''))
ok.append(('l14_n', len(l14['connects']) == 182,
           len(l14['connects']), ''))
meas = [m for m in led['measurements']
        if m.get('phase') == 3075]
ok.append(('meas_exists', len(meas) == 1,
           len(meas), ''))
if meas:
    m0 = meas[0]
    ok.append(('meas_verdict',
               m0['verdict'] == V, '', ''))
    ok.append(('meas_hashes',
               m0['hashes']['npz_sha256_8']
               == 'ce51f6f6'
               and m0['hashes']['result_sha256_8']
               == '200461ce'
               and m0['hashes']['script_sha256_8']
               == '1a70c74f', '', ''))
    ok.append(('meas_in_l14',
               any(isinstance(c, dict)
                   and c.get('meas_id')
                   == 'meas3075_omega_p72_supermodular_'
                      'structure'
                   for c in l14['connects']),
               '', ''))
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
ok.append(('ledger_sha',
           hashlib.sha256(
               blob.encode('utf-8')
           ).hexdigest()[:8] == 'e580acde',
           '', ''))

# ---------- MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
i75 = memo.find('## Phase 3075:')
i74 = memo.find('## Phase 3074:')
ok.append(('memo_phase75', i75 > 0, i75, ''))
ok.append(('memo_after_74',
           i74 >= 0 and i75 > i74, i74, i75))
ok.append(('memo_tail',
           memo.rstrip().endswith(
               u'\u5373\u8fdb 3076 A\u3002'),
           memo.rstrip()[-30:], ''))
for kw in (u'supermodular_diffuse',
           u'M\u00f6bius',
           u'0.34525139007733663',
           u'0.30888883323097094',
           u'0.3292310776105696',
           u'0.5055599285568148',
           u'0.7380952380952381',
           u'16472',
           u'\u53cc\u533a',
           u'\u5b8c\u5907\u6027',
           u'17496'):
    ok.append(('memo_kw_' + kw[:12],
               kw in memo, '', ''))
ok.append(('memo_created',
           u'[%s]' % exe['created'] in memo,
           exe['created'], ''))

# ---------- audit / wlog / memory ----------
aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_37',
           u'## \u4e09\u5341\u4e03\u30013075 \u589e'
           u'\u8865' in aud, '', ''))
wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3075', 'Phase 3075' in wl,
           '', ''))
mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3075', 'max=3075' in mw,
           '', ''))

n_pass = sum(1 for c in ok if c[1])
for c in ok:
    if not c[1]:
        print('FAIL: %s expected=%r got=%r'
              % (c[0], c[2], c[3]))
print('VERIFY %d/%d'
      % (n_pass, len(ok)))
if n_pass == len(ok):
    print('VERIFY_OK')
