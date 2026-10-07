# -*- coding: utf-8 -*-
"""Phase 3074 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3074'
     r'\omega_p71_capacity_law')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3074_omega_'
                 r'p71_capacity_law.py')

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
    ('npz8', '1dcaff27',
     sha8(R + r'\omega_p71_capacity_law.npz')),
    ('result8', '995d5a24',
     sha8(R + r'\result.json')),
    ('exec8', 'e71c3ad4',
     sha8(R + r'\execution.json')),
    ('script8', 'bdbdf942', sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))
ok.append(('seal_npz8', seal['npz_sha256_8']
           == '1dcaff27',
           seal['npz_sha256_8'], ''))
ok.append(('seal_result8', seal['result_sha256_8']
           == '995d5a24', '', ''))
ok.append(('seal_exec8', seal['exec_sha256_8']
           == 'e71c3ad4', '', ''))
ok.append(('seal_script8', seal['script_sha256_8']
           == 'bdbdf942', '', ''))

V = 'capacity_law_hill'
ok.append(('verdict', res['verdict'] == V,
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
an = res['anchors']
for nm in ('a1', 'a8', 'a9', 'a10', 'a11', 'b8',
           'b0', 'b1', 'b4', 'b6'):
    ok.append(('anchor_' + nm,
               an[nm + '_ok'] is True
               and an[nm + '_diff'] == 0.0, '', ''))
ok.append(('anchor_aref',
           an['aref_ok'] is True
           and an['aref_diff'] == 0.0, '', ''))
ok.append(('anchor_b7',
           an['b7_ok'] is True
           and an['b7a_diff'] == 0.0
           and an['b7c_diff'] == 0.0, '', ''))
ok.append(('a2lens', an['a2lens_max']
           == 0.062412261962890625, '', ''))
st = res['stats']
ok.append(('medc34', st['med_c_34']
           == 0.1487826048372403, '', ''))
ok.append(('pa34', st['pa34']
           == 0.29685845971107483, '', ''))
ok.append(('pf34', st['pf34']
           == 0.2371114194393158, '', ''))
ok.append(('top8', st['top8']
           == [20, 7, 1, 14, 26, 0, 2, 24],
           '', ''))
ok.append(('r_all32', st['r_all32']
           == -0.5717521069904176, '', ''))
ok.append(('a_u8', st['a_u8']
           == 0.5322981028358011, '', ''))
ok.append(('a_best', st['a_best_subset']
           == {'mask': 255,
               'amp': 0.5322981028358011}, '', ''))
ok.append(('x_u8', st['x_u8']
           == 1.219138699856578, '', ''))
ok.append(('x_all32', st['x_all32']
           == 1.9917079240985371, '', ''))
ok.append(('n_sub', st['n_sub_checked'] == 16472,
           st['n_sub_checked'], ''))
ok.append(('n_viol', st['n_viol'] == 4359,
           st['n_viol'], ''))
ok.append(('viol_rate', abs(st['viol_rate']
           - 0.26463088878096164) < 1e-12, '', ''))
ok.append(('sub_ok_false',
           st['submodular_ok'] is False, '', ''))
fits = st['fits']
ok.append(('fits_keys',
           set(fits) == {'add', 'exp', 'log',
                         'hill'}, '', ''))
ok.append(('add_err_u8', fits['add']['err_u8']
           == 0.6868405970207769, '', ''))
ok.append(('add_err_32', fits['add']['err_all32']
           == 1.4199558171081197, '', ''))
ok.append(('exp_err_u8', fits['exp']['err_u8']
           == 0.03984606949404468, '', ''))
ok.append(('exp_err_32', fits['exp']['err_all32']
           == 0.05364586169328667, '', ''))
ok.append(('exp_gates', fits['exp']['pass_u8']
           is True
           and fits['exp']['pass_all32'] is False,
           '', ''))
ok.append(('log_err_u8', fits['log']['err_u8']
           == 0.1325037080319692, '', ''))
ok.append(('hill_p_le2', fits['hill']['p_le2']
           == [0.7142857142857143,
               0.45409444455802084,
               1.1857142857142857], '', ''))
ok.append(('hill_sse_le2',
           fits['hill']['sse_le2']
           == 0.010513313833541609, '', ''))
ok.append(('hill_pred_u8', fits['hill']['pred_u8']
           == 0.5452333231103856, '', ''))
ok.append(('hill_err_u8', fits['hill']['err_u8']
           == 0.012935220274584491, '', ''))
ok.append(('hill_pred_32',
           fits['hill']['pred_all32']
           == 0.6088086522043087, '', ''))
ok.append(('hill_err_32', fits['hill']['err_all32']
           == 0.03705654521389112, '', ''))
ok.append(('hill_gates', fits['hill']['pass_u8']
           is True
           and fits['hill']['pass_all32'] is True,
           '', ''))
ok.append(('hill_sse_all',
           fits['hill']['sse_all']
           == 0.6588243184287355, '', ''))
ok.append(('forwards', res['forwards'] == 6207,
           res['forwards'], ''))

# ---------- ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ok.append(('ledger_n', len(led['measurements'])
           == 213, len(led['measurements']), ''))
ok.append(('l14_n', len(l14['connects']) == 181,
           len(l14['connects']), ''))
meas = [m for m in led['measurements']
        if m.get('phase') == 3074]
ok.append(('meas_exists', len(meas) == 1,
           len(meas), ''))
if meas:
    m0 = meas[0]
    ok.append(('meas_verdict',
               m0['verdict'] == V, '', ''))
    ok.append(('meas_hashes',
               m0['hashes']['npz_sha256_8']
               == '1dcaff27'
               and m0['hashes']['result_sha256_8']
               == '995d5a24'
               and m0['hashes']['script_sha256_8']
               == 'bdbdf942', '', ''))
    ok.append(('meas_in_l14',
               any(isinstance(c, dict)
                   and c.get('meas_id')
                   == 'meas3074_omega_p71_capacity_'
                      'law'
                   for c in l14['connects']),
               '', ''))
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
ok.append(('ledger_sha',
           hashlib.sha256(
               blob.encode('utf-8')
           ).hexdigest()[:8] == 'baa4e21b',
           '', ''))

# ---------- MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
i74 = memo.find('## Phase 3074:')
i73 = memo.find('## Phase 3073:')
ok.append(('memo_phase74', i74 > 0, i74, ''))
ok.append(('memo_after_73',
           i73 >= 0 and i74 > i73, i73, i74))
ok.append(('memo_tail',
           memo.rstrip().endswith(
               u'\u5373\u8fdb 3075 A\u3002'),
           memo.rstrip()[-30:], ''))
for kw in (u'capacity_law_hill',
           u'Hill',
           u'0.012935220274584491',
           u'0.03705654521389112',
           u'0.7142857142857143',
           u'1.1857142857142857',
           u'16472',
           u'26.5',
           u'\u5bb9\u91cf\u5b9a\u5f8b',
           u'\u8d85\u6a21',
           u'-0.5717521069904176'):
    ok.append(('memo_kw_' + kw[:12],
               kw in memo, '', ''))
ok.append(('memo_created',
           u'[%s]' % exe['created'] in memo,
           exe['created'], ''))

# ---------- audit / wlog / memory ----------
aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_36',
           u'## \u4e09\u5341\u516d\u30013074 \u589e'
           u'\u8865' in aud, '', ''))
wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3074', 'Phase 3074' in wl,
           '', ''))
mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3074', 'max=3074' in mw,
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
