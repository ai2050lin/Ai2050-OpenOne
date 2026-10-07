# -*- coding: utf-8 -*-
"""Phase 3073 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3073'
     r'\omega_p70_head_interaction')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3073_omega_'
                 r'p70_head_interaction.py')

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
    ('npz8', 'd2cdc763',
     sha8(R + r'\omega_p70_head_'
           r'interaction.npz')),
    ('result8', 'f4262f1d',
     sha8(R + r'\result.json')),
    ('exec8', 'b4e5ffa8',
     sha8(R + r'\execution.json')),
    ('script8', 'df1f0862', sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))
ok.append(('seal_npz8', seal['npz_sha256_8']
           == 'd2cdc763',
           seal['npz_sha256_8'], ''))
ok.append(('seal_result8', seal['result_sha256_8']
           == 'f4262f1d', '', ''))
ok.append(('seal_exec8', seal['exec_sha256_8']
           == 'b4e5ffa8', '', ''))
ok.append(('seal_script8', seal['script_sha256_8']
           == 'df1f0862', '', ''))

V = 'higher_order_required'
ok.append(('verdict', res['verdict'] == V,
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
an = res['anchors']
for nm in ('a1', 'a8', 'a9', 'a10', 'b8',
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
ok.append(('hl34', st['hl34_med']
           == 0.0018137704767155584, '', ''))
ok.append(('top8', st['top8']
           == [20, 7, 1, 14, 26, 0, 2, 24],
           '', ''))
ok.append(('r1_h20', st['r1'][0]
           == -0.22156993894701427, '', ''))
ok.append(('r1_h7', st['r1'][1]
           == -0.21310123529223413, '', ''))
ok.append(('r1_h1', st['r1'][2]
           == -0.20872026764068408, '', ''))
ok.append(('r1_h24', st['r1'][7]
           == -0.060069811227881575, '', ''))
ok.append(('imax', st['i_pair_max']
           == 0.10387255949605928, '', ''))
ok.append(('iargmax', st['i_argmax_pair']
           == [2, 3], '', ''))
ok.append(('nipos', st['n_ipos'] == 24, '', ''))
ok.append(('nineg', st['n_ineg'] == 4, '', ''))
ok.append(('r_u8', st['r_u8']
           == -0.5322981028358011, '', ''))
ok.append(('r_u8_pred', st['r_u8_pred']
           == -0.21182383293868212, '', ''))
ok.append(('pred_err', st['pred_err']
           == 0.320474269897119, '', ''))
ok.append(('t3', st['t3'] == [0, 1, 2], '', ''))
ok.append(('r_t3', st['r_t3']
           == -0.37527143927508755, '', ''))
ok.append(('r_t3_pred', st['r_t3_pred']
           == -0.42269733769576967, '', ''))
ok.append(('err3', st['err3']
           == 0.04742589842068212, '', ''))
ok.append(('inter_sig', st['inter_significant']
           is True, '', ''))
ok.append(('sp_I_r1', st['sp_I_r1prod']
           == 0.8215654077723045, '', ''))
ok.append(('sp_I_dah', st['sp_I_dahprod']
           == -0.6464148877941983, '', ''))
ok.append(('forwards', res['forwards'] == 975,
           res['forwards'], ''))

# ---------- ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ok.append(('ledger_n', len(led['measurements'])
           == 212, len(led['measurements']), ''))
ok.append(('l14_n', len(l14['connects']) == 180,
           len(l14['connects']), ''))
meas = [m for m in led['measurements']
        if m.get('phase') == 3073]
ok.append(('meas_exists', len(meas) == 1,
           len(meas), ''))
if meas:
    m0 = meas[0]
    ok.append(('meas_verdict',
               m0['verdict'] == V, '', ''))
    ok.append(('meas_hashes',
               m0['hashes']['npz_sha256_8']
               == 'd2cdc763'
               and m0['hashes']['result_sha256_8']
               == 'f4262f1d'
               and m0['hashes']['script_sha256_8']
               == 'df1f0862', '', ''))
    ok.append(('meas_in_l14',
               any(isinstance(c, dict)
                   and c.get('meas_id')
                   == 'meas3073_omega_p70_head_'
                      'interaction'
                   for c in l14['connects']),
               '', ''))
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
ok.append(('ledger_sha',
           hashlib.sha256(
               blob.encode('utf-8')
           ).hexdigest()[:8] == 'def8a9db',
           '', ''))

# ---------- MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
i73 = memo.find('## Phase 3073:')
i72 = memo.find('## Phase 3072:')
ok.append(('memo_phase73', i73 > 0, i73, ''))
ok.append(('memo_after_72',
           i72 >= 0 and i73 > i72, i72, i73))
ok.append(('memo_tail',
           memo.rstrip().endswith(
               u'\u5373\u8fdb 3074 A\u3002'),
           memo.rstrip()[-30:], ''))
for kw in (u'higher_order_required',
           u'\u5f3a\u6b21\u53ef\u52a0',
           u'\u4e8c\u9636\u5c55\u5f00',
           u'\u9971\u548c',
           u'\u4ea4\u4e92\u77e9\u9635',
           u'0.320474269897119',
           u'0.8215654077723045'):
    ok.append(('memo_kw_' + kw[:12],
               kw in memo, '', ''))
ok.append(('memo_created',
           u'[%s]' % exe['created'] in memo,
           exe['created'], ''))

# ---------- audit / wlog / memory ----------
aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_35',
           u'## \u4e09\u5341\u4e94\u30013073 \u589e'
           u'\u8865' in aud, '', ''))
wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3073', 'Phase 3073' in wl,
           '', ''))
mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3073', 'max=3073' in mw,
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
