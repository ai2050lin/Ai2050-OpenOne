# -*- coding: utf-8 -*-
"""Phase 3072 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3072'
     r'\omega_p69_focal_head_lineage')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3072_omega_'
                 r'p69_focal_head_lineage.py')

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
    ('npz8', 'e04811bf',
     sha8(R + r'\omega_p69_focal_head_'
           r'lineage.npz')),
    ('result8', '308c540f',
     sha8(R + r'\result.json')),
    ('exec8', '3a6659aa',
     sha8(R + r'\execution.json')),
    ('script8', '94ed8c63', sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))
ok.append(('seal_npz8', seal['npz_sha256_8']
           == 'e04811bf',
           seal['npz_sha256_8'], ''))
ok.append(('seal_result8', seal['result_sha256_8']
           == '308c540f', '', ''))
ok.append(('seal_exec8', seal['exec_sha256_8']
           == '3a6659aa', '', ''))
ok.append(('seal_script8', seal['script_sha256_8']
           == '94ed8c63', '', ''))

V = 'focal_lineage_full'
ok.append(('verdict', res['verdict'] == V,
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
an = res['anchors']
for nm in ('a1', 'a7', 'a8', 'a9', 'b9', 'b10',
           'b8', 'b0', 'b1', 'b4', 'b6'):
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
dc = st['d34_cproj']
ok.append(('d34_attn', dc['attn']
           == 639.8104587682893, '', ''))
ok.append(('d34_mlp', dc['mlp']
           == -95.91242909117048, '', ''))
ok.append(('d34_out', dc['out']
           == 735.8797832320352, '', ''))
d5 = st['d35_cproj']
ok.append(('d35_attn', d5['attn']
           == -14.567961614698053, '', ''))
ok.append(('d35_mlp', d5['mlp']
           == -924.1991208657355, '', ''))
ok.append(('d35_out', d5['out']
           == -191.78616219035163, '', ''))
ok.append(('lincheck', st['lincheck_34']
           == 0.9999932519350134, '', ''))
pb = st['probe']
ok.append(('b9_diff', pb['b9_diff'] == 0.0, '', ''))
ok.append(('b10_diff', pb['b10_diff'] == 0.0,
           '', ''))
ok.append(('b10_35_max', pb['b10_35_max']
           == 0.76220703125, '', ''))
ok.append(('probe_ok', pb['probe_ok'] is True,
           '', ''))
ln = st['lineage']
ok.append(('focal5', ln['focal5']
           == [20, 7, 1, 14, 26], '', ''))
ok.append(('med_rel_joint', ln['med_rel_joint']
           == 0.001736582162660769, '', ''))
ok.append(('top8_min_cos', ln['top8_min_cos']
           == 0.9999986141965966, '', ''))
ok.append(('all32_med_cos', ln['all32_med_cos']
           == 0.9999986168914943, '', ''))
ls = st['listener']
ok.append(('sp_m34_absR', ls['sp_m34_absR']
           == 0.5454545454545454, '', ''))
ok.append(('sp_m34_absDah', ls['sp_m34_absDah']
           == 0.40725806451612906, '', ''))
ok.append(('sp_m35_absR35', ls['sp_m35_absR35']
           == -0.4274193548387097, '', ''))
m34 = ls['m34_med']
for h, v in ((20, 0.988), (7, 0.979),
             (1, 0.978), (14, 0.990),
             (26, 0.994)):
    ok.append(('m34_h%d' % h,
               abs(m34[h] - v) < 2e-3,
               m34[h], v))
ov = st['ov']
ok.append(('ov_sel', ov['sel_heads']
           == [1, 6, 7, 14, 20, 24, 26], '', ''))
ok.append(('ov_trank_h1', ov['target_rank_med'][0]
           == 74738.5, '', ''))
ok.append(('forwards', res['forwards'] == 111,
           res['forwards'], ''))

# ---------- ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ok.append(('ledger_n', len(led['measurements'])
           == 211, len(led['measurements']), ''))
ok.append(('l14_n', len(l14['connects']) == 179,
           len(l14['connects']), ''))
meas = [m for m in led['measurements']
        if m.get('phase') == 3072]
ok.append(('meas_exists', len(meas) == 1,
           len(meas), ''))
if meas:
    m0 = meas[0]
    ok.append(('meas_verdict',
               m0['verdict'] == V, '', ''))
    ok.append(('meas_hashes',
               m0['hashes']['npz_sha256_8']
               == 'e04811bf'
               and m0['hashes']['result_sha256_8']
               == '308c540f'
               and m0['hashes']['script_sha256_8']
               == '94ed8c63', '', ''))
    ok.append(('meas_in_l14',
               any(isinstance(c, dict)
                   and c.get('meas_id')
                   == 'meas3072_omega_p69_focal_'
                      'head_lineage'
                   for c in l14['connects']),
               '', ''))
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
ok.append(('ledger_sha',
           hashlib.sha256(
               blob.encode('utf-8')
           ).hexdigest()[:8] == 'a34ab515',
           '', ''))

# ---------- MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
i72 = memo.find('## Phase 3072:')
i71 = memo.find('## Phase 3071:')
ok.append(('memo_phase72', i72 > 0, i72, ''))
ok.append(('memo_after_71',
           i71 >= 0 and i72 > i71, i71, i72))
ok.append(('memo_tail',
           memo.rstrip().endswith(
               u'\u5373\u8fdb 3073 A\u3002'),
           memo.rstrip()[-30:], ''))
for kw in (u'focal_lineage_full',
           u'\u7ebf\u6027\u8c31\u7cfb',
           u'\u503e\u542c\u8005',
           u'theatre',
           u'\u8c31\u7cfb\u6052\u7b49\u5f0f',
           u'g(h)=h//4',
           '0.001736582162660769'):
    ok.append(('memo_kw_' + kw[:12],
               kw in memo, '', ''))
ok.append(('memo_created',
           u'[%s]' % exe['created'] in memo,
           exe['created'], ''))

# ---------- audit / wlog / memory ----------
aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_34',
           u'## \u4e09\u5341\u56db\u30013072 \u589e'
           u'\u8865' in aud, '', ''))
wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3072', 'Phase 3072' in wl,
           '', ''))
mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3072', 'max=3072' in mw,
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
