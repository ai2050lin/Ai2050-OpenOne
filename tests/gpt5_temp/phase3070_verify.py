# -*- coding: utf-8 -*-
"""Phase 3070 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3070'
     r'\omega_p67_l34_suppression_anatomy')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3070_omega_'
                 r'p67_l34_suppression_anatomy.py')

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
    ('npz8', seal['npz_sha256_8'],
     sha8(R + r'\omega_p67_l34_'
           r'suppression_anatomy.npz')),
    ('result8', seal['result_sha256_8'],
     sha8(R + r'\result.json')),
    ('exec8', seal['exec_sha256_8'],
     sha8(R + r'\execution.json')),
    ('script8', seal['script_sha256_8'],
     sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))

V = 'suppression_same_block_attn'
ok.append(('verdict', res['verdict'] == V,
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
an = res['anchors']
for nm in ('a1', 'a4', 'b8', 'aref', 'b0',
           'b1', 'b4', 'b6'):
    ok.append(('anchor_' + nm,
               an[nm + '_ok'] is True
               and an[nm + '_diff'] == 0.0, '', ''))
ok.append(('anchor_b7',
           an['b7_ok'] is True
           and an['b7a_diff'] == 0.0
           and an['b7c_diff'] == 0.0, '', ''))
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
ok.append(('d34_x0', dc['x'] == 0.0, '', ''))
d5 = st['d35_cproj']
ok.append(('d35_mlp', d5['mlp']
           == -924.1991208657355, '', ''))
ok.append(('d35_attn', d5['attn']
           == -14.567961614698053, '', ''))
nm_ = st['norm_med']
ok.append(('norm_a34', nm_['a34']
           == 260.1191951417096, '', ''))
ok.append(('norm_m34', nm_['m34']
           == 88.3442006058601, '', ''))
ok.append(('lincheck', st['lincheck_34']
           == 0.9999932519350134, '', ''))
rc = st['recov']
ok.append(('recov_g2', rc['g2_L34_ATN_ALL']
           == -0.5717521069904176, '', ''))
ok.append(('recov_g1', rc['g1_L34_MLP_ALL']
           == -0.03954917325691043, '', ''))
ok.append(('recov_g4', rc['g4_L35_ATN_ALL']
           == -0.04466772880418762, '', ''))
ok.append(('recov_g6', rc['g6_L34MLP_L35MLP']
           == 0.5867071840192604, '', ''))
ok.append(('recov_g7', rc['g7_L34_MLP_RAND']
           == 0.011506255629808754, '', ''))
cp = st['c_perm']
ok.append(('cperm_g2', cp['g2_L34_ATN_ALL']
           == -0.4229695021531773, '', ''))
ok.append(('cperm_g3eqg2',
           cp['g3_L34_BOTH']
           == cp['g2_L34_ATN_ALL'], '', ''))
ok.append(('fwd231', res['forwards'] == 231,
           '', ''))

led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
n = len(led['measurements'])
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
nl14 = len(l14['connects'])
ok.append(('ledger_n', n == 209, n, 209))
ok.append(('l14_n', nl14 == 177, nl14, 177))
has3070 = any(m.get('phase') == 3070
              for m in led['measurements'])
ok.append(('ledger_3070', has3070, '', ''))
sha_pop = dict(led)
expect = sha_pop.pop('ledger_sha256_8')
blob = json.dumps(sha_pop, sort_keys=True,
                  ensure_ascii=False)
calc = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ok.append(('ledger_sha', calc == expect, calc,
           expect))

memo = io.open(MEMO, encoding='utf-8').read()
i70 = memo.find('## Phase 3070:')
i69 = memo.find('## Phase 3069:')
ok.append(('memo_3070_present', i70 >= 0, i70, ''))
ok.append(('memo_3070_after_3069',
           i70 > i69 >= 0, i70, i69))
ok.append(('memo_no_3071', i70 > 0
           and '## Phase 3071:' not in memo[i70:],
           '', ''))
for key in ('suppression_same_block_attn',
            '640', '-96', '-0.572', '-0.040',
            '-924', '260', '88', '231', 'aref',
            '91b26c3a', '3a3f58d9', '98939a93',
            'ced332b1', '\u6d8c\u73b0',
            '\u5934\u7ea7\u5206\u89e3'):
    ok.append(('memo_key_' + key,
               key in memo[i70:], '', ''))
ok.append(('memo_len', len(memo) > 80000,
           len(memo), ''))

aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_32',
           '## 三十二、3070 增补' in aud, '', ''))

wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3070', 'Phase 3070' in wl, '', ''))

mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3070', 'max=3070' in mw,
           '', ''))

allok = all(t[1] for t in ok)
for t in ok:
    print('%-28s %s %s %s' % (t[0], 'PASS' if t[1]
                              else 'FAIL', t[2], t[3]))
print('VERIFY_%s %d items' % (
    'OK' if allok else 'FAILED', len(ok)))
