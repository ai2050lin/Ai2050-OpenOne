# -*- coding: utf-8 -*-
"""Phase 3071 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3071'
     r'\omega_p68_attn_head_decomp')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3071_omega_'
                 r'p68_attn_head_decomp.py')

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
    ('npz8', 'a66a88a4',
     sha8(R + r'\omega_p68_attn_head_'
           r'decomp.npz')),
    ('result8', '2b871972',
     sha8(R + r'\result.json')),
    ('exec8', '598d3e06',
     sha8(R + r'\execution.json')),
    ('script8', '0d4a5f63', sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))
ok.append(('seal_npz8', seal['npz_sha256_8']
           == 'a66a88a4',
           seal['npz_sha256_8'], ''))
ok.append(('seal_result8', seal['result_sha256_8']
           == '2b871972', '', ''))
ok.append(('seal_exec8', seal['exec_sha256_8']
           == '598d3e06', '', ''))
ok.append(('seal_script8', seal['script_sha256_8']
           == '0d4a5f63', '', ''))

V = 'attn_heads_focal'
ok.append(('verdict', res['verdict'] == V,
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
an = res['anchors']
for nm in ('a1', 'a4', 'a5', 'a6', 'a7', 'b8',
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
dc = st['d34_cproj']
ok.append(('d34_attn', dc['attn']
           == 639.8104587682893, '', ''))
ok.append(('d34_mlp', dc['mlp']
           == -95.91242909117048, '', ''))
ok.append(('d34_out', dc['out']
           == 735.8797832320352, '', ''))
d5 = st['d35_cproj']
ok.append(('d35_mlp', d5['mlp']
           == -924.1991208657355, '', ''))
ok.append(('d35_attn', d5['attn']
           == -14.567961614698053, '', ''))
ok.append(('lincheck', st['lincheck_34']
           == 0.9999932519350134, '', ''))
ok.append(('cperm_g1', st['c_perm']
           ['g1_L34_MLP_ALL']
           == 0.10923343158032986, '', ''))
ok.append(('cperm_gA', st['c_perm']
           ['gA_L34_ATN_ALL']
           == -0.4229695021531773, '', ''))
ok.append(('cperm_gB', st['c_perm']
           ['gB_L35_ATN_ALL']
           == 0.10411487603305267, '', ''))
ok.append(('recov_gA', st['recov']
           ['gA_L34_ATN_ALL']
           == -0.5717521069904176, '', ''))
hd = st['head']
ok.append(('nneg', hd['n_neg'] == 16, '', ''))
ok.append(('capture8', hd['capture8']
           == 0.8355652268128613, '', ''))
ok.append(('top8', hd['top8']
           == [20, 7, 1, 14, 26, 0, 2, 24],
           '', ''))
ok.append(('r34_h20', hd['r34'][20]
           == -0.22156993894701427, '', ''))
ok.append(('r34_h7', hd['r34'][7]
           == -0.21310123529223413, '', ''))
ok.append(('r34_h1', hd['r34'][1]
           == -0.20872026764068408, '', ''))
ok.append(('r34_h14', hd['r34'][14]
           == -0.16739848114763328, '', ''))
ok.append(('r34_h26', hd['r34'][26]
           == -0.1574770002625661, '', ''))
ok.append(('l35_max', hd['l35_max']
           == 0.04555673004619386, '', ''))
ok.append(('l35_flat', hd['l35_flat'] is True,
           '', ''))
ok.append(('rank_corr', hd['rank_corr']
           == -0.23313782991202345, '', ''))
ok.append(('headlin34', hd['headlin34_med']
           == 0.0018137704767155584, '', ''))
ok.append(('headsum34', hd['headsum34']
           == 1056.7015853447947, '', ''))
ok.append(('quarters', hd['r_quarters']
           == [-0.3343023417835643,
               0.450906761459989,
               -0.10244623290529631,
               -0.22896925428307716], '', ''))
ok.append(('no_opb', hd['has_o_proj_bias']
           is False, '', ''))
ok.append(('fwd1767', res['forwards'] == 1767,
           '', ''))

led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
n = len(led['measurements'])
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
nl14 = len(l14['connects'])
ok.append(('ledger_n', n == 210, n, 210))
ok.append(('l14_n', nl14 == 178, nl14, 178))
has3071 = any(m.get('phase') == 3071
              for m in led['measurements'])
ok.append(('ledger_3071', has3071, '', ''))
sha_pop = dict(led)
expect = sha_pop.pop('ledger_sha256_8')
blob = json.dumps(sha_pop, sort_keys=True,
                  ensure_ascii=False)
calc = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ok.append(('ledger_sha', calc == expect, calc,
           expect))

memo = io.open(MEMO, encoding='utf-8').read()
i71 = memo.find('## Phase 3071:')
i70 = memo.find('## Phase 3070:')
ok.append(('memo_3071_present', i71 >= 0, i71, ''))
ok.append(('memo_3071_after_3070',
           i71 > i70 >= 0, i71, i70))
ok.append(('memo_no_3072', i71 > 0
           and '## Phase 3072:' not in memo[i71:],
           '', ''))
for key in ('attn_heads_focal', '0.836', 'h20',
            '0.046', '-0.233', '1767', 'axis=0',
            '0d4a5f63', '2b871972', 'a66a88a4',
            '598d3e06',
            '\u7126\u70b9\u5934',
            '\u8f74\u9677\u9631',
            '\u4f20\u8f93'):
    ok.append(('memo_key_' + key,
               key in memo[i71:], '', ''))
ok.append(('memo_len', len(memo) > 80000,
           len(memo), ''))

aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_33',
           '## 三十三、3071 增补' in aud, '', ''))

wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3071', 'Phase 3071' in wl, '', ''))

mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3071', 'max=3071' in mw,
           '', ''))

allok = all(t[1] for t in ok)
for t in ok:
    print('%-28s %s %s %s' % (t[0], 'PASS' if t[1]
                              else 'FAIL', t[2], t[3]))
print('VERIFY_%s %d items' % (
    'OK' if allok else 'FAILED', len(ok)))
