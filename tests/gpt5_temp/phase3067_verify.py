# -*- coding: utf-8 -*-
"""Phase 3067 independent verify: recompute all
seals from disk bytes, ledger shas, MEMO placement
and key numbers, audit/wlog/memory presence."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3067'
     r'\omega_p64_mlp_conditional_reversal')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCRIPT = ROOT + (r'\tests\glm5\phase3067_omega_'
                 r'p64_mlp_conditional_reversal.py')

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
     sha8(R + r'\omega_p64_mlp_conditional_'
           r'reversal.npz')),
    ('result8', seal['result_sha256_8'],
     sha8(R + r'\result.json')),
    ('exec8', seal['exec_sha256_8'],
     sha8(R + r'\execution.json')),
    ('script8', seal['script_sha256_8'],
     sha8(SCRIPT)),
]
for nm, a, b in checks:
    ok.append((nm, a == b, a, b))

ok.append(('verdict', res['verdict'] ==
           'mlp_reversal_localized_shared',
           res['verdict'], ''))
ok.append(('setup', res['anchors']['setup_ok']
           is True, '', ''))
ok.append(('a1', res['anchors']['a1_diff'] == 0.0,
           '', ''))

led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
n = len(led['measurements'])
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
nl14 = len(l14['connects'])
ok.append(('ledger_n', n == 206, n, 206))
ok.append(('l14_n', nl14 == 174, nl14, 174))
has3067 = any(m.get('phase') == 3067
              for m in led['measurements'])
ok.append(('ledger_3067', has3067, '', ''))
sha_pop = dict(led)
expect = sha_pop.pop('ledger_sha256_8')
blob = json.dumps(sha_pop, sort_keys=True,
                  ensure_ascii=False)
calc = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ok.append(('ledger_sha', calc == expect, calc,
           expect))

memo = io.open(MEMO, encoding='utf-8').read()
i67 = memo.find('## Phase 3067:')
i66 = memo.find('## Phase 3066:')
ok.append(('memo_3067_present', i67 >= 0, i67, ''))
ok.append(('memo_3067_after_3066',
           i67 > i66 >= 0, i67, i66))
for key in ('mlp_reversal_localized_shared',
            '0.706', '-0.0008', '0.389',
            '0.992', '0.976', '0.028', '0.538',
            '+0.350', '158'):
    ok.append(('memo_key_' + key,
               key in memo[i67:], '', ''))
ok.append(('memo_len', len(memo) > 80000,
           len(memo), ''))

aud = io.open(AUDIT, encoding='utf-8').read()
ok.append(('audit_29',
           '## 二十九、3067 增补' in aud, '', ''))

wl = io.open(WLOG, encoding='utf-8').read()
ok.append(('wlog_3067', 'Phase 3067' in wl, '', ''))

mw = io.open(MEMO_W, encoding='utf-8').read()
ok.append(('memory_max3067', 'max=3067' in mw,
           '', ''))

allok = all(t[1] for t in ok)
for t in ok:
    print('%-28s %s %s %s' % (t[0], 'PASS' if t[1]
                              else 'FAIL', t[2], t[3]))
print('VERIFY_%s %d items' % (
    'OK' if allok else 'FAILED', len(ok)))
