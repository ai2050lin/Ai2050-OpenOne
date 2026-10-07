# -*- coding: utf-8 -*-
"""Phase 3145 independent disk verify:
re-checks every write of the closeout
against the REAL disk (fresh reads, no
in-memory state)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D45 = os.path.join(
    RDIR, 'phase3145',
    'omega_p143_v1clip_pc1spec_'
    'headsign_residcausal')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
P_LOG = (ROOT + '\\' + '.workbuddy'
         + '\\' + 'memory' + '\\'
         + '2026-09-30.md')
P_MEM = (ROOT + '\\' + '.workbuddy'
         + '\\' + 'memory' + '\\'
         + 'MEMORY.md')
CKS = []
N = [0]


def ck(name, ok):
    N[0] += 1
    CKS.append('%s %s' % (
        'PASS' if ok else 'FAIL', name))


# 1. result.json + hash + verdict
fp = os.path.join(D45, 'result.json')
ck('result.json exists', os.path.exists(fp))
raw = io.open(fp, 'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
ck('result sha8=9d9cc6d1',
   sha == '9d9cc6d1')
r = json.loads(raw.decode('utf-8'))
ck('smoke False', r['smoke'] is False)
ck('verdict 20-tag', r['verdict'].count(
    '|') == 19 and 'a_3144_ok'
   in r['verdict'] and 'coverage_full'
   in r['verdict'])
ck('seal 46c0187b',
   str(r['seal_sha8']) == '46c0187b')
ck('bit 5/5 in verdict',
   'repro_bit_5|repro_bit_ok'
   in r['verdict'])
ck('resid_anchor_ok in verdict',
   'resid_anchor_ok' in r['verdict'])
ck('tail_dose_mono in verdict',
   'tail_dose_mono' in r['verdict'])
ck('answer_side_yes in verdict',
   'answer_side_yes' in r['verdict'])

# 2. seal + npz + log
ck('design_seal.json exists',
   os.path.exists(os.path.join(
       D45, 'design_seal.json')))
ck('p143_readout.npz exists',
   os.path.exists(os.path.join(
       D45, 'p143_readout.npz')))
lp = os.path.join(D45, 'run_log.txt')
ck('run_log exists', os.path.exists(lp))
logt = io.open(lp, encoding='utf-8').read()
ck('log has DONE',
   'DONE (verdict' in logt)
ck('log has 5/5 bit',
   'D replay: +2 bit anchors (total '
   '5/5)' in logt)
ck('log dvec19 drift 0',
   'drift 0.00e+00' in logt)
ck('log xphase P 1.0',
   'xphase P: session-base vs z26 '
   'bit-match 1.0000 (128/128)'
   in logt)
ck('log xphase A1 1.0',
   'xphase A1: session-base vs z26 '
   'bit-match 1.0000 (128/128)'
   in logt)
ck('ckpt cleaned',
   not os.path.exists(os.path.join(
       D45, 'p143_ckpt.pkl')))

# 3. ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
n = len(led['measurements'])
ck('ledger n=282', n == 282)
last = led['measurements'][-1]
ck('ledger last phase=3145',
   last.get('phase') == 3145)
ck('ledger 3145 sha',
   last['hashes']['result_sha256_8']
   == '9d9cc6d1')
ck('ledger 3145 seal',
   last['hashes']['seal_sha256_8']
   == '46c0187b')
ck('ledger 3145 verdict matches',
   last['verdict'] == r['verdict'])
ck('ledger no dup 3145',
   sum(1 for m in led['measurements']
       if m.get('phase') == 3145) == 1)

# 4. MEMO
mt = io.open(MEMO, encoding='utf-8').read()
ck('MEMO Phase 3145 section',
   mt.count('## Phase 3145') == 1)
ck('MEMO 3146 prereg',
   '3146（Ω-P144）预注册' in mt)
ck('MEMO 3145 anchors',
   'result sha8=9d9cc6d1' in mt)
ck('MEMO 3145 ledger n=282',
   'ledger n=282' in mt)
ck('MEMO T4 28th',
   'T4 第28 Phase' in mt)
ck('MEMO amp correction',
   'v1_amp_dv29dom' in mt)
ck('MEMO tail dose',
   'tail_dose_mono' in mt)
ck('MEMO answer_side_yes',
   'answer_side_yes' in mt)

# 5. daily log
ck('daily log exists',
   os.path.exists(P_LOG))
dt = io.open(P_LOG, encoding='utf-8').read()
ck('daily log 3145 entry',
   'Phase 3145 (Omega-P143) 闭环'
   in dt)
ck('daily log verdict',
   'v1_amp_dv29dom' in dt)
ck('daily log sha',
   '9d9cc6d1' in dt)

# 6. MEMORY.md
mm = io.open(P_MEM, encoding='utf-8').read()
ck('MEMORY max=3145',
   '- max=3145，下一 3146' in mm)
ck('MEMORY old line removed',
   '- max=3144，下一 3145' not in mm)
ck('MEMORY tags v1amp',
   'v1_amp_dv29dom' in mm)
ck('MEMORY tags taildose',
   'tail_dose_mono' in mm)
ck('MEMORY tags residcausal',
   'resid_causal_active' in mm
   or '0.984/0.422' in mm)
ck('MEMORY 3146 items',
   '3146' in mm)

# 7. script on disk
sp = (ROOT + r'\tests\glm5'
      r'\phase3145_omega_p143_'
      r'v1clip_pc1spec_headsign_'
      r'residcausal.py')
ck('script exists', os.path.exists(sp))
st = io.open(sp, encoding='utf-8').read()
ck('script no _rows bug',
   'batch = _rows[' not in st)
ck('script RES44 anchored',
   "RES44_SHA = 'def947d8'" in st)

fails = [c for c in CKS
         if c.startswith('FAIL')]
for c in CKS:
    print(c)
print('TOTAL %d/%d PASS'
      % (N[0] - len(fails), N[0]))
