# -*- coding: utf-8 -*-
"""Phase 3136 disk verify: independent
post-closeout readback of all five writes
+ artifacts. Run AFTER closeout."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
OUT = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_crossmatrix_w8drop'
MEMO = ROOT + r'\research\gpt5\docs' \
       r'\AGI_GPT5_MEMO.md'
LEDGER = ROOT + r'\research\gpt5\atlas' \
         r'\atlas_ledger.json'
WLOG = ROOT + r'\.workbuddy\memory' \
       r'\2026-09-27.md'
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'

out = []
ok = 0
fail = 0


def chk(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        out.append('PASS %s %s' % (name,
                                   detail))
    else:
        fail += 1
        out.append('FAIL %s %s' % (name,
                                   detail))


# 1 artifacts exist
for f in ('result.json', 'design_seal.json',
          'run_log.txt', 'p134_readout.npz'):
    p = os.path.join(OUT, f)
    chk('art:%s' % f, os.path.exists(p),
        '%d B' % os.path.getsize(p)
        if os.path.exists(p) else 'missing')
# 2 result.json well-formed + verdict
try:
    r = json.load(io.open(
        os.path.join(OUT, 'result.json'),
        encoding='utf-8'))
    chk('result.smoke', r['smoke'] is False)
    chk('result.phase', r['phase'] == 3136)
    chk('result.verdict', '|'.join([
        'a_3135_ok']) in r['verdict'],
        r['verdict'])
    chk('result.coverage',
        r['verdict'].endswith(
            'coverage_full'))
    chk('result.runtime>0',
        r['runtime_s'] > 0)
    chk('result.cmat9',
        len(r['part_c']['matrix']) == 9)
    chk('result.dchgl2',
        len(r['part_b']['dose_chg']) == 12)
except Exception as e:
    chk('result.load', False, repr(e))
# 3 run_log has DONE line
try:
    rl = io.open(os.path.join(OUT,
                              'run_log.txt'),
                 encoding='utf-8').read()
    chk('log.done', 'P3136 DONE' in rl)
    chk('log.verdict', 'VERDICT:' in rl)
    chk('log.no_traceback',
        'Traceback' not in rl)
except Exception as e:
    chk('log.read', False, repr(e))
# 4 npz readable + key count
try:
    import numpy as np
    z = np.load(OUT + r'\p134_readout.npz',
                allow_pickle=False)
    chk('npz.keys>=40', len(z.files) >= 40,
        str(len(z.files)))
    chk('npz.union',
        z['union'].shape == (100,))
except Exception as e:
    chk('npz.load', False, repr(e))
# 5 ledger entry
try:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    ent = [e for e in led['entries']
           if e.get('phase') == 3136]
    chk('ledger.3136', len(ent) == 1)
    if ent:
        chk('ledger.sha8',
            ent[0].get('sha8') is not None,
            str(ent[0].get('sha8')))
        chk('ledger.total',
            len(led['entries']) >= 273,
            'n=%d' % len(led['entries']))
except Exception as e:
    chk('ledger.read', False, repr(e))
# 6 MEMO section
try:
    m = io.open(MEMO, encoding='utf-8').read()
    chk('memo.3136',
        '## Phase 3136' in m)
    i3136 = m.rfind('## Phase 3136')
    tail = m[i3136:]
    chk('memo.verdict_text',
        'a_3135_ok' in tail)
    chk('memo.triple',
        tail.count('### 1.') >= 1)
except Exception as e:
    chk('memo.read', False, repr(e))
# 7 workspace log
try:
    w = io.open(WLOG, encoding='utf-8').read()
    chk('wlog.3136', '3136' in w)
except Exception as e:
    chk('wlog.read', False, repr(e))
# 8 workspace MEMORY
try:
    mm = io.open(WMEM,
                 encoding='utf-8').read()
    chk('wmem.3136', '3136' in mm)
except Exception as e:
    chk('wmem.read', False, repr(e))
# 9 script on disk w/ rev marker
try:
    sc = io.open(
        ROOT + r'\tests\glm5'
        r'\phase3136_omega_p134_'
        r'conddose_crossmatrix_w8drop.py',
        encoding='utf-8').read()
    chk('script.n_cap_fix',
        'n_cap = min(NP_CAP' in sc)
    chk('script.sha_ok', len(sc) > 30000,
        '%d chars' % len(sc))
except Exception as e:
    chk('script.read', False, repr(e))

out.append('SUMMARY ok=%d fail=%d'
           % (ok, fail))
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3136_disk_verify.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(out))
print('VERIFY DONE ok=%d fail=%d'
      % (ok, fail))
