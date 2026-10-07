# -*- coding: utf-8 -*-
"""Post-closeout disk verification for Phase 3065.
Independently recomputes the ledger sha and the four
seal shas (npz/result/script/exec) from real disk
bytes, then checks every closeout write landed.
"""
import hashlib
import io
import json
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + (r'\.workbuddy\memory'
               r'\2026-09-21.md')
MEM = ROOT + (r'\.workbuddy\memory\MEMORY.md')
RD = (ROOT + r'\tests\glm5\result'
      r'\rdc_query_construction_20260913'
      r'\phase3065\omega_p62_v_sign_'
      r'orchestration')
NPZ = RD + r'\omega_p62_v_sign_orchestration.npz'
RESJ = RD + r'\result.json'
SEALJ = RD + r'\seal.json'
EXECJ = RD + r'\execution.json'
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3065_omega_p62_v_sign_'
          r'orchestration.py')
OUT = RD + r'\verify_log.txt'

o = []


def sha8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20),
                          b''):
            h.update(chunk)
    return h.hexdigest()[:8]


# 0) seal shas recomputed from disk bytes
seal = json.load(io.open(SEALJ, encoding='utf-8'))
for key, path in (('npz_sha256_8', NPZ),
                  ('result_sha256_8', RESJ),
                  ('script_sha256_8', SCRIPT),
                  ('exec_sha256_8', EXECJ)):
    rec = sha8(path)
    ok = rec == seal[key]
    o.append('seal %s saved=%s recomputed=%s '
             'match=%s' % (key, seal[key], rec, ok))
    assert ok, 'SEAL MISMATCH ' + key

# 1) ledger: sha recompute + entry check
led = json.load(io.open(LEDGER, encoding='utf-8'))
saved = led.get('ledger_sha256_8')
chk = dict(led)
chk.pop('ledger_sha256_8')
blob = json.dumps(chk, sort_keys=True,
                  ensure_ascii=False)
rec = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
n = len(led['measurements'])
ph5 = [m for m in led['measurements']
       if m.get('phase') == 3065]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
o.append('ledger sha saved=%s recomputed=%s '
         'match=%s' % (saved, rec, saved == rec))
o.append('ledger n=%d (expect 204) phase3065 '
         'entries=%d (expect 1) l14=%d (expect '
         '172)' % (n, len(ph5),
                   len(l14['connects'])))
assert saved == rec, 'SHA MISMATCH'
assert n == 204 and len(ph5) == 1
assert len(l14['connects']) == 172
assert ph5[0]['verdict'] == 'sign_decoupled_all3'

# 2) MEMO: Phase 3065 section after 3064
memo = io.open(MEMO, encoding='utf-8').read()
i65 = memo.rfind('## Phase 3065:')
i64 = memo.rfind('## Phase 3064:')
o.append('memo len=%d phase3065 at=%d after3064='
         '%s' % (len(memo), i65, i65 > i64 > 0))
assert i65 > i64 > 0
tail = memo[i65:i65 + 300]
mt = re.match(
    r'## Phase 3065: .* \[2026-09-21 '
    r'\d{2}:\d{2}:\d{2}\]', tail)
assert mt is not None, 'TITLE/TIMESTAMP BAD'
o.append('memo title ok: %s'
         % tail.split('\n')[0])
assert 'sign_decoupled_all3' in tail
# key numbers present in the 3065 section
sec = memo[i65:]
iend = sec.find('## Phase 3066:')
if iend < 0:
    iend = len(sec)
sec = sec[:iend]
for key in (u'+0.130', u'\u22120.356', u'+0.913',
            'sign_decoupled_all3'):
    assert key in sec, 'MEMO MISSING ' + key
o.append('memo key numbers ok')

# 3) audit addendum 27
aud = io.open(AUDIT, encoding='utf-8').read()
i27 = aud.rfind('## 二十七、3065 增补')
i26 = aud.rfind('## 二十六、3064 增补')
o.append('audit len=%d add27 at=%d after26=%s'
         % (len(aud), i27, i27 > i26 > 0))
assert i27 > i26 > 0
assert 'sign_decoupled_all3' in aud[i27:]

# 4) workspace daily log
wl = io.open(WLOG, encoding='utf-8').read()
o.append('wlog len=%d has3065=%s'
         % (len(wl), 'Phase 3065' in wl))
assert 'Phase 3065' in wl

# 5) MEMORY.md
mem = io.open(MEM, encoding='utf-8').read()
o.append('memory len=%d max3065=%s has_sign=%s'
         % (len(mem), 'max=3065' in mem,
            'sign_decoupled' in mem))
assert 'max=3065' in mem
assert 'sign_decoupled' in mem
assert len(mem) < 3000

# 6) result.json verdict spot check
res = json.load(io.open(RESJ, encoding='utf-8'))
o.append('result verdict=%s'
         % res.get('verdict'))
assert res.get('verdict') == 'sign_decoupled_all3'

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('VERIFY_OK')
