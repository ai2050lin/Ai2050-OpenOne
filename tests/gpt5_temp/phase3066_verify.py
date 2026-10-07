# -*- coding: utf-8 -*-
"""Post-closeout disk verification for Phase 3066.
Independently recomputes the ledger sha and the four
seal shas (npz/result/script/exec) from real disk
bytes, then checks every closeout write landed."""
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
      r'\phase3066\omega_p63_last_layer_'
      r'flip_anatomy')
NPZ = RD + (r'\omega_p63_last_layer_flip_'
            r'anatomy.npz')
RESJ = RD + r'\result.json'
SEALJ = RD + r'\seal.json'
EXECJ = RD + r'\execution.json'
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3066_omega_p63_last_layer_'
          r'flip_anatomy.py')
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
ph6 = [m for m in led['measurements']
       if m.get('phase') == 3066]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
o.append('ledger sha saved=%s recomputed=%s '
         'match=%s' % (saved, rec, saved == rec))
o.append('ledger n=%d (expect 205) phase3066 '
         'entries=%d (expect 1) l14=%d (expect '
         '173)' % (n, len(ph6),
                   len(l14['connects'])))
assert saved == rec, 'SHA MISMATCH'
assert n == 205 and len(ph6) == 1
assert len(l14['connects']) == 173
assert ph6[0]['verdict'] == ('sign_flip_'
                             'downstream_'
                             'distributed_'
                             'heads_dist')

# 2) MEMO: Phase 3066 section after 3065
memo = io.open(MEMO, encoding='utf-8').read()
i66 = memo.rfind('## Phase 3066:')
i65 = memo.rfind('## Phase 3065:')
o.append('memo len=%d phase3066 at=%d after3065='
         '%s' % (len(memo), i66, i66 > i65 > 0))
assert i66 > i65 > 0
tail = memo[i66:i66 + 300]
mt = re.match(
    r'## Phase 3066: .* \[2026-09-21 '
    r'\d{2}:\d{2}:\d{2}\]', tail)
assert mt is not None, 'TITLE/TIMESTAMP BAD'
o.append('memo title ok: %s'
         % tail.split('\n')[0])
assert 'sign_flip_downstream_distributed' in tail
sec = memo[i66:]
iend = sec.find('## Phase 3067:')
if iend < 0:
    iend = len(sec)
sec = sec[:iend]
for key in (u'\u22120.302', u'+0.199',
            u'\u22120.356', u'8.6 pct',
            'sign_flip_downstream_distributed'):
    assert key in sec, 'MEMO MISSING ' + key
o.append('memo key numbers ok')

# 3) audit addendum 28
aud = io.open(AUDIT, encoding='utf-8').read()
i28 = aud.rfind('## 二十八、3066 增补')
i27 = aud.rfind('## 二十七、3065 增补')
o.append('audit len=%d add28 at=%d after27=%s'
         % (len(aud), i28, i28 > i27 > 0))
assert i28 > i27 > 0
assert 'sign_flip_downstream' in aud[i28:]

# 4) workspace daily log
wl = io.open(WLOG, encoding='utf-8').read()
o.append('wlog len=%d has3066=%s'
         % (len(wl), 'Phase 3066' in wl))
assert 'Phase 3066' in wl

# 5) MEMORY.md
mem = io.open(MEM, encoding='utf-8').read()
o.append('memory len=%d max3066=%s has_flip=%s'
         % (len(mem), 'max=3066' in mem,
            'sign_flip_downstream' in mem))
assert 'max=3066' in mem
assert 'sign_flip_downstream' in mem
assert len(mem) < 3000

# 6) result.json verdict spot check
res = json.load(io.open(RESJ, encoding='utf-8'))
o.append('result verdict=%s'
         % res.get('verdict'))
assert res.get('verdict') == ('sign_flip_'
                              'downstream_'
                              'distributed_'
                              'heads_dist')

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('VERIFY_OK')
