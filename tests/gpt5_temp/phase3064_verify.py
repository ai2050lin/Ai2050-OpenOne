# -*- coding: utf-8 -*-
"""Post-closeout disk verification for Phase 3064.
Independently recomputes the ledger sha and checks every
closeout write actually landed on disk.
"""
import hashlib
import io
import json

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
OUT = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3064\omega_p61_ds7b_chain_'
       r'replication\verify_log.txt')

o = []

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
ph4 = [m for m in led['measurements']
       if m.get('phase') == 3064]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
o.append('ledger sha saved=%s recomputed=%s '
         'match=%s' % (saved, rec, saved == rec))
o.append('ledger n=%d (expect 203) phase3064 '
         'entries=%d (expect 1) l14=%d (expect '
         '171)' % (n, len(ph4),
                   len(l14['connects'])))
assert saved == rec, 'SHA MISMATCH'
assert n == 203 and len(ph4) == 1
assert len(l14['connects']) == 171
assert ph4[0]['verdict'] == 'chain_fragmented_ds7b'

# 2) MEMO: Phase 3064 section at tail
memo = io.open(MEMO, encoding='utf-8').read()
i64 = memo.rfind('## Phase 3064:')
o.append('memo len=%d phase3064 at=%d tail_ok=%s'
         % (len(memo), i64,
            i64 > memo.rfind('## Phase 3063:')))
assert i64 > 0
assert i64 > memo.rfind('## Phase 3063:')
tail = memo[i64:i64 + 400]
assert 'chain_fragmented_ds7b' in tail
assert '[2026-09-21 15:40:41]' in tail
o.append('memo title ok: %s'
         % tail.split('\n')[0])

# 3) audit addendum 26
aud = io.open(AUDIT, encoding='utf-8').read()
i26 = aud.rfind('## 二十六、3064 增补')
i25 = aud.rfind('## 二十五、3063 增补')
o.append('audit len=%d add26 at=%d after25=%s'
         % (len(aud), i26, i26 > i25 > 0))
assert i26 > i25 > 0
assert 'chain_fragmented_ds7b' in aud[i26:]

# 4) workspace daily log
wl = io.open(WLOG, encoding='utf-8').read()
o.append('wlog len=%d has3064=%s'
         % (len(wl), 'Phase 3064' in wl))
assert 'Phase 3064' in wl

# 5) MEMORY.md
mem = io.open(MEM, encoding='utf-8').read()
o.append('memory len=%d max3064=%s has_ds7b=%s'
         % (len(mem), 'max=3064' in mem,
            'DS7B' in mem))
assert 'max=3064' in mem
assert 'chain_fragmented' in mem
assert len(mem) < 3000

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('VERIFY_OK')
