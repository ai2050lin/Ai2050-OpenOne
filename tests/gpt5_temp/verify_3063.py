# verify all 3063 closeout writes on real disk
import json
import os
import hashlib
import io

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\verify_3063.txt')
ROOT = r'D:\AI2050\Ai2050-OpenOne'
lines = []
ok = True

# 1. ledger
lp = ROOT + r'\research\gpt5\atlas\atlas_ledger.json'
led = json.load(io.open(lp, encoding='utf-8'))
ms = led['measurements']
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
has3063 = any(m.get('phase') == 3063 for m in ms)
lines.append('ledger n=%d has3063=%s l14conn=%d'
             % (len(ms), has3063,
                len(l14['connects'])))
sha_now = hashlib.sha256(json.dumps(
    {k: v for k, v in led.items()
     if k != 'ledger_sha256_8'},
    sort_keys=True,
    ensure_ascii=False).encode('utf-8')
).hexdigest()[:8]
lines.append('ledger sha recomputed=%s stored=%s '
             'match=%s' % (sha_now,
                           led['ledger_sha256_8'],
                           sha_now
                           == led['ledger_sha256_8']))
ok = ok and has3063 and len(ms) == 202 \
    and len(l14['connects']) == 170 \
    and sha_now == led['ledger_sha256_8']

# 2. MEMO
mp = ROOT + (r'\research\gpt5\docs'
             r'\AGI_GPT5_MEMO.md')
memo = io.open(mp, encoding='utf-8').read()
has = '## Phase 3063:' in memo
lines.append('memo has Phase3063=%s len=%d '
             'verdict_in=%s'
             % (has, len(memo),
                'adversarial_same_source_qwen'
                in memo))
ok = ok and has
idx = memo.find('## Phase 3063:')
lines.append('memo3063 head=%s'
             % memo[idx:idx + 120].replace('\n', ' '))

# 3. audit
ap = ROOT + (r'\research\gpt5\docs'
             r'\hdmcc_knowledge_map_review_'
             r'20260921.md')
aud = io.open(ap, encoding='utf-8').read()
has25 = '## 二十五、3063 增补' in aud
lines.append('audit has25=%s len=%d' % (has25,
                                        len(aud)))
ok = ok and has25

# 4. workspace log
wlp = os.path.join(ROOT, '.workbuddy', 'memory',
                   '2026-09-21.md')
wl_ok = os.path.exists(wlp)
wl_has = wl_ok and 'Phase 3063' in io.open(
    wlp, encoding='utf-8').read()
lines.append('wlog exists=%s has3063=%s'
             % (wl_ok, wl_has))
ok = ok and wl_has

# 5. MEMORY.md
mmp = os.path.join(ROOT, '.workbuddy', 'memory',
                   'MEMORY.md')
mm_ok = os.path.exists(mmp)
mm_len = len(io.open(mmp, encoding='utf-8').read()) \
    if mm_ok else 0
mm_has = mm_ok and 'max=3063' in io.open(
    mmp, encoding='utf-8').read()
lines.append('MEMORY.md exists=%s max3063=%s '
             'len=%d' % (mm_ok, mm_has, mm_len))
ok = ok and mm_has and mm_len < 3000

lines.append('ALL_OK=%s' % ok)
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
