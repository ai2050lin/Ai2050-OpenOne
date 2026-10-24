# -*- coding: utf-8 -*-
"""Phase 3240 meta-fix: res_sha8 determinism (exclude created/runtime_s).
Syncs the sealed res reference 20002d42 -> ffb97506 across ledger entry #392,
MEMO, MEMORY, dual wlogs; recomputes the ledger post hash. No measurement
number is touched (pure sealing-metadata repair, mirrors the 3238 CRLF
normalization precedent).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
OLD, NEW = '20002d42', 'ffb97506'
PREV_POST = '016f79aa'
REPORT = []

# 1 ledger entry text sync + post recompute
raw = open(LED, 'rb').read()
led = json.loads(raw.decode('utf-8'))
e = [x for x in led['measurements'] if x.get('phase') == 3240]
assert len(e) == 1
det = e[0]['detail']
if OLD in det:
    e[0]['detail'] = det.replace('res %s' % OLD, 'res %s' % NEW)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('1 ledger: entry #392 res ref synced %s->%s, post now %s'
                  % (OLD, NEW, new_post))
else:
    new_post = led['ledger_sha256_8']
    assert NEW in json.dumps(e[0], ensure_ascii=False)
    REPORT.append('1 ledger: already synced (post=%s)' % new_post)
# chain recheck
chk = json.load(open(LED, encoding='utf-8'))
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1).encode()).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8']
REPORT.append('1 ledger chain recheck OK (n=%d, post=%s)' % (len(chk['measurements']), new_post))

# 2 MEMO
mb = open(MEMO, 'rb').read()
if ('res **`%s`**' % OLD).encode() in mb:
    mb2 = mb.replace(('res **`%s`**' % OLD).encode(), ('res **`%s`**' % NEW).encode())
    open(MEMO, 'wb').write(mb2)
    REPORT.append('2 memo: res ref synced')
else:
    REPORT.append('2 memo: already synced')
assert (b'20002d42' not in open(MEMO, 'rb').read())

# 3 wlogs
for p in (WLOG_WS, WLOG_U):
    t = open(p, encoding='utf-8').read()
    if 'res %s' % OLD in t:
        open(p, 'w', encoding='utf-8', newline='').write(
            t.replace('res %s' % OLD, 'res %s' % NEW))
        REPORT.append('3 wlog %s: synced' % p)
    else:
        REPORT.append('3 wlog %s: already synced' % p)

# 4 MEMORY
mem = open(MEM, encoding='utf-8').read()
if ('res `%s`' % OLD) in mem:
    mem = mem.replace('res `%s`' % OLD, 'res `%s`' % NEW)
    assert len(mem) < 3000
    open(MEM, 'w', encoding='utf-8', newline='').write(mem)
    REPORT.append('4 memory: res ref synced (len=%d)' % len(mem))
else:
    REPORT.append('4 memory: already synced')

print(chr(10).join(REPORT))
print('POST=%s' % new_post)
