# -*- coding: utf-8 -*-
# Phase 3270 closeout: five writes (idempotent)
import json, hashlib, os, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
WLOG = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md')
WMIR = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

FREEZE = "2f3b1327"
RES_SHA = "3660dc49"
ART_SHA = "b6039d3e"

def dump(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1)

# ---------------- write 1: ledger append #422 ----------------
led = json.load(open(LED, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == '3270' for m in ms):
    print('write1: #422 already present (idempotent skip)')
else:
    prev_post = led['ledger_sha256_8']
    assert prev_post == '9ce29f05' and len(ms) == 426, (prev_post, len(ms))
    rec = {
        'phase': '3270', 'code': 'G5-H33',
        'name': 'owner_report_registered',
        'created': datetime.datetime.now().isoformat(),
        'evidence_level': 'governance',
        'freeze_sha8': FREEZE, 'res_sha8': RES_SHA,
        'artifact': 'research/gpt5/docs/OWNER_STATUS_REPORT.md (%s)' % ART_SHA,
        'summary': ('Owner status consolidated report (report-only, zero GPU, zero '
                    'ledger measurement write): human-readable OWNER_STATUS_REPORT.md '
                    'generated from measured disk state — incumbent tri-account '
                    '(gap v1.13 814ce808 / registry v1.36 78d37211 / ledger #421 '
                    '9ce29f05 n=426), closure chain timeline 3239-3269 (N entries '
                    'measured, evidence distribution), u* dual-track verdicts, F/GAP/'
                    'APPX account matrix, 5 pending owner decisions, standby snapshot '
                    'history. All figures disk-measured, no memory. GATE 6/6.'),
        'prev_post': prev_post,
    }
    cp = dict(led)
    cp['measurements'] = ms + [rec]
    cp['ledger_sha256_8'] = prev_post
    new_post = hashlib.sha256(dump(cp).encode('utf-8')).hexdigest()[:8]
    cp['ledger_sha256_8'] = new_post
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(dump(cp))
    print('write1: ledger #422 appended, post=%s (n=%d)' % (new_post, len(ms) + 1))

# ---------------- write 2: MEMO ----------------
marker = '## Phase 3270 (G5-H33):'
if marker in open(MEMO, encoding='utf-8').read():
    print('write2: MEMO section already present (idempotent skip)')
else:
    led2 = json.load(open(LED, encoding='utf-8'))
    post = led2['ledger_sha256_8']
    sec = """
## Phase 3270 (G5-H33): 所有者状态综合报告（report-only）

- **verdict**: `owner_report_registered`（GATE 6/6，零 GPU、零账本测量写入）。
- **交付**：`research/gpt5/docs/OWNER_STATUS_REPORT.md`（sha8 `b6039d3e`，13.3k 字符）——面向所有者的人类可读收线状态综合报告，六节：现役三账 / 收线链 3239–3269 时间线 / u\\* 双轨道几何定谳 / F·GAP·APPX 账目矩阵 / 5 项待所有者裁决 / 待命快照历史。全部数值磁盘实测（含端点协程直调核验），不依赖记忆。
- 产物快照 `phase3270/g5h33_owner_report/owner_report_snapshot.md`（同 sha8 `b6039d3e`）；freeze `2f3b1327` / res `3660dc49` / ledger #422 post `EC_POST_SLOT`（n=427）。
- **后续**：维持 STANDBY。报告即为待命期权威汇总——所有者裁决任一待决项或点名执行相后恢复实质推进。
"""
    sec = sec.replace('EC_POST_SLOT', post)
    with open(MEMO, 'a', encoding='utf-8', newline='') as f:
        f.write(sec)
    print('write2: MEMO section appended (post=%s)' % post)

# ---------------- writes 3/4: wlogs ----------------
led3 = json.load(open(LED, encoding='utf-8'))
wline = ("- 3270 G5-H33 owner_report_registered（GATE 6/6，report-only）：交付 "
         "OWNER_STATUS_REPORT.md（b6039d3e，13.3k 字符，六节：现役三账/收线链时间线/"
         "u* 双轨道定谳/账目矩阵/5 项待决/待命历史），全部磁盘实测。freeze 2f3b1327 / "
         "res 3660dc49 / ledger #422 post EC_POST_SLOT（n=427）。维持待命。")

def wlog_append(p, tag):
    txt = open(p, encoding='utf-8').read()
    if '3270 G5-H33' in txt:
        print('write(%s): already present (idempotent skip)' % tag)
        return
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write('\n## 3270 ' + wline.replace('EC_POST_SLOT',
                                             led3['ledger_sha256_8']) + '\n')
    print('write(%s): appended' % tag)

wlog_append(WLOG, 'wlog')
wlog_append(WMIR, 'wmirror')

# ---------------- write 5: MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '**3270' in mem:
    print('write5: MEMORY already has 3270 (idempotent skip)')
else:
    o = '- **3269**：待命期复核——三账/端点/APPX 矩阵字节级不变、无并发 append；STANDBY_CONFIRMED。#421。'
    n = (o + '\n'
         '- **3270**：所有者状态综合报告 OWNER_STATUS_REPORT.md（六节磁盘实测）交付。#422。')
    if o in mem:
        mem = mem.replace(o, n)
    else:
        mem = mem.rstrip('\n') + '\n' + n + '\n'
    while len(mem) > 3000:
        lines = mem.split('\n')
        cand = [(len(l), i) for i, l in enumerate(lines)
                if l.startswith('- **32') and '**3270' not in l and len(l) > 40]
        if not cand:
            break
        _ln, i = min(cand)
        lines.pop(i)
        mem = '\n'.join(lines)
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    print('write5: MEMORY updated (len=%d)' % len(mem))

print('CLOSEOUT DONE')
