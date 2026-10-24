# -*- coding: utf-8 -*-
# Phase 3269 closeout: five writes (idempotent)
import json, hashlib, os, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
WLOG = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md')
WMIR = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

FREEZE = "12d5491d"
RES_SHA = "820201b2"
ART_SHA = "0013779e"

def dump(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1)

# ---------------- write 1: ledger append #421 ----------------
led = json.load(open(LED, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == '3269' for m in ms):
    print('write1: #421 already present (idempotent skip)')
else:
    prev_post = led['ledger_sha256_8']
    assert prev_post == 'b7407287' and len(ms) == 425, (prev_post, len(ms))
    rec = {
        'phase': '3269', 'code': 'G5-H32',
        'name': 'standby_refresh_registered',
        'created': datetime.datetime.now().isoformat(),
        'evidence_level': 'governance',
        'freeze_sha8': FREEZE, 'res_sha8': RES_SHA,
        'artifact': 'tests/glm5/result/rdc_query_construction_20260913/phase3269/'
                    'g5h32_standby_refresh/standby_refresh_v1.json (%s)' % ART_SHA,
        'summary': ('Standby refresh recheck (report-only, zero GPU, zero ledger '
                    'measurement write): nothing moved during standby window — gap '
                    'v1.13 814ce808 / registry v1.36 78d37211 byte-identical to 3268 '
                    'snapshot; ledger n=425 post b7407287 tail=3268 (only own-line '
                    'closeout #420, concurrent_append_detected=False); endpoints live '
                    '(gaps 1.13 appendix 7 / registry 1.36 upgrade_log 15); manifest '
                    'v1/v2 + APPX matrix unchanged; 5 owner-approval items untouched. '
                    'Chain head recompute PASS. GATE 6/6. STANDBY CONFIRMED.'),
        'prev_post': prev_post,
    }
    cp = dict(led)
    cp['measurements'] = ms + [rec]
    cp['ledger_sha256_8'] = prev_post
    new_post = hashlib.sha256(dump(cp).encode('utf-8')).hexdigest()[:8]
    cp['ledger_sha256_8'] = new_post
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(dump(cp))
    print('write1: ledger #421 appended, post=%s (n=%d)' % (new_post, len(ms) + 1))

# ---------------- write 2: MEMO ----------------
marker = '## Phase 3269 (G5-H32):'
if marker in open(MEMO, encoding='utf-8').read():
    print('write2: MEMO section already present (idempotent skip)')
else:
    led2 = json.load(open(LED, encoding='utf-8'))
    post = led2['ledger_sha256_8']
    sec = """
## Phase 3269 (G5-H32): 待命期复核（report-only 轻量刷新）

- **verdict**: `standby_refresh_registered`（GATE 6/6，零 GPU、零账本测量写入）。
- **结论：STANDBY_CONFIRMED**——待命窗口内无任何移动：gap v1.13（`814ce808`）/ registry v1.36（`78d37211`）与 3268 快照**字节一致**；ledger n=425 post `b7407287` 尾条 3268（快照后仅本线 closeout #420，**concurrent_append_detected=False**）；端点实机可见（gaps 1.13 appendix 7 / registry 1.36 upgrade_log 15）；manifest v1/v2 + APPX-1..7 矩阵不变；5 项所有者批准事项不变。
- 产物 `standby_refresh_v1.json`（sha8 `0013779e`）；freeze `12d5491d` / res `820201b2` / ledger #421 post `EC_POST_SLOT`（n=426）。
- **后续**：维持 STANDBY。再次「继续」将继续执行同类轻量刷新臂；实质性推进须所有者指令（批准待决项 / 点名 APPX-2/3 执行相 / 新方向）。
"""
    sec = sec.replace('EC_POST_SLOT', post)
    with open(MEMO, 'a', encoding='utf-8', newline='') as f:
        f.write(sec)
    print('write2: MEMO section appended (post=%s)' % post)

# ---------------- writes 3/4: wlogs ----------------
led3 = json.load(open(LED, encoding='utf-8'))
wline = ("- 3269 G5-H32 standby_refresh_registered（GATE 6/6，report-only 轻量刷新）："
         "STANDBY_CONFIRMED——gap v1.13 814ce808 / registry v1.36 78d37211 与 3268 快照字节一致；"
         "ledger n=425 post b7407287 尾条 3268（仅本线 closeout #420，无并发 append）；"
         "端点实机 + manifest v1/v2 + APPX 矩阵不变；5 项批准事项不变。产物 0013779e / "
         "freeze 12d5491d / res 820201b2 / ledger #421 post EC_POST_SLOT（n=426）。维持待命。")

def wlog_append(p, tag):
    txt = open(p, encoding='utf-8').read()
    if '3269 G5-H32' in txt:
        print('write(%s): already present (idempotent skip)' % tag)
        return
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write('\n## 3269 ' + wline.replace('EC_POST_SLOT',
                                             led3['ledger_sha256_8']) + '\n')
    print('write(%s): appended' % tag)

wlog_append(WLOG, 'wlog')
wlog_append(WMIR, 'wmirror')

# ---------------- write 5: MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '**3269' in mem:
    print('write5: MEMORY already has 3269 (idempotent skip)')
else:
    o = '- **3268**：收线待命快照（report-only）——三账+端点实机核验、APPX 矩阵 7/7、STANDBY 声明（5 项待批准）。#420。项目待命。'
    n = ('- **3268**：收线待命快照（report-only）——三账+端点实机核验、APPX 矩阵 7/7、STANDBY 声明（5 项待批准）。#420。\n'
         '- **3269**：待命期复核——三账/端点/APPX 矩阵字节级不变、无并发 append；STANDBY_CONFIRMED。#421。')
    if o in mem:
        mem = mem.replace(o, n)
    else:
        mem = mem.rstrip('\n') + '\n' + n + '\n'
    while len(mem) > 3000:
        lines = mem.split('\n')
        cand = [(len(l), i) for i, l in enumerate(lines)
                if l.startswith('- **32') and '**3269' not in l and len(l) > 40]
        if not cand:
            break
        _ln, i = min(cand)
        lines.pop(i)
        mem = '\n'.join(lines)
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    print('write5: MEMORY updated (len=%d)' % len(mem))

print('CLOSEOUT DONE')
