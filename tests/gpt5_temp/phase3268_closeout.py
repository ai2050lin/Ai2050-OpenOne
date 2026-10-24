# -*- coding: utf-8 -*-
# Phase 3268 closeout: five writes (idempotent)
import json, hashlib, os, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
WLOG = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md')
WMIR = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

FREEZE = "bf5a0dfb"
RES_SHA = "1f6d94b0"
ART_SHA = "48308482"

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def dump(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1)

# ---------------- write 1: ledger append #420 ----------------
led = json.load(open(LED, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == '3268' for m in ms):
    print('write1: #420 already present (idempotent skip)')
else:
    prev_post = led['ledger_sha256_8']
    assert prev_post == '6f2a9170' and len(ms) == 424, (prev_post, len(ms))
    rec = {
        'phase': '3268', 'code': 'G5-H31',
        'name': 'closure_standby_registered',
        'created': datetime.datetime.now().isoformat(),
        'evidence_level': 'governance',
        'freeze_sha8': FREEZE, 'res_sha8': RES_SHA,
        'artifact': 'tests/glm5/result/rdc_query_construction_20260913/phase3268/'
                    'g5h31_closure_standby/standby_snapshot_v1.json (%s)' % ART_SHA,
        'summary': ('Closure standby snapshot (report-only, zero GPU, zero ledger '
                    'measurement write): arm a incumbent tri-account + live endpoints '
                    '(gaps v1.13 appendix 7 / registry v1.36 upgrade_log 15, chain '
                    'heads [1.13,1.12,1.11]/[1.36,1.35,...] parseable); arm b closure '
                    'package integrity (closure_summary + manifest v2 57531ec3 + '
                    'manifest v1 bc212d52 sealed bytes; APPX-1..7 status matrix '
                    'measured==declared 7/7); arm c STANDBY declaration (5 items '
                    'require owner approval: fruit-K4 deletion, lcorr op4, APPX-1 '
                    'quantization, APPX-2/3 execution, APPX-4/5 governance; no '
                    'auto-resume). Chain head recompute g5 PASS. GATE 6/6.'),
        'prev_post': prev_post,
    }
    cp = dict(led)
    cp['measurements'] = ms + [rec]
    cp['ledger_sha256_8'] = prev_post
    new_post = hashlib.sha256(dump(cp).encode('utf-8')).hexdigest()[:8]
    cp['ledger_sha256_8'] = new_post
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(dump(cp))
    print('write1: ledger #420 appended, post=%s (n=%d)' % (new_post, len(ms) + 1))

# ---------------- write 2: MEMO ----------------
marker = '## Phase 3268 (G5-H31):'
if marker in open(MEMO, encoding='utf-8').read():
    print('write2: MEMO section already present (idempotent skip)')
else:
    led2 = json.load(open(LED, encoding='utf-8'))
    post = led2['ledger_sha256_8']
    sec = """
## Phase 3268 (G5-H31): 收线待命快照（report-only）

- **verdict**: `closure_standby_registered`（GATE 6/6，零 GPU、零账本测量写入）。
- **Arm a（现役三账 + 端点实机）**：gap v1.13（`814ce808`，appendix 7）/ registry v1.36（`78d37211`，upgrade_log 15，anchors 28）/ ledger #419（n=424，post `6f2a9170`，尾条 3267）；`atlas_gaps`/`atlas_registry` 协程直调实机可见（链头 [1.13,1.12,1.11] / [1.36,1.35,…] 均可解析）。
- **Arm b（收线包完整性）**：closure_summary_3239_3255.md 在盘；N 线 manifest v2（`57531ec3`）+ v1 封存字节（`bc212d52`）不动；**APPX-1..7 status 矩阵实测==声明 7/7**（1/2/3 open、4/5 open_crossline、6/7 recorded）。
- **Arm c（STANDBY 声明）**：五项须所有者显式批准——水果类/K4 删除、lcorr op4 施加、APPX-1 量化口径、APPX-2/3 执行相（协议已备 `ce73984d`）、APPX-4/5 治理动作；无 auto-resume（「继续」只执行默认 report-only 快照臂）。
- 产物 `standby_snapshot_v1.json`（sha8 `48308482`）；freeze `bf5a0dfb` / res `1f6d94b0` / ledger #420 post `EC_POST_SLOT`（n=425）。链头重算（g5）通过。
- **后续**：项目转入 STANDBY——无预注册执行相；下一 phase 由用户指令触发（批准任一待决项，或点名 APPX-2/3 执行相）。
"""
    sec = sec.replace('EC_POST_SLOT', post)
    with open(MEMO, 'a', encoding='utf-8', newline='') as f:
        f.write(sec)
    print('write2: MEMO section appended (post=%s)' % post)

# ---------------- writes 3/4: wlogs ----------------
led3 = json.load(open(LED, encoding='utf-8'))
wline = ("- 3268 G5-H31 closure_standby_registered（GATE 6/6，report-only）：现役三账快照 "
         "gap v1.13 814ce808 / registry v1.36 78d37211 / ledger #419 n=424 post 6f2a9170；"
         "端点实机 gaps v1.13 appendix 7 + registry v1.36 upgrade_log 15（链头均可解析）；"
         "收线包完整性 manifest v2 57531ec3 + v1 封存 bc212d52 + APPX 矩阵实测==声明 7/7；"
         "STANDBY 声明：5 项须所有者批准（水果类/K4、lcorr op4、APPX-1 量化、APPX-2/3 执行、"
         "APPX-4/5 治理），无 auto-resume。产物 48308482 / freeze bf5a0dfb / res 1f6d94b0 / "
         "ledger #420 post EC_POST_SLOT（n=425）。项目转入待命。")

def wlog_append(p, tag):
    txt = open(p, encoding='utf-8').read()
    if '3268 G5-H31' in txt:
        print('write(%s): already present (idempotent skip)' % tag)
        return
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write('\n## 3268 ' + wline.replace('EC_POST_SLOT',
                                             led3['ledger_sha256_8']) + '\n')
    print('write(%s): appended' % tag)

wlog_append(WLOG, 'wlog')
wlog_append(WMIR, 'wmirror')

# ---------------- write 5: MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '**3268' in mem:
    print('write5: MEMORY already has 3268 (idempotent skip)')
else:
    o = '- **挂账**：水果类/K4 删除待确认。'
    n = ('- **挂账**：水果类/K4 删除待确认。\n'
         '- **3268**：收线待命快照（report-only）——三账+端点实机核验、APPX 矩阵 7/7、'
         'STANDBY 声明（5 项待批准）。#420。项目待命。')
    if o in mem:
        mem = mem.replace(o, n)
    else:
        mem = mem.rstrip('\n') + '\n' + n + '\n'
    while len(mem) > 3000:
        lines = mem.split('\n')
        cand = [(len(l), i) for i, l in enumerate(lines)
                if l.startswith('- **32') and '**3268' not in l and len(l) > 40]
        if not cand:
            break
        _ln, i = min(cand)
        lines.pop(i)
        mem = '\n'.join(lines)
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    print('write5: MEMORY updated (len=%d)' % len(mem))

print('CLOSEOUT DONE')
