# -*- coding: utf-8 -*-
"""把 Phase 15 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。

纪律（Phase 12 教训 #12）：**自检锚点必须先对「追加源文件」预检**，
否则会出现「追加成功但锚点缺失」→ 不得不按字节回滚重来。本脚本先预检、再追加。

纪律（Phase 14 加固）：追加前读 `memo_baseline_preappend_phase15.json`（由 closeout_phase15.py 冻结），
用其中的 `bytes / sha8 / lines / phase_headings` 作为**前缀锚**逐字节复核「追加前文件未变」。

纪律（Phase 15 修正 #：前缀锚 bug）：前缀锚必须是**追加前的完整原始字节**（含 BOM 与结尾 CRLF）。
旧写法用 `BOM + txt.rstrip(CRLF)` 作前缀，若原文件以 CRLF 结尾则其长度比 `raw0` 少 2 字节，
`sha256(rb[:len(prefix)])` 必然 ≠ `sha256(raw0)` ⇒ 追加成功后仍抛断言（假失败）。
正确关系：`rb = BOM + txt.rstrip(CRLF) + CRLF + CRLF + body + CRLF`，故 `rb` **以 raw0 为前缀**。
本版另加**幂等保护**：若检测到 MEMO 已含 Phase 15 节，则跳过写入、只做纯复核。
"""
import os
import io
import time
import hashlib
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase15')
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(P15T, 'memo_append_phase15.md')
BASE = os.path.join(P15T, 'memo_baseline_preappend_phase15.json')
OUT = os.path.join(P15, 'verify_append_phase15.txt')

HDRKEY = '## Phase 15: 跨模型复算「统一剖面」'

o = []


def w(s=''):
    o.append(str(s))
    print(s)


# ---------- -1. 读预追加基线（前缀锚） ----------
BB = json.load(io.open(BASE, encoding='utf-8'))
BASE_BYTES = int(BB['bytes'])
BASE_SHA8 = str(BB['sha8'])
BASE_LINES = int(BB.get('lines', 0))
BASE_PHASES = int(BB.get('phase_headings', 14))
w('=== [-1] 预追加基线（memo_baseline_preappend_phase15.json） ===')
w('  bytes=%d lines=%d phase_headings=%d sha8=%s tag=%s'
  % (BASE_BYTES, BASE_LINES, BASE_PHASES, BASE_SHA8, BB.get('tag')))

ANCHORS = [
    HDRKEY, 'N2h1-α-8',
    'NF4_FAITHFUL', 'max|dxhalf|', 'argmax_w_x', 'XH_RANGE',
    'CONC_JUDGE_INVALID_X_ALL', 'CONC_JUDGE_ALIVE_X',
    'ARGS_GAP_LAYERSTACK', 'ARGS_GAP_4B_SPECIFIC', 'ARGS_GAP_MIXED',
    'null95', '裕度', '置换零假设', 'L_star_own', 'B_cat',
    'qwen3-14b', 'glm4-9b', 'nf4', 'bf16', '7.317', '0.036', '200×',
    'F3', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
    '第一性原理', '(z)', '(aa)', 'Phase 16', 'HARKing',
    'spearman', '0.70', '逐位',
]
OPTIONAL = ['HARKing']          # 若正文未出现，仅提示不阻断

# ---------- 0. 预检：锚点必须已存在于追加源文件 ----------
src = io.open(APP, encoding='utf-8').read()
REQ = [k for k in ANCHORS if k not in OPTIONAL]
pre_miss = [k for k in REQ if k not in src]
w('')
w('=== [0] 追加源预检 ===')
w('源文件 %s bytes=%d lines=%d' % (os.path.basename(APP), len(src.encode('utf-8')),
                                    len(src.splitlines())))
for k in ANCHORS:
    c = src.count(k)
    flag = 'OK' if c >= 1 else ('(optional)' if k in OPTIONAL else '!! MISSING')
    w('  %-44s count=%d %s' % (k, c, flag))
w('源文件缺失（必选）锚点数 = %d' % len(pre_miss))
assert not pre_miss, '追加源文件缺少锚点（先修源文件再追加）: %s' % pre_miss
w('  ==> 预检通过（%d 个必选锚点全部存在于源文件）' % len(REQ))

# ---------- 0b. 幂等探测 + 前缀锚 ----------
raw0 = open(MEMO, 'rb').read()
txt = raw0.decode('utf-8-sig')
ALREADY = (HDRKEY in txt)
w('')
w('=== [0b] 追加前前缀锚复核 ===')
w('  实际 bytes=%d sha8=%s' % (len(raw0), hashlib.sha256(raw0).hexdigest()[:8]))
assert raw0[:3] == b'\xef\xbb\xbf', 'BOM 丢失'
_now_sha8 = hashlib.sha256(raw0).hexdigest()[:8]
if ALREADY:
    w('  ==> MEMO 已含 Phase 15 节 ⇒ **纯复核模式**（跳过写入；前缀锚改为与冻结基线 sha8 比对）')
elif BASE_SHA8 == 'PENDING_APPEND':
    w('  基线为 PENDING_APPEND（证明 MEMO 尚未追加）⇒ 允许继续')
elif _now_sha8 != BASE_SHA8 or len(raw0) != BASE_BYTES:
    raise AssertionError('MEMO 已被改动（基线 %d B/%s vs 实际 %d B/%s）——中止，勿覆盖'
                         % (BASE_BYTES, BASE_SHA8, len(raw0), _now_sha8))
else:
    w('  ==> 与基线一致（%d B / %s）' % (BASE_BYTES, BASE_SHA8))

# ---------- 1. 追加（幂等） ----------
n0_b, n0_l = len(raw0), len(raw0.split(b'\r\n'))
if not ALREADY:
    body = io.open(APP, encoding='utf-8').read().strip('\r\n')
    new = txt.rstrip('\r\n') + '\r\n\r\n' + body + '\r\n'
    out = b'\xef\xbb\xbf' + new.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
    open(MEMO, 'wb').write(out)
    w('')
    w('=== [1] 追加执行 ===')
    w('  写入完成（非幂等路径）')
else:
    n0_b, n0_l = BASE_BYTES, BASE_LINES
    w('')
    w('=== [1] 追加执行 ===')
    w('  跳过写入（幂等路径：Phase 15 节已存在）')

# ---------- 2. 落盘复核（真实磁盘） ----------
rb = open(MEMO, 'rb').read()
t2 = rb.decode('utf-8-sig')
lines = t2.splitlines()
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 15')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('')
w('=== [2] 追加结果 ===')
w('append: bytes %d -> %d (+%d) ; lines %d -> %d (+%d)'
  % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines), len(lines) - n0_l))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'),
                                  rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('sha8   = %s' % hashlib.sha256(rb).hexdigest()[:8])
w('Phase 15 标题行 = %s' % hdr)
w('Phase 标题总数 = %d（追加前 %d）' % (len(allh), BASE_PHASES))

# --- 前缀锚：必须用「追加前的完整原始字节」---
if ALREADY:
    pre_raw = raw0[:BASE_BYTES]
    pre_sha8 = BASE_SHA8
else:
    pre_raw = raw0
    pre_sha8 = _now_sha8
prefix_ok = rb.startswith(pre_raw)
anchor_ok = (prefix_ok and len(rb) >= len(pre_raw)
             and hashlib.sha256(rb[:len(pre_raw)]).hexdigest()[:8] == pre_sha8)
w('前缀锚（长度 %d / 期望 sha8 %s）: startswith=%s sha_match=%s'
  % (len(pre_raw), pre_sha8, prefix_ok, anchor_ok))

w('')
w('=== [3] 锚点落盘复核 ===')
miss = []
for k in ANCHORS:
    c = t2.count(k)
    w('  anchor %-46s count=%d %s' % (k, c, 'OK' if c >= 1 else ('(optional)' if k in OPTIONAL else '!! MISSING')))
    if c < 1 and k not in OPTIONAL:
        miss.append(k)
w('缺失（必选）锚点数 = %d' % len(miss))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')

assert not miss, '锚点缺失: %s' % miss
assert len(hdr) == 1, 'Phase 15 标题不唯一: %s' % hdr
assert len(allh) == BASE_PHASES + 1, 'Phase 标题数应为 %d，实为 %d' % (BASE_PHASES + 1, len(allh))
assert rb[:3] == b'\xef\xbb\xbf'
assert rb.count(b'\n') - rb.count(b'\r\n') == 0, 'bare_lf 不为 0'
assert prefix_ok, '前缀被改动'
assert anchor_ok, '前缀锚 sha8 不匹配'
print('ALL CHECKS PASSED ->', OUT)
