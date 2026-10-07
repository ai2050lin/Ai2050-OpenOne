# -*- coding: utf-8 -*-
"""把 Phase 20 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。

沿用 Phase 15–19 已修正的纪律：
  (1) 前缀锚必须是**追加前的完整原始字节**（含 BOM 与结尾 CRLF）；
  (2) 幂等探测：若 MEMO 已含 Phase 20 节则跳过写入、只做纯复核。
"""
import os
import io
import time
import hashlib
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(P20T, 'memo_append_phase20.md')
BASE = os.path.join(P20T, 'memo_baseline_preappend_phase20.json')
OUT = os.path.join(P20T, 'verify_append_phase20.txt')

HDRKEY = '## Phase 20:'
o = []


def w(s=''):
    o.append(str(s))
    print(s)


BB = json.load(io.open(BASE, encoding='utf-8'))
BASE_BYTES = int(BB['bytes'])
BASE_SHA8 = str(BB['sha8'])
BASE_LINES = int(BB.get('lines', 0))
BASE_PHASES = int(BB.get('phase_headings', 19))
w('=== [-1] 预追加基线（memo_baseline_preappend_phase20.json） ===')
w('  bytes=%d lines=%d phase_headings=%d sha8=%s tag=%s'
  % (BASE_BYTES, BASE_LINES, BASE_PHASES, BASE_SHA8, BB.get('tag')))

ANCHORS = [
    HDRKEY, 'N2h1-α-13',
    'com_B', 'com_layer', 'b_{c,ℓ}', 'com_V', 'REACH', 'share_mlp_beh', 'xhalf',
    'A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16',
    'qwen3-4b', 'glm4-9b', 'Qwen3-14B', 'nf4', 'bf16',
    'segfault', 'offload', 'E-sper', 'E-scope', 'E-probefull', 'E-baseline', 'E-xhdom', 'E-rho',
    'Phase 21',
]

src = io.open(APP, encoding='utf-8').read()
pre_miss = [k for k in ANCHORS if k not in src]
w('')
w('=== [0] 追加源预检 ===')
w('源文件 %s bytes=%d lines=%d' % (os.path.basename(APP), len(src.encode('utf-8')), len(src.splitlines())))
for k in ANCHORS:
    w('  %-36s count=%d %s' % (k, src.count(k), 'OK' if src.count(k) >= 1 else '!! MISSING'))
assert not pre_miss, '追加源缺少锚点: %s' % pre_miss
w('  ==> 预检通过（%d 个锚点全部存在）' % len(ANCHORS))

raw0 = open(MEMO, 'rb').read()
txt = raw0.decode('utf-8-sig')
ALREADY = (HDRKEY in txt)
w('')
w('=== [0b] 追加前复核 ===')
w('  实际 bytes=%d sha8=%s' % (len(raw0), hashlib.sha256(raw0).hexdigest()[:8]))
assert raw0[:3] == b'\xef\xbb\xbf', 'BOM 丢失'
_now_sha8 = hashlib.sha256(raw0).hexdigest()[:8]
if ALREADY:
    w('  ==> MEMO 已含 Phase 20 节 ⇒ **纯复核模式**（跳过写入）')
elif _now_sha8 != BASE_SHA8 or len(raw0) != BASE_BYTES:
    raise AssertionError('MEMO 已被改动（基线 %d B/%s vs 实际 %d B/%s）——中止'
                         % (BASE_BYTES, BASE_SHA8, len(raw0), _now_sha8))
else:
    w('  ==> 与基线一致（%d B / %s）' % (BASE_BYTES, BASE_SHA8))

n0_b, n0_l = len(raw0), len(raw0.split(b'\r\n'))
w('')
w('=== [1] 追加执行 ===')
if not ALREADY:
    body = io.open(APP, encoding='utf-8').read().strip('\r\n')
    new = txt.rstrip('\r\n') + '\r\n\r\n' + body + '\r\n'
    out = b'\xef\xbb\xbf' + new.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
    open(MEMO, 'wb').write(out)
    w('  写入完成（非幂等路径）')
else:
    n0_b, n0_l = BASE_BYTES, BASE_LINES
    w('  跳过写入（幂等路径）')

rb = open(MEMO, 'rb').read()
t2 = rb.decode('utf-8-sig')
lines = t2.split('\r\n')
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 20')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('')
w('=== [2] 追加结果 ===')
w('append: bytes %d -> %d (+%d) ; lines %d -> %d (+%d)'
  % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines), len(lines) - n0_l))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'),
                                  rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('sha8   = %s' % hashlib.sha256(rb).hexdigest()[:8])
w('Phase 20 标题行 = %s' % hdr)
w('Phase 标题总数 = %d（追加前 %d）' % (len(allh), BASE_PHASES))

pre_raw = raw0[:BASE_BYTES] if ALREADY else raw0
pre_sha8 = BASE_SHA8 if ALREADY else _now_sha8
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
    w('  anchor %-36s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    if c < 1:
        miss.append(k)
w('缺失锚点数 = %d' % len(miss))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')

assert not miss, '锚点缺失: %s' % miss
assert len(hdr) == 1, 'Phase 20 标题不唯一: %s' % hdr
assert len(allh) == BASE_PHASES + 1, 'Phase 标题数应为 %d，实为 %d' % (BASE_PHASES + 1, len(allh))
assert rb[:3] == b'\xef\xbb\xbf'
assert rb.count(b'\n') - rb.count(b'\r\n') == 0, 'bare_lf 不为 0'
assert prefix_ok and anchor_ok, '前缀锚失败'
print('ALL CHECKS PASSED ->', OUT)
