# -*- coding: utf-8 -*-
"""把 do_append_phase21.py 的 ANCHORS 块替换为与 memo_append_phase21.md 实际用词一致的版本。

纪律：精确块匹配 + count==1 断言 + 回读复核 + 对追加源逐一 count 预检。
只读 memo_append（不改），只改 do_append 脚本。
"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'deepseek', 'Phase21', 'do_append_phase21.py')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21', 'memo_append_phase21.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21', '_patch_anchors_report.txt')

o = []


def w(s=''):
    o.append(str(s))


OLD = (
    "ANCHORS = [\n"
    "    HDRKEY, 'N2h1-\u03b1-14',\n"
    "    'share_v', 'com_V', 'com_layer', 'com_B', 'b_{c,\u2113}', 'w_l', 'vec_budget',\n"
    "    'A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16',\n"
    "    'qwen3-4b', 'glm4-9b', 'Qwen3-14B', 'nf4', 'bf16',\n"
    "    'segfault', 'offload', 'identity-probe', 'identity-probe \u63a2\u9488', 'I_nl',\n"
    "    'Phase 22', 'P8',\n"
    "]"
)

NEW = (
    "ANCHORS = [\n"
    "    HDRKEY, 'N2h1-alpha-14',\n"
    "    'share_v', 'com_V', 'com_layer', 'vec_budget', 'b_{c,l}', 'w_l',\n"
    "    'A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16',\n"
    "    'qwen3-4b', 'glm4-9b', 'Qwen3-14B', 'nf4', 'bf16',\n"
    "    'segfault', 'offload', '\u5355\u4f4d\u9635\u63a2\u9488', 'I_nl',\n"
    "    '\u540e\u7eed\u6b7b\u7ebf', 'P8', 'max_head_share_v', 'W.max_head_share',\n"
    "]"
)

raw = open(SRC, 'rb').read()
bom = raw[:3] == b'\xef\xbb\xbf'
txt = raw.decode('utf-8-sig')
w('=== patch do_append ANCHORS ===')
w('SRC = %s' % SRC)
w('SRC bytes(before) = %d  sha8=%s  bom=%s' % (len(raw), hashlib.sha256(raw).hexdigest()[:8], bom))

n_old = txt.count(OLD)
n_new = txt.count(NEW)
w('OLD block occurrences = %d' % n_old)
w('NEW block occurrences (should be 0 before patch) = %d' % n_new)
assert n_old == 1, 'OLD 块匹配数 != 1 -> %d' % n_old
assert n_new == 0, 'NEW 块已存在 -> %d' % n_new

newtxt = txt.replace(OLD, NEW)
assert newtxt != txt
out = ('\ufeff' if bom else '') + newtxt
open(SRC, 'wb').write(out.encode('utf-8'))

# 回读复核
raw2 = open(SRC, 'rb').read()
t2 = raw2.decode('utf-8-sig')
w('SRC bytes(after)  = %d  sha8=%s' % (len(raw2), hashlib.sha256(raw2).hexdigest()[:8]))
w('after: OLD count=%d  NEW count=%d' % (t2.count(OLD), t2.count(NEW)))
w('')
w('=== 锚点实存性预检（对 memo_append_phase21.md）===')
src = io.open(APP, encoding='utf-8-sig').read()
anchors = [
    '## Phase 21:', 'N2h1-alpha-14',
    'share_v', 'com_V', 'com_layer', 'vec_budget', 'b_{c,l}', 'w_l',
    'A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16',
    'qwen3-4b', 'glm4-9b', 'Qwen3-14B', 'nf4', 'bf16',
    'segfault', 'offload', '单位阵探针', 'I_nl',
    '后续死线', 'P8', 'max_head_share_v', 'W.max_head_share',
]
miss = []
for k in anchors:
    c = src.count(k)
    w('  %-24s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    if c < 1:
        miss.append(k)
w('缺失锚点 = %d %s' % (len(miss), miss))
w('')
w('=== 一致性核对：do_append 内 ANCHORS 列表 vs 上面 anchors 列表 ===')
import re
m = re.search(r'ANCHORS = \[(.*?)\n\]', t2, re.S)
listed = re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1))
w('do_append ANCHORS 条目 = %s' % listed)
w('列条目数 = %d ; 预检条目数 = %d' % (len(listed), len(anchors)))
w('hdr 键在列条目中 = %s' % ('## Phase 21:' in listed))
w('missing entries in listed = %s' % [a for a in anchors if a not in listed and a != '## Phase 21:'])
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('PATCH OK ->', OUT)

assert not miss, '追加源仍缺锚点: %s' % miss
assert len(listed) == len(anchors) - 1, '列条目数不符: %d vs %d' % (len(listed), len(anchors) - 1)
