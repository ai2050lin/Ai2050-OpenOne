# -*- coding: utf-8 -*-
"""Phase 13 收尾补丁：修正 disk_verify_phase13.py 的 [H_docs] 三处**陈旧常量**。

背景：A8 勘误触发「MEMO / Ledger / wlog 三链按基线 sha256 回滚 → 重跑 whole chain」，
追加后的 MEMO 字节数与 sha8 已从 308702 / 0c192b97 变为 308863 / 4f9b4574，
baseline.bytes 同步变化。复核脚本里的期望值仍是回滚前的旧值，导致 3 个**伪 FAIL**。

本补丁只改期望常量，不动任何判据逻辑。
纪律：逐处 assert count==1 + 回读复核（铁律 o）。
"""
import io
import os
import hashlib

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'disk_verify_phase13.py')
src = io.open(P, encoding='utf-8').read()

PAIRS = [
    ("('MEMO bytes == 308702', len(mb) == 308702),",
     "('MEMO bytes == 308863', len(mb) == 308863),"),
    ("('MEMO sha8 == 0c192b97', hashlib.sha256(mb).hexdigest()[:8] == '0c192b97'),",
     "('MEMO sha8 == 4f9b4574', hashlib.sha256(mb).hexdigest()[:8] == '4f9b4574'),"),
    ("('baseline.bytes == 308702', BSE['bytes'] == 308702),",
     "('baseline.bytes == 308863', BSE['bytes'] == 308863),"),
]

for old, new in PAIRS:
    n = src.count(old)
    assert n == 1, 'count(%r) == %d != 1' % (old[:40], n)
    src = src.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='').write(src)

# --- 回读复核（读真实磁盘）---
back = io.open(P, encoding='utf-8').read()
for _, new in PAIRS:
    assert back.count(new) == 1, 'readback missing %r' % new[:40]
assert back.count('308702') == 0, '残留旧常量 308702'
assert back.count('0c192b97') == 0, '残留旧 sha8 0c192b97'

# py_compile 自检
import py_compile
py_compile.compile(P, doraise=True)

print('[1] 三处陈旧常量已更新（count==1 逐处通过）')
print('    MEMO bytes 308702 -> 308863 ; sha8 0c192b97 -> 4f9b4574 ; baseline.bytes -> 308863')
print('    py_compile OK ; readback OK ; 旧值残留 0')
print('    file sha8 =', hashlib.sha256(io.open(P, 'rb').read()).hexdigest()[:8])
