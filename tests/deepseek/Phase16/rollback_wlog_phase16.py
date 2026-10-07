# -*- coding: utf-8 -*-
"""回滚当日 wlog 中本轮刚追加的「## Phase 16 / N2h1-α-9」段（同轮修正，P15 先例）。
判据：以 `## Phase 16 / N2h1-` 标题为界，取 rfind 之前的全部内容为回滚点。
校验：回滚后文件不得再含该标题；并打印 sha256 供记录。
"""
import os
import hashlib

WLOG = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
MARK = '## Phase 16 / N2h1-'.encode('utf-8')

b = open(WLOG, 'rb').read()
i = b.rfind(MARK)
assert i > 0, '未找到 Phase 16 段标记'
pre = b[:i].rstrip(b'\r\n') + b'\r\n'
open(WLOG, 'wb').write(pre)

b2 = open(WLOG, 'rb').read()
print('bytes %d -> %d (%+d)' % (len(b), len(b2), len(b2) - len(b)))
print('contains Phase16 heading:', MARK in b2)
print('contains Phase15 heading:', '## Phase 15 / N2h1-'.encode('utf-8') in b2)
print('sha256 =', hashlib.sha256(b2).hexdigest())
print('tail:', repr(b2[-120:]))
assert MARK not in b2, '回滚失败：段仍在'
print('ROLLBACK OK')
