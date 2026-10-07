# -*- coding: utf-8 -*-
"""修 disk_verify_phase21.py 第 55 行：8-char sha8 与 64-hex seal_sha256 的错配比较。"""
import os
import hashlib

SRC = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase21\disk_verify_phase21.py'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase21\_patch_diskverify_report.txt'
o = []


def w(s=''):
    o.append(str(s))


OLD = ("chk('seal sha 与 exec.seal_sha256 一致', h8(SEAL) == EX['seal_sha256'], "
       "h8(SEAL), EX['seal_sha256'][:8])")
NEW = ("_seal_full = hashlib.sha256(open(SEAL, 'rb').read()).hexdigest()\n"
       "chk('seal sha 与 exec.seal_sha256 一致', _seal_full == EX['seal_sha256'], "
       "_seal_full[:8], EX['seal_sha256'][:8])")

raw = open(SRC, 'rb').read()
bom = raw[:3] == b'\xef\xbb\xbf'
t = raw.decode('utf-8-sig')
n = t.count(OLD)
w('OLD count = %d' % n)
assert n == 1, 'OLD 匹配数 != 1 -> %d' % n
t2 = t.replace(OLD, NEW)
assert t2 != t and t2.count(NEW) == 1
open(SRC, 'wb').write((('\ufeff' if bom else '') + t2).encode('utf-8'))
rb = open(SRC, 'rb').read().decode('utf-8-sig')
w('after: OLD=%d NEW=%d' % (rb.count(OLD), rb.count(NEW)))

# 独立核对：seal 全 sha 与 exec.seal_sha256 确实一致
P21T = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase21'
import io
import json
SEAL = os.path.join(P21T, 'N2h1a14_design_seal.json')
EXEC = os.path.join(P21T, 'execution_phase21.json')
EX = json.load(io.open(EXEC, encoding='utf-8'))
full = hashlib.sha256(open(SEAL, 'rb').read()).hexdigest()
w('')
w('seal full sha256 = %s' % full)
w('exec.seal_sha256  = %s' % EX['seal_sha256'])
w('MATCH = %s' % (full == EX['seal_sha256']))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('PATCH DISKVERIFY OK')
