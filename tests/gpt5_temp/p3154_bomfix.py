# -*- coding: utf-8 -*-
# p3154_bomfix.py: 修复 MEMO 头部 —— 字面 "\ufeff" 6 字节 -> 真实 BOM EF BB BF
import hashlib

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
b = open(MEMO, 'rb').read()
assert b[:6] == b'\\ufeff', ('unexpected head', b[:12])
assert b[6:14] == b'## AGI R', ('head content mismatch', b[6:20])
fixed = b'\xef\xbb\xbf' + b[6:]
with open(MEMO, 'wb') as f:
    f.write(fixed)
chk = open(MEMO, 'rb').read()
print('head now:', chk[:12])
print('len:', len(chk), 'sha8:', hashlib.sha256(chk).hexdigest()[:8])
txt = chk.decode('utf-8')
print('starts with BOM+title:', txt.startswith('\ufeff## AGI'))
print('marker present:', 'G1-P4' in txt)
