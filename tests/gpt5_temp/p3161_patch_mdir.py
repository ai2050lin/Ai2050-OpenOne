# -*- coding: utf-8 -*-
"""Point qwen3-14b at the pre-quantized NF4 checkpoint (load-path only, design unchanged)."""
import py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3161_g4p4_head_attribution.py'
raw = open(P, 'rb').read()
s = raw.decode('utf-8')

OLD = "MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}"
NEW = ("MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'glm4': 'glm4-9b-chat-hf'}\n"
       "# 14b: pre-quantized NF4 checkpoint (converted once under tf4.57 streaming loader;\n"
       "# 5.14 on-the-fly bnb load materializes 29.5GB bf16 in RAM -> segfault, issue #43032\n"
       "# family). Same bnb 0.50.2 NF4 kernel; diag5 verified slot semantics = stock 5.14.\n"
       "MDIR_MAP['qwen3-14b'] = 'Qwen3-14B-bnb-nf4'")

cnt = s.count(OLD)
assert cnt == 1, 'OLD count=%d' % cnt
s2 = s.replace(OLD, NEW)
assert '\r' not in s2
open(P, 'wb').write(s2.encode('utf-8'))
s3 = open(P, 'rb').read().decode('utf-8')
assert s3 == s2, 'write/read mismatch'
assert s3.count("MDIR_MAP['qwen3-14b'] = 'Qwen3-14B-bnb-nf4'") == 1
py_compile.compile(P, doraise=True)
print('PATCH OK: MDIR_MAP[14b] -> Qwen3-14B-bnb-nf4, py_compile passed, size %d->%d' % (len(raw), len(s3.encode('utf-8'))))
