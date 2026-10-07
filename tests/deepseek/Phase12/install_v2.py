# -*- coding: utf-8 -*-
"""把 v2 主脚本原子替换到正式路径，并保留 v1 备份。"""
import os, io, shutil, hashlib, py_compile

D = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase12'
SRC = os.path.join(D, 'n2h1a5_swap_alloc_v2.py')
DST = os.path.join(D, 'n2h1a5_swap_alloc.py')
BAK = os.path.join(D, 'n2h1a5_swap_alloc_v1.bak.py')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


assert os.path.exists(SRC), SRC
if os.path.exists(DST):
    shutil.copy2(DST, BAK)
    print('[swap] v1 backed up -> %s (%d bytes sha8 %s)' % (os.path.basename(BAK), os.path.getsize(BAK), sha(BAK)[:8]))
shutil.copy2(SRC, DST)
t = io.open(DST, encoding='utf-8').read()
assert 'xhalf' in t and 'amend1' in t and 'LAST_POS_STATE_SUFFICIENT' in t
assert "J_INJECT[int(_k)] = _v['jump_ratio']" in t
py_compile.compile(DST, doraise=True)
print('[swap] v2 installed -> %s (%d bytes sha8 %s)' % (os.path.basename(DST), os.path.getsize(DST), sha(DST)[:8]))
print('[swap] py_compile OK')
os.remove(SRC)
print('[swap] temp v2 removed ; DONE')
