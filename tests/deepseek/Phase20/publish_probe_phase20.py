# -*- coding: utf-8 -*-
"""把探针臂记录发布到 **seal 冻结声明**的路径（`probe_files`）。
seal（38cfbcd6）在冻结时声明：
  probe_files.A0_nf4  = tests\\deepseek_temp\\Phase20\\_probe20_A0_nf4.json
  probe_files.A0_bf16 = tests\\deepseek_temp\\Phase20\\_probe20_A0_bf16.json
而驱动 `run_phase20_split.py` 在 PROBE 下按主脚本的落盘约定写成
  _armrec20_probe_A0_nf4.json / _armrec20_probe_A0_bf16.json
⇒ 本节把后者**复制**为前者（不删原件），使**冻结声明成立**（不改 seal、不动指纹链）。

幂等：目标已存在且 sha256 相同则跳过。
"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
PAIRS = [('_armrec20_probe_A0_nf4.json', '_probe20_A0_nf4.json'),
         ('_armrec20_probe_A0_bf16.json', '_probe20_A0_bf16.json')]


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


for src, dst in PAIRS:
    sp, dp = os.path.join(P20T, src), os.path.join(P20T, dst)
    assert os.path.exists(sp), '缺探针源文件: %s（probe 尚未完成？）' % sp
    sb = open(sp, 'rb').read()
    if os.path.exists(dp) and open(dp, 'rb').read() == sb:
        print('[skip] %s == %s (sha8=%s)' % (dst, src, sha8(sp)))
        continue
    io.open(dp, 'wb').write(sb)
    print('[pub ] %s -> %s  %d B  sha8=%s' % (src, dst, len(sb), sha8(dp)))
print('PUBLISH OK')
