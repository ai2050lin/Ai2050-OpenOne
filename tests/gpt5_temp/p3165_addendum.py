# -*- coding: utf-8 -*-
# 3165 addendum: numeric sensitivity band note (append-only, idempotent, CRLF/BOM safe)
import io
import shutil

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
DAILY = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-09.md'
MARK = '3165 补记'

raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if MARK in norm:
    print('memo: addendum already present, skip')
else:
    block = (
        '\n### ' + MARK + '（复核期发现，正式观测后追加）\n\n'
        '独立复核期定位一个**数值敏感带**：K_readout×S_class 的 top-1 主角在 S_class 的 '
        'float32/float64 口径下分别为 67.000° / 66.726°（差 0.27°）——S_class 行空间病态'
        '（10 个类质心差分方向近相关，即 2881 J4 互斥结构的另一面，行空间最小奇异值极小）'
        '对 3.7e-9 级 float32 量化扰动作出放大响应。**两侧均远离 30°/15° 预注册门，'
        '判决 separable 不变**；登记口径 = 主脚本 float32 round-trip 链（res 67.000）。'
        'verify 复算已改为同 dtype 链后 24/24 ALL PASS。教训：病态行空间子空间的主角度'
        '读数须登记 dtype 口径；门判决与敏感带分离陈述。\n\n\n---\n\n\n')
    new = norm + block
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3165b')
    with open(MEMO, 'wb') as f:
        f.write(out)
    print('memo: addendum appended')

dtxt = io.open(DAILY, encoding='utf-8').read()
line = '- **3165 补记**：K_readout×S_class 主角色值敏感带（float32 67.000° / float64 66.726°，' \
       'S_class 行空间病态放大 3.7e-9 量化扰动）；两侧远离 30/15 门，判决不变；verify 同 dtype 链 24/24。'
if '3165 补记' in dtxt:
    print('daily: already present, skip')
else:
    if not dtxt.endswith('\n'):
        dtxt += '\n'
    dtxt += line + '\n'
    with io.open(DAILY, 'w', encoding='utf-8', newline='') as f:
        f.write(dtxt)
    print('daily: appended')
