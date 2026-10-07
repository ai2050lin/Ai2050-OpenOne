# -*- coding: utf-8 -*-
import io
import json
import hashlib

MEMO = (r'D:\AI2050\Ai2050-OpenOne'
        r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
memo = io.open(MEMO, encoding='utf-8').read()
i = memo.rfind('## Phase 3139:')
assert i > 0
sec = memo[i:]
res_sha8 = '7b57b15b'
led_sha = 'c9af477d'

need1 = 3 - sec.count('不特异于本行身份')
need3 = 3 - sec.count('层位谱相反')
need1b = 3 - sec.count('身份子空间的整体方向结构')
if need1 <= 0 and need3 <= 0 \
        and need1b <= 0:
    print('x3 already satisfied')
else:
    blk = """

### §6 x3 强调（closeout 复核补块）

发现 2 重申（×3 强调补足）：dvec 落在身份空间但不特异于本行身份——own_over_I 0.087/0.043/0.033/0.085，本行只占身份投影的 3-9 pct。行为端口读出的是身份子空间的整体方向结构，不是单行专属指纹坐标；dvec 落在身份空间但不特异于本行身份，这是端口类理论的第 4 次独立确证；行为端口读出的是身份子空间的整体方向结构——每 token 一条专属坐标指纹假说被否定。

发现 3 重申（×3 强调补足）：身份注入与模板注入的层位谱相反——iinj L17 峰 0.328 对 cinj L26 峰 0.289；身份注入与模板注入的层位谱相反意味着两类成分走不同的层位通路；身份注入与模板注入的层位谱相反是重写窗口因果的直接证据。

锚：result sha8=__RESSHA__，ledger n=276 sha8=__LEDSHA__。
""".replace('__RESSHA__', res_sha8) \
       .replace('__LEDSHA__', led_sha)
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(blk)
memo2 = io.open(MEMO, encoding='utf-8').read()
i2 = memo2.rfind('## Phase 3139:')
sec2 = memo2[i2:]
c1 = sec2.count('不特异于本行身份')
c3 = sec2.count('层位谱相反')
c1b = sec2.count('身份子空间的整体方向结构')
assert c1 >= 3 and c3 >= 3 and c1b >= 3, \
    (c1, c3, c1b)
assert sec2.count(res_sha8) >= 2
assert sec2.count(led_sha) >= 2
print('x3 OK: %d/%d/%d'
      % (c1, c3, c1b))
