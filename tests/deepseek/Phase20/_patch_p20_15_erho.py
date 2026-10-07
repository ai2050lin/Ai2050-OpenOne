# -*- coding: utf-8 -*-
"""补丁 15：登记 [E-rho]（收尾自查抓到的渲染缺陷）到 gen_memo §11，并把锚点加入 do_append。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')
LOG = []


def rep(fname, old, new, tag, count=1):
    p = os.path.join(D20, fname)
    s = io.open(p, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, '[%s] 期望 %d 处，实为 %d：%s' % (fname, count, n, tag)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
    LOG.append('  OK  %-24s %s' % (fname, tag))


GM = 'gen_memo_phase20.py'
OLD = ("  '教训：**「沿用某容差」时必须把该容差的标定域一并写进判据文字**。')\n"
       "A('')\n"
       "A('### 12. 下一步（死线）')\n")
NEW = ("  '教训：**「沿用某容差」时必须把该容差的标定域一并写进判据文字**。')\n"
       "A('- **[E-rho] 渲染缺陷（收尾自查抓到、交付前已修）**：`rho_b_all` 是 `dict{rho,resid_med,resid_p90}`，'\n"
       "  '首版 §0/§4 误用标量取值器 `q()` 去取它 ⇒ 生成件里打印出原始 dict。修：新增 `qr()` 只取 `.rho`；'\n"
       "  'MEMO 的 Phase 20 节在交付前**回滚重生成**（前缀锚逐字节恢复后重跑追加链）。'\n"
       "  '教训：**取值器必须按被取对象的类型分层（标量 / 带 delta 的 dict / 嵌套 dict）**。')\n"
       "A('')\n"
       "A('### 12. 下一步（死线）')\n")
rep(GM, OLD, NEW, 'E-rho 勘误')

rep('do_append_phase20.py',
    "    'segfault', 'offload', 'E-sper', 'E-scope', 'E-probefull', 'E-baseline', 'E-xhdom',\n",
    "    'segfault', 'offload', 'E-sper', 'E-scope', 'E-probefull', 'E-baseline', 'E-xhdom', 'E-rho',\n",
    '锚点加 E-rho')

io.open(os.path.join(D20, '_patch_p20_15_erho.log'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
print('patched:', len(LOG))
