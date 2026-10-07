# -*- coding: utf-8 -*-
"""MEMORY.md 定稿复核：体积 / 结构 / 铁律字母 / 过期残留。"""
import io
import os
import re
import hashlib

p = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\verify_memory_phase14.txt'
b = open(p, 'rb').read()
t = b.decode('utf-8')

must = ['(a) 份额用', '(b) SMOKE', '(c) 门裕度', '(d) 预注册/修正案记 sha8', '(e) 行为端与层内端同测',
        '(f)「充分」放大臂', '(g) 曲线判据', '(h) 冒烟产物落', '(i) 冒烟截断网格', '(j) 跨位点分',
        '(k) 判据符号与物理方向一致', '(l) `r_ℓ` 差 3–17 倍', '(m) 读数端机制必做', '(n) 固定基必报',
        '(o) **同消息多 Edit', '(p) 非线性/极值统计量', '(q)「构造决定的位点」', '(r) 端点构造饱和',
        '(s)「首次达比例」', '(t) 集中度须', '(u)「不可排序」须写区间口径',
        '(v)「新独立口径」', '(w) 面板级 vs 逐对', '(x) 预注册预测符号须与自身', '(y) GQA 禁用']
bad = ['n=**296**', '308,863 B / 3149 行', '（46 条', '（v) 面板级', '(w) GQA', '(x) 禁用 `hidden',
       '(t)「另一个独立口径」', '(u) 面板级恒等式', '(v) 预注册预测', '(h) 跨位点分', '(i) 判据符号须',
       '(j) `r_ℓ` 可差', '(k) 读数端解释机制', '(l) 固定基须报', '(m)「应由构造决定的位点」',
       '(n) 端点量若构造饱和', '(q)「首次达比例」型判据', '(r) 集中度判据须', '(s)「不可排序」型结论须']
anchors = ['n=**297**', '346,025 B / 3,427 行 / sha8 c9b4b3f5', '七次 P8–P14', 'P14】', 'Phase 15'.replace('Phase ', 'P15 '),
           '50 坑', '20 教训', '0.002722', '0.6998', 'FAMILY_TRANSFER_BOTH', 'null 95 分位']

secs = re.split(r'(?m)^(## .*)$', t)
L = ['=== verify MEMORY.md（Phase 14 定稿）===',
     'bytes=%d chars=%d lines=%d sha8=%s bom=%s bare_lf=%d' % (
         len(b), len(t), len(b.split(b'\n')), hashlib.sha256(b).hexdigest()[:8],
         b[:3] == b'\xef\xbb\xbf', b.count(b'\n') - b.count(b'\r\n')), '']
L.append('--- 分区体积 ---')
for i in range(1, len(secs), 2):
    L.append('  %-40s %5d B' % (secs[i][:40], len(secs[i + 1].encode('utf-8'))))
L.append('')
L.append('--- 锚点 ---')
for a in anchors:
    L.append('  %-40s %d' % (a, t.count(a)))
L.append('')
L.append('--- 铁律字母（须全 1）---')
for a in must:
    L.append('  %-38s %d' % (a[:38], t.count(a)))
L.append('')
L.append('--- 过期/错位残留（须全 0）---')
for a in bad:
    L.append('  %-38s %d' % (a[:38], t.count(a)))
ok = (all(t.count(a) == 1 for a in must) and all(t.count(a) == 0 for a in bad)
      and all(t.count(a) >= 1 for a in anchors) and len(b) < 10000)
L += ['', 'ALL OK' if ok else 'HAS FAIL', '']
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
