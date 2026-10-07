# -*- coding: utf-8 -*-
import os
ROOT = r'D:\AI2050\Ai2050-OpenOne'
rep = []

# ---------- 1) 工作区日志（幂等 append）----------
log = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
T = open(log, encoding='utf-8').read()
if '## N1 探针：主轴三段分工' in T:
    rep.append('wlog: N1 section already present, skipped')
else:
    add = open(os.path.join(ROOT, 'gpt5_temp', 'wlog_n1_section.md'), encoding='utf-8').read()
    b0 = os.path.getsize(log)
    with open(log, 'a', encoding='utf-8', newline='') as f:
        f.write('\n' + add)
    rep.append('wlog %d -> %d' % (b0, os.path.getsize(log)))
T = open(log, encoding='utf-8').read()
rep.append('wlog has N1 section: %s' % ('## N1 探针：主轴三段分工' in T))
rep.append('wlog bytes now: %d' % len(T.encode('utf-8')))

# ---------- 2) MEMORY.md ----------
mem = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
Tm = open(mem, encoding='utf-8').read()
b1 = len(Tm.encode('utf-8'))
anchor = "软门记录型优先于硬 assert（防长跑尾炸）。"
if 'MEMO 非 Phase 节有被外部进程删除的风险' in Tm:
    rep.append('MEMORY: item4 already present, skipped')
else:
    assert Tm.count(anchor) == 1, 'anchor count %d' % Tm.count(anchor)
    Tm = Tm.replace(anchor, anchor + "\n4. **MEMO 非 Phase 节有被外部进程删除的风险**（2026-10-01 事故）：03:15 存在的 `## 设计草案` + `## 探索性探针 E1` 两节在 03:43 消失（恰好 250 行，Phase 3149 随之上移）。按 Phase 的 closeout 脚本含 `open(*,'w')` 模式且项目存在 memcompress 实践。对策：非 Phase 交付**必须同时留 standalone 文档**；MEMO 追加后须二次 Grep 复核存在性。")

n1sec = open(os.path.join(ROOT, 'gpt5_temp', 'memory_n1_section.md'), encoding='utf-8').read()
if '## 主轴三段分工（N1，2026-10-01，6 模型）' not in Tm:
    Tm = Tm.rstrip('\n') + '\n' + n1sec
with open(mem, 'w', encoding='utf-8', newline='') as f:
    f.write(Tm)
T2 = open(mem, encoding='utf-8').read()
rep.append('MEMORY.md %d -> %d bytes' % (b1, len(T2.encode('utf-8'))))
rep.append('MEMORY has item4: %s' % ('MEMO 非 Phase 节有被外部进程删除的风险' in T2))
rep.append('MEMORY has N1 sec: %s' % ('## 主轴三段分工（N1，2026-10-01，6 模型）' in T2))

open(os.path.join(ROOT, 'gpt5_temp', 'verify_memory_n1.txt'), 'w', encoding='utf-8').write('\n'.join(rep))
print('done')
