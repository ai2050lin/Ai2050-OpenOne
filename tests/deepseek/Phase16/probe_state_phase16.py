# -*- coding: utf-8 -*-
"""Phase 16 开工前状态探针（只读）。
落点: tests/deepseek_temp/Phase16/_probe_state.txt
"""
import io
import json
import os
import hashlib
import time

R = r'D:\AI2050\Ai2050-OpenOne'
OUTDIR = R + r'\tests\deepseek_temp\Phase16'
os.makedirs(OUTDIR, exist_ok=True)
OUT = OUTDIR + r'\_probe_state.txt'

L = []


def p(*a):
    L.append(' '.join(str(x) for x in a))


def stat(path):
    if not os.path.exists(path):
        return (None, None)
    b = open(path, 'rb').read()
    return (len(b), hashlib.sha256(b).hexdigest()[:8])


p('=== [0] 时间 ===')
p('now_local', time.strftime('%Y-%m-%d %H:%M:%S'))

p('')
p('=== [1] 关键文件 ===')
for n, rel in [('MEMORY', r'\.workbuddy\memory\MEMORY.md'),
               ('wlog1002', r'\.workbuddy\memory\2026-10-02.md'),
               ('MEMO', r'\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'),
               ('ledger', r'\research\gpt5\atlas\atlas_ledger.json'),
               ('baseline', r'\tests\deepseek_temp\_infra\memo_baseline.json'),
               ('result15', r'\tests\deepseek_temp\Phase15\result_phase15.json'),
               ('present15', r'\tests\deepseek_temp\Phase15\present_phase15.html'),
               ('seal15', r'\tests\deepseek_temp\Phase15\N2h1a8_design_seal.json'),
               ('amend1', r'\tests\deepseek_temp\Phase15\N2h1a8_design_seal_amend1.json'),
               ('exec15', r'\tests\deepseek_temp\Phase15\execution_phase15.json')]:
    sz, s8 = stat(R + rel)
    p('%-10s %10s B  sha8=%s' % (n, sz, s8))

p('')
p('=== [2] MEMORY.md 规模（是否仍超限）===')
mp = R + r'\.workbuddy\memory\MEMORY.md'
if os.path.exists(mp):
    s = io.open(mp, encoding='utf-8').read()
    p('chars', len(s), 'bytes', len(s.encode('utf-8')))
    p('headings:', [ln[:70] for ln in s.split('\n') if ln.startswith('#')])

p('')
p('=== [3] MEMO 尾部 ===')
memo = R + r'\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
b = open(memo, 'rb').read()
d = b.decode('utf-8-sig')
lines = d.split('\r\n') if '\r\n' in d else d.split('\n')
p('bytes', len(b), 'bom', b[:3] == b'\xef\xbb\xbf',
  'crlf', b.count(b'\r\n'), 'bare_lf', b.count(b'\n') - b.count(b'\r\n'))
hdrs = [(i + 1, ln) for i, ln in enumerate(lines) if ln.startswith('## Phase ')]
p('phase_headings n =', len(hdrs))
for i, h in hdrs[-4:]:
    p('   L%-6d %s' % (i, h[:100]))
p('--- 尾部 1800 字符 ---')
p(d[-1800:])

p('')
p('=== [4] result_phase15.json 结构 ===')
rp = R + r'\tests\deepseek_temp\Phase15\result_phase15.json'
if os.path.exists(rp):
    res = json.load(io.open(rp, encoding='utf-8'))
    p('top_keys:', sorted(res.keys()))
    p('joint_verdict:', json.dumps(res.get('joint_verdict'), ensure_ascii=False))
    p('verdicts:', json.dumps(res.get('verdicts'), ensure_ascii=False)[:800])
    for ac in ('E1C', 'E2C', 'E3L', 'E4C', 'E5C', 'E6C'):
        if ac in res:
            v = res[ac]
            if isinstance(v, dict):
                arms = sorted(v.keys())
                p('%s arms=%s' % (ac, arms))
                for a in arms:
                    p('   %s subkeys=%s' % (a, sorted(v[a].keys()) if isinstance(v[a], dict) else type(v[a]).__name__))
    # 打印 E3L / E5C 全文（短）
    for ac in ('E3L', 'E5C'):
        if ac in res:
            p('--- %s 全文 ---' % ac)
            p(json.dumps(res[ac], ensure_ascii=False, indent=1)[:4000])

p('')
p('=== [5] Ledger ===')
lp = R + r'\research\gpt5\atlas\atlas_ledger.json'
if os.path.exists(lp):
    led = json.load(io.open(lp, encoding='utf-8'))
    ms = led['measurements']
    p('n =', len(ms))
    for m in ms[-3:]:
        p('   ', json.dumps({k: m.get(k) for k in ('phase', 'line', 'verdict', 'ts', 'sha8', 'title') if k in m},
                            ensure_ascii=False)[:300])

p('')
p('=== [6] baseline ===')
bp = R + r'\tests\deepseek_temp\_infra\memo_baseline.json'
if os.path.exists(bp):
    bs = json.load(io.open(bp, encoding='utf-8'))
    p(json.dumps({k: bs.get(k) for k in ('tag', 'bytes', 'lines', 'sha8') if k in bs}, ensure_ascii=False))
    p('history_n =', len(bs.get('history', []) or []))

open(OUT, 'w', encoding='utf-8', newline='\r\n').write('\r\n'.join(L) + '\r\n')
print('WROTE', OUT, len(L), 'lines')
