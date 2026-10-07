# -*- coding: utf-8 -*-
"""Phase 8 开工侦察：GPU 空闲 / 在跑 Phase / 死线原文 / 可复用资产。"""
import os, io, re, time, glob, json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'recon_phase8.txt')
o = []
o.append('recon at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))

# 1. GPU
try:
    import torch
    o.append('torch %s cuda %s' % (torch.__version__, torch.cuda.is_available()))
    if torch.cuda.is_available():
        free, tot = torch.cuda.mem_get_info()
        o.append('gpu free %.2f GB / total %.2f GB' % (free / 1e9, tot / 1e9))
        o.append('device %s' % torch.cuda.get_device_name(0))
except Exception as e:
    o.append('torch fail %r' % e)
try:
    import subprocess
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv'], capture_output=True, text=True)
    o.append('nvidia-smi compute-apps rc=%s' % r.returncode)
    o.append(r.stdout.strip()[:400])
except Exception as e:
    o.append('nvidia-smi fail %r' % e)

# 2. 在跑 Phase 检测（deepseek 线 & glm5 线）
for pat in [os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase*'),
            os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913', 'phase*')]:
    hits = sorted(glob.glob(pat))
    o.append('--- glob %s -> %d' % (pat.replace(ROOT, '.'), len(hits)))

# 3. 备忘录结构 + Phase 6/7 后续节
memo = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
T = io.open(memo, encoding='utf-8-sig').read()
L = T.splitlines()
o.append('memo bytes %d lines %d' % (os.path.getsize(memo), len(L)))
heads = [(i + 1, l) for i, l in enumerate(L) if l.startswith('## ')]
o.append('--- sections ---')
for i, l in heads:
    o.append('  L%-5d %s' % (i, l[:100]))

def section(start_kw, stop_kws=('## ',)):
    s = None
    for i, l in enumerate(L):
        if l.startswith(start_kw):
            s = i
            break
    if s is None:
        return ''
    e = len(L)
    for j in range(s + 1, len(L)):
        if any(L[j].startswith(k) for k in stop_kws):
            e = j
            break
    return '\n'.join(L[s:e])

# Phase 6 §8 后续与资源
sec6 = section('## Phase 6')
m = re.search(r'### 8\.?\s*后续与资源(.*?)(?=\n### |\Z)', sec6, re.S)
o.append('--- Phase 6 后续与资源 ---')
o.append((m.group(1).strip()[:2600] if m else 'NOT FOUND'))
sec7 = section('## Phase 7')
m7 = re.search(r'### 7\.?\s*后续与资源(.*?)(?=\n### |\Z)', sec7, re.S)
o.append('--- Phase 7 后续与资源 ---')
o.append((m7.group(1).strip()[:2600] if m7 else 'NOT FOUND'))

# 4. N2h1 资产
o.append('--- N2h1 / N1 资产 ---')
for d in [os.path.join(ROOT, 'tests', 'deepseek', 'Phase4'),
          os.path.join(ROOT, 'tests', 'deepseek', 'Phase6'),
          os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase6')]:
    if os.path.isdir(d):
        o.append('  [%s]' % d.replace(ROOT, '.'))
        for f in sorted(os.listdir(d)):
            o.append('     %-46s %8d' % (f, os.path.getsize(os.path.join(d, f))))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('wrote', OUT, len(o), 'lines')
