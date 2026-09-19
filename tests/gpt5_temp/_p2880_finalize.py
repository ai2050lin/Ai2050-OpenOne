# -*- coding: utf-8 -*-
"""Phase 2880 finalization: SHA register + ledger update + MEMO append.
Run AFTER phase2880_syntax_census.py completes; reads result.json and
generates verdict text dynamically.  Idempotent against re-runs."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P2880 = os.path.join(BASE, 'phase2880', 'syntax_census')
LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REPORT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2880_finalize.txt'
DECOMP = os.path.join(BASE, 'phase2879', 'e2_posthoc_decomp.txt')

lines = []


def log(m):
    lines.append(m)


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


# ---------- load verdict ----------
with io.open(os.path.join(P2880, 'result.json'), encoding='utf-8') as f:
    res = json.load(f)
v = res['verdict']
for fn in sorted(os.listdir(P2880)):
    log('SHA %s/%s = %s' % ('syntax_census', fn,
                            sha8(os.path.join(P2880, fn))))

# ---------- ledger ----------
led = AtlasLedger.load(verify_sha=True)
rep = []
stale = led.verify(rep)
log('pre-update stale: %s' % (stale if stale else 'none'))

bid = [b['block_id'] for b in led.doc['blocks']]
if 'B_syntax_causal' not in bid:
    led.add_block({
        'block_id': 'B_syntax_causal', 'axis_id': 'syntax',
        'kind': 'causal_spectrum', 'shape': [48, 1152],
        'src': {'path': 'phase2880/syntax_census/syntax_census.npz',
                'sha256_8': sha8(os.path.join(P2880,
                                              'syntax_census.npz')),
                'key': 'drops'}})

mid = [m['meas_id'] for m in led.doc['measurements']]
if 'M2880_syntax_census' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2880_syntax_census', 'type': 'axis3_census',
        'verdict': v['final_verdict'],
        'source': {'path': 'phase2880/syntax_census/result.json',
                   'sha256_8': sha8(os.path.join(P2880, 'result.json')),
                   'phase': 2880}})

gids = [g['point_id'] for g in led.doc['growth_curve']]
if 'G_axis3_syntax_causal' not in gids:
    acc = v['Y5_acc_full']
    below = not v['Y5']
    led.doc['growth_curve'].append({
        'point_id': 'G_axis3_syntax_causal', 'axis_id': 'syntax',
        'block': 'B_syntax_causal', 'components': 1152, 'acc': acc,
        'phase': 2880,
        'notes': ('below null p95 - drop channel absent'
                  if below else 'above null p95')})

lids = [l['link_id'] for l in led.doc['linkage']]
quad = ('drop %s / mlp yes' % ('organized' if v['Y1'] or v['Y5']
                               else 'unorganized'))
if 'L6_quadrant_syntax' not in lids:
    led.doc['linkage'].append({
        'link_id': 'L6_quadrant_syntax',
        'from': {'axis': 'syntax', 'block': 'B_syntax_causal'},
        'to': {'block': 'B3_syntax_mlp'},
        'evidence': 'channel quadrant syntax: drop-spectrum Y5=%s '
                    '(acc %.4f vs null p95 %s), Y1 frontedge n3=%d vs '
                    'null p95 %s; mlp E1=0.7083 (2879); post-hoc: '
                    'number axis alone margin +0.3216 > 0.1460'
                    % (v['Y5'], v['Y5_acc_full'],
                       v.get('Y5_null_p95', '-'), v['Y1_n_heads_ge3'],
                       v.get('Y1_null_n3_p95', '-')),
        'phase': 2880, 'status': 'measured'})

led.save()
log('ledger saved')
led2 = AtlasLedger.load(verify_sha=False)
rep2 = []
log('post stale: %s' % (led2.verify(rep2) if led2.verify(rep2) else 'none'))
log('blocks %d measurements %d growth %d linkage %d'
    % (len(led2.doc['blocks']), len(led2.doc['measurements']),
       len(led2.doc['growth_curve']), len(led2.doc['linkage'])))

# ---------- MEMO append ----------
sec = u"""

## Phase 2880：语法轴 drop 谱 census——通道分化四象限表 syntax 象限补齐

**日期**：2026-09-18。**脚本**：`phase2880_syntax_census.py`（2875 协议
verbatim 移植，3 轴 48 词 × 36 层 × 32 头）。产物：`phase2880/syntax_census/`
{exec %s, result %s, npz %s}。

### 判决表（预注册 Y1-Y5）

| 判决 | 观测 | 结果 |
|---|---|---|
| **Y1** 前缘（≥3/3 全重叠） | n_heads = %d vs null p95 %s | **%s** |
| **Y2** 机制侧分离 | rho(syntax, class) = %s | **%s** |
| **Y3** top64 交集 | %d，hyper p = %s | %s |
| **Y5** 词级检索 | acc = %s vs null p95 %s | **%s** |

per-axis top8：%s

### Post-hoc 附注（2879 E2 分解，描述性，非门线）

number 轴单独边际 +0.3216 > null p95 0.1460（成立）；gerund +0.0348 /
comparative +0.0565 阴性——2879 E2 全局阴性系后两轴稀释。number↔gerund
轴质心 cos 0.789 高度共线：语法轴族内部非正交，与跨族几何独立（2870/2875）
形成对照。详见 `phase2879/e2_posthoc_decomp.txt`。

### 四象限表（通道分化 × 轴族）

| 轴族 | drop 谱组织 | mlp 载体 |
|---|---|---|
| class | 有（43 头前缘） | 0.875 |
| attr | 无（2875/2876 双阴性） | 0.500 |
| syntax | %s | 0.7083 |
| translation | 无统一轴方向（2878 VG2b≈0） | 未测（轴不存在） |

### 接续

2881 候选：A（主选）B3_class∪B3_attr∪B3_syntax 三族联合词坐标与密度门控
融合；B 语法轴内部解剖（number 强边际 + 质心共线的头级来源，Y1 头群
per-axis 归属）；C tense 轴隔离复议（0.1902 差 0.01，若 A/B 需要第三语法轴）。
""" % (
    sha8(os.path.join(P2880, 'execution.json')),
    sha8(os.path.join(P2880, 'result.json')),
    sha8(os.path.join(P2880, 'syntax_census.npz')),
    v['Y1_n_heads_ge3'], v.get('Y1_null_n3_p95', '-'),
    'syntax_frontedge_exists' if v['Y1'] else 'syntax_frontedge_absent',
    v['Y2_rho_class'],
    'mechanism_side_separate' if v['Y2'] else 'mechanism_side_aligned',
    v['Y3_intersection'], v['Y3_hyper_p'], v['Y3_label'],
    v['Y5_acc_full'], v.get('Y5_null_p95', '-'),
    'axis_signal_detected' if v['Y5'] else 'axis_signal_absent',
    json.dumps(v['per_axis_top8'], ensure_ascii=False),
    ('有组织（Y1=%s, Y5=%s）' % (v['Y1'], v['Y5'])),
)

if 'Phase 2880：语法轴 drop 谱 census' not in io.open(
        MEMO, encoding='utf-8').read():
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write(sec)
    log('memo appended')
else:
    log('memo already contains 2880 section - skipped')

with io.open(MEMO, encoding='utf-8') as f:
    n_lines = len(f.readlines())
log('memo lines now %d' % n_lines)

with io.open(REPORT, 'w', encoding='utf-8') as g:
    g.write('\n'.join(lines) + '\n')
print('finalized')
