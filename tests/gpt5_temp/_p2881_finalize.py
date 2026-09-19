# -*- coding: utf-8 -*-
"""Phase 2881 finalization: SHA + ledger + MEMO (idempotent)."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P = os.path.join(BASE, 'phase2881', 'joint_word_coords')
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REPORT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2881_finalize.txt'

lines = []


def log(m):
    lines.append(m)


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


with io.open(os.path.join(P, 'result.json'), encoding='utf-8') as f:
    res = json.load(f)
v1 = res['v1']
j1, j1l = res['J1']['verdict'], res['J1']
j2, j2l = res['J2']['verdict'], res['J2']
j3 = res['J3']['transfer_matrix_rows=fa_dirs_cols=fb_words']
j4 = res['J4']

for fn in sorted(os.listdir(P)):
    log('SHA %s/%s = %s' % ('joint_word_coords', fn, sha8(os.path.join(
        P, fn))))

led = AtlasLedger.load(verify_sha=True)
bid = [b['block_id'] for b in led.doc['blocks']]
if 'B3_joint_3fam' not in bid:
    led.add_block({
        'block_id': 'B3_joint_3fam', 'axis_id': 'class',
        'kind': 'mlp_response', 'shape': [170, 210],
        'src': {'path': 'phase2881/joint_word_coords/'
                        'joint_word_coords.npz',
                'sha256_8': sha8(os.path.join(
                    P, 'joint_word_coords.npz')),
                'key': 'B3_joint'},
        'notes': '170 words x 21 family directions x 10 layers '
                 '(class/attr/syntax blocks 100/80/30 cols)'})

mid = [m['meas_id'] for m in led.doc['measurements']]
if 'M2881_joint_coords' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2881_joint_coords', 'type': 'joint_spectrum',
        'verdict': 'v1=true(det 0.0)/J1=joint_spectrum_carries_all_'
                   'families(acc 0.7471 vs null p95 0.0882, 8.5x)/'
                   'J2=gated_fusion_generalizes(own 0.7235 -> alpha*=0.5, '
                   'delta* 0.0529 > 0.0412)/J3 transfer all>0.76/'
                   'J4 family centroids mutually negative '
                   '(class-attr -0.65)',
        'source': {'path': 'phase2881/joint_word_coords/result.json',
                   'sha256_8': sha8(os.path.join(P, 'result.json')),
                   'phase': 2881}})

gids = [g['point_id'] for g in led.doc['growth_curve']]
if 'G_joint_3fam' not in gids:
    led.doc['growth_curve'].append({
        'point_id': 'G_joint_3fam', 'axis_id': 'class',
        'block': 'B3_joint_3fam', 'components': 210,
        'acc': j1l['acc_joint'], 'phase': 2881,
        'notes': 'three-family joint spectrum: 21 fine labels retrieved '
                 'at 8.5x null; gated fusion alpha*=0.5 generalizes '
                 'the 2869 law across families'})

lids = [l['link_id'] for l in led.doc['linkage']]
if 'L7_joint_channel' not in lids:
    led.doc['linkage'].append({
        'link_id': 'L7_joint_channel', 'from': {'block': 'B3_joint_3fam',
                                                'dim': 210},
        'to': {'kind': 'mlp_response', 'families': ['class', 'attr',
                                                    'syntax']},
        'evidence': 'all 21 fine labels retrievable from ONE joint mlp '
                    'spectrum (8.5x null); cross-family fusion adds '
                    '+0.053 at alpha*=0.5 (2869 law generalizes); '
                    'family centroids mutually negative (separated '
                    'subspaces of one channel, not three channels)',
        'phase': 2881, 'status': 'confirmed'})

led.save()
log('ledger saved')
led2 = AtlasLedger.load(verify_sha=False)
rep2 = []
st = led2.verify(rep2)
log('post stale: %s' % (st if st else 'none'))
log('blocks %d measurements %d growth %d linkage %d'
    % (len(led2.doc['blocks']), len(led2.doc['measurements']),
       len(led2.doc['growth_curve']), len(led2.doc['linkage'])))

sec = u"""

## Phase 2881：三族联合词坐标——一个 mlp 通道承载全部三个族谱（门控融合跨族推广）

**日期**：2026-09-18。**脚本**：`phase2881_joint_word_coords.py`（67.6s，
170 词 × 21 方向 × 10 层统一重测）。产物：`phase2881/joint_word_coords/`
{exec %s, result %s, npz %s}。

词表：class 80（2867 CATS verbatim）+ attr 42（2874 存活轴）+ syntax 48
（2878 存活轴）；方向 = class 10 类质心差分（2861 式，W_U）+ attr 8 +
syntax 3 = 21 方向 × L26-35 = 210 维联合谱（块 100/80/30）。

### 判决表（预注册 J1-J4）

| 判决 | 观测 | 结果 |
|---|---|---|
| **J1** 联合细标签检索（21 类） | acc = **0.7471** vs null p95 0.0882（**8.5×**） | **joint_spectrum_carries_all_families** |
| **J2** 密度门控融合 | own 0.7235 → **α\*=0.5**，Δ\* = **+0.0529** > null p95 0.0412 | **gated_fusion_generalizes** |
| **J3** 跨族迁移矩阵（最近质心） | 全部 >0.76（class 行 0.93-0.98 最强；syntax→attr 0.76 最弱） | 跨族方向可读出任意族标签 |
| **J4** 族质心余弦 | class↔attr **−0.65** / class↔syntax −0.33 / attr↔syntax −0.14；own acc class 0.80 / attr 0.52 / syntax 0.75 | 族子空间互斥分离 |

### 核心结论（重复三遍）

**一个 mlp 响应通道承载全部三个族谱：21 个细标签从同一 210 维联合谱以
8.5× null 检索成功；2869 密度门控定律跨族推广（α\*=0.5 增益 +0.053）；
族质心两两负相关——三个族是同一通道内互斥分离的子空间，不是三个通道。
载体曲线收官：class 0.875 / attr 0.500 / syntax 0.7083（own-block）→
联合谱细标签 0.7471（8.5× null）。族谱工程（V-族谱 → LLM 响应图谱）的
通道侧整合完成。**

### 登记差异

per-family own acc 与单族 phase 略异（class 0.80 vs 0.875@2867、attr
0.524 vs 0.500@2877、syntax 0.75 vs 0.708@2879）——统一协议重测 +
conds 细节差异，方向一致性保持（attr/syntax 几乎复现，class 略低系
2861 单词 conds 与 2-token conds 差异）。

### 接续

族谱-图谱整合线收官。下一步转入：A Atlas Ledger v2 升级（新增
syntax/translation 轴、joint 块、象限表与三族迁移矩阵入 spec）；
B 语法轴内部解剖（number 强边际 + tense 复议）；C 研究方向总结与
下一战役规划（见总结节）。
""" % (
    sha8(os.path.join(P, 'execution.json')),
    sha8(os.path.join(P, 'result.json')),
    sha8(os.path.join(P, 'joint_word_coords.npz')),
)

if 'Phase 2881：三族联合词坐标' not in io.open(
        MEMO, encoding='utf-8').read():
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write(sec)
    log('memo appended')
else:
    log('memo already has 2881 - skipped')

with io.open(MEMO, encoding='utf-8') as f:
    log('memo lines now %d' % len(f.readlines()))

with io.open(REPORT, 'w', encoding='utf-8') as g:
    g.write('\n'.join(lines) + '\n')
print('finalized')
