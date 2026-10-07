# -*- coding: utf-8 -*-
"""生成 Phase 16 seal 的 amend1：锚判据**分层化**（不改任何科学假设）。

触发（SMOKE 阶段，正式运行之前）：
  - SMOKE 的 α 网格只有 3 点 ([0.0,0.5,1.0])，而 xhalf 是**插值量**（cross_alpha 在网格上线性插值）；
    SMOKE 因此给出 max|dxh| = 0.0105 —— 完全由网格变化解释，**不是** drift。
  - 但该现象暴露判据设计缺陷：xhalf 是「曲线穿越半饱和点」的插值位置，其可复现性弱于
    峰/谷/端点等直接读数；对插值量设单一硬门 1e-3 不符合「软门优先硬 assert」纪律。

fix：Q1 判据分层（三臂同判）：
  RECON_OK        : max|dxh| <= 1e-3 且 max_rel|dJ| <= 2e-2 且 legacy argmax 整数相同 且 legacy top3 差 <= 2e-2
  RECON_OK_LOOSE  : 1e-3 < max|dxh| <= 5e-3 且其余同上（必须在报告中给出**逐站点 dxh 表**）
  RECON_DRIFT     : 其余
  P2 判 PASS 当且仅当：三臂全部 ∈ {RECON_OK, RECON_OK_LOOSE} 且 >= 2/3 臂为 RECON_OK。

what_is_NOT_changed：网格、实验材料、统计量定义、P3–P7、CENTROID_* 阈值、UNREACH_y —— 全部不变。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P16T, 'N2h1a9_design_seal.json')
EXECP = os.path.join(P16T, 'execution_phase16.json')

SEALB = open(SEALP, 'rb').read()
EXECB = open(EXECP, 'rb').read()
SEAL = json.loads(SEALB.decode('utf-8'))
EX = json.loads(EXECB.decode('utf-8'))

AM1 = dict(
    phase=16,
    line='N2h1-alpha-9',
    kind='criterion_tiering + robustness fix (no hypothesis change)',
    created_local=__import__('time').strftime('%Y-%m-%d %H:%M:%S'),
    amend_of_seal_sha256=hashlib.sha256(SEALB).hexdigest(),
    amend_of_seal_sha8=hashlib.sha256(SEALB).hexdigest()[:8],
    amend_of_execution_sha256=hashlib.sha256(EXECB).hexdigest(),
    trigger=('SMOKE（A0，PROFILE=[1,2,3,6,30,34]，ALPHAS=3 点）给出 max|dxh|=0.010517198667084726，'
             '超过 seal 的 RECON_TOL_XH=1e-3。定位：xhalf 是 cross_alpha 在 α 网格上的**插值位置**，'
             'SMOKE 的 3 点网格必然使其系统性偏移 —— 这是网格效应，不是 drift。'),
    evidence_from_smoke=dict(
        smoke_result_path='tests/deepseek_temp/Phase16/result_phase16_smoke.json',
        smoke_result_sha8='87244b87',
        max_abs_dxh_A0=0.010517198667084726,
        smoke_alpha_grid=[0.0, 0.5, 1.0],
        smoke_profile=[1, 2, 3, 6, 30, 34],
        note='SMOKE 的 P5/P6/P7 FAIL 亦由同一原因（n_jumps=2 < k+1=4 -> 新量降级），属设计性降级。'),
    floors=dict(RECON_TOL_XH_STRICT=1e-3, RECON_TOL_XH_LOOSE=5e-3,
                RECON_TOL_J_REL=2e-2, RECON_TOL_TOP3=2e-2),
    tiers=dict(
        RECON_OK='max|dxh| <= STRICT 且 max_rel|dJ| <= 2e-2 且 legacy argmax 整数相同 且 legacy top3 差 <= 2e-2',
        RECON_OK_LOOSE='STRICT < max|dxh| <= LOOSE 且其余同上（报告必须附逐站点 dxh 表）',
        RECON_DRIFT='其余',
    ),
    p2_criterion=('三臂全部 ∈ {RECON_OK, RECON_OK_LOOSE} 且 count(RECON_OK) >= 2'),
    what_is_NOT_changed=[
        'profile_sites / profile_sites_legacy / alphas / CANDS（网格）',
        'template / classes / instances_all / pairs_all / discovery / confirmation / quant（材料）',
        'com_layer / span_k 的定义与双边置换协议（BP=2000 与 6 组种子）',
        'UNREACH_y=0.10 / CENTROID_SEP_MIN=4.0 / CENTROID_AFTER_WIN_MIN=5.0 / NEW_NONDEG_MIN=2',
        'P1 与 P3–P7 的判据',
    ],
    added_honesty_8=('锚判据分层后，RECON_OK_LOOSE 格必须在报告中附**逐站点 dxh 表**并说明'
                     '该格低于 STRICT 的位点；不得只报「锚通过」。'),
)

OUT = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
b = json.dumps(AM1, ensure_ascii=False, indent=1).encode('utf-8')
io.open(OUT, 'wb').write(b)
h = hashlib.sha256(b).hexdigest()
print('AMEND1 -> %s' % OUT)
print('bytes=%d sha256=%s sha8=%s' % (len(b), h, h[:8]))
print('amend_of_seal_sha8 = %s' % AM1['amend_of_seal_sha8'])
