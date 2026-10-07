# -*- coding: utf-8 -*-
"""生成 Phase 16 execution（冻结）：引用 seal sha256，材料与 Phase 15 逐字节一致，
并新增 anchor_result_sha256 / anchor_values / profile_sites_legacy。"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P16T, 'N2h1a9_design_seal.json')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


SEALB = open(SEALP, 'rb').read()
SEAL = json.loads(SEALB.decode('utf-8'))
AM1P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
AM1B = open(AM1P, 'rb').read()
AM1 = json.loads(AM1B.decode('utf-8'))
assert AM1['amend_of_seal_sha256'] == hashlib.sha256(SEALB).hexdigest(), 'amend1 与 seal 不匹配'
EX15 = json.load(io.open(os.path.join(P15T, 'execution_phase15.json'), encoding='utf-8'))
RES15B = open(os.path.join(P15T, 'result_phase15.json'), 'rb').read()
RES15 = json.loads(RES15B.decode('utf-8'))

ANCH = {}
for a in EX15['arm_order']:
    ANCH[a] = dict(
        FULL_SWAP=RES15['E2_full_swap'][a]['FULL_SWAP'],
        L_star_own=RES15['E3_localize'][a]['L_star_own'],
        xhalf_by_site={str(s): RES15['E4_summary'][a]['xhalf'][i]
                       for i, s in enumerate(RES15['E4_summary'][a]['sites'])},
        J_by_site={str(s): RES15['E4_summary'][a]['J'][i]
                   for i, s in enumerate(RES15['E4_summary'][a]['sites'])},
        legacy_top3_x=RES15['E5_concentration'][a]['top3_x'],
        legacy_top3_j=RES15['E5_concentration'][a]['top3_j'],
        legacy_argmax_w_x=RES15['E5_concentration'][a]['argmax_w_x'],
        legacy_argmax_w_j=RES15['E5_concentration'][a]['argmax_w_j'],
        legacy_null95_x=RES15['E5_concentration'][a]['null_x']['null95'],
        legacy_null95_j=RES15['E5_concentration'][a]['null_j']['null95'],
        legacy_margin_x=RES15['E5_concentration'][a]['margin_x'],
        legacy_margin_j=RES15['E5_concentration'][a]['margin_j'],
    )

EX = dict(
    phase=16,
    line='N2h1-alpha-9',
    revision='v2 ( amend1: 锚判据分层；SMOKE 显示 xhalf 为插值量，硬门不适合 )',
    seal_sha256=hashlib.sha256(SEALB).hexdigest(),
    seal_sha8=hashlib.sha256(SEALB).hexdigest()[:8],
    seal_path='tests/deepseek_temp/Phase16/N2h1a9_design_seal.json',
    amend1_path='tests/deepseek_temp/Phase16/N2h1a9_design_seal_amend1.json',
    amend1_sha256=hashlib.sha256(AM1B).hexdigest(),
    amend1_sha8=hashlib.sha256(AM1B).hexdigest()[:8],
    amend1_kind=AM1['kind'],
    amend1_floors=AM1['floors'],
    amend1_p2_criterion=AM1['p2_criterion'],
    amend1_of_execution_sha256=AM1['amend_of_execution_sha256'],
    amend1_trigger=AM1['trigger'],
    anchor_phase=15,
    anchor_result_path='tests/deepseek_temp/Phase15/result_phase15.json',
    anchor_result_sha256=hashlib.sha256(RES15B).hexdigest(),
    anchor_values=ANCH,

    arm_order=EX15['arm_order'],
    arms=EX15['arms'],
    template=EX15['template'],
    classes=EX15['classes'],
    instances_all=EX15['instances_all'],
    pairs_all=EX15['pairs_all'],
    discovery=EX15['discovery'],
    confirmation=EX15['confirmation'],
    quant=EX15['quant'],
    sup_id=EX15['sup_id'],
    sup_id_semantics=SEAL['sup_id_semantics'],

    profile_sites=SEAL['profile_sites'],
    profile_sites_legacy=SEAL['profile_sites_legacy'],
    alphas=SEAL['alphas'],
    localize=EX15['localize'],
    W=SEAL['concentration']['W'],
    xh_frac=EX15['xh_frac'],
    bootstrap=SEAL['bootstrap'],
    floors=SEAL['floors'],
    inheritance=EX15['inheritance'],
    concentration=SEAL['concentration'],
    reachability=SEAL['reachability'],
    predictions=SEAL['predictions'],
    honesty=SEAL['honesty'],
)

OUT = os.path.join(P16T, 'execution_phase16.json')
b = json.dumps(EX, ensure_ascii=False, indent=1).encode('utf-8')
io.open(OUT, 'wb').write(b)
h = hashlib.sha256(b).hexdigest()
print('EXEC -> %s' % OUT)
print('bytes=%d sha256=%s sha8=%s' % (len(b), h, h[:8]))
print('seal_sha8=%s ; anchor_result_sha8=%s' % (EX['seal_sha8'], EX['anchor_result_sha256'][:8]))
print('profile_sites n=%d ; legacy n=%d ; floors=%s' %
      (len(EX['profile_sites']), len(EX['profile_sites_legacy']), json.dumps(EX['floors'], ensure_ascii=False)))
