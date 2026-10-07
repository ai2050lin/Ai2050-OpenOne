# -*- coding: utf-8 -*-
"""
Phase 15 execution 冻结生成器。
把运行所需的全部参数写入 execution_phase15.json，并把 seal 的 sha256 钉死（主脚本启动时校验）。
"""
import io
import os
import json
import hashlib
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
SEALP = os.path.join(P15, 'N2h1a8_design_seal.json')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


S = json.load(io.open(SEALP, encoding='utf-8'))
AM1P = os.path.join(P15, 'N2h1a8_design_seal_amend1.json')
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
E12 = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12',
                                     'execution_phase12.json'), encoding='utf-8'))

EXEC = dict(
    phase=15,
    revision='v2 ( amend1: sup_id 逐臂解析 )',
    name=S['name'],
    seal_path='tests/deepseek_temp/Phase15/N2h1a8_design_seal.json',
    seal_sha256=sha(SEALP),
    seal_sha8=sha(SEALP)[:8],
    amend1_path='tests/deepseek_temp/Phase15/N2h1a8_design_seal_amend1.json',
    amend1_sha256=sha(AM1P),
    amend1_sha8=sha(AM1P)[:8],
    amend1_kind=AM1['kind'],
    template=S['template'],
    classes=S['panel']['classes'],
    sup_id=S['panel']['sup_id'],
    sup_id_semantics=('qwen 族参考值。**实际口径 = 逐臂由该臂 tokenizer 现场解析**（amend1 / 新断言 F1b）：'
                      '6/6 类别词必须单 token 且 decode 可逆；A0/A2 解析结果与参考值逐位相同，'
                      'A1(glm4-9b) 不同（vocab 151329 vs 151643）。'),
    sup_id_per_arm={k: v['sup_id'] for k, v in AM1['tokenizer_probe'].items()},
    discovery=S['panel']['discovery'],
    confirmation=S['panel']['confirmation'],
    instances_all=S['panel']['instances_all'],
    pairs_all=S['panel']['pairs_all'],
    profile_sites=S['profile_sites'],
    alphas=S['alphas'],
    W=3,
    xh_frac=0.5,
    bootstrap=dict(BP=2000, seed=20261002, scheme='permutation of jumps'),
    quant=S['quantization'],
    localize=S['localize_arm'],
    arms={k: dict(model=v['model'], dir=v['dir'], role=v['role'],
                  expected=v['expected'], config_sha256=v['config_sha256'],
                  config_sha8=v['config_sha8'], ckpt_gb=v['ckpt_gb'])
          for k, v in S['arms'].items()},
    arm_order=S['arm_order'],
    floors=S['floors'],
    decision=S['decision'],
    predictions=[p['id'] for p in S['pre_registered_predictions']],
    inheritance=dict(
        FULL_SWAP_12=S['inheritance_anchors']['inherited_published']['FULL_SWAP_12'],
        XH_12_by_site=S['inheritance_anchors']['inherited_published']['XH_12_by_site'],
        J_swap_12_by_site=S['inheritance_anchors']['inherited_published']['J_swap_12_by_site'],
        recover_12_by_site=S['inheritance_anchors']['inherited_published']['recover_12_by_site'],
        XH_RANGE_12=S['inheritance_anchors']['inherited_published']['XH_RANGE_12'],
        SHARE_X_13=S['inheritance_anchors']['inherited_published']['SHARE_X_13'],
        SHARE_J_13=S['inheritance_anchors']['inherited_published']['SHARE_J_13'],
        MODE_X_13=S['inheritance_anchors']['inherited_published']['MODE_X_13'],
        MODE_J_13=S['inheritance_anchors']['inherited_published']['MODE_J_13'],
    ),
    result_keys=['E0_selfcheck', 'E1_capture', 'E2_full_swap', 'E3_localize', 'E4_profile',
                 'E5_concentration', 'E6_calibration', 'predictions_check', 'verdict',
                 'floors', 'arms_meta', 'extra'],
    smoke_env=('SMOKE=1 且未设 ARMS -> 仅 A0 臂；profile_sites 取前 3；alphas 取 [0.0,0.5,1.0]；'
               'CANDS 取 [4,6,20]；BP=200。SMOKE=1 + ARMS=<id> -> 只跑该臂的粗网格（用于装置门复核）。'),
    frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
)

OUT = os.path.join(P15, 'execution_phase15.json')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(json.dumps(EXEC, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('EXEC -> %s' % OUT)
print('bytes=%d sha256=%s' % (len(b), hashlib.sha256(b).hexdigest()))
print('sha8=%s ; seal_sha8=%s' % (hashlib.sha256(b).hexdigest()[:8], EXEC['seal_sha8']))
print('arm_order=%s' % EXEC['arm_order'])
print('profile_sites=%d alphas=%d W=%d BP=%d' %
      (len(EXEC['profile_sites']), len(EXEC['alphas']), EXEC['W'], EXEC['bootstrap']['BP']))
