# -*- coding: utf-8 -*-
"""
Phase 21 执行冻结生成器：把 seal 的设计**实例化**为运行输入（重述面板/臂/锚/容差），
并在运行时用 seal_sha256 做 drift 断言（实现与 seal 逐字一致）。

用法：python tests/deepseek/Phase21/gen_exec_phase21.py
产出：tests/deepseek_temp/Phase21/execution_phase21.json
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
SEALP = os.path.join(P21T, 'N2h1a14_design_seal.json')
P8RESULT = os.path.join(P8T, 'result_phase8.json')
P8EXEC = os.path.join(P8T, 'execution_phase8.json')
EXEC_OUT = os.path.join(P21T, 'execution_phase21.json')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace('/', '\\')


S = json.load(io.open(SEALP, encoding='utf-8'))
assert S['phase'] == 21 and S['kind'] == 'design_seal'

EXEC = dict(
    phase=21,
    line='N2h1-alpha-14',
    kind='execution',
    created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    seal_sha256=sha(SEALP),
    seal_bytes=os.path.getsize(SEALP),
    anchor_p8_result_path=rel(P8RESULT),
    anchor_p8_result_sha256=sha(P8RESULT),
    anchor_p8_exec_path=rel(P8EXEC),
    anchor_p8_exec_sha256=sha(P8EXEC),
    template=S['template'],
    classes=S['classes'],
    instances_all=S['instances_all'],
    pairs_all=S['pairs_all'],
    discovery=S['discovery'],
    confirmation=S['confirmation'],
    quant_nf4=S['quant_nf4'],
    quant_bf16=S['quant_bf16'],
    components=S['components'],
    arms=S['arms'],
    arm_order=S['arm_order'],
    anchors=S['anchors'],
    floors=S['floors'],
    predictions=S['predictions'],
    verdict_tree=S['verdict_tree'],
    V_rand_per_pair=S['V_rand_per_pair'],
    seed=S['seed'],
    target_metric='M1 share_v（向量预算，精确可加）为第一指标；M2 W 为权重实现级；M3 dDonor 为效应侧（第二指标）。',
    note='唯一自变量 = 数值精度（nf4 vs bf16）。面板/U 构造/层/判据逐字继承 P8（U 在 discovery 上估计）；'
         'A0_bf16 必须复现 P8 冻结锚（P8 即 bf16）；A1 为跨模型 holdout（无 P8 锚）。'
         'A2（Qwen3-14B）不参与（bf16 segfault，P19 实测）。',
)

with io.open(EXEC_OUT, 'w', encoding='utf-8', newline='\n') as f:
    f.write(json.dumps(EXEC, ensure_ascii=False, indent=1))
print('WROTE', EXEC_OUT, os.path.getsize(EXEC_OUT), 'B')
print('seal_sha8', sha(SEALP)[:8])
print('p8_result_sha8', sha(P8RESULT)[:8], 'p8_exec_sha8', sha(P8EXEC)[:8])
