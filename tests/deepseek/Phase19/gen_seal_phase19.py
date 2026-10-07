# -*- coding: utf-8 -*-
"""
Phase 19 (N2h1-alpha-12) 预注册 seal 生成器。

题：写入向量谱 w_ell 与其质心 com_V 的**量化口径稳健性**（nf4 vs bf16）。
材料逐字节继承 Phase 17（template/classes/instances/pairs/quant 全部原样搬运）。
锚：Phase 17 result 的 com_V 族（A0=26.1501, A1=26.7037, A2=26.6749），现场读入。

臂集（量化口径为唯一自变量；A2-bf16 实测不可加载 -> 只以 nf4 参与装置门）：
  A0_nf4  qwen3-4b   nf4   （校准臂：必须逐位复现 P17 锚）
  A0_bf16 qwen3-4b   bf16  （核心检验臂）
  A1_nf4  glm4-9b    nf4   （校准臂 2：复现 P17 锚 A1）
  A1_bf16 glm4-9b    bf16  （跨家族 holdout 检验臂；需 CPU offload）
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
os.makedirs(P19T, exist_ok=True)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


EX17P = os.path.join(P17T, 'execution_phase17.json')
R17P = os.path.join(P17T, 'result_phase17.json')
S17P = os.path.join(P17T, 'N2h1a10_design_seal.json')
EX17 = json.load(io.open(EX17P, encoding='utf-8'))
R17 = json.load(io.open(R17P, encoding='utf-8'))

# ---- 探针证据（A0 双口径；A1/A2 可加载性）
PB = json.load(io.open(os.path.join(P19T, '_probe19_A0_both.json'), encoding='utf-8'))
P1B = json.load(io.open(os.path.join(P19T, '_probe19_A1_bf16.json'), encoding='utf-8'))
_A2P = os.path.join(P19T, '_probe19_A2_bf16.json')
P2B = (json.load(io.open(_A2P, encoding='utf-8')) if os.path.exists(_A2P)
       else {'runs': {'bf16': {'load_ok': False, 'error': 'segfault (exit 139) during weight loading'}}})
R_NF4 = PB['runs']['nf4']
R_BF16 = PB['runs']['bf16']
CMP = PB['compare']

# ---- P17 锚（现场读入）
MAP17 = {'A0_nf4': 'A0_calib_qwen3-4b-nf4', 'A1_nf4': 'A1_glm4-9b-nf4', 'A2_nf4': 'A2_qwen3-14b-nf4'}
ANCH = {}
for a19, a17 in MAP17.items():
    E = R17['arms'][a17]['E5_com_V']
    ANCH[a19] = dict(
        com_V=float(E['com_V']), com_V_mlp=float(E['com_V_mlp']), com_V_attn=float(E['com_V_attn']),
        com_V_top1head=float(E['com_V_top1head']), com_V_full=float(E['com_V_full']),
        median_reach=float(E['median_reach']), nb=[int(x) for x in E['neighbourhood']],
        share_mlp_nb=float(E['share_mlp_nb']), share_attn_nb=float(E['share_attn_nb']),
        top1_head_nb=int(E['top1_head_nb']), top1_head_share_nb=float(E['top1_head_share_nb']),
        argmax_w_layer=int(E['argmax_w_layer']), reach=[int(x) for x in E['reach']])
A2_REF = ANCH.pop('A2_nf4')          # A2 只作参考（bf16 不可行，不入判据）

# ---- 材料（逐字继承 P17）
TMPL = EX17['template']
SUPS = list(EX17['classes'])
INST_ALL = EX17['instances_all']
PAIRS_ALL = EX17['pairs_all']
DISC = EX17['discovery']
CONF = EX17['confirmation']
QUANT_NF4 = EX17['quant']

QUANT_BF16 = dict(scheme='bfloat16 (no quantization)',
                  dtype='bfloat16',
                  attn_implementation=QUANT_NF4['attn_implementation'],   # 与 nf4 臂逐字一致
                  device_map=QUANT_NF4['device_map'],
                  max_memory=QUANT_NF4['max_memory'],
                  low_cpu_mem_usage=True,
                  why=('除量化外与 nf4 臂**逐项一致**（同 attn_implementation / device_map / max_memory）。'
                       'A0-bf16 单卡 8.0GB 可载；A1-bf16 18.8GB 需 CPU offload；'
                       'A2-bf16 29.5GB 实测加载期 segfault（见 loadability）。'))

ARMS = {
    'A0_nf4': dict(model='qwen3-4b', dir='qwen3-4b', quant='nf4', offload=False, ckpt_gb=8.04,
                   role='量化口径校准臂 1（必须逐位复现 Phase 17 锚 com_V=%.4f）' % ANCH['A0_nf4']['com_V'],
                   expected=EX17['arms']['A0_calib_qwen3-4b-nf4']['expected'],
                   config_sha8=EX17['arms']['A0_calib_qwen3-4b-nf4']['config_sha8']),
    'A0_bf16': dict(model='qwen3-4b', dir='qwen3-4b', quant='bf16', offload=False, ckpt_gb=8.04,
                    role='核心检验臂（bf16，同模型同尺度；探针已知，非 holdout）',
                    expected=EX17['arms']['A0_calib_qwen3-4b-nf4']['expected'],
                    config_sha8=EX17['arms']['A0_calib_qwen3-4b-nf4']['config_sha8']),
    'A1_nf4': dict(model='glm4-9b-chat-hf', dir='glm4-9b-chat-hf', quant='nf4', offload=False, ckpt_gb=18.8,
                   role='量化口径校准臂 2（必须逐位复现 Phase 17 锚 com_V=%.4f）' % ANCH['A1_nf4']['com_V'],
                   expected=EX17['arms']['A1_glm4-9b-nf4']['expected'],
                   config_sha8=EX17['arms']['A1_glm4-9b-nf4']['config_sha8']),
    'A1_bf16': dict(model='glm4-9b-chat-hf', dir='glm4-9b-chat-hf', quant='bf16', offload=True, ckpt_gb=18.8,
                    role='跨家族 holdout 检验臂（bf16；seal 前未观测其任何研究量）',
                    expected=EX17['arms']['A1_glm4-9b-nf4']['expected'],
                    config_sha8=EX17['arms']['A1_glm4-9b-nf4']['config_sha8']),
}
ARM_ORDER = ['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16']

SEED_BASE = 20261003
SEED_NULL = {'comv_all': SEED_BASE + 71, 'comv_mlp': SEED_BASE + 83}

SEAL = {
    'phase': 19,
    'line': 'N2h1-alpha-12',
    'title': ('写入向量谱 w_ell 与其质心 com_V 的量化口径稳健性：'
              '在同模型同尺度上以 bf16 复算，排除「深端集中」是 nf4 量化地板效应'),
    'kind': 'design_seal',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'supersedes': None,

    'motivation': {
        'gap_1': ('Phase 17 的 `quant.why` **自己写明**：为保持三臂同一数值口径统一改为 nf4，'
                  '并声明「A0 臂专职量化对其结论的影响」。**该检查从未执行**。'
                  'P17 的头条 DEEP_ALL（com_V = %.4f / %.4f / %.4f，全部深端）完全建立在 nf4 读数上。'
                  % (ANCH['A0_nf4']['com_V'], ANCH['A1_nf4']['com_V'], A2_REF['com_V'])),
        'gap_2': ('Phase 18 在同一 nf4 口径上又建了一层：行为预算 b 的质心 com_B，'
                  '其核心结论「com_B 比 com_V 浅 2.5-6.6 层」是**两个 nf4 量之差**。'
                  '若量化口径本身给 w_ell 带来系统位移，则该「差」的绝对值需重新标定。'),
        'gap_3': ('Phase 12 的 **bf16** 未投影残差谱 diff_norms 在深端爆炸（L6 %.1f -> L30 %.1f），'
                  '**方向支持**「深端大」，但对象不同（未投影残差 vs 投影组件和），不能直接外推。'
                  % (24.94, 240.13)),
        'fix': ('① 在同模型同尺度（同 template/classes/instances/pairs/U_l 口径/区间求和质心/REACH 域）上'
                '以 bf16 复算 w_ell 谱与 com_V 族；② 与 nf4 臂做**同 Phase 配对**（不是跨 Phase 比对）；'
                '③ 报告 delta(com_V)、谱秩相关、同位点相对残差；④ 校准臂逐位复现 P17 锚。'),
    },

    'material_source': {
        'from_execution': 'tests\\deepseek_temp\\Phase17\\execution_phase17.json',
        'from_execution_sha256': fsha(EX17P),
        'from_result_sha256': fsha(R17P),
        'from_seal_sha256': fsha(S17P),
        'inherited_keys': ['template', 'classes', 'instances_all', 'pairs_all', 'discovery',
                           'confirmation', 'quant(nf4)', 'profile_sites', 'span_ks',
                           'neighbourhood_width'],
        'note': ('禁止改动词表/实例/配对/模板/nf4 口径；bf16 口径是本 Phase 的**唯一**新自由度，'
                 '且已在 floors 与 honesty 中预注册其风险。'),
    },

    'template': TMPL,
    'classes': SUPS,
    'instances_all': INST_ALL,
    'pairs_all': PAIRS_ALL,
    'discovery': DISC,
    'confirmation': CONF,
    'quant_nf4': QUANT_NF4,
    'quant_bf16': QUANT_BF16,
    'profile_sites': EX17['profile_sites'],
    'sup_id_semantics': ('逐臂由该臂 tokenizer 现场解析类别词 id，并断言 6/6 恰为单 token 且 decode 可逆（F1b）；'
                         '禁止跨词表沿用任何硬编码 id（Phase 15 amend1 事故）。'
                         '注意 bf16 与 nf4 共用同一 tokenizer => 两臂的 sup_id 必须相同。'),

    'invariants': {
        'what_changes': 'ONLY the numeric precision of the forward pass (nf4 4-bit vs bf16).',
        'what_is_frozen': [
            'template / classes / instances_all / pairs_all / discovery / confirmation',
            'U_l 口径：全 41 实例按类平均 -> 类别质心矩阵 -> SVD 取 rank = n_classes - 1 = 5',
            '质量定义：w_l = mean over discovery pairs of ||P_{U_l}(Delta_inc_l)||',
            '质心定义：REACH 的相邻位点**区间求和** + 中点（com_of_mass，逐字继承 P17）',
            'REACH 域：本 Phase 采用该臂在 P17 冻结的 REACH（跨口径保持不变，用于配对）',
            '邻域宽度 NBW = 2；bootstrap BP 与 seeds 逐字继承',
        ],
    },

    'measures': {
        'E2_fidelity': 'arch = ||(h_{l+1}-h_l) - (o_proj(o_l) + m_l)|| / ||h_{l+1}-h_l|| ; blk = ||sum_h head_block - o_proj(v)|| / ||o_proj(v)||',
        'E3_U': 'U_l（rank=5）',
        'E4_profile': 'w_all / w_attn / w_mlp / w_top1（投影范数，逐对平均）+ **d_norm**（未投影 ||Delta_inc_l||，对齐 Phase 12 diff_norms 口径，描述性）',
        'E5_com_V': 'com_V(all/mlp/attn/top1head/full) + median(REACH) + nb + share_mlp_nb / share_attn_nb / top1_head_share_nb + argmax_w_layer',
        'E6_null': 'com_V 的置换零假设（保留质量多重集、随机重排到 REACH 位点，BP 次，双侧 5/95）',
        'E7_anchor': '（仅 nf4 臂）从 P17 result 现场读入锚并与本臂实测**逐位**断言',
        'E8_pair': '（MERGE 期）同 Phase 配对：delta(com_V) = |com_V(nf4) - com_V(bf16)|；spearman(w_nf4, w_bf16)；median/p90 相对残差；argmax 是否同位；share_mlp_nb 是否同侧',
    },

    'floors': {
        'P19_FID_ARCH': 3.0e-2,
        'P19_FID_BLK': 1.0e-2,
        'CALIB_TOL_COMV': 1.0e-3,
        'QUANT_TOL_COMV': 2.0,
        'RHO_SHAPE_MIN': 0.80,
        'MLP_DOM_MIN': 0.50,
        'DEEP_MEDIAN': 0.5,
        'NULL_ALPHA': 0.05,
    },

    'predictions': {
        'P1': dict(name='装置门 —— 两个 nf4 校准臂逐位复现 P17 锚',
                   claim=('A0_nf4 复现 com_V=%.4f / mlp=%.4f / attn=%.4f / median(REACH)=%.1f / nb=%s / '
                          'argmax_w=L%s 且差值 <= CALIB_TOL_COMV；A1_nf4 同样复现 com_V=%.4f。'
                          '保真度门 arch <= %.1e 且 blk <= %.1e（全臂）。'
                          % (ANCH['A0_nf4']['com_V'], ANCH['A0_nf4']['com_V_mlp'], ANCH['A0_nf4']['com_V_attn'],
                             ANCH['A0_nf4']['median_reach'], ANCH['A0_nf4']['nb'], ANCH['A0_nf4']['argmax_w_layer'],
                             ANCH['A1_nf4']['com_V'], 3.0e-2, 1.0e-2)),
                   rationale='跨 Phase 复用 nf4 读数；若装置漂移则「量化敏感度」的配对基线不成立。',
                   falsified_if='任一校准臂的 com_V 与 P17 锚差 > 1e-3，或保真度门超限。'),
        'P2': dict(name='核心（A0，探针已知）—— 量化口径不移动质心',
                   claim=('A0 的 |com_V(bf16) - com_V(nf4)| <= QUANT_TOL_COMV=%.1f 层，'
                          '且 bf16 的 com_V >= median(REACH)（仍深端）。' % 2.0),
                   rationale=('探针实测 delta=%.4f 层（%.2f%%）、spearman(w_nf4,w_bf16)=%.4f、'
                              'argmax 同为 L%s => 预注册该结果在正式运行中复现。'
                              % (CMP['com_V_delta'], 100.0 * CMP['com_V_delta'] / CMP['com_V_nf4'],
                                 CMP['spearman_w'], CMP['argmax_bf16'])),
                   falsified_if='正式运行中 A0 的 delta > %.1f 层，或其 bf16 com_V < median(REACH)。' % 2.0),
        'P3': dict(name='holdout 主预测 —— 跨家族（A1）的量化稳健性',
                   claim='A1 的 |com_V(bf16) - com_V(nf4)| <= QUANT_TOL_COMV=2.0 层。',
                   rationale=('A0 显示量化敏感度极小；若这是**数值误差的共性**（4-bit 反量化的系统偏差'
                              '远小于层间质心差），则 GLM 家族同判。A1 的任何研究量在 seal 前未观测。'),
                   falsified_if='A1 的 delta > 2.0 层。'),
        'P4': dict(name='holdout 主预测 —— A1 的 bf16 仍是「组件过半 + 深端」',
                   claim='A1 的 bf16 上 share_mlp_nb > 0.50 且 com_V >= median(REACH)。',
                   rationale='若 nf4 的 MLP 主导份额与深端位置都是量化假象，则 bf16 会推翻它。',
                   falsified_if='A1 的 bf16 share_mlp_nb <= 0.50 或 com_V < median(REACH)。'),
        'P5': dict(name='谱形状稳健（描述性）',
                   claim='每对配对的 spearman(w_nf4, w_bf16) >= RHO_SHAPE_MIN=0.80，并报告 median/p90 相对残差。',
                   rationale='谱形状（哪一层大）若稳健，则 w_ell 作为「位置量」的口径是可信的。',
                   falsified_if='任一对配对的 spearman < 0.80。'),
    },

    'verdict_tree': {
        'Q0': 'device/apparatus gate：Q0_device + F1(T=2) + F1b(tokenizer 单 token) + determinism + hook 效应',
        'Q1': 'fidelity gate：arch / blk 与 floors 比较 -> FID_PASS / FID_FAIL',
        'Q2': 'calibration gate（nf4 臂）：com_V 与 P17 锚差 <= CALIB_TOL_COMV -> CALIB_OK / CALIB_DRIFT',
        'Q3': 'quant sensitivity（核心）：每对 |com_V(nf4) - com_V(bf16)| 与 QUANT_TOL_COMV -> '
              'QUANT_STABLE / QUANT_SENSITIVE；联合 QUANT_STABLE_ALL / PARTIAL / SENSITIVE',
        'Q4': 'spectrum：spearman(w_nf4,w_bf16) 与 RHO_SHAPE_MIN -> SPECTRUM_CONSISTENT / DISTORTED',
        'Q5': 'attribution：bf16 的 share_mlp_nb 与 MLP_DOM_MIN -> MLP_DOM_RETAINED / MLP_DOM_LOST',
        'Q6': 'deep：bf16 的 com_V 与 median(REACH) -> DEEP_RETAINED / DEEP_LOST',
        'Q7': 'null：置换零假设（all / mlp）在 nf4 与 bf16 上各报，tail 一致性',
    },

    'honesty': {
        'H1': ('探针在 seal 前只在 **A0** 上计算了研究量（nf4 与 bf16 都算，因为 A0 是校准/装置臂，'
               '与 P17 的 A0 角色一致）。**A1 的 bf16 研究量在 seal 前未被观测** —— 对 A1 只做了'
               'loadonly 探针（加载 + 一次前向 + 记内存），未计算任何 w_ell / com_V / share。'),
        'H2': ('bf16 与 nf4 的差异含**两源**：量化误差 + 线性层 kernel 路径（Linear4bit 反量化 vs bf16 Linear）。'
               '同 attn_implementation="eager" 已控注意力 kernel，但反量化路径无法消除 => '
               'delta 不等于「纯量化误差」，只能说「量化口径的整体影响」。'),
        'H3': 'w_ell 是**激活级**分解（hook o_proj 输入与 MLP 输出），**不是权重级实现证明**（承 N2h1-alpha-1 挂账）。',
        'H4': 'U_l 由全 41 实例估计 => 轻微选择性泄漏；本 Phase 的「泛化」只指**跨量化口径**与**跨臂**，不指跨词表/跨实例。',
        'H5': '保真度门容差来自 nf4 实测地板（P17 探针 1.62e-2 / 3.59e-3），不是理论误差界。',
        'H6': ('**A1_bf16 需 CPU offload**（18.8GB > 14GiB GPU 上限）=> 引入「分片执行」第三源。'
               '该臂的读数按预注册仍入 P3/P4 硬门，但若出现超容差，必须先在 A0（无 offload）上排除'
               '分片效应后才可归因于量化。'),
        'H7': 'QUANT_TOL_COMV=2.0 层由**邻域宽度 nb=±2** 给出（「仍落在同一读位槽」的朴素定义），非迁就观测。',
        'H8': ('**A2-bf16 实测不可加载**（29.5GB，加载 19%% 时 segfault，与 P17 的 nf4 切换理由一致）'
               '=> A2 只以 nf4 参与装置门，不参与量化敏感度对照。这是本 Phase 的**覆盖限界**：'
               '结论的跨口径稳健性只在 qwen3-4b 与 glm4-9b 两个模型上验证。'),
        'H9': '本 Phase **不重测** P17 的行为量（J / com_layer），也不重测 P18 的行为预算 b；只回答「w_ell 谱与 com_V 是否量化稳健」。',
    },

    'probe_evidence': {
        'arm_A0_nf4': dict(com_V=R_NF4['com_V'], com_V_mlp=R_NF4['com_V_mlp'], com_V_attn=R_NF4['com_V_attn'],
                           median_reach=R_NF4['median_reach'], nb=R_NF4['nb'],
                           share_mlp_nb=R_NF4['share_mlp_nb'], argmax_w_layer=R_NF4['argmax_w_layer'],
                           n_pairs=R_NF4['n_pairs'], n_inst=R_NF4['n_inst'], rank=R_NF4['rank'],
                           load_s=R_NF4['load_s'], device_hist=R_NF4['device_hist']),
        'arm_A0_bf16': dict(com_V=R_BF16['com_V'], com_V_mlp=R_BF16['com_V_mlp'], com_V_attn=R_BF16['com_V_attn'],
                            median_reach=R_BF16['median_reach'], nb=R_BF16['nb'],
                            share_mlp_nb=R_BF16['share_mlp_nb'], argmax_w_layer=R_BF16['argmax_w_layer'],
                            n_pairs=R_BF16['n_pairs'], n_inst=R_BF16['n_inst'], rank=R_BF16['rank'],
                            load_s=R_BF16['load_s'], device_hist=R_BF16['device_hist']),
        'pair_delta': dict(com_V_delta=CMP['com_V_delta'], rel_delta=100.0 * CMP['com_V_delta'] / CMP['com_V_nf4'],
                           spearman_w=CMP['spearman_w'], median_rel_resid=CMP['median_rel_resid'],
                           p90_rel_resid=CMP['p90_rel_resid'],
                           argmax_nf4=CMP['argmax_nf4'], argmax_bf16=CMP['argmax_bf16'],
                           share_mlp_nb_nf4=CMP['share_mlp_nb_nf4'], share_mlp_nb_bf16=CMP['share_mlp_nb_bf16']),
        'note': ('以上为 A0 探针的全部读数（nf4 与 bf16），txt 原件 _probe19_A0_both.txt；'
                 'A0_nf4 与 P17 冻结锚**逐位一致**（见 calibration）。'),
        'not_computed_before_seal': ['A1 的任何 w_ell / com_V / share 量',
                                     'A1 的谱秩相关与相对残差',
                                     '置换零假设的 A1 分位', 'A2 的任何研究量'],
    },

    'loadability': {
        'A0_bf16': dict(ok=True, ckpt_gb=8.04, device_hist=R_BF16['device_hist'], load_s=R_BF16['load_s'],
                        ram_avail_after_gb=(R_BF16['ram_after'][1] if 'ram_after' in R_BF16 else None),
                        gpu_free_after_gb=(R_BF16['gpu_after'][0] if 'gpu_after' in R_BF16 else None)),
        'A1_bf16': dict(ok=bool(P1B['runs']['bf16'].get('load_ok')), ckpt_gb=18.8,
                        device_hist=P1B['runs']['bf16'].get('device_hist'),
                        load_s=P1B['runs']['bf16'].get('load_s'),
                        fwd_ok=P1B['runs']['bf16'].get('fwd_ok'),
                        gpu_free_after_gb=(P1B['runs']['bf16'].get('gpu_after') or [None])[0],
                        ram_avail_after_gb=(P1B['runs']['bf16'].get('ram_after') or [None, None])[1],
                        note='需 CPU offload（device_hist 含 meta 占位）。'),
        'A2_bf16': dict(ok=False, ckpt_gb=29.54, failure='segfault (exit 139) at ~19% of weight loading',
                        note=('与 P17 quant.why 记录一致：29.5GB bf16 需 CPU 侧 >15GB，本机 RAM 不足以在'
                              '不换页的前提下装载 => 加载期硬崩（无 Python 栈）。')),
    },

    'why_not_a_HARKing_violation': (
        '探针在 seal 前只测了 A0（其读数全部抄录在 probe_evidence 里，读者可逐项核对），'
        '并对 A1/A2 只做了 **loadonly**（加载 + 一次前向 + 内存记录，未计算任何研究量）。'
        '本 Phase 的 holdout 判据落在 **A1 的 bf16**（P3/P4），其任何研究量在冻结前未被观测。'
        '阈值全部由机制无关理由给出：P19_FID_* 沿用 P17 的实测地板；CALIB_TOL_COMV=1e-3 是数值恒等；'
        'QUANT_TOL_COMV=2.0 是邻域宽度；RHO_SHAPE_MIN=0.80 是「强单调」的朴素下限；'
        'MLP_DOM_MIN=0.50 是「过半」。没有任何阈值是为迁就观测值而设。'),

    'may_falsify_the_whole_line': (
        '若 P2/P3 FAIL（bf16 的 com_V 与 nf4 差 > 2 层，或 bf16 塌回中段），则「向量写入质量深端集中」'
        '在 nf4 下是一个**量化口径效应** => P17 的 DEEP_ALL 必须整体撤回，P18 的 com_B vs com_V 的 '
        'gap（也是纯 nf4 量之差）必须重新标定，且 P8-P18 所有基于 nf4 的「深端」表述都要加同一限界。'
        '反之，若 QUANT_STABLE_ALL 且 bf16 仍深端，则 N2h1-alpha 线的 nf4 结论获得跨量化口径的稳健性支持。'),

    'anchor_values': ANCH,
    'anchor_reference_only': {'A2_nf4': A2_REF},
    'arm_order': ARM_ORDER,
    'arms': ARMS,
    'bootstrap': {'BP': 2000, 'seed': SEED_BASE, 'seeds': dict(SEED_NULL),
                  'note': 'com_V 置换零假设两条线各一独立种子（comv_all / comv_mlp）；脚本从 exec 读种子。'},
    'neighbourhood_width': 2,
}

OUT = os.path.join(P19T, 'N2h1a12_design_seal.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(SEAL, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('seal_sha8 =', hashlib.sha256(b).hexdigest()[:8])
print('material: exec17=%s result17=%s seal17=%s' % (fsha(EX17P)[:8], fsha(R17P)[:8], fsha(S17P)[:8]))
print('arms:', ARM_ORDER)
print('anchor com_V:', {a: round(ANCH[a]['com_V'], 4) for a in ANCH if a in ('A0_nf4', 'A1_nf4')},
      '| A2(ref):', round(A2_REF['com_V'], 4))
print('probe delta=%.4f  rho=%.4f  resid_med=%.4f' % (CMP['com_V_delta'], CMP['spearman_w'], CMP['median_rel_resid']))
print('loadability: A0=%s A1=%s A2=%s' % (True, P1B['runs']['bf16'].get('load_ok'), False))
