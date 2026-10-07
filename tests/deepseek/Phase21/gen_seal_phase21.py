# -*- coding: utf-8 -*-
"""
Phase 21 / N2h1-alpha-14 设计 seal 生成器（观测前冻结）。
=========================================================
主题：**组件级向量预算 `share_v` 与权重实现级容量的跨精度稳健性（nf4 vs bf16）**。

动机（P20 §11 写死的死线）：
  P8（N2h1-alpha）在 **bf16** 下算出 L6 写入算子的组件预算分解
      `share_v`（向量预算，第一指标）：MLP = 0.4717 / 最大单头 head14 = 0.0742；
      `W`（权重实现级容量）：max_head_share = 0.1329 / MLP-vs-attn 比值。
  而 P16/P17/P18（把该预算推广到逐层、并加上行为侧与写入窗剖面）**全部在 nf4 口径下**。
  P19 补了**向量侧**的跨精度（`w_l` 谱 / `com_V`），P20 补了**行为侧**与**剖面侧**的跨精度。
  ⇒ 仍然缺的是 **P8 那条线上的两个量本身**：
      (M1) 组件级向量预算 `share_v`（第一指标，**精确可加**）；
      (M2) 权重实现级容量 `W`（纯权重算术，无前向）。
  若这两个量在 nf4↔bf16 下不同号/不同幅，则「分布式搬运 / MLP 是最大单一写入方」这条
  主线结论就带有**未声明的精度依赖**。

唯一自变量 = 数值精度（bitsandbytes nf4 (4bit) ↔ torch.bfloat16）。其余逐字继承。

用法：python tests/deepseek/Phase21/gen_seal_phase21.py
产出：tests/deepseek_temp/Phase21/N2h1a14_design_seal.json
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
SEAL_OUT = os.path.join(P21T, 'N2h1a14_design_seal.json')

P20EXEC = os.path.join(P20T, 'execution_phase20.json')
P8RESULT = os.path.join(P8T, 'result_phase8.json')
P8EXEC = os.path.join(P8T, 'execution_phase8.json')


def sha256_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace('/', '\\')


os.makedirs(P21T, exist_ok=True)

EX20 = json.load(io.open(P20EXEC, encoding='utf-8'))
R8 = json.load(io.open(P8RESULT, encoding='utf-8'))

# --- 臂：逐字继承 P20 的四臂（模型/目录/量化/offload/期望维度），只改 role 与 primary 层
ARMS = {}
for aid in ['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16']:
    a = EX20['arms'][aid]
    ARMS[aid] = dict(
        model=a['model'], dir=a['dir'], quant=a['quant'], offload=bool(a['offload']),
        expected=a['expected'], config_sha256=a['config_sha256'],
    )
ARMS['A0_nf4']['role'] = '校准臂：nf4 口径（P8 锚为 bf16 ⇒ 本臂是**新精度**）'
ARMS['A0_bf16']['role'] = '装置校准臂：bf16 口径，必须复现 P8 冻结锚（P8 即 bf16）'
ARMS['A1_nf4']['role'] = '跨家族 holdout（nf4）'
ARMS['A1_bf16']['role'] = '跨家族 holdout（bf16，需 CPU offload）'

# primary 层（= 各臂写入窗 / 相邻最大增量层，取自 P16 冻结 L*_own）
PRIMARY = {'A0_nf4': 6, 'A0_bf16': 6, 'A1_nf4': 3, 'A1_bf16': 3}
L_STAR = {'A0_nf4': 6, 'A0_bf16': 6, 'A1_nf4': 3, 'A1_bf16': 3}
for aid, v in PRIMARY.items():
    ARMS[aid]['primary_layer'] = v
    ARMS[aid]['L_star_own'] = L_STAR[aid]

# --- 冻结锚：P8 的 bf16 锚（仅 qwen3-4b；A1 无 P8 锚 ⇒ 纯 holdout）
A8 = R8['amend1']
W8 = R8['W']
ANCHOR_P8 = dict(
    model='qwen3-4b',
    result_sha256=sha256_file(P8RESULT),
    exec_sha256=sha256_file(P8EXEC),
    primary_layer=int(R8['layers']['primary']),
    share_v_mlp=float(A8['share_v']['mlp']),
    max_head_share_v=float(A8['max_head_share_v']),
    argmax_head_v=str(A8['argmax_head_v']),
    max_head_share_eff=float(A8['max_head_share_eff']),
    I_nl=float(A8['I_nl']),
    loo_vec_top1=float(A8['loo_vec_top1']),
    vec_budget_mlp=float(A8['vec_budget']['mlp']),
    W_max_head_share=float(W8['max_head_share']),
    W_argmax_head=int(W8['argmax_head']),
    W_mlp_share_vs_attn=float(W8['mlp_share_vs_attn']),
    T_diff6_dDonor=float(R8['T']['diff6']['dDonor']),
    T_diff5_dDonor=float(R8['T']['diff5']['dDonor']),
    T_attn_all_dDonor=float(R8['T']['attn_all']['dDonor']),
    T_mlp_dDonor=float(R8['T']['mlp']['dDonor']),
    verdict=str(R8['verdict']),
    gates=dict(R8['gates']),
)

SEAL = dict(
    phase=21,
    line='N2h1-alpha-14',
    kind='design_seal',
    created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    title='组件级向量预算 share_v 与权重实现级容量的跨精度稳健性（nf4 vs bf16）',
    motivation=dict(
        gap_1='P8 的组件预算分解（share_v 与 W 容量）在 **bf16** 下算出；'
              'P16/P17/P18 把同一预算推广到逐层/行为侧/剖面侧时**全在 nf4 口径**。',
        gap_2='P19 只补了**向量侧**（w_l 谱 / com_V）的跨精度；P20 只补了**行为侧**（b_c,l）与'
              '**剖面侧**（com_layer）。P8 线上的**两个原始量本身**从未在另一精度下复算。',
        gap_3='P8 的第一指标 `share_v(mlp) = 0.47166` 距判据阈值 **0.50 仅 0.0283** ⇒ '
              '「分布式（MLP 未过半）」这条结论对精度**高度敏感**，必须显式检验。',
        fix='把 P8 的 M1（向量预算 share_v）与 M2（权重容量 W）在**同一装置**下对 {nf4, bf16} × '
            '{qwen3-4b, glm4-9b} 四臂复算；A0_bf16 必须复现 P8 冻结锚（装置校准）。',
        falsifiers='若 nf4 下 `share_v(mlp) > 0.50` 或 `max_head_share_v > 0.30`（任一臂），'
                   '则 P8 的「G1 分布式搬运」带有未声明的精度依赖，须降级为「nf4 口径下成立」。',
    ),
    material_source=dict(
        p8_result=dict(path=rel(P8RESULT), sha256=sha256_file(P8RESULT)),
        p8_execution=dict(path=rel(P8EXEC), sha256=sha256_file(P8EXEC)),
        p20_execution=dict(path=rel(P20EXEC), sha256=sha256_file(P20EXEC)),
    ),
    lineage_precision_gap=dict(
        P8='bf16（dtype=torch.bfloat16, .to(cuda), attn=eager）',
        P9_to_P14='bf16（P9 的 D1@a=1 = 10.574739583333335 逐位等于 P8 T[diff6]）',
        P15_to_P20='nf4（bitsandbytes 4bit；P19/P20 另加 bf16 对照臂）',
        consequence='P8 的 share_v/W 与 P17 的 w_l/com_V 属**不同精度谱系**；本 Phase 在同一装置内闭合该缺口。',
    ),
    invariants=dict(
        what_changes='数值精度：bitsandbytes nf4 (4bit, double-quant, compute=bf16) ↔ torch.bfloat16。',
        what_is_frozen='模板 `%s是一种` / 6 类词 / 41 实例 / 24 discovery 配对 / 17 confirmation 配对 / '
                       'U_l 在 **discovery 上估计**（秩 = n_classes-1 = 5，逐字继承 P8）/ 位点 = 各臂写入窗 '
                       'L*_own（A0=L6、A1=L3）/ 判据阈值（0.30 / 0.50）/ 容差。',
    ),
    measures=dict(
        E0='装置自检：确定性双前向 diff、hook 生效、维度门（L/hid/heads/head_dim/o_proj_in）。',
        E1='capture 41 实例：hidden_states 全层 + o_proj 输入（pre-hook）+ MLP 输出（hook）@写入窗。',
        E2='U_l：discovery 24 实例按类平均 → 类别质心差 SVD → 取前 n_classes-1 = 5 个方向（P8 口径）。',
        E3='M1 向量预算：V_c = mean_disc ||P_U(delta_c)||，c in {32 头, MLP}；share_v = V_c / sum_c V_c。',
        E4='M1 派生：max_head_share_v / argmax_head_v / share_v(mlp) / loo_vec_top1 = 1 - max_head_share_v。',
        E5='M2 权重容量：per_head = ||U @ W_o[:, h*HD:(h+1)*HD]||^2；mlp_cap = ||U @ W_down||^2；'
           'W.max_head_share = max(per_head)/sum；W.argmax_head；W.mlp_share_vs_attn = mlp_cap/sum。',
        E6='M3 效应侧（第二指标，非第一）：T[nm].dDonor = mean_disc [score_of(patch) - BASE.sd0]，'
           'nm in {32 头, mlp, diff5, attn_all, diff6}；I_nl = |dDonor(diff6)| / sum_c |dDonor(c)|。',
        E7='M3 派生：max_head_share_eff（效率份额最大单头）/ 地板 V_rand(5/对) 与 M_mismatch(1/对)。',
        E8='跨精度配对：同一模型的 nf4 与 bf16 臂逐量配对（Δshare_v、Δmax_head_share_v、'
           'spearman(share_v_nf4, share_v_bf16)、argmax 不变）。',
        E9='确认集（n=17）同带复核。',
        E10='装置校准：A0_bf16 对 P8 冻结锚逐量比对（容差 CALIB_TOL）。',
    ),
    floors=dict(
        P21_FID_ARCH=0.03,
        P21_FID_BLK=0.01,
        CALIB_TOL_SHARE_V=0.0001,
        CALIB_TOL_W=0.0001,
        QUANT_TOL_SHARE_V=0.05,
        QUANT_TOL_MAXHEAD_V=0.05,
        QUANT_TOL_W=0.05,
        SPEARMAN_MIN=0.80,
        G1_MAXHEAD_V=0.30,
        G1_MLP_SHARE_V=0.50,
        FLOOR_FRAC=0.10,
        NULL_ALPHA=0.05,
    ),
    predictions=[
        'P1 装置校准：A0_bf16 复现 P8 冻结锚（share_v(mlp) / max_head_share_v / argmax_head_v / I_nl / W.max_head_share），容差 1e-4。',
        'P2 share_v(mlp) 跨精度保持：|Δ(nf4-bf16)| <= 0.05（两模型）。',
        'P3 max_head_share_v 跨精度保持：|Δ| <= 0.05（两模型）。',
        'P4 argmax_head_v 跨精度不变（同一头号，两模型）。',
        'P5 G1 核心门（max_head_share_v <= 0.30 且 share_v(mlp) <= 0.50）在两精度两模型**都**成立。',
        'P6 W 权重容量跨精度保持：|Δmax_head_share| <= 0.05 且 argmax_head 不变。',
        'P7 32+1 维 share_v 的秩稳定：spearman(share_v_nf4, share_v_bf16) >= 0.80（两模型）。',
        'P8 确认集（n=17）同带：share_v(mlp) 与 max_head_share_v 的 G1 落判与 discovery 一致。',
        'P9 地板：V_rand 与 M_mismatch 的 |dDonor| < 10% max|comp dDonor|（仅 M3 有效应侧时）。',
    ],
    verdict_tree=dict(
        Q0='四臂 Q0_device 合格且 F4/F5 维度门通过。',
        Q1='保真度：arch 保真 <= 0.03 且 blk 保真 <= 0.01。',
        Q2='装置校准：A0_bf16 复现 P8 冻结锚（<= 1e-4）。',
        Q3='share_v(mlp) 跨精度稳健 <= 0.05（两模型）。',
        Q4='max_head_share_v 跨精度稳健 <= 0.05（两模型）。',
        Q5='argmax_head_v 跨精度不变（两模型）。',
        Q6='G1 核心门在 nf4 与 bf16 两精度**都**成立（两模型）。',
        Q7='W 容量跨精度稳健且 argmax_head 不变。',
        Q8='share_v 谱秩相关 >= 0.80（两模型）。',
        Q9='确认集（n=17）同带。',
        Q10='地板 V_rand / M_mismatch 合格。',
    ),
    honesty=[
        'A2（Qwen3-14B）不参与：bf16 腿在 29.5GB 加载至 ~19% 处 segfault（P19 实测）⇒ 覆盖只在两模型。',
        'A1_bf16 需 CPU offload ⇒ 含「分片执行」第三源（与量化、反量化 kernel 并列）。',
        'A1 无 P8 冻结锚（P8 只跑 qwen3-4b）⇒ A1 是纯跨模型 holdout，其「校准」只能靠 P16 的 L*_own。',
        'A1 的写入窗层 = L3（P16 L*_own=3），与 A0 的 L6 不同 ⇒ 跨模型比较是「各臂自身写入窗」之比。',
        'M1（share_v）是**向量预算**（精确可加）；M3（dDonor）是**效应份额**（不可加）——两者不得混用（P8 教训）。',
        '本 Phase 只回答「P8 线上的两个量是否精度稳健」；**不**把 P8 的 G1 升格为跨模型普适结论。',
        '「n 个类别的类别质心差 SVD 秩 = n_classes-1 = 5」是 U 的构造定义；U 只证「可写」，不证「被读」。',
    ],
    why_not_a_HARKing_violation='全部判据（阈值 0.30/0.50、容差 0.05/1e-4、秩相关下限 0.80）与'
                                '面板/域/层/U 构造均在**观测前**冻结于本 seal；predictions 逐条先行登记。',
    may_falsify_the_whole_line='若 share_v(mlp) 在 nf4 下越过 0.50，则「分布式搬运」须降级为 nf4 口径结论。',
    arms=ARMS,
    arm_order=['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16'],
    anchors=dict(A0_p8_bf16=ANCHOR_P8),
    template=EX20['template'],
    classes=list(EX20['classes']),
    instances_all=list(EX20['instances_all']),
    pairs_all=list(EX20['pairs_all']),
    discovery=list(EX20['discovery']),
    confirmation=list(EX20['confirmation']),
    quant_nf4=dict(EX20['quant_nf4']),
    quant_bf16=dict(EX20['quant_bf16']),
    components=[('head%d' % h) for h in range(32)] + ['mlp'],
    seed=20261001,
    V_rand_per_pair=5,
    next_step='Phase 22 = 若 G1 精度稳健则把「分布式」写进拼图；否则降级并追查 MLP 份额的精度敏感来源。',
)

body = json.dumps(SEAL, ensure_ascii=False, indent=1)
with io.open(SEAL_OUT, 'w', encoding='utf-8', newline='\n') as f:
    f.write(body)

b = open(SEAL_OUT, 'rb').read()
print('WROTE', SEAL_OUT)
print('bytes', len(b), 'sha256', hashlib.sha256(b).hexdigest())
print('sha8', hashlib.sha256(b).hexdigest()[:8])
print('predictions', len(SEAL['predictions']), 'arms', list(SEAL['arms'].keys()))
print('P8 anchor share_v(mlp)=%.6f max_head_share_v=%.6f argmax=%s' %
      (ANCHOR_P8['share_v_mlp'], ANCHOR_P8['max_head_share_v'], ANCHOR_P8['argmax_head_v']))
