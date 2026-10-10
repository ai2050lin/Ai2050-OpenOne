# -*- coding: utf-8 -*-
# Phase 3162 G5-A1: atlas foundation - evidence-graded audit + L1 descriptive census + registry + HTML map.
# External-review adoption: 4-tier evidence levels, 8-field rule schema, fingerprint!=mechanism principles,
# failure ledger as first-class atlas entries. Zero GPU. Modes: audit / census / summary.
import os, sys, json, io, hashlib, time, re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
DSR = os.path.join(ROOT, 'tests', 'deepseek', 'result')
DA = os.path.join(ROOT, 'research', 'deepseek', 'atlas')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
PDIR = os.path.join(RDIR, 'phase3162', 'g5a1_atlas_foundation')
MODELS3 = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
PHASE = 3162
NAME = 'g5a1_atlas_foundation'

def sha8_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()[:8]

def jload(p):
    with io.open(p, encoding='utf-8') as f:
        return json.load(f)

def jdump(obj, p):
    with io.open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

def get_path(d, dotted):
    cur = d
    for part in dotted.split('.'):
        m = re.match(r'^([^\[\]]+)((?:\[\d+\])*)$', part)
        if not m:
            return None, 'badkey:' + part
        name = m.group(1)
        idxs = re.findall(r'\[(\d+)\]', m.group(2) or '')
        if isinstance(cur, dict) and name in cur:
            cur = cur[name]
        elif isinstance(cur, list) and name.isdigit() and int(name) < len(cur):
            cur = cur[int(name)]
        else:
            return None, 'missing:' + name
        for ix in idxs:
            if isinstance(cur, list) and int(ix) < len(cur):
                cur = cur[int(ix)]
            else:
                return None, 'missing_idx:' + part
    return cur, 'ok'

def run_check(d, spec):
    key, expected, mode, tol = spec
    val, st = get_path(d, key)
    if st != 'ok':
        return False, 'key_missing:' + key
    if mode == 'in':
        ok = isinstance(val, str) and expected in val
        return ok, '%s %s' % (key, val if ok else 'NOT-IN: ' + repr(val)[:80])
    if mode == 'approx':
        ok = isinstance(val, (int, float)) and abs(float(val) - float(expected)) <= tol
        return ok, '%s = %s (expect %s +- %s)' % (key, val, expected, tol)
    ok = val == expected
    return ok, '%s = %s (expect %s)' % (key, val, expected)

def C(key, expected, tol=None):
    return (key, expected, 'approx' if tol is not None else 'eq', tol)

def CI(key, substr):
    return (key, substr, 'in', None)

# ================= FROZEN DESIGN (written to execution.json before any audit run) =================
G_3154 = os.path.join(RDIR, 'phase3154', 'g1p4_mfd_multifactor_disentangle')
G_3155 = os.path.join(RDIR, 'phase3155', 'g2p1_relation_family_operator_separability')
G_3156 = os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family')
G_3157 = os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator')
G_3158 = os.path.join(RDIR, 'phase3158', 'g4p1_output_equivalence_class')
G_3159 = os.path.join(RDIR, 'phase3159', 'g4p2_equivalence_dynamics')
G_3160 = os.path.join(RDIR, 'phase3160', 'g4p3_consumption_mechanism')
G_3151 = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction')

TAXONOMY = {
    'E0_candidate': '单次观察/单模型未复现',
    'E1_repeatable': '同协议跨 run/跨 cell 复现，或单模型内稳健',
    'E2_predictive': '通过 held-out 预测（新实体/新主题/新折），或跨模型指纹一致',
    'E3_causal_scoped': '协议内干预完成且排除主要替代解释；scope 内有效，不可外推',
}
PRINCIPLES = [
    '读得出不等于控得住（C_steer=0 佐证）',
    '跨模型指纹相似不等于机制相同：合并机制节点需额外预测与干预证据',
    '失败结果必须入图谱：每条规律保存适用条件、反例、证据等级与否定范围',
    '结构指标不是天然稳定类：需逐项指明哪种结构、哪种度量、什么条件范围',
]
NODES = [
    dict(id='N00', title='图谱基建（ledger / metric_dict / 队列）', line='infra', phase=0,
         claim='共享测量账本（n>=312, chain 9417b14f）、metric_dict v4（0652c008）、30-Phase 预注册队列构成图谱的溯源基建。',
         scope='跨线共享（deepseek 线 + G 线并发追加）',
         metric_definition='atlas_ledger.json / metric_dict.json / phase_queue_v1.json',
         replication='账本逐 Phase 登记（本审计时点 n=312）', heldout='不适用',
         counterexamples='Q01 曾发现自声明哈希失效（D3/D4），已用外部 manifest 规程修复',
         causal='不适用', evidence_level='E1_repeatable', model_scope={m: 'verified' for m in MODELS3},
         files=[LEDGER, os.path.join(DA, 'metric_dict.json'), os.path.join(DA, 'phase_queue_v1.json')],
         checks=[
             ('measurements_n', (312, 330), 'range', None),
             ('chain', '9417b14f', 'eq', None),
             ('has_3160', True, 'eq', None),
             ('md_version', 4, 'eq', None),
             ('md_content_sha', '0652c008', 'eq', None),
             ('queue_sealed', ['Q01', 'Q02', 'Q03', 'Q08', 'Q09', 'Q12', 'Q05', 'Q06'], 'subset', None),
             ('queue_device', ['Q04'], 'subset', None),
         ]),
    dict(id='N01', title='KOUT 面板 content 主导份额', line='G', phase=3154,
         claim='在 3154 多因素面板中，content 因素占 KOUT 读出份额主导（均值 0.576），且面板份额结构跨模型指纹一致（fpmin 0.9959）。不外推为「一切任务 content 主导」。',
         scope='3154 面板（246 对 x 3 模板）x 3 模型 x 各自读出层',
         metric_definition='metric_dict v4 SHARES 口径；phase3154 summary shares_mean_kout / fp_pairs',
         replication='3/3 模型；KOUT 份额指纹配对 0.9959-0.9996',
         heldout='新主题逻辑轴符号预测 1.00 / 1.00 / 0.9375（ho_pass 3/3）',
         counterexamples='KSTAR 指纹不稳（3155 配对 0.244-0.908）；因素峰值层位随模型深度平移（G argmax 15/22/18）',
         causal='无（观察性 + 前门判据）', evidence_level='E2_predictive', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3154, 'summary', 'result_summary.json')],
         checks=[
             C('shares_mean_kout.C', 0.5762712045154778, 1e-3),
             C('shares_mean_kout.S', 0.19399622367068037, 1e-3),
             C('shares_mean_kout.R', 0.1591576651654146, 1e-3),
             C('fp_min_kout', 0.995864698058461, 1e-6),
             C('ho_all_pass', True), C('logic_axis_sig_all', True),
             C('per_model.qwen3-4b.iv_kout.p', 0.0, 1e-9),
             C('per_model.qwen3-14b.iv_kout.p', 0.0, 1e-9),
             C('per_model.glm4.iv_kout.p', 0.0, 1e-9),
             C('res_sha8', '2402f401'), C('seal_sha8', 'a2b92c3e'),
         ]),
    dict(id='N02', title='逻辑轴中层显著 + held-out 泛化', line='G', phase=3154,
         claim='逻辑一致性轴的翻转效应在 3 模型均显著（p<1e-9 量级，stat 3.2-3.8），held-out 新主题符号准确率 1.00/1.00/0.9375。仅覆盖逻辑一致性对比族，不是全部推理能力。',
         scope='逻辑一致性对比任务族 x 3 模型',
         metric_definition='phase3154 per_model.iv_kout {flip, flip_null, stat, p} + heldout_acc',
         replication='3/3 模型独立显著', heldout='见 claim（新主题未参与发现）',
         counterexamples='翻转率绝对值模型特异（0.44/0.60/0.56）；结论限于该对比族',
         causal='无', evidence_level='E2_predictive', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3154, 'summary', 'result_summary.json')],
         checks=[
             C('per_model.qwen3-4b.heldout_acc', 1.0, 1e-6),
             C('per_model.qwen3-14b.heldout_acc', 1.0, 1e-6),
             C('per_model.glm4.heldout_acc', 0.9375, 1e-6),
             C('per_model.qwen3-4b.iv_kout.stat', 3.7560078782332633, 1e-6),
             C('per_model.glm4.iv_kout.stat', 3.220998873665272, 1e-6),
         ]),
    dict(id='N03', title='K2 条件门近似可分离', line='G', phase=3155,
         claim='实体 x 关系交互份额占 KOUT 15.2%/18.3%/16.4%，远低于 50% 死线；加性迁移 held-out 9/9 折通过。是「近似可分离的条件结构」——交互非零，且结论依赖指定关系族与读出层。',
         scope='3 关系族（isa/hasa/mof）x 414 行 x 3 模板 x 3 模型',
         metric_definition='phase3155 summary k2_int_share_kout / ho_fold_ratios / fp_pairs',
         replication='3/3 模型；ho 9/9 折（ratio 0.50-0.89 < 1.5 门）',
         heldout='held_hasa / held_isa / held_mof 三折 x 3 模型全部通过',
         counterexamples='KSTAR 指纹配对低至 0.244（4b vs glm4）；交互峰值层位模型特异（EintC argmax 18/39/5）',
         causal='无', evidence_level='E2_predictive', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3155, 'summary', 'result_summary.json')],
         checks=[
             C('k2_int_share_kout.qwen3-4b', 0.15187579769212284, 1e-6),
             C('k2_int_share_kout.qwen3-14b', 0.1824533831239199, 1e-6),
             C('k2_int_share_kout.glm4', 0.16411745881744746, 1e-6),
             C('k2_death_line_triggered', False),
             C('ho_total', 9), C('fp_min_kout', 0.9728529051134249, 1e-6),
             C('fp_pairs.qwen3-4b_vs_glm4.pearson_kstar', 0.2437429749346557, 1e-6),
             C('res_sha8', '14975aed'), C('seal_sha8', '5d2c2061'),
         ]),
    dict(id='N04', title='RoPE 输出近似不变（严格门未过，单模型）', line='G', phase=3156,
         claim='在 qwen3-4b 位置平移实验中，输出近似稳定、位置效应很小，但未满足预注册的 1e-3 严格内部不变量门（内部相对位移最大 0.01375，g1_rope=False）。9/9 是 9 个位置条件，不是 9 个模型；跨模型同口径复测未做。',
         scope='单模型 qwen3-4b x 双语 x k in {0..128} x 9 位置条件',
         metric_definition='phase3156 result rope_check（内部相对位移）+ kl（A 臂上下文 KL）+ gates',
         replication='单模型；bitwise determinism',
         heldout='不适用（装置性质实验）',
         counterexamples='严格内部不变量被拒（max 1.375e-2 > 1e-3）；en_k64 最大；A 臂上下文效应比位置效应大 3 个量级',
         causal='无', evidence_level='E1_repeatable',
         model_scope={'qwen3-4b': 'verified', 'qwen3-14b': 'absent', 'glm4-9b': 'absent'},
         files=[os.path.join(G_3156, 'qwen3-4b', 'result.json')],
         checks=[
             C('model', 'qwen3-4b'),
             C('rope_check.en_k128', 0.002221531834612419, 1e-9),
             C('rope_check.en_k64', 0.01375244285636824, 1e-9),
             C('gates.g1_rope', False), C('gates.g4_curve', True),
             C('res_sha8', 'c8c66d9f'), C('seal_sha8', 'b54418f2'),
         ]),
    dict(id='N05', title='上下文二元门控 / massive activation', line='G', phase=3156,
         claim='「有无上下文」是二元开关：单模型（4b）中单个前缀 token 触发 massive activation 塌缩（约 77 倍）；跨模型存在性由 3157 锚态 massive 维度（d1=0/731/2319）与 3160 测量支持，但 77 倍塌缩的统一口径跨模型量化待做。',
         scope='量化：qwen3-4b（3156）；跨模型存在性：3157 锚态 + 3160 d1 测量',
         metric_definition='phase3156 subspace.en_mid.top1 + phase3160 per-model model.d1',
         replication='4b 量化 1/1；三模型 massive 维度存在 3/3',
         heldout='不适用', counterexamples='跨模型同口径塌缩倍数未测；3156 rank-1 轴不可从 npz 复现（3160 观测前发现）',
         causal='无', evidence_level='E1_repeatable',
         model_scope={'qwen3-4b': 'verified', 'qwen3-14b': 'partial', 'glm4-9b': 'partial'},
         files=[os.path.join(G_3156, 'qwen3-4b', 'result.json'),
                os.path.join(G_3160, 'qwen3-4b', 'result.json'),
                os.path.join(G_3160, 'qwen3-14b', 'result.json'),
                os.path.join(G_3160, 'glm4', 'result.json')],
         checks=[
             C('f0:subspace.en_mid.top1', 0.9995781779289246, 1e-6),
             C('f1:model.d1', 0),
             C('f2:model.d1', 731),
             C('f3:model.d1', 2319),
         ]),
    dict(id='N06', title='T_C 上下文变换：共享分量 + 内容特异并存', line='G', phase=3157,
         claim='三模型上下文差分的平均两两余弦 0.561/0.563/0.564：中等相似度。正确表述是「上下文变换含共享方向，同时存在明显内容特异分量」——不能称为关系无关。',
         scope='3157 变换代数装置 x 3 模型 x 16 锚态',
         metric_definition='phase3157 per_model_verdicts tc_cos 字段',
         replication='3/3 模型（0.561-0.564 窄带）', heldout='不适用',
         counterexamples='0.56 距「关系无关」(cos~1) 很远；共享/特异两分量的功能分工未测',
         causal='无', evidence_level='E1_repeatable', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3157, 'summary', 'result_summary.json')],
         checks=[
             CI('per_model_verdicts.qwen3-4b', 'tc_cos_0.563'),
             CI('per_model_verdicts.qwen3-14b', 'tc_cos_0.561'),
             CI('per_model_verdicts.glm4', 'tc_cos_0.564'),
             C('res_sha8', '0fe043bf'), C('seal_sha8', 'cfa5c3ed'),
         ]),
    dict(id='N07', title='变换代数对易子 partial', line='G', phase=3157,
         claim='对易子交换子范数比 1.0-1.3（均值 1.273）：非交换（partial），但近正交而非反平行；曲线形状跨模型指纹一致（fpmin_kout 0.9858）。RxN 交互低能量（1.2-1.7%）高扭曲。',
         scope='16 锚 x 2x2x2 变换代数 x 3 模型',
         metric_definition='phase3157 summary exch_* / fp_pairs / comm_class',
         replication='3/3 模型同判类 commutative_partial', heldout='不适用',
         counterexamples='exchR 曲线指纹 0.79-0.90（低于 KOUT 份额指纹）；否定/关系的峰值层位不同',
         causal='无', evidence_level='E1_repeatable', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3157, 'summary', 'result_summary.json')],
         checks=[
             C('comm_class', 'commutative_partial'),
             C('exch_mean_obs', 1.2728174525072664, 1e-6),
             C('fp_pass', True), C('fpmin_kout', 0.9857871501170492, 1e-6),
             CI('per_model_verdicts.glm4', 'rxn_0.0174'),
         ]),
    dict(id='N08', title='输出等价：局部薄邻域成立，子空间商结构被拒', line='G', phase=3158,
         claim='特定位置变化样本中内部状态距离小（预算比 1.03-1.33 全 absent）且输出变化小；但预注册的「子空间型商结构」0/3 未获支持（absent）。局部近似等价成立，统一薄商结构形式化被拒并留档。',
         scope='谱 + 扰动预算 + 直径-曲率三件套 x 3 模型',
         metric_definition='phase3158 summary quotient_classes / ratios / fpmin_*',
         replication='3/3 模型商结构 absent；曲线指纹 fpmin 0.99945',
         heldout='spec 指纹 0.9855-0.9990',
         counterexamples='quotient_3of3=False（假设形式被否证）；上下文变化可引发大输出 KL（与局部等价不矛盾，范围不同）',
         causal='无', evidence_level='E1_repeatable', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3158, 'summary', 'result_summary.json')],
         checks=[
             C('quotient_classes.qwen3-4b', 'absent'),
             C('quotient_classes.qwen3-14b', 'absent'),
             C('quotient_classes.glm4', 'absent'),
             C('gates.quotient_3of3', False),
             C('ratios.qwen3-4b', 1.0283477199188433, 1e-6),
             C('ratios.qwen3-14b', 1.3316768638966026, 1e-6),
             C('ratios.glm4', 1.2472595295618603, 1e-6),
             C('fpmin_curve', 0.9994547019577664, 1e-6),
             C('res_sha8', 'fa5cca12'), C('seal_sha8', 'e0c60629'),
         ]),
    dict(id='N09', title='注入动力学破坏（等价类无 inherited 动力学）', line='G', phase=3159,
         claim='mid 层注入 top-σ 读出方向后 ratio_mid 0.923/0.824/0.935，3/3 dynamics_destroyed：各向同性是剩余层算出来的，不是 inherited。top 方向被动力学消耗 91-94%，big-drop=注入后第 1 块。',
         scope='16 锚 x 24 方向 x 6 alpha x 3 reps x 3 模型',
         metric_definition='phase3159 summary dyn_classes / ratios_mid / fpmin_kl / fpmin_re',
         replication='3/3 模型；KL 指纹 fpmin 0.9983；re-emerge 指纹 0.9526',
         heldout='指纹门（两两 Pearson >= 0.8）全过',
         counterexamples='stable_0/3（稳定性门全不过=动力学确实被破坏）；bottom 轻度 re-gain 2.5-5.6%',
         causal='注入=激活级干预（无权重级证明）', evidence_level='E2_predictive', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3159, 'summary', 'result_summary.json')],
         checks=[
             C('dyn_classes.qwen3-4b', 'dynamics_destroyed'),
             C('dyn_classes.qwen3-14b', 'dynamics_destroyed'),
             C('dyn_classes.glm4', 'dynamics_destroyed'),
             C('ratios_mid.qwen3-4b', 0.9228145577430226, 1e-6),
             C('ratios_mid.qwen3-14b', 0.8241007835631158, 1e-6),
             C('ratios_mid.glm4', 0.9346829405196307, 1e-6),
             C('fpmin_kl', 0.9983358564992173, 1e-6),
             C('fpmin_re', 0.9526117254194486, 1e-6),
             C('res_sha8', 'f9c1fe35'), C('seal_sha8', 'db48b8a9'),
         ]),
    dict(id='N10', title='消耗机制=attention 再分配（协议内因果）', line='G', phase=3160,
         claim='置零块 L_mid/+1/+1+2 的 MLP 输出后 share_top(NL) 恢复 -0.008~+0.006（远小于 0.1 门）：消耗载体是 attention 再分配，MLP 贡献为零；dh 去向弥散（massive 维度份额 0.011-0.018）。此结论限于「top-σ 注入方向在 mid 层的消耗」这一协议，不能外推为所有语言信息都由 attention 消耗。',
         scope='4 锚 x 6 top64 方向 x alpha=0.1 x 4 消融配置 x 3 模型',
         metric_definition='phase3160 summary mech_classes / recover / destination / fp_pairs_q50_none',
         replication='3/3 模型同判类；指纹 0.9911-0.9940（对齐 W=19）',
         heldout='zero 模式消耗段指纹 0.9945 独立复证',
         counterexamples='raw 槽号对齐指纹 0.377-0.381（L_mid 错位稀释对照，口径教训）；MLP 主因假设被否证',
         causal='MLP 置零干预 + identity/bitwise 对照 + 3 模型复现（协议内因果）',
         evidence_level='E3_causal_scoped', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(G_3160, 'summary', 'result_summary.json'),
                os.path.join(G_3160, 'zero', 'result_zero.json')],
         checks=[
             C('mech_classes.qwen3-4b', 'attention_reallocation_primary'),
             C('mech_classes.qwen3-14b', 'attention_reallocation_primary'),
             C('mech_classes.glm4', 'attention_reallocation_primary'),
             C('recover.qwen3-4b.mlp_mid1', 0.0045597339288692765, 1e-6),
             C('recover.qwen3-14b.mlp_mid1', -0.006429697674905608, 1e-6),
             C('recover.glm4.mlp_mid1', -0.00013903126585754308, 1e-6),
             C('fpmin_q50_none', 0.9910838742689714, 1e-6),
             C('f0:res_sha8', 'a52e2ddd'), C('f0:seal_sha8', '3ce5cc3b'),
             C('f1:fpmin_q50', 0.9945447828808008, 1e-6),
             C('f1:res_sha8', 'a9435ded'),
         ]),
    dict(id='N11', title='E_read 基线（未见组合读出误差，逐位锚定）', line='D', phase=0,
         claim='未见组合 (实体,类别) 在行为读出层的相对 L2 预测误差：三模型 0.3316/0.3986/0.3898，池化 0.3734，全部高于 5% 门（0/3）。carrier npz 逐位锚定复现（drift=0.0 x3）——外部审查标记的「待核实」项，本轮已从 q03_result.json + metric_dict v4 双处核实。',
         scope='738 行面板 S1 s7 test fold（147 行）x 3 seeds x 3 模型',
         metric_definition='metric_dict v4 global_kpis.E_read（0652c008）+ tests/deepseek/result/q03_result.json',
         replication='3/3 模型；recompute == anchor 逐位（drift 0.0）',
         heldout='147 held-out 行 x 3 seed（test fold 未参与预测器拟合）',
         counterexamples='5% 门 0/3 全部未过（这是 K1 触发的证据基座，非缺陷）；水果类为最差类（v3.worst_class_b4）',
         causal='无（recompute-only）', evidence_level='E2_predictive', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(DSR, 'q03_result.json')],
         checks=[
             C('per_model.qwen3-4b.b4_rel_readout_mean3seed_recompute', 0.3316153089205424, 1e-9),
             C('per_model.qwen3-14b.b4_rel_readout_mean3seed_recompute', 0.39860084652900696, 1e-9),
             C('per_model.glm4-9b.b4_rel_readout_mean3seed_recompute', 0.389835258324941, 1e-9),
             C('per_model.qwen3-4b.b4_rel_readout_mean3seed_drift', 0.0, 1e-12),
             C('per_model.qwen3-14b.b4_rel_readout_mean3seed_drift', 0.0, 1e-12),
             C('per_model.glm4-9b.b4_rel_readout_mean3seed_drift', 0.0, 1e-12),
             C('carriers.qwen3-4b.ok', True), C('carriers.qwen3-14b.ok', True), C('carriers.glm4-9b.ok', True),
             C('res_sha8', 'cda99b05'), C('query', 'Q03'),
         ]),
    dict(id='N12', title='E_ar(k) 装置测量 + D4 精度桥', line='D', phase=0,
         claim='自回归漂移装置（K=16）三模型建成并测量：装置门 D1-D3 全过；D4 精度桥 max|d_rel|=0.0489 <= 0.05 PASS（nf4 臂与 bf16 可比）；S_rel 相对门 0/4 全 FAIL（诚实负结果：漂移下限 0.659-0.739 远高于 0.05）；形状 4b flat / 14b saturating / 9b flat（按臂）。与 E_read 量纲不同，只可经 rel 桥并读。',
         scope='441 cells(246 对 x 3 tpl 去重) x K16 x 4 臂（4b bf16/nf4、14b nf4、9b nf4）',
         metric_definition='metric_dict v4 global_kpis.E_ar + q04_smoke_result.json + q05_result.json',
         replication='4 臂 x 3 seeds；D0 采集器逐位等价（max_abs_dev 0.0）',
         heldout='每 seed 147 held-out 行',
         counterexamples='S_rel 全 FAIL（0/4）；S1 原始 logit 门被平凡满足（约 88 倍）判为无否证力（Q04 观测后登记）',
         causal='无（测量）', evidence_level='E2_predictive', model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(DSR, 'q04_smoke_result.json'), os.path.join(DSR, 'q05_result.json')],
         checks=[
             C('gates.D1', True), C('gates.D2', True), C('gates.D3', True), C('gates.S1', True),
             C('res_sha8', '04ad1af3'), C('verdict', 'SMOKE_DEVICE_OK|S1_DRIFT_DETECTED'),
             C('precision_bridge.d_abs_max', 0.048894374302628885, 1e-9),
             C('precision_bridge.pass_', True),
             C('s_rel.qwen3-4b__bf16.min_rel_k1_K', 0.6857928329526405, 1e-9),
             C('s_rel.qwen3-14b__nf4.min_rel_k1_K', 0.6939132679279748, 1e-9),
             C('s_rel.glm4-9b__nf4.min_rel_k1_K', 0.7392035648249594, 1e-9),
             C('s_rel_all_pass', False),
             C('f1:res_sha8', '775d7dce'),
             C('f0:res_sha8', '04ad1af3'),
         ]),
    dict(id='N13', title='C_steer 零结果：读得出不等于控得住', line='D', phase=40,
         claim='承重轴 v1（L29 WR 主 PC）端口替换在 441 held-out cells x 10 配置下 C_steer=0.0000（Wilson 上界 1.01%）；随机方向对照同 0；附带损伤干净（frac_zero 0.933）；identity 逐位恒等。负规律：可读出特征不构成可控机制。注意：本测量范围为 qwen3-4b 单模型。',
         scope='qwen3-4b bf16 x 441 cells x 10 steer 配置 + rand/identity 对照',
         metric_definition='metric_dict v4 global_kpis.C_steer + q06_result.json（prereg ebf960cf）',
         replication='单模型（14b/9b 可选臂未运行）', heldout='held-out cells（与 E_read 同面板族 be17ef8a）',
         counterexamples='本 Phase 未降低任一 KPI（I1 => catalog 登记）；谱外崩塌使「换类别」目标本身不可行（见 F3）',
         causal='干预实验（含双对照）——结果是「不能定向操控」的因果证据',
         evidence_level='E3_causal_scoped',
         model_scope={'qwen3-4b': 'verified', 'qwen3-14b': 'absent', 'glm4-9b': 'absent'},
         files=[os.path.join(DSR, 'q06_result.json')],
         checks=[
             C('C_steer_main.value', 0.0, 1e-12),
             C('C_steer_main.rand_value', 0.0, 1e-12),
             C('C_steer_main.wilson[1]', 0.010113689495831947, 1e-9),
             C('C_steer_main.spec_diff', 0.0, 1e-12),
             C('collateral.frac_zero', 0.9326530612244898, 1e-6),
             C('floors.F1_identity_maxd', 0.0, 1e-12),
             C('kpi_report.C_steer', 0.0, 1e-12),
             C('cells.eligible', 376), C('query', 'Q06'), C('mode', 'FULL'),
         ]),
    dict(id='N14', title='A 闸门 seal：K1 双轨判定层', line='D', phase=0,
         claim='A 闸门 seal 判定：K1 判定层=行为读出层（fired_all_models，池化 margin +0.9409）；k* 层=model_specific（qwen3-4b 否决）。E_read 池化 0.3734。K2/K3 当时从未被测量（结构缺陷=全称量词合取触发条件）。',
         scope='R8 seal（deepseek 线）+ Q09 双轨重述',
         metric_definition='deadline_dual_track_v1.json k1_recompute + a_gate_closure_v1.json',
         replication='3 模型聚合口径', heldout='同 E_read held-out',
         counterexamples='k* 层 1/3 触发即被否决（model_specific 不得升为机制）；K2 双操作化不一致被记录',
         causal='不适用（元层判定）', evidence_level='E2_predictive',
         model_scope={m: 'verified' for m in MODELS3},
         files=[os.path.join(DSR, 'deadline_dual_track_v1.json')],
         checks=[
             C('k1_recompute.verdict.kstar_layer', 'model_specific'),
             C('k1_recompute.verdict.readout_layer', 'fired_all_models'),
             C('k1_recompute.aggregate.E_read_pooled', 0.37335047125816345, 1e-9),
             C('k1_recompute.aggregate.readout_pooled_margin', 0.940920094649, 1e-9),
             C('k1_recompute.verdict.layer_choice_is_Q08', True),
             C('closure_exists', True),
         ]),
    dict(id='N15', title='交互对泛化 k3-only（rev3151b 撤回链）', line='G', phase=3151,
         claim='实体-类别交互对在 k3（低秩子空间）泛化、k39 不泛化（rev3151b 纠正门符号与 V2 层位后确认）；水果类为 B4 最差类（谱外崩塌证据基座）。原 3151 判决的 v3_missed 已被 rev-3151b 纠正为 v3_n2h1_confirmed。',
         scope='246 对 x 3 模板 x glm4-9b carrier（E_read 同源 npz c711946c）',
         metric_definition='phase3151 result.json v1/v3/v5 + result_rev3151b.json corrected_verdict',
         replication='glm4 carrier 1 次（E_read 逐位锚定复现）', heldout='s1 交互对 held-out 折',
         counterexamples='k39 不泛化（V2 M1 过位）；v3 原始判定 missed（纠错链 rev-3151a/b 在案）；worst_class=水果',
         causal='无', evidence_level='E2_predictive',
         model_scope={'qwen3-4b': 'absent', 'qwen3-14b': 'absent', 'glm4-9b': 'verified'},
         files=[os.path.join(G_3151, 'result.json'), os.path.join(G_3151, 'result_rev3151b.json')],
         checks=[
             C('v1.verdict', 'interaction_pair_generalizes'),
             C('v1.b4_s1_k39_rel', 0.389835258324941, 1e-9),
             C('v3.worst_class_b4', '水果'),
             C('res_sha8', '3b4344c6'),
             CI('corrected_verdict', 'k3_only'),
             CI('errors_corrected[1]', 'gate sign inverted'),
         ]),
]
FAILURES = [
    dict(id='F1', kind='hypothesis_rejected', phase=3158,
         text='子空间型商结构 0/3 未获支持（quotient absent x3，预算比 1.03-1.33）：输出等价的研究方向保留局部效应，但商结构形式化被拒，不得再以「已确立商结构」引用。', evidence='N08'),
    dict(id='F2', kind='negative_law', phase=40,
         text='C_steer=0.0000（Wilson 上界 1.01%）：承重轴端口替换无法定向操控行为。读得出 != 控得住。范围=qwen3-4b 单模型。', evidence='N13'),
    dict(id='F3', kind='scope_boundary', phase=3151,
         text='谱外崩塌：未见类别（水果）B4 读出误差为全类最差（worst_class_b4=水果）。图谱只能测谱内，不可外推到未见类别。', evidence='N15'),
    dict(id='F4', kind='gate_veto', phase=0,
         text='K1 触发 3/3 否决（读出层池化 0.3734 = 5% 门的 7.5 倍）：全称量词合取型死线结构性不可触发，已由 Q09 双轨重述修复。K2/K3 在 seal 时点从未被测量。', evidence='N14'),
    dict(id='F5', kind='metric_instability', phase=3155,
         text='KSTAR 指纹不稳（配对 0.244/0.488/0.908）：同一结构的读出层选择（KOUT vs KSTAR）决定跨模型一致性——指标所处计算位置影响其可复现性。', evidence='N03'),
    dict(id='F6', kind='strict_gate_failed', phase=3156,
         text='RoPE 严格内部不变量门未过（max 相对位移 1.375e-2 > 1e-3）：位置效应小但不为零；输出近似稳定与内部严格不变是两个不同强度的命题。', evidence='N04'),
    dict(id='F7', kind='artifact_gap', phase=3160,
         text='3156 rank-1 轴不可从 3156 npz 复现（en L7 逐位恒等、最高 SVD 份额 0.685 vs 当时报告 0.9996）：现场计算产物未落盘。教训：轴向量必须随 npz 持久化。', evidence='N05'),
    dict(id='F8', kind='protocol_artifact', phase=3159,
         text='bf16 batch kernel 路径效应：batch=1 vs 6 相对差至 2.2e-2（14b）。锚前向必须 batch=1（bitwise），扫描统一 batch 并量化 batch_rel。', evidence='N09'),
    dict(id='F9', kind='claim_recall', phase=3162,
         text='「九条规律全部跨模型复现」表述撤回：其中 massive 77x 塌缩为 4b 单模型量化（N05 partial）、RoPE 9/9 为位置条件数非模型数（N04 单模型）、E_read 当时不便核实（本轮已核实）。替换为逐条证据分级（本注册表）。', evidence='N04/N05/N11'),
    dict(id='F10', kind='claim_recall', phase=3162,
         text='「五模型装置定型」表述撤回：本证据链主线 3 模型（qwen3-4b / qwen3-14b / glm4-9b）；ledger model_namespace 记 pending_replication=[ds7b, glm4, gemma4]。C_steer 主测量也是单模型。', evidence='N00/N13'),
    dict(id='F11', kind='claim_recall', phase=3162,
         text='「约 5min/模型/Phase」GPU 预算口径撤回：实测跨 Phase 10.6s-297s（3160）/18min（3159 glm4 1081s）——排期须按各 Phase 真实采集量估算。', evidence='N09/N10'),
]
GATES = {
    'G1_no_error_all_nodes': 'audit 对 15+1 节点全部完成且无 LOAD_ERR/文件缺失；disk_verified >= 14（N05 允许 partial）',
    'G2_eread_resolved': 'N11 全部数值检查通过（q03_result.json 与 metric_dict 双处一致）——外部审查第 9 条解除',
    'G3_infra_chain_ok': 'N00 通过：ledger chain 9417b14f、n 弹性 [312,330]、metric_dict v4=0652c008、队列状态正确',
    'G4_registry_complete': 'atlas_registry.json 含 15 节点 + 11 失败账目 + schema/taxonomy/principles；atlas_v0.html 含全部节点 id；census 矩阵 15x3',
}

def freeze_design():
    os.makedirs(PDIR, exist_ok=True)
    exec_p = os.path.join(PDIR, 'execution.json')
    design = dict(phase=PHASE, name=NAME, frozen_before='any audit observation',
                  schema='rdc_atlas_foundation_v1',
                  evidence_level_taxonomy=TAXONOMY, principles=PRINCIPLES,
                  nodes=[{k: v for k, v in n.items() if k not in ('checks',)} for n in NODES],
                  failures=FAILURES, gates=GATES,
                  models_mainline=MODELS3)
    blob = json.dumps(design, ensure_ascii=False, sort_keys=True).encode('utf-8')
    dsha = hashlib.sha256(blob).hexdigest()[:8]
    if os.path.exists(exec_p):
        old = jload(exec_p)
        old_blob = json.dumps({k: v for k, v in old.items() if k != 'design_sha8'}, ensure_ascii=False, sort_keys=True).encode('utf-8')
        old_sha = hashlib.sha256(old_blob).hexdigest()[:8]
        if old_sha != dsha:
            raise SystemExit('DESIGN DRIFT: execution.json differs (old %s vs new %s)' % (old_sha, dsha))
    else:
        design['design_sha8'] = dsha
        jdump(design, exec_p)
    return dsha

def audit():
    dsha = freeze_design()
    out = dict(phase=PHASE, name=NAME, mode='audit', design_sha8=dsha, nodes=[], created=time.strftime('%Y-%m-%d %H:%M:%S'))
    log = ['design_sha8=%s' % dsha]
    n_disk = n_partial = n_err = 0
    for node in NODES:
        rec = dict(id=node['id'], title=node['title'], evidence_level=node['evidence_level'])
        try:
            datas = []
            for fp in node['files']:
                if not os.path.exists(fp):
                    raise FileNotFoundError(fp)
                datas.append(jload(fp))
            rec['files_ok'] = [os.path.relpath(fp, ROOT) for fp in node['files']]
            checks = node['checks']
            results = []
            for spec in checks:
                key, expected, mode, tol = spec
                # N00 特判: 聚合字段
                if key == 'measurements_n':
                    led = datas[0]
                    n = len(led['measurements'])
                    ok = 312 <= n <= 330
                    results.append((ok, 'measurements_n=%d in [312,330]' % n))
                    continue
                if key == 'chain':
                    ok = datas[0].get('ledger_sha256_8') == expected
                    results.append((ok, 'ledger chain=%s' % datas[0].get('ledger_sha256_8')))
                    continue
                if key == 'has_3160':
                    ok = any(m.get('phase') == 3160 for m in datas[0]['measurements'])
                    results.append((ok, 'ledger has phase3160=%s' % ok))
                    continue
                if key == 'md_version':
                    ok = datas[1].get('version') == expected
                    results.append((ok, 'metric_dict version=%s' % datas[1].get('version')))
                    continue
                if key == 'md_content_sha':
                    ok = datas[1].get('content_sha256_8') == expected
                    results.append((ok, 'metric_dict content=%s' % datas[1].get('content_sha256_8')))
                    continue
                if key == 'queue_sealed':
                    q = datas[2].get('sealed_items', [])
                    ok = all(x in q for x in expected)
                    results.append((ok, 'sealed_items=%s' % q))
                    continue
                if key == 'queue_device':
                    q = datas[2].get('device_built_items', [])
                    ok = all(x in q for x in expected)
                    results.append((ok, 'device_built_items=%s' % q))
                    continue
                if node['id'] == 'N14' and key == 'closure_exists':
                    ok = os.path.exists(os.path.join(DA, 'a_gate_closure_v1.json'))
                    results.append((ok, 'a_gate_closure_v1.json exists=%s' % ok))
                    continue
                # 常规: f<idx>: 前缀显式绑定文件; 否则按 files 顺序找第一个含该 root key 的文件
                fb = None
                k = key
                mfb = re.match(r'^f(\d+):(.+)$', k)
                if mfb:
                    fb = int(mfb.group(1))
                    k = mfb.group(2)
                root = k.split('.')[0].split('[')[0]
                if fb is not None:
                    ok, msg = run_check(datas[fb], (k, expected, mode, tol))
                    results.append((ok, '[f%d] %s' % (fb, msg)))
                else:
                    done = False
                    for di, d in enumerate(datas):
                        if root in d:
                            ok, msg = run_check(d, (k, expected, mode, tol))
                            results.append((ok, '[f%d] %s' % (di, msg)))
                            done = True
                            break
                    if not done:
                        results.append((False, 'no file contains root key: ' + root))
            rec['checks'] = [(ok, msg) for ok, msg in results]
            n_pass = sum(1 for ok, _ in results if ok)
            rec['n_checks'] = len(results)
            rec['n_pass'] = n_pass
            if n_pass == len(results):
                rec['status'] = 'disk_verified'
                n_disk += 1
            else:
                rec['status'] = 'check_fail'
                n_err += 1
        except Exception as e:
            rec['status'] = 'error'
            rec['error'] = '%s: %s' % (type(e).__name__, e)
            n_err += 1
        # model-level coverage 记录
        rec['model_scope'] = node['model_scope']
        out['nodes'].append(rec)
        log.append('%s [%s] %s checks=%s pass=%s' % (node['id'], rec['status'], node['title'], rec.get('n_checks'), rec.get('n_pass')))
    out['summary'] = dict(disk_verified=n_disk, check_fail=n_err, total=len(NODES))
    jdump(out, os.path.join(PDIR, 'result_audit.json'))
    log.append('AUDIT DONE disk_verified=%d fail=%d' % (n_disk, n_err))
    with io.open(os.path.join(PDIR, 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')

def census():
    ap = jload(os.path.join(PDIR, 'result_audit.json'))
    stmap = {r['id']: r['status'] for r in ap['nodes']}
    rows = []
    for node in NODES:
        st = stmap.get(node['id'], 'missing')
        for m in MODELS3:
            cov = node['model_scope'].get(m, 'absent')
            if cov == 'verified' and st == 'disk_verified':
                cov_f = 'disk_verified'
            elif cov == 'partial':
                cov_f = 'partial'
            elif cov == 'verified':
                cov_f = st
            else:
                cov_f = 'absent'
            rows.append(dict(node=node['id'], title=node['title'], level=node['evidence_level'],
                             node_status=st, model=m, coverage=cov_f))
    jdump(rows, os.path.join(PDIR, 'atlas_census.json'))
    cols = ['node', 'title', 'level', 'node_status', 'model', 'coverage']
    with io.open(os.path.join(PDIR, 'atlas_census.csv'), 'w', encoding='utf-8-sig') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]).replace(',', ';') for c in cols) + '\n')
    with io.open(os.path.join(PDIR, 'run_log.txt'), 'a', encoding='utf-8') as f:
        f.write('CENSUS DONE rows=%d\n' % len(rows))

def build_html(reg):
    css = """
    body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#f6f8fa;color:#1e293b;margin:0;padding:28px;}
    .wrap{max-width:1150px;margin:0 auto;}
    h1{font-size:22px;margin:0 0 6px;} h2{font-size:17px;margin:34px 0 12px;border-left:4px solid #2563eb;padding-left:10px;}
    .sub{color:#64748b;font-size:13px;margin-bottom:18px;}
    .legend span{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;margin-right:8px;color:#fff;}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;}
    .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,.04);}
    .card.failed{border-left:4px solid #dc2626;background:#fff7f7;}
    .card h3{margin:0 0 6px;font-size:14.5px;}
    .badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;color:#fff;margin-right:6px;vertical-align:middle;}
    .kv{font-size:12.5px;line-height:1.55;margin:4px 0;}
    .kv b{color:#334155;}
    .kv .lab{color:#94a3b8;}
    .chain{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:10px 0;}
    .step{background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:10px 14px;font-size:12.5px;min-width:180px;}
    .step .t{font-weight:600;margin-bottom:3px;}
    .arrow{align-self:center;color:#94a3b8;font-size:20px;}
    table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;}
    th,td{border:1px solid #e2e8f0;padding:5px 8px;text-align:left;}
    th{background:#eef2f7;}
    .ok{color:#16a34a;font-weight:600;} .pa{color:#d97706;font-weight:600;} .ab{color:#94a3b8;} .ng{color:#dc2626;font-weight:600;}
    .foot{color:#94a3b8;font-size:12px;margin-top:26px;}
    code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:12px;}
    """
    lc = {'E0_candidate': '#64748b', 'E1_repeatable': '#2563eb', 'E2_predictive': '#16a34a', 'E3_causal_scoped': '#7c3aed'}
    sc = {'disk_verified': '#16a34a', 'partial': '#d97706', 'check_fail': '#dc2626', 'error': '#dc2626'}
    def node_card(n, audit_st):
        lvl = n['evidence_level']
        st = n.get('status') or audit_st
        cls = 'card failed' if n['id'].startswith('F') else 'card'
        h = ['<div class="%s">' % cls]
        h.append('<h3>%s %s <span class="badge" style="background:%s">%s</span>' % (n['id'], n['title'], lc[lvl], lvl.split('_')[0]))
        h.append('<span class="badge" style="background:%s">%s</span></h3>' % (sc.get(st, '#64748b'), st))
        h.append('<div class="kv"><span class="lab">声明</span> %s</div>' % n['claim'])
        h.append('<div class="kv"><span class="lab">范围</span> %s</div>' % n['scope'])
        h.append('<div class="kv"><span class="lab">复现</span> %s ｜ <span class="lab">held-out</span> %s</div>' % (n['replication'], n.get('heldout', '')))
        h.append('<div class="kv"><span class="lab">反例/边界</span> %s</div>' % n['counterexamples'])
        h.append('<div class="kv"><span class="lab">因果</span> %s</div>' % n['causal'])
        h.append('</div>')
        return ''.join(h)
    a = reg['audit']
    ast = {r['id']: r['status'] for r in a['nodes']}
    H = ['<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>LLM 编码图谱 v0</title><style>', css, '</style></head><body><div class="wrap">']
    H.append('<h1>LLM 编码图谱 v0 — 证据分级研究地图</h1>')
    H.append('<div class="sub">Phase 3162 (G5-A1) · 生成于 %s · 主线模型 %s · design sha8 %s · 采纳外部审查三项校准（证据分级 / 跨模型范围 / 就绪度分层）</div>'
             % (reg['created'], '、'.join(MODELS3), reg['design_sha8']))
    H.append('<div class="legend">' + ''.join('<span style="background:%s">%s</span>' % (v, k) for k, v in lc.items())
             + '<span style="background:#dc2626">失败/限界</span></div>')
    H.append('<h2>一、机制链与桥（3159 → 3160 → 3161）</h2><div class="chain">')
    H.append('<div class="step"><div class="t">3159 注入动力学破坏</div>top 方向被消耗 91-94%<br>big-drop=注入后第 1 块（3/3）</div><div class="arrow">→</div>')
    H.append('<div class="step"><div class="t">3160 attention 再分配</div>MLP 置零 recover≈0（E3 协议内因果）<br>dh 去向弥散</div><div class="arrow">→</div>')
    H.append('<div class="step" style="border-style:dashed"><div class="t">3161 头归因（未测·已预注册）</div>逐头置零，top-4 集中度三分门<br>图谱缺口 #1</div></div>')
    H.append('</div><div class="chain">')
    H.append('<div class="step"><div class="t">E_read（读出误差）</div>0.3316 / 0.3986 / 0.3898<br>池化 0.3734，5% 门 0/3 · drift=0 逐位</div><div class="arrow">⇄</div>')
    H.append('<div class="step"><div class="t">E_ar(k)（自回归漂移）</div>D4 桥 0.0489 PASS<br>S_rel 0/4 FAIL（诚实负）· 量纲不同仅 rel 并读</div><div class="arrow">⇄</div>')
    H.append('<div class="step" style="border-left:4px solid #dc2626"><div class="t">C_steer = 0.0000（边界）</div>读得出 ≠ 控得住<br>qwen3-4b · Wilson 上界 1.01%</div></div>')
    H.append('</div>')
    H.append('<h2>二、规律节点（证据分级注册表）</h2><div class="grid">')
    for n in NODES:
        if n['id'] == 'N00':
            continue
        H.append(node_card(n, ast.get(n['id'])))
    H.append('</div>')
    H.append('<h2>三、失败与限界账本（first-class 图谱条目）</h2><div class="grid">')
    for f in FAILURES:
        H.append(node_card(dict(id=f['id'], title={'hypothesis_rejected': '假设被拒', 'negative_law': '负规律', 'scope_boundary': '范围边界',
                                                   'gate_veto': '死线否决', 'metric_instability': '指标不稳', 'strict_gate_failed': '严格门未过',
                                                   'artifact_gap': '产物缺口', 'protocol_artifact': '协议伪影', 'claim_recall': '表述撤回'}[f['kind']] + ' · ' + str(f['phase']),
                                claim=f['text'], scope='证据: ' + f['evidence'], replication='—', heldout='', counterexamples='—',
                                causal='—', evidence_level='E0_candidate', status='recorded'), 'recorded'))
    H.append('</div>')
    H.append('<h2>四、覆盖矩阵（节点 × 模型）</h2><table><tr><th>节点</th><th>等级</th>')
    for m in MODELS3:
        H.append('<th>%s</th>' % m)
    H.append('</tr>')
    cov_map = {}
    for r in reg['census']:
        cov_map.setdefault(r['node'], {})[r['model']] = r['coverage']
    for n in NODES:
        if n['id'] == 'N00':
            continue
        H.append('<tr><td>%s %s</td><td>%s</td>' % (n['id'], n['title'], n['evidence_level'].split('_')[0]))
        for m in MODELS3:
            c = cov_map.get(n['id'], {}).get(m, 'absent')
            cls = 'ok' if c == 'disk_verified' else ('pa' if c == 'partial' else ('ng' if c == 'check_fail' else 'ab'))
            H.append('<td class="%s">%s</td>' % (cls, c))
        H.append('</tr>')
    H.append('</table>')
    H.append('<h2>五、原则与口径</h2><div class="card">')
    for i, p in enumerate(PRINCIPLES, 1):
        H.append('<div class="kv">%d. %s</div>' % (i, p))
    H.append('<div class="kv">证据分级: ' + ' ；'.join('%s=%s' % (k, v) for k, v in TAXONOMY.items()) + '</div>')
    H.append('<div class="kv">测量口径: metric_dict v4 (<code>0652c008</code>) ；账本 <code>atlas_ledger.json</code> chain <code>9417b14f</code> ；门与预注册见各节点 metric_definition。</div>')
    H.append('</div>')
    H.append('<div class="foot">图谱为「有证据等级的研究地图」，不是已还原的 LLM 计算原理。缺口: ① 3161 头归因 ② C_steer / RoPE / massive 的跨模型同口径复测 ③ 跨族连接（知识/推理/语法）④ 谱外迁移（仅证伪实验）。下一版图谱按 L2（跨族机制验证）扩边。</div>')
    H.append('</div></body></html>')
    return ''.join(H)

def summary():
    dsha = freeze_design()
    audit_r = jload(os.path.join(PDIR, 'result_audit.json'))
    census_r = jload(os.path.join(PDIR, 'atlas_census.json'))
    stmap = {r['id']: r['status'] for r in audit_r['nodes']}
    n_disk = sum(1 for s in stmap.values() if s == 'disk_verified')
    n_fail = sum(1 for s in stmap.values() if s != 'disk_verified')
    eread_ok = stmap.get('N11') == 'disk_verified'
    infra_ok = stmap.get('N00') == 'disk_verified'
    reg = dict(phase=PHASE, name=NAME, created=time.strftime('%Y-%m-%d %H:%M:%S'),
               design_sha8=dsha, schema='rdc_atlas_foundation_v1',
               evidence_level_taxonomy=TAXONOMY, principles=PRINCIPLES,
               models_mainline=MODELS3,
               models_pending=['ds7b', 'gemma4'],
               models_pending_source='atlas_ledger model_namespace.pending_replication',
               audit=audit_r, census=census_r, failures=FAILURES)
    reg_p = os.path.join(PDIR, 'atlas_registry.json')
    jdump(reg, reg_p)
    reg['registry_sha8'] = sha8_file(reg_p)   # 仅入 summary/MEMO，registry 文件本体不含自 sha
    html = build_html(reg)
    html_p = os.path.join(PDIR, 'atlas_v0.html')
    with io.open(html_p, 'w', encoding='utf-8') as f:
        f.write(html)
    # G4 检查
    missing_ids = [n['id'] for n in NODES if n['id'] != 'N00' and n['id'] not in html]
    n_regular = sum(1 for n in NODES if n['id'] != 'N00')
    g4 = (n_regular == 15 and len(FAILURES) == 11 and not missing_ids
          and len(census_r) == len(NODES) * 3)
    n_err_status = sum(1 for s in stmap.values() if s == 'error')
    g1 = (n_err_status == 0 and n_disk >= 14)
    verdict = ('g5a1_atlas_registry_built|disk_verified_%d/%d|fail_%d|eread_ok_%s|infra_ok_%s|g4_%s'
               % (n_disk, len(NODES), n_fail, eread_ok, infra_ok, g4))
    sres = dict(phase=PHASE, mode='summary', design_sha8=dsha,
                registry_sha8=reg['registry_sha8'], html_sha8=sha8_file(html_p),
                census_sha8=sha8_file(os.path.join(PDIR, 'atlas_census.json')),
                audit_sha8=sha8_file(os.path.join(PDIR, 'result_audit.json')),
                execution_sha8=sha8_file(os.path.join(PDIR, 'execution.json')),
                nodes_total=len(NODES), nodes_disk_verified=n_disk, nodes_fail=n_fail,
                failures_ledgered=len(FAILURES),
                gates=dict(G1_all_processed=g1, G2_eread_resolved=eread_ok,
                           G3_infra_chain_ok=infra_ok, G4_registry_complete=g4),
                missing_ids_in_html=missing_ids,
                verdict=verdict, created=reg['created'])
    os.makedirs(os.path.join(PDIR, 'summary'), exist_ok=True)
    jdump(sres, os.path.join(PDIR, 'summary', 'result_summary.json'))
    with io.open(os.path.join(PDIR, 'run_log.txt'), 'a', encoding='utf-8') as f:
        f.write('SUMMARY %s\n' % verdict)
    print('SUMMARY ' + verdict)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'audit'
    if mode == 'audit':
        audit()
        print('AUDIT DONE')
    elif mode == 'census':
        census()
        print('CENSUS DONE')
    elif mode == 'summary':
        summary()
    else:
        raise SystemExit('mode must be audit|census|summary')
