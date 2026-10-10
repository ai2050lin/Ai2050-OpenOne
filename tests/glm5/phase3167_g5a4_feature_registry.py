# -*- coding: utf-8 -*-
"""Phase 3167 G5-A4: atlas registry v1 - cross-model stable feature registry (zero GPU).

Prereg (MEMO 3166 tail): build the atlas v1 feature registry. Every feature carries the
8-field schema (statement / evidence level E0-E3 / model scope / evidence phase anchors /
counter-evidence / replication protocol / values / scope limits). Sources = 3162 atlas
foundation (16 nodes x 122 checks) + gap-2/3 closure readings (3164/3165/3166) +
mechanism chain 3159-3166 + N-line A-gate seal. Gate: every feature needs >=2 anchors
from >=2 distinct phases + explicit model scope. All values are rendered on the fly
from sealed result files on disk (no hardcoded numbers into the registry payload).

SMOKE: P3167_SMOKE=1  (subset of anchors/features, all device gates still run).
"""
import io, json, os, sys, hashlib, datetime

ROOT = r'D:\AI2050\Ai2050-OpenOne'
GLM = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(GLM, 'phase3167', 'g5a4_feature_registry')
SMOKE = os.environ.get('P3167_SMOKE', '') == '1'
LOG = []

def log(s):
    LOG.append(str(s))
    print(s)

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# ---------------------------------------------------------------- frozen design
DESIGN = {
    'phase': 3167,
    'name': 'g5a4_feature_registry',
    'version': 'atlas_registry_v1',
    'mode': 'zero_gpu_registry_build',
    'schema_fields': ['id', 'family', 'statement', 'evidence_level', 'model_scope',
                      'anchors', 'counter_evidence', 'replication', 'values', 'scope_limits'],
    'evidence_taxonomy_verbatim_from': 'phase3162 atlas_registry.json evidence_level_taxonomy',
    'taxonomy': {
        'E0_candidate': '单次观察/单模型未复现',
        'E1_repeatable': '同协议跨 run/跨 cell 复现，或单模型内稳健',
        'E2_predictive': '通过 held-out 预测（新实体/新主题/新折），或跨模型指纹一致',
        'E3_causal_scoped': '协议内干预完成且排除主要替代解释；scope 内有效，不可外推',
    },
    'gates': {
        'G1_anchor_files': 'every anchor file exists and byte sha8 matches SHA_ANCHOR (frozen before SMOKE)',
        'G2_anchor_independence': 'each feature has >=2 anchors from >=2 distinct phases',
        'G3_model_scope': 'each feature model_scope is a non-empty subset of models_mainline',
        'G4_field_asserts': 'every assert path in every anchor matches the on-disk value (float tol 1e-9)',
        'G5_level_consistency': 'E3 needs an intervention-role anchor; E2 needs a cross_model or held_out anchor; E0 forbidden in v1',
        'G6_values_rendered': 'registry values are read from disk at run time (values_spec), never hardcoded',
        'G7_lineage': 'taxonomy+principles verbatim from 3162; failures F1-F11 carried verbatim + F12 appended; upgrade_log records every level/scope change vs 3162 nodes',
    },
    'models_mainline': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'failures_added': ['F12'],
    'smoke': {
        'anchors': ['p3154', 'p3162', 'p3165', 'p3166', 'q03', 'agate'],
        'features': ['FTR-01', 'FTR-03', 'FTR-05', 'FTR-17'],
    },
    'features_count_expected': 20,
    'float_tol_abs': 1e-9,
}

# ---------------------------------------------------------------- sources (all sealed)
SRC = {
    'p3154': GLM + r'\phase3154\g1p4_mfd_multifactor_disentangle\summary\result_summary.json',
    'p3155': GLM + r'\phase3155\g2p1_relation_family_operator_separability\summary\result_summary.json',
    'p3156': GLM + r'\phase3156\g3p1_position_shift_family\qwen3-4b\result.json',
    'p3157': GLM + r'\phase3157\g2p2_transform_algebra_commutator\summary\result_summary.json',
    'p3158': GLM + r'\phase3158\g4p1_output_equivalence_class\summary\result_summary.json',
    'p3159': GLM + r'\phase3159\g4p2_equivalence_dynamics\summary\result_summary.json',
    'p3160': GLM + r'\phase3160\g4p3_consumption_mechanism\summary\result_summary.json',
    'p3161': GLM + r'\phase3161\g4p4_head_attribution\summary\result_summary.json',
    'p3162': GLM + r'\phase3162\g5a1_atlas_foundation\atlas_registry.json',
    'p3163': GLM + r'\phase3163\g4p5_redundancy\summary\result_summary.json',
    'p3164a': GLM + r'\phase3164\g5a2_c_steer\result_summary.json',
    'p3164b': GLM + r'\phase3164\g5a2b_position_shift_cross_model\summary\result_summary.json',
    'p3164c': GLM + r'\phase3164\g5a2c_massive_cross_model\summary\result_summary.json',
    'p3165': GLM + r'\phase3165\g5a3_family_alignment\result.json',
    'p3166': GLM + r'\phase3166\g5a3b_logic_direction\result.json',
    'p3151': GLM + r'\phase3151\g1p1_combo_additive_vs_interaction\result_rev3151b.json',
    'q03': os.path.join(ROOT, r'tests\deepseek\result\q03_result.json'),
    'q05': os.path.join(ROOT, r'tests\deepseek\result\q05_result.json'),
    'q06': os.path.join(ROOT, r'tests\deepseek\result\q06_result.json'),
    'agate': os.path.join(ROOT, r'research\deepseek\atlas\a_gate_closure_v1.json'),
}

# byte sha8 probed 2026-10-09 (never from memory)
SHA_ANCHOR = {
    'p3154': '9fede5d5', 'p3155': '46a5034d', 'p3156': '9e67e780', 'p3157': '71959b6a',
    'p3158': '7ed85b36', 'p3159': 'a552f590', 'p3160': '2d97d24b', 'p3161': 'ec0e4488',
    'p3162': '00f15e98', 'p3163': 'db151a50', 'p3164a': 'e7a1f67d', 'p3164b': '2d5329dc',
    'p3164c': 'b5aaf29b', 'p3165': '511d9b13', 'p3166': '448af595', 'p3151': 'dd0cc176',
    'q03': '57827730', 'q05': '1d7beefc', 'q06': '6f571ac7', 'agate': '24c60160',
}

# ---------------------------------------------------------------- feature table
# assert key syntax: dotted path into the source json; 'node:NNN.field' resolves
# inside p3162 audit['nodes'] by node id.
F = []

F.append({
    'id': 'FTR-01', 'family': 'K', 'upgrade_from': 'N01', 'level_change': False,
    'statement': 'KOUT 残差流读出面板由 content 分量主导：mean 份额 C=57.6%、S=19.4%、R=15.9%（L/G/D 合计约 7%），逻辑轴中层显著且 held-out 全过；跨模型 KOUT 指纹 fpmin=0.996。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3154', 'phase': 3154, 'role': 'primary_measurement', 'tags': ['held_out', 'cross_model'],
         'asserts': {'shares_mean_kout.C': 0.5762712045154778, 'shares_mean_kout.S': 0.19399622367068037,
                     'shares_mean_kout.R': 0.1591576651654146, 'fp_min_kout': 0.995864698058461,
                     'ho_all_pass': True, 'logic_axis_sig_all': True,
                     'res_sha8': '2402f401', 'seal_sha8': 'a2b92c3e'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N01.status': 'disk_verified', 'node:N01.n_pass': 11, 'node:N01.n_checks': 11}},
    ],
    'counter_evidence': ['F5: 同一结构换读出层（KSTAR）指纹不稳 0.244-0.908 —— 份额读数绑定 KOUT 计算位置'],
    'replication': '读出层 logit 分解面板（多因素 L/S/C/G/D/R 份额回归 + held-out 新主题门），口径见 metric_dict v4；3157 同面板复测 fpmin=0.986。',
    'values_spec': [('p3154', 'shares_mean_kout.C', 'C_share'), ('p3154', 'shares_mean_kout.S', 'S_share'),
                    ('p3154', 'shares_mean_kout.R', 'R_share'), ('p3154', 'fp_min_kout', 'fpmin_kout')],
    'scope_limits': '读出层（logit 读出位置）度量；份额绑定 KOUT 层位口径；模型=主线三模型。',
})

F.append({
    'id': 'FTR-02', 'family': 'K', 'upgrade_from': 'N03', 'level_change': False,
    'statement': 'K2 条件门近似可分离：φ_ℓ(c) 与 W_ℓ 的交互份额 15.2%/18.2%/16.4% 远低于 50% 死线，k2_separable_all=True、死线未触发（held-out 加性迁移 9/9）。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3155', 'phase': 3155, 'role': 'primary_measurement', 'tags': ['held_out', 'cross_model'],
         'asserts': {'k2_separable_all': True, 'k2_death_line_triggered': False,
                     'k2_int_share_kout.qwen3-4b': 0.15187579769212284,
                     'k2_int_share_kout.qwen3-14b': 0.1824533831239199,
                     'k2_int_share_kout.glm4': 0.16411745881744746,
                     'res_sha8': '14975aed', 'seal_sha8': '5d2c2061'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N03.status': 'disk_verified', 'node:N03.n_pass': 9, 'node:N03.n_checks': 9}},
    ],
    'counter_evidence': [],
    'replication': '关系族算子可分离性协议（交互份额分解 + held-out 关系泛化门 1.5x），G2-P1 死线协议。',
    'values_spec': [('p3155', 'k2_int_share_kout.qwen3-4b', 'int_share_4b'),
                    ('p3155', 'k2_int_share_kout.qwen3-14b', 'int_share_14b'),
                    ('p3155', 'k2_int_share_kout.glm4', 'int_share_glm4')],
    'scope_limits': 'KOUT 读出位置；关系族=G 线面板内族；「近似可分离」不等于完全可加（交互份额非零）。',
})

F.append({
    'id': 'FTR-03', 'family': 'R', 'upgrade_from': 'N02', 'level_change': False,
    'statement': '逻辑（真/假命题）信号在中层稳定存在且可泛化：3154 逻辑轴回归三模型全显著 + held-out 全过；3166 真/假对比方向严格 leave-one-entity-out AUC 0.856/0.863/0.781（三模型独立方法复现）。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3154', 'phase': 3154, 'role': 'primary_measurement', 'tags': ['held_out', 'cross_model'],
         'asserts': {'logic_axis_sig_all': True, 'ho_all_pass': True}},
        {'src': 'p3166', 'phase': 3166, 'role': 'independent_direction_readout', 'tags': ['held_out', 'cross_model'],
         'asserts': {'per_model.qwen3-4b.device.G2.auc_mean': 0.8564,
                     'per_model.qwen3-14b.device.G2.auc_mean': 0.8634,
                     'per_model.glm4.device.G2.auc_mean': 0.781,
                     'res_sha8': '89b3f320', 'seal_sha8': '263de0ab'}},
    ],
    'counter_evidence': ['3152 K1 判定：k*=3 浅层无逻辑可读信号（K1 not_triggered）——逻辑信号限于中层以上'],
    'replication': '两条独立协议：(a) 逻辑轴份额回归 + held-out 新主题；(b) 真/假双臂均值差方向 + 严格 LOEO 分类（同实体外推）。',
    'values_spec': [('p3166', 'per_model.qwen3-4b.device.G2.auc_mean', 'loeo_auc_4b'),
                    ('p3166', 'per_model.qwen3-14b.device.G2.auc_mean', 'loeo_auc_14b'),
                    ('p3166', 'per_model.glm4.device.G2.auc_mean', 'loeo_auc_glm4')],
    'scope_limits': 'G 线 logic-tag 面板内；k*=3 浅层方向尺度塌缩约 3 个量级（3166 aux 槽读数）。',
})

F.append({
    'id': 'FTR-04', 'family': 'S', 'upgrade_from': None, 'level_change': False,
    'statement': '类词方向配方（2881 联合词坐标 J4 质心差分，float32 round-trip 链）跨模型重建成功：三模型各重建 10 方向/100 词 ok(n=10 words=100)，S_class 不再是 4b 专属。',
    'evidence_level': 'E1_repeatable',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3165', 'phase': 3165, 'role': 'primary_4b_build', 'tags': ['cross_model'],
         'asserts': {'pairwise_4b.K_readout__S_class.top1_deg': 67.0, 'res_sha8': '9b0fe9c9', 'seal_sha8': 'e966a4e5'}},
        {'src': 'p3166', 'phase': 3166, 'role': 'cross_model_rebuild', 'tags': ['cross_model'],
         'asserts': {'per_model.qwen3-4b.S_class_rebuild': 'ok(n=10 words=100)',
                     'per_model.qwen3-14b.S_class_rebuild': 'ok(n=10 words=100)',
                     'per_model.glm4.S_class_rebuild': 'ok(n=10 words=100)'}},
    ],
    'counter_evidence': ['数值敏感带：S_class 行空间病态，主角读数对 3.7e-9 级量化扰动敏感（float32 链 67.000 度 vs float64 66.726 度，差 0.27 度；判决不变）——登记口径=主脚本 float32 链'],
    'replication': '2881 配方 verbatim（质心=全部 single_tok 词均值，dW=质心-其余质心均值，float32 round-trip）；词序对拍 tl74/tl78/tl81。',
    'values_spec': [('p3165', 'pairwise_4b.K_readout__S_class.top1_deg', 'angle_4b_deg')],
    'scope_limits': '2881 面板 6 类词表内；dtype 口径=float32 round-trip 链（换链角度读数移动约 0.3 度）。',
})

F.append({
    'id': 'FTR-05', 'family': 'SxK', 'upgrade_from': None, 'level_change': False,
    'statement': '类词子空间与知识读出子空间几何可分且跨模型：K_readout×S_class 主角 67.0/71.9/74.0 度（三模型全过 30 度可分门，无半能量共享方向），4b 角度在 3166 独立 phase 对拍锚逐位复现（67.000 度）。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3165', 'phase': 3165, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'pairwise_4b.K_readout__S_class.top1_deg': 67.0, 'res_sha8': '9b0fe9c9'}},
        {'src': 'p3166', 'phase': 3166, 'role': 'independent_phase_replication', 'tags': ['cross_model'],
         'asserts': {'per_model.qwen3-4b.census.K_readout__S_class.top1_deg': 67.0,
                     'per_model.qwen3-14b.census.K_readout__S_class.top1_deg': 71.9,
                     'per_model.glm4.census.K_readout__S_class.top1_deg': 74.014}},
    ],
    'counter_evidence': ['dtype 敏感带（见 FTR-04）——角度读数须带 dtype 口径引用'],
    'replication': '主角度谱协议（QR+svd，clip 后 arccos 度数）；3165 构建 4b 全谱，3166 跨模型 census+4b 对拍锚（tol 0.05 度）。',
    'values_spec': [('p3166', 'per_model.qwen3-4b.census.K_readout__S_class.top1_deg', 'angle_4b_deg'),
                    ('p3166', 'per_model.qwen3-14b.census.K_readout__S_class.top1_deg', 'angle_14b_deg'),
                    ('p3166', 'per_model.glm4.census.K_readout__S_class.top1_deg', 'angle_glm4_deg')],
    'scope_limits': '统一 D 模型读出空间；S_class 10 方向；可分门=30 度（预注册 3165）。',
})

F.append({
    'id': 'FTR-06', 'family': 'S', 'upgrade_from': None, 'level_change': False,
    'statement': '属性/语法词方向子空间（4b 构建）与知识读出、逻辑方向子空间均几何可分：K_readout×S_attr=68.1 度、×S_syntax=58.5 度（3165），R_logic×S_attr=85.6 度、×S_syntax=86.3 度（3166 独立确认）——词表方向目前仅 4b（跨模型 pending）。',
    'evidence_level': 'E1_repeatable',
    'model_scope': ['qwen3-4b'],
    'anchors': [
        {'src': 'p3165', 'phase': 3165, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'pairwise_4b.K_readout__S_attr.top1_deg': 68.094,
                     'pairwise_4b.K_readout__S_syntax.top1_deg': 58.488}},
        {'src': 'p3166', 'phase': 3166, 'role': 'independent_phase_replication', 'tags': ['cross_model'],
         'asserts': {'per_model.qwen3-4b.census.R_logic__S_attr.top1_deg': 85.57,
                     'per_model.qwen3-4b.census.R_logic__S_syntax.top1_deg': 86.318}},
    ],
    'counter_evidence': ['S_attr/S_syntax 尚无 14b/glm4 重建（词表绑定 4b tokenizer 面）——跨模型结论不适用；p3166 census 的 S_attr/S_syntax 仅 4b 存在'],
    'replication': '2874/2878 词表 dW_unit + 主角度谱（3165）；3166 独立 census 对同一 S 子空间第二读数（R 侧）。',
    'values_spec': [('p3165', 'pairwise_4b.K_readout__S_attr.top1_deg', 'attr_vs_k_deg'),
                    ('p3165', 'pairwise_4b.K_readout__S_syntax.top1_deg', 'syntax_vs_k_deg'),
                    ('p3166', 'per_model.qwen3-4b.census.R_logic__S_attr.top1_deg', 'attr_vs_r_deg'),
                    ('p3166', 'per_model.qwen3-4b.census.R_logic__S_syntax.top1_deg', 'syntax_vs_r_deg')],
    'scope_limits': 'qwen3-4b 专属（model_scope 显式）；跨模型迁移待词表移植。',
})

F.append({
    'id': 'FTR-07', 'family': 'RxS', 'upgrade_from': None, 'level_change': False,
    'statement': '逻辑方向不是类词方向的混淆：R_logic×S_class 主角 73.7/81.4/80.8 度 separable×3 且 not_confounded（对比方向对类词方向做回归后逻辑 AUC 不塌）。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3166', 'phase': 3166, 'role': 'primary_measurement', 'tags': ['cross_model', 'held_out'],
         'asserts': {'per_model.qwen3-4b.census.R_logic__S_class.top1_deg': 73.704,
                     'per_model.qwen3-14b.census.R_logic__S_class.top1_deg': 81.381,
                     'per_model.glm4.census.R_logic__S_class.top1_deg': 80.77,
                     'overall.confound': 'not_confounded',
                     'res_sha8': '89b3f320'}},
        {'src': 'p3165', 'phase': 3165, 'role': 'independent_baseline', 'tags': ['cross_model'],
         'asserts': {'pairwise_4b.K_readout__S_class.top1_deg': 67.0}},
    ],
    'counter_evidence': [],
    'replication': '3166 普查协议（R×S 主角 + 混淆回归对照）；3165 K×S 基线独立确认类词方向位置。',
    'values_spec': [('p3166', 'per_model.qwen3-4b.census.R_logic__S_class.top1_deg', 'angle_4b_deg'),
                    ('p3166', 'per_model.qwen3-14b.census.R_logic__S_class.top1_deg', 'angle_14b_deg'),
                    ('p3166', 'per_model.glm4.census.R_logic__S_class.top1_deg', 'angle_glm4_deg')],
    'scope_limits': 'R_logic=top8 对比方向子空间（主槽 NL）；混淆对照=类词回归后 AUC 保持。',
})

F.append({
    'id': 'FTR-08', 'family': 'RxK', 'upgrade_from': None, 'level_change': False,
    'statement': '逻辑方向与实体子空间弱混合且模型特异：R×K_entity 主角 45.8/26.6/46.0 度（mixed），14b 落弱分离带并有 2 个半能量共享方向（4b/glm4 为 0）——全表唯一族间弱混合读数。',
    'evidence_level': 'E1_repeatable',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3166', 'phase': 3166, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'per_model.qwen3-4b.census.R_logic__K_entity.top1_deg': 45.845,
                     'per_model.qwen3-14b.census.R_logic__K_entity.top1_deg': 26.641,
                     'per_model.glm4.census.R_logic__K_entity.top1_deg': 46.005,
                     'per_model.qwen3-14b.census.R_logic__K_entity.eff_ge05': 2}},
        {'src': 'p3165', 'phase': 3165, 'role': 'independent_baseline', 'tags': ['cross_model'],
         'asserts': {'pairwise_4b.K_readout__K_entity.top1_deg': 69.823}},
    ],
    'counter_evidence': ['14b 共享方向的语义解释未定位（转图谱附录不阻塞）'],
    'replication': '3166 普查（R×K_entity 主角+有效维数）；3165 K 族内部结构基线（69.8 度）。',
    'values_spec': [('p3166', 'per_model.qwen3-4b.census.R_logic__K_entity.top1_deg', 'angle_4b_deg'),
                    ('p3166', 'per_model.qwen3-14b.census.R_logic__K_entity.top1_deg', 'angle_14b_deg'),
                    ('p3166', 'per_model.glm4.census.R_logic__K_entity.top1_deg', 'angle_glm4_deg')],
    'scope_limits': 'mixed 判定=落在 15-30 度弱分离带；仅 14b 有共享维；解释挂账。',
})

F.append({
    'id': 'FTR-09', 'family': 'mech', 'upgrade_from': 'N09', 'level_change': False,
    'statement': '等价类动力学非继承：中层注入读出谱方向后 dynamics_destroyed×3（ratio_mid 0.92/0.82/0.94），各向同性是剩余层算出来的而非 inherited；big-drop=注入后第 1 块，top 方向被动力学消耗 91-94%。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3159', 'phase': 3159, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'dyn_classes.qwen3-4b': 'dynamics_destroyed', 'dyn_classes.qwen3-14b': 'dynamics_destroyed',
                     'dyn_classes.glm4': 'dynamics_destroyed',
                     'ratios_mid.qwen3-4b': 0.9228145577430226, 'ratios_mid.qwen3-14b': 0.8241007835631158,
                     'ratios_mid.glm4': 0.9346829405196307, 'fpmin_kl': 0.9983358564992173,
                     'res_sha8': 'f9c1fe35', 'seal_sha8': 'db48b8a9'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N09.status': 'disk_verified', 'node:N09.n_pass': 10, 'node:N09.n_checks': 10}},
    ],
    'counter_evidence': ['F8: bf16 batch kernel 路径效应 2.2e-2 —— 锚前向 batch=1 bitwise，扫描统一 batch 并量化 batch_rel'],
    'replication': '16 锚×24 方向×6α 注入协议；指纹 KL>=0.998 + re-emerge>=0.95；方法学（hook 槽位/批效应）见 3159 addendum。',
    'values_spec': [('p3159', 'ratios_mid.qwen3-4b', 'ratio_mid_4b'),
                    ('p3159', 'ratios_mid.qwen3-14b', 'ratio_mid_14b'),
                    ('p3159', 'ratios_mid.glm4', 'ratio_mid_glm4')],
    'scope_limits': 'mid 层注入；方向=读出谱 top；α 网格协议内。',
})

F.append({
    'id': 'FTR-10', 'family': 'mech', 'upgrade_from': 'N10', 'level_change': False,
    'statement': '注入消耗的执行者第一步：GPU 层消融置零 L_mid/+1/+1+2 的 MLP 输出 recover 仅 -0.008~+0.006（远小于 0.1 门）→ MLP 无贡献，旋转由 attention 再分配执行（attention_reallocation_primary×3）。',
    'evidence_level': 'E3_causal_scoped',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3160', 'phase': 3160, 'role': 'primary_intervention', 'tags': ['intervention', 'cross_model'],
         'asserts': {'mech_classes.qwen3-4b': 'attention_reallocation_primary',
                     'mech_classes.qwen3-14b': 'attention_reallocation_primary',
                     'mech_classes.glm4': 'attention_reallocation_primary',
                     'recover.qwen3-4b.mlp_mid': 0.001858057977614794,
                     'recover.qwen3-14b.mlp_mid': -0.0030678492037642053,
                     'recover.glm4.mlp_mid': -0.0017226047550051903,
                     'res_sha8': 'a52e2ddd', 'seal_sha8': '3ce5cc3b'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N10.status': 'disk_verified', 'node:N10.n_pass': 11, 'node:N10.n_checks': 11,
                     'node:N10.evidence_level': 'E3_causal_scoped'}},
    ],
    'counter_evidence': ['F7: 3156 rank-1 轴不可从 npz 复现 → 3160 改用 3157 锚态 massive 维度（观测前修正）；3161 进一步修正排除法结论（见 FTR-11）'],
    'replication': '4 锚×6 top 方向×α=0.1 GPU 层消融协议；zero 分位曲线对齐指纹 0.9945。',
    'values_spec': [('p3160', 'recover.qwen3-4b.mlp_mid', 'recover_mlp_mid_4b'),
                    ('p3160', 'recover.qwen3-14b.mlp_mid', 'recover_mlp_mid_14b'),
                    ('p3160', 'recover.glm4.mlp_mid', 'recover_mlp_mid_glm4')],
    'scope_limits': '协议内因果（α=0.1、消融块=L_mid..+2）；「attention 再分配」在 3161 后理解为全流分布式中的早段瞬态执行。',
})

F.append({
    'id': 'FTR-11', 'family': 'mech', 'upgrade_from': 'N10', 'level_change': False,
    'statement': '消耗机制链收官：逐头置零 o_proj 输入无单点头贡献（T=0.056/0.030/0.007 均小于 0.1，ctrl 约等于 0）且整块恒等化不改变消耗（redundant_closing×3）→ 消耗无单点执行者=残差流冗余分布式性质（3159→3160→3161→3163 判闭）。',
    'evidence_level': 'E3_causal_scoped',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3161', 'phase': 3161, 'role': 'primary_intervention', 'tags': ['intervention', 'cross_model'],
         'asserts': {'cls.qwen3-4b': 'consumption_not_in_attn_out', 'cls.qwen3-14b': 'consumption_not_in_attn_out',
                     'cls.glm4': 'consumption_not_in_attn_out',
                     'T.qwen3-4b': 0.05628812471664884, 'T.qwen3-14b': 0.029683420838937883,
                     'T.glm4': 0.00717614111324244, 'res_sha8': '694395ad', 'seal_sha8': '000a7be7'}},
        {'src': 'p3163', 'phase': 3163, 'role': 'redundancy_intervention', 'tags': ['intervention', 'cross_model'],
         'asserts': {'clsA.qwen3-4b': 'redundant_closing', 'clsA.qwen3-14b': 'redundant_closing',
                     'clsA.glm4': 'redundant_closing',
                     'clsB.qwen3-4b': 'mlp_or_residual_primary', 'clsB.qwen3-14b': 'mlp_or_residual_primary',
                     'clsB.glm4': 'mlp_or_residual_primary',
                     'res_sha8': '8b99262c', 'seal_sha8': '48aa1445'}},
    ],
    'counter_evidence': ['3163 C 门（联合恒等化份额对照）14b 装置 fail（c_ok 2/3）——已登记，不影响 A/B 双判；3160 排除法结论被 3161 修正为分布式（机制结论更强而非更弱）'],
    'replication': '3161 逐头置零协议（块 L_mid/+1/+2，KV 不动，o_proj 输入才有头语义；head_dim 128 显式）+ 3163 联合置零恒等块+扩展窗。',
    'values_spec': [('p3161', 'T.qwen3-4b', 'T_concentration_4b'), ('p3161', 'T.qwen3-14b', 'T_concentration_14b'),
                    ('p3161', 'T.glm4', 'T_concentration_glm4')],
    'scope_limits': '置零类干预协议内；「无单点执行者」不排除 attention 早段瞬态参与（3161 早段差 0.046-0.051）。',
})

F.append({
    'id': 'FTR-12', 'family': 'context', 'upgrade_from': 'N05', 'level_change': True,
    'statement': '上下文二元门控：单 token 前缀即塌缩 massive activation（11072→428）且与 k 无关——has-context 二元门而非渐进上下文积累；3164 跨模型 supported×3（massive 维度 4b d1=0 / 14b d1=731 / glm4 d1=2319）。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3156', 'phase': 3156, 'role': 'primary_4b_measurement', 'tags': ['cross_model'],
         'asserts': {'ctx_effect.zh_k1.7': 103.15874481201172, 'top1.zh_k1': 0.0, 'res_sha8': 'c8c66d9f'}},
        {'src': 'p3164c', 'phase': 3164, 'role': 'cross_model_recheck', 'tags': ['cross_model'],
         'asserts': {'class_agreement_all3': True, 'd1_refs.qwen3-4b': 0, 'd1_refs.qwen3-14b': 731,
                     'd1_refs.glm4': 2319, 'res_sha8': '4d3174be', 'seal_sha8': 'b2ce1e42'}},
    ],
    'counter_evidence': ['massive 77x 塌缩幅度读数绑定 4b 量化口径（F9/F10 校准）——跨模型判定用 supported 门非幅度复现'],
    'replication': '双臂（真实前缀/位置重置）×k∈0..128；3164c 用 3157 锚态 massive 维度（3156 rank-1 轴不可复现后的修正锚）。',
    'values_spec': [('p3164c', 'd1_refs.qwen3-4b', 'd1_4b'), ('p3164c', 'd1_refs.qwen3-14b', 'd1_14b'),
                    ('p3164c', 'd1_refs.glm4', 'd1_glm4')],
    'scope_limits': '输出级门控读数；d1 维度索引用 3157 锚态口径；NF4 量化模型幅度读数带容差标注。',
})

F.append({
    'id': 'FTR-13', 'family': 'context', 'upgrade_from': 'N04', 'level_change': True,
    'statement': 'RoPE 纯相对性（输出级）：位置平移族输出近似不变（4b KL_B 小于等于 0.0023、top1_B 9/9），3164 跨模型 supported×3；严格内部不变量未过（max 相对位移 1.375e-2 大于 1e-3，F6）——输出稳定与内部严格不变是两个强度。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3156', 'phase': 3156, 'role': 'primary_4b_measurement', 'tags': ['cross_model'],
         'asserts': {'rope_check.zh_k1': 0.0006142156442533614, 'top1.zh_k1': 0.0, 'top1.en_k16': 1.0,
                     'res_sha8': 'c8c66d9f'}},
        {'src': 'p3164b', 'phase': 3164, 'role': 'cross_model_recheck', 'tags': ['cross_model'],
         'asserts': {'class_agreement_14b_glm4': True, 'res_sha8': 'd48dcbb7', 'seal_sha8': '55326bbe'}},
    ],
    'counter_evidence': ['F6: 严格内部不变量门未过——不得以「位置完全无内部效应」引用本特征'],
    'replication': '双臂位置平移协议×k 网格；跨模型门=supported（3156 协议同构移植）。',
    'values_spec': [('p3156', 'rope_check.zh_k1', 'kl_shift_zh_k1_4b'), ('p3156', 'rope_check.en_k128', 'kl_shift_en_k128_4b')],
    'scope_limits': '输出分布级；k 小于等于 128 窗；内部不变量保留 F6 否定读数。',
})

F.append({
    'id': 'FTR-14', 'family': 'context', 'upgrade_from': 'N06+N07', 'level_change': False,
    'statement': '上下文变换 T_C 的几何：共享分量关系无关（三模型 cos 约 0.56 一致）+ 内容特异并存；变换代数对易子 partial（exch 1.0-1.3=近正交非反平行；RxN 方差 1.2-1.7% 但几何扭曲大=低能量高扭曲型）。',
    'evidence_level': 'E1_repeatable',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3157', 'phase': 3157, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'fpmin_kout': 0.9857871501170492, 'exch_mean_obs': 1.2728174525072664,
                     'comm_class': 'commutative_partial', 'res_sha8': '0fe043bf', 'seal_sha8': 'cfa5c3ed'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N06.status': 'disk_verified', 'node:N07.status': 'disk_verified'}},
    ],
    'counter_evidence': ['对易子 partial——不得引用为「变换代数可交换」；T_C 的 cos=0.56 是关系无关共享分量，非全空间'],
    'replication': '16×2×2×2=128 行×3 模型对易子协议 + T_C 关系对照；指纹 fpmin 0.986。',
    'values_spec': [('p3157', 'exch_mean_obs', 'exch_mean'), ('p3157', 'fpmin_kout', 'fpmin_kout')],
    'scope_limits': 'mid 层锚态；对易子判定=exch 区间 1.0-1.3 partial；三模型一致部分仅 T_C cos。',
})

F.append({
    'id': 'FTR-15', 'family': 'mech', 'upgrade_from': 'N08', 'level_change': False,
    'statement': '输出等价的边界：局部薄邻域等价成立、子空间型商结构被拒（quotient 0/3 mixed，预算比 1.03/1.33/1.25）——商结构形式化不得再以「已确立」引用（F1）。',
    'evidence_level': 'E1_repeatable',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3158', 'phase': 3158, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'ratios.qwen3-4b': 1.0283477199188433, 'ratios.qwen3-14b': 1.3316768638966026,
                     'ratios.glm4': 1.2472595295618603, 'res_sha8': 'fa5cca12', 'seal_sha8': 'e0c60629'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N08.status': 'disk_verified', 'node:N08.n_pass': 10, 'node:N08.n_checks': 10}},
    ],
    'counter_evidence': ['F1: 子空间商结构 0/3 被拒（本特征即失败入图谱）'],
    'replication': '输出等价类协议（quotient 预算比 + 局部薄邻域曲线）；fpmin_spec/curve 双指纹。',
    'values_spec': [('p3158', 'ratios.qwen3-4b', 'budget_ratio_4b'), ('p3158', 'ratios.qwen3-14b', 'budget_ratio_14b'),
                    ('p3158', 'ratios.glm4', 'budget_ratio_glm4')],
    'scope_limits': 'top64 读出子空间口径；「局部等价」不外推为全局商结构。',
})

F.append({
    'id': 'FTR-16', 'family': 'control', 'upgrade_from': 'N13', 'level_change': True,
    'statement': '读得出不等于控得住（跨模型）：C_steer=0（4b 主测量 Wilson 上界 1.01%，rand 对照同 0，collateral 干净 frac0=0.933），3164 同构移植 14b/glm4 zero_like_q06×3——承重轴=生成稳定性轴非类身份杠杆。',
    'evidence_level': 'E3_causal_scoped',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'q06', 'phase': 40, 'role': 'primary_intervention', 'tags': ['intervention'],
         'asserts': {'C_steer_main.value': 0.0, 'C_steer_main.rand_value': 0.0,
                     'C_steer_main.wilson.1': 0.010113689495831947,
                     'collateral.frac_zero': 0.9326530612244898, 'res_sha8': '5f88ed7e'}},
        {'src': 'p3164a', 'phase': 3164, 'role': 'cross_model_intervention', 'tags': ['intervention', 'cross_model'],
         'asserts': {'ref_cls_4b': 'zero_like_q06', 'class_agreement_14b_glm4': True,
                     'res_sha8': '2114b4dc', 'seal_sha8': '8022c23d'}},
    ],
    'counter_evidence': ['F2: C_steer=0 首测（qwen3-4b 单模型）——3164 后升级为跨模型同向；M14 不对称（clip 移除破坏 66% 生成 vs 替换移动读出小于 1 logit）保留解释'],
    'replication': 'v1 承重轴（WR 主 PC 同构移植）+ x 端口替换，441 held-out cells×10 配置；跨模型同口径门（Wilson 上界+collateral 对照）。',
    'values_spec': [('q06', 'C_steer_main.value', 'c_steer_4b'), ('q06', 'C_steer_main.wilson.1', 'wilson_hi_4b'),
                    ('q06', 'collateral.frac_zero', 'collat_frac0_4b')],
    'scope_limits': 'v1 轴假设下；「零定向控制」不可外推为「任意轴不可控」（未测其他轴族）。',
})

F.append({
    'id': 'FTR-17', 'family': 'readout', 'upgrade_from': 'N11', 'level_change': False,
    'statement': 'E_read 未见组合读出误差基线：池化 0.3734（三模型 0.332/0.399/0.390），5% 门 0/3（最小 6.63x 超门）——未见类别×属性组合的读出误差结构性超门；三 collect.npz 锚逐位复现 drift=0.00e+00。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'q03', 'phase': 37, 'role': 'primary_measurement', 'tags': ['cross_model', 'held_out'],
         'asserts': {'summary.pooled_mean': 0.37335047125816345, 'summary.gate_pass_frac': '0/3',
                     'summary.min_E_x': 6.632306178410848,
                     'per_model.qwen3-4b.b4_rel_readout_mean3seed_drift': 0.0, 'res_sha8': 'cda99b05'}},
        {'src': 'agate', 'phase': 'R8', 'role': 'gate_seal_citation', 'tags': ['held_out'],
         'asserts': {'gate_closed': True, 'k1_reverdict.judging_layer_under_Q08_A': 'readout（行为读出层）'}},
    ],
    'counter_evidence': ['F3: 谱外（未见类别水果）B4 读出误差全类最差——E_read 基线只覆盖谱内未见组合，谱外另列（FTR-20）'],
    'replication': 'Q03 面板协议（246 组合×37 词×3 种子），held-out 指纹；A 闸门 seal 引用同值（k1_reverdict readout 层判定）。',
    'values_spec': [('q03', 'summary.pooled_mean', 'pooled_eread'), ('q03', 'summary.min_E_x', 'min_over_gate_x'),
                    ('q03', 'per_model.qwen3-4b.b4_rel_readout_mean3seed_drift', 'drift_4b')],
    'scope_limits': '归一化 MSE 口径（与 E_ar 的 logit L1 不同量纲，只可并排读）；谱内未见组合。',
})

F.append({
    'id': 'FTR-18', 'family': 'readout', 'upgrade_from': 'N12', 'level_change': False,
    'statement': 'E_ar(k) 装置与精度桥：四臂 738×K16 测量完成，D4 精度桥 max|delta_rel|=0.0489 小于 0.05 门 PASS；形状 4b=flat（bf16/nf4 双精度）/14b=saturating/glm4=flat；S_rel 0/4 过门（报告性，不阻塞）。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'q05', 'phase': 39, 'role': 'primary_measurement', 'tags': ['cross_model'],
         'asserts': {'shape.qwen3-4b__bf16.shape': 'flat', 'shape.qwen3-4b__nf4.shape': 'flat',
                     'shape.qwen3-14b__nf4.shape': 'saturating', 'shape.glm4-9b__nf4.shape': 'flat',
                     'precision_bridge.d_abs_max': 0.048894374302628885, 'precision_bridge.thr': 0.05,
                     'precision_bridge.pass_': True, 'res_sha8': '775d7dce'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N12.status': 'disk_verified', 'node:N12.n_pass': 14, 'node:N12.n_checks': 14}},
    ],
    'counter_evidence': ['S_rel 0/4 过门——min_rel 0.686-0.739 未达预注册门（report_only）；形状跨精度 4b 一致（bf16/nf4 双臂桥接）'],
    'replication': 'Q05 四臂协议（4b bf16 + 4b/14b/glm4 NF4），D4 桥=同模型双精度逐 k 相对差；metric_dict v4 口径。',
    'values_spec': [('q05', 'precision_bridge.d_abs_max', 'd4_bridge_max'), ('q05', 'shape.qwen3-14b__nf4.G', 'G_14b')],
    'scope_limits': 'logit L1 量纲（与 E_read 只可并排读）；NF4 精度策略=RAM 约束下的冻结决定（bf16 offload 不可行）。',
})

F.append({
    'id': 'FTR-19', 'family': 'gate', 'upgrade_from': 'N14', 'level_change': False,
    'statement': 'K1 双轨判定层（A 闸门 seal Q08=甲）：同一死线在 k* 层与读出层判定相反——k* 层 3/3 否决、读出层池化 0.3734=门 7.5x 触发；「条件齿轮组=算子代数」降级 descriptive；k* model_specific。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'agate', 'phase': 'R8', 'role': 'gate_seal', 'tags': ['held_out'],
         'asserts': {'gate_closed': True, 'k1_reverdict.judging_layer_under_Q08_A': 'readout（行为读出层）',
                     'deadlines.K1.state': 'FIRED (under Q08=甲)'}},
        {'src': 'p3162', 'phase': 3162, 'role': 'independent_disk_audit', 'tags': ['audit'],
         'asserts': {'node:N14.status': 'disk_verified', 'node:N14.n_pass': 6, 'node:N14.n_checks': 6}},
    ],
    'counter_evidence': ['F4: 全称量词合取型死线结构性不可触发（原始 K1 形式 3/3 否决）——双轨重述后才可判定；K2/K3 在 seal 时点未被测量（挂账）'],
    'replication': 'a_gate_closure_v1.json seal（Q08=甲全接受）；读出层判定引用 q03 E_read 同源值（0.3734）。',
    'values_spec': [('q03', 'summary.pooled_mean', 'pooled_eread_gate_value')],
    'scope_limits': 'Q08 语义下；判定层=readout（行为层）；K2/K3 仍未测量（挂账保留）。',
})

F.append({
    'id': 'FTR-20', 'family': 'limit', 'upgrade_from': None, 'level_change': False,
    'statement': '谱外崩塌（两线独立确认）：未见类别（水果）B4 读出误差全类最差（3151 rev 链 b4_worst_fold=shuiguo(1.2226)，v3 confirmed），与 N 线「未见类别仍崩（水果 0.04/0.05）」限界互证——图谱只能测谱内，不可外推到未见类别。',
    'evidence_level': 'E2_predictive',
    'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
    'anchors': [
        {'src': 'p3151', 'phase': 3151, 'role': 'primary_measurement_rev_chain', 'tags': ['held_out'],
         'asserts': {'evidence.v3.confirmed': True, 'evidence.v3.b4_worst_fold_k39': 'shuiguo(1.2226)',
                     'evidence.v4_s3_b4_mean': 0.6114, 'seal_sha8': '6409274e'}},
        {'src': 'q03', 'phase': 37, 'role': 'independent_line_confirmation', 'tags': ['held_out', 'cross_model'],
         'asserts': {'summary.gate_pass_frac': '0/3', 'summary.min_E_x': 6.632306178410848}},
    ],
    'counter_evidence': ['本特征即限界（三核心限界之一）；3151 原始交互泛化全称表述已由 rev3151b 撤回为 k3-only（N15/F4 链）'],
    'replication': '3151 面板 B4 fold 读出（rev3151b 修正链 seal a566dc96）+ N 线 Q03 未见组合超门（独立线证据）。',
    'values_spec': [('p3151', 'evidence.v4_s3_b4_mean', 'b4_mean_s3'), ('q03', 'summary.min_E_x', 'eread_min_over_gate_x')],
    'scope_limits': '水果类=2881 谱外；「谱内可测」不外推谱外；N 线数值（0.04/0.05）来自 deepseek 线口径。',
})

FEATURES = F

# ---------------------------------------------------------------- helpers
def getpath(obj, dotted):
    cur = obj
    for part in dotted.split('.'):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur

def get_assert(R, key):
    # 'node:NNN.field' -> p3162 audit nodes lookup by id
    if key.startswith('node:'):
        rest = key[len('node:'):]
        nid, field = rest.split('.', 1)
        for n in R['audit']['nodes']:
            if n['id'] == nid:
                return getpath(n, field)
        raise KeyError('audit node not found: %s' % nid)
    return getpath(R, key)

def close(a, b, tol):
    return abs(a - b) <= max(tol, 1e-9 * abs(b) if b else 0.0)

# ---------------------------------------------------------------- freeze
def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    # design hash excludes created/design_sha8 so re-runs stay stable
    core = {k: v for k, v in DESIGN.items() if k not in ('created', 'design_sha8')}
    raw = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
    d8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    body = dict(core)
    body['created'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body['design_sha8'] = d8
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s (edit DESIGN or delete file)'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        with io.open(exep, 'w', encoding='utf-8') as f:
            f.write(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True))
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8

# ---------------------------------------------------------------- run
def run():
    d8 = freeze()
    bad = []
    regs = {}

    # G1: anchor files + byte sha
    need = set()
    for ft in FEATURES:
        for a in ft['anchors']:
            need.add(a['src'])
    if SMOKE:
        need = set(DESIGN['smoke']['anchors']) & need
    for tag in sorted(need):
        p = SRC[tag]
        if not os.path.exists(p):
            bad.append('G1 missing %s (%s)' % (tag, p)); continue
        h = sha8(p)
        if h != SHA_ANCHOR[tag]:
            bad.append('G1 sha %s=%s expect %s' % (tag, h, SHA_ANCHOR[tag]))
    log('G1 anchor files checked: %d/%d' % (len(need), len(SHA_ANCHOR)))
    regs['G1_files_checked'] = len(need)

    # load
    R = {}
    for tag in need:
        R[tag] = json.load(io.open(SRC[tag], encoding='utf-8'))

    feats = [ft for ft in FEATURES if (not SMOKE or ft['id'] in DESIGN['smoke']['features'])]
    log('features to register: %d (smoke=%s)' % (len(feats), SMOKE))

    # G2-G6 per feature
    for ft in feats:
        fid = ft['id']
        # G2 independence
        phases = set(str(a['phase']) for a in ft['anchors'])
        if len(ft['anchors']) < 2 or len(phases) < 2:
            bad.append('G2 %s anchors=%d phases=%d' % (fid, len(ft['anchors']), len(phases)))
        # G3 scope
        ms = ft['model_scope']
        if not ms or not set(ms) <= set(DESIGN['models_mainline']):
            bad.append('G3 %s model_scope=%s' % (fid, ms))
        # G4 asserts + G5 tags + G6 values
        vals = {}
        for a in ft['anchors']:
            src = a['src']
            if src not in R:
                bad.append('G4 %s anchor %s not loaded (smoke subset?)' % (fid, src)); continue
            if not a.get('tags'):
                bad.append('G5 %s anchor %s missing tags' % (fid, src))
            for k, exp in sorted(a['asserts'].items()):
                try:
                    got = get_assert(R[src], k)
                except Exception as e:
                    bad.append('G4 %s %s.%s path error %s' % (fid, src, k, e)); continue
                if isinstance(exp, float) and isinstance(got, (int, float)):
                    if not close(float(got), exp, DESIGN['float_tol_abs']):
                        bad.append('G4 %s %s.%s got %r expect %r' % (fid, src, k, got, exp))
                elif isinstance(exp, bool) or isinstance(got, bool):
                    if bool(got) != bool(exp):
                        bad.append('G4 %s %s.%s got %r expect %r' % (fid, src, k, got, exp))
                else:
                    if str(got) != str(exp):
                        bad.append('G4 %s %s.%s got %r expect %r' % (fid, src, k, got, exp))
            if src not in ('p3162',):
                for st, path, okey in ft.get('values_spec', []):
                    if st == src:
                        try:
                            vals[okey] = get_assert(R[src], path)
                        except Exception as e:
                            bad.append('G6 %s values %s path error %s' % (fid, path, e))
        # G5 level consistency
        roles = [a.get('role', '') for a in ft['anchors']]
        tags_all = [t for a in ft['anchors'] for t in a.get('tags', [])]
        lv = ft['evidence_level']
        if lv == 'E3_causal_scoped' and not any('intervention' in r for r in roles):
            bad.append('G5 %s E3 without intervention anchor' % fid)
        if lv == 'E2_predictive' and not any(t in ('cross_model', 'held_out') for t in tags_all):
            bad.append('G5 %s E2 without cross_model/held_out anchor' % fid)
        if lv == 'E0_candidate':
            bad.append('G5 %s E0 forbidden in v1' % fid)
        ft['values'] = vals
        regs[fid] = {'anchors_n': len(ft['anchors']), 'phases': sorted(phases), 'level': lv,
                     'scope': ms, 'values_n': len(vals)}

    nG4 = sum(len(a['asserts']) for ft in feats for a in ft['anchors'] if a['src'] in R)
    log('G4 field asserts evaluated: %d' % nG4)
    regs['G4_asserts_evaluated'] = nG4

    # G7 lineage: load 3162 registry, carry failures + append F12
    reg3162 = json.load(io.open(SRC['p3162'], encoding='utf-8'))
    fails = [dict(x) for x in reg3162['failures']]
    assert len(fails) == 11, 'F ledger n=%d expect 11' % len(fails)
    fails.append({
        'id': 'F12', 'kind': 'claim_precision', 'phase': '3165->3166',
        'text': 'ledger 3165 detail 以「3151/3152 H 为纯真命题面板」作为 R 族 pending_material 理由——3166 复核证明该面板 41 实体×6 类全组合天然含真/假反事实双臂（理由不精确）；R 族方向已由 3166 零 GPU 补采完成，pending 解除。',
        'evidence': 'p3166 res 89b3f320 / MEMO 3166 节 / ledger 3166 detail',
    })
    log('G7 failures carried: F1-F11 verbatim + F12 appended (n=%d)' % len(fails))
    regs['G7_failures_n'] = len(fails)

    # weak ledger check (shared file, concurrent writers possible: existence + n >= 318 + has 3166 entry)
    ledp = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
    led = json.load(io.open(ledp, encoding='utf-8'))
    ms = led['measurements']
    has3166 = any(str(m.get('phase')) == '3166' for m in ms)
    if len(ms) < 318 or not has3166:
        bad.append('G7 ledger n=%d has3166=%s' % (len(ms), has3166))
    log('ledger: n=%d has3166=%s (weak check, shared file)' % (len(ms), has3166))
    regs['ledger_n'] = len(ms)

    if bad:
        log('FAILURES (%d):' % len(bad))
        for b in bad:
            log('  ' + b)
        raise SystemExit('device gates failed')

    # ---------------- assemble registry v1
    registry = {
        'phase': 3167, 'name': 'g5a4_feature_registry', 'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'schema': 'rdc_atlas_feature_registry_v1',
        'design_sha8': d8,
        'evidence_level_taxonomy': DESIGN['taxonomy'],
        'principles': reg3162['principles'],
        'models_mainline': DESIGN['models_mainline'],
        'models_pending': reg3162.get('models_pending', []),
        'upgrade_log': [
            {'node': 'N04->FTR-13', 'change': 'E1_repeatable->E2_predictive; scope 4b->3 models',
             'reason': '3164b cross-model rope_relative_supported x3 (res d48dcbb7)'},
            {'node': 'N05->FTR-12', 'change': 'E1_repeatable->E2_predictive; scope 4b->3 models',
             'reason': '3164c cross-model massive_context_gate_supported x3 (res 4d3174be)'},
            {'node': 'N13->FTR-16', 'change': 'E3 scope 4b->3 models (zero verdict cross-model same direction)',
             'reason': '3164a zero_like_q06 x3 (res 2114b4dc)'},
        ],
        'features': [{k: ft[k] for k in DESIGN['schema_fields']} for ft in feats],
        'failures': fails,
        'provenance': {'anchor_files': {t: {'path': SRC[t].replace(ROOT + os.sep, ''), 'sha8': SHA_ANCHOR[t]} for t in sorted(SHA_ANCHOR)}},
    }
    out_reg = os.path.join(OUTDIR, 'atlas_registry_v1.json')
    with io.open(out_reg, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=1)
    log('registry v1 written: %s (features=%d, failures=%d)' % (out_reg, len(registry['features']), len(fails)))

    # ---------------- result + seal
    n_lv = {}
    for ft in feats:
        n_lv[ft['evidence_level']] = n_lv.get(ft['evidence_level'], 0) + 1
    n_e3 = sum(1 for ft in feats if ft['level_change'])
    verdict = 'g5a4_registry_v1|%d_features|anchors_%d|asserts_%d_ok|E2_%d_E1_%d_E3_%d|upgrades_%d|failures_12' % (
        len(feats), regs['G1_files_checked'], regs['G4_asserts_evaluated'],
        n_lv.get('E2_predictive', 0), n_lv.get('E1_repeatable', 0), n_lv.get('E3_causal_scoped', 0), n_e3)
    summary = {
        'phase': 3167, 'name': 'g5a4_feature_registry', 'smoke': SMOKE,
        'verdict': verdict,
        'features_n': len(registry['features']), 'failures_n': len(fails),
        'anchors_checked': regs['G1_files_checked'], 'asserts_evaluated': regs['G4_asserts_evaluated'],
        'per_feature': regs,
        'ledger_n_weak': regs['ledger_n'],
        'upgrade_log': registry['upgrade_log'],
    }
    raw = json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    summary['res_sha8'] = res8
    if SMOKE:
        outp = os.path.join(OUTDIR, 'smoke_result.json')
    else:
        mid = json.dumps({k: v for k, v in summary.items() if k != 'seal_sha8'}, ensure_ascii=False, indent=1, sort_keys=True)
        seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
        summary['seal_sha8'] = seal8
        outp = os.path.join(OUTDIR, 'result.json')
    with io.open(outp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with io.open(os.path.join(OUTDIR, 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    log('DONE smoke=%s res_sha8=%s%s' % (SMOKE, res8, (' seal_sha8=' + seal8) if not SMOKE else ''))
    return res8

if __name__ == '__main__':
    run()
