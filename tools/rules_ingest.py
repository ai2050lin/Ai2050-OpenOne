"""把已经做过的测试登记成规则账本（幂等：按 id upsert，重复跑不会产生重复条目）。

每条规则都来自实测，数字与 tempdata / design 里的记录一致；evidence 里写明脚本与数据文件，
方便回溯。retest 字段是这套系统的核心用途：
  block  —— 已有足够证据，口径不变时再测属于重复劳动
  need_n —— 方向已知但样本不足，重测必须上到指定题量才可能改变判定
  allow  —— 口径变了（换模型 / 换指标 / 换干预方式）可以重测
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
MODEL = 'Qwen3-4B'

R = []


def add(rid, domain, topic, claim, status, confidence, criterion, measure,
        evidence, retest, why, tags=(), keywords=None, supersedes=None):
    R.append({
        'id': rid,
        'domain': domain,
        'topic': topic,
        'claim': claim,
        'status': status,
        'confidence': confidence,
        'criterion': criterion,
        'measure': measure,
        'evidence': evidence,
        'retest': retest or {},
        'why': why,
        'tags': tags,
        'keywords': keywords or [],
        'supersedes': supersedes or [],
    })


def ev(phase, script, data, note):
    return {'phase': phase, 'script': script, 'data': data, 'model': MODEL, 'note': note}


BLOCK = lambda n, note: {'status': 'block', 'count': n, 'note': note}
NEEDN = lambda n, need, note: {'status': 'need_n', 'count': n, 'need_n': need, 'note': note}
ALLOW = lambda n, note: {'status': 'allow', 'count': n, 'note': note}

# ---------------- MLP 单元 ----------------

add('R-MLP-01', 'mlp', '单元重要性排序',
    '一阶归因（attribution patching）选出的 top-1% MLP 单元，跨数据集是稳定的：在 C-Eval 上选出的集合拿到 CMMLU 上仍是同一批。',
    'confirmed', 'high',
    'top-97 重叠 ≥20（随机期望 0.97）且 |Δ| 排序 spearman ≥0.30',
    {'overlap': '65/97', 'spearman': 0.607, 'kl_ratio': 54, 'n_layer': 24},
    [ev('H1', 'tools/mech_holdout.py', 'tempdata/holdout_L24.json', 'C-Eval→CMMLU，两套 169 题抽样')],
    BLOCK(3, '两套抽样（科目配对 / 全科目随机）结果几乎相同（0.607 / 0.605），与科目构成无关'),
    '这是本项目唯一可跨数据集引用的单元级结论：稳定性来自模型权重本身，不是题目用词。')

add('R-MLP-02', 'mlp', '单元级必要性',
    '删掉一阶归因 top-1% 的 MLP 单元，会使生成任务的答案正确率显著下降（L34 已证实；L24 样本不足，未判定）。',
    'confirmed', 'high',
    '只在 base 本来就能解析出答案的题上比较（条件的 None 仍算错）：下降 ≥5 点，且 P vs 随机删除 的 McNemar p<0.05',
    {'acc_base': 0.733, 'acc_top1pct': 0.660, 'acc_random': 0.720, 'drop_pp': -7.3,
     'p_top_vs_random': 0.0074, 'p_top_vs_base': 0.0213, 'p_random_vs_base': 1.0, 'n': 150, 'layer': 34,
     'L24_n_valid': 35, 'L24_p': 0.5, 'L24_note': '严格口径仅 35 题，方向甚至 P 略高（0.943 vs 0.914），无法判定'},
    [ev('H5+更正', 'tools/mech_cot.py', 'tempdata/mech_cot_L34_n169.json', 'L34 全 169 题，按「base 有效」口径重算'),
     ev('H4', 'tools/mech_cot.py', 'tempdata/mech_cot_L24.json', 'L24 40 题（严格 35 题）P=rand p=0.500 ⇒ 样本不足，未判定')],
    BLOCK(5, 'L34 已定论（不需要重测）；L24 需要 169 题重测才能判定。口径见 R-MET-06'),
    '原判 refuted 是解析失败造成的假象：base 有 19 题没写出答案行（记作 None → 算错），'
    '删 top-97 把其中 18 题修好了（格式修复），于是 P 的正确率被抬高、base 被压低，两者看起来相等。'
    '只在 base 有效的 150 题上比：P 0.660 vs base 0.733（−7.3 点），随机删 0.720 与 base 无差异（p=1.0）。'
    'L24 那轮的 40 题严格子集只有 35 题，p=0.5，且方向相反（P 0.943 > base 0.914），'
    '⇒ **不能说 L24 不成立，只能说未判定**。是否「必要性取决于层集中度」需要 L24 上 169 题重测。')

add('R-MLP-03', 'mlp', '杠杆识别',
    '删掉 top-1% 单元会显著改变模型给出的答案（改答率显著高于删同量随机单元）。',
    'confirmed', 'high',
    '改答配对的 McNemar p<0.01，且改答率 ≥2× 随机对照',
    {'only_top_changed': 16, 'only_random_changed': 0, 'p': '<0.0001', 'rate_top': 0.132,
     'rate_random': 0.021, 'ratio': 6.3, 'n_valid': 144, 'raw_only_top': 28, 'raw_only_random': 4},
    [ev('H5+更正', 'tools/mech_cot_analyze.py', 'tempdata/mech_cot_L34_n169.json',
        'L34 169 题；严格口径 = 两侧都给出有效答案的 144 题')],
    BLOCK(1, '严格口径下 只P改 16 / 只rand改 0，效应量极大，不需要更多题量'),
    '这条是"杠杆真实存在"的最强行为证据。严格口径（排除解析失败题）下更强：0.132 vs 0.021（6.3×），'
    '且随机删除一次都没改答。原口径的 28/4 里混入了格式修复，见 R-MET-06。')

add('R-MLP-04', 'mlp', '编码方式',
    '高杠杆 MLP 单元是成对正反编码（差分对）：删掉它们时，破坏正确计算与修复错误计算同时发生，净效应接近零。',
    'refuted', 'high',
    '行为层的净效应当接近零（由错变对 ≈ 由对变错）；排除解析失败题后仍应如此；'
    '若真是差分对，拆开单删也应只剩一半后果、合删才有全部后果',
    {'orig_wrong_to_right': 11, 'orig_right_to_wrong': 14, 'orig_net': -3,
     'corrected_wrong_to_right': 3, 'corrected_right_to_wrong': 13, 'corrected_net': -10,
     'logit_abs_sum': 633.7, 'logit_net_sum': 16.1,
     'h6_split': {'pos45_delta_pt': -5.11, 'neg45_delta_pt': -4.38, 'both90_delta_pt': -5.11,
                  'pos_vs_neg_p': 1.0, 'pos_vs_both_p': 1.0}},
    [ev('H5+更正', 'tools/mech_cot_analyze.py', 'tempdata/mech_cot_L34_n169.json',
        '行为层：原口径 14 错/11 对；排除解析失败后 13 错/3 对（净 −10）'),
     ev('H6-严格', 'tools/mech_pair_sign_strict.py', 'tempdata/pair_sign_L34_strict.json',
        '拆开单删：两侧各掉 5 点左右且彼此无差异；合删 90 个不再多掉（饱和，非抵消）'),
     ev("C5'", 'tools/mech_unit_attr.py', 'tempdata/unit_attr_L24.npz', 'logit 层：Σ|Δ|=633.7 而 ΣΔ=16.1')],
    BLOCK(2, 'H5 从净效应否定、H6 从拆分实验否定，两条路径同一结论；别再用拆分重做'),
    '「双向抵消」不成立：排除解析失败题后净效应是 −10（不是 −3），P 显著差于随机删除。'
    'H6 进一步否掉差分对的形状要求：差分对必须"删一半只有一半后果"，实测删 45 个就掉 5 点、'
    '删 90 个还是 5 点（饱和）。logit 层的 ΣΔ 很小（16.1）仍然成立，但那是 KL/长尾口径的性质（见 R-MET-01），'
    '不能外推到行为层。⇒ 符号没有功能意义，「冗余饱和池」（R-MLP-16）才是这批单元的真实形状。')

add('R-MLP-05', 'mlp', '模块结构',
    '按共现把 MLP 单元聚成组，得到的组是"参数共享的功能子模块"（组内写方向应当一致）。',
    'refuted', 'high',
    '组内写方向平均 |cos| 显著高于零模型（同规模随机单元对）',
    {'cos_within': 0.017, 'cos_null': 0.0198, 'lift': 1.0, 'jaccard_lift': '4.66–14.45'},
    [ev('A2', 'server/mech_pair_service.py', 'data/mech/unit_clusters.json', '5 层 k=24 聚类 + 写方向亲和')],
    BLOCK(2, '共现结构真实（Jaccard lift 5–14×），但组内方向与随机不可分，结论明确'),
    '共现分组是"语境模块"（同类文本里一起被激活），不是"计算模块"（共享参数、协同工作）。'
    '这是整个 C 段路线失败的直接原因。')

add('R-MLP-06', 'mlp', '模块结构',
    '共现聚类在换随机种子 / 换样本后能稳定复现。',
    'refuted', 'high',
    '换初值或重采样 80% 样本后，两次聚类的 Jaccard ≥0.6',
    {'jaccard_seed': [0.434, 0.414, 0.455], 'jaccard_resample80': [0.187, 0.282, 0.325]},
    [ev("C2'", 'tools/mech_cluster_stability.py', 'tempdata/cluster_stability.json', 'KMeans 换初值 × 3 / 窗口重采样 80% × 3')],
    BLOCK(1, '不足 0.5 的复现度已经足以否定"稳定模块"'),
    '簇是采样噪声的产物；任何以簇为单位的因果结论都不成立。')

add('R-MLP-07', 'mlp', '模块结构',
    '共现聚类的簇对应可命名的语义类别（token 类型 / 语种 / 标点）。',
    'refuted', 'high',
    'AMI(簇, 类别) 显著高于零（≥0.3）',
    {'ami_char': '0.045–0.184', 'ami_language': '0.041–0.144', 'punct_ratio': '0.115 vs 0.093'},
    [ev("C2'", 'tools/mech_token_types.py', 'tempdata/cluster_token_types.json', '簇 × 字符类 / 语种 的互信息'),
     ev("C2'", 'tools/mech_cluster_axis.py', 'tempdata/cluster_axis.json', '簇的读出词构成')],
    BLOCK(1, 'AMI 接近零基线，且 C19 的标点富集只有 1.24×，已撤回上轮"排版模块"的判断'),
    '上轮把 C19 判成"排版模块"是采样偏差（全层真正最偏标点的是 C8）。教训：簇不承载语义。')

add('R-MLP-08', 'mlp', '层结构',
    '层间差异可以用"集中度"（top-1% 占 Σ|Δ| 的比例）量化，且集中度与跨数据集稳定性严格同向。',
    'confirmed', 'high',
    'share_top1pct（top-1% 占 Σ|Δ|）三层单调，且与 top-97 重叠数、spearman 同向；'
    '注意 share 与 kl_ratio（删 1% 的 KL / 删整层的 KL）是两个不同的量，不可互替',
    {'share_top1pct_L14': 0.079, 'share_top1pct_L24': 0.147, 'share_top1pct_L34': 0.370,
     'kl_ratio_L14': 0.086, 'kl_ratio_L24': 1.000, 'kl_ratio_L34': 1.115,
     'overlap_L14': 37, 'overlap_L24': 65, 'overlap_L34': 81,
     'spearman_L14': 0.150, 'spearman_L24': 0.607, 'spearman_L34': 0.761},
    [ev('H1b/H1c', 'tools/mech_holdout.py', 'tempdata/holdout_L14.json', '同脚本 LAYER=14/24/34'),
     ev("C4'", 'tools/mech_unit_attr.py', 'tempdata/unit_attr_L14.npz', '四层单元归因')],
    BLOCK(3, '三层各跑一次，单调关系清楚'),
    '这是唯一一个"可预筛层"的量化指标：先测集中度，再决定值不值得做单元级分析。')

add('R-MLP-09', 'mlp', '层结构',
    'L30 是沉默层（删掉整层对答案几乎无影响），不是"分布式层"。',
    'confirmed', 'high',
    '整层消融 KL <0.02 且翻转数不高于随机',
    {'full_kl_L30': 0.0128, 'conc_L30': 0.176, 'compare_L24_kl': 0.1708},
    [ev("C4'", 'tools/mech_unit_attr.py', 'tempdata/unit_attr_L30.npz', 'L30 60 题')],
    BLOCK(2, 'KL 比 L24 低一个数量级'),
    '上轮把 L30 读成"分布式"是错的：集中度与 L24 同级，但整层效应量小 13 倍。'
    '四层结构是三状态（分布式 / 杠杆层 / 沉默层），不是一条曲线。')

add('R-MLP-10', 'mlp', '重要性预测',
    '用权重几何（列范数 / 能量 / 有效秩）可以预测 MLP 单元的重要性排序。',
    'refuted', 'high',
    '与 |t| 的相关系数 ≥0.6',
    {'corr_col_norm': 0.174, 'corr_ov_pr': 0.026, 'corr_w_letter_max': 0.349},
    [ev('A1', 'server/mech_service.py', 'data/mech/heads_dir.json', 'OV 写方向 / QK 读方向 / 有效秩')],
    BLOCK(2, '三种几何量都低相关，且有反例（L24H31 能量 0.042 但 t=14.6；L35H24 能量 0.274 但 t=−15.2）'),
    '看权重猜不出谁重要，只能实测。这条否证了"剪枝按范数"的常见做法在这个模型上的有效性。')

add('R-MLP-11', 'mlp', '重要性预测',
    '用"读取增益"（下游 W_K/W_V 能读到多少）可以预测 MLP 单元的重要性。',
    'refuted', 'high',
    'corr(增益, |t|) ≥0.5 且逐对稳定',
    {'corr_mean': 0.226, 'corr_median': 0.138, 'pairs': 41},
    [ev("G'", 'server/mech_pair_service.py', 'data/mech/pair_gain.json', '5 层 × 12 配对，256 随机向量标准化')],
    BLOCK(1, '预注册 kill criterion 命中，G 环节已从流水线移除'),
    '这条失败后流水线由 W→R→G→C→P 简化为 W→R→C→P，省掉一整个环节。')

add('R-MLP-12', 'mlp', '读出结构',
    '集中式晚期层（L34）的 top-97 中存在可直接推出选项字母的单元，且相对全层显著富集。',
    'confirmed', 'medium',
    'top-97 中 readout#1 为字母的单元数 ≥3，且富集 ≥5×',
    {'n_letter_units': 3, 'share_of_effect': 0.089, 'baseline': '20/9728=0.21%', 'enrichment': 15, 'L14_L24_count': 0},
    [ev('H4附录', 'tools/mech_top_units.py', 'tempdata/top_units_L34.md', '普查 readout_top × 单元归因交集')],
    BLOCK(1, '普查数据是静态的，重算不会变；但只覆盖 9.7% 效应量，不能推广成"整层=字母单元"'),
    '第三条独立证据（不依赖 KL、不依赖行为）。同时说明其余 94 个单元的 readout 词仍无族——'
    '"例外正是那几个推字母的"。')

add('R-MLP-13', 'mlp', '必要性',
    '删掉整层 9728 个 MLP 单元会翻转大量答案。',
    'refuted', 'high',
    '翻转数显著高于随机（≥20/169）且答案分布的熵明显上升',
    {'flip': '5/169', 'd_margin': 0.020, 'entropy': '0.621→0.395', 'top1_still_letter': 1.00},
    [ev("C5'", 'tools/mech_degrade_check.py', 'tempdata/degrade_check_L24.json', '30 题退化检验'),
     ev('H5', 'tools/mech_cot.py', 'tempdata/mech_cot_L34_n169.json', 'L34 169 题：整层 acc 0.800 vs base 0.850')],
    BLOCK(3, '三种读数（翻转 / margin / 熵）一致指向"没崩，反而更确定"'),
    '回答不是"层不重要"，而是层的输出被下游层冗余补偿；同时 KL 指标测的是长尾（见 R-MET-01）。')

add('R-MLP-14', 'mlp', '符号结构',
    '高杠杆单元的符号有功能意义：只删 Δ>0 的一侧与只删 Δ<0 的一侧，行为效应方向相反。',
    'refuted', 'high',
    '数量与幅度双匹配（各 45 个，Σ|Δ| 48.77 / 45.54）后，net(pos) 与 net(neg) 符号相反且 |net|≥5；严格口径（排除解析失败题）',
    {'n_strict': 137, 'base_acc': 0.752,
     'pos': {'acc': 0.701, 'delta_pt': -5.11, 'net': -7, 'flip': '8/1', 'p_vs_base': 0.0391},
     'neg': {'acc': 0.708, 'delta_pt': -4.38, 'net': -6, 'flip': '9/3', 'p_vs_base': 0.1460},
     'pos_vs_neg_p': 1.0, 'rand97_delta_pt': 0.73},
    [ev('H6', 'tools/mech_pair_sign.py', 'tempdata/pair_sign_L34.json',
        'L34 169 题，base/pos/neg/both 四条件，89 分钟，base 复现 169/169'),
     ev('H6-严格', 'tools/mech_pair_sign_strict.py', 'tempdata/pair_sign_L34_strict.json',
        '严格子集 n=137；两侧都掉分且彼此无差异（p=1.0000）')],
    BLOCK(4, '已直接测完。符号方向不成立，别再用「正/负侧」切分重做'),
    '两侧都掉分（−5.11 / −4.38 点）、彼此无差异（p=1.0000），没有"方向相反"。'
    '符号是数值属性，不是功能单位——继续在正负号上找区分是死路。'
    '脚本自带 verdict 曾报「无向抖动」，那是 None 记错造成的二次误读（见 R-MET-07）。')

add('R-MLP-15', 'mlp', '符号结构',
    '同一侧单元的 ×0 与 ×2 应把正确率推向 base 的两侧（放大与删除方向相反）。',
    'open', 'medium',
    'acc(amp_neg) < acc(base) < acc(amp_pos) 且两端差 ≥5 点；严格口径',
    {'amp': 2.0, 'n_units_each': 45, 'state': 'queued'},
    [ev('H7', 'tools/mech_ampl.py', 'tempdata/ampl_L34.json', '复用 H6 的单元表，避免两套定义')],
    ALLOW(0, '排队中。H6 完成后跑'),
    '这一条不依赖「删除」这个操作本身（H5 已见删除会让输出变模板化），是对 H6 的独立复核。')

add('R-MLP-16', 'mlp', '冗余结构',
    'L34 的 top-1% 单元是一个冗余饱和的共享功能池：删池内任一半与删全部的行为损伤相当，而同层随机删除几乎无损。',
    'refuted', 'high',
    '强形式「任一半」需 top-40 也能打出 ≥3pt 损伤；弱形式「整类 vs 整池」只需 pos45≈both90；严格口径',
    {'n_strict': 137, 'base_acc': 0.752,
     'pos45_delta_pt': -5.11, 'neg45_delta_pt': -4.38, 'both90_delta_pt': -5.11,
     'rand97_delta_pt': 0.73,
     'h8_top40_delta_pt': -0.74, 'h8_top40_sum_abs_delta': 73.63, 'h8_top97_sum_abs_delta': 96.50,
     'h10_tailP_delta_pt': -0.72, 'h10_tailN_delta_pt': -5.04},
    [ev('H6-严格', 'tools/mech_pair_sign_strict.py', 'tempdata/pair_sign_L34_strict.json',
        '删 45 / 45 / 90 三档 vs 随机 97'),
     ev('H8-反证', 'tools/mech_dose.py', 'tempdata/dose_L34.json',
        'top-40 占归因总量 76% 却只有 −0.74pt'),
     ev('H10-反证', 'tools/mech_headtail.py', 'tempdata/headtail_L34.json',
        'tailP(27) −0.72pt 而 tailN(23) −5.04pt，单元数与 Σ|Δ| 相近却差 4 倍')],
    BLOCK(1, '「任一半就饱和」已被 H8 与 H10 两轮独立推翻；准确说法见 R-MLP-19（近二值池依赖）'),
    '强形式被推翻：删 40 个（含 pos45 的 18 个、neg45 的 22 个、占 top-97 归因量 76%）几乎无损。'
    '弱形式（把某一符号类删到接近干净）成立，但 H10 显示决定因素是「是否覆盖到承载单元」，'
    '不是数量也不是符号 → 改写为 R-MLP-19。')

add('R-MLP-17', 'mlp', '冗余结构',
    'L34 的 top-1% 池存在一个可测的饱和点：删到某个 k 之后的额外删除不再显著增加行为损伤。',
    'confirmed', 'medium',
    'Δ(k) 达到 Δ(97) 的 80%（−4.74 pt）时的最小 k 记 k80；k∈{10,20,40,60,80}，同 SEED 20261101，'
    'base 须与 H5/H6 逐题一致否则整轮作废',
    {'curve_pt': {'10': 0.0, '20': -1.48, '40': -0.74, '60': -3.60, '80': -5.04, '97': -5.93},
     'k80': 80, 'flat_below': 40, 'cliff': '40→80',
     'damaged_questions': {'10': 3, '20': 5, '40': 4, '60': 10, '80': 13, '97': 14},
     'n_strict': 135, 'base_acc': 0.7481},
    [ev('H8', 'tools/mech_dose.py', 'tempdata/dose_L34.json', 'k10/20/40：40 名之前全平'),
     ev('H10', 'tools/mech_headtail.py', 'tempdata/headtail_L34.json', 'k60/80：悬崖在 40–80 之间')],
    BLOCK(1, 'k∈{10,20,40,60,80} 已测完，曲线已覆盖 40–97 全段；再测中间点只是重复'),
    '剂量饱和确实存在但位置很高（k80 = 80/97），而且**题目级饱和远早于剂量级饱和**：'
    'tailN 的 23 个单元就打坏了 11/14 道会被打坏的题。'
    '所以「饱和」有两个不同的量，不可混用：删到 80 个才达到 80% 的准确率损伤，'
    '但删到 23 个就已经打坏了几乎全部会被打坏的题。')

add('R-MLP-18', 'mlp', '归因有效性',
    '一阶归因的 |Δ| 排名可以预测单元的行为必要性：|Δ| 越大的单元删掉越痛。',
    'refuted', 'high',
    '删 top-k（按 |Δ| 降序）的损伤应随 k 单调增强，且 top-40（占归因量 76%）必须打出 ≥3pt；同层随机删除作对照',
    {'top40_delta_pt': -0.74, 'top40_sum_abs_frac_of_top97': 0.763,
     'pos45_delta_pt': -5.11, 'tailP27_delta_pt': -0.72, 'tailN23_delta_pt': -5.04,
     'rand97_delta_pt': 0.73, 'top40_damaged_q': 4, 'tailN23_damaged_q': 11,
     'per_question_attr_auc': 'H11 三条判据全否'},
    [ev('H8', 'tools/mech_dose.py', 'tempdata/dose_L34.json', 'top-40 占归因量 76% 却 −0.74pt'),
     ev('H10', 'tools/mech_headtail.py', 'tempdata/headtail_L34.json', 'tailN 23 个 ≈ 整池 97 个'),
     ev('H11', 'tools/mech_frag_attr.py', 'tempdata/frag_attr_L34.json',
        '逐题归因区分不了脆弱题与对照题（重叠 26.75 vs 25.67）')],
    BLOCK(1, '三轮独立证据（H8 剂量、H10 头尾、H11 逐题）都指向同一结论：不要在池内按 |Δ| 排序'),
    'top-40 拿走 76% 的归因量却只打坏 4 道题；tailN 的 23 个占 9.43/96.5 = 9.8% 却打坏 11 道。'
    '而且逐题归因也不能预测哪道题会坏。⇒ 一阶归因只能用来选「池」，不能用来选「池内成员」，'
    '更不能用来预测单题依赖。后续所有「top-k」口径的结论必须改口径（整类删除 / 留一 / 承载集定位）。')

add('R-MLP-19', 'mlp', '冗余结构',
    'L34 top-1% 的必要性是近二值的池依赖：覆盖到承载单元就一次性打坏同一小批题，覆盖不到就是零损伤。',
    'confirmed', 'high',
    'base 答对的题里，被各子集打坏的应是同一批题（重叠高），且损伤不随子集大小线性增长',
    {'base_correct_q': 110, 'damaged_by_pool': 14, 'damaged_by_random97': 1,
     'damaged_by_all_8_conditions': [73, 5], 'damaged_by_ge6_conditions': 7,
     'fragile_subjects': {'chinese_civil_service_exam': 5, 'high_school_geography': 3,
                          'others': 4},
     'specificity_vs_random': '14x'},
    [ev('H5+H6+H8+H10 合并逐题', 'tools/mech_dose.py / mech_headtail.py',
        'tempdata/dose_L34.json + headtail_L34.json',
        '11 个条件的逐题结果并集；qi=73、5 被全部 8 个条件打坏')],
    BLOCK(1, '这一条已经把「哪些题会坏」定死（12–14 道），不要再按子集比较准确率百分点——分辨率只有几道题'),
    '这是本轮最重要的结构结论：不是「池内某类单元重要」，而是「池子里有一小批题依赖的承载单元，'
    '任何覆盖到它们的子集都会打坏同一批题」。'
    '⇒ 「−5.9 点」实际上是「12 道题」，所有子集比较的误差都以几道题计，'
    '所以 pos45 / neg45 / both90 / k60 / k80 之间的无差异是分辨率不足，不是真的等效。')

add('R-MLP-20', 'mlp', '归因有效性',
    '单题的一阶单元归因能预测该题在消融下会不会被弄坏（脆弱性可预测）。',
    'refuted', 'high',
    '脆弱题与对照题的归因画像必须有可测差异：与全局 top-97 重叠、top-20 集中度、承载集覆盖，'
    '任一项相对差 ≥20% 或重叠差 ≥10',
    {'overlap_global97': {'fragile': 26.75, 'control': 25.67},
     'conc_top20': {'fragile': 0.093, 'control': 0.091, 'rel': 1.022},
     'in_tailN': {'fragile': 3.67, 'control': 4.75},
     'in_tailP': {'fragile': 5.75, 'control': 5.667}},
    [ev('H11', 'tools/mech_frag_attr.py', 'tempdata/frag_attr_L34.json',
        '12 脆弱 + 12 对照，三条判据全不成立')],
    BLOCK(1, '已被直接测过并否证：再做「用归因预测哪道题会坏」就是重复'),
    '常见误差是「脆弱题 top-20 集中度 0.093 > 对照 0.091 所以更集中」——'
    '相对差 1.022 倍是噪声。U3 判据最初只写不等号，第一次跑就误判为成立（见 R-MET-09）。')

add('R-MLP-21', 'mlp', '冗余结构',
    'L34 tailN 池里存在可点名的「承载核」：[767, 1540, 35] 三个单元承担 tailN 损伤的 3/4，'
    '且损伤题集完全被 tailN 包含。',
    'confirmed', 'high',
    '三个单元（占 tailN Σ|Δ| 12.7%、全层排名 62/66/67）全量 169 题上打坏 9/110 道 base 对的题，'
    '且这 9 道全部 ⊆ tailN 的 12 道（C1 零例外）；归因量 2.2 倍的 core5 只打坏 5 道',
    {'core3': [767, 1540, 35], 'core3_sum_abs': 1.196, 'tailN_sum_abs': 9.43,
     'share_of_tailN': 0.127, 'layer_rank': [62, 66, 67],
     'n_base_right': 110, 'core3_damaged': 9, 'core5_damaged': 5, 'tailN_damaged': 12,
     'core3_delta_pt': -4.67, 'tailN_delta_pt': -6.67, 'core5_delta_pt': -3.33,
     'core3_not_in_tailN': 0, 'tailN_not_in_core3': [21, 25, 78],
     'same_107q_broken': {'core3': 8, 'core5': 3, 'tailN': 10, 'k97': 13, 'rand97': 3},
     'write_dir_pairwise_cos_max': 0.029,
     'p_values': {'core3_vs_base': 0.0654, 'core3_vs_rand97': 0.0703,
                  'core3_vs_core5': 0.3438, 'core3_vs_tailN': 0.25,
                  'tailN_vs_base': 0.0129}},
    [ev('H12', 'tools/mech_carrier.py', 'tempdata/carrier_L34.json',
        '自适应二分 7 个条件，17 题，判据 localized_3（打坏 7/17）'),
     ev('H13', 'tools/mech_core.py', 'tempdata/core_L34.json',
        '169 题 × 4 条件全量确认，C1–C5 全过，base 对齐 H8 150/150'),
     ev('H14-pre', 'tools/mech_core_wiring.py', 'tempdata/core_wiring_L34.json',
        '只读权重：三单元写方向两两 |cos| ≤ 0.029，最近邻均不在 tailN 内 ⇒ 不是副本'),
     ev('H5 参照', 'tools/mech_cot.py', 'tempdata/mech_cot_L34_n169.json',
        'k97 / rand97 拉进同一批 107 题做同口径对比')],
    BLOCK(2, 'H12（17 题）与 H13（169 题）两轮独立确认；结构结论已定，不要再按子集比准确率百分点'),
    'H11 已证「单元级归因不能预测单题脆弱性」，但那只否掉了「用归因画像挑单元」这条路径，'
    '没有否掉「承载核存在」。H12 换测法（自适应二分 + 少题高重复）后 3 个单元就复现了 7 道，'
    'H13 在全量上确认 9/12 且零例外。两条自然对照最能说明问题：Σ|Δ| 几乎相等（2.626 vs 2.588）'
    '的两半损伤 2 vs 7；同一父集合内 Σ|Δ| 更高的那一半反而更轻。'
    '**只能读结构不能读显著**：效应只有 9–10 道题，core3 vs base p=0.065、vs rand97 p=0.070、'
    'vs core5 p=0.344 都不过 0.05，唯一显著的是 tailN vs base（0.0129）。'
    '题池只有 169 道，这是分辨率的硬上限——不得把本条写成「显著比随机更伤」。')

add('R-MLP-22', 'mlp', '冗余结构',
    '脆弱题可按二分树分成五档「最小充分集」（3 / 5 / 11 / 23 个单元 / 打不坏）。',
    'refuted', 'high',
    '逐题记录「第一次被打坏时所在的最小子集大小」，若同一批题落在同一档则同质，'
    '若跨越 ≥3 档则非同质',
    {'tier_claim_17q': {'3': [5, 29, 41, 73, 94, 97, 125], '5': [78], '11': [25, 55],
                        '23': [21], 'none': [68]},
     'single_unit_767_damaged_17q': [5, 25, 29, 41, 73, 78, 94, 97, 125],
     'counterexamples': {'qi=78': 'H12 归到 A′(5) 档，但 767 单点即可打坏',
                         'qi=25': 'H12 归到「需 A(11)」档，但 767 单点即可打坏'},
     'core3_subset_of_767': True},
    [ev('H12', 'tools/mech_carrier.py', 'tempdata/carrier_L34.json',
        '二分树逐轮 damaged 列表（分档的来源）'),
     ev('H15', 'tools/mech_core_split.py', 'tempdata/core_split_L34.json',
        '补测单点：767 单点打坏 9 道 ⊇ core3 的 7 道，含被归到 A′/11 档的 78、25')],
    BLOCK(1, '已被 H15 直接否证；不要再引用「五档最小充分集」'),
    '错在方法：二分只回答「这条路径上哪一半有用」，不回答「最小集合是哪个」。'
    '而且 H12 的 while len(cur) > 3 直接封死了单点，等于预设了答案的下限。'
    'qi=78 被 A′(5) 打坏也被 767 单点打坏、qi=25 被 A(11) 打坏也被 767 单点打坏 ⇒ '
    '「档」是路径的产物。正确表述见 R-MET-10（分档必须逐单点测）和 R-MLP-23（单点承载）。')

add('R-MLP-23', 'mlp', '冗余结构',
    'L34 的单个单元 767 打坏 11/110 道 base 对的题（保留率 90.0%），与 core3 三个一起（9 道）、'
    'tailN 23 个（12 道）同量级；但它既不包含 core3 的全部损伤，也不满足预注册的随机对照阈值。',
    'confirmed', 'medium',
    '单点在全量 110 道 base 对的题上打坏 ≥7 道（O1）；O2 用 10 个随机单点在 17 题上做对照；'
    'O3 要求单点题集包含 core3 的 9 道；O4 要求不劣于 core3',
    {'single': 767, 'core3': [767, 1540, 35],
     'n_base_ok': 110, 'reserve_rate': {'s767': 0.9, 'core3': 0.918, 'core5': 0.955, 'tailN': 0.891},
     'damaged_full': {'s767': 11, 'core3': 9, 'core5': 5, 'tailN': 12},
     's767_set': [5, 11, 25, 29, 41, 61, 73, 78, 94, 97, 125],
     'core3_set': [5, 29, 41, 61, 73, 94, 97, 125, 147],
     'overlap': 8, 's767_only': [11, 25, 78], 'core3_only': [147],
     'O1': True, 'O2': False, 'O3': False, 'O4': True,
     'change_rate_150': {'s767': 0.107, 'core3': 0.093, 'tailN': 0.147, 'k97': 0.14, 'rand97': 0.06},
     'no_letter_bias': {'A': 3, 'B': 1, 'C': 5, 'D': 4},
     'caliber_fix': 'H13 的 delta_pt 是净效应/150，H17 是毛损伤/110，两者不可相减；'
                    '本条的保留率统一按 110 道 base 对的题算'},
    [ev('H15', 'tools/mech_core_split.py', 'tempdata/core_split_L34.json',
        '7 条件 × 17 题：s767 打坏 9，core3 打坏 7'),
     ev('H17', 'tools/mech_core_one.py', 'tempdata/core_one_L34.json',
        '169 题全量 + 10 个随机单点：O1/O4 过，O2/O3 不过；base 对齐 H13 169/169'),
     ev('H18', 'tools/mech_logit_effect.py', 'tempdata/logit_effect_L34.json',
        '答案位置 logit 差分：推高换行族 9–10.7，机制的定性见 R-MLP-25')],
    BLOCK(3, 'H15（17 题）+ H17（169 题）+ H18（机制）三轮一致；不要再按子集比准确率百分点，'
             '也不要写成「单点比三合一更狠」——差 2 道题在 n=110 上没有分辨力'),
    '「1 ≈ 3 ≈ 23」这个量级关系是两轮独立测出来的，但它只说明**打坏几道题**这件事与单元数无关，'
    '不说明 767 是「知识承载单元」——H18 把机制指向格式（见 R-MLP-25）。'
    'O2 判否的原因不是特异性不存在，而是判据用「10 次抽样的最大值」当零分布标尺，'
    '被一道通用脆性题（qi=73）抬高了一个（见 R-MLP-24 与 R-MET-11）。'
    '按身份比：767 的 9 道里有 6 道是 10 个随机单点一次都没碰到的。')

add('R-MLP-24', 'mlp', '冗余结构',
    '存在与「删哪个单元」无关的通用脆性题：qi=73 被 10 个随机单点里的 8 个弄坏，'
    '且两种模态都有（真改答 / 解析崩）。',
    'confirmed', 'high',
    '同一批题上，随机单点损伤的并集若集中在极少数几道上，且这些题同时被主条件与对照组打坏，'
    '则判为 pool-generic',
    {'qi': 73, 'gold': 'A', 'base': 'A',
     'random_single_hits': {'n_units': 10, 'n_hit_73': 8,
                            'union_all': [25, 73, 78], 'counts': {'73': 8, '25': 3, '78': 1}},
     'modalities': {'s767': 'no_answer_line（解析失败）',
                    'r2098': 'answer_C（真改答，gold 是 A）',
                    'r7964': 'no_answer_line（解析失败）'},
     's767_6_exclusive_of_null': [5, 29, 41, 94, 97, 125]},
    [ev('H17', 'tools/mech_core_one.py', 'tempdata/core_one_L34.json',
        '10 个随机单点 × 17 题的损伤清单'),
     ev('H17-probe', 'tools/mech_probe_frag.py', 'tempdata/probe_frag_L34.json',
        '三个单元各跑一次 qi=73/25/78，记录 pred 与原文尾部')],
    BLOCK(2, '两级证据：计数（8/10）+ 原文（真改答与解析崩各一例）。'
             '引用「12 道脆弱题」时必须区分 pool-generic 与 pool-specific'),
    '这一条修正 R-MLP-19/22 的名单口径：旧的「12 道脆弱题」是八个条件的并集，'
    '里面混着至少一道与单元无关的通用脆性题。'
    '后果很实际——用计数判特异性时，通用脆性题会让主条件与随机对照各加一，差别被压缩。')

add('R-MLP-25', 'mlp', '单元语义',
    '单元 767 是格式承载单元而不是知识单元：它在「分析：」之后抑制换行，消融后模型另起一段重写分析，'
    '答案随之漂移；字母层面的判断几乎不受它影响。',
    'confirmed', 'high',
    '三条同时成立：① 答案位置换行族 logit 抬高 ≥5 且 11 道题推同一批 token；'
    '② 固定前缀（teacher forcing）下字母 argmax 不变；'
    '③ 自由生成时首 token 变空白/换行 ≥5 道且格式崩坏的题答案随之改变',
    {'q': [5, 11, 25, 29, 41, 61, 73, 78, 94, 97, 125],
     'H18_promoted': {'\n': [9.06, 10.69], '\r': [4.75, 5.63], '\r\n': [3.28, 3.81]},
     'H18_top5_jaccard': 1.0, 'H18_letter_delta_max': 0.125,
     'H18_abs_median_delta': {'s767': 0.125, 'rand_max': 0.0156, 'ratio': 8.0},
     'H19_fixed_prefix': {'same_argmax': '9/9', 'kl_median': 0.0,
                          'non_saturated': {'qi=29': 0.0021, 'qi=94': 0.0019},
                          'caveat': '7/9 道 p=1.000 饱和 ⇒ 单独不能下结论'},
     'H20_prefix_sweep': {'argmax_changed': {'25%': 0, '50%': 0, '75%': 0, '100%': 1},
                          'margin_shift_median_25pct': 0.031, 'margin_shift_max': 0.12,
                          'letter_prob_median_25pct': 0.584,
                          'gold_gap_range_25pct': [-5.2, 1.7]},
     'H21_free_gen': {'first_tok_ws': {'base': '1/11', 'ablated': '8/11'},
                      'format_broken': 7, 'ans_changed_among_broken': 7,
                      'non_newline_but_changed': [41, 61, 73]},
     'design_waste': 'H19 第一版强制「 答案：」前缀不中性（qi=5 base 由 D 变 C，因为删掉了推理），'
                     '第一个样本即废弃，代价 75 秒'},
    [ev('H18', 'tools/mech_logit_effect.py', 'tempdata/logit_effect_L34.json',
        '11 道坏题 × 5 条件前向 logit 差分'),
     ev('H19', 'tools/mech_format_test.py', 'tempdata/format_test_L34.json',
        '固定 base 前缀的 teacher forcing；v1 废弃原因记在 design_note'),
     ev('H20', 'tools/mech_prefix_scan.py', 'tempdata/prefix_scan_L34.json',
        '25/50/75/100% 四档前缀，读出非饱和'),
     ev('H21', 'tools/mech_head_tokens.py', 'tempdata/head_tokens_L34.json',
        '自由生成的首 3 token + 重复度 + 解析结果')],
    BLOCK(4, 'H18 机制 + H19 固定前缀 + H20 前缀扫描 + H21 行为闭环，四轮一致；'
             '不要再写「767 承载 10 道题的知识」，要写「767 维持答题格式」'),
    '这条改写了 H12–H17 的定性：单元级必要性真实存在、高度特异，但机制是格式/轨迹。'
    'H20 是四条里最硬的一条——25% 前缀上读出不再饱和（字母概率中位 0.58、gold 间隔 −5.2…+1.7），'
    '而 767 对间隔的位移上界只有 0.12 个 logit，与换行的 9–10.7 差两个数量级。'
    'H19 单独看是弱证据（7/9 饱和），必须配 H20 才能下结论——这一点在文档里已写明。'
    '下一步见 R-MET-12：用格式不变读出重做池级必要性。')

add('R-MET-01', 'metric', '重要性度量',
    'KL 散度可以作为 MLP 单元「重要性 / 必要性」的度量。',
    'refuted', 'high',
    'top-k 的扰动能量应集中在高概率 token 上（energy_share_top20 ≥3× 均匀基线）',
    {'energy_share_top20': 0.0007, 'uniform_baseline': 0.00013, 'ratio': 5.4,
     'letter_share': 0.0002},
    [ev('H3b', 'tools/mech_delta_eff.py', 'tempdata/mech_delta_eff.json',
        'KL 与答案位能量分解')],
    BLOCK(2, '不要重测。KL 只能作长尾抖动的描述量，不能作必要性判据'),
    'KL 测的是 logit 长尾的抖动幅度，不是答案决策的因果贡献。'
    '用 KL 排序会把「让输出变毛糙的单元」误判成「重要单元」。这是本线最早也最重要的一条方法论否证。')

add('R-MET-02', 'metric', '对照组设计',
    '删掉「同规模的随机单元」是匹配的弱对照，可以用来和删 top-k 对比。',
    'refuted', 'high',
    '随机对照的扰动规模应与实验组同量级（中心化范数比 0.5–2）',
    {'norm_ratio_raw': 8.16, 'norm_ratio_centered': 0.13, 'kl_random': 0.00089},
    [ev('H3', 'tools/mech_delta_cos.py', 'tempdata/mech_delta_cos.json', '随机对照的范数与 KL')],
    BLOCK(1, '不要重测；随机对照必须报中心化范数比，否则「没影响」是无效结论'),
    '删除小批随机单元约等于给 logits 加一个常数，softmax 完全不动。'
    '所以「删随机 97 没影响」不能证明「被删的随机单元不重要」，只能证明这个对照太弱。')

add('R-MET-03', 'metric', '对照组设计',
    '未中心化的 Δlogits 范数可以直接用来比较不同干预的扰动规模。',
    'refuted', 'high',
    '两种干预的范数比与它们的 KL 比同向',
    {'raw_ratio_top1pct': 1.73, 'raw_ratio_random': 8.16, 'kl_ratio': 54},
    [ev('H3', 'tools/mech_delta_cos.py', 'tempdata/mech_delta_cos.json', '两套范数口径对比')],
    BLOCK(1, '不要重测；比较扰动规模前必须投影到与输出均值正交的子空间'),
    '必须先把 Δlogits 投影到与输出均值正交的子空间再比范数，否则常数平移完全污染比较。')

add('R-MET-04', 'metric', '样本量',
    'n=40 的生成任务足以判定「删 top-k 造成 7.5 个百分点的下降」。',
    'refuted', 'high',
    'McNemar 配对检验 p<0.05（不能只看点估计差）',
    {'p_at_n40': 0.453, 'discordant': '5:2', 'need_n': 684, 'effect_disappeared_at_n169': True},
    [ev('H4→H5', 'tools/mech_cot_analyze.py', 'tempdata/mech_cot_L34_n169.json',
        '40 题与 169 题两轮对照')],
    BLOCK(2, '不要重测；任何「掉几个点」的结论必须给 p 与 discordant 计数'),
    'n=40 时 7.5 点的差距 p 只有 0.453。need_n=684 是错误口径下估的，实为 120–150。')

add('R-MET-05', 'metric', '效应结构',
    '分组消融的效应近似可加（各组 KL 之和 ≈ 整层 KL）。',
    'confirmed', 'high',
    '各组之和 / 整层的比值 ∈[0.7, 1.2]',
    {'sum_groups': 0.1474, 'full_layer': 0.1708, 'ratio': 0.86, 'n_groups': 24},
    [ev('C1', 'tools/mech_c1_full_run.py', 'data/mech/ablate_L24.json', '24 组分组消融')],
    BLOCK(1, '不要重测'),
    '加性成立说明层内没有强协同，但也说明「用组做单位」不会放大效应——'
    '真正的问题是组本身不对应功能（见 R-MLP-05）。')

add('R-MET-06', 'metric', '口径',
    '生成类实验的解析失败可以按统一口径合并处理，正确率与改答率可以共用同一分母。',
    'refuted', 'high',
    '必须分别报告：① 解析失败数；② 正确率用「base 有效」作分母；③ 改答率用「两侧都有效」作分母',
    {'restored': True,
     'why': '解析失败在不同条件下数量不同（本线实测 base 19 / core3 7 / k97 1 / rand97 19，169 题），'
            '把它计入或剔除会改变分母与结论方向',
     'example': 'H5 的「P = rand」就是被解析失败污染出来的',
     'note': '本条在一次账本脚本误覆盖后按当日工作日志重建，只保留了日志里记录的数字'},
    [ev('H5→H6', 'tools/mech_pair_sign_strict.py', 'tempdata/pair_sign_L34_strict.json',
        '严格口径分析器：三档分母分别报')],
    BLOCK(5, '不要重测；这是本线最危险的坑（同一份数据两种口径结论完全相反）'),
    '写新脚本时「正确率用哪个分母」必须显式写在 docstring 里，并在输出里同时给两种口径的数。')

add('R-MET-07', 'metric', '判据实现',
    '脚本内置的 verdict 可以按与最终报告不同的口径计算。',
    'refuted', 'high',
    '判据的计算口径必须与结论口径同源；脚本内置 verdict 要与独立复核脚本逐项对齐',
    {'restored': True,
     'case': 'H6 的 mech_pair_sign.py 把 None 记错：net(neg) 从 −6 抹成 0、net(both) 从 −8 抹成 −1，'
             '于是读出「S3 抵消成立 / 杠杆是无向抖动」',
     'fix': '另写严格口径分析器重算，结论翻转为「差分对否证，但特异性极强」',
     'note': '本条在一次账本脚本误覆盖后按当日工作日志重建'},
    [ev('H6', 'tools/mech_pair_sign_strict.py', 'tempdata/pair_sign_L34_strict.json',
        '严格口径重算 + 同批 rand97 对照')],
    BLOCK(1, '不要重测；新实验先写严格口径分析器，再反校脚本内置 verdict'),
    '这是「判据实现与结论口径不同源」的第一个案例，后来在 H8/H11/H17 反复出现（见 R-MET-08/09/11）。')

add('R-MET-10', 'metric', '实验设计',
    '自适应二分的切分路径可以直接当作「最小充分集」的判定结果使用。',
    'refuted', 'high',
    '补测单点与成对：若单点即可复现被归到「多单元档」的题，则二分路径给出的档位无效',
    {'bisection_tier': {'qi=78': 'A′(5) 档', 'qi=25': '需 A(11) 档'},
     'single_unit_result': {'767 单点': '同时打坏 78 和 25'},
     'core3_damaged_7q': '完全被 767 单点的 9 道包含',
     'lower_bound_set_by_code': 'while len(cur) > 3'},
    [ev('H12→H15', 'tools/mech_carrier.py → mech_core_split.py',
        'tempdata/carrier_L34.json + core_split_L34.json',
        '二分树的分档 vs 逐单点补测')],
    BLOCK(1, '已直接否证；任何「最小集合」结论都要逐单点/逐对补测，不能引用二分路径'),
    '二分只回答「这条路径上哪一半还有用」，它优化的是轮数不是集合最小性；'
    '再叠一个 while len(cur) > 3 的下限，等于把答案的下限写进了代码。'
    '这次是便宜的一课（17 题 × 7 条件 15 分钟就把三合一推翻了）。'
    '**以后凡是「找到了核心集合」，必须补跑其所有真子集**，否则「最小」二字无依据。')

add('R-MET-11', 'metric', '实验设计',
    '用「N 个随机样本的最大值 ≤ 阈值」判特异性。',
    'refuted', 'high',
    '同一批评测上同时报两件事：数量（主条件 vs 随机组的最大/分布）与身份（随机组损伤并集、'
    '主条件的独有题集）。只报数量即判该判据不合格',
    {'pre_registered': 'rand_max <= 1（10 个随机单点在 17 题上）',
     'observed': {'rand_max': 2, 'rand_counts': [1, 1, 2, 2, 2, 2, 1, 0, 1, 0],
                  'rand_union': [25, 73, 78], 's767': 9,
                  's767_exclusive_of_union': [5, 29, 41, 94, 97, 125]},
     'why_failed': '随机组最大值被一道通用脆性题（qi=73，8/10 个随机单元都能弄坏）抬高了一，'
                   '阈值 1 因此被击穿；而按身份看随机组的并集只有 3 道题、767 有 6 道在其外',
     'same_family': ['R-MET-07（判据读法错）', 'R-MET-09（相对差阈值缺失）']},
    [ev('H17', 'tools/mech_core_one.py', 'tempdata/core_one_L34.json',
        '预注册 O2 判否，但同一份数据按身份重读给出相反结论')],
    BLOCK(1, '已直接否证；凡是「随机对照」必须同时报并集与差集，否则「特异性」二字无依据'),
    '这是本线第五次「判据实现与结论口径不同源」：判据写的是一个能被单点异常值击穿的阈值，'
    '结论要的是「主条件命中了随机对照命不中的目标」。'
    '更一般的检查项：**任何判据都必须先问「它在什么情况下会给出另一种结果」**，'
    '给不出两种结果的判据（如 H16 的 A2、H14-pre 的 W4）一票作废。')

add('R-MET-12', 'metric', '实验设计',
    '用「答对/答错」当因变量测单元（或单元集合）的必要性。',
    'refuted', 'high',
    '内容必要性必须用格式不变的读出复测：固定 base 自已写好的前缀，测答案位置的 gold 间隔被压缩多少；'
    '只报答案对错不算测到内容',
    {'what_went_wrong': 'L34 上被测得「最有必要」的单点（767）实为格式单元：'
                        '它对答案位置换行族 logit 的影响是 9–10.7，对字母间隔只有 0.12',
     'why_systematic': '格式是全模型瓶颈、只需极少数单元；知识是分布的 ⇒ '
                       '以答案对错为因变量时，格式单元会被系统性排到第一位',
     'evidence': {'H18': 'newline +9~10.7 vs letter ±0.12',
                  'H19': 'fixed prefix 9/9 字母不变',
                  'H20': '四档前缀上间隔位移 ≤0.12',
                  'H21': '首 token 空白 1/11 → 8/11，格式崩的 7 道答案全变'}},
    [ev('H18–H21', 'mech_logit_effect / mech_format_test / mech_prefix_scan / mech_head_tokens',
        'tempdata/logit_effect_L34.json + format_test + prefix_scan + head_tokens',
        '四轮递进：logit 位移 → 固定前缀 → 前缀扫描 → 自由生成首 token')],
    BLOCK(1, '已直接否证；任何「某单元承载某概念」的结论必须先用格式不变读出复核'),
    '这是本线代价最大的一次方法学错误：H5–H17 十三个实验都用「答对/答错」当因变量，'
    '方向没错（行为是真因变量），但**它不能区分格式与内容**。'
    '修法不是换掉行为指标，而是加一个格式不变的对照读出：同一批题、同一前缀、只比字母间隔。'
    'H22 就是按这个读出重做池级必要性。')

add('R-MET-08', 'metric', '判据实现',
    '剂量曲线类实验里，若没有任何剂量触发预设阈值，可以据此判定曲线形状（例如「接近线性」）。',
    'refuted', 'high',
    '阈值未触发时必须报「未定位 + 相邻两点实测值」，不得输出形状结论；形状判定要求实测点覆盖 ≥3 个量级或至少两个超过 80% 基准',
    {'case': 'H8 mech_dose.py',
     'false_reading': '饱和点高于 40 → 曲线接近线性',
     'actual': {'k40': -0.74, 'k97': -5.93, 'linear_pred_at_k40': -2.4},
     'cause': 'k80 为 None 时脚本走 else 分支直接打印「接近线性」'},
    [ev('H8', 'tools/mech_dose.py', 'tempdata/dose_L34.json',
        '实测 40 名 −0.74、97 名 −5.93；真线性该在 40 名给出 −2.4，差 3.3 倍')],
    BLOCK(1, '这是项目里第三次「脚本读法 ≠ 数据形状」（H5 分母、H6 verdict、H8 形状）'),
    '阈值未触发只说明「不在已测区间内」，完全不能推出形状。'
    '要判形状必须实测覆盖：H8 只测到 40，而悬崖在 40–97 之间，任何结论都越界。'
    '⇒ 曲线实验必须至少测到基准的 60–80%，否则只说「未定位」，并给出相邻两点让它可见。')

add('R-MET-09', 'metric', '判据实现',
    '判据里写清方向（例如「A 的集中度 > B 的集中度」）就够了，不必给效应量阈值。',
    'refuted', 'high',
    '每条数值判据都必须带效应量阈值或相对差下限（≥20% / ≥10 个百分点 / p<0.05），'
    '并同时输出实测值与阈值，避免 0.093 vs 0.091 这种噪声差被判成立',
    {'case': 'H11 U3',
     'v1': {'fragile': 0.093, 'control': 0.091, 'rule': 'fragile > control', 'verdict': 'true'},
     'v2': {'rule': 'fragile >= 1.2 * control', 'verdict': 'false'},
     'false_reading': '脆弱题更集中'},
    [ev('H11', 'tools/mech_frag_attr.py', 'tempdata/frag_attr_L34.json',
        '相对差 1.022 倍被 v1 判为「更集中」')],
    BLOCK(1, '这是本线第四次同类错误（H5 分母、H6 verdict、H8 形状、H11 判据），都出在「判据太松」'),
    '只写不等号的判据在数值噪声面前必然成立。'
    '⇒ 所有判据模板改为「实测值 + 阈值 + 差值」三件套，并在输出里同时打印；'
    '凡是只写 `>` / `<` 的地方，一律视为未定义判据。')

# ---------------- 注意力头 ----------------

add('R-ATT-01', 'attn', '读出位置',
    '注意力头对答案的贡献集中在晚期层、在"答案位"读出；问题位与选项位几乎没有显著头。',
    'confirmed', 'high',
    '|t|>4（Bonferroni）的显著头按读出位置分布明显不均',
    {'n_significant': 58, 'layer_range': 'L24–L35', 'at_q_end': 0, 'at_opt_end': 2, 'at_answer': 58},
    [ev('A1', 'server/mech_service.py', 'data/mech/pos_profile.json', '三读出位 × 五分段')],
    BLOCK(2, '三个位置各跑一次，分布极不均匀（58 / 2 / 0）'),
    '这条确立了"头级归因"的可行口径：只看答案位。也让 58 个显著头成为后续分析的主力单元。')

add('R-ATT-02', 'attn', '重要性预测',
    '用头的权重几何（写方向对答案字母的投影 / 能量 / 有效秩）可以预测该头的 DLA 显著性。',
    'refuted', 'high',
    'corr(几何量, |t|) ≥0.6',
    {'corr_w_letter_max': 0.349, 'corr_energy': 0.174, 'corr_ov_pr': 0.026,
     'significant_vs_not_ratio': 1.15},
    [ev('A1', 'server/mech_service.py', 'data/mech/heads_dir.json', '1152 头 OV/QK 分解')],
    BLOCK(1, '三个几何量全部低相关，显著头只有非显著头的 1.15×'),
    '与 R-MLP-10 同构：权重几何在这个模型上不能替代实测。')

add('R-ATT-03', 'attn', '头类型',
    'prev-token 头参与答案读出（在 DLA 显著头中占相当比例）。',
    'refuted', 'high',
    'DLA 显著头中 prev-token 类型 ≥5 个',
    {'prev_in_significant': 0, 'n_significant': 58, 'n_prev_total': 74},
    [ev('A1', 'server/mech_service.py', 'data/mech/head_types.json', 'induction / prev / dup / sink 打分')],
    BLOCK(1, '0/58 是干净的零'),
    'prev-token 头（74 个，集中 L0–L6）做的是局部复制，不参与答案决策——'
    '这本身是有价值的分工证据，不必再测。')

add('R-ATT-04', 'attn', '头类型',
    'induction 头与答案 DLA 显著头显著重叠，是答案读出的重要载体。',
    'confirmed', 'medium',
    'DLA 显著头中 induction 类型 ≥10 个（随机期望约 58×46/1152≈2.3）',
    {'induction_in_significant': 13, 'n_significant': 58, 'n_induction_total': 46, 'expected_random': 2.3},
    [ev('A1', 'server/mech_service.py', 'data/mech/head_types.json', '最强 L13H13 0.9827 / L22H23 0.9716')],
    BLOCK(1, '5.7× 富集，方向明确'),
    'induction 头双带分布（L2/L4/L7 + L13/L20–25/L28–33），与答案位显著头带（L24–35）在后一带重合。')

add('R-ATT-05', 'attn', '重要性预测',
    '头在选项上的注意力质量可以预测它的 DLA 显著性。',
    'refuted', 'high',
    'corr(注意力质量, |t|) ≥0.5',
    {'corr_all_heads': 0.103},
    [ev('A1', 'server/mech_service.py', 'data/mech/pos_profile.json', '全头注意力质量 vs |t|')],
    BLOCK(1, '0.103 远低于阈值'),
    '但这条失败换来一个更好的发现：早看晚写（见 R-ATT-06）。')

add('R-ATT-06', 'attn', '分工',
    '存在"早看晚写"分工：早期层（L0–L6）的头对选项注意力高达 0.85–0.98 但对答案 DLA 几乎为零，晚期层同时具备两者。',
    'confirmed', 'high',
    '早层头注意力质量 ≥0.8 且 |t|<1；晚层头 ≥0.75 且 |t|>8',
    {'early_quality': '0.85–0.98', 'early_t': '<1', 'late_examples': 'L26H26 0.80/t=11.2, L24H31 0.78/t=14.6'},
    [ev('A1', 'tools/mech_pos_probe.py', 'data/mech/pos_profile.json', '逐头注意力分段构成 × DLA')],
    BLOCK(1, '两条带同时存在，且 P6（注意力位置决定贡献）已被证伪'),
    '原假设"头在答案位就读答案"被否证，换成这条更强的分工描述。')

add('R-ATT-07', 'attn', '分辨率',
    '在 GQA 结构下可以对每个注意力头独立做归因（头级分辨率 32）。',
    'refuted', 'high',
    '同组内 4 个头的读取通路应彼此独立',
    {'n_query_heads': 32, 'n_kv_heads': 8, 'group_size': 4, 'true_dof': '9728×8'},
    [ev("G'", 'server/mech_pair_service.py', 'data/mech/pair_gain.json', '同组 4 头读取增益逐位相等')],
    BLOCK(1, '结构事实，不需要重测'),
    '32 个 query 头只有 8 套 K/V。所有"头级"结论必须降一档理解——'
    '同组 4 头共享读写通路，"独立头"是假象。')

# ---------------- 方法 ----------------

add('R-MTH-01', 'method', '干预方式',
    '在输出处投影掉组平均写方向（dir 干预）可以定位该组的功能，且效应可被匹配对照分辨。',
    'inconclusive', 'medium',
    'dir 干预 KL ≥3× 匹配 sham/gauss 对照，且与层主写方向 |cos| 可解释',
    {'kl_c18': 0.02592, 'sham_mean': 0.00209, 'ratio': 12.4, 'cos_with_layer_dir': '≤0.09'},
    [ev('C1', 'tools/mech_c1_full_run.py', 'data/mech/ablate_L24.json', '24 组 dir 干预 + sham0/sham1/gauss0/gauss1')],
    NEEDN(1, 2, '需要第二层（如 L14 分布式层）复现才能定论；单层 24 组不足'),
    'KL 是 sham 的 12.4×，但投影方向与层主写方向的 |cos|≤0.09——'
    '权重几何看不见、只有干预能看见。这个反差本身可能只是"投影掉了噪声子空间"。')

add('R-MTH-02', 'method', '干预方式',
    'steering（沿方向加）与 ablation（删掉）在同一组单元上效应强度相当。',
    'open', 'low',
    '同组同规模的加 vs 删，行为改变量比值 ∈[0.5, 2]',
    {'status': 'not_tested'},
    [],
    ALLOW(0, '从未做过。若 R-MLP-04 的差分对假说成立，steering 应远强于 ablation'),
    '这是差分对假说最具判别力的一条：加是不对称操作、删是对称操作。'
    '如果 steering ≫ ablation，就能解释整个领域"控制有效、消融无效"的普遍困惑。')

# 查重用的补充关键词（中英混写时，光靠 claim 的二元字组命中率不够）
KEYWORDS = {
    'R-MLP-01': ['held-out', '跨数据集', 'top-97', 'overlap', '复现', 'spearman'],
    'R-MLP-02': ['删掉', 'top-1%', '正确率', '下降', '必要性', 'ablate'],
    'R-MLP-03': ['改答', '改变答案', '杠杆', 'leverage', '随机对照'],
    'R-MLP-05': ['聚类', 'cluster', 'KMeans', '共现', '模块', '写方向', 'cos'],
    'R-MLP-08': ['集中度', 'concentration', '层结构', '预筛'],
    'R-MLP-10': ['权重', '范数', '几何', 'OV', 'energy', '有效秩', '预测', '剪枝'],
    'R-MET-01': ['KL', '散度', '长尾', '扰动', 'energy', 'top-20'],
    'R-MET-02': ['随机', '对照', 'random', 'rand97', '弱对照', '范数'],
    'R-MET-04': ['n=40', '40', '样本量', '功效', 'McNemar', '掉点', '169', '684'],
    'R-ATT-01': ['DLA', '显著头', '答案位', 'q_end', 'opt_end', '58'],
    'R-ATT-02': ['权重', '几何', '投影', 'energy', '有效秩', '预测'],
    'R-ATT-07': ['GQA', 'KV', '分组', '分辨率', '独立头'],
    'R-MLP-16': ['饱和', '冗余', '备份', '共享池', '删一半', '池内', 'saturation'],
    'R-MLP-17': ['剂量', 'dose', '饱和点', 'k80', '曲线', '单调', '悬崖'],
    'R-MLP-18': ['排序', '排名', 'top-k', '绝对值', '归因幅度', '头部', '惰性',
                 'rank', '重要单元', '取前几个', '删前几十个'],
    'R-MLP-19': ['脆弱题', '同一批题', '集中损伤', '12 道', '承载单元', '近二值', '池依赖'],
    'R-MLP-20': ['预测脆弱', '哪道题会坏', '逐题归因', '单题归因', '预测性'],
    'R-MLP-21': ['承载核', '最小承载集', '哪几个单元', '会承载单元', '3 个单元', '二分',
                 'binary search', '定位承载', '767', '1540'],
    'R-MLP-22': ['脆弱题分类', '最小充分集合', '几档', '分层', '不是同一类', '哪几道题'],
    'R-MLP-23': ['单点', '单个单元', '一个单元', '767', '最小集合', '承载核大小',
                 '三合一', '为什么是三个'],
    'R-MET-10': ['二分', '自适应二分', '最小充分集', '子集', '下限', '搜索停止'],
    'R-MET-11': ['随机对照', '特异性', '计数', '身份', '并集', '零分布', '通用脆性'],
    'R-MLP-24': ['通用脆性', '任何单点', '73', 'pool-generic', '题池噪声'],
    'R-MLP-25': ['换行', '格式', '前缀', 'logit 差分', '767 算什么', '格式单元'],
    'R-MET-12': ['答对答错', '因变量', '格式不变', 'teacher forcing', '内容必要性', '格式混淆'],
    'R-MET-08': ['线性', '形状', '未触发', '阈值', '曲线读法', '读法'],
    'R-MET-09': ['判据', '效应量', '阈值', '不等号', '噪声差', '误判成立'],
    'R-MTH-02': ['steering', '方向干预', '加方向', '控制', '加'],
}
for _r in R:
    if _r['id'] in KEYWORDS:
        _r['keywords'] = KEYWORDS[_r['id']]


def main():
    now = time.strftime('%Y-%m-%dT%H:%M:%S')
    if LEDGER.exists():
        try:
            old = json.loads(LEDGER.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            old = {}
    else:
        old = {}
    prev = {r.get('id'): r for r in old.get('rules', []) or []}

    rules = []
    for r in R:
        r = dict(r)
        r['created_at'] = (prev.get(r['id']) or {}).get('created_at', now)
        r['updated_at'] = now
        rules.append(r)

    out = {
        'schema': 'rule_ledger.v1',
        'version': int(old.get('version') or 1),
        'updated_at': now,
        'model_scope': MODEL,
        'policy': ('一条规则只在给定模型 + 给定口径下成立；口径不变时不重复测。'
                   'retest.status=block 表示已有足够证据，再测属于重复劳动；'
                   'need_n 表示结论方向已知但样本不足，重测必须上到该题量才可能改变判定；'
                   'allow 表示从未做过或口径已变。'),
        'rules': rules,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
    from collections import Counter
    c = Counter(r['status'] for r in rules)
    rt = Counter(r['retest']['status'] for r in rules)
    print('写入 %s：%d 条规则' % (LEDGER, len(rules)))
    print('  状态：' + '  '.join('%s=%d' % (k, v) for k, v in c.items()))
    print('  复查：' + '  '.join('%s=%d' % (k, v) for k, v in rt.items()))
    print('  累计已测次数：%d' % sum(r['retest'].get('count') or 0 for r in rules))


if __name__ == '__main__':
    main()
