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

# ---------------- 注意力头：模式破解线（2026-11-01 一夜九轮） ----------------

add('R-ATT-08', 'attn', '头级功能定位',
    '1152 个头里存在一组约 263 个「角色定位头」：注意力落点等于所问角色（施事/受事/客体），显著高于随机落点。',
    'refuted', 'high',
    '落点命中率显著高于 null（null=0.344），且主动/被动两种句式同时成立',
    {'n_pass': 263, 'n_total': 1152, 'null': 0.344, 'median_layer': 28,
     'pass_inband_cov': 0.23, 'fail_inband_cov': 0.052,
     'raw_gain_k60': 0.389, 'aligned_gain': 0.101, 'band_delta': -0.028},
    [ev('Q1', 'tools/pat_q1_unbind.py', 'results/2026-11-01/*_pat_q1_unbind.json', '24 三元组 × 2 句式 × 3 角色，行为门 70/72'),
     ev('Q2-A', 'tools/pat_q2a_confound.py', 'results/2026-11-01/*_pat_q2a_confound.json', '2×2 混杂分解（是否通过筛选 × 是否低层）')],
    BLOCK(3, '密度混杂已定量：通过组层内覆盖率 23%、未通过组 5.2%；对齐后效应 +0.389 → +0.101，整带为 −0.028'),
    '本线第一次「看起来很干净」的假阳性。筛选本身把高覆盖率（更容易被消融影响）的头选进来了，'
    '所谓「落点精准」其实是「删得多所以掉得多」。口径不变时不要再测头级落点。')

add('R-ATT-09', 'attn', '角色编码形式',
    '末位读出只接受一个约 53° 的「角度门锥」：位移方向落在锥内近全效、锥外归零，锥的朝向逐样本不同。',
    'refuted', 'high',
    '固定长度扫 θ∈[0°,90°]，D/R/M/Z 四臂应呈阶跃（近 1 → 0），且 cos(Δ_i,Δ_j) 应有共享轴',
    {'D_0deg': 1.000, 'R_33deg': 0.958, 'M_61deg': 0.479, 'Z_90deg': 0.000,
     'cos_pairs_mean': 0.245, 'cos_with_consensus': 0.482},
    [ev('Q2-C2', 'tools/pat_q2c2_direction.py', 'results/2026-11-01/*_pat_q2c2_direction.json', '同长度四方向扫描'),
     ev('Q2-L2', 'tools/pat_q2l2_controls.py', 'results/2026-11-01/050009_pat_q2l2_controls.json', '换层/换位置/换目标词三控制')],
    BLOCK(4, '死因是测量对象错：tied embedding + 末层 RMSNorm ⇒ 加 E[X]−E[Y] 就抬高 X 的 logit，近乎代数恒等式。注入 L20–L35 全部 1.000、注入宾语答宾语 0.886'),
    '「锥」是 LM head 线性读出的几何，不是角色编码。由此得出一条硬规则：'
    'tied-embedding 模型上的任何「注入差向量」实验都不能用来推断机制（D9）。')

add('R-ATT-10', 'attn', '角色编码形式',
    '三种角色在末位可分离为一组基（d1 施事 / d2 受事 / d3 客体），且 d3 不影响答案。',
    'refuted', 'high',
    'd1/d2 翻转率 ≈1.0、d3 ≈0（答案不变），且 orth 对照 ≈随机',
    {'d1': 0.977, 'lex1': 1.000, 'd2': 1.000, 'd3_unchanged': 1.000,
     'orth1': 0.159, 'rand': 0.091, 'cos_d1_d2': 0.545},
    [ev('Q2-L', 'tools/pat_q2l_decompose.py', 'results/2026-11-01/045825_pat_q2l_decompose.json', '三格全同基座设计，n=44，覆盖 12 对人物')],
    BLOCK(2, '自判平凡：d1 的效力全部由词法分量承担（剥到 λ=1 后 1.000 → 0.159），所谓「角色基」就是「目标词嵌入差」'),
    '目标词不必出现在句中也成立（d2 指向从未出现的名字，翻转 1.000），'
    '说明这不是角色推理，是输出可被直接转向任意词表项。')

add('R-ATT-11', 'attn', '承载结构',
    '存在必需头、必需层或「必需致病组合」——删掉它们会让行为崩溃。',
    'refuted', 'high',
    '每层只保留 k 个头（k≤4），12 次随机抽样应出现部分崩溃；两次触底样本的交集应显著大于随机期望',
    {'k': 4, 'exact_1.000_runs': '12/12', 'head_coverage': '119/128 (93%)', 'chance_overlap': 32, 'observed_overlap': 32},
    [ev('Q2-H', 'tools/pat_q2h_core.py', 'results/2026-11-01/*_pat_q2h_core.json', 'L0–L3 内 k_per∈{1,2,4,8,12,16}，12 次嵌套抽样')],
    BLOCK(3, '已四路交叉验证（Q2-D / D2 / G / H）：k≤4 时 12/12 精确 1.000，两次触底交集等于随机期望'),
    '结论是容量分布式：没有任何单点承担功能，损伤是累积量而不是结构量。'
    '不要再问「哪个头负责什么」，这个问题在本模型上没有答案。')

add('R-ATT-12', 'attn', '承载剖面',
    '注意力对行为的承载呈倒 U 形：中层最强、两端最弱。',
    'refuted', 'high',
    '逐层单独全消融的行为 drop 应呈倒 U（中层峰、两端低）',
    {'k': [10, 30, 60, 100, 150, 263], 'drop': [0.000, 0.022, 0.117, 0.239, 0.451, 0.667],
     'shape': 'monotone'},
    [ev('Q2-D', 'tools/pat_q2d_partial.py', 'results/2026-11-01/*_pat_q2d_partial.json', '层内比例 p 扫描'),
     ev('Q2-A3', 'tools/pat_q2a3_layerwise.py', 'results/2026-11-01/*_pat_q2a3_layerwise.json', '35/36 层单独全消 ≤0.007（仅 L0 = 0.153）')],
    BLOCK(2, '实测单调递增，无峰值；drop 随删掉的头数单调上升并在地板处饱和'),
    '「中层是瓶颈」是视觉模型的经验移植。在语言模型的行为读出上，'
    '承载量靠近似等于「删掉了多少」，与层位置几乎无关。')

add('R-ATT-13', 'attn', '对齐与承载',
    '对答案最敏感的头（对齐头）就是承载行为的头（承载头）——两者是同一批。',
    'refuted', 'high',
    '按对齐分数筛出的头，其消融 drop 应显著高于同层随机头',
    {'aligned_heads': 263, 'aligned_in_L20_35': 260,
     'carry_L0_3': 0.674, 'carry_L16_19': 0.660, 'carry_L28_35': 0.000,
     'phase_gap_layers': 20},
    [ev('Q2-D2', 'tools/pat_q2d2_bandfull.py', 'results/2026-11-01/*_pat_q2d2_bandfull.json', '9 带 × 128 头全消')],
    BLOCK(2, '两者反相约 20 层：263 个对齐头里 260 个落在 L20–35，而 L28–35 的承载量为 0.000'),
    '这是本线最重要的结构性事实：**「看起来在处理这个信息」与「缺了会坏」是两件不同的事，'
    '而且发生在不同的层**。任何用对齐分数当功能指标的做法都要先过这一关。')

add('R-ATT-14', 'attn', '损伤定律',
    '注意力消融造成的损伤 ≈ 覆盖率 × 集中度：删得越广、越集中在少数层，掉得越多。',
    'confirmed', 'high',
    '同等头数下，集中在少数层的 drop 显著高于铺满全层；比值应 >2',
    {'wide_128_4layers': 0.674, 'flat_144_36layers': 0.197, 'ratio': 3.4,
     'floor': 0.3333, 'max_drop': 0.667},
    [ev('Q2-E', 'tools/pat_q2e_additivity.py', 'results/2026-11-01/*_pat_q2e_additivity.json', '双带可加性'),
     ev('Q2-G', 'tools/pat_q2g_distribution.py', 'results/2026-11-01/*_pat_q2g_distribution.json', '全 36 层、只扫每层 k')],
    BLOCK(3, '3.4× 差距稳定复现；轻度超可加、重度饱和'),
    '这条可以直接用来给任何「删 X 掉 Y」的结论做期望值校准：'
    '看到 drop 之前先算它本该掉多少。串联共享通路 + 损伤预算，是当前最稳的一条定量规律。')

add('R-ATT-15', 'attn', '读出汇聚',
    '末位存在一个读出汇聚点 L33：L33 之前上游还能改写末位内容，L33 之后不能。',
    'confirmed', 'high',
    '真实激活修补（非注入）：把干净句的施事位置激活写回坏句，看答案恢复率随写入层的衰减点',
    {'patch_L0_29': 1.00, 'patch_L30_32': 'decay', 'patch_L33_on': 0.00,
     'last_pos': 'opposite', 'third_entity_decay': 'L32 (0.21)'},
    [ev('Q2-C1', 'tools/pat_q2c1_patch.py', 'results/2026-11-01/*_pat_q2c1_patch.json', '实体互换配对 patch，写入 hidden_states[L+1]')],
    BLOCK(2, '阶跃而非连续：L32 仍有 1.00，L33 直接 0.00'),
    '推论：**角色解算发生在 L33 之前**。所以任何「角色编码长什么样」的问题，'
    '都必须在 L20–L32 窗口里问，L33 及以后问到的都是读出几何。')

# ---------------- 指标与对照：模式破解线新增 ----------------

add('R-MET-13', 'metric', '干预效度',
    '在末位注入「目标词的嵌入差」，可以用来研究角色编码机制。',
    'refuted', 'high',
    '注入后行为改变应随注入层变化（若为机制，应只在读出点附近生效）',
    {'inject_L20_35_all': 1.000, 'inject_q0': 0.091, 'inject_object_answer_object': 0.886},
    [ev('Q2-L2', 'tools/pat_q2l2_controls.py', 'results/2026-11-01/050009_pat_q2l2_controls.json', '换层/换位置/换目标词三控制，n=44')],
    BLOCK(3, '与层完全无关（L20/25/29/31/33/35 全部 1.000），换位置到句首即掉到 0.091'),
    'tied-embedding 结构下这是代数恒等式，不是机制。'
    '**任何注入类实验，必须先证明它在不该生效的地方不生效**（D9）。')

add('R-MET-14', 'metric', '混杂控制',
    '两组头数不同时，可以直接比较消融造成的 drop 大小。',
    'refuted', 'high',
    '比较前须把「层内覆盖率」配平；未配平时 drop 差异应由覆盖率解释',
    {'raw_diff_k60': 0.389, 'coverage_pass': 0.23, 'coverage_fail': 0.052,
     'aligned_diff': 0.101, 'band_diff': -0.028},
    [ev('Q2-A', 'tools/pat_q2a_confound.py', 'results/2026-11-01/*_pat_q2a_confound.json', '2×2 分解')],
    BLOCK(2, '配平后效应从 +0.389 塌到 +0.101，整带反向'),
    '这是本线最贵的教训：一个 +0.389 的「显著效应」，'
    '配平覆盖率之后变成 −0.028。**报 drop 必报三元口径：头数、占总头比、层内覆盖率**（D1）。')

add('R-MET-15', 'metric', '分布读法',
    '对多次随机抽样的结果取均值，可以代表该条件下的典型效应。',
    'refuted', 'high',
    '取均值前须检查分布；极差 ≥0.30 或双峰时均值无意义',
    {'observed_range': [0.34, 1.00], 'shape': 'bimodal', 'mean_used_before': True},
    [ev('Q2-F', 'tools/pat_q2f_hetero.py', 'results/2026-11-01/*_pat_q2f_hetero.json', '12 次抽样异质性')],
    BLOCK(4, '本线已在三处重犯（Q2-F / Q2-C2c / 早期 A64），每次都是双峰被均值抹平成「中等效应」'),
    '正确做法是逐次配对比较，或报告分布本身。均值 + 一个 ±0.04 的标准误，'
    '会掩盖「一半样本 1.00、一半 0.34」这种结构（D5）。')

add('R-MET-16', 'metric', '样本量',
    'SMOKE（小样本预跑）上得到的效应方向与大小，可以外推到全量。',
    'refuted', 'high',
    'SMOKE 与全量的效应方向应一致；且 SMOKE 必须覆盖全量的全部结构维度',
    {'smoke_n': 24, 'smoke_effect': 0.292, 'full_n': 44, 'full_effect': -0.028},
    [ev('Q2-A', 'tools/pat_q2a_confound.py', 'results/2026-11-01/*_pat_q2a_confound.json', 'SMOKE 与全量反向'),
     ev('Q2-C2c', 'tools/pat_q2c2c_samelength.py', 'results/2026-11-01/*_pat_q2c2c_samelength.json', 'SMOKE 用 triples(4)（4 对全同人物，cos 0.936）vs 全量 0.245')],
    BLOCK(3, '两次反向：n=24 的 +0.292 → 全量 −0.028；SMOKE cos 0.936 → 全量 0.245'),
    'SMOKE 只能验证管线跑得通，**不能作为证据**。'
    '而且 SMOKE 的样本必须覆盖全量的每一个结构维度（D8）：4 个三元组共用同一对人物，就是一个维度没覆盖。')

add('R-MET-17', 'metric', '对照匹配',
    '方向类对照只需要配平投影长度，不必同时配平向量长度。',
    'refuted', 'high',
    '同幅度对照须同时控 |位移| 与投影；缩放因子依赖样本量时必须报其分布与最大值',
    {'scale_factor_max': 2.07, 'cos_median': 0.482, 'curve_shape': 'non-monotone'},
    [ev('Q2-C2', 'tools/pat_q2c2_direction.py', 'results/2026-11-01/*_pat_q2c2_direction.json', 'S 臂走 alpha·||Δ||/cos')],
    BLOCK(2, 'S 臂实际位移是目标的 2.07 倍，曲线非单调（0.639→0.611）是过冲不是额外效力'),
    '「配平了投影」听起来比「配平了长度」更严格，实际反而引入了未控的幅度差。'
    '超 cap 的臂应该直接不跑，而不是跑完再解释（D7）。')

add('R-MET-18', 'metric', '行为地板',
    '三选一强制选择任务的行为地板是 1/3，任何 drop 的上限是 0.667。',
    'confirmed', 'high',
    '强制在三个候选首 token 上取 argmax，随机基线的期望准确率 = 1/3',
    {'floor': 0.3333, 'max_drop': 0.667, 'observed_floor_hits': 'Q2-E 三条件全部 0.3333'},
    [ev('Q2-G', 'tools/pat_q2g_distribution.py', 'results/2026-11-01/*_pat_q2g_distribution.json', '全 36 层分布扫描')],
    BLOCK(3, '解析可算，且实测三次触底'),
    '报 drop 前必须先报「离地板还有多远」（D4）。'
    '本项目曾因撞地板而误判「三条件同效」，实际三者都已经到底，无法区分。')

add('R-MET-19', 'metric', '判据构造',
    '判据里取 min(两臂) 可以避免某个臂失效导致的假阳性。',
    'refuted', 'medium',
    '若某个臂在基线条件下恒为零，min 会把整个判据钉死在零',
    {'case': 'Q2-C1 判据 min(agent0, agent1)', 'agent1_baseline': 0.0},
    [ev('Q2-C1', 'tools/pat_q2c1_patch.py', 'results/2026-11-01/*_pat_q2c1_patch.json', '变量名取错导致恒零量进入 min')],
    BLOCK(1, '复合判据的每个分量都必须单独报自己的剖面'),
    '保守的判据构造（取 min / 取交）会引入「恒零分量」这类结构性失效，'
    '而且失效方向是「什么都判不出来」，看起来像「没效应」而不是「判据坏了」（D6 补充）。')

# ---------------- 实验方法：从九轮里抽出的框架级规则 ----------------

add('R-MTH-03', 'method', '结论分级',
    '分布剖面（E1）可以独立作为「规律」；干预实验（E2）单独只能作为「线索」。',
    'confirmed', 'high',
    '同一条结论若只由 E2 支撑，在下一轮换任何对照后都应预期失效',
    {'E2_conclusions_retracted': 3, 'E2_survived': 0, 'E1_conclusions_retracted': 0},
    [ev('PSF-1', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '四层证据 + 升格规则')],
    BLOCK(1, '框架级规则，不需要重测'),
    '一夜九轮里被撤回的结论 100% 来自 E2（点陈述），没有一条来自 E1（剖面）。'
    '**这是本项目「无限测试」循环的直接成因**：用点陈述承担了规律的角色（D10）。')

add('R-MTH-04', 'method', '对照管理',
    '对照组可以随着实验推进逐轮补充——每轮发现新混杂就加一个新对照。',
    'refuted', 'high',
    '对照族必须在实验前一次性冻结；新增对照须升级族版本并重测全部历史结论',
    {'frozen_transforms': 8, 'rounds_before_freeze': 'unbounded'},
    [ev('PSF-1', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '变换族 T-v1 冻结 8 个')],
    BLOCK(1, '已观测到循环：Q1 死于密度、Q2-C2 死于长度、Q2-L 死于读出，每轮都是「临时想到一个对照」'),
    '对照空间不封闭 ⇒ 等价于无限次假设检验。'
    '**冻结的代价才是闸门**：新增 t9 要求重测全部历史，这个成本让「边测边加」变得不划算（D11）。')

add('R-MTH-05', 'method', '搜索预算',
    '一个假设方向测不出结果时，可以换更细的切片继续测，直到测出来为止。',
    'refuted', 'high',
    '每个假设家族设 3 轮上限；3 轮内拿不到「8 个变换中至少 4 个不变」即永久关闭',
    {'family_budget': 3, 'pass_threshold': '4/8'},
    [ev('PSF-1', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '闸 1 预算封顶')],
    BLOCK(1, 'D→D2→G→H 就是这个模式的实例：一轮比一轮细，最后得出「无必需头」'),
    '更细的切片不会带来新信息，只会带来新的混杂。'
    '**关闭一个方向比测出它更有价值**——它把候选空间真正缩小了（D12）。')

add('R-MTH-06', 'method', '排除集',
    '过程账本（谁在什么时候跑了什么）足以支撑后续决策，不需要单独维护「已排除集」。',
    'refuted', 'high',
    '每轮必须让排除集至少 +1；未增大的轮次记 NULL 且禁止在同一子空间再开 Phase',
    {'ledger_records': 455, 'excluded_set_v0': 6},
    [ev('PSF-1', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '排除集 E-01…E-06 首次落盘')],
    BLOCK(1, '455 条 ledger 记录里没有任何「什么已被永久排除」的索引'),
    '过程账支撑复现，排除集支撑决策。**没有后者，死掉的方向换个包装又回来**——'
    '这是循环真正的燃料，不是「假设不够好」。')

add('R-MTH-07', 'method', '分布式合并',
    '多个分片跑同一套实验时，把各分片的结论汇总起来（多数票 / 平均）即可得到全局结论。',
    'refuted', 'high',
    '不变量取交集（全片成立才算）、候选规律取并集（单片入池但须双片复现）、剖面逐片保留不取平均',
    {'shards': 1280, 'merge_invariant': 'intersection', 'merge_candidate': 'union', 'merge_profile': 'no_average'},
    [ev('PSF-1', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '§3.5 分布式结构')],
    BLOCK(1, '尚未实跑，规则来自 D5 与「单片结论不可迁移」的两次实测'),
    '最容易错的三处：① 用并集算不变量 ⇒ 假阳性爆炸；② 剖面取平均 ⇒ 双峰被抹平；'
    '③ 分片各自用不同判据。因此**汇总器只收原始量、不收结论**（D13）。')

add('R-MTH-08', 'method', '失败归因',
    '实验没有产出规律时，应当换假设继续找。',
    'refuted', 'high',
    '先判定是「假设错」还是「测量钝」：3 轮内无任何量达到 4/8 不变 ⇒ 换测量而非换假设',
    {'threshold': '3 rounds × inv<4/8'},
    [ev('PSF-1', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '§6 框架自检条款')],
    BLOCK(1, '历史 100% 的失败都被归因为「假设不对」，从未被归因为「工具钝」'),
    '两种失败看起来完全一样（都是没结果），但修法相反。'
    '**不区分它们，就会永远在换假设，而工具一直是钝的**——这是上万个测试的总病根。')

add('R-MTH-09', 'method', '齿痕可测性',
    '注意力分数可以按「钥匙的三项分解」精确拆开：s = q·k_cmn + q·k_id + q·k_ctx，三项正交、和为恒等。',
    'confirmed', 'high',
    '拆解后三项方差份额之和 ≈1；若某项占比 >0.95 且另两项 ≈0，说明分解退化（未扣共同方向）',
    {'n_cells': 114, 'share_cmn_med': 0.543, 'share_id_L0': 0.95, 'share_ctx_med': 0.009, 'share_sum_med': 1.05},
    [ev('Q3-T2', 'tools/pat_q3t_t2.py', 'results/2026-11-01/055903_pat_q3t_t2.json', '114 格，行为门 113/114')],
    BLOCK(1, '同步前向的代数分解，无消融/注入/移植，三种旧混杂均不适用；份额和 1.05 说明三项略正相关'),
    '这是本线第一个不含任何旧混杂的测量。前提是**必须先扣掉所有词共有的方向**再取身份方向，'
    '否则身份项会吃掉全部能量（share_id 假性 = 1.00）。')

add('R-MTH-10', 'method', '齿痕剖面',
    'token 的「齿痕」（key 的身份差异）沿层的形状是 U 形：低层高、中层被上下文稀释、高层回升。',
    'refuted', 'high',
    'R_id(l,h,w) = ||mean_i k_i|| / mean_i||k_i||，M=20；形状须落在 {U形, 单调降, 单调升, 平坦} 之一',
    {'sep_L0': 0.002, 'sep_peak': 0.230, 'peak_layer': 13, 'sep_L35': 0.042,
     'match_all_layers': 1.000, 'fold_over_chance': 30.0},
    [ev('Q3-T1', 'tools/pat_q3t_t1.py', 'results/2026-11-01/055514_pat_q3t_t1.json', '30 词 x 20 上下文 x 2 模板 = 1200 条')],
    BLOCK(2, '四选一形状全落空：前中差 0.007（不是 U）、rho -0.169/+0.194（不是单调）、极差 0.227（不是平）'),
    '实测形状是「L0 精确为零 → L13 峰 0.23 → L35 收至 0.04」，四选一里没有这一档。'
    '但形状跨位置可复现（P9 vs P0 Spearman 0.835），是不变量候选。'
    '身份可分性反而满格（30x 基线，全层 1.000）——绝对差异小不等于不可分。')

add('R-MTH-11', 'method', '融合复用',
    '不同语义角色在 logit 差上复用同一批稀疏字典列（reuse ≥ 3），即融合是结构性复用而非临时拼装。',
    'refuted', 'high',
    '稀疏分解得 |D|≤64，任意 logit 差用 ≤3 列重建 R²≥0.9；reuse(role) = Jaccard / 随机期望',
    {'r2_holdout': 0.819, 'r2_holdout_best_fold': 0.888, 'r2_insample': 0.989,
     'reuse_ag': 1.788, 'reuse_re': 1.609, 'reuse_th': 1.300, 'reuse_median': 1.609,
     'reuse_same_entity': 7.884, 'reuse_diff_entity': 1.236,
     'pca_shared_rank8': 0.653, 'random_expectation_jaccard': 0.0604},
    [ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json',
        '144 格 = 24 三元组 x 2 句式 x 3 查询，V=39，K=32，OMP≤3 非零，按三元组 5 折')],
    BLOCK(1, 'held-out R² 0.819 < 0.90（5 折最好 0.888 也没到）；reuse 1.30–1.79 远低于 3.0'),
    '但复用**确实存在**，只是索引不同：同答案实体 7.88x vs 不同答案 1.24x，比值 6.4 倍。'
    '正确的替代陈述不是「没有复用」，而是「复用按答案实体组织，不按角色组织」。'
    '另：共享 rank-8 子空间只解释 65.3%，说明跨三元组没有可迁移的共享低维模式。')

# ---------------- 注意力头：Q3-T 线新增（2026-11-01） ----------------

add('R-ATT-16', 'attn', '上下文写入',
    '上下文通过改写 key 来决定注意力落点——同一 token 在不同句子里的 key 会有实质变化。',
    'refuted', 'high',
    '同一 token 跨 20 个不同上下文的 key 方向 cos 应显著低于不同 token 之间；上下文特异方差份额应占可观比例',
    {'cos_same_across_ctx': 0.998, 'cos_diff_words_L0': 0.9976, 'cos_diff_words_L35': 0.9472,
     'share_ctx_median': 0.009, 'share_ctx_max': 0.03},
    [ev('Q3-T1', 'tools/pat_q3t_t1.py', 'results/2026-11-01/055514_pat_q3t_t1.json', '1200 条，位置严格对齐'),
     ev('Q3-T2', 'tools/pat_q3t_t2.py', 'results/2026-11-01/055903_pat_q3t_t2.json', '三项方差分解')],
    BLOCK(2, '同词跨上下文 cos 0.998 与异词 cos 0.9976 几乎相等；上下文特异份额中位 0.009、全层 ≤0.03'),
    '限定说明：k_ctx 定义为「偏离该词跨上下文平均方向」的部分，衡量的是上下文的**特异**贡献，'
    '共有部分被算进 k_cmn。所以成立的窄命题是：**句子与句子之间的差异几乎不改 key**。'
    '推论：角色选择不在 k 侧，必须去 q 侧找——这解释了头侧九轮全部失败。')

# ---------------- 指标：Q3-T 线新增 ----------------

add('R-MET-20', 'metric', '份额指标',
    '方差份额（share）是样本集无关的内禀量，可以在混合不同条件的样本上直接估计。',
    'refuted', 'high',
    '同一份额在至少两个条件分组内分别重算，降幅差应 <20%',
    {'pooled_drop': 0.66, 'per_group_drop': [0.26, 0.23, 0.26], 'inflation': 2.5},
    [ev('Q3-T2', 'tools/pat_q3t_t2.py', 'results/2026-11-01/055903_pat_q3t_t2.json', '按 ag/re/th 分组重算')],
    BLOCK(1, '混合 114 格得降幅 0.66，按查询分组得 0.23–0.26，夸大 2.5 倍'),
    'share 是相对于所选样本集的条件量。混合互不相同的条件会把它抬高。'
    '**报份额必须同时报至少两个分组的值**（D14）；这也适用于所有"方差解释比例"类指标。')

add('R-MET-21', 'metric', '分解构造',
    '把「同一 token 跨上下文的平均方向」直接当作身份方向，可以把 key 拆成身份分量 + 上下文分量。',
    'refuted', 'high',
    '分解后身份份额应显著小于 1；若 ≈1 说明身份项吞掉了共同方向',
    {'naive_share_id': 1.00, 'fixed_share_id_L0': 0.95, 'fixed_share_ctx': 0.009},
    [ev('Q3-T2', 'tools/pat_q3t_t2.py', 'results/2026-11-01/055903_pat_q3t_t2.json', 'SMOKE 暴露后退回重做')],
    BLOCK(1, '所有 key 两两 cos 0.998，均值方向就是主方向，投影后残差≈0'),
    '正确顺序是三步：先取「所有词所有上下文共有」的方向 g，扣掉它，'
    '再在各词的去共同残差里取身份方向（D15）。省掉第一步会让分解退化成"几乎全部能量都算身份"。')

# ---------------- 注意力头 / 指标 / 方法：Q3-T3 线新增（2026-11-01） ----------------

add('R-ATT-17', 'attn', '融合形态',
    '下一词分布由少数几个可复用的语义角色模式相加而成，即「多模式融合」在输出端有角色专属列族。',
    'refuted', 'high',
    'held-out R²≥0.90 且 reuse(role)≥3.0（bootstrap 95% CI 下界 >1.5）；对照：跨角色复用与同实体复用',
    {'cos_within_role': 0.9676, 'cos_cross_role': 0.9556, 'role_cos_gap': 0.012,
     'reuse_role_range': [1.30, 1.79], 'reuse_cross_role': 1.18,
     'reuse_same_entity': 7.884, 'reuse_diff_entity': 1.236, 'entity_role_ratio': 6.4},
    [ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json',
        '144 格，去角色均值版并列')],
    BLOCK(1, '三种查询的 Δ 余弦：角色内 0.9676 vs 角色间 0.9556，只差 0.012；'
             'reuse 角色内 1.30–1.79（阈值 3.0）'),
    '角色专属列族从原料上就不成立——三种查询的输出分布形状差异远小于它与自身的相似度。'
    '原担心的「答案词表差异」混杂因此没有出现（若由词表主导，角色间余弦应明显更低，实测只低 0.012）。'
    '输出端真正的结构是**按答案实体**组织的：同答案 7.88x vs 不同答案 1.24x。'
    '这与 3281 的「写入强度 D_norm 预测动态份额（ρ=0.97）」是同一件事从输出端的镜像：'
    '结构住在 q·k 匹配强度里，不在可复用的输出原子集合里。')

add('R-MET-22', 'metric', '稀疏分解',
    '稀疏字典分解的 in-sample R² 可以证明「分解成立」。',
    'refuted', 'high',
    '字典容量 K 接近数据维度 V 时必须报 held-out R²；in-sample 不作证据',
    {'k_dict': 32, 'v': 39, 'r2_insample': 0.989, 'r2_holdout': 0.819, 'gap': 0.170,
     'best_fold_holdout': 0.888},
    [ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json', '按三元组 5 折')],
    BLOCK(1, 'in-sample 0.989 vs held-out 0.819，差 0.17；5 折中最好一折 0.888 仍未过 0.90'),
    'K≈V 时 in-sample R² 必然接近 1（过完备字典对训练点近乎插值），拿它当证据是自欺（D17）。'
    'held-out 差距全部来自泛化，而泛化才是「结构可复用」这个词的实际含义。')

add('R-MET-23', 'metric', 'bootstrap',
    'bootstrap 重采样时保留 (i,i) 自配对不会影响置信区间。',
    'refuted', 'high',
    'CI 下界必须 ≤ 点估计；若 CI 下界 > 点估计 ⇒ 估计器有缺陷，不是发现',
    {'point_est': 1.788, 'ci_lo_before_fix': 1.813, 'ci_hi_before_fix': 2.456,
     'ci_lo_after_fix': 1.52, 'ci_hi_after_fix': 2.10},
    [ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json', '修复后重跑全量')],
    BLOCK(1, '修复前 CI 下界 1.813 > 点估计 1.788（结构性不可能）；修复后 CI [1.52, 2.10] 正常包住'),
    '对集合相似度做 bootstrap 时，同一元素被抽中两次形成的 (i,i) 配对 Jaccard 恒为 1.0，'
    '会把整个分布整体抬高（D18）。凡重采样后仍按「两两配对」聚合的统计量都中这一枪。')

add('R-MTH-12', 'method', '界的宣称',
    '可以用「全局 rank-3 PCA 的 R²」作为「每格 ≤3 原子稀疏编码」的 R² 上界。',
    'refuted', 'high',
    '声称 A 是 B 的界，须先证明两者的可行约束集有包含关系',
    {'pca_rank3': 0.477, 'omp3_measured': 0.541, 'violation': 0.064},
    [ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json',
        'SMOKE 中 OMP 反超「上界」而暴露')],
    BLOCK(1, 'OMP(3) 实测 0.541 高于所声称的上界 0.477——上界被自己的测量推翻'),
    '「每格 ≤3 原子」只约束该格落在**它自选的** 3 个原子张成的子空间里，不同格可选不同原子；'
    '全局 rank-3 PCA 约束所有格共享同一子空间。两者约束集无包含关系，不存在谁界谁（D19）。'
    '凡「某受限模型是另一受限模型的界」的说法，先检查约束集包含关系再开口。')

# ---------------- 方法：PSF-2 范式修订（2026-11-01，Q3-T 三连 NULL 触发） ----------------

add('R-MTH-13', 'method', '必要性单元',
    '「消除后就无效」是判定机制有效性的必要条件，且对任何分析单元（头 / 层 / 神经元 / 位置）同等适用。',
    'refuted', 'high',
    '同一层带上比较「消融参数」与「改位置内容」两种干预；若前者为 0 而后者为满值，则必要性只存在于后者',
    {'ablate_133_heads_L28_35': 0.000, 'patch_L33_single_position': 1.00,
     'ablate_93pct_heads_L0_3': 0.000, 'no_necessary_head': True, 'no_necessary_layer': True},
    [ev('Q2-C1', 'tools/pat_q2c1.py', '—', '消融 133 头 vs patch 单位置，同段对照'),
     ev('Q2-E/F/G', '—', '—', 'L0–L3 内 93% 头可消融零损伤')],
    BLOCK(1, '同段 L28–L35：消融全部 133 头 drop 0.000，patch L33 一个位置翻转 1.00'),
    '消融测的是**容量与冗余**，不是功能；分布式系统按构造必然回答"删掉这个不坏"，'
    '所以消融类结论只在"容量"意义上成立，**永久降级为容量测量、不作机制证据**（D20）。'
    '必要性真实存在，只是住在"位置上的内容"里，不在"头的参数"里。'
    '推论：本项目不必要求每条结论都过消融关（D21 每 Phase 一个干预锚点即可），但不能一条都没有（D22）。')

add('R-MTH-14', 'method', '关闭判据',
    '一个假设家族跑满固定轮数（如 3 轮）即可判定该方向不存在。',
    'refuted', 'high',
    '关闭前须报本轮 n 下的最小可检测效应 MDE；MDE > 关心的效应量时只能判「未测出」，记 pending 而非关闭',
    {'rounds_cap_was': 3, 't3_fold_r2_range': [0.753, 0.888], 't3_mde_est': 0.03,
     't1_n': 1200, 't1_judged_effect': 0.10},
    [ev('PSF-2', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '§9.2 功效闸'),
     ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json',
        '5 折无一达阈，MDE≈0.03 ⇒ 结论稳')],
    BLOCK(1, '轮数是功效的错误代理变量：弱效应小样本被误关，强效应 3 轮又过剩'),
    '「跑完了」≠「测不出」。若某轮 n=36、效应量 0.05、MDE 0.20，那轮没有资格产出 NULL。'
    '算力本身不设上限——但**规模必须由功效反推**，而不是由轮数或时长决定（D23）。'
    '本修订部分替代 D12：轮数只作辅助，功效闸才是关闭依据。')

add('R-MTH-15', 'method', '题型',
    '定位型问题（某功能存在于哪些头 / 哪些层 / 哪些列 / 哪一段）是寻找编码机制的有效途径。',
    'refuted', 'medium',
    '若连续 3 个侧（头 / k / 输出端）的定位测量全部 NULL，则判题型错而非侧错，须换为预测型判据',
    {'rounds_total': 12, 'sides_tested': ['heads', 'k_side', 'output_side'],
     'localization_nulls': 12, 'prediction_type_rounds': 0},
    [ev('PAT-9R', '—', 'research/gpt5/docs/PATTERN_Q_FINDINGS_20261101.md', '头侧九轮全 NULL'),
     ev('Q3-T1', 'tools/pat_q3t_t1.py', 'results/2026-11-01/055514_pat_q3t_t1.json', 'k 侧 NULL'),
     ev('Q3-T2', 'tools/pat_q3t_t2.py', 'results/2026-11-01/055903_pat_q3t_t2.json', 'k 侧分解 NULL'),
     ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json', '输出端 NULL')],
    BLOCK(1, '12 轮同一题型（定位）全部 NULL，且每换一侧都再次确认「没有那个地方」'),
    '定位问题假设"某处存着 X"。分布式表示里这个假设可能根本不成立——不是没找到，是**没有那个地方**。'
    '这是循环的真机制：不是假设不够好、不是算力不够，是**问错了题型**。'
    '改法：问"执行的是什么规则"，用**预测准确率 vs 基线**判决（D24）。'
    '预测型是唯一自带备择、无法用细分换结果的判据形态。'
    '注意这条是 12/12 的归纳结论，不是形式证明，故 confidence 记 medium。')


# ---------------- 注意力头 / 方法：Q4-R 线新增（2026-11-01，PSF-2 首个预测型 Phase） ----------------

add('R-ATT-18', 'attn', '查询选槽',
    '「查询在哪个槽上」这件事无法用一条不含答案词的规则预测，只能靠语言学知识。',
    'refuted', 'high',
    '规则只输出槽位、不含答案词；held-out 准确率 ≥0.80 且 ≥随机+0.40；'
    '同时报位置规则（不看查询）作备择，二者差 ≥0.30',
    {'acc_R1_holdout': 1.000, 'acc_R3_position': 0.3333, 'random': 0.3333,
     'permutation_rate': 1.000, 'lstar': 28, 'lstar_folds_all_same': True,
     'band_onset': 'L19-L24', 'low_layer_equals_position_rule': True,
     'mcnemar_b': 96, 'mcnemar_c': 0, 'p': 2.5e-29},
    [ev('Q4-R', 'tools/pat_q4r.py', 'results/2026-11-01/063929_pat_q4r.json',
        '144 格，24 三元组 x 2 句式 x 3 查询，独立夹具 324 次前向')],
    BLOCK(1, 'R1 144/144（5 折全选 l*=28）；R3 位置规则恰 0.333；置换一致 48/48'),
    '规则 =「末位 query 在三个实体上注意力最大的槽」，不含任何答案词，槽里的词由句子给出。'
    'L18 及以前 R1 与 R3 同为 0.333 ⇒ 低层 argmax 退化为位置选择，查询尚未起作用；'
    'L21 → 0.59，L24 → 0.97，L28 起满分。⇒ **路由决策在 L19–L24，读出在 L33**（与 Q2-C1 衔接）。'
    '注意：R2（静态码）也达 1.000，但 T1 已测同词跨上下文 cos 0.998，k̄≈k 近乎恒等，'
    '故该条只排除"上下文特异改写影响选槽"，不是独立证据。')

add('R-MTH-16', 'method', '判据形态',
    '换测量侧（头 → k → 输出端）能突破"找不到承载者"的困境。',
    'refuted', 'medium',
    '若三个侧连续 NULL，则判题型错而非侧错；改预测型判据后应在 2 轮内产出可报告准确率',
    {'sides_null': 3, 'prediction_type_rounds_to_first_hit': 1,
     'first_hit_acc': 1.000, 'baseline': 0.3333},
    [ev('Q4-R', 'tools/pat_q4r.py', 'results/2026-11-01/063929_pat_q4r.json',
        'PSF-2 §9.3 首个预测型 Phase，一次命中'),
     ev('PSF-2', 'research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md', '—', '§9.5 自检条款')],
    BLOCK(1, '三个侧全 NULL 后，改预测型判据第 1 轮即得 1.000 vs 0.333（p=2.5e-29）'),
    'PSF-2 §9.5 的自检条款预言「预测型判据应在 2 轮内产出可报告准确率而非 NULL」，'
    '实测 1 轮命中 ⇒ 自检通过。'
    '**关键不在"换了哪一侧"，在"换了题型"**：定位问"某处存着 X"，预测问"哪条规则能算出 X"。'
    '前者在分布式表示里系统性失败，后者不依赖"那个地方存在"。')

add('R-MET-24', 'metric', '剖面报告',
    '逐层剖面可以按固定步长抽样打印，抽样不改变结论。',
    'refuted', 'high',
    '剖面必须以完整数组落盘/打印；抽样只作摘要，且须同时给出完整版本',
    {'sampling_step': 3, 'n_layers': 36, 'printed_upto': 33,
     'missed_layers': [34, 35], 'L35_actual': 0.562, 'claimed': 'L28 起稳定满分'},
    [ev('Q4-R', 'tools/pat_q4r.py', 'results/2026-11-01/063929_pat_q4r.json',
        'profile_R1 完整 36 层，L35 塌陷由独立复核发现')],
    BLOCK(1, 'profile[::3] 打印到 L33，跳过 L34/L35；实测 L35 = 0.562，与"稳定满分"相反'),
    '抽样打印是**报告环节的泄漏**：不改数据，但系统性掩盖末端效应。'
    '36 不是步长 3 的端点友好数，而最后几层恰是注意力重新分配最常发生处。'
    '与 D5（双峰禁用均值）、D13（剖面不平均）同类：聚合可以，但聚合不得替换原始分布（D25）。')

add('R-ATT-19', 'attn', '维度分工',
    '注意力头按语义维度分工：不同头分别编码语言、风格、逻辑、语法等维度，同层聚合即多维融合。',
    'refuted', 'high',
    '有信号头（三槽覆盖≥0.20）之间的槽选择一致性 ≥0.80，且按 k 聚合的准确率必须**饱和在 <1**；'
    '若聚合能到 1.000 则分工不成立',
    {'unanimity': 0.875, 'k1': 0.638, 'k32': 1.000, 'sink_vs_content_r': -0.371,
     'n_cells': 144, 'n_signal_heads_median': 16},
    [ev('Q5-A', 'tools/pat_q5a.py', 'results/2026-11-01/070525_pat_q5a.json',
        'L28 k 曲线单调升到 1.000；一致性 0.875')],
    BLOCK(1, '若头分组指向不同槽，聚合必然饱和在中间值（组间抵消），不可能到 1.000。实测到 1.000'),
    '唯一真实的分工是**二分类功能分工**：寄存器（sink）头 vs 内容头，'
    '逐头 sink 质量 [0.087, 0.938]、与指对率 r=-0.371。不是四类语义维度。'
    '这是「多维度加权融合」假设在注意力层的直接否证（negative #23）。')

add('R-ATT-20', 'attn', '注意力主导项',
    '注意力主要落在候选实体上，候选之间的份额即竞争结果。',
    'refuted', 'high',
    '报告位置分解：sink 位置质量、候选槽总覆盖、其余；候选槽覆盖 <0.40 时不得用「候选之间竞争」表述',
    {'sink_pos0': 0.462, 'three_slots': 0.100, 'rest': 0.44, 'pos1': 0.000},
    [ev('Q5-A', 'tools/pat_q5a.py', 'results/2026-11-01/070525_pat_q5a.json',
        'L28 位置分解，pos0=<|im_start|>')],
    BLOCK(1, '三实体槽只占 10%，46% 在句首 sink。Q4-R 的 1.000 是在这 10% 的质量差里取得的'),
    '「竞争」的真实战场是**寄存器 vs 内容**。这条同时是对我自己框架的勘误：'
    '预注册设了 A4（覆盖≥0.40）作为框架前置闸，它在跑完时判 FRAMEWORK_INVALID，'
    '避免了我在错误坐标系里继续解释 A1/A2/A3。')

add('R-MET-25', 'metric', '证据累积',
    '分布式的意思是「没有单独必需的头」，仅此而已。',
    'confirmed', 'high',
    '随机取 k 头聚合并报 k 曲线（k=1,2,4,8,16,32，每点重采样≥200 次）；只报「无必需头」不报曲线视为未测',
    {'k1': 0.638, 'k2': 0.743, 'k4': 0.909, 'k8': 0.957, 'k16': 0.995, 'k32': 1.000,
     'single_head_median': 0.67, 'single_head_max': 1.00},
    [ev('Q5-A', 'tools/pat_q5a.py', 'results/2026-11-01/070525_pat_q5a.json',
        'L28，单调无饱和点，需约 16 头达 0.995')],
    ALLOW(1, '这是把九轮消融的定性结论「93% 头可消融而零损伤」升级为定量：证据按 1/√k 量级累积'),
    'k 曲线还兼作实现正确性检验：四层 k=32 得到 0.590/0.653/0.972/1.000，'
    '与 Q4-R 独立测得的剖面逐位相同。两份独立实现互证，比任何自检都硬。')

add('R-MTH-17', 'method', '框架前置闸',
    '预注册只需冻结判据阈值；判断框架本身是否适用的条件可以事后讨论。',
    'refuted', 'high',
    '预注册必须包含至少一条「框架前置闸」：一个若不过就判框架失效、而非判假设失效的量',
    {'q5a_gate': 'A4 cover>=0.40', 'actual': 0.100, 'verdict': 'FRAMEWORK_INVALID'},
    [ev('Q5-A', 'research/gpt5/docs/PAT_Q5A_PREREG_v1.md',
        'results/2026-11-01/070525_pat_q5a.json', 'A4 未过，直接判框架失效')],
    ALLOW(1, 'Q5-A 实测：A1/A2 未过、A3 过、A4 未过。若无 A4 会拿 A1/A2/A3 继续在错误坐标系解释'),
    '这是 D6（判据须自带备择）的框架级推广：'
    '不只要求判据能区分备择假设，还要求**能区分「假设错」与「坐标系错」**。'
    '两者处置完全不同：前者换假设，后者换测量单位。混在一起就是无限循环的入口。')

add('R-ATT-21', 'attn', '关系方向',
    '语义关系在残差流里以「共享方向」存在：同一关系的不同实例，其差向量 Δ=v_target−v_source 互相共线。',
    'confirmed', 'high',
    '族内「目标词不同」的配对 cos ≥0.30 且族间 margin ≥0.20，**且族内置换（保持目标词多重集）p<0.05**；'
    '须报 37 层的达阈层占比，不得用单层极值',
    {'F3_colour': {'A1b_layers': '30/37', 'margin': 0.359, 'p_perm': 0.000},
     'F4_size': {'A1b_layers': '33/37', 'margin': 0.466, 'p_perm': 0.030},
     'F1_taxonomic': {'A1b_layers': '2/37', 'margin_at_peak': 0.256, 'band': 'L13-L24'},
     'F5_antonym': {'A1b_layers': '0/37', 'peak_within': 0.134},
     'family_cross_F3_F4': 0.081, 'family_cross_F1_F2': 0.208},
    [ev('Q6-A', 'tools/pat_q6a.py', 'results/2026-11-01/072150_pat_q6a.json',
        '54 实例 78 词 3 载体句，200 次置换，1000 bootstrap')],
    ALLOW(1, '**只对属性类（颜色/大小）成立**；反义族 within 仅 0.134，分类族只有 L13–L24 窄带'),
    '这是「复用差分」在本项目第一次以正确对照取得的正面证据。'
    '三个对照缺一不可：A1b 挡「指向同一目标词」、A1p 置换挡「目标词本身相似」、族×族矩阵挡「跨词区跃迁」。'
    '族×族是关键：F3 与 F4 同为名→修饰却只有 +0.081，说明载体是语义关系不是词性。')

add('R-ATT-22', 'attn', '层级编码',
    '分类层级（苹果→水果→食物）由同一支可复用的「抽象方向」反复施加实现。',
    'refuted', 'high',
    '同链 cos(Δ(a→b), Δ(b→c)) 显著为正（对照为跨链配对）；若为负或近零则否证',
    {'same_chain_median': -0.246, 'cross_chain': -0.005, 'peak': -0.172,
     'layers_negative': '37/37', 'radial_monotone_peak': 0.60,
     'radial_layers_pass': '0/37'},
    [ev('Q6-A', 'tools/pat_q6a.py', 'results/2026-11-01/072150_pat_q6a.json',
        '10 条链，37 层全负')],
    BLOCK(1, '同链余弦 37 层全为负、中位 −0.246，而跨链对照精确为 0。'
             '预注册的径向替代解释（越抽象越靠均值）也否证：达阈 0/37'),
    '**负值是系统性的、有结构的**，不是噪声；但「为什么是负的」本轮无解释，'
    '作为开放异常带入下一轮，不得据此推断任何替代机制（E-07/E-08 见 findings §5）。')

add('R-MET-26', 'metric', '置换零假设',
    '用「族内配对 cos 中位」判断关系方向时，跨族比较足以排除目标词相似的伪影。',
    'refuted', 'high',
    '判断关系方向必须做**族内置换**（打乱源—目标配对、**保持目标词多重集不变**），并与跨族对照同时报',
    {'F4_target_self_cos': 0.82, 'F3_target_self_cos': 0.85, 'F1_target_self_cos': 0.80,
     'why': '同族目标词彼此 cos 高达 0.80-0.85，Δ 天然相似'},
    [ev('Q6-A', 'tools/pat_q6a.py', 'results/2026-11-01/072150_pat_q6a.json',
        '目标词自身 cos 0.65-0.85，跨族对照无法吸收')],
    BLOCK(1, 'F4 只有小/大两个目标、两者 cos 0.82，A1b 仍达阈 33/37；'
             '不做置换会把「目标词像」误判成「关系方向」'),
    '与 Q3-T3 的「答案词表混杂」同型，但更隐蔽：T3 混的是角色，这里混的是**族内目标词的彼此相似**。'
    '置换保持多重集，因此把它精确吸收。')

add('R-MET-27', 'metric', 'bootstrap 口径',
    'bootstrap 重采样只要次数够多就可靠，口径可与点估计不同。',
    'refuted', 'high',
    'bootstrap 的重采样单位、留出划分、层选择必须与点估计**完全一致**；'
    '并断言 CI 必须包住点估计，否则判 ESTIMATOR_BUG',
    {'q3t_bug': 'CI 下界 1.813 > 点估计 1.788',
     'q6a_bug': 'CI [0.61,0.84] 整体高于点估计 0.196',
     'cause': '点估计按词留出、bootstrap 用样本内全实例'},
    [ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json',
        '自配对抬高 CI'),
     ev('Q6-A', 'tools/pat_q6a.py', 'results/2026-11-01/072150_pat_q6a.json',
        '样本内/折外口径混用')],
    BLOCK(2, '**同一类错在不同 Phase 连续出现两次**（D18 已立过，但只写了「跳自配对」这一种成因）'),
    '这是 D26 的规则化。关键判据：**CI 不含点估计 ⇒ 估计器错，不是发现**。'
    '两次的实际成因不同（自配对 / 口径混用），说明根因是「估计器与统计量不对齐」，'
    '不是某一个特定 bug ⇒ 故改为断言式检查而不是逐条排查。')

add('R-MTH-18', 'method', '功效闸执行',
    '判据的效应量阈值一旦未达到，就可以判该项不成立。',
    'refuted', 'high',
    '若 CI 上界仍高于判据阈值，只能判 PENDING_power 并给出所需 n；不得判 NOT_*',
    {'A3_acc': 0.375, 'A3_CI': [0.219, 0.562], 'threshold': 0.52, 'n': 32, 'n_needed': 170},
    [ev('Q6-A', 'tools/pat_q6a.py', 'results/2026-11-01/072150_pat_q6a.json',
        'A3 CI 上界 0.562 > 阈值 0.52，脚本原判 NOT_PREDICTIVE 被 D23 覆盖为 PENDING')],
    BLOCK(1, '脚本按阈值判 NOT_PREDICTIVE，但 D23 要求 CI 上界也低于阈值才可判死'),
    '这是 D23 第一次真正改变判决结果——此前 D23 只是写在框架里。'
    '同时给出反推：A3 要判死需约 170 个留出实例，现只有 32。')

add('R-ATT-23', 'attn', '层级编码方向',
    '分类层级（苹果→水果→食物）由同一支可复用的「抽象方向」反复施加实现。',
    'refuted', 'high',
    '同链 cos(Δ(a→b), Δ(b→c)) 必须 ≥0.30（已达阈的共享方向量级）才可称方向；'
    '**判据只能看绝对水平，不得看正负号**——两跳共享中间词，符号为负是几何必然',
    {'obs_all_layer_median': -0.2461, 'obs_max_any_layer': -0.1716,
     'null_rand_median': -0.3994, 'null_level_median': -0.3075,
     'geom_share': 0.399, 'superordinate_share': 0.0919, 'chain_share': 0.0614,
     'ref_F3_median': 0.323, 'ref_F4_median': 0.376,
     'ref_F3_peak': 0.395, 'ref_F4_peak': 0.494,
     't_gt1_frac_obs': 0.0, 't_gt1_frac_null': 0.0, 'rho_median': 0.969},
    [ev('Q6-B2', 'tools/pat_q6b2.py', 'results/2026-11-01/073538_pat_q6b2.json',
        '10 链 × 37 层 × 200 零抽样；G0 闸通过（null −0.399 ∈ [−0.65,−0.35]）')],
    BLOCK(2, 'obs 在**任何一层**最高只到 −0.172，而同一运行里已确立的共享方向是 +0.323/+0.376；'
             '差距 0.49–0.55，不是"弱"，是"不同量级"。'),
    '**上一轮（Q6-A）拒斥层级的理由必须作废**：−0.246 里 0.399 是几何、0.092 是上位词区域、'
    '只有 0.061 属于具体链条。结论保住是因为绝对水平不够，不是因为那个负号。'
    '**−0.246 从此不得单独引用**（负结果#25）。')

add('R-ATT-24', 'attn', '属性知识存储形态',
    '有限参数以低秩因子分解的形式存储「多物体 × 多属性」，故属性知识可由矩阵补全恢复。',
    'refuted', 'high',
    'held-out 格子 R² 必须随秩上升；若 **R=0（仅主效应）最优、加秩单调变差**，则结构不是低秩',
    {'r2_at_R0': 0.197, 'r2_at_R1': 0.118, 'r2_at_R3': -0.406, 'r2_at_R4': -1.059,
     'r2_null_R0': 0.016, 'r2_null_R3': -0.631, 'r2_inverted_at_R': [4, 5, 11],
     'var_row': 0.106, 'var_col': 0.315, 'var_interaction': 0.579,
     'peak_ratio_median': 0.50, 'active_values_per_type_median': 1.0,
     'active_values_per_type_mean': 1.18,
     'cells': 336, 'active_indices': 99, 'compression': '3.4x'},
    [ev('Q6-B', 'tools/pat_q6b.py', 'results/2026-11-01/074005_pat_q6b.json',
        '28 物体 × 12 属性值；C1 8/8+8/8、C2 0.655、C3 6.56 nat 四门全过')],
    BLOCK(1, 'interaction 占 58% 方差却一格都外推不了；真实数据在各秩均优于打乱零分布 ⇒ '
             '结构为真但**不是低秩**，是稀疏'),
    '正确描述是**稀疏指针**：每个实体每属性类型只激活 1.18 个值，且只能取 12 元封闭码本的下标。'
    '与 Q3-T3（复用按答案实体 7.88× vs 按角色 1.24×）同向：**复用的单位是实体，不是关系或属性**。'
    'owner「有限参数记无限属性」成立，但机制是稀疏查表不是低秩分解（负结果#26）。')

add('R-MET-28', 'metric', '差向量余弦基线',
    '两条差向量的余弦可以直接解释为「两个关系的相似度」。',
    'refuted', 'high',
    '凡比较共享词的两条差向量，必须同时报**换词零假设**（保持共享词与符号、只换另一端），'
    '且结论只允许依据「观测 − 零假设」，不得依据原始符号',
    {'iso_prediction': -0.5, 'measured_null_median': -0.3994,
     'measured_obs_median': -0.2461, 'semantic_share': 0.0614,
     'why': 'Δ1=b−a 与 Δ2=c−b 共享 b 且符号相反 ⇒ E[Δ1·Δ2]=−σ²、E[|Δ1||Δ2|]=2σ² ⇒ E[cos]=−0.5'},
    [ev('Q6-B2', 'tools/pat_q6b2.py', 'results/2026-11-01/073538_pat_q6b2.json',
        'null_rand 全层中位 −0.3994，实证各向同性预测')],
    BLOCK(1, 'Q6-A 把 −0.246 当成「层级不是方向」的证据，实际几何基线就是 −0.40；'
             '跨链对照 =0 是因为跨链不共享词，根本无法判定同链负值来源'),
    '这条覆盖绝大多数类比与组合性探针（它们天然比较共享成分的差向量）。'
    '**判据形态**：不看符号看绝对水平；**对照形态**：换词而非换链。')

add('R-MET-29', 'metric', '子空间重叠判据',
    '用「向量落在某子空间之外的份额」可以判定该向量与该子空间是否解耦。',
    'refuted', 'high',
    '在高维（≫子空间维度）空间中，投影份额类判据**饱和**，无分辨力；'
    '须改用回归 R² / CCA 等**有尺度**的统计量',
    {'dec_attr': '0.95–1.00', 'dec_adjective_null': '0.97–1.00', 'diff_range': [-0.0445, -0.0005],
     'dim': 2560, 'subspace_dim': 3, 'volume_ratio': '3/2560 ≈ 0.12%'},
    [ev('Q6-B', 'tools/pat_q6b.py', 'results/2026-11-01/074005_pat_q6b.json',
        '属性值与同词性形容词的 dec 都≈0.99，diff≈0'),
     ev('Q3-T1', 'tools/pat_q3t_t1.py', 'results/2026-11-01/055514_pat_q6t_t1.json',
        'R_id = 1.000 假性满格，同类错第一次'),
     ev('Q3-T3', 'tools/pat_q3t_t3.py', 'results/2026-11-01/061142_pat_q3t_t3.json',
        'share_id 假性 1.00，同类错第二次')],
    BLOCK(3, '**同一类错第三次**：分母含天然极小量（k/H、Var(mean)/Var(个体)、3/2560），'
             '判据被钉在天花板，对一切都成立'),
    '这条把前两次的具体补丁（R-MET-13 换主判据、R-MET-21 三步分解）升格为**形态判据**：'
    '凡是「份额/占比/比例」类统计量，先算它在**随机对象**上的值；若随机对象已接近上限，'
    '该判据不可用。')

add('R-MTH-19', 'method', '续写打分边界',
    '用 teacher-forcing 求「续写总 log-prob」可以测量某个词在上下文中的合适度。',
    'refuted', 'high',
    'tokenize 后必须检查续写首 token 的 offset 起点是否 ≥ 续写起点；'
    '若被并入前文（如 `很大` 是一个 token），则该次比较作废。**优先改用「前缀末位单 token 条件概率」**',
    {'bad_case': '「苹果」通常很大。 → 很大(66,68)，起点 66 < 续写起点 67 ⇒ 被排除',
     'symptom': '大小门 0/8（恰好全错，系统性反转）',
     'fixed_by': 'M[o,a] = log softmax(logits[前缀末位])[id(a)]，前向 672→84'},
    [ev('Q6-B', 'tools/pat_q6b.py', 'results/2026-11-01/074005_pat_q6b.json',
        '改口径后 C1 颜色 8/8、大小 8/8，margin 最高 20.3 nat')],
    BLOCK(1, '中文分词会把「很+形容词」「颜色字+色」并成单 token，续写总 log-prob 静默失效；'
             '门 0/8 是唯一可观测的指纹'),
    '推广：**任何**基于「续写评分」的测量都要先打印一次 token offset 表。'
    '单 token 条件概率口径同时更省（一次前向读出整个属性类型的候选）且无边界问题。')

add('R-MTH-20', 'method', '判决分级',
    '只要「观测 − 零假设」显著，就可以判定该效应成立。',
    'refuted', 'high',
    '判决必须分两级并同时报：**主判看绝对水平**（是否达到该类现象的量级）、'
    '**归因看相对差值**（是否显著高于零假设）。两者不得压成一个数',
    {'q6b2_smoke_diff': 0.19, 'q6b2_obs_absolute': -0.25, 'threshold': 0.30,
     'reference_established': [0.323, 0.376],
     'failure_mode': '差值 +0.19 看似成立，但绝对水平 −0.25 离阈值差 0.55'},
    [ev('Q6-B2', 'tools/pat_q6b2.py', 'results/2026-11-01/073538_pat_q6b2.json',
        'SMOKE 首版按单判决输出 HIERARCHY_PARTIAL，绝对水平 −0.18 从未达阈')],
    BLOCK(1, '把两级压成一级会产出「显著 ⇒ 成立」的 M1 式误读；'
             '且**必须同报参照线**，否则 −0.25 与 +0.40 无法在同一刻度比较'),
    '配套要求：**参照线必须在同一次运行、同一估计器下测得**（本轮 F3 +0.323 / F4 +0.376 即如此），'
    '不得从历史结果里搬。')

add('R-ATT-25', 'attention', '关系码不绑连接词',
    '句间关系（顺承/转折/因果）在模型内部若被编码，必然依附于显式连接词这类表面标记。',
    'refuted', 'high',
    '关系码在**序列末位**即可线性读出，且**去掉全部连接词**只掉 2.5 点（0.9136 → 0.8889）；'
    '只换一个标点（逗号→句号）0.9444。连接词位置本身 1.0000 恰恰说明它不能当证据',
    {'content_heldout': 0.9136, 'connective_removed': 0.8889, 'punct_swap': 0.9444,
     'connective_position': 1.0, 'floor': 0.3333},
    [ev('Q7-A', 'tools/pat_q7a.py', 'results/2026-11-01/080226_pat_q7a.json',
        'J1 0.9136 CI[0.8272,0.9815]；J2 S1S2->S3 0.8889 CI[0.7778,0.9815]；J0 三表面精确 0.333333')],
    BLOCK(1, '若码绑连接词，S1S2->S3 迁移会塌到 1/3；实测只掉 2.5 点，差 0.56'),
    '推广：**测量关系类量的位置必须在"关系已完全确定"处**（此处是第二句末），'
    '而不是关系标记处。在标记处测得的高准确率是同义反复。')

add('R-ATT-26', 'attention', '关系是三点二维不是一条轴',
    '若干语义关系（顺承/转折/因果）可以排在同一条「关系轴」上，像颜色-大小那样是一支方向。',
    'refuted', 'high',
    '三关系在判别平面里构成**等边三角形**（两两 cos ≈ −0.50、第一轴方差占比 0.501）；'
    '秩压缩曲线 rank-1 只到 0.6111、rank-2 即满血 0.9136 ⇒ 存在一条共享轴但它只承载 2/3',
    {'rank1': 0.6111, 'rank2': 0.9136, 'rank3': 0.9136,
     'centroid_cos': [-0.501, -0.499, -0.500], 'axis1_share': 0.501},
    [ev('Q7-A', 'tools/pat_q7a.py', 'results/2026-11-01/080226_pat_q7a.json',
        'J3 秩曲线 + 层16 质心几何（描述性）')],
    BLOCK(1, '一条轴装不下三个等距点；rank-1 与 rank-2 差 0.30，远超内容级 CI 宽度的一半'),
    '与 Q6-A/B2 合看：属性类是「一支方向」（颜色 0.323、大小 0.376），'
    '层级是「区域」，句间关系是「三点二维」。**三类模式的几何形态各不相同**，'
    '所以不存在一张统一的地图。⚠ 此处的 −0.5 是正三角形签名，与 Q6-B2 的 −0.5 无关。')

add('R-MET-30', 'metric', '探针必须带四条基线',
    '隐藏态线性探针在留出条件下准确率高于随机地板，就说明该属性被「编码」了。',
    'refuted', 'high',
    '任何隐藏态探针结果必须同时报四条基线：**字符词袋**（同折同 alpha）、**长度**、'
    '**主导单特征**、以及**第 1 层同口径值**。缺任何一条都不可归因',
    {'hidden': 0.9136, 'char_bag': 0.8025, 'length': 0.3704, 'single_neg_bit': 0.6481,
     'layer1': 0.7840, 'mcnemar_p': 0.0051, 'floor': 0.3333},
    [ev('Q7-A', 'tools/pat_q7a.py', 'results/2026-11-01/080226_pat_q7a.json',
        '隐藏态 148 对 / 词袋 130 对，McNemar p=0.0051；L1 已达 0.7840 ≈ 词袋 0.8025'),
     ev('Q7-A', 'tools/pat_q7a_verify.py', 'results/2026-11-01/080226_pat_q7a.json',
        '12/12 复核通过，含双基线复算')],
    BLOCK(1, '没有词袋基线时 0.91 看起来像"抽象码"，有基线后只剩 +11 点，'
             '且 L1 已 0.78 ⇒ 大部分可读性在任何组合加工之前就存在'),
    '措辞纪律：只允许说「**超出词法的真实信息 +X 点**」，不允许说「已找到抽象码」。'
    '若 X < 0.05，一律判 NOT_BEYOND_LEXICON。')

add('R-MTH-21', 'method', '秩压缩曲线的饱和点',
    '把 C 类探针的权矩阵截断到 rank-k，若需要 k=C 才满血，就说明 C 个类各自有独立方向。',
    'refuted', 'high',
    'C 个类质心最多张 **C−1 维**判别子空间，rank-C 与 rank-(C−1) 恒等（除非类协方差不同）。'
    '故「需要 rank-C」这一行**在数学上不可达**，判决表被结构性地压向中间档',
    {'n_classes': 3, 'rank2': 0.9136, 'rank3': 0.9136, 'equal': True},
    [ev('Q7-A', 'tools/pat_q7a.py', 'results/2026-11-01/080226_pat_q7a.json',
        'rank2 = rank3 = 0.9136，预注册的 PRIVATE_DIRECTIONS 行不可达')],
    BLOCK(1, '三个质心张 2 维是线性代数恒等式，不是经验发现；'
             '把它当成「三支私有方向」的判据等于预置了一个永假的备择'),
    '修正：秩曲线的饱和秩必须显式写成 **C−1**；若确实需要「各自独立」这个备择，'
    '类数须 ≥ 4，或改用「两两转移矩阵 + 共享子空间残差」的问法。')

add('R-MTH-22', 'method', '行为读数不是合格证',
    '可以用一个行为读数（元语言判断、似然比较等）给手写刺激材料发"确实表达了该语义"的合格证，'
    '并据此设一道跑前硬闸。',
    'refuted', 'high',
    '同一批 54 个句对的三种行为口径互相打架：元语言三选一 0.5926（但转折仅 0.278）、'
    '全序列似然 0.4444、B 条件似然去偏 0.4815（转折 0.611）。'
    '**行为读数只能作为诊断，不能作为闸**；多口径不一致本身就是证据',
    {'meta_label': 0.5926, 'meta_label_contrast_only': 0.278,
     'full_seq_likelihood': 0.4444, 'cond_likelihood_debiased': 0.4815,
     'cond_likelihood_contrast': 0.611, 'floor': 0.3333},
    [ev('Q7-A', 'tools/pat_q7a.py', 'results/2026-11-01/080226_pat_q7a.json',
        'SMOKE 阶段三口径全跑，判决链最终不引入 G0')],
    BLOCK(1, '标签先验使模型几乎不吐"转折"；「然后」约束最弱故似然最高。'
             '两者都是方言偏差，不是材料缺陷'),
    '处置范式：行为闸只在结果为 NULL/WEAK 时启用，用来区分「模型不编码」与「材料没表达」；'
    '**永远不能用来认证 PASS**。所有口径必须同时入账，否则就是 gate shopping。')

add('R-MET-31', 'metric', '前缀同一性硬断言',
    '位置对齐可以靠肉眼核对渲染文本与 token 下标来保证。',
    'refuted', 'high',
    '构造刺激时让同一单元内除目标因子外的**前缀逐字相同**，则任意探针在该位置上的预测必然相同，'
    '准确率**必须精确等于 1/C**。这条恒等式是最省的位置对齐断言',
    {'n_classes': 3, 'expected': 0.333333, 'measured_S1': 0.333333,
     'measured_S2': 0.333333, 'measured_S3': 0.333333, 'max_dev': 0.0},
    [ev('Q7-A', 'tools/pat_q7a.py', 'results/2026-11-01/080226_pat_q7a.json',
        'P_AE 三表面精确 1/3，最大偏离 0.00e+00'),
     ev('Q7-A', 'tools/pat_q7a_verify.py', 'results/2026-11-01/080226_pat_q7a.json',
        'V4 独立复算同样精确 1/3')],
    BLOCK(1, '过去三轮的位置 bug（token offset、下标当值、抽样漏层）都是肉眼核对漏掉的；'
             '恒等式断言不依赖核对者的注意力'),
    '推广：任何「同单元多条件」的实验都应预留一个**理论上必须等于地板的测量点**，'
    '一旦偏离即 PIPELINE_LEAK，不解释数据。')

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
    'R-ATT-08': ['角色', '定位头', '落点', 'attention head', 'role', '施事', '受事',
                 '主体', '宾语', '谁', 'agent', 'recipient', '263'],
    'R-ATT-09': ['锥', '角度', '方向', '注入', 'conical', 'direction', '53度', '角色方向'],
    'R-ATT-10': ['角色基', 'basis', 'd1', 'd2', 'd3', '可分离', '解耦', '三角色'],
    'R-ATT-11': ['必需', '关键头', 'essential', 'critical', '崩溃', '组合', '缺了会坏'],
    'R-ATT-12': ['倒U', '中层', '瓶颈', 'middle', '分布形状', '哪层最重要'],
    'R-ATT-13': ['对齐', '承载', '同一批', 'alignment', 'functional', '筛选', '反相'],
    'R-ATT-14': ['覆盖率', '集中度', '损伤', 'damage', '消融掉多少', '删多少', '线性'],
    'R-ATT-15': ['读出', '汇聚', 'L33', '解算', 'readout', '什么时候决定', 'patch 在哪层生效'],
    'R-MET-13': ['注入', 'embedding diff', '嵌入差', 'steering', '激活相加', '向量相加'],
    'R-MET-14': ['混杂', 'confound', '两组不同', '头数不同', '配平', '覆盖'],
    'R-MET-15': ['均值', '平均', 'mean', '双峰', 'bimodal', '方差', '抽样'],
    'R-MET-16': ['smoke', '小样本', '预跑', '外推', 'n=24', '样本不够', '先跑几题'],
    'R-MET-17': ['对照', '长度', '投影', '过冲', '匹配', 'control', '幅度'],
    'R-MET-18': ['地板', 'floor', '三选一', '随机基线', '1/3', '天花板'],
    'R-MET-19': ['判据', 'min', '复合判据', '恒零', '失效'],
    'R-MTH-03': ['规律', '结论', '证据等级', 'E1', 'E2', '剖面', '点陈述', '能不能叫规律'],
    'R-MTH-04': ['对照族', '冻结', '变换', 'transform', '边测边加', '无限检验'],
    'R-MTH-05': ['预算', '关闭', '止损', '轮数', '什么时候放弃', '子空间'],
    'R-MTH-06': ['排除集', '否证', '排除', 'excluded', '已死', '重复走弯路'],
    'R-MTH-07': ['分布式', '分片', 'shard', '合并', '汇总', '多机', '并行'],
    'R-MTH-08': ['失败', '归因', '换假设', '工具', '分辨力', '没结果'],
    'R-MTH-09': ['齿痕', 'key 分解', '身份分量', '上下文分量', 'qk', '归因比'],
    'R-MTH-10': ['R_id', '齿痕强度', '身份占比', 'token identity', '词嵌入', '剖面形状'],
    'R-MTH-11': ['复用', '字典', '稀疏', 'dictionary', 'reuse', '融合', '多模式'],
    'R-ATT-16': ['上下文', 'context', 'key 变', '改写', '齿痕', 'cos', '同一 token'],
    'R-MET-20': ['份额', 'share', '方差解释', '分组', '混合', 'conditional'],
    'R-MET-21': ['分解', 'decomposition', '身份方向', '共同方向', '投影', '正交'],
    'R-ATT-17': ['输出端', '列族', '多模式', '融合形态', '答案实体', '角色模式', 'logit 差'],
    'R-MET-22': ['in-sample', 'held-out', '过完备', '字典容量', '泛化', 'R2'],
    'R-MET-23': ['bootstrap', '置信区间', '自配对', '重采样', 'CI', 'Jaccard'],
    'R-MTH-12': ['上界', '界', 'upper bound', '约束集', '包含关系', 'PCA'],
    'R-MTH-13': ['消融', '必要性', '单元', '位置内容', 'patch', '容量', '范式'],
    'R-MTH-14': ['关闭', '轮数', '功效', 'MDE', 'pending', '误关'],
    'R-MTH-15': ['定位', '题型', '预测', '规则', '机制', '无限循环'],
    'R-ATT-18': ['查询', '选槽', '槽位', 'argmax', '注意力质量', '路由', '预测准确率'],
    'R-MTH-16': ['预测型', '判据形态', '题型', '定位', '突破', '自检'],
    'R-MET-24': ['剖面', '抽样', '打印', '完整数组', '末端', '漏层'],
    'R-ATT-19': ['维度', '分工', '多维度', '融合', '头', '聚合', 'k 曲线'],
    'R-ATT-20': ['sink', '位置分解', '覆盖', '主导', '竞争', '寄存器'],
    'R-MET-25': ['k 曲线', '累积', '分布式', '证据', '必需头'],
    'R-MTH-17': ['框架', '前置闸', '坐标系', '失效', '预注册'],
    'R-ATT-21': ['关系', '方向', '差分', 'cos', '复用', '属性', '颜色', '大小'],
    'R-ATT-22': ['层级', '抽象', '链', '上位', '径向', '层级编码'],
    'R-MET-26': ['置换', 'permutation', '零假设', '目标词', '混杂'],
    'R-MET-27': ['bootstrap', 'CI', '置信区间', '口径', '估计器'],
    'R-MTH-18': ['功效', 'MDE', 'pending', '功效闸', '所需样本量'],
    'R-ATT-23': ['层级', '抽象', '链', '方向', '上位', '几何', '零假设'],
    'R-ATT-24': ['属性', '低秩', '稀疏', '秩', '压缩', '码本', '矩阵补全'],
    'R-MET-28': ['差向量', '余弦', '共享词', '几何基线', '零假设', '换词'],
    'R-MET-29': ['子空间', '投影', '份额', '饱和', '解耦', '分辨力'],
    'R-MTH-19': ['续写', '打分', 'token', '分词', '边界', 'log-prob'],
    'R-MTH-20': ['判决', '绝对', '相对', '分级', '参照线'],
    'R-ATT-25': ['关系', '连接词', '句间', '转折', '因果', '顺承', '表面', '解耦'],
    'R-ATT-26': ['关系', '轴', '等边', '三角形', '秩', '压缩', '二维', '质心'],
    'R-MET-30': ['探针', '基线', '词袋', 'bag', '长度', '单特征', '第 1 层', '归因'],
    'R-MTH-21': ['秩', 'rank', '饱和', '类数', '独立方向', '截断'],
    'R-MTH-22': ['行为', '闸', '合格证', '口径', '元语言', '似然', 'gate'],
    'R-MET-31': ['前缀', '对齐', '断言', '地板', '恒等式', '位置', '索引'],
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
        r = {**prev.pop(r['id'], {}), **r}
        r.setdefault('created_at', now)
        r['updated_at'] = now
        rules.append(r)
    rules.extend(prev.values())

    out = {
        **old,
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
