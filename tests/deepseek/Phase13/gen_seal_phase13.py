# -*- coding: utf-8 -*-
"""Phase 13 预注册（seal）生成器 —— N2h1-alpha-6 位点间配对 bootstrap。

冻结时机：在任何 Phase 13 观测之前。可复现性依据（**均为 Phase 12 已发表量**，
不产生新信息）：`_feas_probe.py` 已用落盘逐对矩阵逐位复现 Phase 12 的
J_ci(18)/top3_share_x/rho_xhalf/R_ci 与两个 2000 值置换零假设，max|d|=0.000e+00。

铁律 (o)：写入走 Python 脚本；本脚本写 seal 后回读复核 + py_compile 不适用（非 py 模块）。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T12 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
S13 = os.path.join(ROOT, 'tests', 'deepseek')
os.makedirs(T13, exist_ok=True)
os.makedirs(os.path.join(S13, 'Phase13'), exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


R12 = os.path.join(T12, 'result_phase12.json')
E12 = os.path.join(T12, 'execution_phase12.json')
SEAL12 = os.path.join(T12, 'N2h1a5_design_seal.json')
AM12 = os.path.join(T12, 'N2h1a5_design_seal_amend1.json')
REP12 = os.path.join(T12, 'n2h1a5_report_qwen3-4b.txt')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')

SEAL = {
    "phase": 13,
    "name": "N2h1-alpha-6 / pairwise-site paired bootstrap (discriminability + concentration coordinate-dependence)",
    "frozen_at_local": "2026-10-02 01:20",
    "frozen_by": "agent (deepseek line)",
    "kind": "zero_extra_forward_reanalysis",

    "one_sentence": (
        "用配对 bootstrap（同一 idx_b 下 J_b(l_i) - J_b(l_{i+1}) 的 95% 带是否含 0）把 "
        "Phase 11 B3 的\"相邻位点不可排序\"从诊断升级为判决，并用量化的尾部概率 "
        "P(share >= 0.60) / P(share <= 0.40) 决断 Phase 12 的 G2_mid；同时检验一个预先声明的"
        "定位假设：Phase 12 的 top3_share_x 是由深尾（L34 反弹）而非浅端过渡驱动的，"
        "从而\"集中度\"在 xhalf 与 J_swap 两个坐标上指向不同的少数层。"
    ),

    "motivation": {
        "from_phase11_B3": (
            "Phase 11 B3：16/17 相邻位点的 J 置信区间互相重叠 ⇒ 位点 J 不可排序；"
            "当时只能用独立区间口径，结论是诊断级而非判决级。"
        ),
        "from_phase12_G2": (
            "Phase 12 G2：top3_share_x = 0.5745，bootstrap 带 [0.3867, 0.8316] 同时跨 0.60"
            "（少层主导）与 0.40（逐层累积）⇒ 按预注册规则判 ALLOCATION_AMBIGUOUS。"
            "带之所以宽，部分来自\"对不可排序的量做集中度统计\"。"
        ),
        "from_phase12_G4": (
            "Phase 12 G4：确认集 4 位点 rho(xhalf)=+0.80 与发现集 -0.7833 符号相反；"
            "事后归因于采样密度不足（4 位点落在非单调剖面的不同支）。需要一个配对检验来判定"
            "L30 谷 / L34 反弹是否为真信号。"
        ),
        "new_observation_motivating_the_pre_registered_localization": (
            "Phase 12 已发表 xhalf.jumps（17 个）中，绝对最大的单个跳变是**最后一个** "
            "+0.0645（L32->L34），占 XH_RANGE=0.10939 的 58.9%；其次才是浅端的 "
            "-0.0263（L9->L10）与 -0.0247（L22->L24）。"
            "=> top3_share_x = 0.5745 的 argmax 窗口极可能在**最深窗口**（j14..j16，即 "
            "L28->L34 段）而非浅端。这一读数完全来自 Phase 12 已封存产物，"
            "故本预测的冻结**不**构成 HARKing。"
        ),
        "death_line_source": (
            "Phase 12 备忘录 §8 最高优先：\"对相邻位点做配对 bootstrap（同一 idx_b 下 "
            "J_b(l_i) - J_b(l_{i+1}) 的 95% 带是否含 0），把\"哪些相邻对真的可分辨\"从诊断"
            "升级为判决\"; 并附\"配对差口径比独立区间口径更紧，是决断集中度问题的正确工具\"。"
        ),
        "secondary_candidate_not_executed": (
            "Phase 12 §8 第二候选（(a) alpha 网格低端加密 / (b) 逐层累积代换 prefix swap）"
            "**需要新前向**，本轮不执行，写入 §8 作为 Phase 14 死线。"
            "本轮以 A8（alpha 网格留一稳健性）作为分辨率问题的**零前向代理**。"
        ),
    },

    "model": {
        "name": "qwen3-4b",
        "note": "本轮**不加载模型**（assert torch 未导入）；全部量来自 Phase 12 落盘矩阵。",
        "layers": {"L": 36, "primary": 6, "pre": 5},
    },

    "object": {
        "source_of_truth": "tests/deepseek_temp/Phase12/result_phase12.json",
        "reconstruction_recipe": {
            "FS_VEC": "np.array([FULL_SWAP_pairs[w] for w in E2['6'][0]['order']]); FULL_SWAP=mean(FS_VEC)",
            "PM_swap": "np.stack([np.array(E2_pairs[str(s)],float) for s in sites.profile],0)  # (18,14,24)",
            "PM_R": "np.array(E6_pairs,float)  # (14,24)",
            "PM_conf": "np.stack([np.array(E5_pairs[str(s)],float) for s in sites.conf],0)  # (4,4,17)",
        },
        "pair_alignment_verified": {
            "order_identical_across_all_sites_and_alphas": True,
            "n_discovery_pairs": 24,
            "n_confirmation_pairs": 17,
            "order_words": ["苹果", "香蕉", "梨", "西瓜", "狗", "猫", "老虎", "大象", "汽车", "火车",
                            "飞机", "摩托车", "桌子", "椅子", "床", "沙发", "铁", "铜", "铝", "金",
                            "红", "蓝", "绿", "黄"],
        },
        "sites": {
            "discovery_profile": [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            "confirmation": [7, 11, 20, 34],
            "readout_R_used_in_A0": True,
        },
    },

    "dose_coordinate": {
        "alpha_grid": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        "unit_trap_A": "xhalf 是 alpha（无量纲替换比例）；J 是斜率比（无量纲）。两者与 Phase 8-11 的注入剂量坐标（绝对/相对范数）**不可比**，只可跨坐标比**秩与形状**。",
        "unit_trap_B": "本轮所有判决量均在 xhalf/J 自身坐标内做配对差，不引入新的物理单位。",
    },

    "statistics_definition": {
        "J_only": "max(adjacent slope) / median(remaining slopes)，xs >= 0.01 过滤，14 点网格。**逐字复制 Phase 12**（极值比统计量 ⇒ bootstrap 分布右偏）。",
        "J_alt_iqr": "A9 用：max(adjacent slope) / IQR(remaining slopes)，用于检验陡度定义不改变秩结构。",
        "xhalf": "cross_alpha(alpha_grid, y, 0.5)，y=dDonor/FULL_SWAP 的逐对均值曲线，线性插值。",
        "delta_paired": "Delta_b(i) = F_b(site_i) - F_b(site_{i+1})，同一 bx idx_b（配对）。F ∈ {J, xhalf}。",
        "band": "percentile 2.5/97.5，B=2000。",
        "label": "DECISIVE_DOWN (hi<0) / DECISIVE_UP (lo>0) / TIE (lo<=0<=hi)。",
    },

    "arms": {
        "A0_replicate_phase12": {
            "what": "重放 BRNG 流，逐位复现 Phase 12 的 J_ci(18x3 分位)/top3_share_x_ci/top3_share_recover_ci/rho_recover/rho_xhalf/R_ci(recover,xhalf) 与两个 2000 值置换零假设。",
            "role": "装置锚（非判决量）",
            "assertion": "max|d| == 0.0 （逐位）",
        },
        "A0b_replicate_confirmation_band": {
            "what": "继续重放 BRNG（confirmation 段），复现 Phase 12 的 E5.rho_boot_xhalf 的 lo/hi/med。",
            "role": "第二个装置锚",
            "assertion": "max|d| < 1e-12",
        },
        "A1_delta_J": {
            "what": "17 个相邻对的 Delta_b(J)，观测 Delta_hat、percentile 带、DECISIVE/TIE 标签。",
            "role": "主判决量 1（B3 的直接延伸）",
        },
        "A2_delta_xhalf": {
            "what": "同 A1，坐标换成 xhalf（18 个位点全部 XH 非 None，无需剔除）。",
            "role": "主判决量 2（G4 符号矛盾 + G2 口径）",
        },
        "A3_tightening_attribution": {
            "what": "逐对计算 sd_paired = sd(Delta_b)、sd_indep = sqrt(var(J_b[:,i])+var(J_b[:,i+1]))、rho_pair = corr(J_b[:,i], J_b[:,i+1])、tighten = sd_paired/sd_indep。",
            "role": "把\"配对口径更紧\"从断言变成归因",
            "hard_assertions": [
                "rho_pair > 0  => sd_paired < sd_indep （逐对，全部满足）",
                "sd_indep^2 - sd_paired^2 == 2*cov(J_i,J_{i+1})  (max|d| < 1e-12，纯代数)",
            ],
        },
        "A4_concentration_tail_probabilities": {
            "what": "对 xhalf 与 J_swap 两个坐标分别算并报 P(share>=0.60)、P(share<=0.40)、P(mid)，以及 bootstrap 中 argmax 窗口起始索引的频次直方图。窗口 W=3。",
            "role": "把 Phase 12 的\"带跨阈值\"升级为量化尾部概率 + 定位",
        },
        "A5_independent_vs_paired_count": {
            "what": "独立区间口径（Phase 12 的 J_ci 带两两不重叠）与配对口径（A1）的可分辨对数对比；报收紧比中位数。",
            "role": "把 Phase 11 B3 的\"16/17 重叠\"升级为\"配对下有多少对可分辨\"",
        },
        "A6_deep_tail_reversal": {
            "what": "对 (L28->L30)、(L30->L32)、(L32->L34) 三对的 Delta 带（xhalf 与 J 各一份），判定 L30 谷 / L34 反弹是否为真信号。",
            "role": "回应 Phase 12 G4 的符号矛盾",
        },
        "A7_confirmation_paired": {
            "what": "确认集（4 位点 -> 3 对，n=17 pairs）的 Delta 带；分母沿用 Phase 12 确认集口径 FULL_SWAP 常数（逐字复现便于 A0b 锚定）。",
            "role": "方向一致性检查（不具细结构复现能力）",
        },
        "A8_alpha_grid_leave_one_out": {
            "what": "逐个剔除 alpha_grid 中除 0.0 与 1.0 外的每个点，重算每位点 xhalf、XH_RANGE、top3_share_x；报 min/max/极差。",
            "role": "分辨率稳健性（零前向代理 Phase 14 的网格加密）",
        },
        "A9_steepness_statistic_robustness": {
            "what": "用 J_alt_iqr 重算 18 位点剖面，报 spearman(J, J_alt)、以及 J_alt 下的 N_dec。",
            "role": "检验判决不是 max/median 比这一特定统计量的产物",
        },
    },

    "metrics": {
        "primary": [
            "N_dec_J, N_dec_X, N_dec_indep_J",
            "Delta_J_obs[17], Delta_J_band[17], Delta_J_label[17]",
            "Delta_X_obs[17], Delta_X_band[17], Delta_X_label[17]",
            "P_few_x, P_acc_x, P_mid_x, P_few_j, P_acc_j, P_mid_j",
            "win_argmax_hist_x, win_argmax_hist_j",
            "P13_verdict",
        ],
        "secondary": [
            "sd_paired[17], sd_indep[17], rho_pair[17], tighten[17]",
            "A8 range_le1_top3, range_le1_XH_RANGE",
            "A9 spearman(J,J_alt), N_dec_J_alt",
            "A7 conf Delta bands (3 pairs, xhalf + J)",
        ],
    },

    "curve_classifier_parameterized": {
        "note": "沿用 Phase 10/11/12 逐字口径，本轮未使用（本 Phase 主量为配对差）。",
        "sig": "y = A*sigmoid(k*(x-x*))，A=max(y)",
        "k_grid": "1..60 step 1", "x_star_grid": "[x_min,x_max] step 0.005",
        "classes": {"UNREACH": "|y|max<0.15", "S_STRONG": "R2_log>=0.98 & J>=3.0 & k_log>=2.0",
                    "S_WEAK": "R2_log>=0.95 & J>=2.0", "GRADUAL": "R2_log>=0.95 & J<2.0",
                    "LINEAR": "R2_lin>=0.97 & R2_log<0.95"},
    },

    "bootstrap": {
        "scheme": "pair-level percentile bootstrap (B=2000, B_perm=2000)",
        "seed": 20261001,
        "seed_note": "与 Phase 12 **同一 seed**，且 BRNG 的首个消费点即主循环 ⇒ 可逐位复现（A0/A0b 的前提）。独立段的第二次抽样不在本轮。",
        "ci": "percentile 2.5/97.5",
        "paired_key": "同一 b 下所有位点/坐标共用同一 idx_b。",
        "zero_extra_forward": True,
        "cpu_only": True,
    },

    "pre_registered_predictions": {
        "P1_xhalf_concentration_is_deep": (
            "xhalf 的 bootstrap argmax 窗口众数 = 最后一个窗口（起始 jump 索引 14，覆盖 L28->L34），"
            "频次 >= 0.50。理由：Phase 12 已发表 jumps 中 |+0.0645| 为最大且占总变程 58.9%。"
        ),
        "P2_J_concentration_is_shallow": (
            "J_swap 的 bootstrap argmax 窗口众数落在**最前**三个窗口（起始索引 0/1/2，覆盖 L6->L12），"
            "频次 >= 0.50。理由：Phase 12 已发表 J_swap 剖面 25.35->15.49->8.72 的骤降在浅端。"
        ),
        "P3_jump_profiles_not_aligned": (
            "spearman(|jumps_x|, |jumps_j|) <= 0.2（两个坐标的跳变大小剖面不对齐）。"
        ),
        "falsifiable": "三条预测任一被否证，则\"两个坐标指向不同少数层\"的定位叙述必须改写；若三条同时成立，则 Phase 12 的 G2_mid 应重新解释为坐标系依赖性而非纯抽样不确定性。",
    },

    "decision": {
        "G0p_precondition": {
            "require": "A0 max|d| == 0.0 AND A0b max|d| < 1e-12",
            "fail_verdict": "DEVICE_ANCHOR_FAILED",
        },
        "D_discriminability": {
            "N_dec_J": "# of 17 adjacent pairs with band excluding 0 (J_swap)",
            "N_dec_X": "same for xhalf",
            "expectation_recorded_before_run": "N_dec_X 仍可能为 0：xhalf 的 XH_RANGE 只 0.1094，单对 Delta 的绝对量级 ~0.003-0.06，可能全部落在带内。",
        },
        "D_concentration": {
            "P_few_x": "P(top3_share_x_b >= 0.60)",
            "P_acc_x": "P(top3_share_x_b <= 0.40)",
            "P_few_j": "P(top3_share_j_b >= 0.60)",
            "P_acc_j": "P(top3_share_j_b <= 0.40)",
            "W_x": "argmax window start index (mode over bootstrap) for xhalf",
            "W_j": "same for J_swap",
            "coord_dep_rule": "|share_x|>=0.40 AND |share_j|>=0.40 AND |W_x - W_j| >= 3",
        },
        "verdict_table": [
            ["G0p fail", "DEVICE_ANCHOR_FAILED"],
            ["N_dec_J == 0 AND N_dec_X == 0", "PAIRED_TEST_UNINFORMATIVE"],
            ["coord_dep_rule true", "CONCENTRATION_COORDINATE_DEPENDENT"],
            ["P_few_x >= 0.95 AND P_few_j >= 0.95", "CONCENTRATION_FEW_LAYER_ROBUST"],
            ["P_acc_x >= 0.95 AND P_acc_j >= 0.95", "CONCENTRATION_ACCUMULATE_ROBUST"],
            ["otherwise", "CONCENTRATION_UNDECIDED"],
        ],
        "P13_verdict_scope": "本判决只解决\"集中度是否是坐标系无关性质\"；**不**直接断定\"少层主导\"或\"逐层累积\"为真。",
    },

    "floors": {
        "F14_A0_bit_replication": "A0: max|d| == 0.0 across J_ci(18x3) + top3_share_x_ci(3) + top3_share_recover_ci(3) + rho_recover(3) + rho_xhalf(3) + R_ci(4) + perm_x(2000) + perm_rec(2000).",
        "F15_telescoping_hat": "sum_i Delta_hat(i) == J_hat(0) - J_hat(16) (max|d| < 1e-12).",
        "F16_telescoping_boot": "per b: sum_i Delta_b(i) == J_b[b,0] - J_b[b,16] (max|d| < 1e-12).",
        "F17_variance_decomposition": "sd_indep^2 - sd_paired^2 == 2*cov (max|d| < 1e-12).",
        "F18_band_contains_hat_fraction": "report frac of the 17 pairs whose band contains Delta_hat (NOT asserted; percentile bands need not contain the point estimate).",
        "F19_probability_conservation": "|P_few + P_acc + P_mid - 1| < 1e-12 for both coordinates.",
        "F20_confirmation_delta_present": "A7 yields 3 Delta bands for xhalf and 3 for J, all finite.",
        "F21_no_forward_no_torch": "assert 'torch' not in sys.modules at entry AND at exit; assert no model files opened.",
        "F22_conf_bootstrap_replicate": "A0b: E5.rho_boot_xhalf lo/hi/med replication.",
        "F23_grid_LOO_finite": "A8: all 12 leave-one-out variants give finite XH_RANGE and top3_share_x.",
    },

    "honesty": [
        "1. 配对 bootstrap 只刻画**同一受试集内**的重采样不确定性；带变窄**不**提高跨实例/跨类别的可迁移性（N2h1 leave-class-out 0.45-0.50、水果类 0.04-0.05 的限界不变）。",
        "2. J_only 是极值比（max slope / median slope），bootstrap 分布右偏（Phase 12 已见 L6 hat 25.35 / 带 [16.77, 57.24]）；Delta 带**不对称**，DECISIVE 标签对右尾敏感。必须同时报 med 与 hat 的偏离。",
        "3. xhalf 的配对带受**alpha 网格分辨率**限制（14 点，浅端 0-0.2 只 5 点）；TIE 标签可能被网格粗化人为放大。A8 只给**下界意义**的稳健性，不替代真正的网格加密。",
        "4. 17 个相邻对**不独立**（共用位点）⇒ 不做多重比较校正的量级声明；N_dec 只作**描述性计数**，其含义不由 p 值定义。",
        "5. 确认集只有 4 位点（3 对）⇒ 只作方向一致性检查，不能复现 17 对的细结构。",
        "6. 全部量仍是**激活级**、**单模型（qwen3-4b）**、**单模板（X是一种）**、**发现集 n=24**。",
        "7. A0/A0b 的逐位复现**只证明实现与 Phase 12 一致**（同一数据、同一实现谱系），**不**构成对 Phase 12 结论的独立验证。",
        "8. 本轮为**纯再分析**：不产生新前向、不涉及模型权重；所得结论仍是\"读出传递函数\"层面的，非机制实现级。",
        "9. pre_registered_predictions 的三条**是在冻结前写下的**；若被否证必须如实记录，不得事后改写成\"本来就没预期\"。",
        "10. 结论若为 CONCENTRATION_COORDINATE_DEPENDENT，则不得声称\"少层主导\"或\"逐层累积\"中的任一个为装置的固有性质；只能说\"在 xhalf 坐标下集中，在 J 坐标下不集中（或反之）\"。",
    ],

    "may_falsify_the_whole_line": {
        "condition": "N_dec_J == 0 AND N_dec_X == 0 AND P_few/P_acc 四个尾部概率全部 < 0.95。",
        "verdict": "PAIRED_TEST_UNINFORMATIVE",
        "consequence": "在 qwen3-4b/发现集上，\"层贡献\"在抽样不确定性下**不可分解**；配对口径不足以决断集中度。下一步必须从**再分析**转向**新干预**（prefix swap 第三坐标）与**跨模型**，而不是继续在 n=24 上加密统计。",
    },

    "artifacts": {
        "script": "tests/deepseek/Phase13/n2h1a6_paired_site.py",
        "exec": "tests/deepseek_temp/Phase13/execution_phase13.json",
        "result": "tests/deepseek_temp/Phase13/result_phase13.json",
        "report": "tests/deepseek_temp/Phase13/n2h1a6_report_qwen3-4b.txt",
        "judgement": "tests/deepseek_temp/Phase13/judgement_phase13.json",
        "disk_verify": "tests/deepseek/Phase13/disk_verify_phase13.py",
        "feas_probe": "tests/deepseek/Phase13/_feas_probe.py -> tests/deepseek_temp/Phase13/_feas_probe.txt",
    },

    "inheritance_anchors": {
        "phase12_result_sha256": sha(R12),
        "phase12_result_sha8": sha(R12)[:8],
        "phase12_exec_sha256": sha(E12),
        "phase12_seal_sha256": sha(SEAL12),
        "phase12_amend1_sha256": sha(AM12),
        "phase12_report_sha256": sha(REP12),
        "memo_sha256_pre_append": sha(MEMO),
        "memo_sha256_pre_append_8": sha(MEMO)[:8],
        "ledger_sha256_pre": sha(LEDGER),
        "ledger_sha256_pre_8": sha(LEDGER)[:8],
    },

    "why_not_a_HARKing_violation": (
        "本轮所有判据、阈值、预测与裁决表均在**任何 Phase 13 观测之前**写入本文件。"
        "唯一引用的既有数字（xhalf.jumps 的 +0.0645 最大、J_swap 剖面的浅端骤降）"
        "均来自 Phase 12 **已封存并已在备忘录发表**的产物，属\"已有知识\"而非\"本轮数据窥视\"。"
        "并且 _feas_probe.py 的复现只重建了 Phase 12 已发表的量（J_ci/top3/rho/perm），"
        "未计算任何 Delta 或尾部概率。"
    ),
}

OUT = os.path.join(T13, 'N2h1a6_design_seal.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(SEAL, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE', OUT)
print('bytes', len(b), 'sha256', hashlib.sha256(b).hexdigest())
print('sha8', hashlib.sha256(b).hexdigest()[:8])

# 回读复核
J = json.load(io.open(OUT, encoding='utf-8'))
print('readback keys', len(J.keys()))
assert J['phase'] == 13
assert J['bootstrap']['seed'] == 20261001
assert J['inheritance_anchors']['phase12_result_sha8'] == '7bf4510a'
assert len(J['pre_registered_predictions']) == 4
assert len(J['floors']) == 10
assert len(J['honesty']) == 10
assert len(J['decision']['verdict_table']) == 6
print('READBACK OK')
