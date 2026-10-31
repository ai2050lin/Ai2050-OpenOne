# 规则账本（自动导出）

> 每条规则 = 一条可判定命题 + 结论 + 证据。口径不变时不要重复测。

## 已证实（9）

### R-ATT-01 注意力头对答案的贡献集中在晚期层、在"答案位"读出；问题位与选项位几乎没有显著头。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：|t|>4（Bonferroni）的显著头按读出位置分布明显不均
- 关键数字：n_significant=58，layer_range=L24–L35，at_q_end=0，at_opt_end=2，at_answer=58
- 为什么：这条确立了"头级归因"的可行口径：只看答案位。也让 58 个显著头成为后续分析的主力单元。
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/pos_profile.json`

### R-ATT-04 induction 头与答案 DLA 显著头显著重叠，是答案读出的重要载体。

- 领域：注意力头 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：DLA 显著头中 induction 类型 ≥10 个（随机期望约 58×46/1152≈2.3）
- 关键数字：induction_in_significant=13，n_significant=58，n_induction_total=46，expected_random=2.3
- 为什么：induction 头双带分布（L2/L4/L7 + L13/L20–25/L28–33），与答案位显著头带（L24–35）在后一带重合。
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/head_types.json`

### R-ATT-06 存在"早看晚写"分工：早期层（L0–L6）的头对选项注意力高达 0.85–0.98 但对答案 DLA 几乎为零，晚期层同时具备两者。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：早层头注意力质量 ≥0.8 且 |t|<1；晚层头 ≥0.75 且 |t|>8
- 关键数字：early_quality=0.85–0.98，early_t=<1，late_examples=L26H26 0.80/t=11.2, L24H31 0.78/t=14.6
- 为什么：原假设"头在答案位就读答案"被否证，换成这条更强的分工描述。
- 证据：A1 ｜ `tools/mech_pos_probe.py` ｜ `data/mech/pos_profile.json`

### R-MET-05 分组消融的效应近似可加（各组 KL 之和 ≈ 整层 KL）。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：各组之和 / 整层的比值 ∈[0.7, 1.2]
- 关键数字：sum_groups=0.1474，full_layer=0.1708，ratio=0.86，n_groups=24
- 为什么：加性成立说明层内没有强协同，但也说明"用组做单位"并不会放大效应——真正的问题是组本身不对应功能（见 R-MLP-05）。
- 证据：C1 ｜ `tools/mech_c1_full_run.py` ｜ `data/mech/ablate_L24.json`

### R-MLP-01 一阶归因（attribution patching）选出的 top-1% MLP 单元，跨数据集是稳定的：在 C-Eval 上选出的集合拿到 CMMLU 上仍是同一批。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：top-97 重叠 ≥20（随机期望 0.97）且 |Δ| 排序 spearman ≥0.30
- 关键数字：overlap=65/97，spearman=0.607，kl_ratio=54，n_layer=24
- 为什么：这是本项目唯一可跨数据集引用的单元级结论：稳定性来自模型权重本身，不是题目用词。
- 证据：H1 ｜ `tools/mech_holdout.py` ｜ `tempdata/holdout_L24.json`

### R-MLP-03 删掉 top-1% 单元会显著改变模型给出的答案（改答率显著高于删同量随机单元）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：改答配对的 McNemar p<0.01，且改答率 ≥2× 随机对照
- 关键数字：only_top_changed=28，only_random_changed=4，p=<0.0001，rate_top=0.231，rate_random=0.089，ratio=2.6
- 为什么：这条是"杠杆真实存在"的最强行为证据。它同时说明：用正确率判杠杆会得到假阴性——因为改答是双向的（见 R-MLP-04）。
- 证据：H5 ｜ `tools/mech_cot_analyze.py` ｜ `tempdata/mech_cot_L34_n169.json`

### R-MLP-08 层间差异可以用"集中度"（top-1% 占 Σ|Δ| 的比例）量化，且集中度与跨数据集稳定性严格同向。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：三层集中度单调，且与 top-97 重叠数、spearman 同向变化
- 关键数字：conc_L14=0.079，conc_L24=0.147，conc_L34=0.37，overlap_L14=37，overlap_L24=65，overlap_L34=81，spearman_L14=0.15，spearman_L24=0.607，spearman_L34=0.761
- 为什么：这是唯一一个"可预筛层"的量化指标：先测集中度，再决定值不值得做单元级分析。
- 证据：H1b/H1c ｜ `tools/mech_holdout.py` ｜ `tempdata/holdout_L14.json`
- 证据：C4' ｜ `tools/mech_unit_attr.py` ｜ `tempdata/unit_attr_L14.npz`

### R-MLP-09 L30 是沉默层（删掉整层对答案几乎无影响），不是"分布式层"。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：整层消融 KL <0.02 且翻转数不高于随机
- 关键数字：full_kl_L30=0.0128，conc_L30=0.176，compare_L24_kl=0.1708
- 为什么：上轮把 L30 读成"分布式"是错的：集中度与 L24 同级，但整层效应量小 13 倍。四层结构是三状态（分布式 / 杠杆层 / 沉默层），不是一条曲线。
- 证据：C4' ｜ `tools/mech_unit_attr.py` ｜ `tempdata/unit_attr_L30.npz`

### R-MLP-12 集中式晚期层（L34）的 top-97 中存在可直接推出选项字母的单元，且相对全层显著富集。

- 领域：MLP 单元 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：top-97 中 readout#1 为字母的单元数 ≥3，且富集 ≥5×
- 关键数字：n_letter_units=3，share_of_effect=0.089，baseline=20/9728=0.21%，enrichment=15，L14_L24_count=0
- 为什么：第三条独立证据（不依赖 KL、不依赖行为）。同时说明其余 94 个单元的 readout 词仍无族——"例外正是那几个推字母的"。
- 证据：H4附录 ｜ `tools/mech_top_units.py` ｜ `tempdata/top_units_L34.md`

## 已否证（15）

### R-ATT-02 用头的权重几何（写方向对答案字母的投影 / 能量 / 有效秩）可以预测该头的 DLA 显著性。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：corr(几何量, |t|) ≥0.6
- 关键数字：corr_w_letter_max=0.349，corr_energy=0.174，corr_ov_pr=0.026，significant_vs_not_ratio=1.15
- 为什么：与 R-MLP-10 同构：权重几何在这个模型上不能替代实测。
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/heads_dir.json`

### R-ATT-03 prev-token 头参与答案读出（在 DLA 显著头中占相当比例）。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：DLA 显著头中 prev-token 类型 ≥5 个
- 关键数字：prev_in_significant=0，n_significant=58，n_prev_total=74
- 为什么：prev-token 头（74 个，集中 L0–L6）做的是局部复制，不参与答案决策——这本身是有价值的分工证据，不必再测。
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/head_types.json`

### R-ATT-05 头在选项上的注意力质量可以预测它的 DLA 显著性。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：corr(注意力质量, |t|) ≥0.5
- 关键数字：corr_all_heads=0.103
- 为什么：但这条失败换来一个更好的发现：早看晚写（见 R-ATT-06）。
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/pos_profile.json`

### R-ATT-07 在 GQA 结构下可以对每个注意力头独立做归因（头级分辨率 32）。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同组内 4 个头的读取通路应彼此独立
- 关键数字：n_query_heads=32，n_kv_heads=8，group_size=4，true_dof=9728×8
- 为什么：32 个 query 头只有 8 套 K/V。所有"头级"结论必须降一档理解——同组 4 头共享读写通路，"独立头"是假象。
- 证据：G' ｜ `server/mech_pair_service.py` ｜ `data/mech/pair_gain.json`

### R-MET-01 KL 散度可以作为 MLP 单元"重要性 / 必要性"的度量。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：top-k 的扰动能量应集中在高概率 token 上（energy_share_top20 ≥3× 均匀基线）
- 关键数字：energy_share_top20=0.0007，uniform_baseline=0.00013，ratio=5.4，letter_share=0.0002
- 为什么：KL 测的是 logit 长尾的抖动幅度，不是答案决策的因果贡献。用 KL 排序会把"让输出变毛糙的单元"误判成"重要单元"。这是本轮最重要的一条方法论否证。
- 证据：H3b ｜ `tools/mech_delta_eff.py` ｜ `tempdata/mech_delta_eff.json`

### R-MET-02 删掉"同规模的随机单元"是匹配的弱对照，可以用来和删 top-k 对比。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：随机对照的扰动规模应与实验组同量级（中心化范数比 0.5–2）
- 关键数字：norm_ratio_raw=8.16，norm_ratio_centered=0.13，kl_random=0.00089
- 为什么：删除小批随机单元约等于给 logits 加一个常数，softmax 完全不动。所以"删随机 97 没影响"不能证明"被删的随机单元不重要"，只能证明这个对照太弱。
- 证据：H3 ｜ `tools/mech_delta_cos.py` ｜ `tempdata/mech_delta_cos.json`

### R-MET-03 未中心化的 Δlogits 范数可以直接用来比较不同干预的扰动规模。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：两种干预的范数比与它们的 KL 比同向
- 关键数字：raw_ratio_top1pct=1.73，raw_ratio_random=8.16，kl_ratio=54
- 为什么：必须先把 Δlogits 投影到与输出均值正交的子空间再比范数，否则常数平移完全污染比较。
- 证据：H3 ｜ `tools/mech_delta_cos.py` ｜ `tempdata/mech_delta_cos.json`

### R-MET-04 n=40 的生成任务足以判定"删 top-k 造成 7.5 个百分点的下降"。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：McNemar 配对检验 p<0.05（不能只看点估计差）
- 关键数字：p_at_n40=0.453，discordant=5:2，need_n=684，effect_disappeared_at_n169=True
- 为什么：n=40 时 7.5 点的差距 p 只有 0.453。任何"掉几个百分点"的结论必须给 McNemar p 与 discordant 计数。
- 证据：H4→H5 ｜ `tools/mech_cot_analyze.py` ｜ `tempdata/mech_cot_L34_n169.json`

### R-MLP-02 删掉一阶归因 top-1% 的 MLP 单元，会使生成任务的答案正确率显著下降。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 4 次 ｜ 不要重测
- 判据：McNemar 配对检验 p<0.05 且正确率下降 ≥10 个百分点
- 关键数字：acc_base=0.651，acc_top1pct=0.633，acc_random=0.663，p=0.42，n=169，layer=34
- 为什么：L34 40 题曾测到 −7.5 点，扩到 169 题后消失（前 40 题 base 0.850 vs 全量 0.651，是偏容易子集）。结论：删 top-k 不是测必要性的有效操作。
- 证据：H5 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L34_n169.json`
- 证据：H4 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L24.json`

### R-MLP-05 按共现把 MLP 单元聚成组，得到的组是"参数共享的功能子模块"（组内写方向应当一致）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：组内写方向平均 |cos| 显著高于零模型（同规模随机单元对）
- 关键数字：cos_within=0.017，cos_null=0.0198，lift=1.0，jaccard_lift=4.66–14.45
- 为什么：共现分组是"语境模块"（同类文本里一起被激活），不是"计算模块"（共享参数、协同工作）。这是整个 C 段路线失败的直接原因。
- 证据：A2 ｜ `server/mech_pair_service.py` ｜ `data/mech/unit_clusters.json`

### R-MLP-06 共现聚类在换随机种子 / 换样本后能稳定复现。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：换初值或重采样 80% 样本后，两次聚类的 Jaccard ≥0.6
- 关键数字：jaccard_seed=[0.434, 0.414, 0.455]，jaccard_resample80=[0.187, 0.282, 0.325]
- 为什么：簇是采样噪声的产物；任何以簇为单位的因果结论都不成立。
- 证据：C2' ｜ `tools/mech_cluster_stability.py` ｜ `tempdata/cluster_stability.json`

### R-MLP-07 共现聚类的簇对应可命名的语义类别（token 类型 / 语种 / 标点）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：AMI(簇, 类别) 显著高于零（≥0.3）
- 关键数字：ami_char=0.045–0.184，ami_language=0.041–0.144，punct_ratio=0.115 vs 0.093
- 为什么：上轮把 C19 判成"排版模块"是采样偏差（全层真正最偏标点的是 C8）。教训：簇不承载语义。
- 证据：C2' ｜ `tools/mech_token_types.py` ｜ `tempdata/cluster_token_types.json`
- 证据：C2' ｜ `tools/mech_cluster_axis.py` ｜ `tempdata/cluster_axis.json`

### R-MLP-10 用权重几何（列范数 / 能量 / 有效秩）可以预测 MLP 单元的重要性排序。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：与 |t| 的相关系数 ≥0.6
- 关键数字：corr_col_norm=0.174，corr_ov_pr=0.026，corr_w_letter_max=0.349
- 为什么：看权重猜不出谁重要，只能实测。这条否证了"剪枝按范数"的常见做法在这个模型上的有效性。
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/heads_dir.json`

### R-MLP-11 用"读取增益"（下游 W_K/W_V 能读到多少）可以预测 MLP 单元的重要性。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：corr(增益, |t|) ≥0.5 且逐对稳定
- 关键数字：corr_mean=0.226，corr_median=0.138，pairs=41
- 为什么：这条失败后流水线由 W→R→G→C→P 简化为 W→R→C→P，省掉一整个环节。
- 证据：G' ｜ `server/mech_pair_service.py` ｜ `data/mech/pair_gain.json`

### R-MLP-13 删掉整层 9728 个 MLP 单元会翻转大量答案。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：翻转数显著高于随机（≥20/169）且答案分布的熵明显上升
- 关键数字：flip=5/169，d_margin=0.02，entropy=0.621→0.395，top1_still_letter=1.0
- 为什么：回答不是"层不重要"，而是层的输出被下游层冗余补偿；同时 KL 指标测的是长尾（见 R-MET-01）。
- 证据：C5' ｜ `tools/mech_degrade_check.py` ｜ `tempdata/degrade_check_L24.json`
- 证据：H5 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L34_n169.json`

## 未决（2）

### R-MTH-01 在输出处投影掉组平均写方向（dir 干预）可以定位该组的功能，且效应可被匹配对照分辨。

- 领域：实验方法 ｜ 置信：medium ｜ 已测 1 次 ｜ 重测需加样本
- 判据：dir 干预 KL ≥3× 匹配 sham/gauss 对照，且与层主写方向 |cos| 可解释
- 关键数字：kl_c18=0.02592，sham_mean=0.00209，ratio=12.4，cos_with_layer_dir=≤0.09
- 为什么：KL 是 sham 的 12.4×，但投影方向与层主写方向的 |cos|≤0.09——权重几何看不见、只有干预能看见。这个反差本身可能只是"投影掉了噪声子空间"。
- 证据：C1 ｜ `tools/mech_c1_full_run.py` ｜ `data/mech/ablate_L24.json`

### R-MLP-04 高杠杆 MLP 单元是成对正反编码（差分对）：删掉它们时，破坏正确计算与修复错误计算同时发生，净效应接近零。

- 领域：MLP 单元 ｜ 置信：medium ｜ 已测 1 次 ｜ 重测需加样本
- 判据：单侧删除（只删高 +Δ 或只删高 −Δ）应产生有方向的偏差；双侧删除只产生无方向抖动
- 关键数字：wrong_to_right=11，right_to_wrong=14，cancel_ratio_behaviour=0.88，logit_abs_sum=633.7，logit_net_sum=16.1，cancel_ratio_logit=0.975
- 为什么：两层、两种因变量给出同一个抵消率（0.88 与 0.975），这是"差分编码"最直接的证据；但它仍是假说——缺单侧删除的对照。
- 证据：H5 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L34_n169.json`
- 证据：C5' ｜ `tools/mech_unit_attr.py` ｜ `tempdata/unit_attr_L24.npz`

## 待测（1）

### R-MTH-02 steering（沿方向加）与 ablation（删掉）在同一组单元上效应强度相当。

- 领域：实验方法 ｜ 置信：low ｜ 已测 0 次 ｜ 可以重测
- 判据：同组同规模的加 vs 删，行为改变量比值 ∈[0.5, 2]
- 关键数字：status=not_tested
- 为什么：这是差分对假说最具判别力的一条：加是不对称操作、删是对称操作。如果 steering ≫ ablation，就能解释整个领域"控制有效、消融无效"的普遍困惑。
