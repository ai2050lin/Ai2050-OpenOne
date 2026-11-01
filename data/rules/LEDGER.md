# 规则账本（自动导出）

> 每条规则 = 一条可判定命题 + 结论 + 证据。口径不变时不要重复测。

> 由 `python tools/rules_export_md.py` 从 `data/rules/ledger.json` 生成，不要手改。

## 已证实（23）

### R-ATT-01 注意力头对答案的贡献集中在晚期层、在"答案位"读出；问题位与选项位几乎没有显著头。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：|t|>4（Bonferroni）的显著头按读出位置分布明显不均
- 关键数字：n_significant=58，layer_range=L24–L35，at_q_end=0，at_opt_end=2，at_answer=58
- 为什么：这条确立了"头级归因"的可行口径：只看答案位。也让 58 个显著头成为后续分析的主力单元。
- 复查：三个位置各跑一次，分布极不均匀（58 / 2 / 0）
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/pos_profile.json`

### R-ATT-04 induction 头与答案 DLA 显著头显著重叠，是答案读出的重要载体。

- 领域：注意力头 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：DLA 显著头中 induction 类型 ≥10 个（随机期望约 58×46/1152≈2.3）
- 关键数字：induction_in_significant=13，n_significant=58，n_induction_total=46，expected_random=2.3
- 为什么：induction 头双带分布（L2/L4/L7 + L13/L20–25/L28–33），与答案位显著头带（L24–35）在后一带重合。
- 复查：5.7× 富集，方向明确
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/head_types.json`

### R-ATT-06 存在"早看晚写"分工：早期层（L0–L6）的头对选项注意力高达 0.85–0.98 但对答案 DLA 几乎为零，晚期层同时具备两者。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：早层头注意力质量 ≥0.8 且 |t|<1；晚层头 ≥0.75 且 |t|>8
- 关键数字：early_quality=0.85–0.98，early_t=<1，late_examples=L26H26 0.80/t=11.2, L24H31 0.78/t=14.6
- 为什么：原假设"头在答案位就读答案"被否证，换成这条更强的分工描述。
- 复查：两条带同时存在，且 P6（注意力位置决定贡献）已被证伪
- 证据：A1 ｜ `tools/mech_pos_probe.py` ｜ `data/mech/pos_profile.json`

### R-ATT-14 注意力消融造成的损伤 ≈ 覆盖率 × 集中度：删得越广、越集中在少数层，掉得越多。

- 领域：注意力头 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：同等头数下，集中在少数层的 drop 显著高于铺满全层；比值应 >2
- 关键数字：wide_128_4layers=0.674，flat_144_36layers=0.197，ratio=3.4，floor=0.3333，max_drop=0.667
- 为什么：这条可以直接用来给任何「删 X 掉 Y」的结论做期望值校准：看到 drop 之前先算它本该掉多少。串联共享通路 + 损伤预算，是当前最稳的一条定量规律。
- 复查：3.4× 差距稳定复现；轻度超可加、重度饱和
- 证据：Q2-E ｜ `tools/pat_q2e_additivity.py` ｜ `results/2026-11-01/*_pat_q2e_additivity.json`
- 证据：Q2-G ｜ `tools/pat_q2g_distribution.py` ｜ `results/2026-11-01/*_pat_q2g_distribution.json`

### R-ATT-15 末位存在一个读出汇聚点 L33：L33 之前上游还能改写末位内容，L33 之后不能。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：真实激活修补（非注入）：把干净句的施事位置激活写回坏句，看答案恢复率随写入层的衰减点
- 关键数字：patch_L0_29=1.0，patch_L30_32=decay，patch_L33_on=0.0，last_pos=opposite，third_entity_decay=L32 (0.21)
- 为什么：推论：**角色解算发生在 L33 之前**。所以任何「角色编码长什么样」的问题，都必须在 L20–L32 窗口里问，L33 及以后问到的都是读出几何。
- 复查：阶跃而非连续：L32 仍有 1.00，L33 直接 0.00
- 证据：Q2-C1 ｜ `tools/pat_q2c1_patch.py` ｜ `results/2026-11-01/*_pat_q2c1_patch.json`

### R-ATT-21 语义关系在残差流里以「共享方向」存在：同一关系的不同实例，其差向量 Δ=v_target−v_source 互相共线。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 可以重测
- 判据：族内「目标词不同」的配对 cos ≥0.30 且族间 margin ≥0.20，**且族内置换（保持目标词多重集）p<0.05**；须报 37 层的达阈层占比，不得用单层极值
- 关键数字：F3_colour={'A1b_layers': '30/37', 'margin': 0.359, 'p_perm': 0.0}，F4_size={'A1b_layers': '33/37', 'margin': 0.466, 'p_perm': 0.03}，F1_taxonomic={'A1b_layers': '2/37', 'margin_at_peak': 0.256, 'band': 'L13-L24'}，F5_antonym={'A1b_layers': '0/37', 'peak_within': 0.134}，family_cross_F3_F4=0.081，family_cross_F1_F2=0.208
- 为什么：这是「复用差分」在本项目第一次以正确对照取得的正面证据。三个对照缺一不可：A1b 挡「指向同一目标词」、A1p 置换挡「目标词本身相似」、族×族矩阵挡「跨词区跃迁」。族×族是关键：F3 与 F4 同为名→修饰却只有 +0.081，说明载体是语义关系不是词性。
- 复查：**只对属性类（颜色/大小）成立**；反义族 within 仅 0.134，分类族只有 L13–L24 窄带
- 证据：Q6-A ｜ `tools/pat_q6a.py` ｜ `results/2026-11-01/072150_pat_q6a.json`

### R-MTH-03 分布剖面（E1）可以独立作为「规律」；干预实验（E2）单独只能作为「线索」。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同一条结论若只由 E2 支撑，在下一轮换任何对照后都应预期失效
- 关键数字：E2_conclusions_retracted=3，E2_survived=0，E1_conclusions_retracted=0
- 为什么：一夜九轮里被撤回的结论 100% 来自 E2（点陈述），没有一条来自 E1（剖面）。**这是本项目「无限测试」循环的直接成因**：用点陈述承担了规律的角色（D10）。
- 复查：框架级规则，不需要重测
- 证据：PSF-1 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-09 注意力分数可以按「钥匙的三项分解」精确拆开：s = q·k_cmn + q·k_id + q·k_ctx，三项正交、和为恒等。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：拆解后三项方差份额之和 ≈1；若某项占比 >0.95 且另两项 ≈0，说明分解退化（未扣共同方向）
- 关键数字：n_cells=114，share_cmn_med=0.543，share_id_L0=0.95，share_ctx_med=0.009，share_sum_med=1.05
- 为什么：这是本线第一个不含任何旧混杂的测量。前提是**必须先扣掉所有词共有的方向**再取身份方向，否则身份项会吃掉全部能量（share_id 假性 = 1.00）。
- 复查：同步前向的代数分解，无消融/注入/移植，三种旧混杂均不适用；份额和 1.05 说明三项略正相关
- 证据：Q3-T2 ｜ `tools/pat_q3t_t2.py` ｜ `results/2026-11-01/055903_pat_q3t_t2.json`

### R-MET-05 分组消融的效应近似可加（各组 KL 之和 ≈ 整层 KL）。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：各组之和 / 整层的比值 ∈[0.7, 1.2]
- 关键数字：sum_groups=0.1474，full_layer=0.1708，ratio=0.86，n_groups=24
- 为什么：加性成立说明层内没有强协同，但也说明「用组做单位」不会放大效应——真正的问题是组本身不对应功能（见 R-MLP-05）。
- 复查：不要重测
- 证据：C1 ｜ `tools/mech_c1_full_run.py` ｜ `data/mech/ablate_L24.json`

### R-MET-18 三选一强制选择任务的行为地板是 1/3，任何 drop 的上限是 0.667。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：强制在三个候选首 token 上取 argmax，随机基线的期望准确率 = 1/3
- 关键数字：floor=0.3333，max_drop=0.667，observed_floor_hits=Q2-E 三条件全部 0.3333
- 为什么：报 drop 前必须先报「离地板还有多远」（D4）。本项目曾因撞地板而误判「三条件同效」，实际三者都已经到底，无法区分。
- 复查：解析可算，且实测三次触底
- 证据：Q2-G ｜ `tools/pat_q2g_distribution.py` ｜ `results/2026-11-01/*_pat_q2g_distribution.json`

### R-MET-25 分布式的意思是「没有单独必需的头」，仅此而已。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 可以重测
- 判据：随机取 k 头聚合并报 k 曲线（k=1,2,4,8,16,32，每点重采样≥200 次）；只报「无必需头」不报曲线视为未测
- 关键数字：k1=0.638，k2=0.743，k4=0.909，k8=0.957，k16=0.995，k32=1.0，single_head_median=0.67，single_head_max=1.0
- 为什么：k 曲线还兼作实现正确性检验：四层 k=32 得到 0.590/0.653/0.972/1.000，与 Q4-R 独立测得的剖面逐位相同。两份独立实现互证，比任何自检都硬。
- 复查：这是把九轮消融的定性结论「93% 头可消融而零损伤」升级为定量：证据按 1/√k 量级累积
- 证据：Q5-A ｜ `tools/pat_q5a.py` ｜ `results/2026-11-01/070525_pat_q5a.json`

### R-MLP-01 一阶归因（attribution patching）选出的 top-1% MLP 单元，跨数据集是稳定的：在 C-Eval 上选出的集合拿到 CMMLU 上仍是同一批。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：top-97 重叠 ≥20（随机期望 0.97）且 |Δ| 排序 spearman ≥0.30
- 关键数字：overlap=65/97，spearman=0.607，kl_ratio=54，n_layer=24
- 为什么：这是本项目唯一可跨数据集引用的单元级结论：稳定性来自模型权重本身，不是题目用词。
- 复查：两套抽样（科目配对 / 全科目随机）结果几乎相同（0.607 / 0.605），与科目构成无关
- 证据：H1 ｜ `tools/mech_holdout.py` ｜ `tempdata/holdout_L24.json`

### R-MLP-02 删掉一阶归因 top-1% 的 MLP 单元，会使生成任务的答案正确率显著下降（L34 已证实；L24 样本不足，未判定）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 5 次 ｜ 不要重测
- 判据：只在 base 本来就能解析出答案的题上比较（条件的 None 仍算错）：下降 ≥5 点，且 P vs 随机删除 的 McNemar p<0.05
- 关键数字：acc_base=0.733，acc_top1pct=0.66，acc_random=0.72，drop_pp=-7.3，p_top_vs_random=0.0074，p_top_vs_base=0.0213，p_random_vs_base=1.0，n=150，layer=34，L24_n_valid=35，L24_p=0.5，L24_note=严格口径仅 35 题，方向甚至 P 略高（0.943 vs 0.914），无法判定
- 为什么：原判 refuted 是解析失败造成的假象：base 有 19 题没写出答案行（记作 None → 算错），删 top-97 把其中 18 题修好了（格式修复），于是 P 的正确率被抬高、base 被压低，两者看起来相等。只在 base 有效的 150 题上比：P 0.660 vs base 0.733（−7.3 点），随机删 0.720 与 base 无差异（p=1.0）。L24 那轮的 40 题严格子集只有 35 题，p=0.5，且方向相反（P 0.943 > base 0.914），⇒ **不能说 L24 不成立，只能说未判定**。是否「必要性取决于层集中度」需要 L24 上 169 题重测。
- 复查：L34 已定论（不需要重测）；L24 需要 169 题重测才能判定。口径见 R-MET-06
- 证据：H5+更正 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L34_n169.json`
- 证据：H4 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L24.json`

### R-MLP-03 删掉 top-1% 单元会显著改变模型给出的答案（改答率显著高于删同量随机单元）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：改答配对的 McNemar p<0.01，且改答率 ≥2× 随机对照
- 关键数字：only_top_changed=16，only_random_changed=0，p=<0.0001，rate_top=0.132，rate_random=0.021，ratio=6.3，n_valid=144，raw_only_top=28，raw_only_random=4
- 为什么：这条是"杠杆真实存在"的最强行为证据。严格口径（排除解析失败题）下更强：0.132 vs 0.021（6.3×），且随机删除一次都没改答。原口径的 28/4 里混入了格式修复，见 R-MET-06。
- 复查：严格口径下 只P改 16 / 只rand改 0，效应量极大，不需要更多题量
- 证据：H5+更正 ｜ `tools/mech_cot_analyze.py` ｜ `tempdata/mech_cot_L34_n169.json`

### R-MLP-08 层间差异可以用"集中度"（top-1% 占 Σ|Δ| 的比例）量化，且集中度与跨数据集稳定性严格同向。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：share_top1pct（top-1% 占 Σ|Δ|）三层单调，且与 top-97 重叠数、spearman 同向；注意 share 与 kl_ratio（删 1% 的 KL / 删整层的 KL）是两个不同的量，不可互替
- 关键数字：share_top1pct_L14=0.079，share_top1pct_L24=0.147，share_top1pct_L34=0.37，kl_ratio_L14=0.086，kl_ratio_L24=1.0，kl_ratio_L34=1.115，overlap_L14=37，overlap_L24=65，overlap_L34=81，spearman_L14=0.15，spearman_L24=0.607，spearman_L34=0.761
- 为什么：这是唯一一个"可预筛层"的量化指标：先测集中度，再决定值不值得做单元级分析。
- 复查：三层各跑一次，单调关系清楚
- 证据：H1b/H1c ｜ `tools/mech_holdout.py` ｜ `tempdata/holdout_L14.json`
- 证据：C4' ｜ `tools/mech_unit_attr.py` ｜ `tempdata/unit_attr_L14.npz`

### R-MLP-09 L30 是沉默层（删掉整层对答案几乎无影响），不是"分布式层"。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：整层消融 KL <0.02 且翻转数不高于随机
- 关键数字：full_kl_L30=0.0128，conc_L30=0.176，compare_L24_kl=0.1708
- 为什么：上轮把 L30 读成"分布式"是错的：集中度与 L24 同级，但整层效应量小 13 倍。四层结构是三状态（分布式 / 杠杆层 / 沉默层），不是一条曲线。
- 复查：KL 比 L24 低一个数量级
- 证据：C4' ｜ `tools/mech_unit_attr.py` ｜ `tempdata/unit_attr_L30.npz`

### R-MLP-12 集中式晚期层（L34）的 top-97 中存在可直接推出选项字母的单元，且相对全层显著富集。

- 领域：MLP 单元 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：top-97 中 readout#1 为字母的单元数 ≥3，且富集 ≥5×
- 关键数字：n_letter_units=3，share_of_effect=0.089，baseline=20/9728=0.21%，enrichment=15，L14_L24_count=0
- 为什么：第三条独立证据（不依赖 KL、不依赖行为）。同时说明其余 94 个单元的 readout 词仍无族——"例外正是那几个推字母的"。
- 复查：普查数据是静态的，重算不会变；但只覆盖 9.7% 效应量，不能推广成"整层=字母单元"
- 证据：H4附录 ｜ `tools/mech_top_units.py` ｜ `tempdata/top_units_L34.md`

### R-MLP-17 L34 的 top-1% 池存在一个可测的饱和点：删到某个 k 之后的额外删除不再显著增加行为损伤。

- 领域：MLP 单元 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：Δ(k) 达到 Δ(97) 的 80%（−4.74 pt）时的最小 k 记 k80；k∈{10,20,40,60,80}，同 SEED 20261101，base 须与 H5/H6 逐题一致否则整轮作废
- 关键数字：curve_pt={'10': 0.0, '20': -1.48, '40': -0.74, '60': -3.6, '80': -5.04, '97': -5.93}，k80=80，flat_below=40，cliff=40→80，damaged_questions={'10': 3, '20': 5, '40': 4, '60': 10, '80': 13, '97': 14}，n_strict=135，base_acc=0.7481
- 为什么：剂量饱和确实存在但位置很高（k80 = 80/97），而且**题目级饱和远早于剂量级饱和**：tailN 的 23 个单元就打坏了 11/14 道会被打坏的题。所以「饱和」有两个不同的量，不可混用：删到 80 个才达到 80% 的准确率损伤，但删到 23 个就已经打坏了几乎全部会被打坏的题。
- 复查：k∈{10,20,40,60,80} 已测完，曲线已覆盖 40–97 全段；再测中间点只是重复
- 证据：H8 ｜ `tools/mech_dose.py` ｜ `tempdata/dose_L34.json`
- 证据：H10 ｜ `tools/mech_headtail.py` ｜ `tempdata/headtail_L34.json`

### R-MLP-19 L34 top-1% 的必要性是近二值的池依赖：覆盖到承载单元就一次性打坏同一小批题，覆盖不到就是零损伤。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：base 答对的题里，被各子集打坏的应是同一批题（重叠高），且损伤不随子集大小线性增长
- 关键数字：base_correct_q=110，damaged_by_pool=14，damaged_by_random97=1，damaged_by_all_8_conditions=[73, 5]，damaged_by_ge6_conditions=7，fragile_subjects={'chinese_civil_service_exam': 5, 'high_school_geography': 3, 'others': 4}，specificity_vs_random=14x
- 为什么：这是本轮最重要的结构结论：不是「池内某类单元重要」，而是「池子里有一小批题依赖的承载单元，任何覆盖到它们的子集都会打坏同一批题」。⇒ 「−5.9 点」实际上是「12 道题」，所有子集比较的误差都以几道题计，所以 pos45 / neg45 / both90 / k60 / k80 之间的无差异是分辨率不足，不是真的等效。
- 复查：这一条已经把「哪些题会坏」定死（12–14 道），不要再按子集比较准确率百分点——分辨率只有几道题
- 证据：H5+H6+H8+H10 合并逐题 ｜ `tools/mech_dose.py / mech_headtail.py` ｜ `tempdata/dose_L34.json + headtail_L34.json`

### R-MLP-21 L34 tailN 池里存在可点名的「承载核」：[767, 1540, 35] 三个单元承担 tailN 损伤的 3/4，且损伤题集完全被 tailN 包含。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：三个单元（占 tailN Σ|Δ| 12.7%、全层排名 62/66/67）全量 169 题上打坏 9/110 道 base 对的题，且这 9 道全部 ⊆ tailN 的 12 道（C1 零例外）；归因量 2.2 倍的 core5 只打坏 5 道
- 关键数字：core3=[767, 1540, 35]，core3_sum_abs=1.196，tailN_sum_abs=9.43，share_of_tailN=0.127，layer_rank=[62, 66, 67]，n_base_right=110，core3_damaged=9，core5_damaged=5，tailN_damaged=12，core3_delta_pt=-4.67，tailN_delta_pt=-6.67，core5_delta_pt=-3.33，core3_not_in_tailN=0，tailN_not_in_core3=[21, 25, 78]，same_107q_broken={'core3': 8, 'core5': 3, 'tailN': 10, 'k97': 13, 'rand97': 3}，write_dir_pairwise_cos_max=0.029，p_values={'core3_vs_base': 0.0654, 'core3_vs_rand97': 0.0703, 'core3_vs_core5': 0.3438, 'core3_vs_tailN': 0.25, 'tailN_vs_base': 0.0129}
- 为什么：H11 已证「单元级归因不能预测单题脆弱性」，但那只否掉了「用归因画像挑单元」这条路径，没有否掉「承载核存在」。H12 换测法（自适应二分 + 少题高重复）后 3 个单元就复现了 7 道，H13 在全量上确认 9/12 且零例外。两条自然对照最能说明问题：Σ|Δ| 几乎相等（2.626 vs 2.588）的两半损伤 2 vs 7；同一父集合内 Σ|Δ| 更高的那一半反而更轻。**只能读结构不能读显著**：效应只有 9–10 道题，core3 vs base p=0.065、vs rand97 p=0.070、vs core5 p=0.344 都不过 0.05，唯一显著的是 tailN vs base（0.0129）。题池只有 169 道，这是分辨率的硬上限——不得把本条写成「显著比随机更伤」。
- 复查：H12（17 题）与 H13（169 题）两轮独立确认；结构结论已定，不要再按子集比准确率百分点
- 证据：H12 ｜ `tools/mech_carrier.py` ｜ `tempdata/carrier_L34.json`
- 证据：H13 ｜ `tools/mech_core.py` ｜ `tempdata/core_L34.json`
- 证据：H14-pre ｜ `tools/mech_core_wiring.py` ｜ `tempdata/core_wiring_L34.json`
- 证据：H5 参照 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L34_n169.json`

### R-MLP-23 L34 的单个单元 767 打坏 11/110 道 base 对的题（保留率 90.0%），与 core3 三个一起（9 道）、tailN 23 个（12 道）同量级；但它既不包含 core3 的全部损伤，也不满足预注册的随机对照阈值。

- 领域：MLP 单元 ｜ 置信：medium ｜ 已测 3 次 ｜ 不要重测
- 判据：单点在全量 110 道 base 对的题上打坏 ≥7 道（O1）；O2 用 10 个随机单点在 17 题上做对照；O3 要求单点题集包含 core3 的 9 道；O4 要求不劣于 core3
- 关键数字：single=767，core3=[767, 1540, 35]，n_base_ok=110，reserve_rate={'s767': 0.9, 'core3': 0.918, 'core5': 0.955, 'tailN': 0.891}，damaged_full={'s767': 11, 'core3': 9, 'core5': 5, 'tailN': 12}，s767_set=[5, 11, 25, 29, 41, 61, 73, 78, 94, 97, 125]，core3_set=[5, 29, 41, 61, 73, 94, 97, 125, 147]，overlap=8，s767_only=[11, 25, 78]，core3_only=[147]，O1=True，O2=False，O3=False，O4=True，change_rate_150={'s767': 0.107, 'core3': 0.093, 'tailN': 0.147, 'k97': 0.14, 'rand97': 0.06}，no_letter_bias={'A': 3, 'B': 1, 'C': 5, 'D': 4}，caliber_fix=H13 的 delta_pt 是净效应/150，H17 是毛损伤/110，两者不可相减；本条的保留率统一按 110 道 base 对的题算
- 为什么：「1 ≈ 3 ≈ 23」这个量级关系是两轮独立测出来的，但它只说明**打坏几道题**这件事与单元数无关，不说明 767 是「知识承载单元」——H18 把机制指向格式（见 R-MLP-25）。O2 判否的原因不是特异性不存在，而是判据用「10 次抽样的最大值」当零分布标尺，被一道通用脆性题（qi=73）抬高了一个（见 R-MLP-24 与 R-MET-11）。按身份比：767 的 9 道里有 6 道是 10 个随机单点一次都没碰到的。
- 复查：H15（17 题）+ H17（169 题）+ H18（机制）三轮一致；不要再按子集比准确率百分点，也不要写成「单点比三合一更狠」——差 2 道题在 n=110 上没有分辨力
- 证据：H15 ｜ `tools/mech_core_split.py` ｜ `tempdata/core_split_L34.json`
- 证据：H17 ｜ `tools/mech_core_one.py` ｜ `tempdata/core_one_L34.json`
- 证据：H18 ｜ `tools/mech_logit_effect.py` ｜ `tempdata/logit_effect_L34.json`

### R-MLP-24 存在与「删哪个单元」无关的通用脆性题：qi=73 被 10 个随机单点里的 8 个弄坏，且两种模态都有（真改答 / 解析崩）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：同一批题上，随机单点损伤的并集若集中在极少数几道上，且这些题同时被主条件与对照组打坏，则判为 pool-generic
- 关键数字：qi=73，gold=A，base=A，random_single_hits={'n_units': 10, 'n_hit_73': 8, 'union_all': [25, 73, 78], 'counts': {'73': 8, '25': 3, '78': 1}}，modalities={'s767': 'no_answer_line（解析失败）', 'r2098': 'answer_C（真改答，gold 是 A）', 'r7964': 'no_answer_line（解析失败）'}，s767_6_exclusive_of_null=[5, 29, 41, 94, 97, 125]
- 为什么：这一条修正 R-MLP-19/22 的名单口径：旧的「12 道脆弱题」是八个条件的并集，里面混着至少一道与单元无关的通用脆性题。后果很实际——用计数判特异性时，通用脆性题会让主条件与随机对照各加一，差别被压缩。
- 复查：两级证据：计数（8/10）+ 原文（真改答与解析崩各一例）。引用「12 道脆弱题」时必须区分 pool-generic 与 pool-specific
- 证据：H17 ｜ `tools/mech_core_one.py` ｜ `tempdata/core_one_L34.json`
- 证据：H17-probe ｜ `tools/mech_probe_frag.py` ｜ `tempdata/probe_frag_L34.json`

### R-MLP-25 单元 767 是格式承载单元而不是知识单元：它在「分析：」之后抑制换行，消融后模型另起一段重写分析，答案随之漂移；字母层面的判断几乎不受它影响。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 4 次 ｜ 不要重测
- 判据：三条同时成立：① 答案位置换行族 logit 抬高 ≥5 且 11 道题推同一批 token；② 固定前缀（teacher forcing）下字母 argmax 不变；③ 自由生成时首 token 变空白/换行 ≥5 道且格式崩坏的题答案随之改变
- 关键数字：q=[5, 11, 25, 29, 41, 61, 73, 78, 94, 97, 125]，H18_promoted={'\n': [9.06, 10.69], '\r': [4.75, 5.63], '\r\n': [3.28, 3.81]}，H18_top5_jaccard=1.0，H18_letter_delta_max=0.125，H18_abs_median_delta={'s767': 0.125, 'rand_max': 0.0156, 'ratio': 8.0}，H19_fixed_prefix={'same_argmax': '9/9', 'kl_median': 0.0, 'non_saturated': {'qi=29': 0.0021, 'qi=94': 0.0019}, 'caveat': '7/9 道 p=1.000 饱和 ⇒ 单独不能下结论'}，H20_prefix_sweep={'argmax_changed': {'25%': 0, '50%': 0, '75%': 0, '100%': 1}, 'margin_shift_median_25pct': 0.031, 'margin_shift_max': 0.12, 'letter_prob_median_25pct': 0.584, 'gold_gap_range_25pct': [-5.2, 1.7]}，H21_free_gen={'first_tok_ws': {'base': '1/11', 'ablated': '8/11'}, 'format_broken': 7, 'ans_changed_among_broken': 7, 'non_newline_but_changed': [41, 61, 73]}，design_waste=H19 第一版强制「 答案：」前缀不中性（qi=5 base 由 D 变 C，因为删掉了推理），第一个样本即废弃，代价 75 秒
- 为什么：这条改写了 H12–H17 的定性：单元级必要性真实存在、高度特异，但机制是格式/轨迹。H20 是四条里最硬的一条——25% 前缀上读出不再饱和（字母概率中位 0.58、gold 间隔 −5.2…+1.7），而 767 对间隔的位移上界只有 0.12 个 logit，与换行的 9–10.7 差两个数量级。H19 单独看是弱证据（7/9 饱和），必须配 H20 才能下结论——这一点在文档里已写明。下一步见 R-MET-12：用格式不变读出重做池级必要性。
- 复查：H18 机制 + H19 固定前缀 + H20 前缀扫描 + H21 行为闭环，四轮一致；不要再写「767 承载 10 道题的知识」，要写「767 维持答题格式」
- 证据：H18 ｜ `tools/mech_logit_effect.py` ｜ `tempdata/logit_effect_L34.json`
- 证据：H19 ｜ `tools/mech_format_test.py` ｜ `tempdata/format_test_L34.json`
- 证据：H20 ｜ `tools/mech_prefix_scan.py` ｜ `tempdata/prefix_scan_L34.json`
- 证据：H21 ｜ `tools/mech_head_tokens.py` ｜ `tempdata/head_tokens_L34.json`

## 已否证（72）

### R-ATT-02 用头的权重几何（写方向对答案字母的投影 / 能量 / 有效秩）可以预测该头的 DLA 显著性。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：corr(几何量, |t|) ≥0.6
- 关键数字：corr_w_letter_max=0.349，corr_energy=0.174，corr_ov_pr=0.026，significant_vs_not_ratio=1.15
- 为什么：与 R-MLP-10 同构：权重几何在这个模型上不能替代实测。
- 复查：三个几何量全部低相关，显著头只有非显著头的 1.15×
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/heads_dir.json`

### R-ATT-03 prev-token 头参与答案读出（在 DLA 显著头中占相当比例）。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：DLA 显著头中 prev-token 类型 ≥5 个
- 关键数字：prev_in_significant=0，n_significant=58，n_prev_total=74
- 为什么：prev-token 头（74 个，集中 L0–L6）做的是局部复制，不参与答案决策——这本身是有价值的分工证据，不必再测。
- 复查：0/58 是干净的零
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/head_types.json`

### R-ATT-05 头在选项上的注意力质量可以预测它的 DLA 显著性。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：corr(注意力质量, |t|) ≥0.5
- 关键数字：corr_all_heads=0.103
- 为什么：但这条失败换来一个更好的发现：早看晚写（见 R-ATT-06）。
- 复查：0.103 远低于阈值
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/pos_profile.json`

### R-ATT-07 在 GQA 结构下可以对每个注意力头独立做归因（头级分辨率 32）。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同组内 4 个头的读取通路应彼此独立
- 关键数字：n_query_heads=32，n_kv_heads=8，group_size=4，true_dof=9728×8
- 为什么：32 个 query 头只有 8 套 K/V。所有"头级"结论必须降一档理解——同组 4 头共享读写通路，"独立头"是假象。
- 复查：结构事实，不需要重测
- 证据：G' ｜ `server/mech_pair_service.py` ｜ `data/mech/pair_gain.json`

### R-ATT-08 1152 个头里存在一组约 263 个「角色定位头」：注意力落点等于所问角色（施事/受事/客体），显著高于随机落点。

- 领域：注意力头 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：落点命中率显著高于 null（null=0.344），且主动/被动两种句式同时成立
- 关键数字：n_pass=263，n_total=1152，null=0.344，median_layer=28，pass_inband_cov=0.23，fail_inband_cov=0.052，raw_gain_k60=0.389，aligned_gain=0.101，band_delta=-0.028
- 为什么：本线第一次「看起来很干净」的假阳性。筛选本身把高覆盖率（更容易被消融影响）的头选进来了，所谓「落点精准」其实是「删得多所以掉得多」。口径不变时不要再测头级落点。
- 复查：密度混杂已定量：通过组层内覆盖率 23%、未通过组 5.2%；对齐后效应 +0.389 → +0.101，整带为 −0.028
- 证据：Q1 ｜ `tools/pat_q1_unbind.py` ｜ `results/2026-11-01/*_pat_q1_unbind.json`
- 证据：Q2-A ｜ `tools/pat_q2a_confound.py` ｜ `results/2026-11-01/*_pat_q2a_confound.json`

### R-ATT-09 末位读出只接受一个约 53° 的「角度门锥」：位移方向落在锥内近全效、锥外归零，锥的朝向逐样本不同。

- 领域：注意力头 ｜ 置信：high ｜ 已测 4 次 ｜ 不要重测
- 判据：固定长度扫 θ∈[0°,90°]，D/R/M/Z 四臂应呈阶跃（近 1 → 0），且 cos(Δ_i,Δ_j) 应有共享轴
- 关键数字：D_0deg=1.0，R_33deg=0.958，M_61deg=0.479，Z_90deg=0.0，cos_pairs_mean=0.245，cos_with_consensus=0.482
- 为什么：「锥」是 LM head 线性读出的几何，不是角色编码。由此得出一条硬规则：tied-embedding 模型上的任何「注入差向量」实验都不能用来推断机制（D9）。
- 复查：死因是测量对象错：tied embedding + 末层 RMSNorm ⇒ 加 E[X]−E[Y] 就抬高 X 的 logit，近乎代数恒等式。注入 L20–L35 全部 1.000、注入宾语答宾语 0.886
- 证据：Q2-C2 ｜ `tools/pat_q2c2_direction.py` ｜ `results/2026-11-01/*_pat_q2c2_direction.json`
- 证据：Q2-L2 ｜ `tools/pat_q2l2_controls.py` ｜ `results/2026-11-01/050009_pat_q2l2_controls.json`

### R-ATT-10 三种角色在末位可分离为一组基（d1 施事 / d2 受事 / d3 客体），且 d3 不影响答案。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：d1/d2 翻转率 ≈1.0、d3 ≈0（答案不变），且 orth 对照 ≈随机
- 关键数字：d1=0.977，lex1=1.0，d2=1.0，d3_unchanged=1.0，orth1=0.159，rand=0.091，cos_d1_d2=0.545
- 为什么：目标词不必出现在句中也成立（d2 指向从未出现的名字，翻转 1.000），说明这不是角色推理，是输出可被直接转向任意词表项。
- 复查：自判平凡：d1 的效力全部由词法分量承担（剥到 λ=1 后 1.000 → 0.159），所谓「角色基」就是「目标词嵌入差」
- 证据：Q2-L ｜ `tools/pat_q2l_decompose.py` ｜ `results/2026-11-01/045825_pat_q2l_decompose.json`

### R-ATT-11 存在必需头、必需层或「必需致病组合」——删掉它们会让行为崩溃。

- 领域：注意力头 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：每层只保留 k 个头（k≤4），12 次随机抽样应出现部分崩溃；两次触底样本的交集应显著大于随机期望
- 关键数字：k=4，exact_1.000_runs=12/12，head_coverage=119/128 (93%)，chance_overlap=32，observed_overlap=32
- 为什么：结论是容量分布式：没有任何单点承担功能，损伤是累积量而不是结构量。不要再问「哪个头负责什么」，这个问题在本模型上没有答案。
- 复查：已四路交叉验证（Q2-D / D2 / G / H）：k≤4 时 12/12 精确 1.000，两次触底交集等于随机期望
- 证据：Q2-H ｜ `tools/pat_q2h_core.py` ｜ `results/2026-11-01/*_pat_q2h_core.json`

### R-ATT-12 注意力对行为的承载呈倒 U 形：中层最强、两端最弱。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：逐层单独全消融的行为 drop 应呈倒 U（中层峰、两端低）
- 关键数字：k=[10, 30, 60, 100, 150, 263]，drop=[0.0, 0.022, 0.117, 0.239, 0.451, 0.667]，shape=monotone
- 为什么：「中层是瓶颈」是视觉模型的经验移植。在语言模型的行为读出上，承载量靠近似等于「删掉了多少」，与层位置几乎无关。
- 复查：实测单调递增，无峰值；drop 随删掉的头数单调上升并在地板处饱和
- 证据：Q2-D ｜ `tools/pat_q2d_partial.py` ｜ `results/2026-11-01/*_pat_q2d_partial.json`
- 证据：Q2-A3 ｜ `tools/pat_q2a3_layerwise.py` ｜ `results/2026-11-01/*_pat_q2a3_layerwise.json`

### R-ATT-13 对答案最敏感的头（对齐头）就是承载行为的头（承载头）——两者是同一批。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：按对齐分数筛出的头，其消融 drop 应显著高于同层随机头
- 关键数字：aligned_heads=263，aligned_in_L20_35=260，carry_L0_3=0.674，carry_L16_19=0.66，carry_L28_35=0.0，phase_gap_layers=20
- 为什么：这是本线最重要的结构性事实：**「看起来在处理这个信息」与「缺了会坏」是两件不同的事，而且发生在不同的层**。任何用对齐分数当功能指标的做法都要先过这一关。
- 复查：两者反相约 20 层：263 个对齐头里 260 个落在 L20–35，而 L28–35 的承载量为 0.000
- 证据：Q2-D2 ｜ `tools/pat_q2d2_bandfull.py` ｜ `results/2026-11-01/*_pat_q2d2_bandfull.json`

### R-ATT-16 上下文通过改写 key 来决定注意力落点——同一 token 在不同句子里的 key 会有实质变化。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：同一 token 跨 20 个不同上下文的 key 方向 cos 应显著低于不同 token 之间；上下文特异方差份额应占可观比例
- 关键数字：cos_same_across_ctx=0.998，cos_diff_words_L0=0.9976，cos_diff_words_L35=0.9472，share_ctx_median=0.009，share_ctx_max=0.03
- 为什么：限定说明：k_ctx 定义为「偏离该词跨上下文平均方向」的部分，衡量的是上下文的**特异**贡献，共有部分被算进 k_cmn。所以成立的窄命题是：**句子与句子之间的差异几乎不改 key**。推论：角色选择不在 k 侧，必须去 q 侧找——这解释了头侧九轮全部失败。
- 复查：同词跨上下文 cos 0.998 与异词 cos 0.9976 几乎相等；上下文特异份额中位 0.009、全层 ≤0.03
- 证据：Q3-T1 ｜ `tools/pat_q3t_t1.py` ｜ `results/2026-11-01/055514_pat_q3t_t1.json`
- 证据：Q3-T2 ｜ `tools/pat_q3t_t2.py` ｜ `results/2026-11-01/055903_pat_q3t_t2.json`

### R-ATT-17 下一词分布由少数几个可复用的语义角色模式相加而成，即「多模式融合」在输出端有角色专属列族。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：held-out R²≥0.90 且 reuse(role)≥3.0（bootstrap 95% CI 下界 >1.5）；对照：跨角色复用与同实体复用
- 关键数字：cos_within_role=0.9676，cos_cross_role=0.9556，role_cos_gap=0.012，reuse_role_range=[1.3, 1.79]，reuse_cross_role=1.18，reuse_same_entity=7.884，reuse_diff_entity=1.236，entity_role_ratio=6.4
- 为什么：角色专属列族从原料上就不成立——三种查询的输出分布形状差异远小于它与自身的相似度。原担心的「答案词表差异」混杂因此没有出现（若由词表主导，角色间余弦应明显更低，实测只低 0.012）。输出端真正的结构是**按答案实体**组织的：同答案 7.88x vs 不同答案 1.24x。这与 3281 的「写入强度 D_norm 预测动态份额（ρ=0.97）」是同一件事从输出端的镜像：结构住在 q·k 匹配强度里，不在可复用的输出原子集合里。
- 复查：三种查询的 Δ 余弦：角色内 0.9676 vs 角色间 0.9556，只差 0.012；reuse 角色内 1.30–1.79（阈值 3.0）
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-ATT-18 「查询在哪个槽上」这件事无法用一条不含答案词的规则预测，只能靠语言学知识。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：规则只输出槽位、不含答案词；held-out 准确率 ≥0.80 且 ≥随机+0.40；同时报位置规则（不看查询）作备择，二者差 ≥0.30
- 关键数字：acc_R1_holdout=1.0，acc_R3_position=0.3333，random=0.3333，permutation_rate=1.0，lstar=28，lstar_folds_all_same=True，band_onset=L19-L24，low_layer_equals_position_rule=True，mcnemar_b=96，mcnemar_c=0，p=2.5e-29
- 为什么：规则 =「末位 query 在三个实体上注意力最大的槽」，不含任何答案词，槽里的词由句子给出。L18 及以前 R1 与 R3 同为 0.333 ⇒ 低层 argmax 退化为位置选择，查询尚未起作用；L21 → 0.59，L24 → 0.97，L28 起满分。⇒ **路由决策在 L19–L24，读出在 L33**（与 Q2-C1 衔接）。注意：R2（静态码）也达 1.000，但 T1 已测同词跨上下文 cos 0.998，k̄≈k 近乎恒等，故该条只排除"上下文特异改写影响选槽"，不是独立证据。
- 复查：R1 144/144（5 折全选 l*=28）；R3 位置规则恰 0.333；置换一致 48/48
- 证据：Q4-R ｜ `tools/pat_q4r.py` ｜ `results/2026-11-01/063929_pat_q4r.json`

### R-ATT-19 注意力头按语义维度分工：不同头分别编码语言、风格、逻辑、语法等维度，同层聚合即多维融合。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：有信号头（三槽覆盖≥0.20）之间的槽选择一致性 ≥0.80，且按 k 聚合的准确率必须**饱和在 <1**；若聚合能到 1.000 则分工不成立
- 关键数字：unanimity=0.875，k1=0.638，k32=1.0，sink_vs_content_r=-0.371，n_cells=144，n_signal_heads_median=16
- 为什么：唯一真实的分工是**二分类功能分工**：寄存器（sink）头 vs 内容头，逐头 sink 质量 [0.087, 0.938]、与指对率 r=-0.371。不是四类语义维度。这是「多维度加权融合」假设在注意力层的直接否证（negative #23）。
- 复查：若头分组指向不同槽，聚合必然饱和在中间值（组间抵消），不可能到 1.000。实测到 1.000
- 证据：Q5-A ｜ `tools/pat_q5a.py` ｜ `results/2026-11-01/070525_pat_q5a.json`

### R-ATT-20 注意力主要落在候选实体上，候选之间的份额即竞争结果。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：报告位置分解：sink 位置质量、候选槽总覆盖、其余；候选槽覆盖 <0.40 时不得用「候选之间竞争」表述
- 关键数字：sink_pos0=0.462，three_slots=0.1，rest=0.44，pos1=0.0
- 为什么：「竞争」的真实战场是**寄存器 vs 内容**。这条同时是对我自己框架的勘误：预注册设了 A4（覆盖≥0.40）作为框架前置闸，它在跑完时判 FRAMEWORK_INVALID，避免了我在错误坐标系里继续解释 A1/A2/A3。
- 复查：三实体槽只占 10%，46% 在句首 sink。Q4-R 的 1.000 是在这 10% 的质量差里取得的
- 证据：Q5-A ｜ `tools/pat_q5a.py` ｜ `results/2026-11-01/070525_pat_q5a.json`

### R-ATT-22 分类层级（苹果→水果→食物）由同一支可复用的「抽象方向」反复施加实现。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同链 cos(Δ(a→b), Δ(b→c)) 显著为正（对照为跨链配对）；若为负或近零则否证
- 关键数字：same_chain_median=-0.246，cross_chain=-0.005，peak=-0.172，layers_negative=37/37，radial_monotone_peak=0.6，radial_layers_pass=0/37
- 为什么：**负值是系统性的、有结构的**，不是噪声；但「为什么是负的」本轮无解释，作为开放异常带入下一轮，不得据此推断任何替代机制（E-07/E-08 见 findings §5）。
- 复查：同链余弦 37 层全为负、中位 −0.246，而跨链对照精确为 0。预注册的径向替代解释（越抽象越靠均值）也否证：达阈 0/37
- 证据：Q6-A ｜ `tools/pat_q6a.py` ｜ `results/2026-11-01/072150_pat_q6a.json`

### R-ATT-23 分类层级（苹果→水果→食物）由同一支可复用的「抽象方向」反复施加实现。

- 领域：注意力头 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：同链 cos(Δ(a→b), Δ(b→c)) 必须 ≥0.30（已达阈的共享方向量级）才可称方向；**判据只能看绝对水平，不得看正负号**——两跳共享中间词，符号为负是几何必然
- 关键数字：obs_all_layer_median=-0.2461，obs_max_any_layer=-0.1716，null_rand_median=-0.3994，null_level_median=-0.3075，geom_share=0.399，superordinate_share=0.0919，chain_share=0.0614，ref_F3_median=0.323，ref_F4_median=0.376，ref_F3_peak=0.395，ref_F4_peak=0.494，t_gt1_frac_obs=0.0，t_gt1_frac_null=0.0，rho_median=0.969
- 为什么：**上一轮（Q6-A）拒斥层级的理由必须作废**：−0.246 里 0.399 是几何、0.092 是上位词区域、只有 0.061 属于具体链条。结论保住是因为绝对水平不够，不是因为那个负号。**−0.246 从此不得单独引用**（负结果#25）。
- 复查：obs 在**任何一层**最高只到 −0.172，而同一运行里已确立的共享方向是 +0.323/+0.376；差距 0.49–0.55，不是"弱"，是"不同量级"。
- 证据：Q6-B2 ｜ `tools/pat_q6b2.py` ｜ `results/2026-11-01/073538_pat_q6b2.json`

### R-ATT-24 有限参数以低秩因子分解的形式存储「多物体 × 多属性」，故属性知识可由矩阵补全恢复。

- 领域：注意力头 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：held-out 格子 R² 必须随秩上升；若 **R=0（仅主效应）最优、加秩单调变差**，则结构不是低秩
- 关键数字：r2_at_R0=0.197，r2_at_R1=0.118，r2_at_R3=-0.406，r2_at_R4=-1.059，r2_null_R0=0.016，r2_null_R3=-0.631，r2_inverted_at_R=[4, 5, 11]，var_row=0.106，var_col=0.315，var_interaction=0.579，peak_ratio_median=0.5，active_values_per_type_median=1.0，active_values_per_type_mean=1.18，cells=336，active_indices=99，compression=3.4x
- 为什么：正确描述是**稀疏指针**：每个实体每属性类型只激活 1.18 个值，且只能取 12 元封闭码本的下标。与 Q3-T3（复用按答案实体 7.88× vs 按角色 1.24×）同向：**复用的单位是实体，不是关系或属性**。owner「有限参数记无限属性」成立，但机制是稀疏查表不是低秩分解（负结果#26）。
- 复查：interaction 占 58% 方差却一格都外推不了；真实数据在各秩均优于打乱零分布 ⇒ 结构为真但**不是低秩**，是稀疏
- 证据：Q6-B ｜ `tools/pat_q6b.py` ｜ `results/2026-11-01/074005_pat_q6b.json`

### R-MTH-04 对照组可以随着实验推进逐轮补充——每轮发现新混杂就加一个新对照。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：对照族必须在实验前一次性冻结；新增对照须升级族版本并重测全部历史结论
- 关键数字：frozen_transforms=8，rounds_before_freeze=unbounded
- 为什么：对照空间不封闭 ⇒ 等价于无限次假设检验。**冻结的代价才是闸门**：新增 t9 要求重测全部历史，这个成本让「边测边加」变得不划算（D11）。
- 复查：已观测到循环：Q1 死于密度、Q2-C2 死于长度、Q2-L 死于读出，每轮都是「临时想到一个对照」
- 证据：PSF-1 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-05 一个假设方向测不出结果时，可以换更细的切片继续测，直到测出来为止。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：每个假设家族设 3 轮上限；3 轮内拿不到「8 个变换中至少 4 个不变」即永久关闭
- 关键数字：family_budget=3，pass_threshold=4/8
- 为什么：更细的切片不会带来新信息，只会带来新的混杂。**关闭一个方向比测出它更有价值**——它把候选空间真正缩小了（D12）。
- 复查：D→D2→G→H 就是这个模式的实例：一轮比一轮细，最后得出「无必需头」
- 证据：PSF-1 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-06 过程账本（谁在什么时候跑了什么）足以支撑后续决策，不需要单独维护「已排除集」。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：每轮必须让排除集至少 +1；未增大的轮次记 NULL 且禁止在同一子空间再开 Phase
- 关键数字：ledger_records=455，excluded_set_v0=6
- 为什么：过程账支撑复现，排除集支撑决策。**没有后者，死掉的方向换个包装又回来**——这是循环真正的燃料，不是「假设不够好」。
- 复查：455 条 ledger 记录里没有任何「什么已被永久排除」的索引
- 证据：PSF-1 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-07 多个分片跑同一套实验时，把各分片的结论汇总起来（多数票 / 平均）即可得到全局结论。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：不变量取交集（全片成立才算）、候选规律取并集（单片入池但须双片复现）、剖面逐片保留不取平均
- 关键数字：shards=1280，merge_invariant=intersection，merge_candidate=union，merge_profile=no_average
- 为什么：最容易错的三处：① 用并集算不变量 ⇒ 假阳性爆炸；② 剖面取平均 ⇒ 双峰被抹平；③ 分片各自用不同判据。因此**汇总器只收原始量、不收结论**（D13）。
- 复查：尚未实跑，规则来自 D5 与「单片结论不可迁移」的两次实测
- 证据：PSF-1 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-08 实验没有产出规律时，应当换假设继续找。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：先判定是「假设错」还是「测量钝」：3 轮内无任何量达到 4/8 不变 ⇒ 换测量而非换假设
- 关键数字：threshold=3 rounds × inv<4/8
- 为什么：两种失败看起来完全一样（都是没结果），但修法相反。**不区分它们，就会永远在换假设，而工具一直是钝的**——这是上万个测试的总病根。
- 复查：历史 100% 的失败都被归因为「假设不对」，从未被归因为「工具钝」
- 证据：PSF-1 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-10 token 的「齿痕」（key 的身份差异）沿层的形状是 U 形：低层高、中层被上下文稀释、高层回升。

- 领域：实验方法 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：R_id(l,h,w) = ||mean_i k_i|| / mean_i||k_i||，M=20；形状须落在 {U形, 单调降, 单调升, 平坦} 之一
- 关键数字：sep_L0=0.002，sep_peak=0.23，peak_layer=13，sep_L35=0.042，match_all_layers=1.0，fold_over_chance=30.0
- 为什么：实测形状是「L0 精确为零 → L13 峰 0.23 → L35 收至 0.04」，四选一里没有这一档。但形状跨位置可复现（P9 vs P0 Spearman 0.835），是不变量候选。身份可分性反而满格（30x 基线，全层 1.000）——绝对差异小不等于不可分。
- 复查：四选一形状全落空：前中差 0.007（不是 U）、rho -0.169/+0.194（不是单调）、极差 0.227（不是平）
- 证据：Q3-T1 ｜ `tools/pat_q3t_t1.py` ｜ `results/2026-11-01/055514_pat_q3t_t1.json`

### R-MTH-11 不同语义角色在 logit 差上复用同一批稀疏字典列（reuse ≥ 3），即融合是结构性复用而非临时拼装。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：稀疏分解得 |D|≤64，任意 logit 差用 ≤3 列重建 R²≥0.9；reuse(role) = Jaccard / 随机期望
- 关键数字：r2_holdout=0.819，r2_holdout_best_fold=0.888，r2_insample=0.989，reuse_ag=1.788，reuse_re=1.609，reuse_th=1.3，reuse_median=1.609，reuse_same_entity=7.884，reuse_diff_entity=1.236，pca_shared_rank8=0.653，random_expectation_jaccard=0.0604
- 为什么：但复用**确实存在**，只是索引不同：同答案实体 7.88x vs 不同答案 1.24x，比值 6.4 倍。正确的替代陈述不是「没有复用」，而是「复用按答案实体组织，不按角色组织」。另：共享 rank-8 子空间只解释 65.3%，说明跨三元组没有可迁移的共享低维模式。
- 复查：held-out R² 0.819 < 0.90（5 折最好 0.888 也没到）；reuse 1.30–1.79 远低于 3.0
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MTH-12 可以用「全局 rank-3 PCA 的 R²」作为「每格 ≤3 原子稀疏编码」的 R² 上界。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：声称 A 是 B 的界，须先证明两者的可行约束集有包含关系
- 关键数字：pca_rank3=0.477，omp3_measured=0.541，violation=0.064
- 为什么：「每格 ≤3 原子」只约束该格落在**它自选的** 3 个原子张成的子空间里，不同格可选不同原子；全局 rank-3 PCA 约束所有格共享同一子空间。两者约束集无包含关系，不存在谁界谁（D19）。凡「某受限模型是另一受限模型的界」的说法，先检查约束集包含关系再开口。
- 复查：OMP(3) 实测 0.541 高于所声称的上界 0.477——上界被自己的测量推翻
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MTH-13 「消除后就无效」是判定机制有效性的必要条件，且对任何分析单元（头 / 层 / 神经元 / 位置）同等适用。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同一层带上比较「消融参数」与「改位置内容」两种干预；若前者为 0 而后者为满值，则必要性只存在于后者
- 关键数字：ablate_133_heads_L28_35=0.0，patch_L33_single_position=1.0，ablate_93pct_heads_L0_3=0.0，no_necessary_head=True，no_necessary_layer=True
- 为什么：消融测的是**容量与冗余**，不是功能；分布式系统按构造必然回答"删掉这个不坏"，所以消融类结论只在"容量"意义上成立，**永久降级为容量测量、不作机制证据**（D20）。必要性真实存在，只是住在"位置上的内容"里，不在"头的参数"里。推论：本项目不必要求每条结论都过消融关（D21 每 Phase 一个干预锚点即可），但不能一条都没有（D22）。
- 复查：同段 L28–L35：消融全部 133 头 drop 0.000，patch L33 一个位置翻转 1.00
- 证据：Q2-C1 ｜ `tools/pat_q2c1.py` ｜ `—`
- 证据：Q2-E/F/G ｜ `—` ｜ `—`

### R-MTH-14 一个假设家族跑满固定轮数（如 3 轮）即可判定该方向不存在。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：关闭前须报本轮 n 下的最小可检测效应 MDE；MDE > 关心的效应量时只能判「未测出」，记 pending 而非关闭
- 关键数字：rounds_cap_was=3，t3_fold_r2_range=[0.753, 0.888]，t3_mde_est=0.03，t1_n=1200，t1_judged_effect=0.1
- 为什么：「跑完了」≠「测不出」。若某轮 n=36、效应量 0.05、MDE 0.20，那轮没有资格产出 NULL。算力本身不设上限——但**规模必须由功效反推**，而不是由轮数或时长决定（D23）。本修订部分替代 D12：轮数只作辅助，功效闸才是关闭依据。
- 复查：轮数是功效的错误代理变量：弱效应小样本被误关，强效应 3 轮又过剩
- 证据：PSF-2 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MTH-15 定位型问题（某功能存在于哪些头 / 哪些层 / 哪些列 / 哪一段）是寻找编码机制的有效途径。

- 领域：实验方法 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：若连续 3 个侧（头 / k / 输出端）的定位测量全部 NULL，则判题型错而非侧错，须换为预测型判据
- 关键数字：rounds_total=12，sides_tested=['heads', 'k_side', 'output_side']，localization_nulls=12，prediction_type_rounds=0
- 为什么：定位问题假设"某处存着 X"。分布式表示里这个假设可能根本不成立——不是没找到，是**没有那个地方**。这是循环的真机制：不是假设不够好、不是算力不够，是**问错了题型**。改法：问"执行的是什么规则"，用**预测准确率 vs 基线**判决（D24）。预测型是唯一自带备择、无法用细分换结果的判据形态。注意这条是 12/12 的归纳结论，不是形式证明，故 confidence 记 medium。
- 复查：12 轮同一题型（定位）全部 NULL，且每换一侧都再次确认「没有那个地方」
- 证据：PAT-9R ｜ `—` ｜ `research/gpt5/docs/PATTERN_Q_FINDINGS_20261101.md`
- 证据：Q3-T1 ｜ `tools/pat_q3t_t1.py` ｜ `results/2026-11-01/055514_pat_q3t_t1.json`
- 证据：Q3-T2 ｜ `tools/pat_q3t_t2.py` ｜ `results/2026-11-01/055903_pat_q3t_t2.json`
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MTH-16 换测量侧（头 → k → 输出端）能突破"找不到承载者"的困境。

- 领域：实验方法 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：若三个侧连续 NULL，则判题型错而非侧错；改预测型判据后应在 2 轮内产出可报告准确率
- 关键数字：sides_null=3，prediction_type_rounds_to_first_hit=1，first_hit_acc=1.0，baseline=0.3333
- 为什么：PSF-2 §9.5 的自检条款预言「预测型判据应在 2 轮内产出可报告准确率而非 NULL」，实测 1 轮命中 ⇒ 自检通过。**关键不在"换了哪一侧"，在"换了题型"**：定位问"某处存着 X"，预测问"哪条规则能算出 X"。前者在分布式表示里系统性失败，后者不依赖"那个地方存在"。
- 复查：三个侧全 NULL 后，改预测型判据第 1 轮即得 1.000 vs 0.333（p=2.5e-29）
- 证据：Q4-R ｜ `tools/pat_q4r.py` ｜ `results/2026-11-01/063929_pat_q4r.json`
- 证据：PSF-2 ｜ `research/gpt5/docs/PATTERN_SEARCH_FRAMEWORK_v1.md` ｜ `—`

### R-MTH-17 预注册只需冻结判据阈值；判断框架本身是否适用的条件可以事后讨论。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 可以重测
- 判据：预注册必须包含至少一条「框架前置闸」：一个若不过就判框架失效、而非判假设失效的量
- 关键数字：q5a_gate=A4 cover>=0.40，actual=0.1，verdict=FRAMEWORK_INVALID
- 为什么：这是 D6（判据须自带备择）的框架级推广：不只要求判据能区分备择假设，还要求**能区分「假设错」与「坐标系错」**。两者处置完全不同：前者换假设，后者换测量单位。混在一起就是无限循环的入口。
- 复查：Q5-A 实测：A1/A2 未过、A3 过、A4 未过。若无 A4 会拿 A1/A2/A3 继续在错误坐标系解释
- 证据：Q5-A ｜ `research/gpt5/docs/PAT_Q5A_PREREG_v1.md` ｜ `results/2026-11-01/070525_pat_q5a.json`

### R-MTH-18 判据的效应量阈值一旦未达到，就可以判该项不成立。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：若 CI 上界仍高于判据阈值，只能判 PENDING_power 并给出所需 n；不得判 NOT_*
- 关键数字：A3_acc=0.375，A3_CI=[0.219, 0.562]，threshold=0.52，n=32，n_needed=170
- 为什么：这是 D23 第一次真正改变判决结果——此前 D23 只是写在框架里。同时给出反推：A3 要判死需约 170 个留出实例，现只有 32。
- 复查：脚本按阈值判 NOT_PREDICTIVE，但 D23 要求 CI 上界也低于阈值才可判死
- 证据：Q6-A ｜ `tools/pat_q6a.py` ｜ `results/2026-11-01/072150_pat_q6a.json`

### R-MTH-19 用 teacher-forcing 求「续写总 log-prob」可以测量某个词在上下文中的合适度。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：tokenize 后必须检查续写首 token 的 offset 起点是否 ≥ 续写起点；若被并入前文（如 `很大` 是一个 token），则该次比较作废。**优先改用「前缀末位单 token 条件概率」**
- 关键数字：bad_case=「苹果」通常很大。 → 很大(66,68)，起点 66 < 续写起点 67 ⇒ 被排除，symptom=大小门 0/8（恰好全错，系统性反转），fixed_by=M[o,a] = log softmax(logits[前缀末位])[id(a)]，前向 672→84
- 为什么：推广：**任何**基于「续写评分」的测量都要先打印一次 token offset 表。单 token 条件概率口径同时更省（一次前向读出整个属性类型的候选）且无边界问题。
- 复查：中文分词会把「很+形容词」「颜色字+色」并成单 token，续写总 log-prob 静默失效；门 0/8 是唯一可观测的指纹
- 证据：Q6-B ｜ `tools/pat_q6b.py` ｜ `results/2026-11-01/074005_pat_q6b.json`

### R-MTH-20 只要「观测 − 零假设」显著，就可以判定该效应成立。

- 领域：实验方法 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：判决必须分两级并同时报：**主判看绝对水平**（是否达到该类现象的量级）、**归因看相对差值**（是否显著高于零假设）。两者不得压成一个数
- 关键数字：q6b2_smoke_diff=0.19，q6b2_obs_absolute=-0.25，threshold=0.3，reference_established=[0.323, 0.376]，failure_mode=差值 +0.19 看似成立，但绝对水平 −0.25 离阈值差 0.55
- 为什么：配套要求：**参照线必须在同一次运行、同一估计器下测得**（本轮 F3 +0.323 / F4 +0.376 即如此），不得从历史结果里搬。
- 复查：把两级压成一级会产出「显著 ⇒ 成立」的 M1 式误读；且**必须同报参照线**，否则 −0.25 与 +0.40 无法在同一刻度比较
- 证据：Q6-B2 ｜ `tools/pat_q6b2.py` ｜ `results/2026-11-01/073538_pat_q6b2.json`

### R-MET-01 KL 散度可以作为 MLP 单元「重要性 / 必要性」的度量。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：top-k 的扰动能量应集中在高概率 token 上（energy_share_top20 ≥3× 均匀基线）
- 关键数字：energy_share_top20=0.0007，uniform_baseline=0.00013，ratio=5.4，letter_share=0.0002
- 为什么：KL 测的是 logit 长尾的抖动幅度，不是答案决策的因果贡献。用 KL 排序会把「让输出变毛糙的单元」误判成「重要单元」。这是本线最早也最重要的一条方法论否证。
- 复查：不要重测。KL 只能作长尾抖动的描述量，不能作必要性判据
- 证据：H3b ｜ `tools/mech_delta_eff.py` ｜ `tempdata/mech_delta_eff.json`

### R-MET-02 删掉「同规模的随机单元」是匹配的弱对照，可以用来和删 top-k 对比。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：随机对照的扰动规模应与实验组同量级（中心化范数比 0.5–2）
- 关键数字：norm_ratio_raw=8.16，norm_ratio_centered=0.13，kl_random=0.00089
- 为什么：删除小批随机单元约等于给 logits 加一个常数，softmax 完全不动。所以「删随机 97 没影响」不能证明「被删的随机单元不重要」，只能证明这个对照太弱。
- 复查：不要重测；随机对照必须报中心化范数比，否则「没影响」是无效结论
- 证据：H3 ｜ `tools/mech_delta_cos.py` ｜ `tempdata/mech_delta_cos.json`

### R-MET-03 未中心化的 Δlogits 范数可以直接用来比较不同干预的扰动规模。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：两种干预的范数比与它们的 KL 比同向
- 关键数字：raw_ratio_top1pct=1.73，raw_ratio_random=8.16，kl_ratio=54
- 为什么：必须先把 Δlogits 投影到与输出均值正交的子空间再比范数，否则常数平移完全污染比较。
- 复查：不要重测；比较扰动规模前必须投影到与输出均值正交的子空间
- 证据：H3 ｜ `tools/mech_delta_cos.py` ｜ `tempdata/mech_delta_cos.json`

### R-MET-04 n=40 的生成任务足以判定「删 top-k 造成 7.5 个百分点的下降」。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：McNemar 配对检验 p<0.05（不能只看点估计差）
- 关键数字：p_at_n40=0.453，discordant=5:2，need_n=684，effect_disappeared_at_n169=True
- 为什么：n=40 时 7.5 点的差距 p 只有 0.453。need_n=684 是错误口径下估的，实为 120–150。
- 复查：不要重测；任何「掉几个点」的结论必须给 p 与 discordant 计数
- 证据：H4→H5 ｜ `tools/mech_cot_analyze.py` ｜ `tempdata/mech_cot_L34_n169.json`

### R-MET-06 生成类实验的解析失败可以按统一口径合并处理，正确率与改答率可以共用同一分母。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 5 次 ｜ 不要重测
- 判据：必须分别报告：① 解析失败数；② 正确率用「base 有效」作分母；③ 改答率用「两侧都有效」作分母
- 关键数字：restored=True，why=解析失败在不同条件下数量不同（本线实测 base 19 / core3 7 / k97 1 / rand97 19，169 题），把它计入或剔除会改变分母与结论方向，example=H5 的「P = rand」就是被解析失败污染出来的，note=本条在一次账本脚本误覆盖后按当日工作日志重建，只保留了日志里记录的数字
- 为什么：写新脚本时「正确率用哪个分母」必须显式写在 docstring 里，并在输出里同时给两种口径的数。
- 复查：不要重测；这是本线最危险的坑（同一份数据两种口径结论完全相反）
- 证据：H5→H6 ｜ `tools/mech_pair_sign_strict.py` ｜ `tempdata/pair_sign_L34_strict.json`

### R-MET-07 脚本内置的 verdict 可以按与最终报告不同的口径计算。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：判据的计算口径必须与结论口径同源；脚本内置 verdict 要与独立复核脚本逐项对齐
- 关键数字：restored=True，case=H6 的 mech_pair_sign.py 把 None 记错：net(neg) 从 −6 抹成 0、net(both) 从 −8 抹成 −1，于是读出「S3 抵消成立 / 杠杆是无向抖动」，fix=另写严格口径分析器重算，结论翻转为「差分对否证，但特异性极强」，note=本条在一次账本脚本误覆盖后按当日工作日志重建
- 为什么：这是「判据实现与结论口径不同源」的第一个案例，后来在 H8/H11/H17 反复出现（见 R-MET-08/09/11）。
- 复查：不要重测；新实验先写严格口径分析器，再反校脚本内置 verdict
- 证据：H6 ｜ `tools/mech_pair_sign_strict.py` ｜ `tempdata/pair_sign_L34_strict.json`

### R-MET-08 剂量曲线类实验里，若没有任何剂量触发预设阈值，可以据此判定曲线形状（例如「接近线性」）。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：阈值未触发时必须报「未定位 + 相邻两点实测值」，不得输出形状结论；形状判定要求实测点覆盖 ≥3 个量级或至少两个超过 80% 基准
- 关键数字：case=H8 mech_dose.py，false_reading=饱和点高于 40 → 曲线接近线性，actual={'k40': -0.74, 'k97': -5.93, 'linear_pred_at_k40': -2.4}，cause=k80 为 None 时脚本走 else 分支直接打印「接近线性」
- 为什么：阈值未触发只说明「不在已测区间内」，完全不能推出形状。要判形状必须实测覆盖：H8 只测到 40，而悬崖在 40–97 之间，任何结论都越界。⇒ 曲线实验必须至少测到基准的 60–80%，否则只说「未定位」，并给出相邻两点让它可见。
- 复查：这是项目里第三次「脚本读法 ≠ 数据形状」（H5 分母、H6 verdict、H8 形状）
- 证据：H8 ｜ `tools/mech_dose.py` ｜ `tempdata/dose_L34.json`

### R-MET-09 判据里写清方向（例如「A 的集中度 > B 的集中度」）就够了，不必给效应量阈值。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：每条数值判据都必须带效应量阈值或相对差下限（≥20% / ≥10 个百分点 / p<0.05），并同时输出实测值与阈值，避免 0.093 vs 0.091 这种噪声差被判成立
- 关键数字：case=H11 U3，v1={'fragile': 0.093, 'control': 0.091, 'rule': 'fragile > control', 'verdict': 'true'}，v2={'rule': 'fragile >= 1.2 * control', 'verdict': 'false'}，false_reading=脆弱题更集中
- 为什么：只写不等号的判据在数值噪声面前必然成立。⇒ 所有判据模板改为「实测值 + 阈值 + 差值」三件套，并在输出里同时打印；凡是只写 `>` / `<` 的地方，一律视为未定义判据。
- 复查：这是本线第四次同类错误（H5 分母、H6 verdict、H8 形状、H11 判据），都出在「判据太松」
- 证据：H11 ｜ `tools/mech_frag_attr.py` ｜ `tempdata/frag_attr_L34.json`

### R-MET-10 自适应二分的切分路径可以直接当作「最小充分集」的判定结果使用。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：补测单点与成对：若单点即可复现被归到「多单元档」的题，则二分路径给出的档位无效
- 关键数字：bisection_tier={'qi=78': 'A′(5) 档', 'qi=25': '需 A(11) 档'}，single_unit_result={'767 单点': '同时打坏 78 和 25'}，core3_damaged_7q=完全被 767 单点的 9 道包含，lower_bound_set_by_code=while len(cur) > 3
- 为什么：二分只回答「这条路径上哪一半还有用」，它优化的是轮数不是集合最小性；再叠一个 while len(cur) > 3 的下限，等于把答案的下限写进了代码。这次是便宜的一课（17 题 × 7 条件 15 分钟就把三合一推翻了）。**以后凡是「找到了核心集合」，必须补跑其所有真子集**，否则「最小」二字无依据。
- 复查：已直接否证；任何「最小集合」结论都要逐单点/逐对补测，不能引用二分路径
- 证据：H12→H15 ｜ `tools/mech_carrier.py → mech_core_split.py` ｜ `tempdata/carrier_L34.json + core_split_L34.json`

### R-MET-11 用「N 个随机样本的最大值 ≤ 阈值」判特异性。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同一批评测上同时报两件事：数量（主条件 vs 随机组的最大/分布）与身份（随机组损伤并集、主条件的独有题集）。只报数量即判该判据不合格
- 关键数字：pre_registered=rand_max <= 1（10 个随机单点在 17 题上），observed={'rand_max': 2, 'rand_counts': [1, 1, 2, 2, 2, 2, 1, 0, 1, 0], 'rand_union': [25, 73, 78], 's767': 9, 's767_exclusive_of_union': [5, 29, 41, 94, 97, 125]}，why_failed=随机组最大值被一道通用脆性题（qi=73，8/10 个随机单元都能弄坏）抬高了一，阈值 1 因此被击穿；而按身份看随机组的并集只有 3 道题、767 有 6 道在其外，same_family=['R-MET-07（判据读法错）', 'R-MET-09（相对差阈值缺失）']
- 为什么：这是本线第五次「判据实现与结论口径不同源」：判据写的是一个能被单点异常值击穿的阈值，结论要的是「主条件命中了随机对照命不中的目标」。更一般的检查项：**任何判据都必须先问「它在什么情况下会给出另一种结果」**，给不出两种结果的判据（如 H16 的 A2、H14-pre 的 W4）一票作废。
- 复查：已直接否证；凡是「随机对照」必须同时报并集与差集，否则「特异性」二字无依据
- 证据：H17 ｜ `tools/mech_core_one.py` ｜ `tempdata/core_one_L34.json`

### R-MET-12 用「答对/答错」当因变量测单元（或单元集合）的必要性。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：内容必要性必须用格式不变的读出复测：固定 base 自已写好的前缀，测答案位置的 gold 间隔被压缩多少；只报答案对错不算测到内容
- 关键数字：what_went_wrong=L34 上被测得「最有必要」的单点（767）实为格式单元：它对答案位置换行族 logit 的影响是 9–10.7，对字母间隔只有 0.12，why_systematic=格式是全模型瓶颈、只需极少数单元；知识是分布的 ⇒ 以答案对错为因变量时，格式单元会被系统性排到第一位，evidence={'H18': 'newline +9~10.7 vs letter ±0.12', 'H19': 'fixed prefix 9/9 字母不变', 'H20': '四档前缀上间隔位移 ≤0.12', 'H21': '首 token 空白 1/11 → 8/11，格式崩的 7 道答案全变'}
- 为什么：这是本线代价最大的一次方法学错误：H5–H17 十三个实验都用「答对/答错」当因变量，方向没错（行为是真因变量），但**它不能区分格式与内容**。修法不是换掉行为指标，而是加一个格式不变的对照读出：同一批题、同一前缀、只比字母间隔。H22 就是按这个读出重做池级必要性。
- 复查：已直接否证；任何「某单元承载某概念」的结论必须先用格式不变读出复核
- 证据：H18–H21 ｜ `mech_logit_effect / mech_format_test / mech_prefix_scan / mech_head_tokens` ｜ `tempdata/logit_effect_L34.json + format_test + prefix_scan + head_tokens`

### R-MET-13 在末位注入「目标词的嵌入差」，可以用来研究角色编码机制。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：注入后行为改变应随注入层变化（若为机制，应只在读出点附近生效）
- 关键数字：inject_L20_35_all=1.0，inject_q0=0.091，inject_object_answer_object=0.886
- 为什么：tied-embedding 结构下这是代数恒等式，不是机制。**任何注入类实验，必须先证明它在不该生效的地方不生效**（D9）。
- 复查：与层完全无关（L20/25/29/31/33/35 全部 1.000），换位置到句首即掉到 0.091
- 证据：Q2-L2 ｜ `tools/pat_q2l2_controls.py` ｜ `results/2026-11-01/050009_pat_q2l2_controls.json`

### R-MET-14 两组头数不同时，可以直接比较消融造成的 drop 大小。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：比较前须把「层内覆盖率」配平；未配平时 drop 差异应由覆盖率解释
- 关键数字：raw_diff_k60=0.389，coverage_pass=0.23，coverage_fail=0.052，aligned_diff=0.101，band_diff=-0.028
- 为什么：这是本线最贵的教训：一个 +0.389 的「显著效应」，配平覆盖率之后变成 −0.028。**报 drop 必报三元口径：头数、占总头比、层内覆盖率**（D1）。
- 复查：配平后效应从 +0.389 塌到 +0.101，整带反向
- 证据：Q2-A ｜ `tools/pat_q2a_confound.py` ｜ `results/2026-11-01/*_pat_q2a_confound.json`

### R-MET-15 对多次随机抽样的结果取均值，可以代表该条件下的典型效应。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 4 次 ｜ 不要重测
- 判据：取均值前须检查分布；极差 ≥0.30 或双峰时均值无意义
- 关键数字：observed_range=[0.34, 1.0]，shape=bimodal，mean_used_before=True
- 为什么：正确做法是逐次配对比较，或报告分布本身。均值 + 一个 ±0.04 的标准误，会掩盖「一半样本 1.00、一半 0.34」这种结构（D5）。
- 复查：本线已在三处重犯（Q2-F / Q2-C2c / 早期 A64），每次都是双峰被均值抹平成「中等效应」
- 证据：Q2-F ｜ `tools/pat_q2f_hetero.py` ｜ `results/2026-11-01/*_pat_q2f_hetero.json`

### R-MET-16 SMOKE（小样本预跑）上得到的效应方向与大小，可以外推到全量。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：SMOKE 与全量的效应方向应一致；且 SMOKE 必须覆盖全量的全部结构维度
- 关键数字：smoke_n=24，smoke_effect=0.292，full_n=44，full_effect=-0.028
- 为什么：SMOKE 只能验证管线跑得通，**不能作为证据**。而且 SMOKE 的样本必须覆盖全量的每一个结构维度（D8）：4 个三元组共用同一对人物，就是一个维度没覆盖。
- 复查：两次反向：n=24 的 +0.292 → 全量 −0.028；SMOKE cos 0.936 → 全量 0.245
- 证据：Q2-A ｜ `tools/pat_q2a_confound.py` ｜ `results/2026-11-01/*_pat_q2a_confound.json`
- 证据：Q2-C2c ｜ `tools/pat_q2c2c_samelength.py` ｜ `results/2026-11-01/*_pat_q2c2c_samelength.json`

### R-MET-17 方向类对照只需要配平投影长度，不必同时配平向量长度。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：同幅度对照须同时控 |位移| 与投影；缩放因子依赖样本量时必须报其分布与最大值
- 关键数字：scale_factor_max=2.07，cos_median=0.482，curve_shape=non-monotone
- 为什么：「配平了投影」听起来比「配平了长度」更严格，实际反而引入了未控的幅度差。超 cap 的臂应该直接不跑，而不是跑完再解释（D7）。
- 复查：S 臂实际位移是目标的 2.07 倍，曲线非单调（0.639→0.611）是过冲不是额外效力
- 证据：Q2-C2 ｜ `tools/pat_q2c2_direction.py` ｜ `results/2026-11-01/*_pat_q2c2_direction.json`

### R-MET-19 判据里取 min(两臂) 可以避免某个臂失效导致的假阳性。

- 领域：指标与对照 ｜ 置信：medium ｜ 已测 1 次 ｜ 不要重测
- 判据：若某个臂在基线条件下恒为零，min 会把整个判据钉死在零
- 关键数字：case=Q2-C1 判据 min(agent0, agent1)，agent1_baseline=0.0
- 为什么：保守的判据构造（取 min / 取交）会引入「恒零分量」这类结构性失效，而且失效方向是「什么都判不出来」，看起来像「没效应」而不是「判据坏了」（D6 补充）。
- 复查：复合判据的每个分量都必须单独报自己的剖面
- 证据：Q2-C1 ｜ `tools/pat_q2c1_patch.py` ｜ `results/2026-11-01/*_pat_q2c1_patch.json`

### R-MET-20 方差份额（share）是样本集无关的内禀量，可以在混合不同条件的样本上直接估计。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：同一份额在至少两个条件分组内分别重算，降幅差应 <20%
- 关键数字：pooled_drop=0.66，per_group_drop=[0.26, 0.23, 0.26]，inflation=2.5
- 为什么：share 是相对于所选样本集的条件量。混合互不相同的条件会把它抬高。**报份额必须同时报至少两个分组的值**（D14）；这也适用于所有"方差解释比例"类指标。
- 复查：混合 114 格得降幅 0.66，按查询分组得 0.23–0.26，夸大 2.5 倍
- 证据：Q3-T2 ｜ `tools/pat_q3t_t2.py` ｜ `results/2026-11-01/055903_pat_q3t_t2.json`

### R-MET-21 把「同一 token 跨上下文的平均方向」直接当作身份方向，可以把 key 拆成身份分量 + 上下文分量。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：分解后身份份额应显著小于 1；若 ≈1 说明身份项吞掉了共同方向
- 关键数字：naive_share_id=1.0，fixed_share_id_L0=0.95，fixed_share_ctx=0.009
- 为什么：正确顺序是三步：先取「所有词所有上下文共有」的方向 g，扣掉它，再在各词的去共同残差里取身份方向（D15）。省掉第一步会让分解退化成"几乎全部能量都算身份"。
- 复查：所有 key 两两 cos 0.998，均值方向就是主方向，投影后残差≈0
- 证据：Q3-T2 ｜ `tools/pat_q3t_t2.py` ｜ `results/2026-11-01/055903_pat_q3t_t2.json`

### R-MET-22 稀疏字典分解的 in-sample R² 可以证明「分解成立」。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：字典容量 K 接近数据维度 V 时必须报 held-out R²；in-sample 不作证据
- 关键数字：k_dict=32，v=39，r2_insample=0.989，r2_holdout=0.819，gap=0.17，best_fold_holdout=0.888
- 为什么：K≈V 时 in-sample R² 必然接近 1（过完备字典对训练点近乎插值），拿它当证据是自欺（D17）。held-out 差距全部来自泛化，而泛化才是「结构可复用」这个词的实际含义。
- 复查：in-sample 0.989 vs held-out 0.819，差 0.17；5 折中最好一折 0.888 仍未过 0.90
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MET-23 bootstrap 重采样时保留 (i,i) 自配对不会影响置信区间。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：CI 下界必须 ≤ 点估计；若 CI 下界 > 点估计 ⇒ 估计器有缺陷，不是发现
- 关键数字：point_est=1.788，ci_lo_before_fix=1.813，ci_hi_before_fix=2.456，ci_lo_after_fix=1.52，ci_hi_after_fix=2.1
- 为什么：对集合相似度做 bootstrap 时，同一元素被抽中两次形成的 (i,i) 配对 Jaccard 恒为 1.0，会把整个分布整体抬高（D18）。凡重采样后仍按「两两配对」聚合的统计量都中这一枪。
- 复查：修复前 CI 下界 1.813 > 点估计 1.788（结构性不可能）；修复后 CI [1.52, 2.10] 正常包住
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MET-24 逐层剖面可以按固定步长抽样打印，抽样不改变结论。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：剖面必须以完整数组落盘/打印；抽样只作摘要，且须同时给出完整版本
- 关键数字：sampling_step=3，n_layers=36，printed_upto=33，missed_layers=[34, 35]，L35_actual=0.562，claimed=L28 起稳定满分
- 为什么：抽样打印是**报告环节的泄漏**：不改数据，但系统性掩盖末端效应。36 不是步长 3 的端点友好数，而最后几层恰是注意力重新分配最常发生处。与 D5（双峰禁用均值）、D13（剖面不平均）同类：聚合可以，但聚合不得替换原始分布（D25）。
- 复查：profile[::3] 打印到 L33，跳过 L34/L35；实测 L35 = 0.562，与"稳定满分"相反
- 证据：Q4-R ｜ `tools/pat_q4r.py` ｜ `results/2026-11-01/063929_pat_q4r.json`

### R-MET-26 用「族内配对 cos 中位」判断关系方向时，跨族比较足以排除目标词相似的伪影。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：判断关系方向必须做**族内置换**（打乱源—目标配对、**保持目标词多重集不变**），并与跨族对照同时报
- 关键数字：F4_target_self_cos=0.82，F3_target_self_cos=0.85，F1_target_self_cos=0.8，why=同族目标词彼此 cos 高达 0.80-0.85，Δ 天然相似
- 为什么：与 Q3-T3 的「答案词表混杂」同型，但更隐蔽：T3 混的是角色，这里混的是**族内目标词的彼此相似**。置换保持多重集，因此把它精确吸收。
- 复查：F4 只有小/大两个目标、两者 cos 0.82，A1b 仍达阈 33/37；不做置换会把「目标词像」误判成「关系方向」
- 证据：Q6-A ｜ `tools/pat_q6a.py` ｜ `results/2026-11-01/072150_pat_q6a.json`

### R-MET-27 bootstrap 重采样只要次数够多就可靠，口径可与点估计不同。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：bootstrap 的重采样单位、留出划分、层选择必须与点估计**完全一致**；并断言 CI 必须包住点估计，否则判 ESTIMATOR_BUG
- 关键数字：q3t_bug=CI 下界 1.813 > 点估计 1.788，q6a_bug=CI [0.61,0.84] 整体高于点估计 0.196，cause=点估计按词留出、bootstrap 用样本内全实例
- 为什么：这是 D26 的规则化。关键判据：**CI 不含点估计 ⇒ 估计器错，不是发现**。两次的实际成因不同（自配对 / 口径混用），说明根因是「估计器与统计量不对齐」，不是某一个特定 bug ⇒ 故改为断言式检查而不是逐条排查。
- 复查：**同一类错在不同 Phase 连续出现两次**（D18 已立过，但只写了「跳自配对」这一种成因）
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`
- 证据：Q6-A ｜ `tools/pat_q6a.py` ｜ `results/2026-11-01/072150_pat_q6a.json`

### R-MET-28 两条差向量的余弦可以直接解释为「两个关系的相似度」。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：凡比较共享词的两条差向量，必须同时报**换词零假设**（保持共享词与符号、只换另一端），且结论只允许依据「观测 − 零假设」，不得依据原始符号
- 关键数字：iso_prediction=-0.5，measured_null_median=-0.3994，measured_obs_median=-0.2461，semantic_share=0.0614，why=Δ1=b−a 与 Δ2=c−b 共享 b 且符号相反 ⇒ E[Δ1·Δ2]=−σ²、E[|Δ1||Δ2|]=2σ² ⇒ E[cos]=−0.5
- 为什么：这条覆盖绝大多数类比与组合性探针（它们天然比较共享成分的差向量）。**判据形态**：不看符号看绝对水平；**对照形态**：换词而非换链。
- 复查：Q6-A 把 −0.246 当成「层级不是方向」的证据，实际几何基线就是 −0.40；跨链对照 =0 是因为跨链不共享词，根本无法判定同链负值来源
- 证据：Q6-B2 ｜ `tools/pat_q6b2.py` ｜ `results/2026-11-01/073538_pat_q6b2.json`

### R-MET-29 用「向量落在某子空间之外的份额」可以判定该向量与该子空间是否解耦。

- 领域：指标与对照 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：在高维（≫子空间维度）空间中，投影份额类判据**饱和**，无分辨力；须改用回归 R² / CCA 等**有尺度**的统计量
- 关键数字：dec_attr=0.95–1.00，dec_adjective_null=0.97–1.00，diff_range=[-0.0445, -0.0005]，dim=2560，subspace_dim=3，volume_ratio=3/2560 ≈ 0.12%
- 为什么：这条把前两次的具体补丁（R-MET-13 换主判据、R-MET-21 三步分解）升格为**形态判据**：凡是「份额/占比/比例」类统计量，先算它在**随机对象**上的值；若随机对象已接近上限，该判据不可用。
- 复查：**同一类错第三次**：分母含天然极小量（k/H、Var(mean)/Var(个体)、3/2560），判据被钉在天花板，对一切都成立
- 证据：Q6-B ｜ `tools/pat_q6b.py` ｜ `results/2026-11-01/074005_pat_q6b.json`
- 证据：Q3-T1 ｜ `tools/pat_q3t_t1.py` ｜ `results/2026-11-01/055514_pat_q6t_t1.json`
- 证据：Q3-T3 ｜ `tools/pat_q3t_t3.py` ｜ `results/2026-11-01/061142_pat_q3t_t3.json`

### R-MLP-04 高杠杆 MLP 单元是成对正反编码（差分对）：删掉它们时，破坏正确计算与修复错误计算同时发生，净效应接近零。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：行为层的净效应当接近零（由错变对 ≈ 由对变错）；排除解析失败题后仍应如此；若真是差分对，拆开单删也应只剩一半后果、合删才有全部后果
- 关键数字：orig_wrong_to_right=11，orig_right_to_wrong=14，orig_net=-3，corrected_wrong_to_right=3，corrected_right_to_wrong=13，corrected_net=-10，logit_abs_sum=633.7，logit_net_sum=16.1，h6_split={'pos45_delta_pt': -5.11, 'neg45_delta_pt': -4.38, 'both90_delta_pt': -5.11, 'pos_vs_neg_p': 1.0, 'pos_vs_both_p': 1.0}
- 为什么：「双向抵消」不成立：排除解析失败题后净效应是 −10（不是 −3），P 显著差于随机删除。H6 进一步否掉差分对的形状要求：差分对必须"删一半只有一半后果"，实测删 45 个就掉 5 点、删 90 个还是 5 点（饱和）。logit 层的 ΣΔ 很小（16.1）仍然成立，但那是 KL/长尾口径的性质（见 R-MET-01），不能外推到行为层。⇒ 符号没有功能意义，「冗余饱和池」（R-MLP-16）才是这批单元的真实形状。
- 复查：H5 从净效应否定、H6 从拆分实验否定，两条路径同一结论；别再用拆分重做
- 证据：H5+更正 ｜ `tools/mech_cot_analyze.py` ｜ `tempdata/mech_cot_L34_n169.json`
- 证据：H6-严格 ｜ `tools/mech_pair_sign_strict.py` ｜ `tempdata/pair_sign_L34_strict.json`
- 证据：C5' ｜ `tools/mech_unit_attr.py` ｜ `tempdata/unit_attr_L24.npz`

### R-MLP-05 按共现把 MLP 单元聚成组，得到的组是"参数共享的功能子模块"（组内写方向应当一致）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：组内写方向平均 |cos| 显著高于零模型（同规模随机单元对）
- 关键数字：cos_within=0.017，cos_null=0.0198，lift=1.0，jaccard_lift=4.66–14.45
- 为什么：共现分组是"语境模块"（同类文本里一起被激活），不是"计算模块"（共享参数、协同工作）。这是整个 C 段路线失败的直接原因。
- 复查：共现结构真实（Jaccard lift 5–14×），但组内方向与随机不可分，结论明确
- 证据：A2 ｜ `server/mech_pair_service.py` ｜ `data/mech/unit_clusters.json`

### R-MLP-06 共现聚类在换随机种子 / 换样本后能稳定复现。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：换初值或重采样 80% 样本后，两次聚类的 Jaccard ≥0.6
- 关键数字：jaccard_seed=[0.434, 0.414, 0.455]，jaccard_resample80=[0.187, 0.282, 0.325]
- 为什么：簇是采样噪声的产物；任何以簇为单位的因果结论都不成立。
- 复查：不足 0.5 的复现度已经足以否定"稳定模块"
- 证据：C2' ｜ `tools/mech_cluster_stability.py` ｜ `tempdata/cluster_stability.json`

### R-MLP-07 共现聚类的簇对应可命名的语义类别（token 类型 / 语种 / 标点）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：AMI(簇, 类别) 显著高于零（≥0.3）
- 关键数字：ami_char=0.045–0.184，ami_language=0.041–0.144，punct_ratio=0.115 vs 0.093
- 为什么：上轮把 C19 判成"排版模块"是采样偏差（全层真正最偏标点的是 C8）。教训：簇不承载语义。
- 复查：AMI 接近零基线，且 C19 的标点富集只有 1.24×，已撤回上轮"排版模块"的判断
- 证据：C2' ｜ `tools/mech_token_types.py` ｜ `tempdata/cluster_token_types.json`
- 证据：C2' ｜ `tools/mech_cluster_axis.py` ｜ `tempdata/cluster_axis.json`

### R-MLP-10 用权重几何（列范数 / 能量 / 有效秩）可以预测 MLP 单元的重要性排序。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 2 次 ｜ 不要重测
- 判据：与 |t| 的相关系数 ≥0.6
- 关键数字：corr_col_norm=0.174，corr_ov_pr=0.026，corr_w_letter_max=0.349
- 为什么：看权重猜不出谁重要，只能实测。这条否证了"剪枝按范数"的常见做法在这个模型上的有效性。
- 复查：三种几何量都低相关，且有反例（L24H31 能量 0.042 但 t=14.6；L35H24 能量 0.274 但 t=−15.2）
- 证据：A1 ｜ `server/mech_service.py` ｜ `data/mech/heads_dir.json`

### R-MLP-11 用"读取增益"（下游 W_K/W_V 能读到多少）可以预测 MLP 单元的重要性。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：corr(增益, |t|) ≥0.5 且逐对稳定
- 关键数字：corr_mean=0.226，corr_median=0.138，pairs=41
- 为什么：这条失败后流水线由 W→R→G→C→P 简化为 W→R→C→P，省掉一整个环节。
- 复查：预注册 kill criterion 命中，G 环节已从流水线移除
- 证据：G' ｜ `server/mech_pair_service.py` ｜ `data/mech/pair_gain.json`

### R-MLP-13 删掉整层 9728 个 MLP 单元会翻转大量答案。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 3 次 ｜ 不要重测
- 判据：翻转数显著高于随机（≥20/169）且答案分布的熵明显上升
- 关键数字：flip=5/169，d_margin=0.02，entropy=0.621→0.395，top1_still_letter=1.0
- 为什么：回答不是"层不重要"，而是层的输出被下游层冗余补偿；同时 KL 指标测的是长尾（见 R-MET-01）。
- 复查：三种读数（翻转 / margin / 熵）一致指向"没崩，反而更确定"
- 证据：C5' ｜ `tools/mech_degrade_check.py` ｜ `tempdata/degrade_check_L24.json`
- 证据：H5 ｜ `tools/mech_cot.py` ｜ `tempdata/mech_cot_L34_n169.json`

### R-MLP-14 高杠杆单元的符号有功能意义：只删 Δ>0 的一侧与只删 Δ<0 的一侧，行为效应方向相反。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 4 次 ｜ 不要重测
- 判据：数量与幅度双匹配（各 45 个，Σ|Δ| 48.77 / 45.54）后，net(pos) 与 net(neg) 符号相反且 |net|≥5；严格口径（排除解析失败题）
- 关键数字：n_strict=137，base_acc=0.752，pos={'acc': 0.701, 'delta_pt': -5.11, 'net': -7, 'flip': '8/1', 'p_vs_base': 0.0391}，neg={'acc': 0.708, 'delta_pt': -4.38, 'net': -6, 'flip': '9/3', 'p_vs_base': 0.146}，pos_vs_neg_p=1.0，rand97_delta_pt=0.73
- 为什么：两侧都掉分（−5.11 / −4.38 点）、彼此无差异（p=1.0000），没有"方向相反"。符号是数值属性，不是功能单位——继续在正负号上找区分是死路。脚本自带 verdict 曾报「无向抖动」，那是 None 记错造成的二次误读（见 R-MET-07）。
- 复查：已直接测完。符号方向不成立，别再用「正/负侧」切分重做
- 证据：H6 ｜ `tools/mech_pair_sign.py` ｜ `tempdata/pair_sign_L34.json`
- 证据：H6-严格 ｜ `tools/mech_pair_sign_strict.py` ｜ `tempdata/pair_sign_L34_strict.json`

### R-MLP-16 L34 的 top-1% 单元是一个冗余饱和的共享功能池：删池内任一半与删全部的行为损伤相当，而同层随机删除几乎无损。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：强形式「任一半」需 top-40 也能打出 ≥3pt 损伤；弱形式「整类 vs 整池」只需 pos45≈both90；严格口径
- 关键数字：n_strict=137，base_acc=0.752，pos45_delta_pt=-5.11，neg45_delta_pt=-4.38，both90_delta_pt=-5.11，rand97_delta_pt=0.73，h8_top40_delta_pt=-0.74，h8_top40_sum_abs_delta=73.63，h8_top97_sum_abs_delta=96.5，h10_tailP_delta_pt=-0.72，h10_tailN_delta_pt=-5.04
- 为什么：强形式被推翻：删 40 个（含 pos45 的 18 个、neg45 的 22 个、占 top-97 归因量 76%）几乎无损。弱形式（把某一符号类删到接近干净）成立，但 H10 显示决定因素是「是否覆盖到承载单元」，不是数量也不是符号 → 改写为 R-MLP-19。
- 复查：「任一半就饱和」已被 H8 与 H10 两轮独立推翻；准确说法见 R-MLP-19（近二值池依赖）
- 证据：H6-严格 ｜ `tools/mech_pair_sign_strict.py` ｜ `tempdata/pair_sign_L34_strict.json`
- 证据：H8-反证 ｜ `tools/mech_dose.py` ｜ `tempdata/dose_L34.json`
- 证据：H10-反证 ｜ `tools/mech_headtail.py` ｜ `tempdata/headtail_L34.json`

### R-MLP-18 一阶归因的 |Δ| 排名可以预测单元的行为必要性：|Δ| 越大的单元删掉越痛。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：删 top-k（按 |Δ| 降序）的损伤应随 k 单调增强，且 top-40（占归因量 76%）必须打出 ≥3pt；同层随机删除作对照
- 关键数字：top40_delta_pt=-0.74，top40_sum_abs_frac_of_top97=0.763，pos45_delta_pt=-5.11，tailP27_delta_pt=-0.72，tailN23_delta_pt=-5.04，rand97_delta_pt=0.73，top40_damaged_q=4，tailN23_damaged_q=11，per_question_attr_auc=H11 三条判据全否
- 为什么：top-40 拿走 76% 的归因量却只打坏 4 道题；tailN 的 23 个占 9.43/96.5 = 9.8% 却打坏 11 道。而且逐题归因也不能预测哪道题会坏。⇒ 一阶归因只能用来选「池」，不能用来选「池内成员」，更不能用来预测单题依赖。后续所有「top-k」口径的结论必须改口径（整类删除 / 留一 / 承载集定位）。
- 复查：三轮独立证据（H8 剂量、H10 头尾、H11 逐题）都指向同一结论：不要在池内按 |Δ| 排序
- 证据：H8 ｜ `tools/mech_dose.py` ｜ `tempdata/dose_L34.json`
- 证据：H10 ｜ `tools/mech_headtail.py` ｜ `tempdata/headtail_L34.json`
- 证据：H11 ｜ `tools/mech_frag_attr.py` ｜ `tempdata/frag_attr_L34.json`

### R-MLP-20 单题的一阶单元归因能预测该题在消融下会不会被弄坏（脆弱性可预测）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：脆弱题与对照题的归因画像必须有可测差异：与全局 top-97 重叠、top-20 集中度、承载集覆盖，任一项相对差 ≥20% 或重叠差 ≥10
- 关键数字：overlap_global97={'fragile': 26.75, 'control': 25.67}，conc_top20={'fragile': 0.093, 'control': 0.091, 'rel': 1.022}，in_tailN={'fragile': 3.67, 'control': 4.75}，in_tailP={'fragile': 5.75, 'control': 5.667}
- 为什么：常见误差是「脆弱题 top-20 集中度 0.093 > 对照 0.091 所以更集中」——相对差 1.022 倍是噪声。U3 判据最初只写不等号，第一次跑就误判为成立（见 R-MET-09）。
- 复查：已被直接测过并否证：再做「用归因预测哪道题会坏」就是重复
- 证据：H11 ｜ `tools/mech_frag_attr.py` ｜ `tempdata/frag_attr_L34.json`

### R-MLP-22 脆弱题可按二分树分成五档「最小充分集」（3 / 5 / 11 / 23 个单元 / 打不坏）。

- 领域：MLP 单元 ｜ 置信：high ｜ 已测 1 次 ｜ 不要重测
- 判据：逐题记录「第一次被打坏时所在的最小子集大小」，若同一批题落在同一档则同质，若跨越 ≥3 档则非同质
- 关键数字：tier_claim_17q={'3': [5, 29, 41, 73, 94, 97, 125], '5': [78], '11': [25, 55], '23': [21], 'none': [68]}，single_unit_767_damaged_17q=[5, 25, 29, 41, 73, 78, 94, 97, 125]，counterexamples={'qi=78': 'H12 归到 A′(5) 档，但 767 单点即可打坏', 'qi=25': 'H12 归到「需 A(11)」档，但 767 单点即可打坏'}，core3_subset_of_767=True
- 为什么：错在方法：二分只回答「这条路径上哪一半有用」，不回答「最小集合是哪个」。而且 H12 的 while len(cur) > 3 直接封死了单点，等于预设了答案的下限。qi=78 被 A′(5) 打坏也被 767 单点打坏、qi=25 被 A(11) 打坏也被 767 单点打坏 ⇒ 「档」是路径的产物。正确表述见 R-MET-10（分档必须逐单点测）和 R-MLP-23（单点承载）。
- 复查：已被 H15 直接否证；不要再引用「五档最小充分集」
- 证据：H12 ｜ `tools/mech_carrier.py` ｜ `tempdata/carrier_L34.json`
- 证据：H15 ｜ `tools/mech_core_split.py` ｜ `tempdata/core_split_L34.json`

## 未决（1）

### R-MTH-01 在输出处投影掉组平均写方向（dir 干预）可以定位该组的功能，且效应可被匹配对照分辨。

- 领域：实验方法 ｜ 置信：medium ｜ 已测 1 次 ｜ 重测需加样本
- 判据：dir 干预 KL ≥3× 匹配 sham/gauss 对照，且与层主写方向 |cos| 可解释
- 关键数字：kl_c18=0.02592，sham_mean=0.00209，ratio=12.4，cos_with_layer_dir=≤0.09
- 为什么：KL 是 sham 的 12.4×，但投影方向与层主写方向的 |cos|≤0.09——权重几何看不见、只有干预能看见。这个反差本身可能只是"投影掉了噪声子空间"。
- 复查：需要第二层（如 L14 分布式层）复现才能定论；单层 24 组不足
- 证据：C1 ｜ `tools/mech_c1_full_run.py` ｜ `data/mech/ablate_L24.json`

## 待测（2）

### R-MTH-02 steering（沿方向加）与 ablation（删掉）在同一组单元上效应强度相当。

- 领域：实验方法 ｜ 置信：low ｜ 已测 0 次 ｜ 可以重测
- 判据：同组同规模的加 vs 删，行为改变量比值 ∈[0.5, 2]
- 关键数字：status=not_tested
- 为什么：这是差分对假说最具判别力的一条：加是不对称操作、删是对称操作。如果 steering ≫ ablation，就能解释整个领域"控制有效、消融无效"的普遍困惑。
- 复查：从未做过。若 R-MLP-04 的差分对假说成立，steering 应远强于 ablation

### R-MLP-15 同一侧单元的 ×0 与 ×2 应把正确率推向 base 的两侧（放大与删除方向相反）。

- 领域：MLP 单元 ｜ 置信：medium ｜ 已测 0 次 ｜ 可以重测
- 判据：acc(amp_neg) < acc(base) < acc(amp_pos) 且两端差 ≥5 点；严格口径
- 关键数字：amp=2.0，n_units_each=45，state=queued
- 为什么：这一条不依赖「删除」这个操作本身（H5 已见删除会让输出变模板化），是对 H6 的独立复核。
- 复查：排队中。H6 完成后跑
- 证据：H7 ｜ `tools/mech_ampl.py` ｜ `tempdata/ampl_L34.json`
