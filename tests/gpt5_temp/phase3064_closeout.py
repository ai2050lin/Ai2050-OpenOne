# -*- coding: utf-8 -*-
"""Phase 3064 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3064'
     r'\omega_p61_ds7b_chain_replication')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['verdict']
assert verdict == 'chain_fragmented_ds7b', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
st = res['stats']
an = st['anchors']
assert an['b0_recapture_diff'] == 0.0
assert an['b1_sham_diff'] == 0.0
assert an['a233_pre_max'] == 0.0
assert an['setup_ok'] is True
assert abs(an['b5_pastkey_rel']
           - 1.0019564471254183) < 1e-12
s1 = st['S1_ladder']
assert s1['pass'] is False
assert abs(s1['medK_27']
           - 0.8291837740859915) < 1e-12
assert abs(s1['medV_27']
           - 0.9134400687623977) < 1e-12
assert abs(s1['medK_late_mean']
           - 0.633951638729189) < 1e-12
assert abs(s1['medK_early_mean']
           - 0.3859188275918856) < 1e-12
s2 = st['S2_gate']
assert s2['pass'] is True
assert abs(s2['medJ']
           - 0.9168312790259341) < 1e-12
assert abs(s2['k_share']
           - 0.9044017073315151) < 1e-12
assert s2['h_adv'] == 2 and s2['h_pos'] == 1
assert abs(s2['medJh'][2]
           + 0.00368164850787656) < 1e-12
assert abs(s2['medJh'][1]
           - 0.9114034898838066) < 1e-12
s3 = st['S3_pipe']
assert s3['pass'] is False
assert s3['v_adv_tan'] is False
assert abs(s3['med_cos_d']
           + 0.7313192784790501) < 1e-12
assert abs(s3['med_cos_dtan']
           + 0.764841112418096) < 1e-12
assert abs(s3['med_cos_dtan_only']
           - 0.8929396553922329) < 1e-12
assert abs(s3['med_cos_lg_pred']
           - 0.9814523973306835) < 1e-12
assert abs(s3['med_tan_frac']
           - 0.7280099477303159) < 1e-12
assert abs(s3['med_ka_cos_d']
           - 0.5726095159824962) < 1e-12
assert abs(s3['gamma_skew']
           - 3.7555555555555555) < 1e-12
assert abs(s3['sv_share0']
           - 0.9995953205767881) < 1e-12
s4 = st['S4_write']
assert s4['pass'] is True
assert abs(s4['share_body']
           - 0.9843151536809376) < 1e-12
assert abs(s4['share_prefix']
           - 0.008961970694237036) < 1e-12
assert abs(s4['between_body_cos_med']
           - 0.8665572075119747) < 1e-12
s5 = st['S5_source']
assert s5['pass'] is False
assert s5['axis_dominance'] is False
assert s5['ss_count'] == 2
assert abs(s5['ss1_obs']
           - 0.0002837473255608014) < 1e-12
assert abs(s5['ss1_p']
           - 0.17691154422788605) < 1e-12
assert abs(s5['c_pc1_ka']
           - 0.9863879668191888) < 1e-12
assert abs(s5['ss2_p']
           - 0.0004997501249375312) < 1e-12
assert abs(s5['c_pc1_kpos']
           - 0.999286542146401) < 1e-12
assert abs(s5['ss1_p_kpos']
           - 0.0009995002498750624) < 1e-12
assert abs(s5['jaccard_obs']
           - 0.1990632318501171) < 1e-12
assert abs(s5['ss3_p']
           - 0.0004997501249375312) < 1e-12
assert abs(s5['med_ka']
           - 0.5726095159824962) < 1e-12
assert abs(s5['med_cos_dv_vmat']
           - 0.04529082292419627) < 1e-12
assert abs(s5['p_kx']
           - 0.00399800099950025) < 1e-12
assert abs(s5['med_abs_cos_dv_pc1']
           - 0.04417910178581093) < 1e-12
assert s5['channel_V']['ov_s16'] == 10
assert s5['channel_V']['ov_t64'] == 38
assert abs(s5['channel_V']['e16']
           - 0.01082777212725005) < 1e-12
assert abs(s5['channel_V']['e64']
           - 0.6941281678842874) < 1e-12
assert s5['channel_KA']['ov_s16'] == 8
assert s5['channel_KA']['ov_t64'] == 23
assert abs(s5['channel_KA']['e16']
           - 0.012024142545517444) < 1e-12
assert abs(s5['channel_KA']['e64']
           - 0.07764005276342689) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3064
           for m in led['measurements']):
    claim = (
        'Omega-P61 (plan 3064 A) - cross-model '
        'DS7B full-chain replication of the '
        'Omega-P2 mechanism chain on deepseek-'
        'r1-distill-qwen-7b (Qwen2 arch, 28L, '
        'GQA 28q/4kv, hidden 3584, bf16 native '
        'authoritative, 108.9s, ~1600 forwards; '
        'anchors b0/b1 bit 0.0, a233 stage-pre '
        '0.0, b3 finite, b5 past-key rel 1.002 '
        'recorded-only - replacement confirmed '
        'at cache level). Debug history: 5 '
        'launches, 4 script bugs fixed '
        '(transformers 5.14.1 DynamicCache API '
        'probe -> cache.layers[li].keys; S4 '
        'ANOVA off-by-one C_c[cidx-1]; del wud '
        'placement after last use). RESULTS: '
        'verdict chain_fragmented_ds7b (S1-S5 '
        '= F,T,F,T,F). STRUCTURAL components '
        'replicate: S2 gate anatomy (K_share '
        '0.904 >= 0.8; single-head carry '
        '0.9114; 4-kv-head split cleanly '
        'separates adversarial head h2 med '
        '-0.0037 vs positive head h1 +0.9114) '
        'and S4 write decomposition (SS_body '
        '0.984 vs qwen 0.827; body d_eff '
        '~1.006 near-pure single-rank < '
        'prefix 1.433; between-body |cos| '
        '0.867). SYMBOLIC components flip '
        'sign: S1 medV(27)=+0.913 (qwen '
        '-0.34) while medK(27)=0.829 and '
        'late>early replicate; S3 cos_d '
        '-0.731 / cos_dtan -0.765 (qwen '
        'sign) but exact-norm counterfactual '
        'dtan_only=+0.893 (qwen -0.341) - '
        'DS7B negative V effect lives in '
        'radial-through-norm nonlinearity; '
        'S5 axis_dominance=False (V<->'
        'positive-head PC1 0.9993 beats '
        'V<->adversarial 0.986), med_ka '
        '+0.573 (qwen negative). Shared-axis '
        'EXISTENCE is structural (SS2 |cos '
        'PC1| 0.986, SS3 Jaccard 0.199 vs '
        'null 0.0, both p 0.0005; SS1 per-'
        'pair null p 0.177 within-family) '
        'while axis DOMINANCE is qwen-'
        'specific. New: the V arm is an '
        'ALLY of the positive head h1 in '
        'DS7B (per-pair diag perm p 0.001 '
        'vs qwen null) - V-only replacement '
        'is adversarial in qwen3 but not '
        'universally. Omega-P2 chain = '
        'topology layer (universal '
        'candidate) + sign-orchestration '
        'layer (training-specific).')
    meas = {
        'meas_id': 'meas3064_omega_p61_ds7b_'
                   'chain_replication',
        'phase': 3064,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'b0 recapture diff 0.0 '
                   '(LG+KB+VB, 4 prompts); b1 '
                   'sham bit identity; a233 '
                   'stage-pre identity 0.0; b3 '
                   'gamma/wud/TT finite; b5 '
                   'past-key rel 1.002 '
                   '(recorded-only, bf16; '
                   'confirms replacement '
                   'reaches KV cache)',
        'artifacts': {
            'result': 'phase3064/omega_p61_'
                      'ds7b_chain_replication/'
                      'result.json',
            'npz': 'phase3064/omega_p61_'
                   'ds7b_chain_replication/'
                   'omega_p61_ds7b_chain_'
                   'replication.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (bf16 '
                'native; 4 crashed launches '
                'pre-verdict registered - '
                'cache API, ANOVA index, del '
                'placement; all script bugs, '
                'no data issue)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 203
    l14['connects'].append({
        'meas_id': 'meas3064_omega_p61_ds7b_'
                   'chain_replication',
        'phase': 3064,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P61: Omega-P2 '
                        'chain is TWO-LAYER - '
                        'topology (last-layer KV '
                        'fields, gate-region '
                        'single-head carry + '
                        'single-head adversarial '
                        'split h2/h1, body '
                        'single-rank identity SS '
                        '0.984, gamma pipe skew '
                        '3.76, family-level '
                        'shared-axis existence '
                        'SS2/SS3 p 0.0005) '
                        'replicates on DS7B; '
                        'sign orchestration (V '
                        'adversarial flip, '
                        'tangential vs radial '
                        'locus, axis dominance) '
                        'is qwen-specific - DS7B '
                        'V arm allies with '
                        'positive head h1 (|cos '
                        'PC1| 0.9993, per-pair '
                        'p 0.001). Universality '
                        'claims must be stated '
                        'per-layer '
                        '(chain_fragmented_'
                        'ds7b)'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted n=%d l14=%d'
             % (len(led['measurements']),
                len(l14['connects'])))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3064:' not in memo:
    sec = u'''## Phase 3064: Ω-P61 跨模型 DS7B 复刻全链——拓扑普适、符号编排 qwen 特异（chain_fragmented_ds7b） [%(created)s]

**判决：`chain_fragmented_ds7b`**（S1–S5 = [F, T, F, T, F]，setup_ok=True；bf16 权威 108.9s，约 1600 次前向；锚 b0 recapture bit 0.0、b1 sham bit 0.0、a233 stage-pre 0.0、b3 finite；b5 past-key rel=1.002 仅记录——证实替换机械到达 KV cache 存储层）。调试史如实入册：5 次启动、4 个脚本 bug 全部修复后成功，均非数据问题——①b5 DynamicCache 下标（transformers 5.14.1 移除 key_cache/`__getitem__`/to_legacy_cache）；②shim fallback 仍走下标分支→探针定案唯一路径 `cache.layers[li].keys`；③S4 ANOVA off-by-one（cidx 为 1-based 条件 ID，`C_c[cidx]`→`C_c[cidx-1]`）；④`del wud` 误插在 `Gmet=wud.T@wud` 之前（删除必须在最后使用点之后）。

### 问题与设计
**问题**（3063 菜单 A）：Ω-P2 机制链（KV 阶梯→门区→γ 管道→PC1 身份→写入分解→同源对抗）是 qwen3-4b 特异还是语言编码普适？在 deepseek-r1-distill-qwen-7b（Qwen2 架构：28 层、GQA 28q/4kv、hidden 3584、无 k_norm、bf16 原生）上预注册五 stage 复刻：S1 KV 阶梯（28 层×24 对 V-only/K-only）；S2 L27 门区解剖（K_share+4 kv 头分解）；S3 stage 臂+fp64 手工 RMSNorm 反事实；S4 写入分解 ANOVA；S5 同源三检验+特异性对照+通道身份。判决树观测前冻结（execution.json，seed 3010）。

### 核心结果（重复三遍）
**① 拓扑层普适（一）**：S2 门区解剖 PASS——K_share=0.904≥0.8、单头承载 0.9114≥0.6×medJ、4 kv 头干净分离对抗头 h2（medJh=−0.0037）与正头 h1（+0.9114）——DS7B 的 KV 联合场同样由单一 kv 头承载+单一 kv 头对抗。**② 拓扑层普适（二）**：S4 写入分解 PASS——SS_body=0.9843≥0.5（qwen 3061 为 0.827，更强）、body 有效秩≈1.006（近纯单秩）≪ prefix 1.433、body 间 |cos| med=0.867——body 身份是跨前缀不变的近单秩写入方向。**③ 符号层反转（三遍）**：S1 FAIL——K 阶梯复刻（medK(27)=0.829>0.15、late 0.634>early 0.386）但 medV(27)=+0.913 不为负（qwen −0.34）；S3 FAIL——cos_d=−0.731 与 cos_dtan=−0.765 与 qwen 同号，但精确 norm 反事实后 dtan_only=+0.893 为正（qwen −0.341）——DS7B 的 V 臂负效应住在径向分量经 norm 的非线性里，纯切向分量单独反而正对齐；S5 FAIL——ss_count=2（SS2 |cos PC1(V,KA)|=0.986 p=0.0005、SS3 Jaccard=0.199 vs null 0.0 p=0.0005 成立）但 axis_dominance=False（V↔正头 PC1 对齐 0.9993 **高于** V↔对抗头 0.986）且 med_ka=+0.573>0。γ 管道结构在（skew=3.76≥2、sv_share0=0.9996）。

### 结构性解读
Ω-P2 链的普适性第一次被分层测量：**拓扑组件（末层 K+V 场、门区单头承载+单头对抗、body 单秩身份、γ 管道、族级共享轴存在性）跨模型成立；符号编排（V 臂是否翻转、负效应住切向还是径向、对抗轴是否支配）是 qwen3 特异**。新意外：T4 正对照 V↔h1K 的逐对 diag perm p=0.0010 显著（qwen 中为 null）——DS7B 的 V 臂是正头同盟而非对抗者，与 medV=+0.913、dtan_only=+0.893 的正号一致；"V-only 替换=对抗"是模型特异编排，不是链的必要组成。竞争轴在 DS7B 有族级轴（SS2/SS3 p=0.0005）但不再支配——竞争强度是连续量而非二元机制。

### 硬伤与边界
- **bf16 权威**（7.6B fp32 超 16GB 显存）：符号结论尺度（±0.9 vs ±0.3）远超噪声稳健，但 SS p 值精度受限（置换 null 同精度，纪律一致）。
- **蒸馏混杂**：R1-distill 训练轨迹可能改变符号编排——"模型家族"与"蒸馏效应"不可分离。
- **GQA 头数不同**（4kv vs 8kv）：头级比较是结构性的而非逐头映射。
- **S1 判据敏感性**：medV(27)<0 预注册自 qwen 现象——符号反转触发判据失败，这正是 fragmented 判决的正确读法（失败定位于符号组件而非全链缺失）。
- 24 对继承 3061 骨架；SS1 族内逐对 null（p=0.177）与 qwen 一致——共享轴仍是族级。

### 方法论入册
- **transformers 5.14.1 DynamicCache**：key_cache/下标/to_legacy_cache 全部移除，唯一路径 `cache.layers[li].keys`（形状不变 (b,kv,s,d)）——跨版本缓存 API 必须探针先行（tests/gpt5_temp/probe_3064_cache.py）。
- **1-based 条件 ID**：cidx∈{1,2,3} 直接当 np.stack 位置索引是跨脚本移植高发 bug——按 ID 索引堆叠数组必须减 1。
- **del 位置纪律**：内存优化删除必须放在最后使用点之后（`del wud` 插在 Gmet 之前即崩）。

### 智能理论洞察（第一性原理）
跨模型检验把"语言编码数学结构"的普适性问题切成了两层：**组织拓扑 vs 数值编排**。若语言能力来自某种数学结构，则拓扑层应是结构的直接体现——门区单头承载、body 单秩身份、γ 反方差管道、族级共享轴在 7B 蒸馏 Qwen2 上原样复现；而符号编排更像训练动力学的偶然解——同一结构的不同实现。这直接回答 3063 的遗留问题：**"存在共享竞争轴"是结构的，"轴支配关系"是编排的**。下一层第一性原理问题：符号编排由什么决定（训练数据顺序、规模、还是蒸馏目标）？——需要符号的层间溯源实验，即 3065 A。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3064/omega_p61_ds7b_chain_replication/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3065 菜单**——A（主选）**符号编排溯源**：V 臂符号的层间分解（qwen3-4b vs DS7B 各层 V-only 置换的符号引入点+unembed 投影），判决"符号=训练轨迹"假说；B 门区 2D 易感图（(layer, head) 平面完整地图）；C body 指纹下游消费定位（3016 放大追踪法）；D 竞争轴源头定位（共享轴通道支撑的 L35 之前写入来源）；E（新增）DS7B V↔h1 同盟机制溯源（逐对配对 p=0.001 的信号内容是什么）。"好的，继续"即进 3065 A。
''' % {'created': created,
           'script8': seal['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
           'n': len(led['measurements']),
           'l14': len(l14['connects'])}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 二十六、3064 增补' not in aud:
    add = u'''

---

## 二十六、3064 增补：跨模型 DS7B 复刻全链——拓扑普适、符号编排 qwen 特异（Omega-P61，判决 chain_fragmented_ds7b）

1. **拓扑组件跨模型复现**：DS7B（Qwen2 架构、GQA 4kv 头、bf16）上 S2 门区解剖（K_share 0.904、单头承载 0.9114、对抗头 h2 −0.0037 vs 正头 h1 +0.9114）与 S4 写入分解（SS_body 0.9843、body 有效秩 1.006 vs prefix 1.433）PASS——末层 KV 双场、门区单头承载+单头对抗分离、body 单秩身份是 Transformer 层面的候选普适结构。
2. **符号组件模型特异**：S1（medV(27)=+0.913，qwen −0.34）、S3（dtan_only=+0.893，qwen −0.341——DS7B 负效应住径向经 norm 非线性）、S5（axis_dominance=False、med_ka=+0.573）全部败于符号/支配反转；V 臂在 DS7B 是正头 h1 的同盟（|cos PC1|=0.9993、逐对配对 p=0.001），非对抗者——"V-only 替换=对抗"不是普适命题。
3. **共享轴存在性是结构的**：SS2 |cos PC1(V,KA)|=0.986、SS3 Jaccard=0.199（双 p=0.0005，SS1 族内逐对 null p=0.177）——族级共享轴仍在，但支配关系是编排的；竞争强度是连续量而非二元机制。
4. HDMCC 修正：Ω-P2 机制链升级为"拓扑层（普适候选）+符号层（模型编排）"双层结构；跨模型普适性主张必须分层陈述；单一模型上的机制链完整复现（chain_replicated）不再是普适性的充分证据。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-21.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3064' not in prev:
    line = ('- Phase 3064 Omega-P61 cross-model '
            'DS7B full-chain replication: verdict '
            'chain_fragmented_ds7b (bf16 108.9s; '
            '5 launches - 4 script bugs fixed incl '
            'transformers 5.14.1 DynamicCache '
            'probe cache.layers[li].keys, S4 '
            'ANOVA cidx-1 off-by-one, del wud '
            'placement). STRUCTURAL: S2 gate '
            'anatomy PASS (K_share 0.904, '
            'adversarial head h2 -0.004 vs '
            'positive h1 +0.911), S4 write '
            'decomposition PASS (SS_body 0.984, '
            'body d_eff 1.006 < prefix 1.433). '
            'SYMBOLIC FLIPS: S1 medV(27)=+0.913 '
            '(qwen -0.34), S3 dtan_only +0.893 '
            '(qwen -0.341; radial-norm locus), '
            'S5 axis_dominance False (V<->h1 '
            '0.9993 > V<->h_adv 0.986; med_ka '
            '+0.573). Shared-axis existence '
            'structural (SS2 0.986, SS3 Jaccard '
            '0.199, p 0.0005). Omega-P2 = '
            'topology (universal candidate) + '
            'sign orchestration (qwen-specific); '
            'DS7B V arm allies with positive '
            'head h1 (per-pair p=0.001). Audit '
            'addendum 26; ledger 203/L14 171.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3064' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 架构 28L GQA 28q/4kv hidden 3584 bf16 原生）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- SVD 符号任意性用 |cos|；json 禁 numpy 标量；fp64 W_U 用后即 del（必须在最后使用点之后）。
- **3063**：wud vocab-major——G=W_U·W_Uᵀ 写成 wud.T @ wud；同源分族级/逐对两层。
- **3064**：transformers 5.14.1 DynamicCache 唯一路径 cache.layers[li].keys（key_cache/下标/legacy 全移除，先探针）；1-based 条件 ID（cidx 1..3）当位置索引用必须 -1；bf16 符号结论尺度须远超噪声。

## 机制解释审计链（命名前依次检查）
…→KV 阶梯→体位均匀→末层→K 场+头分解→门位易感→norm 投影→γ 预对齐→反方差重加权→通道身份→PC1→写入分解→身份解码→3063 竞争轴（写/门同源，管道内第三轴）→**3064 跨模型：拓扑普适（门区单头承载+对抗分离、body 单秩 SS 0.984、γ 管道、共享轴存在）；符号编排 qwen 特异（V 翻转/切向/支配反转；DS7B V 臂=正头 h1 同盟）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- cmd.exe 经 bash 损坏（参数截断）——删除/列目录用 Python。
- 关键写入后必须 Grep/Read 复核；幻影 Edit 会再现——Python 补丁 assert count==1 唯一可靠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3064）
Ω-P2（3011-3064）：3045-3048 KV 阶梯；3050 末层；3051 K 场+头；3052-3053 门槽+内容自由门；3054 γ 重整；3055 γ 预对齐；3057 反方差重加权；3060 PC1 身份；3061 写入分解；3062 身份解码 opaque；3063 竞争轴（|cos PC1| 0.341、Jaccard 0.138、正对照 0.138/0.019）；**3064 DS7B：chain_fragmented——结构层复现（K_share 0.904、SS_body 0.984、共享轴 p=0.0005），符号层反转（medV +0.913 vs −0.34、dtan_only +0.893、axis_dom=False）**。终局（3064 修订）：承载=门位易感（普适）；方向=语境组装 body 单秩（普适）+竞争轴（存在普适、支配特异）；读出=单轴传输；γ=反方差放大（普适）。

## 下一步
- max=3064，下一个 3065（A 主选 **符号编排溯源**：V 臂符号层间分解 qwen3 vs DS7B；B 门区 2D 易感图；C body 指纹下游消费；D 竞争轴源头定位；E DS7B V↔h1 同盟机制）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3064')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
