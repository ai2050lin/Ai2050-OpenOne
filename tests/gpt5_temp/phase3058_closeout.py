# -*- coding: utf-8 -*-
"""Phase 3058 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3058'
     r'\omega_p55_payload_channel_identity_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
MEMO_W = WLOG_DIR + r'\MEMORY.md'
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
assert verdict == 'gamma_payload_head16_qwen', \
    verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a184_seals_ok'] is True
assert an['a185_gamma_stats_diff'] == 0.0
assert an['a186_tt_diff'] == 0.0
assert an['a187_chain_diff'] == 0.0
assert an['a188_k64_replay_diff'] == 0.0
assert an['a189_ones_diff'] < 0.01
assert an['a190_construction_ok'] is True
t2 = st['T2_identity']
assert abs(t2['colstd_pct_med']
           - 0.136328125) < 1e-12
assert abs(t2['gamma_top64_max'] - 9.75) < 1e-12
t3 = st['T3_vocab_overlap']
assert abs(t3['obs_intersect'] - 21) < 1e-9
assert abs(t3['p_intersect']
           - 0.4767616191904048) < 1e-9
assert abs(t3['p_votes_mass']
           - 0.10744627686156921) < 1e-9
t4 = st['T4_coalition_overlap']
assert t4['obs_coalition_intersect'] == 2
assert abs(t4['p_coalition']
           - 0.48875562218890556) < 1e-9
t6 = st['T6_cum_ladder']
assert abs(t6['r_cum']['8']
           - 0.39421945158399996) < 1e-12
assert abs(t6['r_cum']['16']
           - 0.8702875074411255) < 1e-12
assert abs(t6['r_cum']['32']
           - 0.9851027457043321) < 1e-12
assert abs(t6['med_cos_k64']
           - 0.6710936164405075) < 1e-12
t7 = st['T7_leave_one_in']
assert abs(t7['r_loi_max']
           - 1.0104798091677774) < 1e-12
assert abs(t7['r_loi_min']
           - 0.9367710656126763) < 1e-12
assert abs(t7['r_loi_med']
           - 0.9990873823296611) < 1e-12
t5 = st['T5_vocab_subspace']
assert abs(t5['frac_med']
           - 0.9280925052753397) < 1e-12
assert 'corrections' in res['prereg']
assert res['prereg']['corrections'] == 'none (first run).'

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3058
           for m in led['measurements']):
    claim = (
        'Omega-P55 (plan 3058 A) - top-64 shape '
        'channel identity, run1 fp32 authoritative '
        '(251.7s, zero crashes, anchor core all '
        'bit 0.0: a184 seals 3044-3057, a185 '
        'gamma stats, a186 TT, a187 COS_orig vs '
        'z51 COS_H[7], a188 k=64 arm replay vs '
        'z57 COS_ARM64, a189 ones diff 1.9e-7, '
        'a190 construction). RESULTS: (1) '
        'VERDICT gamma_payload_head16_qwen. (2) '
        'T6 cumulative ladder within S64: r_cum '
        '1=0.080 2=0.162 4=0.345 8=0.394 16='
        '0.870 32=0.985 - the payload needs 16 '
        'channels in combination (r_cum16 = '
        '0.87), single channels are worthless '
        '(r_cum1 = 0.08). (3) T7 leave-one-in '
        '(64 arms, MAIN): EVERY single-channel '
        'marginal is near-full (r 0.937-1.010, '
        'med 0.999) - removing any one channel '
        'does NOT hurt: the 64 channels are '
        'fully redundant one-at-a-time yet '
        'combine into the payload (non-additive, '
        'same signature as the 3051 head loo). '
        '(4) No semantic identity: overlap with '
        'the 2802 signed-vote significant '
        'channels 21 vs expected 20.1 (p=0.48), '
        'votes mass p=0.11, with the 3022 L3 '
        'relay top-64 write channels 2 vs 1.6 '
        '(p=0.49); S64 colstd percentiles med '
        '0.136 (low-variance region). (5) T5 '
        'vocab subspace: 92.8pct (med) of the '
        'gamma-modulated readout row norm flows '
        'through S64; the top tokens are '
        'chat/tool template markers and rare '
        'glyphs (think/tool_call/im_start), not '
        'content words.')
    meas = {
        'meas_id': 'meas3058_omega_p55_payload_'
                   'channel_identity_qwen',
        'phase': 3058,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a184 source seals 3044-3057; '
                   'a185 gamma stats bit 0.0; a186 '
                   'TT diff 0.0; a187 COS_orig '
                   'diff 0.0 vs z51 COS_H[7]; '
                   'a188 k=64 replay diff 0.0 vs '
                   'z57 COS_ARM64; a189 ones diff '
                   '1.9e-7; a190 construction ok; '
                   'a191 nulls on frozen seeds',
        'artifacts': {
            'result': 'phase3058/omega_p55_'
                      'payload_channel_identity_'
                      'qwen/result.json',
            'npz': 'phase3058/omega_p55_'
                   'payload_channel_identity_'
                   'qwen/omega_p55_payload_'
                   'channel_identity_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (251.7s, fp32; '
                '64 leave-one-in arms x 48 forwards '
                '+ 8 ladder arms; zero crashes)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 197
    l14['connects'].append({
        'meas_id': 'meas3058_omega_p55_payload_'
                   'channel_identity_qwen',
        'phase': 3058,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P55: top-64 shape '
                        'channel identity - payload '
                        'is combinatorial (r_cum16 = '
                        '0.87, r_cum1 = 0.08) with '
                        'full one-at-a-time '
                        'redundancy (all 64 '
                        'leave-one-in marginals '
                        '0.937-1.010, med 0.999); NO '
                        'semantic identity (2802 '
                        'votes p=0.48, 3022 coalition '
                        'p=0.49); the S64 subspace '
                        'carries 92.8pct of the '
                        'gamma-modulated readout and '
                        'serves template/control '
                        'tokens (gamma_payload_'
                        'head16_qwen)'})
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
if '## Phase 3058:' not in memo:
    sec = u'''## Phase 3058: Ω-P55 top-64 形状通道身份——组合承重 head16 + 全单通道冗余 + 无语义身份（gamma_payload_head16_qwen） [%(created)s]

**判决：`gamma_payload_head16_qwen`**（run1 fp32 权威 251.7s，锚核心一次全过、零崩溃）。链锚：a184 封印 3044-3057 / a185 γ 统计 bit 0.0 / a186 TT bit 0.0 / a187 COS_orig vs z51 bit 0.0 / **a188 k=64 臂重放 vs z57 COS_ARM64 bit 0.0** / a189 ones 差 1.9e-7 / a190 构造断言。64 个 leave-one-in 臂 × 48 前向 + 8 个 ladder 臂。

### 设计
**T2 身份剖面（离线）**：S64 的 γ 值与 colstd 百分位；**T3 词表重叠（置换）**：S64 ∩ 2802 sig 通道 + |votes| 质量 vs 随机 64 通道 null（R=2000，seed 9971）；**T4 联盟重叠**：w32_τ=Σ_top32 s·dh_j 的跨 tag 均值 top-64 ∩ S64（seed 9972）；**T5 词表子空间内容**：frac_v=‖(W_U[v]⊙(γ−mean))_S64‖/‖W_U[v]⊙(γ−mean)‖；**T6 累积 ladder（前向）**：rank 序 k∈1/2/4/8/16/32 + k=64 重放；**T7 leave-one-in（前向，主检验）**：γ_64 去一通道→mean，64 臂逐通道 marginal。

### 核心结果（重复三遍）
**① 载荷是组合性的（head16）**：rank 累积 r_cum = 1:0.080 / 2:0.162 / 4:0.345 / 8:**0.394** / 16:**0.870** / 32:0.985——单通道近乎无用（r_cum1=0.08），**16 个通道的组合承载 87pct**。**② 全单通道冗余（T7 主检验）**：64 个 leave-one-in marginal 全部 **0.937–1.010（med 0.999）**——去掉任何单个通道几乎不掉，但去掉一半（cum8）只剩 39pct：**通道逐一皆不承重、组合却承重**——与 3051 头级 loo 全谱非可加同签名，载荷住在联合子空间而非任何单通道。**③ 无语义身份**：与 2802 投票 sig 通道重叠 21 vs 期望 20.1（p=0.48）、votes 质量 p=0.11；与 3022 L3 联盟 top-64 写出通道重叠 2 vs 1.6（p=0.49）——**γ 形状通道与语义投票维、MLP 中继联盟均无关联**。**④ S64 服务模板/控制读出**：colstd 百分位 med 0.136（低方差区），词表 γ-调制读出范数的 **92.8pct（med frac）** 流经 S64；frac top token 全是 `</think>`/`<tool_response>`/`<|im_start|>` 类模板标记与生僻字形，非内容词——**读出预对齐基服务于格式/控制 token 的读出吞吐，不编码语义内容**。

### 机制链定版
γ 读出预对齐终局：**稀疏（~64 通道）+ 组合承重（head16，单通道零边际）+ 无语义身份 + 模板/控制读出主干**。读出侧的"定基"是通用反方差几何为格式 token 修的高速通路；行为方向信息 100pct 由写入侧（KV 门位 + 语境组装）供给。"单点消融不翻行为"（P2）在通道维再现：任何单通道都冗余，子空间整体承重。

### 方法论入册
- **leave-one-in 全冗余判别（3058）**：逐通道 marginal 全高 + 累积 ladder 前段低 = 组合承重签名——单通道归因禁止，必须报组合量与边际谱两件事。
- **大矩阵陷阱再防（3058）**：frac 分母 ‖W_U⊙gmod‖ 按词表行分块（全量 151936×2560 fp64 = 2.9GB）。
- **置换检验向量化**：2000 次 argsort(2560) 批量抽样替代逐次 choice。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3058/omega_p55_payload_channel_identity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3059 菜单**——A（主选）**组合承重的子空间几何**：top-16 通道张成子空间 vs 24 目标方向族的对齐结构（为何 16 通道组合够、8 不够——维数/能量解释）+ 累积谱能量曲线拟合；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ）；D 门区 2D 易感图。"好的，继续"即进 3059 A。
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
if '## 二十、3058 增补' not in aud:
    add = u'''
    
---

## 二十、3058 增补：top-64 形状通道身份——组合承重、无语义身份（Omega-P55，判决 gamma_payload_head16_qwen）

1. **组合承重（head16）**：rank 累积 r=0.08/0.39/0.87/0.99（k=1/8/16/32）——载荷需 16 通道组合；64 个 leave-one-in 边际全部 0.94-1.01——逐通道全冗余，与 3051 头级非可加同签名。
2. **无语义身份**：S64 与 2802 投票 sig（p=0.48）、3022 联盟写出通道（p=0.49）均无关联；colstd 百分位 med 0.136。
3. **模板读出主干**：词表 γ-调制读出的 92.8pct 流经 S64，top token 为 think/tool/im_start 模板标记与生僻字形——读出预对齐基服务格式/控制 token，不编码语义内容。
4. HDMCC 修正：两图谱接口的读出端 = 通用反方差几何为控制 token 修的稀疏高速通路；行为方向信息全部由写入侧供给——"读出侧不携带语义"的又一独立证据。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3058' not in prev:
    line = ('- Phase 3058 Omega-P55 payload '
            'channel identity: verdict '
            'gamma_payload_head16_qwen (run1 fp32 '
            '251.7s, zero crashes; anchors a184-'
            'a190 all pass, a188 k=64 replay bit '
            '0.0). RESULTS: cumulative ladder '
            'r=0.08/0.39/0.87/0.99 (k=1/8/16/32) - '
            'combinatorial head16 payload; all 64 '
            'leave-one-in marginals 0.937-1.010 '
            '(med 0.999) = full one-at-a-time '
            'redundancy, non-additive like the '
            '3051 head loo; NO semantic identity '
            '(2802 votes p=0.48, 3022 coalition '
            'p=0.49); S64 carries 92.8pct of the '
            'gamma-modulated readout, top tokens '
            'are think/tool/im_start template '
            'markers. Audit addendum 20; ledger '
            '197/L14 165.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md rewrite (<=3000 chars) ----------
mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧产物；负结果/锚失败/崩溃/nan 如实登记；verdict 单分支赋值；nan 臂无效须预注册 amendment。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性；廓线反演 ratio≥0.9。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 锚形状核对纪律；捕获库跨相位复用（上游 npz+抽样重捕获 bit 锚+派生量逐对复现锚）。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]。
- 无替换基线走专用 forward_plain；γ 扰动协议：weight swap+restore sham+self-replacement 恒等锚。
- weights-only 相位：零前向离线设计；RMSNorm γ 可含负通道（3057：1/2560）log 前断言正性；GPU 离线统计块先 smoke-test。
- **全词表×hidden 大矩阵必须分块（3058：frac 分母 2.9GB 陷阱）**。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影→γ 读出预对齐→γ 谱清点→白化几何→**通道身份（3058：head16 组合承重+全单通道冗余+无语义身份+模板读出主干）**。禁单点归因：头级 loo（3051）、内容自由门（3053）、通道 loo（3058）三重非可加签名。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ=2560 维强偏斜（max 9.75）；γ⊙W_U=反方差重加权读出基；S64=|γ−mean| top-64。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3058）
Ω-P2（3011-3058）：3045-3048 KV 阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 h7 门槽；3053 内容自由门；3054 径向稀释+γ 重整；3055 γ 预对齐；3056 词表 null；3057 反方差重加权（colstd^−3.31 单独复现增益）；**3058 通道身份=head16 组合承重+全单通道冗余+无语义身份+模板读出主干**。终局：承载=门位易感，方向=语境组装，读出=稀疏反方差组合子空间（控制 token 主干）。

## 下一步
- max=3058，下一个 3059（A 主选 **组合承重的子空间几何**——top-16 通道子空间 vs 24 目标方向族对齐+能量曲线解释 8 不够 16 够；B h0 对抗溯源；C DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
