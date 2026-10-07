# -*- coding: utf-8 -*-
"""Phase 3059 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3059'
     r'\omega_p56_payload_subspace_qwen')
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
assert verdict == 'payload_subspace16_qwen', \
    verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a192_seals_ok'] is True
assert an['a193_gamma_stats_diff'] == 0.0
assert an['a194_tt_diff'] == 0.0
assert an['a195_top64_diff'] == 0.0
assert an['a196_chain_diff'] == 0.0
assert an['a197_dtan_replay_ok'] is True
t2 = st['T2_family_spectrum']
assert abs(t2['d_eff']
           - 1.340331839152761) < 1e-9
assert abs(t2['fam_energy_S16_share']
           - 0.5186384996487651) < 1e-12
assert abs(t2['fam_energy_S64_share']
           - 0.6096349513899847) < 1e-12
assert abs(t2['pc_energy_S16_top8'][0]
           - 0.5192576515832783) < 1e-12
t3 = st['T3_E16_enrichment']
assert abs(t3['med_E16']
           - 0.5183739858571768) < 1e-12
assert abs(t3['null16of64_med']
           - 0.14377366616475543) < 1e-12
assert abs(t3['p16_of64']
           - 0.0004997501249375312) < 1e-9
assert abs(t3['med_E64']
           - 0.6103391615477036) < 1e-12
assert abs(t3['p64_of2560']
           - 0.0004997501249375312) < 1e-9
t5 = st['T5_dtan_subspace']
assert abs(t5['med_DE16']
           - 0.13205730764646595) < 1e-12
assert abs(t5['p_d16_of64']
           - 0.0004997501249375312) < 1e-9
assert 'corrections' in res['prereg']
assert res['prereg']['corrections'] == 'none (first run).'

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3059
           for m in led['measurements']):
    claim = (
        'Omega-P56 (plan 3059 A) - payload '
        'subspace geometry, run1 fp32 '
        'authoritative (24.5s, zero crashes; '
        'a192 seals 3044-3058, a193 gamma stats '
        'bit 0.0, a194 TT bit 0.0, a195 TOP64 '
        'recompute bit 0.0 vs z58, a196 COS_orig '
        'diff 0.0 vs z51 COS_H[7], a197 d_tan '
        'replay medians bit-equal to z55). '
        'RESULTS: (1) VERDICT payload_subspace16_'
        'qwen. (2) The 24 gamma-weighted target '
        'readout preimages v_k = gamma*w_t are '
        'nearly ONE-dimensional: d_eff = 1.34 '
        '(sv0 = 929198 vs sv1 = 21689) - all 24 '
        'targets share a single dominant readout '
        'axis. (3) T3 enrichment: med E16 = 0.518 '
        'vs 16-of-64 null 0.144 (p = 0.0005) and '
        '16-of-2560 null 0.002; med E64 = 0.610 '
        'vs 64-of-2560 null 0.011 (p = 0.0005) - '
        'the family energy concentrates in the '
        'S16 channel coordinates far beyond any '
        'random channel set. (4) T5 write side: '
        'd_tan E16 = 0.132 vs null 0.037 (p = '
        '0.0005), E64 = 0.195 - the write '
        'direction shares the channel support '
        'but at 4x lower concentration; the '
        'readout-side subspace is the dominant '
        'carrier. (5) WHY head16: 16 channels '
        'cover the channel support of the '
        'dominant family PC (52 pct of its '
        'energy in S16 vs 14 pct for the 8-rank '
        'prefix geometry of 3058 r_cum8 = 0.39) '
        '- head16 is a dimensionality '
        'consequence of an effectively 1-d '
        'family with a 16-channel support.')
    meas = {
        'meas_id': 'meas3059_omega_p56_payload_'
                   'subspace_qwen',
        'phase': 3059,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a192 source seals 3044-3058; '
                   'a193 gamma stats bit 0.0; a194 '
                   'TT diff 0.0; a195 TOP64/FLAT64 '
                   'diff 0.0 vs z58; a196 COS_orig '
                   'diff 0.0 vs z51 COS_H[7]; a197 '
                   'd_tan replay medians bit-equal '
                   'to z55',
        'artifacts': {
            'result': 'phase3059/omega_p56_'
                      'payload_subspace_qwen/'
                      'result.json',
            'npz': 'phase3059/omega_p56_'
                   'payload_subspace_qwen/'
                   'omega_p56_payload_subspace_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (24.5s, fp32; '
                '48 capture forwards + offline '
                'SVD/permutation analysis)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 198
    l14['connects'].append({
        'meas_id': 'meas3059_omega_p56_payload_'
                   'subspace_qwen',
        'phase': 3059,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P56: payload '
                        'subspace geometry - the '
                        '24 readout preimages are '
                        'nearly one-dimensional '
                        '(d_eff = 1.34) and their '
                        'energy concentrates in the '
                        'S16 channel coordinates '
                        '(E16 = 0.518 vs null 0.144, '
                        'p = 0.0005; E64 = 0.610 vs '
                        '0.011); write-side d_tan '
                        'shares the support at 4x '
                        'lower concentration (0.132 '
                        'vs 0.037) - head16 is a '
                        'dimensionality consequence '
                        '(payload_subspace16_qwen)'})
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
if '## Phase 3059:' not in memo:
    sec = u'''## Phase 3059: Ω-P56 组合承重的子空间几何——方向族近一维 + S16 通道支撑（payload_subspace16_qwen） [%(created)s]

**判决：`payload_subspace16_qwen`**（run1 fp32 权威 24.5s，锚核心一次全过、零崩溃）。链锚：a192 封印 3044-3058 / a193 γ 统计 bit 0.0 / a194 TT bit 0.0 / a195 TOP64 重算 vs z58 bit 0.0 / a196 COS_orig vs z51 bit 0.0 / a197 d_tan 重放 medians 与 z55 bit 相等。48 捕获前向 + 离线 SVD/置换分析。

### 设计
**T2 方向族谱（离线）**：v_k=γ⊙W_U^T t_k（24×2560）SVD → d_eff + 逐 PC 在 S16/S64 的能量；**T3 E16 富集（主检验）**：med‖v_k[S16]‖²/‖v_k‖² vs 16-of-64（seed 9981）/16-of-2560（seed 9982）随机 null（R=2000）；**T5 写入侧收口**：d_tan 在 S16/S64 的能量 vs 同 null（48 前向捕获）。

### 核心结果（重复三遍）
**① 方向族几乎一维**：24 个读出预像的 SVD **d_eff=1.34**（sv0=929198 ≫ sv1=21689）——so/because/therefore/however/while/yet/although/thus 八体三前缀的 24 个目标共享**单一主读出轴**。**② 家族能量强烈富集于 S16 通道坐标**：med E16=**0.518** vs 16-of-64 null 0.144（**p=0.0005**）、16-of-2560 null 0.002；med E64=**0.610** vs 64-of-2560 null 0.011（p=0.0005）；主 PC 自身能量 51.9pct 在 S16——**16 个通道正好覆盖主 PC 的通道支撑**。**③ 写入侧共享支撑但低浓度**：d_tan E16=**0.132** vs null 0.037（p=0.0005）、E64=0.195——写入方向与读出族同住 S16 支撑，但浓度只有读出侧的 1/4。**④ head16 解释**：3058 的 r_cum8=0.39（top-8 rank 只覆盖部分支撑）→ r_cum16=0.87（覆盖主 PC 支撑）——**head16 是"近一维方向族 × 16 通道支撑"的维数后果**，非任意组合。

### 机制链定版
读出侧终局几何：**γ 反方差重加权把读出基拉成"单主轴 + S16 通道支撑"的窄管道**——所有行为相关读出（24 目标方向族）挤在这一管道里，通道逐一冗余、组合承重。写入侧 d_tan 同支撑低浓度=写入只需把方向送进管道入口，管道本身（γ+S16）完成放大。通道身份无语义（3058）与维数解释（3059）合并：**读出管道是任务无关的传输基建，语义全部在写入侧与语境**。

### 方法论入册
- **方向族 d_eff 判据（3059）**：v_k 族 SVD 有效维数 + 通道支撑富集置换——"组合承重为何需要 k 个通道"的维数分解模板。
- **读写同支撑异浓度对照（3059）**：d_tan vs v_k 在同一通道集的能量对比——读写关系的第四种测量（前三种：路由/内容/门位）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3059/omega_p56_payload_subspace_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3060 菜单**——A（主选）**主读出轴 PC1 的身份**：PC1（sv0=929198）是什么——与 24 个 t_k 的共享分量分解、与 2802 投票谱/unembed 主轴的关系、跨 body 一致性（为何八体共享一轴）；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ 管道）；D 门区 2D 易感图。"好的，继续"即进 3060 A。
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
if '## 二十一、3059 增补' not in aud:
    add = u'''
    
---

## 二十一、3059 增补：组合承重的子空间几何——方向族近一维、S16 支撑（Omega-P56，判决 payload_subspace16_qwen）

1. **方向族近一维**：24 个目标读出预像 d_eff=1.34——八体三前缀共享单一主读出轴；主 PC 能量的 51.9pct 住在 S16 通道坐标。
2. **S16 富集**：med E16=0.518 vs 16-of-64 null 0.144（p=0.0005）、E64=0.610 vs null 0.011——3058 head16 是维数后果：16 通道覆盖近一维方向族的通道支撑。
3. **读写同支撑异浓度**：写入侧 d_tan E16=0.132 vs null 0.037（p=0.0005），浓度为读出的 1/4——写入送方向进管道入口，γ+S16 管道完成放大。
4. HDMCC 修正：两图谱接口的读出端 = 任务无关的窄传输管道（单主轴 × 16 通道支撑），语义全部在写入侧与语境——读出基建与语义内容完全解耦的定量版。
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
if 'Phase 3059' not in prev:
    line = ('- Phase 3059 Omega-P56 payload '
            'subspace geometry: verdict '
            'payload_subspace16_qwen (run1 fp32 '
            '24.5s, zero crashes; a192-a197 all '
            'pass). RESULTS: the 24 gamma-weighted '
            'readout preimages are nearly 1-d '
            '(d_eff=1.34, sv0 dominates); med '
            'E16=0.518 vs 16-of-64 null 0.144 '
            '(p=0.0005), E64=0.610 vs 0.011 - the '
            'family energy lives in the S16 '
            'channel support; write-side d_tan '
            'E16=0.132 vs null 0.037 (4x lower '
            'concentration, shared support). '
            'head16 (3058) is a dimensionality '
            'consequence. Audit addendum 21; '
            'ledger 198/L14 166.\n')
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
- weights-only 相位；RMSNorm γ 可含负通道 log 前断言正性；GPU 块先 smoke-test；全词表×hidden 大矩阵分块。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影→γ 读出预对齐→γ 谱清点→白化几何→通道身份→**子空间几何（3059：方向族 d_eff=1.34 近一维；S16 支撑富集 p=0.0005；写入同支撑 1/4 浓度；head16=维数后果）**。禁单点归因：头级 loo（3051）、内容自由门（3053）、通道 loo（3058）三重非可加签名。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ 强偏斜（max 9.75）；γ⊙W_U=反方差重加权读出基；S16=TOP64[:16] 主支撑。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3059）
Ω-P2（3011-3059）：3045-3048 KV 阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 门槽；3053 内容自由门；3054 径向稀释+γ 重整；3055 γ 预对齐；3056 词表 null；3057 反方差重加权；3058 head16 组合承重+无语义身份；**3059 子空间几何=方向族近一维（d_eff 1.34）× S16 通道支撑（读写同支撑异浓度）**。终局：承载=门位易感，方向=语境组装，读出=单主轴窄管道（任务无关传输基建，语义在写入侧）。

## 下一步
- max=3059，下一个 3060（A 主选 **主读出轴 PC1 身份**——与 24 t_k 共享分量/2802 投票谱/unembed 主轴关系+跨 body 一致性；B h0 对抗溯源；C DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
