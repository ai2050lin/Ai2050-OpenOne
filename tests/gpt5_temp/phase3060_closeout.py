# -*- coding: utf-8 -*-
"""Phase 3060 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3060'
     r'\omega_p57_pc1_identity_qwen')
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
assert verdict == 'pc1_shared_piped_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a198_seals_ok'] is True
assert an['a199_tt_diff'] == 0.0
assert an['a200_gamma_stats_diff'] == 0.0
assert an['a201_top64_diff'] == 0.0
assert an['a202_vmat_diff'] == 0.0
assert an['a203_sv_diff'] == 0.0
assert an['a204_dtan_diff'] == 0.0
t2 = st['T2_shared_decomposition']
assert abs(t2['pc1_var_share']
           - 0.9983339324206975) < 1e-12
assert abs(abs(t2['cos_pc1_vbar'])
           - 0.999671209164029) < 1e-12
assert abs(t2['min_cos_pc1_b']
           - 0.9978803237094269) < 1e-12
assert abs(t2['min_loo_cos']
           - 0.9999857831337567) < 1e-12
assert abs(min(t2['d_eff_b'])
           - 1.0518081260125272) < 1e-12
t3 = st['T3_pc1_channels']
assert abs(t3['E16_pc1']
           - 0.5192576515832783) < 1e-12
assert abs(t3['null16_med']
           - 0.002086033423238257) < 1e-12
assert abs(t3['p_E16']
           - 0.0004997501249375312) < 1e-9
assert abs(t3['E64_pc1']
           - 0.6102624580664004) < 1e-12
assert t3['ov16'] == 11
assert t3['ov64'] == 31
assert abs(t3['rho_pc1_votes']
           - 0.030201609056509864) < 1e-12
assert t3['sig256'] == 86
assert abs(t3['p_sig']
           - 0.24087956021989004) < 1e-9
t4 = st['T4_token_decode']
assert abs(abs(t4['med_cos_pc1_wt'])
           - 0.8512984836424156) < 1e-12
t5 = st['T5_write_side_axis']
assert abs(t5['d_eff_dtan']
           - 8.737576002501802) < 1e-12
assert abs(abs(t5['cos_pc1_dtan_vs_readout'])
           - 0.14189810583595364) < 1e-12
assert 'corrections' in res['prereg']
assert 'run2' in res['prereg']['corrections']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3060
           for m in led['measurements']):
    claim = (
        'Omega-P57 (plan 3060 A) - identity of '
        'the main readout axis PC1, run3 fp32 '
        'weights-only authoritative (21.6s, zero '
        'forwards; a198 seals 3044-3059, a199 TT '
        'bit 0.0, a200 gamma stats bit 0.0, a201 '
        'TOP64 recompute bit 0.0 vs z58, a202 '
        'Vmat recompute bit 0.0 vs z59, a203 '
        'SV/d_eff bit 0.0 vs z59, a204 D_TAN bit '
        '0.0 vs z59). run1 crashed pre-verdict '
        '(T4 per-body matmul shape), run2 '
        'completed but crashed at serialization '
        'and used sign-naive SVD criteria - '
        'registered. RESULTS: (1) VERDICT '
        'pc1_shared_piped_qwen. (2) PC1 is a '
        'perfectly SHARED cross-body axis: '
        'per-body sub-family PC1 |cos| = 0.998-'
        '1.000 (all 8 bodies), d_eff_b = 1.05-'
        '1.14 (each body alone is near 1-d), '
        'leave-one-body-out |cos| >= 0.99999; '
        'PC1 carries 99.8 pct of family '
        'variance and |cos(PC1, v_bar)| = '
        '0.9997 - the shared axis IS the mean '
        'preimage. (3) PC1 lives in the S16/S64 '
        'piped support: E16 = 0.519 vs null '
        '0.002 (p = 0.0005), E64 = 0.610 vs '
        '0.011 (p = 0.0005); top-256 |PC1| '
        'channels overlap S16 11 vs null 1, '
        'TOP64 31 vs null 6 (both p = 0.0005). '
        '(4) NO semantic identity: '
        'Spearman(|PC1|, |votes2802|) = 0.030, '
        'top-256 sig enrichment p = 0.24; top '
        'tokens of W_U@PC1 are '
        'punctuation/format glyphs. (5) '
        'T4: med |cos(PC1, w_t)| = 0.851 - '
        'the per-target w_t directions are '
        'themselves collinear with PC1 (the '
        'whole preimage family is one axis + '
        'small per-target residuals). (6) T5 '
        'write-side asymmetry: d_tan family is '
        'HIGH-dimensional (d_eff = 8.74) and '
        'its PC1 is a DIFFERENT axis '
        '(|cos| = 0.142 vs readout PC1) - the '
        'write side is many-to-one into the '
        'single readout pipe.')
    meas = {
        'meas_id': 'meas3060_omega_p57_pc1_'
                   'identity_qwen',
        'phase': 3060,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a198 source seals 3044-3059; '
                   'a199 TT diff 0.0; a200 gamma '
                   'stats diff 0.0; a201 TOP64/'
                   'FLAT64 diff 0.0 vs z58; a202 '
                   'Vmat diff 0.0 vs z59; a203 '
                   'SV/d_eff diff 0.0 vs z59; '
                   'a204 D_TAN diff 0.0 vs z59',
        'artifacts': {
            'result': 'phase3060/omega_p57_'
                      'pc1_identity_qwen/'
                      'result.json',
            'npz': 'phase3060/omega_p57_'
                   'pc1_identity_qwen/'
                   'omega_p57_pc1_identity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (21.6s, fp32 '
                'weights-only, zero forwards)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 199
    l14['connects'].append({
        'meas_id': 'meas3060_omega_p57_pc1_'
                   'identity_qwen',
        'phase': 3060,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P57: PC1 identity - '
                        'the main readout axis is a '
                        'perfectly shared cross-body '
                        'axis (per-body |cos| = '
                        '0.998-1.000, LOO >= '
                        '0.99999, 99.8 pct variance) '
                        'living in the S16 piped '
                        'support (E16 = 0.519 vs '
                        'null 0.002), with NO '
                        'semantic identity (votes '
                        'rho 0.030, sig p 0.24); '
                        'write-side d_tan is high-'
                        'dimensional (d_eff 8.74) '
                        'with a different PC1 '
                        '(cos 0.14) - many-to-one '
                        'write into a single '
                        'task-agnostic readout pipe '
                        '(pc1_shared_piped_qwen)'})
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
if '## Phase 3060:' not in memo:
    sec = u'''## Phase 3060: Ω-P57 主读出轴 PC1 身份——跨体共享单轴 + 管道支撑 + 写读异轴（pc1_shared_piped_qwen） [%(created)s]

**判决：`pc1_shared_piped_qwen`**（run3 fp32 weights-only 权威 21.6s，零前向）。链锚：a198 封印 3044-3059 / a199 TT bit 0.0 / a200 γ 统计 bit 0.0 / a201 TOP64 重算 vs z58 bit 0.0 / a202 Vmat 重算 vs z59 bit 0.0 / a203 SV/d_eff bit 0.0 / a204 D_TAN bit 0.0。两次非权威如实入册：run1 崩溃于 T4 per-body 矩乘形状（wud.T @ TT[rows] 未转置）；run2 测量全部完成但 (a) result.json 序列化崩溃（np.int64）且 (b) 判据未取绝对值——SVD 符号任意，per-body cos −0.998/−1.0 实为 |cos|=1 完美共享轴，符号脆弱的 fragmented 判决无效，run3 判据改用 |cos|。

### 设计
**T2 共享分解（离线）**：v̄=mean(Vmat)、cos(PC1,v̄)、逐 body（8 组×3 条件）子族 d_eff_b + |cos(PC1_b,PC1)| + leave-one-body-out；**T3 通道身份（主检验，离线）**：PC1 在 S16/S64 能量 vs 16/64-of-2560 置换 null（seed 9983/9984）+ top-256 |PC1| 通道与 S16/S64 重叠置换 + 2802 投票谱 Spearman/sig 富集（seed 9985）；**T4 token 解码**：logits_dir=W_U·PC1 top-15 + cos(PC1,w_t) 谱；**T5 写入侧轴**：z59 D_TAN 族 SVD。

### 核心结果（重复三遍）
**① PC1 是完美的跨体共享轴**：逐 body 子族 PC1 |cos|=**0.998–1.000**（八体全部）、d_eff_b=**1.05–1.14**（每个 body 单独就近一维）、leave-one-body-out |cos|≥**0.99999**；PC1 承载 **99.8pct** 家族方差且 |cos(PC1, v̄)|=0.9997——**共享轴就是平均预像本身**，八个 body 的读出方向族在通道空间里几乎逐位相同。**② PC1 住在 S16/S64 管道支撑里**：E16=**0.519** vs null 0.002（**p=0.0005**）、E64=0.610 vs 0.011（p=0.0005）；top-256 |PC1| 通道与 S16 重叠 **11 vs null 1**、与 TOP64 重叠 **31 vs null 6**（双双 p=0.0005）——主轴是管道轴。**③ 无语义身份**：Spearman(|PC1|,|votes2802|)=0.030、top-256 sig 富集 86 vs 期望 80.5（p=0.24）；W_U·PC1 top token 全为标点/格式/杂字形——主轴不编码语义内容。**④ T4 |cos(PC1, w_t)|=0.851**：每个目标的 w_t 方向本身与 PC1 近共线——整个预像族 = 单轴 + 小的逐目标残差。**⑤ T5 写读异轴不对称**：写入侧 d_tan 族**高维**（d_eff=**8.74**）且其 PC1 与读出 PC1 不同轴（|cos|=**0.142**）——写入是 many-to-one 进单一读出管道。

### 机制链定版
读出管道的最终身份：**单条跨体共享、任务无关、住在 S16 通道支撑里的传输轴**——八体三前缀的全部行为读出共享它（99.8pct 方差），语义不在轴上（无投票/无内容词），而在写入侧与语境（3053/3055）。写入侧高维（d_eff 8.74）与读出侧一维（1.34）的不对称是"有限参数→无限语言"的结构基础之一：**多变的写入方向收敛进固定的读出管道，管道只负责传输与 γ 反方差放大**。

### 方法论入册
- **SVD 符号任意性判据纪律（3060）**：跨族/跨子集比较 PC1 一致性必须用 |cos|——符号脆弱判据可把完美共享轴误判为 fragmented（run2 教训）。
- **共享-残差分解模板（3060）**：族均值 v̄ vs PC1 的 cos + 逐子族 PC1 稳定性 + LOO——"共享轴"主张的三重验证。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3060/omega_p57_pc1_identity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3061 菜单**——A（主选）**写入侧高维分解**：d_tan 族 d_eff=8.74 的结构——逐 body/逐前缀子族维数、8.74 维里是什么（逐目标成分 vs 逐前缀成分）、与 3059 写入 S16 低浓度的调和；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ 管道+PC1 身份）；D 门区 2D 易感图。"好的，继续"即进 3061 A。
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
if '## 二十二、3060 增补' not in aud:
    add = u'''
    
---

## 二十二、3060 增补：主读出轴 PC1 身份——跨体共享单轴、管道支撑、写读异轴（Omega-P57，判决 pc1_shared_piped_qwen）

1. **PC1 是完美的跨体共享轴**：逐 body 子族 PC1 |cos|=0.998-1.000（八体全部近一维 d_eff_b=1.05-1.14），LOO |cos|>=0.99999；PC1 承载 99.8pct 家族方差，共享轴=平均预像本身。
2. **主轴住在管道里**：E16=0.519 vs null 0.002（p=0.0005），top-256 |PC1| 通道与 S16 重叠 11 vs null 1、TOP64 重叠 31 vs null 6——主轴即 3058/3059 的 S16 管道轴。
3. **主轴无语义身份**：投票谱 Spearman=0.030、sig 富集 p=0.24，top token 全为标点/格式字形——管道轴不编码语义，语义在写入侧与语境。
4. **写读异轴不对称**：写入 d_tan 族高维（d_eff=8.74）且 PC1 与读出 PC1 仅 cos 0.14——many-to-one 写入收敛进单一读出管道；"有限参数→无限语言"的结构基础：多变写入 + 固定读出基建。
5. HDMCC 修正：两图谱接口的读出端收口为"单共享轴 × S16 支撑"的任务无关传输轴；语言族→响应图谱的映射是 many-to-one 管道汇入。
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
if 'Phase 3060' not in prev:
    line = ('- Phase 3060 Omega-P57 PC1 identity: '
            'verdict pc1_shared_piped_qwen (run3 '
            'fp32 weights-only 21.6s; a198-a204 '
            'all pass; run1 T4 matmul shape crash, '
            'run2 serialization + sign-naive SVD '
            'criteria - registered). RESULTS: PC1 '
            'is a perfectly shared cross-body axis '
            '(per-body |cos| 0.998-1.000, d_eff_b '
            '1.05-1.14, LOO >= 0.99999, 99.8 pct '
            'variance, = mean preimage); lives in '
            'S16 piped support (E16 0.519 vs null '
            '0.002, p=0.0005; overlap 11 vs 1 / 31 '
            'vs 6); NO semantic identity (votes '
            'rho 0.030, sig p 0.24); write-side '
            'd_tan high-dim (d_eff 8.74) with a '
            'different PC1 (cos 0.14) - '
            'many-to-one write into a single '
            'task-agnostic readout pipe. Audit '
            'addendum 22; ledger 199/L14 167.\n')
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
3. 重跑先删旧产物；负结果/锚失败/崩溃/nan/序列化失败如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性；廓线反演 ratio≥0.9。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 锚形状核对纪律；捕获库跨相位复用（上游 npz+抽样重捕获 bit 锚+派生量逐对复现锚）。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]。
- 无替换基线走专用 forward_plain；γ 扰动协议：weight swap+restore sham+self-replacement 恒等锚。
- weights-only 相位；RMSNorm γ 可含负通道 log 前断言正性；GPU 块先 smoke-test；全词表×hidden 大矩阵分块。
- **SVD 符号任意性纪律（3060）**：跨族/跨子集 PC1 一致性必须用 |cos|——run2 符号脆弱判据把完美共享轴误判 fragmented；json 封存禁 numpy 标量。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影→γ 读出预对齐→γ 谱清点→白化几何→通道身份→子空间几何（3059）→**PC1 身份（3060：跨体共享单轴 |cos|≥0.998×S16 管道支撑×无语义身份；写入 d_tan 高维 d_eff 8.74 异轴 many-to-one）**。禁单点归因：头级 loo（3051）、内容自由门（3053）、通道 loo（3058）三重非可加签名。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ 强偏斜（max 9.75）；γ⊙W_U=反方差重加权读出基；S16=TOP64[:16] 主支撑；读出管道=单共享轴×S16。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3060）
Ω-P2（3011-3060）：3045-3048 KV 阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 门槽；3053 内容自由门；3054 径向稀释+γ 重整；3055 γ 预对齐；3056 词表 null；3057 反方差重加权；3058 head16 组合承重；3059 子空间几何（d_eff 1.34×S16）；**3060 PC1 身份=跨体共享单轴（99.8pct 方差）+S16 管道+无语义+写入高维异轴（8.74 vs 1.34）**。终局：承载=门位易感，方向=语境组装，读出=任务无关共享管道，写入 many-to-one 进管道。

## 下一步
- max=3060，下一个 3061（A 主选 **写入侧高维分解**——d_tan d_eff=8.74 的逐 body/前缀子族维数与成分；B h0 对抗溯源；C DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
