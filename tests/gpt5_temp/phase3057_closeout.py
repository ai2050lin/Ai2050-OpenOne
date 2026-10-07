# -*- coding: utf-8 -*-
"""Phase 3057 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3057'
     r'\omega_p54_whiten_geometry_qwen')
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
assert verdict == 'gamma_whiten_sufficient_qwen', \
    verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a181_seals_ok'] is True
assert an['a177_chain_diff'] == 0.0
assert an['a178_replay_ok'] is True
assert an['a179_gamma_stats_diff'] == 0.0
assert an['a180_tt_diff'] == 0.0
assert an['a182_construction_ok'] is True
assert an['a183_ones_diff'] < 0.01
t2 = st['T2_spectrum']
assert abs(t2['cond_W_U']
           - 57.926188578870025) < 1e-9
assert abs(t2['eff_rank_W_U']
           - 2285.5161867371116) < 1e-6
assert abs(t2['colstd_spread_before']
           - 0.03743024408492013) < 1e-12
assert abs(t2['colstd_spread_after']
           - 0.1183644316141487) < 1e-12
t3 = st['T3_inverse_fit']
assert abs(t3['alpha']
           - 3.307581577305926) < 1e-9
assert abs(t3['R2']
           - 0.41577579929439523) < 1e-12
assert t3['n_nonpositive_gamma'] == 1
assert t3['n_fit_channels'] == 2559
t5 = st['T5_gamma_fit_replay']
assert abs(t5['med_cos_fit']
           - 0.12831293039375533) < 1e-12
assert abs(t5['z55_med_gamma']
           - 0.11143683246599181) < 1e-12
assert abs(t5['med_cos_mean_only']
           - 0.027109528168621805) < 1e-12
assert abs(t5['fit_replay_ratio']
           - 1.1514409334356641) < 1e-12
t4 = st['T4_shape_ladder']
assert abs(t4['med_cos_orig']
           - 0.6804954382830863) < 1e-12
assert abs(t4['med_cos_k0']
           - 0.1899327515084473) < 1e-9
assert abs(t4['r_64']
           - 0.98083461686743) < 1e-9
assert abs(t4['r_256']
           - 0.9975145981646596) < 1e-9
assert 'corrections' in res['prereg']
assert 'run4' in res['prereg']['corrections']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3057
           for m in led['measurements']):
    claim = (
        'Omega-P54 (plan 3057 A) - whitening '
        'geometry verification, run5 fp32 '
        'authoritative (188.1s; runs 1-3 '
        'crashed pre-verdict: G1/G2 on wrong '
        'device, gamma broadcast sliced by '
        'vocab rows, 2.90 GiB colstd_w '
        'allocation; run4 sealed with nan T3/T5 '
        'because gamma has one non-positive '
        'channel (min -0.0221) making log '
        'undefined - run5 amendment: fit on '
        'the 2559 positive channels). RESULTS: '
        '(1) VERDICT gamma_whiten_sufficient_'
        'qwen. (2) T4 shape ladder (forwards): '
        'keeping only the top-64 |gamma-mean| '
        'shape channels recovers r_64 = 0.9808 '
        'of the readout alignment (0.18993 -> '
        '0.6711, orig 0.6805); r_256 = 0.9975 - '
        'the causal payload concentrates in '
        '~64 channels; k = 0 arm reproduces '
        'the 3055 ones med to 1.9e-7 (cos '
        'invariant to positive scaling). (3) '
        'T5 gamma_fit replay: the fitted '
        'inverse column-std profile alpha = '
        '3.3076 (R2 = 0.416, Spearman -0.676) '
        'ALONE reproduces the 3055 '
        'direction-specific gain - med '
        'cos(d_tan, gamma_fit*w_t) = 0.1283 vs '
        '0.1114 with the learned gamma (ratio '
        '1.151, above the 0.9 threshold): the '
        'gain is a GENERIC consequence of '
        'amplifying low-variance readout '
        'channels, no token identity or target '
        'learning required. (4) T2 spectrum: '
        'gamma does NOT equalize the readout '
        'basis - column-std spread INCREASES '
        '0.0374 -> 0.1184, effective rank '
        '2285.5 -> 2201.0, condition number '
        '57.9 -> 2.13e14: inverse-variance '
        'reweighting (anti-correlated with the '
        'raw spectrum), not classical '
        'whitening; refines the 3056 '
        'whitening-filter naming.')
    meas = {
        'meas_id': 'meas3057_omega_p54_whiten_'
                   'geometry_qwen',
        'phase': 3057,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a181 source seals 3044-3056; '
                   'a177 COS_orig diff 0.0 vs z51 '
                   'COS_H[7]; a178 d_tan replay '
                   'bit-equal to z55; a179 gamma '
                   'stats bit 0.0; a180 TT diff '
                   '0.0; a182 construction ok; '
                   'a183 mean-arm vs 3055 ones '
                   'diff 1.9e-7',
        'artifacts': {
            'result': 'phase3057/omega_p54_'
                      'whiten_geometry_qwen/'
                      'result.json',
            'npz': 'phase3057/omega_p54_'
                   'whiten_geometry_qwen/'
                   'omega_p54_whiten_geometry_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run5 authoritative (188.1s, fp32); '
                'runs 1-3 pre-verdict crashes and '
                'run4 nan T3/T5 (negative gamma '
                'channel) registered in '
                'corrections; T3 fit on 2559 '
                'positive channels',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 196
    l14['connects'].append({
        'meas_id': 'meas3057_omega_p54_whiten_'
                   'geometry_qwen',
        'phase': 3057,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P54: whitening '
                        'geometry - top-64 shape '
                        'channels carry r = 0.98 of '
                        'the readout alignment; the '
                        'fitted inverse-column-std '
                        'profile (alpha 3.31) ALONE '
                        'reproduces the 3055 '
                        'direction-specific gain '
                        '(0.1283 vs 0.1114, ratio '
                        '1.15) - the gain is a '
                        'generic consequence of '
                        'amplifying low-variance '
                        'channels, not target '
                        'learning; gamma does NOT '
                        'equalize (spread 0.037 -> '
                        '0.118, cond 2.1e14): '
                        'inverse-variance '
                        'reweighter, refining the '
                        '3056 whitening naming '
                        '(gamma_whiten_sufficient_'
                        'qwen)'})
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
if '## Phase 3057:' not in memo:
    sec = u'''## Phase 3057: Ω-P54 白化几何验证——top-64 形状通道承载 + 反比廓线单独复现方向特异增益（gamma_whiten_sufficient_qwen） [%(created)s]

**判决：`gamma_whiten_sufficient_qwen`**（run5 fp32 权威 188.1s，锚核心全过）。run1-3 判决前崩溃（G1/G2 设备错配；γ 广播误按词表行切片；colstd_w 试图分配 2.90 GiB 大矩阵）——三次修正全入册；run4 完成封存但 T3/T5 产出 nan（γ 含 1 个非正通道 min −0.0221，log 无定义），判决落入 r-only 分支 partial；run5 预注册修正：拟合限于 2559 个正通道并登记排除数。

### 设计
**T2 谱定量（离线）**：W_U vs W_U⊙γ 的 Gram 谱（条件数/有效秩/colstd 离散度）；**T3 反比拟合（离线）**：log γ ~ −log colstd OLS；**T5 γ_fit 重演（离线，核心）**：γ_fit = colstd^(−α̂) 归一化，逐对 cos(d_tan, γ_fit⊙w_t) 对 z55 的 0.1114；**T4 形状 ladder（前向因果，主检验）**：γ_k 保留 top-k 个 |γ−mean| 形状通道（k∈0/64/256/1024），canonical h7 joint 臂 48 前向/臂，恢复率 r_k。

### 核心结果（重复三遍）
**① 因果载荷集中于 ~64 通道**：top-64 形状通道即恢复 **r_64=0.9808**（0.18993→0.6711，orig 0.6805），r_256=0.9975——读出预对齐 payload 挤在 64 个通道里；k=0 臂与 3055 ones 臂差 **1.9e-7**（cos 对正标度不变，逐位验证协议自洽）。**② 反比廓线单独复现方向特异增益**：α̂=**3.3076**（R²=0.416，Spearman −0.676），γ_fit = colstd^(−3.31) 的重演 med cos = **0.1283** vs 学习 γ 的 **0.1114**（ratio **1.151** ≥ 0.9 阈值）——**方向特异增益不需要 token 身份、不需要对目标的学习，是"放大低方差读出通道"的通用几何后果**。**③ γ 不是均衡白化**：colstd 离散度 0.0374→**0.1184（增大）**、有效秩 2285.5→2201.0、条件数 57.9→**2.13e14**——γ 与原始谱反相关地放大差异（inverse-variance reweighter），3056 的"whitening filter"命名修正为**反方差重加权器**。**④ mean-only 对照** = 0.0271（= 无 γ 预像对齐）——增益全部来自形状（通道分配），与 3055 闭环。

### 机制链定版
γ 读出预对齐的完整分解：**γ ≈ colstd^−3.3 型反方差重加权、因果载荷集中 top-64 通道、该廓线单独即可复现方向特异增益**。写入-读出闭环最终版：写入侧（KV 门位易感 + 语境组装）产生 d_tan，读出侧 γ 反方差重加权在低方差通道区放大其对准——"γ 定基"= 反方差定基，方向对准 = 通用几何后果 + 语境写入，二者无需互相知晓。

### 方法论入册
- **γ_fit 反演协议（3057）**：拟合 α̂ → 构造 γ_fit → 离线重演因果观测量——"统计廓线能否替代学习参数"的直接判据（ratio ≥ 0.9 阈值预注册）。
- **形状 ladder（3057）**：|γ−mean| 排序的 top-k 通道保留/替换 mean——定位因果载荷的通道集中度；k=0 臂自动兼作标度不变性 sham。
- **负值通道 log 陷阱（3057 run4）**：RMSNorm γ 可含负通道（1/2560），log 域拟合/重演前必须断言正性并登记排除数。
- **三次判决前崩溃（3057 run1-3）**：设备错配 / 广播切片错维 / 大矩阵分配——GPU 离线统计块先小 smoke-test 再并入主脚本。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3057/omega_p54_whiten_geometry_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3058 菜单**——A（主选）**top-64 形状通道身份**：这 64 个通道是谁（与 unembed 高方差通道、3022 联盟、2802 投票维的交叉比对）+ 逐通道 leave-one-in 剖面 + 低方差通道区的语义内容；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ）；D 门区 2D 易感图。"好的，继续"即进 3058 A。
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
if '## 十九、3057 增补' not in aud:
    add = u'''
    
---

## 十九、3057 增补：白化几何验证——top-64 通道承载、反比廓线充分（Omega-P54，判决 gamma_whiten_sufficient_qwen）

1. **因果载荷集中**：top-64 |γ−mean| 形状通道恢复 98.1pct 的读出对齐（r_256=99.8pct）——γ 预对齐是稀疏通道现象，非弥散重加权。
2. **反比廓线充分**：colstd^(−3.31) 拟合廓线单独复现方向特异增益（0.1283 vs 学习 γ 0.1114，ratio 1.15）——增益是"放大低方差通道"的通用几何后果，无需 token 身份或目标学习。
3. **命名修正**：γ 不均衡列方差（spread 0.037→0.118、cond 2.1e14）——3056 的"whitening filter"应更名"反方差重加权器（inverse-variance reweighter）"。
4. HDMCC 修正：读出端几何 = 稀疏反方差重加权（~64 通道）× 语境写入对准——两图谱接口的读出端是通用频谱整形 + 稀疏通道载荷，方向信息全部由写入侧供给。
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
if 'Phase 3057' not in prev:
    line = ('- Phase 3057 Omega-P54 whitening '
            'geometry: verdict gamma_whiten_'
            'sufficient_qwen (run5 fp32 188.1s; '
            'runs 1-3 pre-verdict crashes - device '
            'mismatch, gamma broadcast slice, '
            '2.9GB alloc; run4 sealed with nan T3/'
            'T5 from one negative gamma channel, '
            'amended in run5). RESULTS: top-64 '
            'shape channels carry r=0.98 of the '
            'readout alignment; fitted colstd^-'
            '3.31 profile ALONE reproduces the '
            'direction-specific gain (0.1283 vs '
            '0.1114, ratio 1.15) - generic '
            'low-variance amplification, no '
            'target learning; gamma does NOT '
            'equalize (spread 0.037->0.118, cond '
            '2.1e14) = inverse-variance '
            'reweighter. Audit addendum 19; '
            'ledger 196/L14 164.\n')
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
3. 重跑先删旧产物；负结果/锚失败/崩溃/nan 如实登记；verdict 单分支赋值；**nan 产出=该臂无效，判决只能落不依赖 nan 的分支，修正须预注册 amendment 后重跑**。
4. 统计纪律：obs/null 同量纲同范围；负对照带符号解读；近似质量判据阈值预注册；随机方向对照判特异性；廓线反演判据（γ_fit 重演 ratio≥0.9）。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 锚形状核对纪律；捕获库跨相位复用（上游 npz+抽样重捕获 bit 锚+派生量逐对复现锚）。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]。
- 无替换基线走专用 forward_plain；γ 扰动协议：weight swap+restore sham+self-replacement 恒等锚。
- weights-only 相位：零前向离线设计，锚全离线 bit 复核。
- **RMSNorm γ 可含负通道（3057 run4：1/2560，min −0.022）——log 域拟合前断言正性并登记排除数**。
- **GPU 离线统计块先 smoke-test（3057 run1-3：设备错配/广播切片错维/2.9GB 分配三连崩）**。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影→γ 读出预对齐→γ 谱清点（词表 null）→**白化几何（3057：top-64 形状通道 r=0.98；colstd^−3.31 廓线单独复现增益 ratio 1.15；γ=反方差重加权器非均衡白化，spread 增大 cond 2.1e14）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ=2560 维强偏斜（max 9.75）；γ⊙W_U=反方差重加权读出基；colstd_w=|γ|⊙colstd 免大矩阵。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3057）
Ω-P2（3011-3057）：3045-3048 KV 五级阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 h7 通用门槽；3053 内容自由门；3054 径向稀释+γ 重整；3055 γ 读出预对齐；3056 词表 null；**3057 白化几何=top-64 通道承载+反比廓线充分（gamma_whiten_sufficient_qwen）**。终局：承载=门位易感，方向=语境组装，读出=稀疏反方差重加权（通用几何后果）。

## 下一步
- max=3057，下一个 3058（A 主选 **top-64 形状通道身份**——与 unembed 高方差通道/3022 联盟/2802 投票维交叉比对+逐通道 leave-one-in；B h0 对抗成分溯源；C DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
