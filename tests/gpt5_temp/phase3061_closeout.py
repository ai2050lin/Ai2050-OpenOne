# -*- coding: utf-8 -*-
"""Phase 3061 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3061'
     r'\omega_p58_write_highdim_qwen')
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
assert verdict == 'write_highdim_body_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a205_seals_ok'] is True
assert an['a206_tt_diff'] == 0.0
assert an['a207_gamma_stats_diff'] == 0.0
assert an['a208_top64_diff'] == 0.0
assert an['a209_dtan_diff'] == 0.0
assert an['a210_vmat_sv_diff'] == 0.0
assert an['a211_pc1_diff'] == 0.0
t2 = st['T2_two_way_decomposition']
assert abs(t2['share_body']
           - 0.8267418375314982) < 1e-12
assert abs(t2['share_prefix']
           - 0.061338517743379464) < 1e-12
assert abs(t2['share_resid']
           - 0.1119196447251223) < 1e-12
t3 = st['T3_subfamily_spectrum']
assert abs(min(t3['d_eff_body'])
           - 1.5299360365254264) < 1e-12
assert abs(max(t3['d_eff_body'])
           - 1.8459141185795829) < 1e-12
assert abs(min(t3['d_eff_prefix'])
           - 4.826019372916492) < 1e-12
assert abs(max(t3['d_eff_prefix'])
           - 5.31250567837251) < 1e-12
assert abs(t3['between_body_mean_abs_cos_med']
           - 0.6921998699656278) < 1e-12
t4 = st['T4_readout_projection']
assert abs(t4['d_eff_resid']
           - 8.804007172838288) < 1e-12
assert abs(t4['cos_dtan_pc1_med']
           - 0.11659034423535636) < 1e-12
t5 = st['T5_channel_support']
assert abs(t5['d_eff_dtan']
           - 8.737576002501802) < 1e-12
assert abs(t5['E16_pc1d']
           - 0.18593984366776234) < 1e-12
assert abs(t5['p_E16']
           - 0.0004997501249375312) < 1e-9
assert t5['ov16'] == 13
assert t5['ov64'] == 38
assert abs(t5['cos_dtan_vmat_med']
           - 0.11143683246599181) < 1e-12
assert 'corrections' in res['prereg']
assert 'run6' in res['prereg']['corrections']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3061
           for m in led['measurements']):
    claim = (
        'Omega-P58 (plan 3061 A) - structure of '
        'the write-side high-dim family, run7 '
        'fp32 weights-only authoritative (15.4s, '
        'zero forwards; a205 seals 3044-3060, '
        'a206 TT bit 0.0, a207 gamma stats bit '
        '0.0, a208 TOP64 bit 0.0 vs z58, a209 '
        'D_TAN bit 0.0 vs z59, a210 Vmat/SV bit '
        '0.0 vs z59, a211 readout PC1 bit 0.0 '
        'vs z60 sign-invariant). run1 crashed '
        'pre-verdict (raw vs CENTERED total SS '
        'in the ANOVA identity), run2 (cross-'
        'space cosine vs vocab-space TT), run3 '
        '(verdict reached; npz key NameError), '
        'run4 (39.1 MiB argsort OOM; fixed by '
        'del wud + row-wise permutations bit-'
        'identical to batched), run5/run6 (npz '
        'key/RHS case mismatches) - all '
        'registered. RESULTS: (1) VERDICT '
        'write_highdim_body_qwen. (2) T2 two-'
        'way ANOVA of the d_tan family: SS_body '
        '= 0.8267, SS_prefix = 0.0613, SS_resid '
        '= 0.1119 - BODY identity carries 83 '
        'pct of the write-side variance; the '
        'd_eff = 8.74 high dimension is 5-dim '
        'body-mean spread + small prefix '
        'modulation. (3) T3: per-body sub-'
        'families (3 prefix rows) d_eff = 1.53-'
        '1.85 (each body alone near 1-d), per-'
        'prefix sub-families (8 body rows) '
        'd_eff = 4.83-5.31 (body directions '
        'spread over ~5 dims); between-body '
        'mean-direction |cos| med 0.692 min '
        '0.579 - distinct but non-orthogonal. '
        '(4) T4: after removing the readout '
        'PC1 the residual family d_eff = 8.80 '
        '(UNCHANGED from 8.74), |cos(d_tan, '
        'PC1_ro)| med 0.117 max 0.163 - the '
        'write family does NOT live on the '
        'readout single axis. (5) T5: write '
        'PC1 DOES live in the S16/S64 channel '
        'support (E16 0.186 vs null 0.005, '
        'E64 0.259 vs 0.021, overlaps 13 vs 2 '
        '/ 38 vs 6, all p = 0.0005) - same '
        'pipe support, different axis within '
        'it; cos(d_tan, Vmat_k) med 0.111.')
    meas = {
        'meas_id': 'meas3061_omega_p58_write_'
                   'highdim_qwen',
        'phase': 3061,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a205 source seals 3044-3060; '
                   'a206 TT diff 0.0; a207 gamma '
                   'stats diff 0.0; a208 TOP64/'
                   'FLAT64 diff 0.0 vs z58; a209 '
                   'D_TAN diff 0.0 vs z59; a210 '
                   'Vmat/SV/d_eff diff 0.0 vs '
                   'z59; a211 readout PC1 diff '
                   '0.0 vs z60 (sign-invariant)',
        'artifacts': {
            'result': 'phase3061/omega_p58_'
                      'write_highdim_qwen/'
                      'result.json',
            'npz': 'phase3061/omega_p58_'
                   'write_highdim_qwen/'
                   'omega_p58_write_highdim_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run7 authoritative (15.4s, fp32 '
                'weights-only, zero forwards); '
                'runs 1-6 non-authoritative, all '
                'registered in corrections',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 200
    l14['connects'].append({
        'meas_id': 'meas3061_omega_p58_write_'
                   'highdim_qwen',
        'phase': 3061,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P58: write-side '
                        'high-dim structure - two-way '
                        'ANOVA: body 0.827 / prefix '
                        '0.061 / resid 0.112; per-'
                        'body sub-families near 1-d '
                        '(1.53-1.85), per-prefix '
                        'sub-families ~5-dim '
                        '(4.83-5.31); write family '
                        'orthogonal to the readout '
                        'PC1 (d_eff_resid 8.80 '
                        'unchanged, |cos| med 0.117) '
                        'but shares the S16/S64 pipe '
                        'support (E16 0.186 vs null '
                        '0.005) - the many-to-one '
                        '"many" is body identity '
                        '(write_highdim_body_qwen)'})
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
if '## Phase 3061:' not in memo:
    sec = u'''## Phase 3061: Ω-P58 写入侧高维分解——body 身份承载 82.7pct + 同管道支撑异轴（write_highdim_body_qwen） [%(created)s]

**判决：`write_highdim_body_qwen`**（run7 fp32 weights-only 权威 15.4s，零前向）。链锚：a205 封印 3044-3060 / a206 TT bit 0.0 / a207 γ 统计 bit 0.0 / a208 TOP64 重算 vs z58 bit 0.0 / a209 D_TAN bit 0.0 vs z59 / a210 Vmat/SV/d_eff bit 0.0 / a211 读出 PC1 vs z60 bit 0.0（符号不变式双侧取 min）。六次非权威如实入册：run1 中心化 SS 恒等式错（分解含 grand-mean 项 24‖d̄‖²，三份额之和=中心化总平方和而非 raw total）；run2 跨空间 cos（TT[k] 是 151936 维词表空间向量，与 2560 维通道空间 d_tan 不同空间，gufunc 维度错）；run3 测量+判决完成但 npz 键名 NameError（B_B/C_C）；run4 T5 置换 null 的 (2000,2560) int64 argsort 仅 39.1 MiB 即 OOM（宿主内存极限，wud fp64 3.1GB 占用）→ del wud + 逐行置换（同 generator 流+同逐行 argsort，与批式 bit 等价、零统计偏离）；run5 只改键名未改 RHS；run6 R_KB 同型。教训入册：**npz 键名与变量大小写须逐项核对，修正时键名与 RHS 同改**。

### 设计
**T2 双因子方差分解（主结构检验，离线）**：d_kb = d̄ + (B_b−d̄) + (C_c−d̄) + R_kb，SS_body/SS_prefix/SS_resid 中心化份额；**T3 子族谱（离线）**：per-body（3 行）与 per-prefix（8 行）子族 d_eff + 体间均值方向 |cos| 矩阵；**T4 读出投影（主检验，离线）**：投影掉读出 PC1（z60）→ 残差族 d_eff；**T5 通道支撑（离线）**：D_TAN 族 SVD PC 能量在 S16/S64 vs 置换 null（seed 9986/9987）+ top-256 重叠（seed 9988）。

### 核心结果（重复三遍）
**① 写入侧高维由 body 语义身份承载**：双因子 ANOVA **SS_body=0.8267**、SS_prefix=0.0613、SS_resid=0.1119——d_eff=8.74 的高维主体是**体间身份差异**（83pct 方差），前缀调制与逐对残差各占 6pct/11pct。**② 子族谱双向不对称**：per-body 子族（3 前缀行）d_eff=**1.53–1.85**（每个 body 内部近一维），per-prefix 子族（8 体行）d_eff=**4.83–5.31**（8 个 body 方向铺开 ~5 维）；体间均值方向 |cos| med **0.692** min 0.579——body 方向互不相同但非正交。**③ 写入族与读出单轴正交**：投影掉读出 PC1 后残差族 d_eff=**8.80（几乎不变）**，|cos(d_tan, PC1_ro)| med **0.117** max 0.163——写入高维不住在读出轴上。**④ 但共享通道支撑**：写入 PC1 的 E16=**0.186** vs null 0.005、E64=0.259 vs 0.021（p=0.0005）；top-256 |PC1_d| 通道与 S16 重叠 **13 vs null 2**、与 TOP64 重叠 **38 vs null 6**（双双 p=0.0005）——**同管道支撑、管内异方向**。cos(d_tan, Vmat_k) med 0.111。

### 机制链定版
写入侧 8.74 维的完整解剖：**~5 维 body 身份（83pct）+ 6pct 前缀调制 + 11pct 逐对残差**。与 3060 合并：写入族铺在 S16 管道支撑的 ~5-9 维子空间里（many-to-one 的 "many" = body 身份维度），读出管道把其中与行为相关的分量沿单一共享轴传输。读写关系修订为**同支撑、异内容**：通道基建共享（γ/S16 是任务无关传输基建），方向内容分属两侧——写入侧语义（body 身份）+ 读出侧单轴。

### 方法论入册
- **中心化 ANOVA 恒等式纪律（3061）**：d_kb = d̄ + A_b + E_c + R_kb 的三份额之和=中心化总平方和，raw total 含 24‖d̄‖²（run1 教训）。
- **跨空间 cos 禁止（3061）**：logit 空间向量与通道空间向量的 cosine 无定义（run2 教训）。
- **npz 键名核对（3061）**：封存块逐项核对键名-变量大小写；修正时键名与 RHS 同步改（run3/5/6 教训）。
- **内存纪律**：fp64 W_U 拷贝 3.1GB 用后即 del；大置换矩阵逐行生成（同流 bit 等价）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3061/omega_p58_write_highdim_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3062 菜单**——A（主选）**body 写入方向的身份解码**：B_b 8 个方向经 W_U 投影的 top token + 与 2802 投票谱/3022 联盟的关系（body 身份维的语义内容）；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ 管道+PC1 身份+写入分解）；D 门区 2D 易感图。"好的，继续"即进 3062 A。
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
if '## 二十三、3061 增补' not in aud:
    add = u'''
    
---

## 二十三、3061 增补：写入侧高维分解——body 身份承载、同支撑异轴（Omega-P58，判决 write_highdim_body_qwen）

1. **写入侧 8.74 维的成分**：双因子 ANOVA（body × prefix）——body 身份承载 **82.7pct** 方差、前缀调制 6.1pct、逐对残差 11.2pct；per-body 子族近一维（d_eff 1.53-1.85），per-prefix 子族 ~5 维（4.83-5.31）——写入高维=8 个 body 方向在 ~5 维中的铺开。
2. **写入族与读出单轴正交**：投影掉读出 PC1 后残差 d_eff=8.80 不变、|cos| med 0.117——3060 的 many-to-one 中 "many" 就是 body 身份维度。
3. **同管道支撑、管内异方向**：写入 PC1 能量 E16=0.186 vs null 0.005（p=0.0005）、top-256 重叠 13 vs 2 / 38 vs 6——读写两侧共享 γ/S16 通道基建，方向内容分属两侧（写入=body 语义，读出=行为单轴）。
4. HDMCC 修正：语言族→响应图谱的映射管道内部有结构——管道支撑（通道基建）是公共的，管道内的轴是读出侧的，写入语义以 ~5 维 body 子空间进管道、由读出单轴选择性传输。
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
if 'Phase 3061' not in prev:
    line = ('- Phase 3061 Omega-P58 write-side '
            'high-dim structure: verdict '
            'write_highdim_body_qwen (run7 fp32 '
            'weights-only 15.4s; a205-a211 all pass; '
            'runs 1-6 registered: centered-SS '
            'identity, cross-space cosine, npz key '
            'case NameErrors x3, 39.1 MiB argsort '
            'OOM fixed by del wud + row-wise perms '
            'bit-identical). RESULTS: two-way ANOVA '
            'body 0.827 / prefix 0.061 / resid '
            '0.112; per-body sub-families d_eff '
            '1.53-1.85, per-prefix 4.83-5.31; '
            'write family orthogonal to readout '
            'PC1 (d_eff_resid 8.80, |cos| med '
            '0.117) but shares S16/S64 pipe '
            'support (E16 0.186 vs 0.005, '
            'overlaps 13 vs 2 / 38 vs 6) - '
            'many-to-one "many" = body identity; '
            'same pipe support, different axis '
            'content. Audit addendum 23; ledger '
            '200/L14 168.\n')
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
- SVD 符号任意性：跨族 PC1 一致性必须用 |cos|；json 封存禁 numpy 标量。
- **3061 新纪律**：①中心化 ANOVA——三份额之和=中心化总平方和（raw total 含 24‖d̄‖²）；②禁跨空间 cos（logit 空间 vs 通道空间无定义）；③npz 封存键名与变量大小写逐项核对，修正时键名与 RHS 同改；④fp64 W_U 拷贝 3.1GB 用后即 del，大置换矩阵逐行生成（同流 bit 等价）。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影→γ 读出预对齐→γ 谱清点→反方差重加权→通道身份→子空间几何→PC1 身份（3060）→**写入侧分解（3061：body 82.7pct×~5 维铺开；读写同 S16 支撑异方向）**。禁单点归因：头级 loo（3051）、内容自由门（3053）、通道 loo（3058）三重非可加签名。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ 强偏斜（max 9.75）；γ⊙W_U=反方差重加权读出基；S16=TOP64[:16] 主支撑；读出管道=单共享轴×S16。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3061）
Ω-P2（3011-3061）：3045-3048 KV 阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 门槽；3053 内容自由门；3054 径向稀释+γ 重整；3055 γ 预对齐；3056 词表 null；3057 反方差重加权；3058 head16 组合承重；3059 子空间几何；3060 PC1=跨体共享单轴+管道；**3061 写入分解=body 82.7pct（~5 维铺开 1.53-1.85/4.83-5.31）+读写同支撑异方向（d_eff_resid 8.80、|cos| 0.117、E16 0.186）**。终局：承载=门位易感，方向=语境组装（body 身份 ~5 维进管道），读出=任务无关单轴传输，γ=反方差放大。

## 下一步
- max=3061，下一个 3062（A 主选 **body 写入方向身份解码**——B_b 8 方向 W_U 投影 top token+与 2802 投票谱/3022 联盟关系；B h0 对抗溯源；C DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
