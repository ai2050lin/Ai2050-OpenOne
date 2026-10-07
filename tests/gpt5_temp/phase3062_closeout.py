# -*- coding: utf-8 -*-
"""Phase 3062 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3062'
     r'\omega_p59_body_identity_decode_qwen')
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
assert verdict == 'body_identity_opaque_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a212_seals_ok'] is True
assert an['a213_tt_diff'] == 0.0
assert an['a214_gamma_stats_diff'] == 0.0
assert an['a215_top64_diff'] == 0.0
assert an['a216_dtan_diff'] == 0.0
assert an['a217_bbody_pc1_diff'] == 0.0
assert an['a218_vmat_sv_diff'] == 0.0
assert an['a219_coal_diff'] == 0.0
t2 = st['T2_vocab_decode']
assert t2['count_margin'] == 0
assert t2['count_self'] == 1
assert abs(t2['margin_obs'][7]
           - 1.03678394169898) < 1e-12
assert abs(t2['p_margin'][7]
           - 0.005994005994005994) < 1e-12
assert abs(t2['p_self'][0]
           - 0.005994005994005994) < 1e-12
assert t2['tok_n'] == [1] * 8
t3 = st['T3_write_read_coupling']
assert abs(t3['diag_mean']
           - 0.005631170764208619) < 1e-12
assert abs(t3['off_mean']
           - -0.0012568050156134998) < 1e-12
assert abs(t3['p_diag']
           - 0.27136431784107945) < 1e-12
t4 = st['T4_channel_identity']
assert t4['ov_s16'] == 10
assert abs(t4['p_s16']
           - 0.0004997501249375312) < 1e-9
assert t4['ov_top64'] == 34
assert t4['ov_coal'] == 6
assert abs(t4['p_coal']
           - 0.6221889055472264) < 1e-12
assert t4['ov_sig'] == 94
assert t4['sig2802_total'] == 805
assert abs(t4['p_sig']
           - 0.03498250874562719) < 1e-12
assert abs(t4['spearman_p_votes']
           - -0.008723222394671996) < 1e-12
assert t4['sig_ch'] == 2
t5 = st['T5_support_distinctness']
assert abs(t5['jaccard_med']
           - 0.07225248458373904) < 1e-12
assert abs(t5['p_jaccard']
           - 0.0004997501249375312) < 1e-9
assert 'corrections' in res['prereg']
assert 'run2' in res['prereg']['corrections']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3062
           for m in led['measurements']):
    claim = (
        'Omega-P59 (plan 3062 A) - identity '
        'decoding of the 8 body write '
        'directions, run3 fp32 weights-only '
        'authoritative (24.2s, zero forwards; '
        'a212 seals 3044-3061, a213 TT bit 0.0, '
        'a214 gamma stats bit 0.0, a215 TOP64 '
        'bit 0.0 vs z58, a216 D_TAN bit 0.0 vs '
        'z59, a217 B_BODY/C_PREFIX/write-PC1 '
        'bit 0.0 vs z61, a218 Vmat/SV bit 0.0 '
        'vs z59, a219 COAL_TOP64 bit 0.0 vs '
        'z58). run1 crashed pre-verdict (the '
        'd_bar/raw decode block was lost to a '
        'phantom Edit while the report section '
        'landed - NameError top8_dbar), run2 '
        'crashed at T3 (row indexing: rows are '
        'c-major with c_idx 0..2, k = c_idx*8 '
        '+ b; the loop used c in 1..3 giving '
        'k up to 31) - both registered. '
        'RESULTS: (1) VERDICT '
        'body_identity_opaque_qwen. (2) T2 '
        'vocab decode NULL: W_U @ (B_b - '
        'd_bar) top-15 tokens are junk '
        '(CJK fragments / code identifiers / '
        'rare glyphs), self-connective rank '
        'pct 0.05-0.88 (none in the top '
        'region), count_margin = 0, count_self '
        '= 1 - body write directions do NOT '
        'point at their own connective; no '
        'vocab-level content. (3) T3 write-'
        'read coupling NULL: diagonal 0.0056 '
        'vs off-diagonal -0.0013 (gap 0.0069, '
        'p = 0.27) - no preferential coupling '
        'to own readout preimages. (4) T4: '
        'identity channels DO live in the pipe '
        'support - top-256 overlap S16 = 10 '
        '(p = 0.0005), TOP64 = 34 (p = 0.0005); '
        'NOT the 3022 relay coalition (6, p = '
        '0.62), not the 2802 voting dims (94 '
        'vs 80.5 expected, p = 0.035 '
        'descriptive only; Spearman -0.009). '
        '(5) T5: per-body top-256 Jaccard med '
        '0.072 vs null 0.0 (p = 0.0005) - '
        'shared pipe support with per-body '
        'distinct sub-supports.')
    meas = {
        'meas_id': 'meas3062_omega_p59_body_'
                   'identity_decode_qwen',
        'phase': 3062,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a212 source seals 3044-3061; '
                   'a213 TT diff 0.0; a214 gamma '
                   'stats diff 0.0; a215 TOP64/'
                   'FLAT64 diff 0.0 vs z58; a216 '
                   'D_TAN diff 0.0 vs z59; a217 '
                   'B_BODY/C_PREFIX/PC1 diff 0.0 '
                   'vs z61; a218 Vmat/SV/d_eff '
                   'diff 0.0 vs z59; a219 '
                   'COAL_TOP64 diff 0.0 vs z58',
        'artifacts': {
            'result': 'phase3062/omega_p59_'
                      'body_identity_decode_'
                      'qwen/result.json',
            'npz': 'phase3062/omega_p59_'
                   'body_identity_decode_'
                   'qwen/omega_p59_body_'
                   'identity_decode_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (24.2s, fp32 '
                'weights-only, zero forwards); '
                'runs 1-2 non-authoritative, '
                'registered in corrections',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 201
    l14['connects'].append({
        'meas_id': 'meas3062_omega_p59_body_'
                   'identity_decode_qwen',
        'phase': 3062,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P59: body write-'
                        'direction identity decode '
                        '- vocab decode NULL (top-'
                        '15 junk, self-connective '
                        'rank pct 0.05-0.88, '
                        'count_margin 0 / count_'
                        'self 1) + write-read '
                        'coupling NULL (diag 0.0056 '
                        'vs off -0.0013, p 0.27) '
                        'BUT pipe support holds '
                        '(top-256 overlap S16 10 / '
                        'TOP64 34, both p 0.0005) '
                        'with per-body distinct '
                        'sub-supports (Jaccard med '
                        '0.072 vs null 0.0); not '
                        '3022 coalition (p 0.62), '
                        'not 2802 voting dims (p '
                        '0.035 descriptive) - '
                        'body identity dims are '
                        'CHANNEL FINGERPRINTS, '
                        'not vocab-readable '
                        'content '
                        '(body_identity_opaque_'
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
if '## Phase 3062:' not in memo:
    sec = u'''## Phase 3062: Ω-P59 body 写入方向身份解码——词表不透明 + 管道支撑细分（body_identity_opaque_qwen） [%(created)s]

**判决：`body_identity_opaque_qwen`**（run3 fp32 weights-only 权威 24.2s，零前向）。链锚：a212 封印 3044-3061 / a213 TT bit 0.0 / a214 γ 统计 bit 0.0 / a215 TOP64 重算 vs z58 bit 0.0 / a216 D_TAN bit 0.0 vs z59 / a217 B_BODY/C_PREFIX/写入 PC1 vs z61 bit 0.0 / a218 Vmat/SV/d_eff bit 0.0 vs z59 / a219 COAL_TOP64 重算 vs z58 bit 0.0。两次非权威如实入册：run1 幻影 Edit（d̄/raw 解码计算块未落盘而报告段落盘，NameError top8_dbar，T2 已日志 count_margin=0/count_self=1）→ Python 补丁 count==1 assert 重插；run2 T3 行索引错（D_TAN/Vmat 行为 c-major、c_idx 0..2，k=c_idx*8+b；误用 c∈1..3 使 k 上探 31）→ rows = b, 8+b, 16+b。

### 设计
**T2 词表解码（主检验，离线）**：logits_ID = W_U @ (B_b−d̄)（fp64，8×151936）逐体 top-15 + margin 旋转 null（R=1000，seed 9991；logits 对方向线性→单位方向 logits 算一次按‖B_ID‖缩放）+ SELF 检验（自连接词 ' so'...' thus' 的 logit 与 rank pct + 跨体特异性 spec）；**T3 读写耦合（主检验，离线）**：M[b1,b2]=mean_c cos(B_ID[b1], Vmat[c_idx*8+b2]) 对角占优 vs 体标签置换（seed 9992）；**T4 通道身份（主检验，离线）**：P=sqrt(mean_b B_ID_b²) top-256 与 S16/TOP64/3022 联盟 COAL_TOP64/2802 sig 重叠置换（seed 9993-9996）+ Spearman(P,|votes|)；**T5 支撑细分（离线）**：per-body top-256 Jaccard vs null（seed 9997）+ d̄/raw B_b 描述性解码。

### 核心结果（重复三遍）
**① 词表解码 null（主检验）**：8 个 body 身份方向的 top-15 全是杂段（CJK 碎片/代码标识符/生僻字形），自身连接词 rank pct **0.054–0.881** 无一进顶部；margin 显著 0/8（max p=0.006 'thus' 未过 0.005 计数门槛）、self+spec 达标 **1/8**——**body 写入方向不写向自己的连接词，词表级内容 opaque**。**② 读写耦合 null**：M 对角 **0.0056** vs 非对角 **−0.0013**（gap 0.0069，p=**0.27**）——写入身份方向与其自身读出预像无特异耦合。**③ 但通道身份成立**：top-256 恒等通道与 S16 重叠 **10**（null med 2，p=**0.0005**）、TOP64 重叠 **34**（null med 6，p=**0.0005**）——身份维住在 γ 管道支撑里；与 3022 联盟重叠 6（p=**0.62** 无关）、2802 sig 94 vs 期望 80.5（p=**0.035** 仅描述性）、Spearman **−0.009**。**④ 支撑细分**：per-body top-256 Jaccard med **0.072** vs null 0.0（p=**0.0005**）——共享管道支撑 + 各体细分支撑不同（小共享核、大体分异）。

### 机制链定版
body 身份维 = 管道支撑内的**方向指纹**：身份可分（3061 体间 |cos| 0.692、通道细分支撑各异）但**不携带词表可读语义**——不指向自身连接词、不占 2802 投票维、不属于 3022 中继联盟。写入侧语义不走词表方向：body 身份是"从哪里写"的路由签名，不是"写什么"的内容向量。读写闭环终版：**语义 = 语境 + 门位路由（3053/3055），通道承载身份指纹（3062），读出管道单轴传输（3060），γ 反方差放大（3057）**——token 解码 opaque 是"头级/符号重要性=关系属性"（3027/3060）在写入侧的再现：身份住在关系结构里，不住在单个 token 读出方向上。

### 方法论入册
- **幻影 Edit 分段落盘（3062 run1）**：多处插入分多次 Edit 时前段可能未落盘而后段落盘——插入块与引用块必须同一次编辑完成，或用 Python 补丁 count==1 assert（本机缺陷第三次确认）。
- **c-major 行索引纪律（3062 run2）**：24 行 = c_idx(0..2)*8+b；与 a206 的 prompt=b*4+c（c∈1..3）是两套索引，混用即 IndexError。
- **旋转 null 线性换算**：logits 对方向线性 → 单位方向 logits 算一次、按 ‖B_ID_b‖ 缩放比较（省 3 倍算力，数值等价）。
- **词表解码 null 双检验**：margin（方向是否"锐利指向"某 token）+ self-token（是否指向**自己的** token）+ spec 符号——防"显著但无内容"与"有内容但不特异"两类误读。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3062/omega_p59_body_identity_decode_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3063 菜单**——A（主选）**h0 对抗成分溯源**（悬置多相位的 B 项：V-only −0.34 与 h0 切向对抗同源检验）；B **跨模型 DS7B 复刻全链**（KV 阶梯+门区+γ 管道+PC1 身份+写入分解+身份解码）；C 门区 2D 易感图；D **body 指纹的下游消费定位**（3016 放大追踪法：谁在读这些通道指纹——门区 K 路由 vs 头读出）。"好的，继续"即进 3063 A。
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
if '## 二十四、3062 增补' not in aud:
    add = u'''
    
---

## 二十四、3062 增补：body 写入方向身份解码——词表不透明、通道指纹成立（Omega-P59，判决 body_identity_opaque_qwen）

1. **词表解码双 null**：8 个 body 身份方向（3061 的 SS_body=82.7pct 载体）经 W_U 投影 top-15 全为杂段，自身连接词 rank pct 0.054-0.881、margin 显著 0/8、self+spec 1/8；读写耦合对角占优 p=0.27——写入方向既不指向自身 token 也与其读出预像无特异耦合。
2. **通道指纹成立**：top-256 恒等通道与 S16 重叠 10 / TOP64 重叠 34（双 p=0.0005），per-body 支撑 Jaccard med 0.072 vs null 0.0——身份维住在管道支撑里且各体细分支撑可分。
3. **归属排除**：与 3022 中继联盟重叠 p=0.62、2802 投票维 p=0.035（描述性）/Spearman -0.009——身份指纹既非语义投票维也非 MLP 中继联盟写出通道。
4. HDMCC 修正：写入侧语义与 token 读出方向解耦——body 身份是通道空间的路由签名（"从哪里写"），词表内容不经过写入方向本身；"关系属性"结论（3027/3060）在写入侧闭合。
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
if 'Phase 3062' not in prev:
    line = ('- Phase 3062 Omega-P59 body write-'
            'direction identity decode: verdict '
            'body_identity_opaque_qwen (run3 fp32 '
            'weights-only 24.2s; a212-a219 all '
            'pass; run1 phantom-Edit NameError, '
            'run2 T3 c-major row-index IndexError, '
            'both registered). RESULTS: vocab '
            'decode NULL (top-15 junk, self rank '
            'pct 0.054-0.881, count_margin 0 / '
            'count_self 1), write-read coupling '
            'NULL (diag 0.0056 vs off -0.0013, '
            'p 0.27), BUT pipe support holds '
            '(top-256 overlap S16 10 / TOP64 34, '
            'both p 0.0005) with per-body distinct '
            'sub-supports (Jaccard 0.072 vs 0.0); '
            'not 3022 coalition (p 0.62), not '
            '2802 voting dims (p 0.035 descr). '
            'Body identity dims = channel '
            'fingerprints, not vocab content. '
            'Audit addendum 24; ledger 201/L14 '
            '169.\n')
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
- weights-only 相位；SVD 符号任意性：跨族 PC1 用 |cos|；json 封存禁 numpy 标量。
- **3061 纪律**：中心化 ANOVA 三份额之和=中心化总平方和；禁跨空间 cos；npz 键名-RHS 同改；fp64 W_U 3.1GB 用后即 del、大置换逐行生成。
- **3062 纪律**：①幻影 Edit 分段落盘——插入块与引用块同一次编辑或 Python 补丁 count==1 assert；②24 行=c_idx(0..2)*8+b（c-major），与 prompt=b*4+c 两套索引勿混；③旋转 null 线性换算：单位方向 logits 算一次按‖B_ID‖缩放；④词表解码双检验 margin+self+spec。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→末层 L35→K 场+h7/h6→门位易感→norm 投影→γ 预对齐→词表 null→反方差重加权→head16→子空间→PC1 身份→写入分解→**身份解码（3062：词表 opaque；通道指纹 p=0.0005×2；非 3022 联盟/2802 投票维）**。禁单点归因：头级 loo、通道 loo、内容自由门三重非可加签名；关系属性结论。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；final norm γ 强偏斜；γ⊙W_U=反方差重加权读出基；S16=TOP64[:16] 主支撑；读出管道=单共享轴×S16。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3062）
Ω-P2（3011-3062）：3045-3048 KV 阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052-3053 门槽+内容自由门；3054 径向稀释；3055 γ 预对齐；3056-3057 词表 null+反方差重加权；3058-3059 head16+子空间；3060 PC1=跨体共享单轴；3061 写入分解=body 82.7pct（~5 维铺开）+读写同支撑异方向；**3062 身份解码=词表 opaque（margin 0/8、self 1/8、耦合 p=0.27）+通道指纹（S16/TOP64 重叠 p=0.0005×2、Jaccard 0.072）+非 3022/2802**。终版：语义=语境+门位路由，通道承载身份指纹（路由签名），读出管道单轴传输，γ 放大。

## 下一步
- max=3062，下一个 3063（A 主选 **h0 对抗成分溯源**——V-only −0.34 与 h0 切向对抗同源检验；B DS7B 复刻全链；C 门区 2D 易感图；D body 指纹下游消费定位——3016 放大追踪法）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
