# -*- coding: utf-8 -*-
"""Phase 3044 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log ->
MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3044'
     r'\omega_p41_field_axis_injection_qwen')
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
verdict = res['final_verdict']
assert verdict == 'fieldaxis_null_qwen', verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a100_dup_prefill_bit'] == 0.0
assert an['a101_dup_all_bit'] == 0.0
assert an['a102_source_seals'] is True
assert an['a103_cross_phase_bit'] == 0.0
assert an['a103_matched'] == 8
assert an['a104_dD_bit'] == 0.0
assert an['a104_dnorm_bit'] == 0.0
assert an['a104_dmed'] <= 1e-12
assert an['a105_integrity_all'] is True
assert an['a105_n_integ_fail'] == 0
assert an['a105_probe_nontarget_max'] == 0.0
assert an['a106_sham_max'] > 0.05
t1 = res['T1_dose']
assert abs(t1['med_rho'] + 0.4) < 1e-12
assert t1['n_pairs'] == 24 and t1['k_pos'] == 6
assert abs(t1['p_binom']
           - 0.9966946244239807) < 1e-12
t1a = res['T1_antisym']
assert abs(t1a['med_cos_s1']
           - 0.5604704363105092) < 1e-12
assert abs(t1a['med_cos_s2']
           - 0.5195530991113031) < 1e-12
t2 = res['T2_specificity']
assert abs(t2['obs_ratio']
           - 0.9768007354157358) < 1e-12
assert abs(t2['null_med']
           - 1.0001567111582461) < 1e-12
assert abs(t2['p_t2'] - 0.63675) < 1e-12
assert t2['n_pairs'] == 24
t3 = res['T3_reproduction']
assert abs(t3['obs_cos']
           - 0.10132944979160571) < 1e-12
assert abs(t3['null_med']
           - 0.09165762290598176) < 1e-12
assert abs(t3['p_t3'] - 0.4) < 1e-12
assert t3['n_pairs'] == 24 and t3['n_dirs'] == 40
t5 = res['T5_layer20']
assert abs(t5['obs_ratio']
           - 3.417754282936703) < 1e-12
assert abs(t5['null_med']
           - 1.0095827419021925) < 1e-12
assert t5['p_t5'] == 0.0
assert t5['n_bodies'] == 8
fl = res['flags']
assert fl['dose_mono'] is False
assert fl['antisym_ok'] is False
assert fl['l20_weaker'] is False
assert fl['n_integ_fail'] == 0
assert abs(an['a106_sham_max']
           - 39.069483372562175) < 1e-9

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3044
           for m in led['measurements']):
    claim = (
        'Omega-P41 (plan 3044 A) - field-axis CAUSAL '
        'injection: alpha_c/ubar3 rebuilt verbatim '
        'from the 3043 npz (a104 displacement-chain '
        'anchor: D3 bit 0.0, norms bit 0.0, alpha '
        'pairwise med-cos == obs_t3 <=1e-12; a103 vs '
        '3037 npz 8/8 bit 0.0); injected +-s*g*dir '
        'into the L3 v_proj kv7 slice at the target '
        'position of the 8 base bodies (scales '
        '0.5/1/2/4, natural norms g_bc from 3043); '
        'a105 readback integrity ALL pass + non-'
        'target positions bit 0.0 (3 full-cache '
        'probes); a106 sham max ||dlg|| 39.07.  '
        'RESULTS: (T2) L3 axis efficiency 0.977x '
        'size-matched random (MC label permutation '
        'p 0.64) - NO readout privilege at the '
        'origin layer; (T3) injection reproduces '
        'the prefix OWN logit shift only at cos '
        '0.101 vs random-dir null 0.092 (p 0.40); '
        '(T1) dose NEGATIVE (med Spearman -0.40, '
        'k_pos 6/24, p 0.997) and antisymmetry '
        'BROKEN (med cos(d+,d-) +0.56 at s=1, '
        '+0.52 at s=2) - the L3 V-perturbation '
        'logit response is dominated by a sign-'
        'independent common component with '
        'attenuating magnitude, consistent with '
        'the general suppression field (3018-3019); '
        '(T5) INVERSION: the L20 shared axis ubar20 '
        'is 3.42x more efficient than size-matched '
        'random at L20 (p<5e-5, 0/20000 MC) - '
        'direction-specific causal potency at L20, '
        'the layer where the CORRELATIONAL field is '
        'weaker (0.44 vs 0.68 alignment).  '
        'CONCLUSION: fieldaxis_null_qwen - the L3 '
        'correlational field axis carries no '
        'privileged causal channel to the readout; '
        'causal field access is layer-shifted to '
        'L20.  NEXT: L20 potency anatomy (why '
        'ubar20 beats random), damping-dose curve, '
        'cross-lingual subspace, nested '
        'orthogonality, cross-model.')
    meas = {
        'meas_id': 'meas3044_omega_p41_field_axis_'
                   'injection_qwen',
        'phase': 3044,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a100/a101/a103/a104 bit 0.0; '
                   'a102 seals; a105 integrity all '
                   '+ probe 0.0; a106 sham 39.07; '
                   'T2 ratio 0.977 p 0.64; T3 cos '
                   '0.101 vs 0.092 p 0.40; T1 med '
                   'rho -0.40 (6/24); antisym +0.56; '
                   'T5 L20 ratio 3.42 p<5e-5',
        'artifacts': {
            'result': 'phase3044/omega_p41_field_'
                      'axis_injection_qwen/'
                      'result.json',
            'npz': 'phase3044/omega_p41_field_'
                   'axis_injection_qwen/'
                   'omega_p41_field_axis_injection_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run2 authoritative (54.8s); run1 '
                'T3 cosine-definition bug (missing '
                '||dlg_pref|| denominator, obs 26.97>1 '
                'exposed it) -> corrected to the '
                'preregistered cosine and fully '
                'rerun; T1/T2/T5 norm-based and '
                'reproduced bit-identically under '
                'frozen seeds; correction registered '
                'in PREREG',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 183
    l14['connects'].append({
        'meas_id': 'meas3044_omega_p41_field_axis_'
                   'injection_qwen',
        'phase': 3044,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P41: field-axis '
                        'causal injection - L3 field '
                        'axis causally INERT at the '
                        'readout (efficiency 0.977x '
                        'random p 0.64; prefix-logit '
                        'reproduction cos 0.101 vs '
                        '0.092 p 0.40; dose NEGATIVE '
                        'med rho -0.40; antisymmetry '
                        'BROKEN +0.56 common-mode) '
                        'while the L20 shared axis is '
                        '3.42x efficient (p<5e-5) - '
                        'causal field access is '
                        'layer-shifted; suppression-'
                        'field phenomenology at L3; '
                        'fieldaxis_null_qwen'})
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
if '## Phase 3044:' not in memo:
    sec = u'''## Phase 3044: Ω-P41 场轴因果注入——L3 场轴读出惰性（效率 0.98×、剂量负、反对称破）；L20 共享轴反转 3.4×（p&lt;5e-5） [%(created)s]

**判决：`fieldaxis_null_qwen`**（run2 权威 54.8s；run1 T3 余弦定义 bug——缺 ||dlg_pref|| 分母，obs=26.97&gt;1 自暴露——登记 correction 修正后全量重跑；T1/T2/T5 范数量在冻结种子下逐位复现；七锚全过）

### 设计（3044 A 主选：场轴因果注入）
α_c/ū3 从 3043 npz 逐字重建（**a104 位移链锚**：D3 位级 0.0、norms 位级 0.0、α 两两 med|cos|==obs_t3 ≤1e-12；a103 vs 3037 npz 8/8）。在 8 个基体句目标位的 **L3 v_proj 输出 kv7 切片**注入 ±s·g_bc·dir（s∈&#123;0.5,1,2,4&#125;，g_bc=3043 自然位移范数），共约 1450 次注入 prefill。**a105 完整性**：全部 readback cos&gt;0.9+ratio∈[0.5,1.5] 通过 + 3 次全缓存探针非目标位**位级 0.0**；**a106 sham**：|s|≥2 时 max||Δlg||=**39.07**——注入确实进链。

### 核心结果（重复三遍）
**① L3 场轴读出惰性**：轴效率 0.977× 随机方向（配对标签置换 MC p=**0.64**）；对前缀自身 logit 位移的复现 cos 仅 **0.101** vs 随机 0.092（p=**0.40**）。**② 剂量为负+反对称破缺**：med Spearman ρ(||Δlg||,s)=**−0.40**（k_pos 6/24，p=0.997）；med cos(Δlg(+s),Δlg(−s))=**+0.56**（s=1）/+0.52（s=2）——L3 V 扰动的 logit 响应由**符号无关公共分量**主导且幅度随剂量衰减，与 3018-3019 通用抑制场一致。**③ L20 反转**：L20 共享轴 ū20 效率 **3.42×** 随机（p&lt;**5e-5**，0/20000 MC）——**方向特异的因果通道在 L20，恰是相关场更弱的层**（对齐 0.44 vs L3 0.68）。

### 机制链更新
前缀场分裂为两层角色：**L3 相关场轴（无因果读出特权，受抑制场衰减）+ L20 因果场轴（方向特异高效）**——"场从哪里可测"与"场从哪里起作用"是不同层。HDMCC"全局引力场"的因果检验：注入不沿相关轴复现前缀效应；场的作用通道需按层定位。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3044/omega_p41_field_axis_injection_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3045 菜单**——A（主选）**L20 效率解剖**：ū20 为何 3.4× 于随机——对 lm_head/读出子空间对齐 vs 层深衰减差；剂量曲线与符号对称性在 L20 重测；B 跨语言共享子空间（EN/ZH 同词深层对齐）；C 嵌套子空间正交性量化（apple/fruit/food）；D 跨模型复刻（五分量+场+注入协议上 DS7B/GLM4）。
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
if '## 六、3044 增补' not in aud:
    add = u'''
    
---

## 六、3044 增补：场轴因果注入结果（Omega-P41，判决 fieldaxis_null_qwen）

对"全局引力场"的因果检验（run2 权威，七锚全过，a104 位移链锚 + a105 注入完整性 + a106 sham 39.07）：

1. **L3 场轴读出惰性**：沿相关场轴注入与等范数随机方向在下游 logits 上无差别（效率 0.977×，p=0.64）；对前缀自身 logit 位移复现 cos 0.101 vs 0.092（p=0.40）——相关结构不等于因果通道。
2. **抑制场签名**：剂量为负（med ρ=−0.40）、反对称破缺（cos(d+,d−)=+0.56 公共分量主导）——L3 V 扰动响应被符号无关地衰减，与 3018-3019 通用抑制场一致。
3. **L20 反转**：ū20 效率 3.42×（p&lt;5e-5）——方向特异因果通道在 L20；"场的可测层"（L3，对齐 0.68）≠"场的作用层"（L20）。
4. 对附件叙事的裁定升级：风格=全局引力场 **降级为"L20 层的因果偏置"**；引力场隐喻需按层拆分（相关投影 vs 因果读出通道）。

据此 3045 菜单 A 定为 L20 效率解剖（ū20 为何高效：读出对齐 vs 层深衰减）。
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
if 'Phase 3044' not in prev:
    line = ('- Phase 3044 Omega-P41 field-axis causal '
            'injection: verdict fieldaxis_null_qwen '
            '(run2 54.8s; run1 T3 cosine-definition bug '
            'obs 26.97>1 -> correction + full rerun; 7 '
            'anchors ok, a104 displacement chain bit '
            '0.0, a105 integrity all, a106 sham 39.07); '
            'L3 axis causally inert (eff 0.977x p 0.64; '
            'repro cos 0.101 vs 0.092 p 0.40; dose '
            'NEGATIVE med rho -0.40; antisym broken '
            '+0.56); L20 INVERSION ubar20 3.42x '
            'efficient (p<5e-5) - causal access is '
            'layer-shifted; audit addendum 6; ledger '
            '183/L14 151.\n')
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
1. 闭环：execution 冻结（PREREG/锚/判决）→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）。
3. 重跑先删旧 execution/result/npz；负结果与判据作废如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格；裸百分号写 %%；log 占位符数=实参数（3042）；置换格保证存在（3043）；锚引用 npz 键名先探针（3043）；**统计量先量纲自检（cos&gt;1 即 bug，3044 run1）**。

## 标准锚与精度
- a1 dirs 重建 2.17e-08；bit 级仅限同文件链/上游全精度；跨相位 npz 锚均 0.0；剂量两端点锚死。
- 干预锚族：重复基链/门开/重复臂位级 0.0+比率门；手工 norm+lm_head 0.15 门；源封印；跨相位链锚（3043 a98 位移 D3/D20；**3044 a104 位移链+轴重建 med-cos==obs_t3 ≤1e-12**）；注入 readback 完整性门 cos&gt;0.9+ratio[0.5,1.5]（bf16 容差）+非目标位 0.0（3044 a105）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；margin n≳40（探索性标注）。
- null 门不设在接收干预的量上；曲率排除饱和区；跨相位锚前核对量纲；守卫零→退化行剔除；小 n 组内去均值→构造匹配置换；几何标签命名前查谱；方差分解边际 SS+标签置换；**注入实验：随机方向等范数对照+配对标签置换 MC（3044）**。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→KV 三分量→读出协议条件性→残差防居中→谱水平复核→位移场双 null+通道泄漏→方差分解主效应→**相关可测层≠因果作用层（3044）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；bf16 hook 配 bf16 列；pre-hook with_kwargs 返回 (args,kwargs)；SVD 行空间基底 Vt[:r].T；**KV 注入=v_proj 输出切片 [kv*128:(kv+1)*128] 前向 hook**（3044）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3044）
Ω-P2（3011-3044）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/曲率操作点/非正交；3037 KV 三分量；3038/3039 协议分层；3040 情景分量 98.3pct+刻板；3041 谱推翻秩1（PR 1.94）；3042 风格场 4.5× 全局共享；3043 场方差分解（体 64pct/前缀 15pct、轴共性 0.647、库外迁移 0.683）；3044 **场轴因果注入：L3 轴读出惰性（0.977×、剂量负、反对称破 +0.56 公共分量）+ L20 反转 3.42×（p&lt;5e-5）→ 因果作用层=L20，非相关最强层 L3**。HDMCC 审计：三定律兼容；正交嵌套 ❌；风格场降级为 L20 因果偏置（审计六节）。

## 下一步
- max=3044，下一个 3045（A 主选 **L20 效率解剖**——ū20 为何 3.4×：对读出子空间对齐 vs 层深衰减差，剂量/对称性 L20 重测；B 跨语言共享子空间；C 嵌套子空间正交性；D 跨模型复刻）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
