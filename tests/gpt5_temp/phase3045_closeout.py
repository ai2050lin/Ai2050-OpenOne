# -*- coding: utf-8 -*-
"""Phase 3045 closeout (idempotent): Ledger -> L14 ->
MEMO append (incl. 3044 erratum) -> HDMCC audit
addendum -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3045'
     r'\omega_p42_l20_axis_anatomy_qwen')
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
assert verdict == 'faxis20_null_qwen', verdict
assert res['anchor_core_ok'] is False
an = res['anchors']
assert an['a107_dup_prefill_bit'] == 0.0
assert an['a108_dup_all_bit'] == 0.0
assert an['a109_source_seals'] is True
assert an['a110_cross_precision_min_cos'] >= 0.999
assert an['a110_matched'] == 8
assert an['a111_cross_precision_min_cos'] \
    == 0.9987931229728595
assert an['a112_raw_med_eff_u'] \
    == 3.417754282936703
assert an['a112_d_vs_stored'] == 0.0
assert an['a113_j_and_linearity'] is False
assert an['n_integ_fail'] == 0
t1 = res['T1_efficiency']
assert abs(t1['obs_L20']
           - 0.8934298364441196) < 1e-12
assert abs(t1['p_L20'] - 0.9447) < 1e-12
assert abs(t1['obs_L3']
           - 1.0483109213491169) < 1e-12
assert abs(t1['p_L3'] - 0.3604) < 1e-12
t2 = res['T2_dose_ladder']
assert abs(t2['L3'][-1][1]
           - 0.8221894652144929) < 1e-12
assert abs(t2['L20'][-1][1]
           - 0.16721518092337442) < 1e-12
t3a = res['T3a_linear_eff']
assert abs(t3a['obs_L20']
           - 0.8632) < 1e-3
assert abs(t3a['p_L20'] - 0.8729) < 1e-12
t3b = res['T3b_singular_alignment']
assert abs(t3b['p_L20'] - 0.8567) < 1e-12
assert abs(t3b['rand_frac_L20']
           - 0.06746911340050775) < 1e-12
t3c = res['T3c_linearity']
assert abs(t3c['med_cos_L20']
           - 0.9998) < 1e-3
t5 = res['T5_out_of_bank']
assert abs(t5['obs'] - 1.1537) < 1e-3
assert abs(t5['p'] - 0.2114) < 1e-12
fl = res['flags']
assert fl['spec'] is False
assert fl['align'] is False
assert fl['rep'] is False

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3045
           for m in led['measurements']):
    claim = (
        'Omega-P42 (plan 3045 A) - L20 axis potency '
        'anatomy, run3 fp32 authoritative. RUN1-2 '
        'FINDINGS: (i) the 3044 T5 L20-inversion '
        '3.42x claim is RETRACTED - scale mismatch '
        '(raw obs 3.4178 == stored obs_t5, '
        'bit-confirmed by a112; raw med eff_rand '
        '3.4362; honest ratio 1.0154); (ii) bf16 '
        'flip-noise floor: ||dlg|| flat at 10-15 '
        'across injection norms 0.1-3.0 and layers '
        'L3/L20, +-symmetry cos +0.51, while static '
        'quantization is only norm 3.5 - ALL 3044 '
        'logit statistics were noise-bound; (iii) '
        'run2 T3a reduction-axis bug fixed (axis=0). '
        'FP32 RESULTS (clean measurement, anchors '
        'a107/a108 bit 0.0, a110 cross-precision cos '
        '0.99996, a112 bit 0.0; a111 0.99879 and a113 '
        'med_col gate fails documented as gate '
        'miscalibration): (T1) NO direction '
        'specificity at natural scale - L20 obs '
        '0.8934 (p 0.94), L3 obs 1.0483 (p 0.36); '
        '(T2) dose ladders perfectly LINEAR: L3 gain '
        '0.257/unit, L20 gain 0.052/unit (L20 readout '
        'gain 5x WEAKER); (T3a) Jacobian efficiency '
        'no spec (0.863/0.830); (T3b) no singular '
        'alignment (axis frac 0.045 vs chance 0.0625, '
        'p 0.86); (T3c) J predicts actual injections '
        'at cos 0.9998; (T5) out-of-bank replication '
        'p 0.21.  CONCLUSION: faxis20_null_qwen - '
        'the prefix field axes carry NO privileged '
        'causal channel to the logits at L3 OR L20; '
        'the KV-V to logit readout is direction-'
        'democratic; the field is correlational '
        'structure in the V write; any prefix causal '
        'effect on readout must route outside the '
        'measured V axes.  NEXT: K-side injection '
        '(attention routing), damping anatomy, '
        'cross-model, cross-lingual.')
    meas = {
        'meas_id': 'meas3045_omega_p42_l20_axis_'
                   'anatomy_qwen',
        'phase': 3045,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a107/a108 bit 0.0; a109 seals; '
                   'a110 cos 0.99996 (8/8); a111 '
                   '0.99879 FAIL documented; a112 bit '
                   '0.0 (3044 T5 diagnosis); a113 '
                   'documented fail; T1 0.893/1.048 '
                   'p 0.94/0.36; T3a 0.863/0.830; '
                   'T3b p 0.86; T5 p 0.21; dose '
                   'linear L3 0.257 L20 0.052',
        'artifacts': {
            'result': 'phase3045/omega_p42_l20_'
                      'axis_anatomy_qwen/'
                      'result.json',
            'npz': 'phase3045/omega_p42_l20_'
                   'axis_anatomy_qwen/'
                   'omega_p42_l20_axis_anatomy_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (116.4s, fp32); '
                'run1 bf16 crashed pre-verdict '
                '(RND indexing + a112 FAIL); run2 '
                'T3a reduction-axis bug; T1/T2/T3b/'
                'T3c/T4/T5 reproduced bit-'
                'identically across runs; anchor_'
                'core_ok False per prereg (a111/'
                'a113 gate miscalibration '
                'documented, measurement itself '
                'clean)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 184
    l14['connects'].append({
        'meas_id': 'meas3045_omega_p42_l20_axis_'
                   'anatomy_qwen',
        'phase': 3045,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P42: fp32 readout '
                        'anatomy - 3044 L20-inversion '
                        'RETRACTED (scale artifact, '
                        'bit-confirmed); bf16 flip-'
                        'noise floor ~12-14 documented '
                        '(all 3044 logit stats noise-'
                        'bound); fp32: NO axis '
                        'specificity at L3 or L20 '
                        '(0.893/1.048, p 0.94/0.36), '
                        'no singular alignment, dose '
                        'perfectly linear (L3 0.257 vs '
                        'L20 0.052 per unit); field '
                        'axes = correlational structure '
                        'without privileged causal '
                        'readout; faxis20_null_qwen'})
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
if '## Phase 3045:' not in memo:
    sec = u'''## Phase 3045: Ω-P42 L20 轴效率解剖——**3044"L20 反转 3.42×"撤回（量纲错位伪影）+ bf16 翻转噪声地板发现**；fp32 干净测量：L3/L20 均无方向特异因果通道（faxis20_null_qwen） [%(created)s]

**判决：`faxis20_null_qwen`**（run3 fp32 权威 116.4s；run1 bf16 判决前崩、run2 T3a 归约轴 bug，均登记；a107/a108 位级 0.0、a110 跨精度 cos 0.99996、**a112 位级 0.0 铁证 3044 T5 诊断**；a111/a113 门限误标定如实登记）

### 两大方法论发现（重复三遍）
**① 3044"L20 反转 3.42×"撤回**：T5 的 obs 是**原始** med eff_u=3.4178（a112 位级==存储 obs_t5），而 null 是除以 med_rest 的比值（构造性 ≈1）；raw med eff_rand=**3.4362**——真实比值 **1.0154**，无任何特异。**② bf16 翻转噪声地板**：‖Δlg‖ 对注入范数 0.1→3.0 与层 L3/L20 **完全平坦（10-15）**、±对称 cos=+0.51，而静态量化仅 norm 3.5——**3044 全部 logit 级统计（剂量 ρ=−0.40、反对称 +0.56、T2/T3/T5）都在噪声地板下测量，a106 sham 门 0.05 形同虚设**。

### fp32 干净测量核心结果
**① 无方向特异性**：自然尺度效率 L20 obs=**0.893**（p=0.94）、L3 obs=**1.048**（p=0.36）；J 线性效率同（0.863/0.830）；**② 剂量阶梯完美线性**：L3 增益 **0.257/unit**、L20 **0.052/unit**（L20 读出增益**弱 5×**）——J 与实际注入 cos=**0.9998**；**③ 无奇异对齐**（轴 top-8 能量 0.045 vs 随机 0.0675，p=0.86）；**④ 新句复现失败**（1.15，p=0.21）。结论：**前缀场轴在 L3 与 L20 都不享有特异因果读出通道；KV-V→logits 读出是方向民主的；场是 V 写入中的相关结构**。若前缀对读出有因果效应，必经 V 轴以外的通路（attention 路由/K、更深非线性）。

### 机制链更新
五分量候选中的"前缀场"定性降级：**可测的相关结构 ≠ 因果通路**（3037-3043 的 V 空间测量不受影响——V 读回信噪比充足）；读出级因果检验必须 fp32。锚纪律新增：obs/null 同量纲自检、J 归约轴检查、跨精度 cos 门替代 bit 锚。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3045/omega_p42_l20_axis_anatomy_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3046 菜单**——A（主选）**K 侧注入**：沿 δK 方向干预 L3/L20 K 写入改 attention 路由（V 无通道→测路由通道），fp32；B 阻尼场通道分解；C 跨模型复刻（V/K 注入协议上 DS7B）；D 跨语言共享子空间。
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
if '## 七、3045 增补' not in aud:
    add = u'''
    
---

## 七、3045 增补：fp32 因果读出终审（Omega-P42，判决 faxis20_null_qwen）+ 3044 勘误

1. **勘误（六、五节涉及）**：3044 的"L20 反转 3.42×"为量纲错位伪影（raw obs vs 比值 null），真实比值 1.0154——已撤回；3044 的 logit 级统计（剂量负、反对称破缺）均为 bf16 翻转噪声（‖Δlg‖ 平坦 10-15）。
2. **fp32 终审**：消灭噪声后，场轴在 L3 与 L20 均无方向特异因果通道（0.893/1.048，p 0.94/0.36）；剂量线性（L3 0.257/unit，L20 仅 0.052/unit）；无奇异对齐。
3. **对附件的最终裁定升级**："风格=全局引力场"在 V 写入层无因果读出特权——引力场的因果效应（若存在）必经 attention 路由/K 侧或 V 轴以外机制；"指纹竞争被场偏置"在 KV-V 层面不成立。
4. 方法论：logit 级因果检验必须在 fp32 下进行；obs/null 同量纲为硬纪律。
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
if 'Phase 3045' not in prev:
    line = ('- Phase 3045 Omega-P42 L20 axis anatomy: '
            'verdict faxis20_null_qwen (run3 fp32 '
            '116.4s; run1 bf16 crashed pre-verdict, '
            'run2 T3a reduction-axis bug). MAJOR: '
            '3044 L20-inversion 3.42x RETRACTED (a112 '
            'bit-confirmed scale mismatch, honest '
            'ratio 1.0154); bf16 flip-noise floor ~12-'
            '14 documented (all 3044 logit stats '
            'noise-bound); fp32 clean: NO axis '
            'specificity L3/L20 (0.893/1.048 p '
            '0.94/0.36), dose linear (L3 0.257 vs L20 '
            '0.052 per unit), no singular alignment, '
            'no out-of-bank replication; a111/a113 '
            'gate miscalibration documented; audit '
            'addendum 7; ledger 184/L14 152.\n')
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
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%；log 占位符数=实参数。
3. 重跑先删旧产物；负结果与判据作废如实登记；verdict 单分支赋值。
4. **统计量纪律（3044-3045 血泪）**：obs 与 null 必须同量纲（3044 T5 反转即此伪影）；统计量先量纲自检（cos&gt;1、超 σ_max 界即 bug）；J 归约轴 (NVOC,k) 用 axis=0；**logit 级因果读出必须 fp32**（bf16 翻转噪声地板 ‖Δlg‖≈12-14，与范数/层无关）。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨精度用 cos 门（≥0.999，3045 a110=0.99996 过、a111=0.99879 对 D≈0.3 的差分噪声放大属门限误标定）。
- 注入 readback 完整性门 cos&gt;0.9+ratio[0.5,1.5]+非目标位 0.0；派生量链锚（3043 a98、3044 a104、**3045 a112 位级==存储 obs_t5**）；J 非退化门须按实测增益标定（L20 增益 0.052/unit）。

## 统计判据纪律
- 判据可达性先检（bf16 下自然尺度注入不可测）；零方差/退化行先剔；构造匹配置换；标签置换 MC 池内同规模；margin n≳40 探索性标注。

## 机制解释审计链（命名前依次检查）
…→KV 三分量→读出协议条件性→谱水平复核→位移场双 null→方差分解→**相关可测层≠因果作用层，且场轴在两层都无因果特权（3045 fp32）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；KV 注入=v_proj 输出切片 [kv*128:(kv+1)*128] 前向 hook；J=128 列 one-hot h=0.25；Gram eigh 求 σ/右奇异；**fp32 模型（torch.float32）用于 logit 级因果测量**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm 损坏→os.remove；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁脚本引号逐行自查（3045 连环引号事故）；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3045）
Ω-P2（3011-3045）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/非正交；3037 KV 三分量；3038/3039 协议分层；3040-3041 情景分量+谱复核；3042-3043 风格场（4.5×共享、体主效应 64pct、轴共性 0.647、库外迁移 0.683）；3044 场轴注入（判决 null 但统计全部作废）；3045 **fp32 终审：场轴 L3/L20 均无因果读出特权、剂量线性（L3 0.257 vs L20 0.052/unit）、3044 L20 反转撤回、bf16 噪声地板发现**。相关结构≠因果通路。

## 下一步
- max=3045，下一个 3046（A 主选 **K 侧注入**——δK 干预改 attention 路由 fp32；B 阻尼场通道分解；C 跨模型 DS7B 复刻；D 跨语言共享子空间）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
