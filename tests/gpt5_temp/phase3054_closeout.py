# -*- coding: utf-8 -*-
"""Phase 3054 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3054'
     r'\omega_p51_norm_projection_qwen')
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
assert verdict == 'norm_tangential_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a162_recapture_diff'] == 0.0
assert an['a163_jdiag_diff'] == 0.0
assert an['a164_postn_dev'] < 1e-3
assert an['a165_sham_diff'] == 0.0
assert an['a166_pre_id_max'] == 0.0
assert an['a167_stage_mlp_diff'] == 0.0
assert an['a168_stage_mlpn_diff'] == 0.0
t2 = st['T2_decomposition']
assert abs(t2['med_cos_d']
           - -0.1372528281075465) < 1e-12
assert abs(t2['med_cos_dtan']
           - 0.1663355023643394) < 1e-12
assert abs(t2['med_cos_drad']
           - -0.6061175307783215) < 1e-12
assert abs(t2['med_frac_rad']
           - 0.5836963485794313) < 1e-12
assert abs(t2['med_tan_frac']
           - 0.8116009604859302) < 1e-12
assert abs(t2['med_rms_b']
           - 15.0848130333124) < 1e-9
t3 = st['T3_first_order']
assert abs(t3['med_cos_lg']
           - 0.9999241121985769) < 1e-12
assert abs(t3['med_rel_err']
           - 0.05440491884613139) < 1e-12
assert abs(t3['med_cos_postn']
           - 0.9994824506359201) < 1e-12
t3b = st['T3b_counterfactual']
assert abs(t3b['med_cos_tan_only']
           - 0.6806543732475432) < 1e-12
assert abs(t3b['med_cos_rad_only']
           - -0.12177011274936675) < 1e-12
t4 = st['T4_h0_arm']
assert abs(t4['med_cos_K0']
           - -0.24919460797697768) < 1e-12
assert abs(t4['med_cos_d']
           - -0.25922395631057515) < 1e-12
assert abs(t4['med_cos_dtan']
           - -0.17860688947711578) < 1e-12
assert abs(t4['med_cos_dtan_only']
           - -0.24920459177363602) < 1e-12
assert abs(t4['med_frac_rad']
           - 0.13095629593163938) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3054
           for m in led['measurements']):
    claim = (
        'Omega-P51 (plan 3054 A) - gate-zone norm '
        'projection mechanism, run1 fp32 '
        'authoritative (47.9s, all anchors bit '
        '0.0 incl. per-pair stage cos vs z53 '
        'STAGE cols 2/3). RESULTS: (1) RADIAL '
        'DILUTION CONFIRMED: the full-layer diff '
        'd = post_i - post_b decomposes into a '
        'tangential part (81.2% of amplitude, '
        'W_U alignment +0.166) and a radial part '
        '(58.4% projection fraction, W_U '
        'alignment -0.606); their mixture gives '
        'the raw -0.1373 - the negative raw '
        'alignment is radial dilution of a '
        'positive tangential signal. (2) '
        'FIRST-ORDER JACOBIAN PREDICTION: pred = '
        'gamma * d_tan / rms_b (RMSNorm '
        'first-order Jacobian) predicts the '
        'actual logit diff with med cos 0.99992 '
        '(rel err 5.4%, cos_postn 0.9995) - '
        'verdict norm_tangential_qwen. (3) '
        'CORRECTION to the prereg hypothesis: '
        'the tangential part is NOT '
        'pre-aligned; the final 0.6805 alignment '
        'is MANUFACTURED by the norm in two '
        'steps - (a) radial attenuation '
        '(rad-only through the exact norm: '
        '-0.606 -> -0.122 contribution, '
        'tan-only through the exact norm: '
        '+0.6807, matching 0.6805), and (b) the '
        'learnable gamma of the final RMSNorm: '
        'since 1/rms is a scalar, the 0.166 -> '
        '0.68 tangential alignment gain is '
        'entirely gamma re-weighting in the W_U '
        'readout - final-norm gamma is a static '
        'readout pre-alignment. (4) h0 NEGATIVE '
        'ARM = TANGENTIAL-ADVERSARIAL: frac_rad '
        'only 0.131, d_tan through the exact '
        'norm still -0.2492 - the exact h0 '
        'field adversarial component lives in '
        'the tangential subspace and the norm '
        'cannot rescue it (unlike the h7 arm '
        'whose negative raw cos is radial '
        'dilution); explains why random K '
        '(+0.49) beats the exact h0 field '
        '(-0.249).')
    meas = {
        'meas_id': 'meas3054_omega_p51_norm_'
                   'projection_qwen',
        'phase': 3054,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a162 re-capture bit 0.0 vs '
                   'z48 + TT 0.0; a163 joint h7 '
                   'diag diff 0.0 vs z51 COS_H[7]; '
                   'a164 postn dev 1.0e-7; a165 '
                   'sham bit 0.0; a166 stage-pre '
                   'identity 0.0; a167/a168 '
                   'per-pair stage cos bit 0.0 vs '
                   'z53 STAGE cols 2/3',
        'artifacts': {
            'result': 'phase3054/omega_p51_'
                      'norm_projection_qwen/'
                      'result.json',
            'npz': 'phase3054/omega_p51_'
                   'norm_projection_qwen/'
                   'omega_p51_norm_projection_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (47.9s, fp32); '
                'decomposition + first-order Jacobian '
                'offline exact-norm counterfactuals; '
                'no null needed (approximation-'
                'quality criterion)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 193
    l14['connects'].append({
        'meas_id': 'meas3054_omega_p51_norm_'
                   'projection_qwen',
        'phase': 3054,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P51: norm projection '
                        'mechanism - raw -0.137 = '
                        'radial dilution (tangent '
                        '+0.17, amplitude 0.81; '
                        'radial -0.61, fraction '
                        '0.58); RMSNorm 1st-order '
                        'Jacobian gamma*P_perp/rms '
                        'predicts the logit diff '
                        'cos 0.9999; final 0.68 is '
                        'MANUFACTURED: radial '
                        'attenuation (-0.61->-0.12) '
                        '+ learnable gamma readout '
                        're-weighting (0.17->0.68, '
                        '1/rms scalar cannot change '
                        'cos) - final-norm gamma = '
                        'static readout pre-'
                        'alignment; h0 negative arm '
                        'is tangential-adversarial '
                        '(frac_rad 0.13, d_tan '
                        'through norm still -0.249) '
                        '(norm_tangential_qwen)'})
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
if '## Phase 3054:' not in memo:
    sec = u'''## Phase 3054: Ω-P51 门区 norm 投影机制——径向稀释 + γ 读出重整（norm_tangential_qwen） [%(created)s]

**判决：`norm_tangential_qwen`**（run1 fp32 权威 47.9s，锚核心一次全过、零崩溃）。链锚 vs z51（a163 COS_H[7] bit 0.0）+ vs z53（a167/a168 逐对 stage cos bit 0.0，STAGE 列 2/3）+ a162 重捕获 / a165 sham / a166 pre 恒等 / a164 postn dev 1.0e-7。

### 设计
逐对分解（canonical h7 joint 臂 = 3053 T3 臂）：d = post_i−post_b 分解 d_tan/d_rad（x̂_b 基）+ 一阶 Jacobian 预测 pred=γ⊙d_tan/rms_b + T3b 离线精确 norm 反事实（d_tan-only / d_rad-only 各自单独过 norm）+ T4 h0 精确场负臂同型分解。无 null——主判据是近似质量判据（一阶预测对实际 logit 差分的 cos）。

### 核心结果（重复三遍）
**① 径向稀释确认**：整层差分 d 分解为切向（幅度占 **81.2%%**、W_U 对齐 **+0.166**）+ 径向（投影占比 **58.4%%**、W_U 对齐 **−0.606**）——混合后即 −0.1373：**负的原始对齐是正切向信号被径向幅度调制稀释**，非切向反转。**② 一阶 Jacobian 预测成功**：pred=γ⊙d_tan/rms_b 对实际 logit 差分 **med cos=0.99992**（rel err 5.4%%，cos_postn 0.9995）→ norm_tangential。**③ 修正预注册假设——0.68 是被"制造"的**：切向本身对齐仅 0.166 而非假设的高对齐；两步制造——(a) **径向衰减**：d_rad 单独过精确 norm 贡献 −0.61→**−0.122**（d_tan-only 过 norm = **0.6807** ≈ 0.6805）；(b) **γ⊙ 静态读出重整**：1/rms 是标量不改 cos，0.166→0.68 的提升**全部来自 final norm 的可学习 γ 在 W_U 读出域的重加权**——final-norm γ 是静态读出预对齐。**④ h0 负臂 = 切向对抗**：frac_rad 仅 **0.131**、d_tan 过 norm 后仍 **−0.2492**——h0 精确场的对抗成分住在切向子空间、norm 无法拯救（对比 h7 臂负 cos_d 是径向稀释、切向本正）；这解释了随机 K（+0.49，无切向对抗）优于 h0 精确场（−0.249）。

### 机制链定版
0.43（注意写入）→ −0.137（MLP 重塑+径向稀释）→ 0.68（norm 两步：径向衰减 + γ 重整）完整解析。**读出侧的"方向成形"由 final norm γ 的静态几何承担——读出预对齐是参数的一部分，与 3027（读出=上下文属性）互补：γ 定基、语境选向**。

### 方法论入册
- **分解-预测-反事实三件套（3054）**：切向/径向分解（x̂_b 基）+ 一阶 Jacobian 预测精度判据（cos_lg≥0.99 分支）+ 离线精确 norm 反事实（d_tan-only/d_rad-only）——三者合用才能区分"保留既有对齐"与"制造对齐"。
- **近似质量判据可替代 null**：当主检验是解析近似的保真度时（预测 vs 实测同 24 对同机制），无需 MC null；判据阈值预注册（0.99/0.95）。
- 逐对 stage cos bit 锚（vs 上游 npz 列）是分解类相位的强链锚：分解建立在逐对 bit 复现之上。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3054/omega_p51_norm_projection_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3055 菜单**——A（主选）**γ 读出预对齐的来源与普适性**：γ⊙W_U 组合几何——哪些词表方向被 γ 预对齐（top 感方向谱）+ 跨层 norm γ 剖面 + γ 扰动因果检验（打乱 γ 对 t 对齐的破坏）；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链；D 门区 2D 易感图。"好的，继续"即进 3055 A。
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
if '## 十六、3054 增补' not in aud:
    add = u'''
    
---

## 十六、3054 增补：norm 投影机制——径向稀释 + γ 读出重整（Omega-P51，判决 norm_tangential_qwen）

1. **0.43→0.68 的放大被完整解析**：整层差分 = 切向（+0.17 对齐、81% 幅度）+ 径向（−0.61 对齐、58% 投影）→ 原始 −0.137 是径向稀释；RMSNorm 一阶 Jacobian（γ⊙P_⊥/rms）对实际 logit 差分预测 cos 0.9999。
2. **γ 读出预对齐（新机制）**：0.166→0.68 的切向对齐提升全部来自 final norm 可学习 γ 的 W_U 域重加权（1/rms 标量不改方向）——读出几何是训练好的静态参数，"深层 KV 写入方向 + γ 预对齐读出"构成写入-读出闭环。
3. **h0 负臂定性**：切向对抗（径向仅 13%、过 norm 仍 −0.249）——精确源场的对抗成分在切向子空间；随机 K 无此负担故更优。B 线（对抗成分溯源）已获一半答案。
4. HDMCC 修正：读出侧不应画成无结构的 lm_head——final norm γ 承担方向成形的最后一环，应画为"γ 预对齐读出基"。
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
if 'Phase 3054' not in prev:
    line = ('- Phase 3054 Omega-P51 norm projection: '
            'verdict norm_tangential_qwen (run1 fp32 '
            '47.9s, all anchors bit 0.0 vs z48/z51/'
            'z53). RESULTS: raw -0.137 = radial '
            'dilution (tangent +0.17 align, 81pct '
            'amplitude; radial -0.61, 58pct frac); '
            '1st-order Jacobian gamma*d_tan/rms '
            'predicts logit diff cos 0.9999; final '
            '0.68 is MANUFACTURED in two steps - '
            'radial attenuation (-0.61->-0.12, '
            'tan-only through exact norm 0.6807) + '
            'learnable gamma readout re-weighting '
            '(0.17->0.68, scalar 1/rms cannot '
            'change cos) - final-norm gamma = '
            'static readout pre-alignment; h0 '
            'negative arm is TANGENTIAL-adversarial '
            '(frac_rad 0.13, through norm still '
            '-0.249). Audit addendum 16; ledger '
            '193/L14 161.\n')
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
3. 重跑先删旧产物；负结果/锚失败/崩溃如实登记；verdict 单分支赋值；非权威 run 登记进 corrections+seal。
4. 统计纪律：obs/null 同量纲同范围；null 限同一行集（3050）；loo 全谱（3051）；跨对迁移 A/B+配对符号翻转（3052）；块内随机 null+跨头对照判内容自由 vs 场易感（3053）；**近似质量判据可替代 null（3054：预测-实测同对同机制，阈值预注册）**。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 锚形状核对纪律（3053 教训）：跨相位锚比较前 assert 上游数组形状。
- 捕获库跨相位复用：上游 npz+抽样重捕获 bit 锚+派生统计量逐对复现锚（3054：vs z53 STAGE 列逐对 bit）。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]；layer 返回裸 tensor。
- 头块切片（3051）：1024=8×128 头主序；块级 null 逐行 norm 匹配该头精确场。
- 分解-预测-反事实三件套（3054）：x̂_b 基切向/径向分解 + 一阶 Jacobian 预测 + 离线精确 norm 反事实，合用才能区分"保留对齐"vs"制造对齐"。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯（3045-3048）→体位均匀（3049）→末层 L35（3050）→K 场+h7/h6（3051）→h7 通用门槽（3052）→门控源：内容自由+场易感分级（3053）→**norm 投影（3054：径向稀释 −0.61/γ 读出重整 0.17→0.68；final-norm γ=静态读出预对齐；h0 负臂=切向对抗）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量；rms_eps=config.rms_norm_eps。
- stage 捕获配 a-pre 恒等 + a-postn 一致性锚；分解量逐对与上游 STAGE 列 bit 对锚。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3054）
Ω-P2（3011-3054）：3011 门控=L3 KV；3045-3048 KV 五级阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 h7 通用门槽；3053 内容自由门+场易感分级；**3054 norm 投影：径向稀释+γ 读出重整（norm_tangential_qwen）**。终局：承载=门位易感，方向=语境组装，读出成形=γ 静态预对齐。

## 下一步
- max=3054，下一个 3055（A 主选 **γ 读出预对齐来源**——γ⊙W_U 组合几何、top 感方向谱、γ 扰动因果检验；B h0 对抗成分溯源；C 跨模型 DS7B 复刻；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
