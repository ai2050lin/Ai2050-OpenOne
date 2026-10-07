# -*- coding: utf-8 -*-
"""Phase 3047 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3047'
     r'\omega_p44_kv_joint_replay_qwen')
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
assert verdict == 'kvjoint_null_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a120_chain_diff'] == 0.0
assert an['a121_dup_diff'] == 0.0
assert an['a122_ok'] is True
assert an['n_integ_fail'] == 0
assert an['n_past_fail'] == 0
assert abs(an['a122_diag_subform']
           - 1.9073486328125e-06) < 1e-15
assert an['a123_sham_diff'] == 0.0
assert abs(an['a124_max_dlg']
           - 99.91247856658396) < 1e-9
assert an['a125_dose_diff'] == 0.0
assert an['a125_ok'] is True
mc = st['T2_arms']['med_cos']
mf = st['T2_arms']['med_frac']
assert abs(mc['K'] + 0.13545346630219604) < 1e-12
assert abs(mc['V'] - 0.1926928143467039) < 1e-12
assert abs(mc['KV'] - 0.17509705530465736) < 1e-12
assert abs(mc['RM'] + 0.0453462434691384) < 1e-12
assert abs(mf['K'] - 0.03655892174316696) < 1e-12
assert abs(mf['V'] - 0.07481395071982919) < 1e-12
assert abs(mf['KV'] - 0.07562273754062158) < 1e-12
assert abs(mf['RM'] - 0.041289854170882656) < 1e-12
nu = st['null']
assert abs(nu['p_cos'] - 0.12437810945273632) < 1e-12
assert abs(nu['med_cos']
           - 0.07152633286653196) < 1e-12
assert abs(nu['max_cos']
           - 0.39412855615389264) < 1e-12
assert nu['R'] == 200 and nu['seed'] == 9850
d3 = st['T3_dose']
assert abs(d3['L3']['med_frac']
           - 0.0007976089701822217) < 1e-12
assert abs(d3['L20']['med_frac']
           - 0.0029855174287045936) < 1e-12
assert abs(d3['low']['med_frac']
           - 0.05874071675966269) < 1e-12
assert abs(d3['high']['med_frac']
           - 0.04768895055520644) < 1e-12
t4 = st['T4_removal']
assert abs(t4['med_rem_frac']
           - 0.041289854170882656) < 1e-12
assert abs(t4['med_rem_cos']
           - 0.0453462434691384) < 1e-12
assert abs(st['t_norms']['med']
           - 367.0807715900762) < 1e-9

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3047
           for m in led['measurements']):
    claim = (
        'Omega-P44 (plan 3047 A) - joint multi-'
        'layer KV replay, run2 fp32 authoritative. '
        'RUN1 REGISTERED: a122 integrity failed '
        '96/96 while past checks passed; probe '
        '(gpt5_temp/probe3047d) root-caused the '
        'CHECK formulation, not the injection: '
        'max|mod-orig-delta| = 1.86e-09 (fp32 '
        'subtraction rounding of (x+d)-x, rel '
        '3.5e-08) while mod == orig+delta is '
        'BIT-EXACT and the f32 cast is exact; '
        'integrity reformulated to the bit-exact '
        'mod == orig+delta check; statistics '
        'reproduced bit-level in run2. DESIGN: '
        'the exact per-layer pre-norm KV '
        'displacements dpre(c,b,l) of the target '
        'position (kv7 slice, all 36 layers, '
        'k_proj AND v_proj) captured on the '
        '48-prompt bank are injected '
        'SIMULTANEOUSLY into the base prompt '
        '(arms K / V / KV) and removed from the '
        'prefix prompt (arm RM). RESULTS: '
        '(T1) chain anchors a120 bit 0.0 (KPRE '
        'L3/L20 and DPREK3/20 vs z46), a121 '
        'duplicate capture bit 0.0, a125 dose '
        'replication bit 0.0 (L3/L20 K-only '
        'subsets equal 3046 T3b frac_e exactly); '
        'fp32 prefix logit displacement t_bc has '
        'norm med 367 (max 959). (T2 PRIMARY) '
        'joint KV replay does NOT reproduce the '
        'prefix direction: med cos KV 0.1751 vs '
        'norm-matched random-direction null med '
        '0.0715 max 0.3941, p = 0.1244; capture '
        'frac KV med 0.0756 (K 0.0366, V 0.0748); '
        'RM removal: rem_frac 0.0413, rem_cos '
        '0.0453 - subtracting ALL target-position '
        'KV displacements from the prefix prompt '
        'moves its logits by only 4.1 pct. (T3) '
        'dose subsets: L3 K-only 0.0008, L20 '
        'K-only 0.0030, layers 0-17 KV 0.0587, '
        '18-35 KV 0.0477 - no layer dominates. '
        'CONCLUSION: kvjoint_null_qwen - the '
        'target-position KV write carries '
        'NOTHING causally at any scope (single V '
        '3045, single K 3046, joint 36-layer '
        'K+V here: 7.6 pct magnitude, '
        'direction not significant, removal '
        '4.1 pct); the prefix logit effect must '
        'flow through the PREFIX-TOKEN KV '
        '(attention targets) or residual/MLP '
        'paths. NEXT: all-position joint KV '
        'replay decomposition (3048 A).')
    meas = {
        'meas_id': 'meas3047_omega_p44_kv_joint_'
                   'replay_qwen',
        'phase': 3047,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a120 bit 0.0 (KPRE+DPREK vs '
                   'z46); a121 bit 0.0; a122 bit-'
                   'exact fail 0 + past fail 0 '
                   '(subform diagnostic 1.9e-06); '
                   'a123 sham bit 0.0; a124 max '
                   'dlg 99.91; a125 dose '
                   'replication bit 0.0',
        'artifacts': {
            'result': 'phase3047/omega_p44_'
                      'kv_joint_replay_qwen/'
                      'result.json',
            'npz': 'phase3047/omega_p44_'
                   'kv_joint_replay_qwen/'
                   'omega_p44_kv_joint_replay_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run2 authoritative (331.6s, fp32); '
                'run1 statistics bit-identical, '
                'a122 check-formulation artifact '
                'probe-root-caused (e_sub 1.86e-09, '
                'e_bit 0.0); integrity discipline '
                'updated: bit-exact mod == '
                'orig+delta, never (x+d)-x',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 186
    l14['connects'].append({
        'meas_id': 'meas3047_omega_p44_kv_joint_'
                   'replay_qwen',
        'phase': 3047,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P44: joint 36-layer '
                        'KV replay upper bound - '
                        'the exact target-position '
                        'KV displacement (K+V, all '
                        'layers, pre-norm kv7) '
                        'injected jointly does NOT '
                        'reproduce the prefix logit '
                        'direction (cos 0.1751 vs '
                        'null max 0.3941, p 0.1244) '
                        'and captures only 7.6 pct '
                        'of its norm; removal arm '
                        'moves prefix logits by '
                        '4.1 pct only; with 3045 '
                        '(single V) and 3046 '
                        '(single K) the KV-carries '
                        'hypothesis is dead at '
                        'every scope - effect must '
                        'live in prefix-token KV '
                        'or residual/MLP paths; '
                        'kvjoint_null_qwen'})
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
if '## Phase 3047:' not in memo:
    sec = u'''## Phase 3047: Ω-P44 多层联合 KV 复放——目标位 KV 全层联合仍不承载前缀效应，KV 承载假设三级否证（kvjoint_null_qwen） [%(created)s]

**判决：`kvjoint_null_qwen`**（run2 fp32 权威 331.6s；run1 的 a122 完整性检查 96/96 失败→探针裁决为**检查公式伪影**而非注入缺陷：max|mod−orig−delta|=1.86e-09 是 fp32 减法 (x+d)−x 的舍入，而 mod == orig+delta **位级精确**、f32 cast 精确——完整性检查改位级形式后 run2 权威；全部统计量跨 run 冻结种子逐位复现）

### 设计
48 句库（verbatim 3042-3046）上捕获目标位**全部 36 层** k_proj/v_proj kv7 切片的 pre-norm 位移 dpre(c,b,l)，**同时**注入基座句（K / V / KV 三臂）并从前缀句**减除**（RM 臂）；null = R=200 逐层范数匹配随机方向联合注入。链锚：a120（KPRE L3/L20 + DPREK3/20 vs z46）bit 0.0、a121 复捕获 bit 0.0、a123 sham bit 0.0、**a125 剂量子集位级复现 3046 T3b**（L3 K-only frac 0.0008、L20 0.0030 与 frac_e3/frac_e20 diff=0.0）。

### 核心结果（重复三遍）
**① 方向不复现**：联合 KV 复放 med cos=**0.1751** vs 随机 null med 0.0715 / max 0.3941，**p=0.1244** 不显著——36 层 K+V 一起打，方向上与随机扰动不可区分。**② 幅度仅 7.6pct**：frac_KV med **0.0756**（K 0.0366、V 0.0748），而 fp32 干净尺度下前缀 logit 位移 med 范数 **367**（max 959）。**③ 擦除无效**：RM 臂从前缀句减去全部目标位 KV 位移，logits 仅动 **4.1pct**（rem_cos 0.0453）——前缀效应几乎完全不住在目标位的 KV 写入里。**④ 无层面主导**：dose 低半层 0.0587 / 高半层 0.0477，浅层略大但无一承载。

### 机制链定版
**KV 承载假设三级否证**：单层 V（3045 无特权）→ 单层 K（3046 仅 0.3pct）→ **全层联合 K+V（本相位：7.6pct 幅度、方向不显著、擦除 4.1pct）**。前缀 logit 效应必经**其他位置的 KV**（前缀 token 自身的 K/V——改变目标位的注意力对象）或**残流/MLP 通路**。3048 A：全位置联合 KV 复放分解（前缀位 KV vs 体位 KV）。

### 方法论入册
**完整性检查禁用 (x+d)−x==d 形式**（fp32 减法舍入必然伪失败）；位级形式 mod == orig+delta（同一 IEEE 运算确定性复现）。fp32 下前缀位移范数 ~367 远大于注入响应（~100）——效应量纲先看 t 范数再定 null 规模。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3047/omega_p44_kv_joint_replay_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3048 菜单**——A（主选）**全位置联合 KV 复放**：捕获前缀句全部位置的 K/V 位移并联合注入（kv 全头扩展），分解 前缀位 KV / 体位 KV / 目标位 KV 的捕获占比——前缀效应的最后一条 KV 通路裁决；B 阻尼场通道分解；C 跨模型 DS7B 复刻（KV 协议）；D 跨语言共享子空间。
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
if '## 九、3047 增补' not in aud:
    add = u'''
    
---

## 九、3047 增补：KV 承载假设终审——全层联合复放上界否证（Omega-P44，判决 kvjoint_null_qwen）

1. **上界检验**：把目标位全部 36 层 K+V 的精确 pre-norm 位移同时注入基座句——方向 cos 0.1751 vs 随机 null max 0.3941（p=0.124），幅度仅捕获前缀位移的 7.6pct；从前缀句减除同一位移仅动其 logits 4.1pct。
2. **"Attention 重新绑定语法路由 / 引力场扭曲指纹竞争"的 KV 通路裁决定案**：单层 V（3045）、单层 K（3046）、全层联合目标位 KV（3047）都不承载。若"语法路由"存在，其载体必是**前缀位置自身的 KV**（改变注意力对象）或残流/MLP——3048 A 全位置复放分解将给出最后裁决。
3. **方法论**：a122 事件再证"检查公式即测量"——(x+d)−x 在 fp32 下不等于 d（1.86e-9 舍入），完整性判据必须用 mod == orig+delta 位级形式。
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
if 'Phase 3047' not in prev:
    line = ('- Phase 3047 Omega-P44 joint multi-layer '
            'KV replay: verdict kvjoint_null_qwen '
            '(run2 fp32 331.6s; run1 a122 96/96 '
            'failure probe-root-caused as a CHECK-'
            'formula artifact - fp32 (x+d)-x '
            'rounding 1.86e-9 while mod == '
            'orig+delta BIT-EXACT; integrity '
            'reformulated, statistics bit-'
            'reproduce). RESULTS: joint 36-layer '
            'K+V target-position replay does not '
            'reproduce prefix direction (cos '
            '0.1751, p 0.1244 vs null max 0.3941), '
            'captures 7.6 pct of the norm (t med '
            '367 in fp32), removal arm moves '
            'prefix logits 4.1 pct only; dose: '
            'L3 0.0008 / L20 0.0030 / low-half '
            '0.0587 / high-half 0.0477 (a125 '
            'bit-level replication of 3046 T3b). '
            'KV-carries hypothesis dead at every '
            'scope (3045 V, 3046 K, 3047 joint); '
            'next 3048 A all-position KV replay '
            'decomposition; audit addendum 9; '
            'ledger 186/L14 154.\n')
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
4. **统计量纪律（3044-3047）**：obs 与 null 同量纲；统计量先量纲自检；**logit 级因果读出必须 fp32**（fp32 下前缀位移 t 范数 ~367，先看效应量纲再定 null 规模）；改函数返回值后全文件查解包。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨精度 cos 门 ≥0.999。
- **干预点纪律（3046）**：先探针 norm 管线（Qwen3 qk-norm 在 k_proj 与 RoPE 之间：pre-norm key 1.71 vs post-norm 20.16，尺度打错空间=过驱动 9× 饱和）；自然尺度/场轴定义在干预点所在空间；捕获顺序先改后录。
- **完整性检查纪律（3047）**：禁用 (x+d)−x==d（fp32 减法舍入必伪失败，1.86e-9）；位级形式 mod == orig+delta；非目标位 past KV bit 0.0。

## 统计判据纪律
- 判据可达性先检；退化行先剔；构造匹配置换；池内标签置换 MC；margin n≳40 标注探索性。

## 机制解释审计链（命名前依次检查）
…→KV 多分量→谱水平复核→场方差分解→相关可测层≠因果作用层→**KV 承载三级否证**：单层 V（3045）→单层 K（3046，qk-norm 陷阱）→全层联合目标位 KV（3047：方向 p=0.124、幅度 7.6pct、擦除 4.1pct）→前缀效应必经前缀位 KV 或残流/MLP。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv（kv7↔q28-31）；KV 注入=v/k_proj 输出切片 hook；fp32 用于 logit 级因果测量；qk-norm：k_norm 在 k_proj 后、RoPE 前。
- output_attentions 的 attn 张量须 detach().cpu()。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3047）
Ω-P2（3011-3047）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/非正交；3037 KV 多分量；3038/3039 协议分层；3040-3041 情景分量+谱复核；3042-3043 风格场（4.5×共享、体主效应 64pct、轴共性 0.647、库外迁移）；3044 作废；3045 fp32 终审 V 无特权+撤回+bf16 噪声地板；3046 qk-norm 发现、K 场真实但单层 key 复放 0.3pct；**3047 全层联合 KV 复放仍 null（方向 p=0.124、7.6pct、擦除 4.1pct）——KV 承载假设死亡**。

## 下一步
- max=3047，下一个 3048（A 主选 **全位置联合 KV 复放**——前缀位/体位/目标位 KV 捕获占比分解，KV 通路最后裁决；B 阻尼场通道分解；C 跨模型 DS7B 复刻；D 跨语言共享子空间）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
