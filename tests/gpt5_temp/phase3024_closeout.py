# -*- coding: utf-8 -*-
"""Phase 3024 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3024'
     r'\omega_p2r_restore_ablation_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
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
assert verdict == 'relay_restore_load_bearing_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_content'] == 11
assert t2['med_js_erase'] == 0.003332
assert t2['med_js_restore_all'] == 0.00157
assert t2['med_js_restore_coal'] == 0.00117
assert t2['med_js_rnd'] == 0.003323
assert t2['collapse_coal_med'] == 0.6574
assert t2['collapse_all_med'] == 0.5105
assert t2['ratio_coal_med'] == 0.3426
assert t2['ratio_rnd_med'] == 0.9538
assert t2['n_plus'] == 11
assert t2['p_binom'] == 0.0005
assert t2['a28_erase_chain_diff'] == 0.0
assert t2['a30_capture_self_diff'] == 0.0
t2b = res['T2b']
assert t2b['med_rest_rel'] == 0.515
assert t2b['rnd_seed'] == 5920
t2c = res['T2c']
assert t2c['med_js_erase_content'] == 0.000118
assert t2c['med_js_coal_content'] == 0.000151
assert t2c['collapse_content'] == -0.0273
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a27_3022'] is True
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a29_3023'] is True
assert res['anchors']['a30_capture_self_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3024
           for m in led['measurements']):
    claim = (
        'Omega-P2r (plan v5 P2) - baseline-'
        'RESTORATION ablation, the principled fix '
        'of the 3023 zero-ablation toxicity: in '
        'the g7-K erase chain, PATCH the selected '
        'neurons down_proj output contribution to '
        'the NO-ERASE chain value (output - '
        'h_idx @ W_idx.T + h_base_idx @ W_idx.T; '
        'h_base/out_base captured from the no-'
        'erase step-2 chain) - baseline function '
        'preserved, only the ERASE-INDUCED relay '
        'change reverted.  Arms per logic tag = '
        'ERASE / RESTORE_ALL / RESTORE_COAL / '
        'RESTORE_RND(12, seed 5920); content tags '
        'ERASE_C / RESTORE_COAL_C.  Verdict '
        'relay_restore_load_bearing_qwen (frozen '
        'map: collapse_coal_med 0.6574 >= 0.5 AND '
        'p_binom 0.0005 <= 0.05 AND ratio_coal_med '
        '0.3426 < ratio_rnd_med 0.9538).  '
        'RESULTS: (i) restoring ONLY the 32-'
        'neuron coalition collapses the erase JS '
        'by 65.7 pct (11/11 tags positive, exact '
        'binomial p 0.0005; med js 0.003332 -> '
        '0.00117) while random-32 restoration is '
        'inert (0.954) - the 3022 relay '
        'coalition IS causally load-bearing once '
        'measured with the baseline preserved; '
        '3023 verdict reinterpreted: its failure '
        'was MEASUREMENT TOXICITY (zero-ablation '
        'removes the 55.5 pct baseline function), '
        'not non-causality; (ii) RESTORE_ALL '
        '(full L3 MLP output patched to no-erase '
        'values) collapses LESS (med 0.5105) '
        'than the coalition-only restore (0.6574) '
        '- the non-coalition erase-induced relay '
        'change partially REBALANCES (works '
        'against the coalition change), direct '
        'evidence of competitive rebalancing '
        'inside L3; (iii) specificity clean: '
        'content collapse -0.0273 (vs -28.27 '
        'under 3023 zero-ablation) - the restore '
        'is logic-specific and toxicity-free by '
        'construction; (iv) restore magnitude '
        'rel 0.515: the coalition erase-induced '
        'output change is 51.5 pct of the L3 MLP '
        'output norm - large and non-degenerate; '
        '(v) chain identity: a28 ERASE arm vs '
        'sealed 3022 js bit-level 0.0 AND a30 '
        'capture self-consistency js(pb, p0) 0.0 '
        'bit-level.  NEXT: decompose the '
        'restore_all - restore_coal gap (non-'
        'coalition rebalancing sign per neuron '
        'band), coalition baseline consumers.')
    meas = {
        'meas_id': 'meas3024_omega_p2r_restore_'
                   'ablation_qwen',
        'phase': 3024,
        'claim': claim,
        'verdict': verdict,
        'anchors': '30/30 (a0-a24 as 3022; a26 3021; '
                   'a27 3022; a28 erase chain vs 3022 '
                   'bit-level 0.0; a29 3023 integrity; '
                   'a30 capture self-consistency 0.0)',
        'artifacts': {
            'result': 'phase3024/omega_p2r_'
                      'restore_ablation_qwen/'
                      'result.json',
            'npz': 'phase3024/omega_p2r_'
                   'restore_ablation_qwen/'
                   'omega_p2r_restore_ablation_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, no crashes; '
                'patch-not-remove resolves the 3023 '
                'toxicity: coalition restoration '
                'collapses erase JS 65.7 pct '
                '(11/11, p 0.0005), random inert, '
                'content clean; RESTORE_ALL < '
                'RESTORE_COAL = in-layer rebalancing '
                'evidence.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 163
    l14['connects'].append({
        'meas_id': 'meas3024_omega_p2r_restore_'
                   'ablation_qwen',
        'phase': 3024,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2r: baseline-'
                        'restoration (patch to '
                        'no-erase values) resolves '
                        'the 3023 toxicity - the '
                        '3022 relay coalition IS '
                        'causally load-bearing '
                        '(collapse 0.657, 11/11, p '
                        '0.0005, random 0.954, '
                        'content -0.027); '
                        'RESTORE_ALL (0.511) < '
                        'RESTORE_COAL (0.657) = '
                        'non-coalition relay change '
                        'rebalances in-layer'})
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
if '## Phase 3024:' not in memo:
    sec = u'''## Phase 3024: Ω-P2r 基线恢复消融——patch 而非清零：中继联盟因果承重成立 [%(created)s]

**判决：`relay_restore_load_bearing_qwen`**（run1 权威一次通过，175.1s，无崩溃，锚 **30/30**：a28 擦除链 vs 3022 js 位级 0.0、**a30 捕获自洽 js(pb,p0) 位级 0.0**、a29 3023 完整性；correction_note 空）

### 设计（3023 机器 verbatim + patch-not-remove）
3023 零消融毒性的原理性修复：**基线恢复**——擦除链 step-2 中把选中神经元的 down_proj 输出贡献替换为**无擦除链的对应值**：`output - h_idx@W_idx.T + h_base_idx@W_idx.T`（h_base/out_base 由无擦除链 down_proj pre/forward hook 捕获）——基线功能逐位保留，只回退**擦除诱导的中继变化**，构造性无毒。每 logic tag 四臂：ERASE / RESTORE_ALL（全 MLP 输出恢复=旁路检验）/ RESTORE_COAL（联盟 32 神经元恢复=主检验）/ RESTORE_RND(12 组，seed 5920)；content 臂同 prompt 首 logic tag 联盟。

### 核心结果（重复三遍）
**① 联盟恢复使擦除 JS 塌缩 65.7pct，因果承重成立**：med js 0.003332 → 0.00117（collapse 0.6574，**11/11 tag 为正，精确二项 p=0.0005**），随机 32 恢复完全惰性（ratio 0.954）——3022 联盟在基线保留的测量下**就是因果承重的**；3023 的"归因≠因果"重新解读：败因是**测量毒性**（零消融连带移除 55.5pct 基线功能），不是非因果。**② RESTORE_ALL(0.5105) < RESTORE_COAL(0.6574)——层内重平衡直接证据**：只恢复 32 个神经元比恢复整个 MLP 输出消掉**更多** JS——非联盟神经元的擦除诱导变化部分**反向抵消**联盟变化（P6 tag 全恢复臂 collapse −0.55 最极端），"消融差分=直接+竞争重平衡"审计链在层内获得直接测量。**③ 特异性干净**：content 塌缩 −0.027（3023 零消融是 −28.27）——恢复 logic 特异且构造性无毒。**④ 幅度非退化**：恢复量 rel=0.515——联盟的擦除诱导输出变化达 MLP 输出范数的 51.5pct。

### 方法论结论
消融设计的第三次教训闭环：**零消融 → 有毒 → 恢复式消融（patch to no-erase value）**是处理"大基线功能集合"的正确工具；代价是需要无擦除链的逐位基线（本相位 a30 自洽锚证明捕获链位级可复现）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3024/omega_p2r_restore_ablation_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3025 = A（主选）**RESTORE_ALL−RESTORE_COAL 差距分解**——非联盟神经元擦除诱导变化的逐带符号分账（重平衡定位：哪些带反向、量多大）；B 联盟基线功能刻画（55pct 范数的下游消费头定位）；C 只恢复 e4-对齐分量；D L31 次峰定位。
''' % {'created': created,
           'script8': exe['script_sha256_8'],
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

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3024' not in prev:
    line = ('- Phase 3024 Omega-P2r: verdict '
            'relay_restore_load_bearing_qwen '
            '(run1 authoritative, 175.1s, anchors '
            '30/30, a28 erase chain vs 3022 and a30 '
            'capture self-consistency both bit-level '
            '0.0); baseline-RESTORATION ablation '
            '(patch coalition down_proj output to '
            'no-erase values in the erase chain) '
            'resolves the 3023 toxicity: coalition '
            'restore collapses erase JS 65.7 pct '
            '(11/11 tags, exact binomial p 0.0005; '
            'med 0.003332 -> 0.00117), random-32 '
            'restore inert (0.954), content collapse '
            '-0.027 vs -28.27 under zero-ablation; '
            'RESTORE_ALL 0.5105 < RESTORE_COAL '
            '0.6574 = non-coalition relay change '
            'REBALANCES in-layer (direct evidence); '
            'restore magnitude rel 0.515; 3022 '
            'coalition IS causally load-bearing '
            'under baseline-preserving measurement; '
            'next = restore_all - restore_coal gap '
            'decomposition (rebalancing bands); '
            'ledger 163/L14 131.\n'
            % ())
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
- 脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\phase{N}\\{arm}\\；临时 gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结（PREREG/锚/判决）→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）。
3. 重跑先删旧 execution/result/npz；负结果如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格（{key} 不替换，2971）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；链身份锚多点（‖e4‖=3.6017、p_m=.686、js 序列、js(pb,p0)=0.0，均 0.0）。

## 统计判据纪律
- 判据可达性先检（永真/不可达禁用）：置换不变 null p≡1（3021）；Jaccard null 中位=0→以 p 值为主（3022）；**消融类设计必预检 ablation-only 副作用 << 信号（毒性门），第三次复发（3023）**；**大基线功能集合禁零消融→用基线恢复（patch to no-erase value，3024 方法论闭环：零消融→有毒→恢复式）**；margin n≳40；quasi-post-hoc 标注；maxT；镜像 −dirs 必配；跨产物对照先查类目对齐。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+竞争重平衡（3024 层内直接证据：RESTORE_ALL 0.511 < RESTORE_COAL 0.657=非联盟变化反向抵消）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；真残差流=decoder-layer pre-hook；恒等门分母与噪声同尺度；npz 存 dict→0-d 对象数组，读回须 .item()（3023 run1）；hook 修改输出用返回值，active 门 prefill 期关=只 step-2 生效；权重与激活同 device；恢复捕获 h_base/out_base 用 detach().clone() 存 GPU bf16。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep；改后必编译检查。
- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接。
- numpy 标量入 json 转 int()/float()；GQA：KV 缓存 8 头。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3024）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/GLM4 86 vs qwen 63pct/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3024）：3011 门控=L3 KV；3013 情景码 LOO .397；3014 剂量非单调=K 路由；3015 K 消费=领先 g7（.390）；3016 放大分布式；3018 **抵消主导**（2.51 nats，MLP 承载）；3019 抵消带=通用抑制场；3020 **读出特异=注入特异**（L4 即时 944×→稀释 28.3×）；3021 **注入=MLP 中继 69pct**（头 31pct GQA 限 group7）；3022 **中继=正性专属稀疏联盟**（top32/9728=82pct，vs 3019 抑制场 Jaccard 0.0 反号）；3023 零消融有毒且集合特异（联盟=55.5pct MLP 范数，反升 2.47×）；3024 **基线恢复消融：联盟因果承重成立**（patch 后 collapse 0.657，11/11 p=0.0005，随机惰性，content −0.03 干净；RESTORE_ALL<COAL=层内重平衡）。门控三级=种子→L3 正性中继联盟→负性均衡场；核心：null 重编码全层分布式涌现；头级重要性=关系属性。

## 下一步
- max=3024，下一个 3025（A 主选 **RESTORE_ALL−RESTORE_COAL 差距分解**——非联盟擦除诱导变化逐带符号分账=重平衡定位；B 联盟基线功能刻画/下游消费头；C 只恢复 e4 对齐分量；D L31 次峰）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
