# -*- coding: utf-8 -*-
"""Phase 2996 closeout: ledger 135 -> L14 connect -> MEMO ->
workspace log -> MEMORY.md.  Idempotent."""
import hashlib
import json
import os
import time

LED = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
       r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\MEMORY.md')
R = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2996'
     r'\omega_f2a_registry_caliber_audit')

out = []

# ---------- ledger ----------
led = json.load(open(LED, encoding='utf-8'))
has = any(m.get('meas_id') == 'meas2996_omega_f2a'
          for m in led['measurements'])
if not has:
    led.pop('ledger_sha256_8')
    led['measurements'].append({
        'meas_id': 'meas2996_omega_f2a',
        'phase': 2996,
        'claim': (
            'Omega-F2a registry CALIBER AUDIT (both models, '
            'both calibers, matched relative-depth trio). '
            'DISCOVERY: 2995 dirs_attn/dirs_mlp were '
            'DEGENERATE - all 40 rows identical (single-slot '
            'capture hook + broadcast sweep assignment); its '
            'T1/T2 used LTOP=39 only and remain VALID by '
            'accident, but T3 per-layer z at L7/10/13 were '
            'L39-direction pseudo-replications -> VOID. '
            'Re-tested with true per-layer axes: GLM4 '
            'weight-side registry z = +30.55/+10.41/+10.15 '
            '(vs 2995 pseudo -1.75/-2.25/-1.89) -> the 2995 '
            '"registry not replicated" verdict is a capture '
            'artifact, card regraded REPLICATED. K1 (2989 '
            'act x down-col caliber): share p=0.0 both '
            'models (qwen L6/L10, glm L10/L13; maxT leg '
            'non-significant exactly as in 2989 itself, '
            'p_lang_maxT was 0.51). K2 qwen: L6 z=-21.9 '
            '(early-layer anti-alignment), L10 +5.8, L12 '
            '+22.2 - weight-side present on BOTH models, '
            'depth-profiled. artifact_q=False: no caliber '
            'mixing issue; registry_robust_across_calibers'),
        'verdict': 'registry_robust_across_calibers',
        'anchors': (
            '12/12 (qa1 d_lang_u 0.00; qa2 det 0.00; qa3/qa4 '
            'act+D vs 2989 npz 0.00 bf16-fused recompute; '
            'ga3/ga4 0.00; ga1 L39 4.4e-9; ga2 degeneracy '
            'confirmed non39 0.385; ga5 obs 2.3e-7; ga6 '
            '1.5e-15; perm reruns identical)'),
        'artifacts': {
            'result':
                'phase2996/omega_f2a_registry_caliber_audit'
                '/result.json',
            'npz':
                'phase2996/omega_f2a_registry_caliber_audit'
                '/omega_f2a_registry_caliber_audit.npz'},
        'hashes': {
            'npz_sha256_8': '327f4838',
            'result_sha256_8': 'd03a47b0',
            'script_sha256_8': '0600f54e'},
        'note': (
            '6 runs: run1 EXEC_2977 dirname; run2 Wd.T '
            'matmul shape; run3 qa1 anchor reworded '
            '2927->2979 (2927 dirs_word is a different '
            '57-word en/L pair set) + act recompute '
            'float64->bf16 (ULP tolerance); run4 split->'
            'fused matmul (cuBLAS path); run5 K1_sig '
            'AND->OR aligned to the 2989 registered rule; '
            'run6 authoritative. Tags: lang axis, len-2, '
            'snapshot only, word sets 74 vs 98 (matched '
            'caliber, not matched vocabulary')}),
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append({
        'phase': 2996,
        'via': 'omega_f2a_registry_caliber_audit',
        'adds': (
            '2995 T3 negative retracted (capture degeneracy '
            'artifact); with true per-layer axes the MLP '
            'registry replicates on GLM4 in BOTH calibers '
            '(K2 z +10..+31, K1 share p=0.0); qwen weight-'
            'side depth profile: absent/negative at L6, '
            'present from L10')})
    hc = hashlib.sha256(json.dumps(
        led, sort_keys=True,
        ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = hc
    json.dump(led, open(LED, 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    out.append('ledger appended hash=%s n=%d'
               % (hc, len(led['measurements'])))
else:
    hc = led['ledger_sha256_8']
    out.append('ledger already has meas2996 (idempotent)')

# ---------- memo ----------
stamp = '2026-09-20 11:54'
tag = '## Phase 2996:'
memo = open(MEMO, encoding='utf-8').read()
if tag + ' Ω-F2a' not in memo:
    section = """## Phase 2996: Ω-F2a 注册表口径审计——2995 T3 判决被捕获退化伪影推翻 [%(stamp)s]

**判决：`registry_robust_across_calibers`**（run6 权威，100.2s，锚 12/12，六重 bit 级 0.00）

### 原理与设计
方案 v4 机制解释审计链（口径混用嫌疑）：2995 T3 以 weight-side 口径判"MLP 注册表不复制"，而 qwen 证据（2989/2991）是 act×down-col 口径——禁止跨口径判决，故双模型（qwen3-4b 74 cells / glm4-9b 98 cells，匹配相对深度三元组 [6,10,12]/[7,10,13]）× 双口径（K1=2989 act·<Wdown[:,i],u> 标签置换 null + top-128 能量份额；K2=2995 unit(up_row)·dirs_mlp Haar null）同测。

### 重大发现：2995 捕获退化（重复三遍）
**2995 的 dirs_attn/dirs_mlp 全 40 行完全相同（max|diff|=0.0）——单槽捕获 hook + 广播 sweep 赋值把逐层轴变成最后一层（L39）的 40 份拷贝。其 T1/T2 只用 LTOP=39（恰为最后触发层）结论侥幸有效；但 T3 的"分层 z"是同一 L39 方向 ×3 的伪重复，判决作废。用真 per-layer 轴重测：GLM4 weight-side 注册表 z = +30.55/+10.41/+10.15（L7/10/13；2995 伪值 −1.75/−2.25/−1.89）——2995 T3"不复制"是捕获伪影，卡片改判 REPLICATED。**

### 核心结果
- K1（2989 口径）：share leg p=0.0 双模型在场（qwen L6/L10 share_obs 0.216/0.240；glm L10/L13 0.161/0.154）；maxT leg 与 2989 自身一致地不显著（qwen 0.81-0.91，2989 p_lang_maxT 本来就是 0.51——该卡从来不靠 maxT 承重）；
- K2（weight-side）：qwen L6 z=−21.9（早层反aligned）→L10 +5.8→L12 +22.2，GLM4 L7 +30.6——双模型双口径均在场，深度剖面不同；
- artifact_q=False：不存在口径混用问题；2995 负结果的唯一根因是捕获退化。判决 `registry_robust_across_calibers`，卡片集 v2 中 2995 T3 改判 replicated。

### 硬伤（5 笔，均删产物重跑，全部登记 correction_note）
run1 EXEC_2977 目录名；run2 ga6 Wd 转置形状；run3 qa1 锚对象错（2927 dirs_word 是另一套 57 词 en/L 配对集，74-cell 轴系谱在 2979 d_lang_u）+ act 重算 float64→bf16（差 2-3 ULP ≈5e-3，门校准错误非捕获错误）；run4 拆分 matmul→fused 单 matmul（cuBLAS 舍入路径不同 4e-4..1e-2）；run5 K1_sig AND→OR（对齐 2989 冻结注册规则）。

### 入账
Ledger 135 / L14 103 / hash %(ledhash)s 复算自洽；seal npz8=%(npz8)s res8=%(res8)s script8=%(script8)s；产物 immutable 落盘。

### 接续（下一个 2997）
A（主选）卡片集 v2 重分级 propagate：2995 三卡复算 + 面板判决 omega_f1_panel_partial→replicated 修订登记；B Ω-F2 真续卡：s_c 开关注入卡 2945/2953 需先在 GLM4 重建 rotation-target 链（2939 dcks/Vt8，独立 phase）；C qwen K2 L6 深度负 z 剖面（早层 weight-side 反对齐=新现象）；D 2989 T3 加密复测 + k 剂量曲线。
"""
    memo += section % {
        'stamp': stamp,
        'ledhash': hc,
        'npz8': '327f4838',
        'res8': 'd03a47b0',
        'script8': '0600f54e'}
    open(MEMO, 'w', encoding='utf-8').write(memo)
    out.append('memo appended')
else:
    out.append('memo already has 2996 (idempotent)')

# residue check
memo = open(MEMO, encoding='utf-8').read()
tail = memo.split('## Phase 2996:')[1]
res_ok = '%(' not in tail
out.append('memo residue-free=%s hashes=%s'
           % (res_ok, all(h in tail for h in
                          ('327f4838', 'd03a47b0',
                           '0600f54e', hc))))

# ---------- workspace log ----------
ws = ''
if os.path.isfile(WSLOG):
    ws = open(WSLOG, encoding='utf-8').read()
if 'Phase 2996' not in ws:
    ws += ('\n- Phase 2996 Ω-F2a 口径审计（run6 权威 100.2s，锚 12/12 bit 级）：'
           '发现 2995 dirs_attn/dirs_mlp 全 40 行退化（单槽 hook+广播赋值），'
           '其 T1/T2（LTOP-only）侥幸有效、T3 分层 z 作废；真 per-layer 重测 '
           'GLM4 weight-side z +10~+31，K1 share p=0.0 双模型在场，'
           '判决 registry_robust_across_calibers，2995 T3 卡改判 replicated。'
           'Ledger 135 hash %s；产物 phase2996/omega_f2a_registry_caliber_audit/。\n'
           % hc)
    open(WSLOG, 'w', encoding='utf-8').write(ws)
    out.append('wslog appended')
else:
    out.append('wslog already has 2996')

# ---------- MEMORY.md ----------
mm = open(MEM, encoding='utf-8').read()
old_next = ('**下一个 2996**：A（主选）Ω-F2 续卡复制'
            '（s_c 开关注入机器卡 2945/2953 + 词盲卡 2940，'
            'GLM4 注入协议重建）；B T3 空间口径复审'
            '（判“注册表不复制”是否口径伪影）；'
            'C L 词轴位跨模型定量；D 2989 T3 加密复测+k 剂量。')
new_next = ('**下一个 2997**：A（主选）卡片集 v2 重分级登记'
            '（2995 T3 改判 replicated，面板 partial→'
            'replicated）；B GLM4 rotation-target 链重建'
            '（Ω-F2 s_c 注入卡前置）；C qwen K2 L6 早层'
            '反对齐剖面；D 2989 T3 加密复测+k 剂量。')
if old_next in mm:
    mm = mm.replace(old_next, new_next, 1)
    out.append('memory next updated')
elif new_next in mm:
    out.append('memory next already updated')
else:
    out.append('MEMORY next MISS - manual check needed')

old_chain = ('2995 Ω-F1 跨模型面板：类分离 p 0.0014、头集中 '
             'maxT 地板、注册表不复制（z 负）=首分化点，'
             'L 词轴位重排；')
new_chain = ('2995 Ω-F1 面板：类分离+头集中复制；'
             '2996 口径审计：2995 T3=dirs 全层退化伪影'
             '（单槽 hook），真分层 K2 z +10~31、K1 share '
             'p=0 双模型双口径在场=registry_robust，'
             'T3 改判 replicated；')
if old_chain in mm:
    mm = mm.replace(old_chain, new_chain, 1)
    out.append('memory chain updated')
elif '2996 口径审计' in mm:
    out.append('memory chain already updated')
else:
    out.append('MEMORY chain MISS - manual check needed')
open(MEM, 'w', encoding='utf-8').write(mm)
out.append('memory chars=%d max2996=%s star2997=%s'
           % (len(mm), 'max=2996' in mm, '**2997**' in mm))

open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\tmp_closeout2996_log.txt', 'w',
     encoding='utf-8').write('\n'.join(out) + '\n')
print('ok')
