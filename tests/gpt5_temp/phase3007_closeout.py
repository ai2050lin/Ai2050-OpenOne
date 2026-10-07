# -*- coding: utf-8 -*-
"""Phase 3007 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3007'
     r'\omega_p2a_generation_trajectory_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
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
assert verdict == 'logic_locked_perturb_divergent_qwen', \
    verdict
assert res['anchor_all_ok'] is True
t1 = res['T1']
assert t1['locked'] is True
assert t1['lock_ratio_obs'] == 0.457
assert t1['med_dS']['logic'] == 11.742
assert t1['med_dS']['content'] == 25.6958
assert t1['n']['logic'] == 21
assert t1['n']['content'] == 320
t2 = res['T2']
assert t2['recov_count'] == 2 and t2['recov'] is False
assert t2['per_prompt']['P3']['xdir'][
    'first_token_div'] == 53
assert t2['per_prompt']['P0']['xdir']['n_token_div'] == 0
assert res['T3']['med_drift_end'] == 45.9661
assert res['anchors']['a1_diff'] < 1e-5
assert res['anchors']['a4_diff'] < 1e-4

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3007
           for m in led['measurements']):
    claim = (
        'Omega-P2a (plan v5 P2, battle two) - FIRST '
        'autoregressive trajectory recorder: greedy '
        'KV-cache generation, 12 fixed prompts x 64 '
        'tokens, per-token final-norm coords in u35/'
        'Vt8 basis (3002 readout machine, a4/a5 '
        'bit-level 7.2e-06/6.3e-06; a10 generation '
        'determinism rel 0.0). Verdict '
        'logic_locked_perturb_divergent_qwen (frozen '
        'gates). RESULTS: (i) T1 LOCKING: language-'
        'axis step deltas med|Ds| logic 11.742 < '
        'func 15.763 < content 25.696; lock ratio '
        '0.457 < 0.5 gate (n logic 21 / content 320) '
        '- logic-word positions lock the readout '
        '(monotone class ordering logic<func<content'
        '<other 21.758); (ii) T2 PERTURB-RECOVER '
        'SEED (t0=16, L17, s=2, v=unit(mean xdir)): '
        'recovered 2/4 (gate 3) => divergent per '
        'frozen gate; but token-level divergence is '
        'RARE (3/4 prompts ZERO token change, P3 '
        'diverges at step 53, 4 tokens) - the '
        'perturbation lives mostly below the greedy '
        'argmax threshold; coordinate distance peaks '
        '~1.5-4.1 and only partially decays; ctrl '
        '(span(Vt8) orth rnd) recovers as often => '
        'generation-state recovery is NOT xdir-'
        'specific at this scale (contrast forward-'
        'pass ~150x specificity, 3002); (iii) T3 '
        'drift: med |s(end)-s_prefill| 45.97, late '
        'window std 25.51 < early 31.58 (mild late '
        'contraction, no runaway). CONCLUSION: the '
        'readout geometry survives into generation '
        'state with class-ordered locking; the '
        'greedy decoder is perturbation-tolerant at '
        'token level but the coordinate trajectory '
        'carries a persistent offset - consistency-'
        'attractor operationalization (P2b) now has '
        'a baseline.')
    meas = {
        'meas_id': 'meas3007_omega_p2a_generation_'
                   'trajectory_qwen',
        'phase': 3007,
        'claim': claim,
        'verdict': verdict,
        'anchors': '11/11 (a0 57 words; a1 2.17e-08; '
                   'a2 0.0; a3 0.0; a4 7.2e-06; '
                   'a5 6.3e-06; a6 sep_f 185.70; '
                   'a7 9.95e-14; a8 ok; a9 3002 '
                   'integrity; a10 gen det 0.0)',
        'artifacts': {
            'result': 'phase3007/omega_p2a_generation_'
                      'trajectory_qwen/result.json',
            'npz': 'phase3007/omega_p2a_generation_'
                   'trajectory_qwen/'
                   'omega_p2a_generation_trajectory_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative (run1 crashed at '
                'first decode step - fin_cap (1,hid) '
                'batch view needed [0] squeeze; '
                'correction_note registered BEFORE '
                'run2).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 146
    l14['connects'].append({
        'meas_id': 'meas3007_omega_p2a_generation_'
                   'trajectory_qwen',
        'phase': 3007,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2a: FIRST autoregressive '
                        'recorder - logic positions lock '
                        'readout (med|Ds| 11.74 vs '
                        'content 25.70, ratio 0.457; '
                        'class order logic<func<content'
                        '<other); perturb-recover seed: '
                        '2/4 recovered (divergent per '
                        'gate) but token divergence '
                        'rare (3/4 zero token change) '
                        '=> perturbation mostly below '
                        'argmax threshold; recovery '
                        'NOT xdir-specific in '
                        'generation state'})
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
if '## Phase 3007:' not in memo:
    sec = u'''## Phase 3007: Ω-P2a 生成轨迹记录器——逻辑词位锁定读出，扰动容忍于 token 级 [%(created)s]

**判决：`logic_locked_perturb_divergent_qwen`**（run2 权威，57.4s，锚 11/11：a4/a5 位级 7.2e-06/6.3e-06 vs 2935、a1 2.17e-08 vs 2927、a10 生成确定性 rel 0.0）

### 设计（方案 v5-P2/P2b，战役二第一块；146 卡中首个自回归在册物）
3002 readout 机器 verbatim 位级锚 + **KV-cache 贪心生成**（batch=1，12 条固定 prompt × 64 token），逐 token 捕获 final-norm 输入坐标（s=u35 投影、c8=Vt8 坐标）；冻结词类表（logic 12 词/func 44 词单 token 过滤零丢弃）；T2 扰动-恢复种子（P0-P3，t0=16，L17，s=2，v=unit(mean xdir)，对照=span(Vt8)⊥随机）；T3 漂移描述性。

### 核心结果（重复三遍）
**① 锁定成立且类序单调**：语言轴步进 med|Δs| logic **11.742** < func 15.763 < content **25.696** < other 21.758，lock ratio **0.457** < 0.5 门（n logic 21 / content 320）——逻辑词位在生成态锁住读出坐标，比内容词位稳 2.2×。**② 扰动-恢复：判决门下 divergent（2/4 < 3），但 token 级分歧罕见**——3/4 prompt 零 token 改变（坐标距离峰 1.5-4.1 部分衰减），仅 P3 在第 53 步分岔（4 token）；扰动主要活在贪心 argmax 阈值之下。**③ 恢复非 xdir 特异**：对照向量恢复频率相同（P2 ctrl 反而 rec=True）——生成态的恢复不继承前向态的 ~150× 方向特异性（3002），方向特异防御是前向解剖量、不是生成量。T3 漂移：med |s(end)-s_pre| 45.97，晚期窗 std 25.51 < 早期 31.58（轻度晚段收缩，无失控）。**

### 对附件空白二/三的第一步回答
生成态几何存活且类序锁定（逻辑词=最稳读出位），支持"逻辑词是长程依赖载体"的进一步操作化；一致性吸引子（P2b）自此有基线：扰动后坐标偏移持续存在但不轻易翻转 greedy 选择——"幻觉分岔"命名继续门控。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3007/omega_p2a_generation_trajectory_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3008 = A（主选）逻辑词锁定升级（2993 签名机器移植生成态 + 长生成 256 token + 逻辑词位 KV 因果探测）；B P2b 扰动-恢复全网格（t0×scale×方向，恢复指数函数化）；C Ω-A2 GLM4 家族 Base 对照（需 GLM4-9B-Base 资产）。
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
if 'Phase 3007' not in prev:
    line = ('- Phase 3007 Omega-P2a FIRST autoregressive '
            'trajectory recorder: verdict '
            'logic_locked_perturb_divergent_qwen; '
            'locking ratio 0.457 (logic 11.74 vs '
            'content 25.70, class order monotone); '
            'perturb-recover 2/4 (divergent per gate) '
            'but 3/4 prompts zero token change => '
            'perturbation below argmax threshold; '
            'recovery not xdir-specific in generation '
            'state; ledger 146/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
