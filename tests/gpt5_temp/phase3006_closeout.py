# -*- coding: utf-8 -*-
"""Phase 3006 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3006'
     r'\omega_a1_base_language_axis_xdir')
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
assert verdict == 'base_shared_eraser', verdict
assert res['anchor_all_ok'] is True
band = res['A']['L17_band']
assert band['med_r1'] == 0.655793
assert band['med_medcos'] == 0.784023
assert band['med_amp_ratio'] == 0.4554
assert res['A']['null_r1_rowgauss'] == 0.022317
assert res['anchors']['a5_diff'] == 0.0
assert res['anchors']['a6_diff'] == 0.0
t3 = res['T3_base']['L17']
assert t3['35']['trk_ratio'] == 118.4392
assert res['a7_chat_mirror']['sep_f_base'] == 192.9341

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3006
           for m in led['measurements']):
    claim = (
        'Omega-A1 (plan v5 P1, alignment trio #1) - '
        'Qwen3-4B-BASE language axis + xdir response '
        '+ operator structure: verdict '
        'base_shared_eraser (frozen 2-D gate). '
        'All-Base-internal geometry (dirs/Vt8/dcks/'
        'xdir rebuilt in-session, a5 rebuild '
        'determinism 0.0; Base tids == 2887 chat '
        'keys, family tokenizer). RESULTS: (i) the '
        'language axis EXISTS in Base and is STRONGER '
        '(sep_f_base 192.9 vs chat 185.7; null0/f '
        'ratio 0.411 vs chat 0.414 - nearly '
        'identical null structure); (ii) the mid-band '
        'machinery is the SAME OPERATOR as chat: '
        'shared low-rank redirect (band med r1 0.656/'
        'medcos 0.784 vs null 0.0223 = 29x; head '
        'layers 18-22 up to r1 0.78/medcos 0.90 - '
        'STRONGER sharing than chat), dose-linear '
        'ratios (L17 s1/s2/s3 = 0.50/1.08/1.29; L15 '
        '0.31/1.03/1.20), xdir-specific ~110x '
        '(sub-random controls 0.009-0.011); (iii) T3: '
        'L17 injection 70->16.96@18 (76 pct loss) '
        'then REGROWS to 118.4@35 = 1.69x injection '
        '(chat: 1.66x), xdir proj 4900->58@18->919@35 '
        'growing with amplitude - same '
        'redirect-with-amplification shape. '
        'INTERPRETATION GATE CAVEAT: the frozen '
        'eraser label comes from band-median '
        'amp_ratio 0.4554 vs the 0.5 gate - a '
        'boundary effect of averaging the early '
        'decay segment (l18-25, amp 0.19-0.35) with '
        'the late amplification segment (l32-35, amp '
        '1.05-1.48); the true profile is '
        'decay-then-amplify, chat-shaped. CONCLUSION: '
        'the chat mid-band machinery (shared low-rank '
        'redirect+amplification, dose-linear, xdir-'
        'specific) is PRE-TRAINING-EMERGENT, not '
        'alignment-carved; Base->chat alignment only '
        'modulates it upward (s2 ratio 1.08->1.50, '
        'r1 0.656->0.715). The GLM4-vs-qwen immunity '
        'gap (23.2x, 3000) is therefore a FAMILY/'
        'recipe difference, not an alignment-state '
        'difference - plan v5 P1 primary question '
        'answered for the qwen family.')
    meas = {
        'meas_id': 'meas3006_omega_a1_base_language_'
                   'axis_xdir',
        'phase': 3006,
        'claim': claim,
        'verdict': verdict,
        'anchors': '6/6 (a0 57 words; a1 tids==2887 '
                   'keys; a2 0.0; a3 8.5e-14; a4 ok; '
                   'a5 0.0 rebuild determinism; a6 0.0)',
        'artifacts': {
            'result': 'phase3006/omega_a1_base_'
                      'language_axis_xdir/result.json',
            'npz': 'phase3006/omega_a1_base_language_'
                   'axis_xdir/'
                   'omega_a1_base_language_axis_xdir'
                   '.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run3 authoritative (run1 a7 scalar '
                'conversion crash - 2935 s_base is '
                'per-word; run2 full pass recorded '
                'before correction_note retro-'
                'registration; no verdict-affecting '
                'defects).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 145
    l14['connects'].append({
        'meas_id': 'meas3006_omega_a1_base_language_'
                   'axis_xdir',
        'phase': 3006,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-A1: Base mid-band = '
                        'SAME shared low-rank redirect+'
                        'amplification operator as chat '
                        '(r1 0.656/medcos 0.784, head '
                        '0.78/0.90; dose-linear 1.08@'
                        's2; T3 tail 1.69x) => machinery '
                        'PRE-TRAINING-EMERGENT, not '
                        'alignment-carved; alignment '
                        'only modulates upward (s2 '
                        '1.08->1.50); GLM4-qwen gap = '
                        'family difference; eraser '
                        'label = band-median boundary '
                        'effect (registered)'})
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
if '## Phase 3006:' not in memo:
    sec = u'''## Phase 3006: Ω-A1 Base 语言轴 + xdir 响应——中带机制预训练涌现，非对齐雕刻 [%(created)s]

**判决：`base_shared_eraser`**（run3 权威，14.1s，锚 6/6：a5 dirs 重建确定性 0.0、a2/a6 0.0、a3 8.5e-14；Base 全内部几何，无 chat 数值锚）

### 设计（方案 v5-P1 对齐三件套第一件）
3004 机器 verbatim + **全 Base 内部几何**：dirs_base 两次重建（a5 确定性 0.0）、Vt8b/u35b、in-session dcks、xdir=dcks_S@Vt8_S（S_IDX verbatim）；tokenizer tid == 2887 chat keys（家族同词表，a1 通过）；null0 seed 2896 verbatim（tids 与 chat 相同）。剂量 L4/L15/L17 × s∈(1,2,3)，sub/full 随机对照，T4 全捕获（轴序无 bug 口径），SVD 算子结构 + 能量分账。**判决改为二维组合门**（算子 shared/scatter/mixed × 功能 amplifier/neutral/eraser）。

### 核心结果（重复三遍）
**① 语言轴 Base 自带且更强**：sep_f_base **192.9** vs chat 185.7（+4%%）；null0/func 比 **0.411** vs chat 0.414（几乎相同）。**② 中带机制与 chat 同构**：共享低秩重定向（band med r1 **0.656**/medcos **0.784** vs null 0.0223=29×；头部 l18-22 高达 **0.78/0.90，比 chat 全带 med 还强**）；剂量线性（L17 s1/s2/s3=0.50/1.08/1.29）；xdir 特异 ~110×（随机对照 0.009-0.011）。**③ T3 剖面同款**：L17 注入 70→16.96@18（76%% 衰减）→**118.4@35（1.69×，chat 1.66×）**，xdir 投影 4900→58@18→919@35 随幅度增长——**redirect-with-amplification 形状完全一致**。结论：**qwen chat 的中带机制（共享低秩重定向+放大、剂量线性、xdir 特异）是预训练涌现的，Base→chat 对齐仅上调调制（s2 ratio 1.08→1.50、r1 0.656→0.715）；3000 的 GLM4-qwen 23.2× 免疫差是家族/配方差异，非对齐状态差异**。**

### 判决标签边界效应（在册澄清）
功能维度 'eraser' 来自 band 全程中位 amp_ratio **0.4554** 恰低于 0.5 门——band 前半衰减段（l18-25，amp 0.19-0.35）与后半放大段（l32-35，amp 1.05-1.48）平均的边界效应；真实剖面=先衰减后放大（chat 形状）。判决按冻结门登记不变，解释以剖面为准。

### 对附件空白一的回答（v5-P1）
对齐训练**不雕刻**放大带——它在 Base 中已完整存在且共享性更强。附件"Base=野马/Chat=警犬"的比喻在 qwen 家族内被否证：qwen base 与 chat 的语言轴与中带机制几乎无差；免疫差异必须到 GLM4 家族找（下一步 Ω-A2：GLM4 家族 Base 对照，需 GLM4-9B-Base 资产）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3006/omega_a1_base_language_axis_xdir/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3007 = A（主选）生成轨迹记录器 v5-P2a（自回归动力学第一块：逻辑词 vs 内容词锁定度，答案见附件空白二/三）；B Ω-A2 GLM4-9B-Base 下载 + 家族对照（补齐空白一）；C Base 剂量曲线加密（s∈(1,2,3,4,6) 定位放大饱和点）。
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
if 'Phase 3006' not in prev:
    line = ('- Phase 3006 Omega-A1 Base language axis '
            '+ xdir: verdict base_shared_eraser (2-D '
            'gate); mid-band machinery SAME as chat '
            '(r1 0.656/medcos 0.784 head 0.78/0.90; '
            'dose-linear 1.08@s2; T3 tail 1.69x vs '
            'chat 1.66x) => PRE-TRAINING-EMERGENT, '
            'alignment only modulates up (s2 1.08->'
            '1.50); GLM4-qwen gap = family difference; '
            'eraser label = band-median boundary '
            'effect; ledger 145/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
