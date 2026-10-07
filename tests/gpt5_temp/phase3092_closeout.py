# -*- coding: utf-8 -*-
"""Phase 3092 closeout (idempotent):
Ledger(measurements + L14) -> MEMO append ->
audit 53 addendum -> wlog -> MEMORY.md ->
closeout_log."""
import hashlib
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3092'
     r'\omega_p89_gate_sensitivity')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
z = np.load(R + r'\omega_p89_gate_sensitivity.npz',
            allow_pickle=False)
verdict = res['verdict']
assert verdict == 'gate_sensitive_substantive', verdict
assert res['forward_free'] is True
assert int(res['forwards']) == 0
assert bool(z['ANCH_BITS_ALL'])
assert str(z['VERDICT']) == verdict
o.append('loaded result/seal/npz ok; npz keys=%s'
         % sorted(map(str, z.files)))

# ---------- Ledger ----------
meas = {
    'meas_id': 'meas3092_omega_p89_'
               'gate_sensitivity',
    'phase': 3092,
    'claim':
        'Omega-P89 (plan A of the 3091 '
        'Stouffer-signature finding) - '
        'G_DS gate sensitivity '
        'formalization, forward-free.  '
        'Four arbitration phases (3081 '
        'DS7B L37, 3085 3B L34, 3087 A2 '
        'GLM4 L37, 3089 GLM4 L38) x 6 '
        'tests (f2_cTT x {T,U} x '
        '{AB,AC,BC}) x 4 gates: G0 '
        'preregistered count (cnt>=4 AND '
        'min_sp>0), G1 signed Stouffer '
        'z>1.645, G2 unsigned Stouffer '
        'z>1.645, G3 Bonferroni (any '
        'p<0.05/6 AND sp>0).  All 16 '
        'bit-checks pass (cnt/bonf/minsp/'
        'unsigned-Stouffer recomputed == '
        'stored GDS_*).  Gate map: G2 '
        'passes {3081, 3089}, G1 passes '
        '{3087}, G3 passes {3081}, G0 '
        'passes none.  VERDICT '
        'gate_sensitive_substantive: '
        'preregistered prediction (only '
        '3089 flips under G2) holds for '
        '3089 (3.1501 vs sign-aware '
        '+1.1012) but not globally - '
        '3081 is the extreme '
        'unsigned-inflation case (z_un '
        '1.7606 vs z_sig +0.0138, '
        'sign-corrected evidence cancels '
        'to zero) and flips under G2 AND '
        'G3; 3087 flips under G1 (signed '
        'pooling z=+2.2967 with cnt=0, '
        'no individually significant '
        'positive) - correlated-variant '
        'pooling is a second, distinct '
        'inflation mode.  G0 remains the '
        'sole decision gate; unsigned '
        'Stouffer formally disqualified.  '
        'CAVEATS: 6 tests are correlated '
        'variants (Stouffer independence '
        'violated for G1/G2); only 4 '
        'phases tested; inherited frozen-'
        'spearman tie-order convention.',
    'verdict': verdict,
    'anchors':
        '16 bit-checks (4 phases x '
        'cnt/bonf/minsp/unsigned-Stouffer '
        'vs stored GDS_*) all True - see '
        'npz ANCH_BITS_ALL; VERDICT key '
        'bit-equal result.json verdict',
    'artifacts': {
        'result': 'phase3092/'
                  'omega_p89_gate_sensitivity/'
                  'result.json',
        'npz': 'phase3092/'
               'omega_p89_gate_sensitivity/'
               'omega_p89_gate_sensitivity.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'forward-free, 0.003s runtime, '
            'no model loads; gate '
            'definitions preregistered in '
            'execution.json before any '
            'gate comparison',
}
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3092
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 230
    l14['connects'].append({
        'meas_id': 'meas3092_omega_p89_'
                   'gate_sensitivity',
        'phase': 3092,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change':
            'Omega-P89: G_DS gate '
            'sensitivity SUBSTANTIVE - gate '
            'map G2={3081,3089} G1={3087} '
            'G3={3081} G0={}; unsigned '
            'Stouffer disqualified as '
            'decision gate (3081 z_un 1.76 '
            'vs z_sig +0.01 extreme '
            'cancellation; 3089 3.15 vs '
            '+1.10); G0 preregistered count '
            'gate stays sole standard - '
            'G_DS absent at all 4 tested '
            'points.  Next: 3093 qwen3-14b '
            'fifth spectrum point (~21k '
            'fw); B 4B trunk anatomy; C R1 '
            'reuse topology panorama'})
    assert len(l14['connects']) == 198
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already has 3092')

# ---------- MEMO append ----------
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3092:' not in memo:
    entry = (
        '\n## Phase 3092: Ω-P89 G_DS 门敏感性正式化'
        '（免前向）——判决 gate_sensitive_substantive，'
        '门选择实质性双向敏感 [%s]\n\n'
        '**状态**: 已执行（免前向 0.003s；四门定义与'
        '预注册预测先于任何门对比冻结于 '
        'execution.json）。\n\n'
        '### 1. 目标与原理\n'
        '3091 C 臂发现 3089 预注册的无符号 Stouffer '
        '（z=3.150）把小 p 一律计为正向证据、与 sp '
        '符号无关，属高估口径。本 Phase 将该发现正式化：'
        '四个仲裁 Phase（3081 DS7B L37 / 3085 3B L34 / '
        '3087 A2 GLM4 L37 / 3089 GLM4 L38）× 6 检验'
        '（f2_cTT×{T,U}×{AB,AC,BC}）× 四门对比：\n'
        '- **G0**（预注册计数门）：cnt_pos≥4 且 '
        'min_sp>0（合取复制标准）；\n'
        '- **G1**（符号感知 Stouffer）：'
        '$$z_{sig}=\\sum_i \\mathrm{sign}(sp_i)'
        '\\cdot \\Phi^{-1}(1-p_i)/\\sqrt{6}>1.645$$；\n'
        '- **G2**（无符号 Stouffer）：'
        '$$z_{un}=\\sum_i \\Phi^{-1}(1-p_i)'
        '/\\sqrt{6}>1.645$$；\n'
        '- **G3**（Bonferroni）：任一 p<0.05/6 且 '
        'sp>0。\n'
        'bit 校验：重算 cnt_pos/n_bonf/min_sp/无符号 '
        'z 与各 Phase 存储 GDS_* 比对。预注册预测：'
        '仅 3089 在 G2 下翻转；判决树 '
        'gate_unsigned_inflates / gate_order_stable / '
        'gate_sensitive_substantive。\n\n'
        '### 2. bit 校验（全过）\n'
        '4 Phase × 4 项（cnt/bonf/minsp/stouffer）'
        '共 16 项重算全部与存储值 bit 级相等'
        '（ANCH_BITS_ALL=True）。\n\n'
        '### 3. 门矩阵\n'
        '| Phase | 存储判决 | cnt | bonf | min_sp | '
        'z_un | z_sig | G0 | G1 | G2 | G3 |\n'
        '|---|---|---|---|---|---|---|---|---|---|---|\n'
        '| 3081 DS7B L37 | ds7b_cos_absent | 1 | 1 | '
        '-0.2670 | 1.7606 | +0.0138 | F | F | T | T |\n'
        '| 3085 3B L34 | third_mixed_absent | 1 | 1 | '
        '-0.5400 | 1.6431 | -1.0063 | F | F | F | F |\n'
        '| 3087 A2 GLM4 L37 | fourth_mixed_absent | 0 '
        '| 0 | -0.2443 | 1.5643 | +2.2967 | F | T | F '
        '| F |\n'
        '| 3089 GLM4 L38 | fourth_l38_mixed_absent | '
        '1 | 0 | -0.4426 | 3.1501 | +1.1012 | F | F | '
        'T | F |\n'
        '门通过集：G2={3081,3089}、G1={3087}、'
        'G3={3081}、G0={}。\n\n'
        '### 4. 分析\n'
        '- **判决 gate_sensitive_substantive**：'
        '预注册预测对 3089 成立（确在 G2 下翻转，'
        '3.1501 vs 符号感知 +1.1012；pred_3089=True），'
        '但全局不成立（others_stable=False）——3081 '
        '在 G2 与 G3 下均翻转、3087 在 G1 下翻转。'
        '门敏感性是实质性的、双向的。\n'
        '- **3081 = 无符号膨胀的极端案例**：z_un=1.7606 '
        'vs z_sig=+0.0138——符号校正后证据精确归零，'
        '无符号 z 全部由 1 个强正（sp=+0.563, '
        'p=0.00555）与 3 个负 sp 的抵消制造。'
        '在"任一证据"型门（G2/G3）下 DS7B L37 会被'
        '误判为 present；合取复制门正确拒绝。\n'
        '- **符号汇聚是第二失败模式**：3087 z_sig='
        '+2.2967 过阈但 cnt=0（无单项显著正，4 个弱正 '
        'p∈[0.07,0.24]）——6 个相关变体加性汇聚制造'
        '显著性，与无符号膨胀机理不同：G1 符号感知但'
        '违反独立性，G2 连符号都不感知。\n'
        '- **G0 维持唯一判决门**：预注册计数门是唯一'
        '全否门——G_DS 假设在全部 4 个检验点（3 架构 × '
        '2 层位）在复制标准下保持 absent。无符号 '
        'Stouffer 从此正式取消判决门资格（3091 发现'
        '升级为纪律）。\n'
        '- **门语义分层**：G0=合取复制标准（逐检验'
        '显著 + 全正）；G1/G2=加性汇聚（跨检验池化）；'
        'G3=单检验多重校正。三者回答不同问题，门矩阵'
        '显示无单调序，不可互换。\n\n'
        '### 5. 硬伤与边界\n'
        '- 每 Phase 6 检验是同一 f2_cTT 族的相关变体，'
        'Stouffer 独立性假设对 G1/G2 双双违反'
        '（正相关下有效样本远小于名义 6）——两门'
        '统计量本身偏乐观。\n'
        '- 仅 4 个仲裁 Phase（3 架构 × 2 层位），'
        '门敏感性图谱是局部的。\n'
        '- G1 方向取 sign(sp)，cnt=0 时汇聚过阈对'
        '变体选择脆弱。\n'
        '- 继承约束：frozen spearman 并列值顺序敏感'
        '（3091 约定：单元序=3088 sorted-key 序）。\n\n'
        '### 6. 结论与接续\n'
        'G_DS 门敏感性正式化完成：无符号 Stouffer '
        '否决、G0 计数门维持、门语义分层记录在案。'
        '接续：3093 qwen3-14b 第五谱点（C 选项，'
        '~21k 前向 GPU，补 R1 底册最大缺口）；'
        '备选 B 4B 主干解剖 / D R1 复用拓扑全景。\n\n'
        '资源消耗：免前向，CPU 0.003s；产物 sealed'
        '（npz8=%s result8=%s exec8=%s '
        'script8=%s）。\n'
        % (ts, seal['npz_sha256_8'],
           seal['result_sha256_8'],
           seal['exec8'],
           seal['script_sha256_8']))
    i3091 = memo.rindex('## Phase 3091:')
    assert i3091 > memo.rindex('## Phase 3090:')
    memo = memo.rstrip('\n') + '\n' + entry
    with io.open(MEMO, 'w',
                 encoding='utf-8') as f:
        f.write(memo)
    o.append('memo appended ts=%s' % ts)
else:
    o.append('memo already has 3092')

# ---------- audit 53 ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '五十三' not in aud:
    aud = aud.rstrip('\n') + (
        '\n\n---\n\n'
        '## 五十三、3092 增补：G_DS 门敏感性正式化'
        '——gate_sensitive_substantive\n'
        '1. **判决**：四仲裁 Phase（3081/3085/3087/'
        '3089）× 6 检验 × 四门对比，GDS_* bit 级复现'
        '全过；门矩阵 G2={3081,3089}、G1={3087}、'
        'G3={3081}、G0={}——门选择实质性双向敏感，'
        '预注册预测（仅 3089 在 G2 翻转）对 3089 '
        '成立但全局不成立。\n'
        '2. **无符号 Stouffer 正式否决**：3081 极端'
        '案例 z_un=1.7606 vs 符号感知 +0.0138（证据'
        '归零，纯抵消制造）；3089 3.1501 vs +1.1012。'
        '无符号口径从此禁作判决门，G0 计数门维持唯一'
        '判决标准——G_DS 在全部 4 检验点 absent 存活。\n'
        '3. **符号汇聚是第二失败模式**：3087 '
        'z_sig=+2.2967 过阈但 cnt=0（无单项显著正）'
        '——6 个相关变体加性汇聚制造显著性，Stouffer '
        '独立性假设对 G1/G2 同样违反。HDMCC 更新：'
        '门语义分层（合取复制/加性汇聚/单检验校正），'
        '不可互换。\n')
    with io.open(AUDIT, 'w',
                 encoding='utf-8') as f:
        f.write(aud)
    o.append('audit 53 appended')
else:
    o.append('audit 53 already')

# ---------- wlog ----------
wl = os.path.join(WLOG_DIR, '2026-09-22.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if verdict not in prev:
    line = ('- Phase 3092 Omega-P89 G_DS gate '
            'sensitivity (forward-free): '
            'verdict %s - bit-checks 4x4 all '
            'OK; gate map G2={3081,3089} '
            'G1={3087} G3={3081} G0={}; '
            'unsigned Stouffer formally '
            'disqualified (3081: z_un 1.76 '
            'vs z_sig +0.01); preregistered '
            'count gate G0 stays sole '
            'decision gate.  Audit 53; '
            'ledger 230/L14 198.\n' % verdict)
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
mem_cur = io.open(MEMO_W,
                  encoding='utf-8').read()
if 'max=3092' not in mem_cur:
    idx = mem_cur.find('## 下一步')
    assert idx > 0
    mem_cur = mem_cur[:idx] + (
        '## 下一步\n'
        '- max=3092，下一个 3093（A qwen3-14b '
        '第五谱点 ~21k 前向 GPU；B 4B 主干解剖；'
        'C R1 复用拓扑全景——reuse_inventory '
        '底册已备）。\n')
    mem_cur = mem_cur.replace(
        '均显著）**。',
        '均显著）→ 3092 G_DS 门敏感性 '
        'gate_sensitive_substantive'
        '（无符号 Stouffer 否决）**。')
    mem_cur = mem_cur.replace(
        '泛化声明分层（跨架构声明目前=4 架构 '
        '1 非 Qwen 点）。',
        '泛化声明分层（跨架构声明目前=4 架构 '
        '1 非 Qwen 点）；无符号 Stouffer '
        '禁作判决门（3092 正式化，G0 计数门'
        '维持唯一标准）。')
    assert len(mem_cur) < 3000, len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3092')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
