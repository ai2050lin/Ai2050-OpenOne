# -*- coding: utf-8 -*-
"""Phase 3137 closeout: five-write chain
(ledger -> MEMO -> wlog -> workspace
MEMORY -> npz already done) + idempotent
guards. Run after full-run DONE."""
import hashlib
import io
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
OUT = RDIR + r'\phase3137' \
      r'\omega_p135_modecoop_' \
      r'k0anat_coorddecomp_l35recheck'
MEMO = ROOT + r'\research\gpt5\docs' \
       r'\AGI_GPT5_MEMO.md'
LEDGER = ROOT + r'\research\gpt5\atlas' \
         r'\atlas_ledger.json'
WLOG = ROOT + r'\.workbuddy\memory' \
       r'\2026-09-28.md'
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'

steps = []

# ---------- read result ----------
r = json.load(io.open(
    os.path.join(OUT, 'result.json'),
    encoding='utf-8'))
assert r['smoke'] is False
assert r['phase'] == 3137
V = r['verdict']
raw = io.open(os.path.join(OUT,
                           'result.json'),
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
assert sha8 == '4f8055bb', sha8
runtime = r['runtime_s']
pe = r['part_e']
pf = r['part_f']
pg = r['part_g']
ph = r['part_h']
mm_e = pe['mode_matrix']
f_res = pf['f_res']
g_res = pg['g_res']
h_res = ph['h_res']

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
MEAS = 'meas3137_omega_p135_modecoop_' \
       'k0anat_coorddecomp_l35recheck'
exists = [e for e in led['measurements']
          if e.get('meas_id') == MEAS]
if exists:
    steps.append('ledger: exists, skip')
else:
    claim = (
        'Omega-P135 (3137, T4 twentieth '
        'phase: prompt-only vs allstep '
        'mode x layer x dose matrix + k0 '
        'anatomy + co36@L17 coordinate '
        'decomposition + L35 union '
        'recheck. Mode gap (s2.0) all '
        'positive: L17 0.7578 L29 0.5547 '
        'L33 0.4609 L38 0.8203 (ratio '
        '7.56x); po dose-monotone all 4 '
        'layers; high-dose po catch-up '
        '(L17 s4.0 0.8125, L38 s4.0 '
        '0.9375) -> mode gap is efficacy '
        'not reachability; 672-row '
        'prompt-only anchor L38 s2.0 '
        '0.089286 bit-level reproduces '
        '3134; k0 negative confirmed '
        '(w8full 0.6484 vs nok0 0.7500, '
        'contrib -0.1016), k0 alone '
        'active 0.1562, flip decomp '
        'only_full=5 only_nok0=18; '
        'k0only==po_l17_s1.0 fstep sha8 '
        'identical (9ba70f6f) semantic '
        'self-proof; co36@L17 flat '
        '(top25 0.1328 vs bot25 0.1172, '
        'no coordinate gradient), co36 '
        'dose monotone 0.0859/0.1406/'
        '0.2266/0.6562, co50:co36 '
        'equipotent dose ratio ~2-4x; '
        'L35 union zero dose-robust '
        '0.0547/0.0703/0.1172 (d4 still '
        'near-zero vs L17 co36 d4 '
        '0.6562); 9 cross-session soft '
        'anchors drift all 0.00e+00; '
        'xphase P=1.0 (7th session) - '
        'verdict ' + V)
    entry = {
        'meas_id': MEAS,
        'phase': 3137,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3137/omega_p135_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p135_readout.npz'},
        'hashes': {
            'result_sha256_8': sha8},
        'anchors': [
            'meas3136_omega_p134_'
            'conddose_crossmatrix_'
            'w8drop'],
        'note': 'mode_gap_all/po_monotone'
                '; k0_negative+alone_active'
                '; co36_l17_flat/dose_'
                'monotone; l35_zero_robust'}
    led['measurements'].append(entry)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
    steps.append('ledger: appended n=%d '
                 'sha8=%s'
                 % (len(led['measurements']),
                    led['ledger_sha256_8']))

# ---------- 2. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3137:' in memo:
    steps.append('memo: exists, skip')
else:
    hhmm = time.strftime('%H:%M')
    sec = []
    sec.append('')
    sec.append('## Phase 3137: Ω-P135 模式'
               '×层位×剂量+k0解剖+坐标分解'
               '（T4 第20 Phase）[%s]' % hhmm)
    sec.append('')
    sec.append('**性质**：T4 第 20 Phase，'
               '3136 MEMO §5 预注册四项执行，'
               'design_seal.json 观测前冻结。'
               'Part A offline 3136+3134 链接'
               '断言（result sha8 3903af46 + '
               'co50 52b126af/union '
               '1e7edb1a + L38 po 672 行锚 '
               '0.0893 硬断言）；Part E 模式'
               '对照矩阵（prompt-only vs '
               'allstep × L17/29/33/38 × '
               'scale {1,2,4} = 24 试验 + '
               '672 行 L38 po s2.0 锚试，'
               '128 行 A1 方向 allstep 行为'
               'chg）；Part F k0 解剖（w8full/'
               'nok0/k0only/drop3 + 行级翻转'
               '分解）；Part G co36@L17 坐标'
               '分解（co36_rank top25/bot25 + '
               'co36/co50 @L17 剂量曲线 '
               '{0.5,1,2,4}/{0.5,1,2}）；'
               'Part H L35 union 剂量复核'
               '（{1,2,4} + co36 d2 对照）。'
               '运行 13998.5s，xphase P=1.0'
               '（第 7 会话）。')
    sec.append('')
    sec.append('### 1. 三大发现（重复三遍）')
    for _ in range(3):
        sec.append(
            '1. **模式×层位×剂量定律：'
            'mode_gap_all|po_dose_monotone**。'
            'allstep/prompt-only 行为差距在'
            '四层全正（s2.0 格 gap L17 '
            '0.7578/L29 0.5547/L33 0.4609/'
            'L38 0.8203，L38 ratio 7.56×）；'
            'po 绝对值普遍弱（L29/33/38 '
            's2.0 仅 0.086/0.055/0.125）但'
            '高剂量可部分追赶（L17 s4.0 '
            '0.8125、L38 s4.0 0.9375 vs '
            'all 1.0）→ **模式差是效力系数'
            '不是可达性门槛**；672 行 L38 '
            'po s2.0 = 0.089286 与 3134 '
            '位级复现（跨会话跨行集锚）。'
            '行为效果主体由解码步承载，'
            'prompt-forward 注入效力低')
    for _ in range(3):
        sec.append(
            '2. **k0 负贡献确认+单独活跃：'
            'k0_negative_confirmed|'
            'k0_alone_active**。w8full '
            '0.6484（drift 0 vs 3136）vs '
            'nok0 0.7500 → k0 净贡献 '
            '−0.1016；k0only 单独 0.15625 '
            '非零（注入有传导但方向为干扰）；'
            '行级翻转分解：nok0 新增 18 行/'
            '丢失 5 行（net +13 = 128×'
            '0.1016 精确吻合）；**fstep_'
            'w8_k0only 与 fstep_po_l17_'
            's1.0 sha8 位级相同（9ba70f6f）'
            '——k0only ≡ prompt-only 语义'
            '自证**，模式对照内部一致')
    for _ in range(3):
        sec.append(
            '3. **错配坐标平坦+L35 零贡献'
            '剂量鲁棒：co36_l17_flat|'
            'coord_dose_monotone|'
            'l35_union_zero_dose_robust**。'
            'co36@L17 错配效果在 50 坐标内'
            '平坦（top25 0.1328 vs bot25 '
            '0.1172，无坐标梯度）→ 错配'
            '有效性来自坐标数量/总注入幅度'
            '而非特定坐标身份；co36 @L17 '
            '剂量单调 0.0859/0.1406/0.2266/'
            '0.6562（d4 跳升），co50:co36 '
            '等效剂量比 ~2–4×；L35 union '
            'd1/2/4 = 0.0547/0.0703/0.1172 '
            '剂量鲁棒近零（d4 仍 ≪ L17 '
            'co36 d4 0.6562）→ **L35 端口'
            '零贡献是质差不是量差**')
    sec.append('')
    sec.append('### 2. 关键数值')
    sec.append(
        'Part E mode_matrix（all vs po，'
        's1/s2/s4）：L17 0.6875/1.0/1.0 '
        'vs 0.156/0.242/0.8125；L29 '
        '0.25/0.6406/1.0 vs 0.078/0.086/'
        '0.125；L33 0.211/0.516/0.992 vs '
        '0.070/0.055/0.234；L38 0.781/'
        '0.945/1.0 vs 0.047/0.125/0.9375。'
        'a672_l38_po_s2 = 0.089286（'
        'first=30）。Part F：w8full '
        '0.6484/nok0 0.7500/k0only '
        '0.1562/drop3 0.4062（后二者 '
        'drift 0 vs 3136）；flips full=83 '
        'nok0=96 both=78 only_full=5 '
        'only_nok0=18。Part G：top25 '
        '0.1328/bot25 0.1172；co36 d0.5/'
        '1/2/4 = 0.0859/0.1406/0.2266/'
        '0.6562；co50 d0.5/1/2 = 0.1406/'
        '0.4219/0.8516（d1 双锚 drift 0）。'
        'Part H：l35union d1/2/4 见上；'
        'l35co36 d2 0.0703（3136 d1 锚 '
        '0.0781 一致弱）。9 软锚 drift 全 '
        '0.00e+00。运行 13998.5s。')
    sec.append('')
    sec.append('### 3. 硬伤')
    sec.append(
        '①po_dose_monotone 判据宽松：L33 '
        'po 0.0703→0.0547→0.2344 中段微'
        '降仍判 True（判据为末格>首格，'
        '非逐格单调）——读数时注意；'
        '②k0 负贡献机制仍未解释（本 '
        'Phase 只完成解剖定位：净负+单独'
        '活跃+行级分解，注意力窗位移/'
        '前缀污染假说未测）；③co50 剂量'
        '只到 d2（0.8516 近饱和未测 d4）；'
        '④co36@L17 坐标维（top/bot 平坦）'
        '与剂量维的交互未做二维网格；'
        '⑤128 行 SCAN_N 与 672 行全集'
        '跨样本集禁令照旧（a672 锚试'
        '除外，其本身就是 672 行）；'
        '⑥po s4.0 高剂量追赶的机制'
        '（饱和域压缩差距？）未分离。')
    sec.append('')
    sec.append('### 4. 机制拼图更新')
    sec.append(
        '①写入时序定律定量化：行为效果由'
        '解码步 k1+ 承载，k0 prompt 注入'
        '净负（−0.102）且单独活跃（0.156）'
        '——注入时序=效力维度；②k0only ≡ '
        'prompt-only（fstep 位级相同）：'
        '模式对照语义统一，历史 3134 '
        'prompt-only 数字可与 k0only 互译；'
        '③错配坐标有效性=数量/幅度驱动'
        '（top/bot 平坦）非身份驱动 → '
        '端口类理论再强化：L17 端口消费'
        '注入总幅度而非精确坐标；④L35 '
        '端口零贡献剂量鲁棒（d4 仍 0.117）'
        '——端口差异是质差；⑤高剂量 po '
        '追赶 → 模式差是效力系数。'
        '**指纹范式文档（FINGERPRINT_'
        'PARADIGM_PLAN.md）已定稿：'
        '3138 起转入 Absolute State Bank '
        '路线（Ω-P136）。**')
    sec.append('')
    sec.append('### 5. 3138 预注册（观察后'
               '冻结）')
    sec.append(
        '按 FINGERPRINT_PARADIGM_PLAN.md '
        'Ω-P136：①Absolute State Bank：'
        'SCAN_N 128 行 × ≥3 模板网格 × '
        '40 层全量 hidden 持久化（fp16 '
        'npz，预算 ≤3GB）；②三分解 '
        'H = B_l（层公共）+ I_w（词身份）'
        '+ C_c（上下文/模板）+ 残差：各'
        '成分 split-half 稳定性 ≥0.9 门 + '
        '探针 AUC 对比（I_w 单独 vs B_l '
        '单独 vs 联合）；③k0 干扰机制初探'
        '（po 注入下注意力窗/前缀位置分析，'
        '时间允许则并入）。锚：3137 result '
        'sha8=4f8055bb、verdict 全等、'
        'drift 全 0 锚组（0.640625/0.515625'
        '/0.945312/0.089286/0.648438/'
        '0.406250/0.140625/0.421875/'
        '0.054688）、co50 52b126af、'
        'co36_rank_order 0f4c25b1、union '
        '1e7edb1a、ledger n=274。')
    sec.append('')
    sec.append('产物：`tests/glm5/result/'
               'rdc_query_construction_'
               '20260913/phase3137/'
               'omega_p135_modecoop_k0anat_'
               'coorddecomp_l35recheck/`'
               '（result.json sha256_8=%s、'
               'design_seal.json、run_log.txt、'
               'p135_readout.npz 29 键）；'
               '脚本 `tests/glm5/phase3137_'
               'omega_p135_modecoop_k0anat_'
               'coorddecomp_l35recheck.py`。'
               % sha8)
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write('\n'.join(sec) + '\n')
    steps.append('memo: appended')

# ---------- 3. wlog ----------
wl = ''
if os.path.exists(WLOG):
    wl = io.open(WLOG,
                 encoding='utf-8').read()
if 'Phase 3137' in wl:
    steps.append('wlog: exists, skip')
else:
    line = ('- Phase 3137 (Ω-P135) 闭环：'
            '正式跑 13998.5s 一次通过（SMOKE '
            '三修：rev-3137a 区间替换事故误删'
            'SEAL→整体重写、rev-3137b a672 '
            'SMOKE 行数 cap、rev-3137c co36'
            '_rank len 36→50 事实修正）；'
            'verdict=a_3136_ok|mode_gap_all|'
            'po_dose_monotone|k0_negative_'
            'confirmed|k0_alone_active|'
            'co36_l17_flat|coord_dose_'
            'monotone|l35_union_zero_dose_'
            'robust|coverage_full；9 软锚 '
            'drift 全 0；k0only≡po_l17_s1.0 '
            'fstep sha8 位级相同；closeout '
            '五写+磁盘复核。\n')
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        if wl and not wl.endswith('\n'):
            f.write('\n')
        f.write(line)
    steps.append('wlog: appended')

# ---------- 4. workspace MEMORY ----------
mm = io.open(WMEM, encoding='utf-8').read()
if 'meas3137' in mm:
    steps.append('wmem: exists, skip')
else:
    mm2 = mm.replace(
        '- max=3136，下一 3137',
        '- 3137（T4）：模式×层位×剂量矩阵 '
        'mode_gap_all（s2.0 gap 0.76/0.55/'
        '0.46/0.82，L38 7.56×）+po 剂量单调'
        '+高剂量 po 追赶（L17 s4 0.81）=模式'
        '差是效力系数非可达性；k0 解剖 net负'
        '−0.102/单独活跃 0.156/翻转行 +18/−5，'
        '**k0only≡po_l17_s1.0（fstep sha8 '
        '位级同）**；co36@L17 top/bot 平坦='
        '错配有效性数量驱动非身份驱动；L35 '
        'union d4 仍 0.117=零贡献质差；a672 '
        'L38 po 0.089286 与 3134 位级复现；'
        '9 软锚 drift 全 0。\n'
        '- max=3137，下一 3138')
    assert mm2 != mm, 'wmem anchor not found'
    mm2 = mm2.replace(
        '**prompt-only×allstep×层位×剂量对照'
        '矩阵 + k0 负贡献解剖（w8[1..8] 对照）'
        '+ co36@L17 错配坐标分解 + L35 union '
        '零贡献剂量复核。**',
        '**3138 起转 Absolute State Bank 路线'
        '（Ω-P136，FINGERPRINT_PARADIGM_PLAN'
        '.md）：128 行×≥3 模板×40 层 hidden '
        '库 + B/I/C 三分解（split-half ≥0.9 门'
        '+探针 AUC 对比）+ k0 干扰机制初探。**')
    with io.open(WMEM, 'w',
                 encoding='utf-8') as f:
        f.write(mm2)
    steps.append('wmem: updated')

steps.append('result sha8=%s runtime=%.0fs'
             % (sha8, runtime))
out = '\n'.join(steps)
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3137_closeout_steps.txt', 'w',
        encoding='utf-8').write(out)
print('CLOSEOUT OK')
print(out)
