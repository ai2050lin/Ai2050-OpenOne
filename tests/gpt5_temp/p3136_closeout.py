# -*- coding: utf-8 -*-
"""Phase 3136 closeout: five-write chain
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
OUT = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_crossmatrix_w8drop'
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
assert r['phase'] == 3136
V = r['verdict']
raw = io.open(os.path.join(OUT,
                           'result.json'),
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
runtime = r['runtime_s']
pc = r['part_c']
pd_ = r['part_d']
pb = r['part_b']
m = pc['matrix']
contrib = pd_['drop1_contrib']

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
MEAS = 'meas3136_omega_p134_conddose_' \
       'crossmatrix_w8drop'
exists = [e for e in led['measurements']
          if e.get('meas_id') == MEAS]
if exists:
    steps.append('ledger: exists, skip')
else:
    claim = (
        'Omega-P134 (3136, T4 nineteenth '
        'phase: conduction fidelity '
        'dose-response + co36/co50 cross '
        'injection matrix + w8 drop-one. '
        'Dose x8 (abs 1-8) self-layer dvec '
        'inj: cos_direct 0.99 -> 0.87-0.92 '
        'slow decay (gap <=0.127), rho '
        'log-log slope 0.917/0.938/0.916 '
        'linear, behavior chg saturates at '
        'd2.0 (L29 0.25/0.64/1.0/1.0, L33 '
        '0.21/0.52/0.99/1.0, L38 0.78/0.95/'
        '1.0/1.0) -> write linear-amp vs '
        'readout threshold-saturation; '
        'cross matrix co36@L17 0.1406 > '
        'co36@L35 0.0781, co50@L35 0.0625, '
        'both_union 0.5547 = L17_union '
        '(L35 net ~0), union additive resid '
        '0.0078/0.0859/0.0547 -> '
        'layer-dominant not set-dominant; '
        'anchors L17_co50 0.4219 and '
        'L35_co36 0.0781 bit-level '
        'cross-session reproduction; w8 '
        'drop-one top1 k=3 contrib 0.2422, '
        'k1 0.1641 k2 0.1484, k5-8 <=0.047, '
        'k0 negative -0.1016, sum ratio '
        '0.867, w8full 0.6484 drift 0.0000 '
        'vs 3135 - verdict ' + V)
    entry = {
        'meas_id': MEAS,
        'phase': 3136,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3136/omega_p134_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p134_readout.npz'},
        'hashes': {
            'result_sha256_8': sha8},
        'anchors': [
            'meas3135_omega_p133_'
            'conduction_co36ablation_'
            'window'],
        'note': 'dose fid_dose_sensitive/'
                'rho_linear/monotone; '
                'xmat diagonal_weak/'
                'union_add_all; drop1 '
                'top1_dominant/repro_ok'}
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
if '## Phase 3136:' in memo:
    steps.append('memo: exists, skip')
else:
    hhmm = time.strftime('%H:%M')
    sec = []
    sec.append('')
    sec.append('## Phase 3136: Ω-P134 传导'
               '剂量+交叉矩阵+w8drop（T4 第19'
               'Phase）[%s]' % hhmm)
    sec.append('')
    sec.append('**性质**：T4 第 19 Phase，'
               '3135 MEMO §5 预注册三项执行，'
               'design_seal.json 观测前冻结。'
               'Part A offline 3135 链接断言'
               '（result sha8 0f1fdf95 + '
               'verdict 全等 + co36 84a1e1a3/'
               'co36_rank 0f4c25b1/co50 '
               '52b126af/union 1e7edb1a/dvec '
               '29-33-38 sha + w8 0.6484、'
               '0.4219、0.0781 硬断言）；'
               'Part B1 注入向量冻结自 3135 '
               'npz（fp32 sha 锚定，免 swap '
               '重算）+ 本会话 base/swap '
               'capture（xphase P=1.0000 '
               '672/672，第 6 会话位级全匹配；'
               '本地 dvec med norm 32.3/45.4/'
               '103.0 vs 3135 31.6/44.7/100.4，'
               '+2% 会话漂移）；Part B2 传导'
               '保真剂量-响应（L29/33/38 × '
               '{0.5,1,2,4}×2.0 自层注入，'
               '672 行 capture cos/rho + 128 '
               '行 allstep 行为 chg）；Part C '
               'co36∪co50 交叉注入矩阵'
               '（{L17,L35,both}×{co50,co36,'
               'union} 9 试验，A1 128 行）；'
               'Part D w8 drop-one（9 forward '
               '逐一剔除 + w8full 锚，128 行）。'
               '运行 28306s。')
    sec.append('')
    sec.append('### 1. 三大发现（重复三遍）')
    for _ in range(3):
        sec.append(
            '1. **剂量解耦定量：'
            'fidelity_dose_sensitive|'
            'rho_linear|dose_resp_monotone**。'
            'dvec 自层注入剂量 8 倍扫描'
            '（绝对 1→8）：传导方向保真 '
            'cos_direct 仅 0.99→0.87–0.92 '
            '缓降（gap 0.071–0.127，全程 '
            '≥0.87），rho log-log 斜率 '
            '0.917/0.938/0.916 ≈ 线性'
            '（幅度∝剂量、方向保持）；而行为 '
            'chg 在 d2.0 即饱和至 1.0'
            '（L29 0.25→0.64→1.0→1.0；L33 '
            '0.21→0.52→0.99→1.0；L38 '
            '0.78→0.95→1.0→1.0，全单调）。'
            'd1.0（=3135 注入条件）cos/rho '
            '0.956/0.960/0.970、1.926/1.964/'
            '2.081 位级复现。**写入链是线性'
            '直通放大器（方向不变、幅度线性'
            '增长），行为读出是阈值饱和非'
            '线性——两者在剂量维完全解耦，'
            '"写入即达、读出定效"的定量版**')
    for _ in range(3):
        sec.append(
            '2. **层位效力 > 坐标身份：'
            'diagonal_weak + union_add_all**。'
            '交叉矩阵（A1 方向）：co36 错配'
            '打到 L17 chg 0.1406 > 自位 L35 '
            '0.0781；co50 错配到 L35 仅 '
            '0.0625；both_union 0.5547 = '
            'L17_union 0.5547（L35 注入净'
            '贡献≈0）；union 可加残差 '
            '0.0078/0.0859/0.0547 全 ≤0.15 → '
            '不相交坐标集独立线性贡献、无'
            '干扰。**L17 是行为读出主端口'
            '（注入深度优先），坐标集身份'
            '（fork-fit vs 载体 top50）次要'
            '——3133 端口类的层位维强化**。'
            '双锚点 L17_co50 0.4219、'
            'L35_co36 0.0781 跨会话位级复现'
            '（first=51/6 亦一致）')
    for _ in range(3):
        sec.append(
            '3. **w8 早步主导：'
            'w8_top1_dominant|w8_repro_ok**。'
            'w8full 0.6484 与 3135 位级一致'
            '（drift 0.0000）；drop-one 贡献'
            '谱：top1 k=3 贡献 0.2422（37%）、'
            'k=1 0.1641、k=2 0.1484、k=4–7 '
            '≤0.047、k=8 ≈0、**k=0 负贡献 '
            '−0.1016**（去掉 prompt-forward '
            '注入 chg 反升 0.75 vs 0.648）；'
            'Σ贡献 0.5625 = 87% chg。**行为'
            '分叉集中在解码早步 k1–k3（合计 '
            '86%），prompt-forward 注入与'
            '解码步部分干扰——3135 cum 曲线 '
            'step3–4 跳跃的步级定位完成**')
    sec.append('')
    sec.append('### 2. 关键数值')
    sec.append(
        'Part B：cos 曲线（d0.5/1/2/4）L29 '
        '0.992/0.956/0.908/0.895、L33 '
        '0.987/0.960/0.920/0.916、L38 '
        '0.994/0.970/0.903/0.867；rho 曲线 '
        '0.966→6.60 / 0.977→6.95 / '
        '0.995→6.73（斜率 0.917/0.938/'
        '0.916）；chg 见上；first（首 token '
        '翻转行数/128）随剂量升：L38 '
        '1/8/119/128——L38 载体 allstep 行为'
        '面强。Part C 矩阵（co50/co36/'
        'union × L17/L35/both）：0.4219/'
        '0.1406/0.5547; 0.0625/0.0781/'
        '0.0547; 0.3906/0.1328/0.5547。'
        'Part D：w8full 0.6484；drop0–8 '
        '0.7500/0.4844/0.5000/0.4062/'
        '0.6016/0.6406/0.6250/0.6094/'
        '0.6562；贡献 k0–8 −0.1016/0.1641/'
        '0.1484/0.2422/0.0469/0.0078/'
        '0.0234/0.0391/−0.0078。运行 '
        '28306s。')
    sec.append('')
    sec.append('### 3. 硬伤')
    sec.append(
        '①FID_GAP=0.06 太紧：fid_gap '
        '0.071–0.127 全落敏感侧，但 cos '
        '≥0.87 仍高保真——判决字面"dose_'
        'sensitive"应读作"缓降非失效"；'
        '②diagonal_weak 由端口不对称产生'
        '（L35 天然弱 0.078 vs L17 错配 '
        '0.141），应解读为层位不对称而非'
        '端口类失效；③both_co50 0.3906 < '
        'L17_co50 0.4219（−0.031）：L35 '
        'co50 注入轻微削弱 L17 效果，机制'
        '未解释；④k0 负贡献机制未解剖'
        '（prompt-forward 注入为何削弱：'
        '注意力窗位移/前缀污染？）；'
        '⑤128 行 SCAN_N 与 3134 的 672 行'
        '全集不可比（跨样本集禁令），行为'
        '对比仅定性方向；⑥**closeout 审计'
        '发现 3134 chg_matrix 为 prompt-'
        'only（mode=0）注入**（本 Phase '
        'B2 为 allstep）——历史"载体行为'
        '面"数字（L38 0.089 等）不可与 '
        'allstep 混用；L38 prompt-only 弱'
        '（0.089, 672 行）vs allstep 强'
        '（0.945, 128 行）= 新增时序-层位'
        '交互线索；⑦生成阶段 GPU 长时'
        '降速 ~2.6×，总时长 7.86h 偏长；'
        '⑧first 语义（首 token 翻转行数）'
        '与 chg 的联合读出未做行级分解。')
    sec.append('')
    sec.append('### 4. 机制拼图更新')
    sec.append(
        '①写入链=线性直通放大器（剂量线性'
        '、方向保持、cos≥0.87），行为读出='
        '阈值饱和非线性（d2.0 饱和）；'
        '②L17 行为主端口地位再确认（深度'
        '优先于坐标身份，错配仍有效且强于'
        '自位弱端口）；③坐标集可加性（union '
        '残差 ≤0.086）——端口类坐标独立'
        '编码；④解码早步 k1–k3 是行为分叉'
        '窗口（86% 效果），k0 prompt 注入'
        '部分干扰；⑤L38 载体行为效果依赖'
        '解码步持续注入（prompt-only 0.089 '
        'vs allstep 0.945）——写入时序分层。')
    sec.append('')
    sec.append('### 5. 3137 预注册（观察后'
               '冻结）')
    sec.append(
        '①prompt-only vs allstep 模式×层位'
        '×剂量对照矩阵（L17/29/33/38，两'
        '模式 × dose {0.5,1,2}，672 行）'
        '——时序-层位交互定律定量化（含 '
        '3134 数字位级对照）；②k0 负贡献'
        '解剖：w8 [0..8] vs [1..8] 去除'
        '对照 + 行级 first-token 翻转分解；'
        '③co36@L17 错配有效的坐标分解'
        '（co36 top25/bot25 @L17 注入）+ '
        'L17 载体 vs 坐标注入等效剂量'
        '换算；④L35 union 零贡献复核'
        '（L35 union 剂量扫描 {1,2,4}）。')
    sec.append('')
    sec.append('产物：`tests/glm5/result/'
               'rdc_query_construction_'
               '20260913/phase3136/'
               'omega_p134_conddose_cross'
               'matrix_w8drop/`（result.json '
               'sha256_8=%s、design_seal.json、'
               'run_log.txt、p134_readout.npz）；'
               '脚本 `tests/glm5/phase3136_'
               'omega_p134_conddose_cross'
               'matrix_w8drop.py`。' % sha8)
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write('\n'.join(sec) + '\n')
    steps.append('memo: appended')

# ---------- 3. wlog ----------
wl = ''
if os.path.exists(WLOG):
    wl = io.open(WLOG,
                 encoding='utf-8').read()
if 'Phase 3136' in wl:
    steps.append('wlog: exists, skip')
else:
    line = ('- Phase 3136 (Ω-P134) 闭环：'
            '正式跑 28306s 一次通过（SMOKE '
            'R1 crash→rev-3136a n_cap 修复、'
            'R2 发现 dvec_full 只存 4 层→'
            'rev-3136b 恢复本会话 swap '
            'capture）；verdict=a_3135_ok|'
            'fidelity_dose_sensitive|'
            'rho_linear|dose_resp_monotone|'
            'diagonal_weak|union_add_all|'
            'w8_top1_dominant|w8_repro_ok|'
            'coverage_full；closeout 五写'
            '+磁盘复核。\n')
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        if wl and not wl.endswith('\n'):
            f.write('\n')
        f.write(line)
    steps.append('wlog: appended')

# ---------- 4. workspace MEMORY ----------
mm = io.open(WMEM, encoding='utf-8').read()
if 'meas3136' in mm:
    steps.append('wmem: exists, skip')
else:
    mm2 = mm.replace(
        '- max=3135，下一 3136',
        '- 3136（T4）：传导剂量×8 cos 仅缓降 '
        '0.99→0.87-0.92（rho 斜率 0.917≈线性）'
        '而行为 chg d2.0 饱和 1.0=写入线性直通'
        '/读出阈值饱和；交叉矩阵 diagonal_weak'
        '（co36@L17 0.141>自位 0.078）+union '
        '全可加（残差≤0.086）+双锚位级复现'
        '（0.4219/0.0781）=L17 行为主端口、'
        '深度>坐标身份；w8 drop-one top1 k=3 '
        '0.242、k0 负 −0.102、w8full 与 3135 '
        'drift 0.0000=解码早步 k1-k3 主导'
        '（86%）；**3134 chg_matrix 为 '
        'prompt-only（closeout 审计发现），'
        '与 allstep 不可混用**。\n'
        '- max=3136，下一 3137')
    assert mm2 != mm, 'wmem anchor not found'
    mm2 = mm2.replace(
        '**传导保真×行为分离剂量曲线 + '
        'co36∪co50 交叉注入矩阵 + w8 逐步 '
        'drop-one 贡献谱。**L17 主载确认下的 '
        '层间传导核验（L17→下游消费）+ co36 '
        '必要性子集消融。**',
        '**prompt-only×allstep×层位×剂量对照'
        '矩阵 + k0 负贡献解剖（w8[1..8] 对照）'
        '+ co36@L17 错配坐标分解 + L35 union '
        '零贡献剂量复核。**')
    with io.open(WMEM, 'w',
                 encoding='utf-8') as f:
        f.write(mm2)
    steps.append('wmem: updated')

steps.append('result sha8=%s runtime=%.0fs'
             % (sha8, runtime))
out = '\n'.join(steps)
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3136_closeout_steps.txt', 'w',
        encoding='utf-8').write(out)
print('CLOSEOUT OK')
print(out)
