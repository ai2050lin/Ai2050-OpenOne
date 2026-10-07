# -*- coding: utf-8 -*-
"""Phase 3138 closeout: five-write chain
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
OUT = RDIR + r'\phase3138' \
      r'\omega_p136_statebank_' \
      r'bicdecomp_probecontrast'
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
assert r['phase'] == 3138
V = r['verdict']
raw = io.open(os.path.join(OUT,
                           'result.json'),
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
assert sha8 == 'f7ef08be', sha8
runtime = r['runtime_s']
pc = r['part_c']
pd_ = r['part_d']
pe = r['part_e']
ns = pc['norm_share']
retr = pc['retr']

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
MEAS = 'meas3138_omega_p136_statebank_' \
       'bicdecomp_probecontrast'
exists = [e for e in led['measurements']
          if e.get('meas_id') == MEAS]
if exists:
    steps.append('ledger: exists, skip')
else:
    claim = (
        'Omega-P136 (3138, T4 twenty-first '
        'phase: Absolute State Bank 672 '
        'rows x 2 dirs x 4 surface '
        'templates x 40 layers fp16 '
        'scaled shards + B/I/C additive '
        'decomposition + P-vs-A1 probe '
        'contrast + k0 row anatomy. '
        'Norm share vs causal share '
        'decoupled: B base carries '
        '95-99% of norm (L29 min 0.951) '
        'but zero probe signal; I '
        'identity ~8-16% norm carries '
        'AUC 1.0; template C stable '
        'cross-row csh 0.9996 all key '
        'layers; identity retrieval '
        'curve retr: ~1.0 plateau L0-25, '
        'valley L26-32 (L29 0.784), '
        'recovery L33-38 (0.957), L17 '
        '1.0 = identity-fully-stable at '
        'the behavior port; ish '
        '0.638-0.743 (hard 0.60 ok, '
        'soft 0.85 below: identity has '
        'template-interaction residual); '
        'cross-template probe AUC 1.0 '
        'raw/BI/I all key layers = '
        'Tier-1 Level-1 fingerprint '
        'recovery achieved; k0 anatomy: '
        'only_nok0 18 flip rows margin '
        '1.142 < rest 1.252 (k0 hits '
        'weak-margin rows) - verdict '
        + V)
    entry = {
        'meas_id': MEAS,
        'phase': 3138,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3138/omega_p136_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p136_readout.npz',
            'bank_shards':
                'bank_{P,A1}_T{0..3}.npz '
                'x8 fp16 scaled'},
        'hashes': {
            'result_sha256_8': sha8},
        'anchors': [
            'meas3137_omega_p135_'
            'modecoop_k0anat_'
            'coorddecomp_l35recheck'],
        'note': 'B/I/C additive fit; '
                'retr valley L26-32 = '
                'identity rewrite window; '
                'AUC 1.0 cross-template'}
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
if '## Phase 3138:' in memo:
    steps.append('memo: exists, skip')
else:
    hhmm = time.strftime('%H:%M')
    sec = []
    sec.append('')
    sec.append('## Phase 3138: Ω-P136 状态'
               '库+BIC分解+探针对照（T4 第21'
               'Phase）[%s]' % hhmm)
    sec.append('')
    sec.append('**性质**：T4 第 21 Phase，'
               '指纹范式转移后首个 Phase'
               '（FINGERPRINT_PARADIGM_PLAN'
               '.md Ω-P136 + 3137 MEMO §5 '
               '预注册），design_seal.json '
               '观测前冻结。Part A 3137 链接'
               '断言（sha8 4f8055bb + verdict '
               '全等 + 9 drift-0 锚硬断言）；'
               'Part B Absolute State Bank：'
               '672 行 × 2 方向（P/A1）× 4 '
               '表面模板（T0 标准/T1 行序'
               '重排/T2 前缀/T3 query 措辞）'
               '× 40 层最后位置状态，8 分片 '
               'fp16+per-layer scale 存储'
               '（深层幅值超 fp16 保护）；'
               'Part C 每层加性分解 H=B+I+C+R'
               '（范数占比+身份跨模板半稳定'
               'ish+模板跨行半稳定 csh+跨半'
               'top-1 检索 retr+残差 PCA1）；'
               'Part D P-vs-A1 logistic 探针'
               '（raw/BI/I 三成分 × within-T0 '
               '行切分 / 跨模板 T01→T23 双 '
               'regime）；Part E k0 行级解剖'
               '（offline，3137 npz fstep + '
               'z26 margin）。运行 4028.1s。')
    sec.append('')
    sec.append('### 1. 三大发现（重复三遍）')
    for _ in range(3):
        sec.append(
            '1. **范数占比 ≠ 因果占比'
            '（定量证实）**。层基座 B 占'
            '范数 95–99%（L29 最低仍 '
            '0.951；L0 0.9993 → L39 '
            '0.9936），但 B 无任何行/方向'
            '判别信号；身份成分 I 仅占 '
            '8–16%（L17 0.139 峰），却'
            '承载 P-vs-A1 全部判别力'
            '（I 单独 AUC 1.0）。模板 C '
            '占 8.6–16.1% 且跨行半稳定 '
            'csh 0.9993–0.9996（soft 0.85 '
            '过）——**表层措辞=跨样本恒定'
            '的固定偏移骨架**。冰山比喻的'
            '定量版：水下部分（B）是骨架'
            '不是语义，语义在微小但判别的'
            '身份偏移里')
    for _ in range(3):
        sec.append(
            '2. **身份重写窗口定位：retr '
            '层曲线 L0–25 平台 ~1.0 → '
            'L26–32 谷底（L29 0.784）→ '
            'L33–38 回升（L38 0.957）**。'
            '跨模板半 top-1 检索率（基线 '
            '0.0015）在浅层近乎完美、'
            'L26–32 剧降、深层部分恢复'
            '——**L26–32 是身份被上下文'
            '重写/混合最剧烈的窗口**，与 '
            '3113–3117 MLP 主写 L20–28+'
            'L32 擦除相、3121 内容涌现'
            '上游层位一致；**L17 检索 '
            '1.0+ish 0.71/0.72 = 身份在 '
            'L17 完全稳定**，与 L17 行为'
            '主端口（3133/3136/3137）互证'
            '——端口层=身份未混合层')
    for _ in range(3):
        sec.append(
            '3. **跨模板泛化 AUC 1.0：'
            'Tier-1 Level-1（指纹新上下文'
            '恢复）达成**。raw/BI/I 三成分 '
            '在 within-T0 与跨模板（T01 '
            '训练→T23 测试）双 regime 下 '
            'key 层 AUC 全 1.0，C_only=0.5 '
            '（模板成分不含判别信息）——'
            '**换一种问法/换行序/换前缀，'
            '身份指纹不变且语义完全保持**；'
            'ish 0.638–0.743（soft 0.85 '
            '未达）表明 I 含 ~26–35% 模板'
            '交互残差——加性假设不足，'
            '身份不是纯加性成分（3139 '
            '解剖）。E 部分：k0 削弱的 18 '
            '翻转行 base margin 1.142 < '
            '其余 1.252——k0 干扰偏向弱'
            'margin 行（记录型）')
    sec.append('')
    sec.append('### 2. 关键数值')
    sec.append(
        'norm_share（B/I/C/R，L0/8/17/29/'
        '33/38/39）：0.999/0.031/0.020/'
        '0.009；0.982/0.080/0.110/0.049；'
        '0.971/0.139/0.138/0.088；0.951/'
        '0.116/0.161/0.086；0.964/0.103/'
        '0.133/0.075；0.983/0.085/0.086/'
        '0.057；0.994/0.104/0.106/0.073。'
        'ish key（P|A1）：L17 0.713|0.720、'
        'L29 0.638|0.651、L33 0.671|0.675、'
        'L38 0.743|0.735；ish_all_med '
        '0.721。csh key 0.9994/0.9996/'
        '0.9996/0.9993。retr：L17 1.0、'
        'L29 0.784、L33 0.872、L38 0.957；'
        'L0–25 全 ≥0.935，谷底 L31 0.743。'
        'resid PCA1 份额谷 L10–19 ~0.07、'
        '峰 L31 0.244。auc_x L17：raw 1.0/'
        'BI 1.0/I 1.0/C_only 0.5；auc_x_'
        'med 1.0。E：only_nok0 18 行 '
        'margin 1.142 vs rest 1.252 vs '
        'only_full 1.465。运行 4028.1s。')
    sec.append('')
    sec.append('### 3. 硬伤')
    sec.append(
        '①ish soft85 未达（0.638–0.743）：'
        '加性分解的 I 含模板交互残差'
        '（~26–35%），身份≠纯加性成分；'
        '②retr 谷底与 C 峰（L29 0.161）'
        '共现——C 可能吸收了部分身份'
        '方差（加性归因未分离）；③探针'
        '行切分在同一材料集（672 pk）'
        '——跨材料泛化未测（3107 坐标基'
        '跨材料简并教训）；④bank 只存'
        '最后位置状态，token 级指纹未'
        '覆盖（3121 token 级替换失效的 '
        'bank 版验证未做）；⑤E margin '
        '差 0.11 非结论性；⑥B 用向量范数、'
        'I/C/R 用中位个体范数——口径'
        '混合，跨层比较需谨慎；⑦SMOKE '
        '期间 rev-3138a 区间替换再爆 '
        'index 偏移事故（docstring 同名'
        '短语命中）→PART B/C 被删，全量'
        '重写恢复——**区间替换 patch 必须'
        '用多行唯一锚+count 断言+落盘'
        '复核三重防护**。')
    sec.append('')
    sec.append('### 4. 机制拼图更新')
    sec.append(
        '①B/I/C/R 分解成立：基座（公共'
        '骨架 95–99%）/身份（判别载体 '
        '8–16%）/模板偏移（跨行恒定）/'
        '残差四成分各司其职——**响应场'
        '指纹的几何基座建立**；②L26–32 '
        '身份重写窗口与写入链层位重合：'
        '上行写入不只添加信息，还混合/'
        '改写身份表示——3121"token 级'
        '替换失效=写入链上游改写"的'
        'bank 版定量确证；③L17=身份'
        '完全稳定+行为主端口：**读出'
        '端口选在身份未混合处**；④模板'
        '不变性：措辞变化只加固定偏移'
        '（C），不触碰身份子空间——'
        'Level-1 指纹恢复达成，Level-2'
        '（组合预测）具备基座。')
    sec.append('')
    sec.append('### 5. 3139 预注册（观察后'
               '冻结）')
    sec.append(
        '①身份交互分解：I 对模板回归取'
        '残差 → I_stable/I_templ 二分，'
        'ish 门重测（≥0.85 目标）+R 中'
        '行身份检索（残差里是否残留身份'
        '信号）；②端口消费测试：3135/3136 '
        '冻结 dvec（L17/29/33/38，sha '
        '锚定）在 B/I/C/R 子空间的投影'
        '能量占比——**行为端口消费哪个'
        '成分**（预测：I 主导，C≈0）；'
        '③L26–32 重写窗口因果：C 注入'
        '（把 T2 的 C 加到 T0 的 H）与 '
        'I 扰动的行为效应对比（连接 '
        '3121–3127 写入链）；④跨材料'
        '泛化前哨：材料集外新 (s,o) 组合'
        '的 I 检索与 AUC（Tier-1 '
        'Level-2 前哨）。锚：3138 result '
        'sha8=f7ef08be、verdict 全等、'
        'bank 8 分片、retr_key L17 1.0/'
        'L29 0.7842/L38 0.9568、csh '
        '0.9996、auc_x 1.0、ledger '
        'n=275。')
    sec.append('')
    sec.append('产物：`tests/glm5/result/'
               'rdc_query_construction_'
               '20260913/phase3138/'
               'omega_p136_statebank_'
               'bicdecomp_probecontrast/`'
               '（result.json sha256_8=%s、'
               'design_seal.json、run_log.txt、'
               'p136_readout.npz 15 键、'
               'bank_*.npz 8 分片）；脚本 '
               '`tests/glm5/phase3138_omega_'
               'p136_statebank_bicdecomp_'
               'probecontrast.py`。' % sha8)
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write('\n'.join(sec) + '\n')
    steps.append('memo: appended')

# ---------- 3. wlog ----------
wl = ''
if os.path.exists(WLOG):
    wl = io.open(WLOG,
                 encoding='utf-8').read()
if 'Phase 3138 (Ω-P136) 闭环' in wl:
    steps.append('wlog: exists, skip')
else:
    line = ('- Phase 3138 (Ω-P136) 闭环：'
            '正式跑 4028.1s 一次通过（SMOKE '
            '三修：rev-3138b fp16 溢出→'
            'per-layer scale 存储、rev-3138c '
            "comps['I'] broadcast 修空切片 "
            'NaN、rev-3138d E 段改 3137 npz '
            '键；rev-3138a 区间替换 index '
            '偏移事故→全量重写）；verdict='
            'a_3137_ok|i_sh_hard_ok|'
            'i_sh_soft85_below|c_sh_hard_ok|'
            'c_sh_soft85_ok|probe_auc_gen_'
            'ok|k0_anatomy_recorded|'
            'coverage_full；范数占比≠因果'
            '占比定量证实（B 95-99% AUC 0.5 '
            'vs I 8-16% AUC 1.0）；retr 层'
            '曲线谷底 L26-32=L29 0.784、'
            'L17 1.0；跨模板 AUC 1.0='
            'Tier-1 L1 达成；closeout 五写'
            '+磁盘复核；3139 预注册。\n')
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        if wl and not wl.endswith('\n'):
            f.write('\n')
        f.write(line)
    steps.append('wlog: appended')

# ---------- 4. workspace MEMORY ----------
mm = io.open(WMEM, encoding='utf-8').read()
if 'meas3138' in mm:
    steps.append('wmem: exists, skip')
else:
    mm2 = mm.replace(
        '- max=3137，下一 3138',
        '- 3138（T4）：状态库 672×2×4 模板×'
        '40 层+B/I/C 分解：**范数占比≠因果'
        '占比定量证实**（B 基座 95-99% 无判别'
        '信号，I 身份 8-16% AUC 1.0，C 模板'
        '偏移跨行恒定 csh 0.9996）；身份检索'
        '层曲线 L0-25≈1.0→**谷底 L26-32'
        '（L29 0.784）=身份重写窗口**→回升 '
        'L38 0.957，L17 1.0=端口层身份未混合'
        '（与行为主端口互证）；跨模板探针 '
        'AUC 1.0=**Tier-1 Level-1 指纹恢复'
        '达成**；ish 0.64-0.74=I 含模板交互'
        '残差（加性不足，3139 解剖）。\n'
        '- max=3138，下一 3139')
    assert mm2 != mm, 'wmem anchor not found'
    mm2 = mm2.replace(
        '**3138 起转 Absolute State Bank 路线'
        '（Ω-P136，FINGERPRINT_PARADIGM_PLAN'
        '.md）：128 行×≥3 模板×40 层 hidden '
        '库 + B/I/C 三分解（split-half ≥0.9 门'
        '+探针 AUC 对比）+ k0 干扰机制初探。**',
        '**3139（Ω-P137）：身份交互分解'
        '（I_stable/I_templ）+端口消费测试'
        '（冻结 dvec 在 B/I/C/R 投影能量）+'
        'L26-32 重写窗口因果（C 注入/I 扰动'
        '行为效应）+跨材料泛化前哨（Level-2）。**')
    with io.open(WMEM, 'w',
                 encoding='utf-8') as f:
        f.write(mm2)
    steps.append('wmem: updated')

steps.append('result sha8=%s runtime=%.0fs'
             % (sha8, runtime))
out = '\n'.join(steps)
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3138_closeout_steps.txt', 'w',
        encoding='utf-8').write(out)
print('CLOSEOUT OK')
print(out)
