# -*- coding: utf-8 -*-
"""Phase 3145 closeout: five writes
(ledger 281->282 + MEMO append incl.
3146 prereg + workspace daily log +
MEMORY.md next-step) with idempotent
guards throughout."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D45 = os.path.join(
    RDIR, 'phase3145',
    'omega_p143_v1clip_pc1spec_'
    'headsign_residcausal')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')

# ---- load result + verify hashes ----
raw = io.open(os.path.join(D45,
                           'result.json'),
              'rb').read()
res_sha = hashlib.sha256(raw).hexdigest()[:8]
assert res_sha == '9d9cc6d1', res_sha
res = json.loads(raw.decode('utf-8'))
assert res['smoke'] is False
V = res['verdict']
EXP = ('a_3144_ok|repro_bit_5|'
       'repro_bit_ok|dvec19_repro_6a0332|'
       'field_self_ok|z35_cos17_ok|'
       'resid_anchor_ok|v1_amp_dv29dom|'
       'v1_causal_dirty|pc1_onset_l29|'
       'pc1_history_dependent|'
       'pc1_hist_lock|'
       'pc1_replay_readout_sim|'
       'head_sign_asym|tail_dose_mono|'
       'tail_sym_break|resid_causal_active|'
       'answer_side_yes|xphase_ok|'
       'coverage_full')
assert V == EXP, V
SEAL_SHA = str(res['seal_sha8'])
assert SEAL_SHA == '46c0187b', SEAL_SHA
assert os.path.exists(
    os.path.join(D45, 'design_seal.json'))
assert os.path.exists(
    os.path.join(D45, 'p143_readout.npz'))

# ---- 1. ledger ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
_n0 = len(led['measurements'])
assert _n0 in (281, 282), _n0
_has = any(m.get('phase') == 3145
           for m in led['measurements'])
if not _has:
    assert _n0 == 281, _n0
    entry = {
        'phase': 3145,
        'name': ('omega_p143_v1clip_'
                 'pc1spec_headsign_'
                 'residcausal'),
        'date': '2026-09-30',
        'kind': ('v1clip_pc1spec_'
                 'headsign_residcausal'),
        'verdict': V,
        'runtime_s': round(
            float(res['runtime_s']), 1),
        'hashes': {
            'result_sha256_8': res_sha,
            'seal_sha256_8': SEAL_SHA},
        'anchors': {
            'res44_sha8': 'def947d8',
            'seal44': 'bcd6fc5e',
            'dvec19_sha8': '6a0332a6',
            'xphase': 'P1.0/A1.1.0',
            'repro_bit': '5/5',
            'resid_anchor': 'share38 '
                            '0.6405/wdn39 '
                            '2.4955/ra_v1 '
                            '0.2287 exact',
            'pc1_sign': '+1'},
        'summary': (
            'v1 attribution: joint v1 '
            'amplification -59.1 is '
            'dominated by dv29 carrier '
            'own downstream dynamics '
            '(-67.0), pc1 contribution '
            '+7.9 (ratio 8.48) -> '
            '3144 finding refined; '
            'v1 clip: base+clip@38 '
            'chg 0.805 (>>0.05) -> v1 '
            'axis is behavioral load-'
            'bearing axis of generation, '
            'clip is non-specific '
            'destruction (dv29/joint '
            'both 0.961), causal test '
            'dirty-blocked; pc1 '
            'spectrum: 131401 dlogit '
            'onset L29 1.20 -> L39 '
            '2.83 (2x jump at L39); '
            'instant 0.0625 vs allstep '
            '0.2031 (ratio 0.31) -> '
            'pc1 interference is step-'
            'wise cumulative; replay '
            'lock 0.875 readout sim; '
            'head sign: pos 0.1406 vs '
            'neg 0.2188 -> directional '
            'asym; tail dose MONOTONE '
            '0.094->0.172(d2)->0.305(d4) '
            '+ sym break at d2 (pos '
            '0.258 vs neg 0.172) -> '
            'tail carries weak '
            'directional signal (3144 '
            'noise verdict refined); '
            'resid top-PC @L19 saturated '
            '1.0 both signs, @L38 pos '
            '0.984 / neg 0.422 -> '
            'directionally differentiated '
            'behavioral sufficiency; '
            'perp39 dyes +1.120 / dno '
            '-1.154 -> yes side.')}
    led['measurements'].append(entry)
    io.open(LEDGER, 'w',
            encoding='utf-8').write(
        json.dumps(led, ensure_ascii=False,
                   indent=1))
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
assert len(led2['measurements']) == 282
assert led2['measurements'][-1][
    'phase'] == 3145
print('1. ledger n=282 OK')

# ---- 2. MEMO append ----
block = '''
## Phase 3145: v1clip承重+pc1累积+尾部剂量+残差因果（T4 第28 Phase）[02:00]

**执行**：phase3145_omega_p143_v1clip_pc1spec_headsign_residcausal.py；SMOKE 两轮（rev-3145a patch1 误删 _rows 定义 NameError→patch2 修复；SMOKE 377s 全链贯通）→ 正式跑一次通过 12906s。设计四组：①v1 放大归因（3144 traj 锚分解）+v1 clip 因果（joint×CLIP_L{30,33,36,38,39}+base/dv29+clip@38 对照）②pc1 token 逐层谱（16 行 teacher-forced L20-39 base/inj）+instant（mode 0）vs allstep+pc1 历史重放 ③d_headpos（HEAD25 sgn+1）+tail 剂量 d{2,4}+tailpos_d2+d_head/d_gbot bit 重放 ④perp38 top-PC @L19/@L38 ±注入+perp39 yes/no 分解。

**锚链**：PART A 3144 sha8=def947d8/seal=bcd6fc5e 全数值断言通过（traj/tokdec/co36sign/resid 锚逐一）；dvec19 重捕获 sha8=6a0332a6（drift 0.00e+00，**第 3 次跨会话位级**）；field 自洽 cos 双 1.0000；xphase P/A1=1.0（128/128）；z35 软锚 0.0102；**C4 resid 锚全命中**（share38 0.6405/wdn_p39 2.4955/ra_v1 0.2287/pnorm38 55.263 与 3144 完全一致）；**bit 锚 5/5**（b_pc1 0.203125 + b_dvec29 0.640625 + b_joint 0.4765625 第 3 次复现 + d_head 0.21875 + d_gbot 0.1171875）。

### §1 发现1：v1 放大归因修正——dv29 载体自身动态主导（×3 强调）
- B2 归因分解（3144 traj 锚纯计算）：amp_joint −59.12、amp_dv29 **−67.02**、pc1 贡献 **+7.90**（ratio |dv29/pc1c|=8.48）→ **v1_amp_dv29dom：joint 的 v1 下游放大（6.7×）主体是 dv29 载体自身的下游动态，pc1 分量在 v1 维度是抵消方向（+10.48→+18.38）**——3144"pc1 分量下游放大 6.7×"的表述需修正：放大的是 joint 总 v1，其载体是 dv29 分量。

### §2 发现2：v1 轴=生成行为承重轴，clip 干预不干净（×3 强调）
- v1 clip 谱（joint）：L30 0.664/L33 0.727/L36 0.789/L38 0.961/L39 0.656；对照 base+clip@38 **0.8047**（>>0.05 dirty gate）、dv29+clip@38 0.9609≈joint+clip@38 0.9609。
- 判决 **v1_causal_dirty**：投影去除 v1 分量本身使 base 生成 80% 行为破坏（first=6 中位）——**v1 轴（L29 WR 主 PC）是生成过程的行为承重轴，任何条件下去除都崩塌**；dv29/joint clip 后相同（0.961）是共同崩塌而非恢复——形态支配假说无法用全 v1 clip 检验（dirty gate 正确拦截假阳性 shrink=1.0）。

### §3 发现3：pc1 干扰=逐步累积竞争；token 抬升从注入层即开始（×3 强调）
- T1：instant（仅 prompt forward 注入）chg 0.0625 vs allstep 0.2031（ratio 0.308）→ **pc1_history_dependent：行为级干扰需要每步重新注入维持**，单步注入几乎无行为效应（tokcls other 7/format 1/answer 0/none 120）。
- T2 token 谱（16 行 base-hist teacher-forced）：131401 dlogit L20-28 全 0.000 → L29 1.201 → L38 1.474 → **L39 2.832**（L39 跳升 1.9×）；134772 同构（L29 1.110→L39 2.408）→ **pc1_onset_l29**：干扰 token 抬升是注入层直接效应，下游单调累积，L39 最后一层再跳升。
- T3 pc1 历史重放：top1d in-spec 0.875（base-hist 同 0.875）、med dh.wdn39 0.746=0.746 → hist_lock+readout_sim：teacher-forced 单步下干扰结构稳定，与 T1 行为级"瞬时无效应"合起来=**干扰是跨步累积过程，非单步竞争翻转**（解释 3144 dlogit(t*)>0 但 argmax 翻转的悖论：单步 logit 差 +0.596 不足以翻转，累积后才翻转）。

### §4 发现4：尾部坐标剂量单调+符号破缺=弱方向性信号（修正 3144；×3 强调）
- D 矩阵（A1，L17）：head 0.2188（sgn−1，bit）/ headpos 0.1406（sgn+1）→ **head_sign_asym**（头部负向效应强于正向 1.56×，方向特异确认）；tail d1 0.0938 → d2 0.1719 → d4 0.3047 **单调**；tailpos_d2 0.2578 vs tail_d2 0.1719（|Δ|=0.086>0.03）→ **tail_sym_break**。
- 判决 **tail_dose_mono**：3144"尾部=符号对称噪声"在剂量维度被修正——尾部坐标携带真实方向性行为信号，但弱于头部（tail 需 2× 剂量 0.172 才接近 head 1× 的 0.219；d4 0.305 反超）；低剂量下方向分量低于噪声底故表现"符号对称"。
- bit 5/5：d_head 0.21875 + d_gbot 0.1171875 现场重放位级命中（3143/3137 锚第 2 次跨会话）。

### §5 发现5：残差 top-PC 行为充分+方向区分；perp39 写 yes 侧（×3 强调）
- E 注入（P 行，幅度 perp_norm38=55.26）：@L19 pos/neg 双 **1.0000**（128 行全 first-step 翻转=饱和）；@L38 pos **0.9844** / neg **0.4219** → **resid_causal_active**：L38 注入保持方向区分度（正向几乎全破坏、负向 42%——非对称翻转）。
- perp39 yes/no 分解：dyes **+1.120** / dno **−1.154**（diff 2.274 vs perp_wdn39 2.4955 自洽）→ **answer_side_yes**：dh19 独立残差成分在 L39 把读出推离 no、推向 yes——结合 3144 perp@wdn L39 2.50 主导：残差方向行为上可注入、方向上区分、读出上写 yes 侧。
- 判决链完整：L19 独立成分（~2/3 能量）→ 沿残差 top-PC 方向 → 行为充分（可注入）→ 写 yes 读出侧。

### §6 综合 + 3146 预注册
3145 五问五答：(1) v1 放大=dv29 载体自身动态（pc1 抵消方向，ratio 8.48）；(2) v1 轴=行为承重轴，clip dirty（base 0.805）；(3) pc1 干扰=逐步累积（instant ratio 0.31），token 抬升 onset L29、L39 跳升；(4) 头部方向特异（asym 1.56×），尾部剂量单调+sym_break=弱方向信号；(5) 残差 top-PC 行为充分+方向区分（l38 0.984/0.422）+yes 侧。

3146（Ω-P144）预注册：
1. 尾部信号定位：d4 剂量下 tail 坐标细分注入（TAIL25 内 top10/bottom15 by 富集分）+ head/tail/gbot/co36full/co50ex 完整剂量矩阵 d{1,2,4}——定位尾部方向信号的坐标来源与剂量响应形状；
2. pc1 累积层位解剖：allstep 注入下逐层 capture 谱（对照瞬时单步）——定位干扰 token 抬升的下游放大层（L39 跳升 1.9× 的来源层）+ L39 跳升与 final norm 层的关系；
3. resid 因果剂量曲线：pcres @L38 剂量 {0.25,0.5,1.0}×perp_norm38（脱离饱和区）——方向区分度（pos−neg gap）随剂量的演化+neg 0.4219 非饱和区的翻转结构（first-step 位置分布）；
4. v1 承重轴干净干预：部分幅度 clip（h−=α(h·v̂)v̂，α∈{0.25,0.5}）+ 仅 decode 步 clip（prompt forward 保留）——找 non-dirty 干预窗口后重测 joint/dv29 行为差距（形态支配假说的二阶检验）。

关键数字：repro 5/5；amp_joint −59.12/amp_dv29 −67.02/pc1c +7.90（ratio 8.48）；bc@38 0.8047/dc 0.9609/jc38 0.9609；instant 0.0625/allstep 0.2031（ratio 0.308）；131401 谱 L29 1.201→L39 2.832；head 0.2188/headpos 0.1406；tail 0.0938→0.1719(d2)→0.3047(d4)/tailpos_d2 0.2578；l19 双 1.0/l38 0.9844/0.4219；dyes +1.120/dno −1.154。

锚：result sha8=9d9cc6d1，seal sha8=46c0187b，dvec19 sha8=6a0332a6，ledger n=282。
'''
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3145' not in t:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(block)
t2 = io.open(MEMO, encoding='utf-8').read()
assert t2.count('## Phase 3145') == 1
assert '3146（Ω-P144）预注册' in t2
print('2. MEMO appended OK')

# ---- 3. workspace daily log ----
p_log = (ROOT + '\\' + '.workbuddy'
         + '\\' + 'memory' + '\\'
         + '2026-09-30.md')
add = '''
## Phase 3145 (Omega-P143) 闭环 [02:00]
- 正式跑一次通过 12906s（SMOKE 377s；rev-3145a：patch1 误删 _rows 定义→NameError，patch2 修复）。verdict: a_3144_ok|repro_bit_5|repro_bit_ok|dvec19_repro_6a0332|field_self_ok|z35_cos17_ok|resid_anchor_ok|v1_amp_dv29dom|v1_causal_dirty|pc1_onset_l29|pc1_history_dependent|pc1_hist_lock|pc1_replay_readout_sim|head_sign_asym|tail_dose_mono|tail_sym_break|resid_causal_active|answer_side_yes|xphase_ok|coverage_full。
- 五发现：①v1 放大归因修正——dv29 载体自身动态主导（−67.0），pc1 贡献 +7.9 抵消方向（ratio 8.48）②v1 轴=行为承重轴（base+clip 0.805 破坏），clip dirty，形态支配假说无法全 clip 检验 ③pc1 干扰=逐步累积（instant 0.0625 vs allstep 0.2031），131401 谱 onset L29→L39 跳升 2.83 ④头部方向特异（0.2188 vs 0.1406）；尾部剂量单调 0.094→0.172→0.305+sym_break=弱方向信号（修正 3144 噪声结论）⑤resid top-PC 行为充分+l38 方向区分（0.984/0.422）+perp39 写 yes 侧（dyes +1.120/dno −1.154）。
- bit 5/5（三 3142 锚第 3 次+d_head/d_gbot 第 2 次）；resid 锚 4/4 精确；dvec19 sha 6a0332a6 drift 0（第 3 次）。
- closeout：ledger n=282（res sha8=9d9cc6d1、seal 46c0187b）+ MEMO（T4 第28Phase，含 3146 预注册）+ 本日志 + MEMORY。'''
if os.path.exists(p_log):
    t = io.open(p_log, encoding='utf-8').read()
else:
    t = ''
    os.makedirs(os.path.dirname(p_log),
                exist_ok=True)
if 'Phase 3145 (Omega-P143) 闭环' not in t:
    io.open(p_log, 'w',
            encoding='utf-8').write(t + add)
t2 = io.open(p_log, encoding='utf-8').read()
assert 'Phase 3145 (Omega-P143) 闭环' in t2
print('3. daily log OK')

# ---- 4. MEMORY.md next-step line ----
p_mem = (ROOT + '\\' + '.workbuddy'
         + '\\' + 'memory' + '\\'
         + 'MEMORY.md')
tm = io.open(p_mem, encoding='utf-8').read()
lines = tm.splitlines()
out_lines = []
changed = False
NEWLINE = ('- max=3145，下一 3146：①尾部信号定位（TAIL25 内富集 top10/bottom15 细分+d{1,2,4} 完整剂量矩阵）②pc1 累积层位解剖（allstep 逐层 capture 谱+L39 跳升来源层）③resid 因果剂量曲线（@L38 d{0.25,0.5,1.0}×pnorm38 脱饱和+方向区分度演化）④v1 承重轴干净干预（部分幅度 α clip+仅 decode 步 clip 后重测 joint/dv29 差距）。**v1 放大=dv29 载体自身动态（pc1c +7.9 抵消，ratio 8.48）；v1 轴=行为承重轴（base+clip 0.805 dirty）；pc1 干扰=逐步累积（instant ratio 0.31，onset L29→L39 跳升 2.83）；头部方向特异+尾部剂量单调 sym_break（弱方向信号，修正 3144）；resid top-PC 行为充分+l38 方向区分 0.984/0.422+answer_side_yes（dyes +1.120/dno −1.154）。**')
for l in lines:
    if l.startswith('- max=3144，下一 3145'):
        out_lines.append(NEWLINE)
        changed = True
    else:
        out_lines.append(l)
if changed:
    io.open(p_mem, 'w',
            encoding='utf-8').write(
        chr(10).join(out_lines) + chr(10))
tm2 = io.open(p_mem, encoding='utf-8').read()
assert 'max=3145' in tm2
assert 'max=3144，下一' not in tm2
print('4. MEMORY.md OK (changed=%s)'
      % changed)
print('CLOSEOUT ALL DONE')
