# -*- coding: utf-8 -*-
"""Phase 3081 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3081'
     r'\omega_p78_ds7b_crossmodel')
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
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['verdict']
assert verdict == 'ds7b_cos_absent', verdict
assert res['forwards'] == 20634
# (3081 result.json has no 'smoke' key; the
# authoritative run identity is verified via
# execution.json created + seal hashes below.)
st = res['stats']
fam = st['families']
for fk in ('A', 'B', 'C'):
    ba = fam[fk]['b_anchors']
    okkeys = [k for k in ba
              if k.endswith('_ok')]
    assert okkeys and all(
        ba[k] is True for k in okkeys), fk
    assert fam[fk]['top8_sel_ok'] is True, fk
cr = st['cross']
g = res['gates']
assert g['G_DS'] is False
assert g['count_sig_pos'] == 1
assert g['min_sp'] < 0
assert g['f2_TAB_p'] > 0.05
assert g['G1'] is False
mig = cr['mig']
assert abs(mig['AB']) < 0.25
assert abs(mig['AC']) < 0.25
assert abs(mig['BC']) < 0.25
e3 = cr['e3']
f2uab = e3['f2_cTT~U_AB']
assert f2uab['sp'] > 0.5 and f2uab['p'] < 0.01
el = res['elapsed']

# Per-pair T/U migration arrays live in the npz
# (result.json keeps only aggregates); SMOKE flag
# is also an npz scalar (authoritative run = 0).
import numpy as np
_Z = np.load(R + r'\omega_p78_ds7b_crossmodel.npz',
             allow_pickle=True)
assert bool(int(_Z['SMOKE'])) is False
assert int(_Z['FORWARDS']) == 20634
assert bool(int(_Z['SETUP_OK'])) is True
assert bool(int(_Z['TOP8_ALL_OK'])) is True


def np_med(cr, rn, key):
    return float(np.median(
        _Z[rn + '_' + key]))


claim = (
    'Omega-P78 (plan 3081 A) - DS7B (deepseek-'
    'r1-distill-qwen-7b, Qwen2 28L GQA 28h/4kv '
    '3584 bf16 15.23 GB) cross-model '
    'replication of the 3079/3080 TT-migration '
    'and cos-lock results; full 3076-protocol '
    'pipeline minus E2H/lens (L_INJ=25, '
    'L_POST=26; no W32 - 16 GB GPU), 20634 '
    'forwards, %.0fs.  PREREGISTERED HONEST '
    'INDEPENDENT TEST of ab_cos_locked: no '
    'DS7B activation data observed before the '
    'run; gate G_DS fixed in the prereg.  '
    'VERDICT ds7b_cos_absent.  (1) MIGRATION '
    'ITSELF IS ABSENT: mig = sp(R1_f, R1_g) '
    'over 28 heads = AB %.4f / AC %.4f / BC '
    '%.4f (qwen3-4b: +0.676/+0.130/-0.088); '
    'per-pair T/U medians all ~0 (T '
    '%+.3f/%+.3f/%+.3f, U %+.3f/%+.3f/%+.3f); '
    'focal top8 cross-family overlap 2/1/2 of '
    '8 (random expectation ~2.3); sp(U,T) = '
    '%.3f/%.3f/%.3f (4B: 0.690/0.487/-0.099) '
    '- there is NO cross-family routing '
    'migration on DS7B to lock.  (2) GATE '
    'FAILS AS PREREGISTERED: G_DS 1/6 '
    'significant (min sp %.4f < 0, Stouffer '
    'z=%.3f); f2~T_AB %+.4f (p=%.3f); G1 (f1 '
    'control) also fails (best %+.4f, p=%.3f) '
    '- neither rank shape nor angle carries '
    'a signal that is not there.  (3) THE '
    'PIPELINE IS SENSITIVE (not a '
    'measurement failure): within-family '
    'focal structure exists - r1 min '
    '-0.465/-0.682/-0.344, n_neg 27/21/26, '
    'capture8 0.566/0.793/0.633, full 28-'
    'head swap R_ALL -0.247/-1.277/-0.340, '
    'med_c 0.6996/0.7468/0.4935; all '
    'b-anchors (b0/b1/b4/b6/b7a/b8) bit 0.0 '
    'in all three families.  (4) RESIDUAL '
    'CONDITIONAL STRUCTURE (exploratory, '
    'ungated): f2~U_AB %+.4f (p=%.5f) is '
    'the only significant f2 test - within '
    'AB, pairs with more aligned TT '
    'directions transfer subset-structure '
    'better even though the average U_AB '
    'is ~0; partial sp(f2|f1) U_AB %+.3f '
    '(p=%.4f); f4~T_AB %.4f (p=%.3f) and '
    'f4~T_AC %.4f (p=%.3f) are NEGATIVE '
    '(2 of 30 tests at p<0.05, within '
    'multiple-comparison noise).  THEORY '
    'UPDATE: the 3080 angle-matching '
    'assembly rule is a MODEL-INTERNAL '
    'regularity of qwen3-4b, not a '
    'model-general property - cross-context '
    'gear reuse is a training-product '
    '(instruct style-prefix handling), not '
    'a Transformer universal.'
    % (el, mig['AB'], mig['AC'], mig['BC'],
       np_med(cr, 'T', 'AB'),
       np_med(cr, 'T', 'AC'),
       np_med(cr, 'T', 'BC'),
       np_med(cr, 'U', 'AB'),
       np_med(cr, 'U', 'AC'),
       np_med(cr, 'U', 'BC'),
       cr['sp_ut']['AB'], cr['sp_ut']['AC'],
       cr['sp_ut']['BC'], g['min_sp'],
       g['stouffer_z'], g['f2_TAB_sp'],
       g['f2_TAB_p'], g['G1_sp'], g['G1_p'],
       f2uab['sp'], f2uab['p'],
       cr['partial']['f2g1_U_AB'],
       cr['partial']['f2g1p_U_AB'],
       e3['f4_sLGb~T_AB']['sp'],
       e3['f4_sLGb~T_AB']['p'],
       e3['f4_sLGb~T_AC']['sp'],
       e3['f4_sLGb~T_AC']['p']))

meas = {
    'meas_id': 'meas3081_omega_p78_'
               'ds7b_crossmodel',
    'phase': 3081,
    'claim': claim,
    'verdict': verdict,
    'anchors': 'setup anchors all bit 0.0 in '
               'all 3 families: b0 bank '
               'recapture, b1 sham self-V '
               'replacement, b3 finite, b4 '
               'delta-x at L_INJ, b6 (x+a)+m=h2 '
               'bf16, b7a identity attn '
               'self-swap, b8 block-output '
               'continuity (injected-forward '
               'zX[L_POST] vs zP[L_INJ]); no '
               'cross-model bit anchors by '
               'design (different model)',
    'artifacts': {
        'result': 'phase3081/omega_p78_'
                  'ds7b_crossmodel/'
                  'result.json',
        'npz': 'phase3081/omega_p78_'
               'ds7b_crossmodel/'
               'omega_p78_ds7b_crossmodel.'
               'npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'run1 authoritative (ds7b bf16, '
            '20634 forwards, 1031s).  Negative '
            'result preregistered as a first-'
            'class outcome.  Caveats: single '
            'model pair (cannot separate DS7B-'
            'specific from 4B-specific); DS7B '
            'is an R1 reasoning distill with '
            'plausibly different style-prefix '
            'handling; L_INJ=25/L_POST=26 vs '
            '4B L34/L35; n=24 pairs; T near-'
            'zero variance makes spearman '
            'noise-sensitive; shared '
            'syntactic frame; migration '
            'defined on the per-family focal '
            'top-8 subset frame.',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3081
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 220
    l14['connects'].append({
        'meas_id': 'meas3081_omega_p78_'
                   'ds7b_crossmodel',
        'phase': 3081,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P78: DS7B '
                        'cross-model replication '
                        '(20634 forwards, honest '
                        'independent test, gate '
                        'fixed in prereg).  '
                        'NEGATIVE: ds7b_cos_'
                        'absent.  Migration '
                        'itself absent (mig '
                        'sp(R1) -0.126/+0.091/'
                        '-0.130 vs 4B '
                        '+0.676/+0.130/-0.088; '
                        'T/U med ~0; top8 '
                        'overlap 2/1/2 ~ '
                        'chance); G_DS 1/6, '
                        'G1 fails; pipeline '
                        'sensitive (within-'
                        'family focal '
                        'structure strong: '
                        'capture8 0.57-0.79, '
                        'R_ALL to -1.28).  '
                        '3080 angle-matching '
                        'rule demoted to '
                        'MODEL-INTERNAL '
                        '(qwen3-4b); '
                        'cross-context gear '
                        'reuse is a training '
                        'product, not a '
                        'Transformer '
                        'universal.  Residual: '
                        'f2~U_AB +0.563 '
                        '(p=0.0056) - TT '
                        'alignment conditions '
                        'subset-transfer even '
                        'at zero mean.  Opens '
                        '3082: A DS7B negative-'
                        'anatomy (no-forward '
                        'npz re-analysis); B '
                        'third-model control; '
                        'C qwen3-4b new-family '
                        'D; D h14/injected-'
                        'state spectrum'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
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
if '## Phase 3081:' not in memo:
    sec = u'''## Phase 3081: Ω-P78 DS7B 跨模型对照——cos 锁定不复现，跨族路由迁移本身缺失（ds7b_cos_absent） [%(created)s]

**判决：`ds7b_cos_absent`**（预注册诚实独立检验：20634 前向、%(el).0f 秒、DS7B=deepseek-r1-distill-qwen-7b bf16 15.23GB 单模型逐一测试；G_DS 门在预注册中先行固定，DS7B 激活数据在运行前零观察）。**G_DS 失败：f2 六检验仅 1/6 显著**（f2~U_AB +0.5626 p=0.00555 唯一显著；min sp=−0.267<0；Stouffer z=1.761；f2~T_AB +0.0991 p=0.64）；G1 对照同样失败（best f1 +0.1904 p=0.376）。**更深一层的主发现：跨族迁移现象本身在 DS7B 上不存在**——mig=sp(R1_f,R1_g) 全部近零（**AB −0.1259 / AC +0.0909 / BC −0.1303**；4B 上 +0.676/+0.130/−0.088）；逐对 T/U 迁移中位数全 ≈0（T med +0.044/−0.010/−0.021；U med −0.002/+0.034/+0.024）；跨族 top8 重叠 2/1/2（随机期望 ≈2.3/8）；sp(U,T)=0.089/0.017/0.228（4B：0.690/0.487/−0.099）。管线敏感性对照（排除"测不出"）：**族内焦点结构健康存在**——r1 min −0.465/−0.682/−0.344、n_neg 27/21/26、capture8 0.566/0.793/0.633、全 28 头 swap R_ALL −0.247/−1.277/−0.340、med_c 0.6996/0.7468/0.4935；三族全部 b 锚（b0/b1/b4/b6/b7a/b8）bit 0.0。

### 问题与设计（3081 A，3080 菜单主选）
**问题**：3079 migration_tt_locked 与 3080 ab_cos_locked（装配泛化=角度匹配）是否模型普遍？**设计**：3076 协议在 DS7B 完整重跑（三族 × 8 body × 4 prefix；L_INJ=25 注入/L_POST=26 观测；28 头扫描 + 255 子集联合 swap + 全头 swap；E2H/lens 因 16GB 显存省略——无 W32，cos 全部来自前向 logits），然后 3079/3080 统计机器原样应用（T/U × f1–f5 共 30 检验，seed 3081，20000 置换）。

### 核心结果（重复三遍）
**① 迁移缺失（一）**：DS7B 的逐头因果谱在三个 prompt 族之间互不相关（mig ≈ 0），焦点头集合几乎不重叠（2/1/2 of 8 ≈ 随机）——**qwen3-4b 上"日常因果与科学因果共享路由"的现象是模型特异的**。**② 门按预注册失败（二）**：没有迁移信号，f1 秩形与 f2 夹角都无从锁定（G_DS 1/6、G1 失败）——阴性判决是门体系的正确输出，不是检验失灵。**③ 残存条件结构（三）**：f2~U_AB +0.5626（p=0.0056，唯一显著）+ 偏 sp(f2|f1) U_AB +0.542（p=0.0092）——AB 族对内，TT 方向对齐更好的对，其 255 子集结构迁移更高（尽管 U_AB 均值≈0）：**夹角信息在 DS7B 上退居为条件调制，不再是主载体**。另 f4~T_AB −0.466（p=0.022）/f4~T_AC −0.456（p=0.028）负相关（30 检验中 2 个 p<0.05，多重比较噪声水平，不立发现）。

### 硬伤与边界
- **单模型对**：一个 4B vs 一个 7B，无法区分"DS7B 特异"还是"4B 特异"——需要第三模型仲裁。
- DS7B 是 R1 推理蒸馏（Qwen2.5-7B 骨干），对 style-prefix 的处理机制可能与 instruct 模型根本不同——"迁移缺失"可能反映蒸馏训练对条件化路由的特化。
- L_INJ=25/L_POST=26（28 层倒数 3/2）vs 4B L34/L35（倒数 2/1）层位差异；n=24 对；T 近零方差下 spearman 噪声敏感；因果范式、共享句法框架；迁移定义依赖逐族 top-8 焦点框架。

### 理论更新（第一性原理）
**3080 的"装配泛化=角度匹配"规则被降格为模型内规律：qwen3-4b 内部，跨语境装配由 TT 方向夹角锁定；DS7B 上，跨语境齿轮复用本身缺失。** 条件化齿轮组的跨语境复用不是 Transformer 架构的普遍属性，而是训练产物（假设：instruct 模型的 style-prefix 共享处理回路 vs R1 蒸馏的域特化路由）。对 AGI 理论的关键启示：**"泛化"必须分层声明——(a) 架构级普遍（如注意力残差结构）、(b) 模型级规律（如 4B 的角度匹配装配）、(c) 族级条件规律（如 AB>AC 迁移不对称）**。3080 的发现在 (b) 层为真，在 (c) 层跨模型为假。可证伪预测改写：若跨模型复用存在，应先看到焦点头集合的跨族重叠显著高于随机——这是比 cos 锁定更基础的前置条件（DS7B 上前置条件已失败）。下一个仲裁实验：第三模型（glm4-9b / qwen2.5-3b）同管线——若 4B 是离群值，角度匹配规律记为 4B 特异性；若 DS7B 是离群值，则 R1 蒸馏特化假说升级。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3081/omega_p78_ds7b_crossmodel/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；20634 forwards / %(el).0fs。

**接续 3082 菜单**——A（主选）**DS7B 阴性结果解剖**（免前向，3081 npz 再分析：DS7B 的 R_S 谱形/TT 方向谱/族内结构 vs 4B 逐项对照——表征"迁移缺失"的形态：是焦点头换了一组、还是因果谱本身去相关、还是幅度层级不同）。B **第三模型仲裁**（glm4-9b 或 qwen2.5-3b 同管线，判定 4B/DS7B 谁离群；重，需前向防 OOM）。C **qwen3-4b 新族 D 对照**（3080 遗留，f2 锁定在新数据复验）。D **h14 全域解剖 / 注入态高阶交互谱**（遗留，免前向为主）。"好的，继续"即进 3082 A。
''' % {'created': created,
       'el': el,
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
if '## 四十三、3081' not in aud:
    add = u'''
---
## 四十三、3081 增补：DS7B 跨模型对照——迁移缺失，角度匹配规则降格为模型内规律（Omega-P78，判决 ds7b_cos_absent）
1. **阴性主发现**：DS7B 上跨族头路由迁移本身不存在——mig=sp(R1_f,R1_g)=−0.126/+0.091/−0.130（4B：+0.676/+0.130/−0.088），逐对 T/U 中位数≈0，跨族 top8 重叠 2/1/2（≈随机）；预注册 G_DS 门 1/6 失败（min sp=−0.267），G1 对照失败。
2. **管线敏感性成立**：族内焦点结构健康（capture8 0.57–0.79、R_ALL 至 −1.28、r1 min −0.68），全 b 锚 bit 0——阴性是现象缺失，不是测量失灵。
3. **残存条件结构**：f2~U_AB +0.5626（p=0.0056）+ 偏 sp(f2|f1) U_AB +0.542——TT 夹角在 DS7B 退居条件调制，非主载体。
4. HDMCC 更新：泛化声明分层——架构级普遍 / 模型级规律 / 族级条件规律；3080 角度匹配属模型级（qwen3-4b）；跨模型复用的前置条件=焦点头集合跨族重叠显著高于随机（DS7B 已失败）。仲裁实验=第三模型。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-22.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3081' not in prev:
    line = ('- Phase 3081 Omega-P78 DS7B '
            'cross-model replication (20634 '
            'forwards, 1031s, preregistered '
            'honest independent test of '
            'ab_cos_locked): verdict '
            'ds7b_cos_absent.  Migration '
            'itself absent on DS7B (mig '
            'sp(R1) -0.126/+0.091/-0.130 vs '
            '4B +0.676/+0.130/-0.088; T/U '
            'medians ~0; top8 overlap 2/1/2 '
            '~ chance); G_DS 1/6 significant '
            '(f2~U_AB +0.563 p=0.0056 only '
            'hit), G1 fails; pipeline '
            'sensitive (within-family focal '
            'structure strong).  3080 '
            'angle-matching rule demoted to '
            'MODEL-INTERNAL; generalization '
            'claims must be layered '
            '(architecture / model / family '
            'level).  All b-anchors bit 0.0; '
            'Audit 43; ledger 220/L14 188.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3081' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16 15.23GB，16GB 卡需省 W32/E2H）、glm4-9b-chat-hf 备用。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（link_id=L14_readout_spectrum_cross_model；verify 需 isinstance 防御）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果如实登记为一等公民；verdict 单分支赋值。
4. 统计纪律：阈值预注册；跨 phase bit 重放必须匹配上游浮点路径（spearman76 教训）；探索性升格=门先行固定+声明非独立（3080）；**泛化声明分层：架构级/模型级/族级（3081 阴性结果教训）**。

## 标准锚与精度
- bit 锚家族：标量行、因果置换、跨 phase 参考数、块链恒等、hook 互证、枚举重放、bit 锚族（3076）、跨源一致性（3077）、前向重建锚（3078）、免前向重放锚组（3079）、跨 npz 数据一致锚（3080）、**跨模型 setup 锚组（3081 b0/b1/b4/b6/b7a/b8 全 bit 0，无跨模型 bit 锚——不同模型按设计）**。
- b8 语义：注入前向内部块链连续性（zX[L_POST] vs 同一前向 zP[L_INJ] 末 token），不是 vs 基座 bank；置换 rng seed=phase 号。

## 机制解释审计链（命名前依次检查）
…→3078 routing_signal_absent→3079 migration_tt_locked→3080 ab_cos_locked（角度匹配，模型内）→**3081 ds7b_cos_absent：DS7B 跨族迁移本身缺失（mig≈0，top8 重叠≈随机）；f1/f2 无信号可锁；残存 f2~U_AB +0.56 条件调制→角度匹配=模型级规律；跨模型复用前置条件=焦点头跨族重叠＞随机；仲裁=第三模型**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat 坏→python shutil.rmtree；日志用 Read；-c stdout 丢→写文件再 Read；Glob 对部分模型目录失效→python os.listdir 为准。
- 关键写入后必须 Grep/Read 复核；改后必编译检查；跨行属性访问用括号。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3081）
Ω-P2（3011-3081）：…3078 routing_signal_absent；3079 migration_tt_locked；3080 ab_cos_locked；**3081 ds7b_cos_absent（跨模型阴性：迁移缺失，角度匹配降格模型级）**。

## 下一步
- max=3081，下一个 3082（A 主选 **DS7B 阴性结果解剖**：免前向 3081 npz 再分析，表征迁移缺失形态 vs 4B；B 第三模型仲裁（glm4-9b/qwen2.5-3b 同管线，重）；C qwen3-4b 新族 D；D h14/注入态高阶谱）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3081')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
