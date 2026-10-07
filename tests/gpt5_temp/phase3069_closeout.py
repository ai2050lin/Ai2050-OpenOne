# -*- coding: utf-8 -*-
"""Phase 3069 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3069'
     r'\omega_p66_cross_layer_mlp_pool')
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
assert verdict == 'cross_layer_distributed', \
    verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a1ext_diff'] == 0.0
assert an['a2set_diff'] == 0 and an['a2set_ok'] is True
assert an['a3_diff'] == 0.0 and an['a3_ok'] is True
assert an['l35ctl_diff'] == 0.0 \
    and an['l35ctl_ok'] is True
assert an['a2lens_max'] == 0.062412261962890625
assert an['a2lens_ok'] is True
assert an['a2b_diff'] < 0.0002
assert an['a2b_ok'] is True
assert an['b0_diff'] == 0.0 and an['b0_ok'] is True
assert an['b1_diff'] == 0.0 and an['b1_ok'] is True
assert an['b3_ok'] is True
assert an['b4_diff'] == 0.0 and an['b4_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b6_diff'] == 0.0 and an['b6_ok'] is True
assert an['b7l_diff'] == 0.0 and an['b7l_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_l']['30'] == 0.5312391051160289
assert st['med_c_l']['31'] == 0.36067533326696655
assert st['med_c_l']['32'] == 0.10106477801389738
assert st['med_c_l']['33'] == 0.44953101303168347
assert st['med_c_l']['34'] == 0.1487826048372403
assert st['med_c_l']['35'] == -0.35579100779974404
assert abs(st['obs_cproj_l']['30']
           - 48.36822341089126) < 1e-9
assert abs(st['obs_cproj_l']['31']
           - 28.466891625071) < 1e-9
assert abs(st['obs_cproj_l']['32']
           - 187.96649777134724) < 1e-9
assert abs(st['obs_cproj_l']['33']
           - 515.5637264191196) < 1e-9
assert abs(st['obs_cproj_l']['34']
           + 95.91242909117048) < 1e-9
assert abs(st['obs_cproj_l']['35']
           + 626.3775671949336) < 1e-9
pl = {p['layer']: p for p in st['per_layer']}
assert abs(pl[33]['recov']['top']
           + 0.14730966510061888) < 1e-12
assert abs(pl[33]['recov']['all']
           + 0.13536460829621094) < 1e-12
assert abs(pl[33]['capture']
           - 1.0882435738170884) < 1e-12
assert abs(pl[34]['recov']['top']
           + 0.0021290933509379995) < 1e-12
assert abs(pl[34]['recov']['all']
           + 0.03954917325691043) < 1e-12
assert abs(pl[34]['capture']
           - 0.05383407984554981) < 1e-12
assert abs(pl[35]['recov']['top']
           - 0.7062164995552644) < 1e-12
assert abs(pl[35]['recov']['all']
           - 0.7317112170947366) < 1e-12
assert abs(pl[35]['capture']
           - 0.9651574050747792) < 1e-12
assert abs(pl[31]['recov']['top']
           + 0.052794721981111015) < 1e-12
assert abs(pl[31]['recov']['all']
           + 0.012251653202798085) < 1e-12
assert abs(pl[32]['recov']['top']
           + 0.029324646082249828) < 1e-12
assert abs(pl[32]['recov']['all']
           - 0.03383022967267002) < 1e-12
assert abs(pl[30]['recov']['top']
           - 0.012780267640154497) < 1e-12
assert abs(pl[30]['recov']['all']
           - 0.0058318937893887535) < 1e-12
assert abs(pl[33]['score_max']
           - 20.287043227814138) < 1e-9
assert abs(pl[34]['score_max']
           - 18.27193613257259) < 1e-9
assert abs(pl[35]['score_max']
           - 13.43401150405407) < 1e-9
assert st['n_conc'] == 2
assert st['l35_control_recov'] == \
    st['l35_control_ref']
assert res['forwards'] == 619

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3069
           for m in led['measurements']):
    claim = (
        'Omega-P66 (plan 3069 A) - qwen3-4b bf16 '
        'cross-layer MLP pool generalization L30-'
        '35, per-layer injection ladder + swap-to-'
        'base (41.1s, 619 forwards; anchors: a1 '
        'rows 34/35 AND a1ext rows 30-35 ALL '
        'bit-exact vs 3066 npz diff 0.0 - the '
        '3066 ladder was full-layer same-protocol; '
        'a2 S_TOP(L35) set equality vs 3067 npz; '
        'a3 PERM(L35,TOP) bit 0.0 vs 3067 npz; '
        'l35ctl recov diff 0.0 vs 3067 ref '
        '0.7062164995552644; med_c reference '
        'assert; b0/b1/b4/b5/b6/b7l bit 0.0; '
        'b3 finite). RESULTS: verdict '
        'cross_layer_distributed (preregistered '
        'n_conc=2). (1) POLARITY STRATIFICATION: '
        'observed MLP self-diff TT projections '
        'L30-33 POSITIVE (+48/+28/+188/+516) and '
        'L34/L35 NEGATIVE (-96/-626) - the sign '
        'flip of MLP writes sits between L33 and '
        'L34, one layer BEFORE the readout-c '
        'sign flip (med_c +0.149@34 -> -0.356@35). '
        '(2) AMPLITUDE STRATIFICATION: ALL-swap '
        'effect |recov_all|: L35 0.732 (206 pct '
        'of its med_c), L33 0.135 (30 pct), L34 '
        '0.040 (27 pct), L32 0.034 (sign-'
        'opposed), L31 0.012 (3.4 pct), L30 '
        '0.006 (0.6 pct) - upstream injections '
        'carry their direct effect in attention/'
        'propagation, NOT in the layer-own MLP; '
        'only the last two MLPs dominate their '
        'own layer effect. (3) POOL CONCENTRATION '
        'IS TRACKED BY EFFECT SIZE: L35 capture '
        '0.965 (top-128 carries 96.5 pct of the '
        'upper bound, random control -0.002), '
        'L33 capture 1.088 (top slightly exceeds '
        'ALL - mild overshoot/masking; random '
        '-0.014, 11x specificity); L34 capture '
        '0.054 - it WRITES negative (obs -96) '
        'but its causal footprint is nearly '
        'zero: a suppressed negative writer that '
        'LOSES the competition; small-effect '
        'layers L30-32 have numerically unstable '
        'captures (0.02 floor, n/a). (4) L32 '
        'sign paradox recorded: obs write '
        'positive (+188) yet swap-to-base RAISES '
        'c (recov_all +0.034) - direct write vs '
        'downstream-mediated effect oppose; '
        'unresolved. Conclusion: 3066 distributed '
        'positive normalization REVISED - '
        'positive contributions exist in L30-33 '
        'MLPs but amplitude concentrates in the '
        'last-two-layer MLPs (L35 >> L33); the '
        'L35 one-layer negative-writer picture '
        '(3067/3068) is the tail of a smooth '
        'depth stratification of both polarity '
        'and amplitude.')
    meas = {
        'meas_id': 'meas3069_omega_p66_cross_'
                   'layer_mlp_pool',
        'phase': 3069,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 rows 34/35 + a1ext rows '
                   '30-35 bit 0.0 vs 3066 npz '
                   '(hard); a2 S_TOP(L35) set '
                   'equality vs 3067 npz (hard); '
                   'a3 PERM(L35,TOP) bit 0.0 vs '
                   '3067 npz (hard); l35ctl diff '
                   '0.0 vs 3067 ref (hard); med_c '
                   'reference assert; a2b 9.4e-05; '
                   'b0/b1/b4/b5/b6/b7l bit 0.0; '
                   'b3 finite',
        'artifacts': {
            'result': 'phase3069/omega_p66_'
                      'cross_layer_mlp_pool/'
                      'result.json',
            'npz': 'phase3069/omega_p66_'
                   'cross_layer_mlp_pool/'
                   'omega_p66_cross_layer_'
                   'mlp_pool.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke '
                'zero-crash 5th consecutive '
                'phase; smoke 8-pair verdict '
                'concentrated was OVERTURNED by '
                'the 24-pair authoritative run - '
                'preregistered statistics '
                'discipline held)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 208
    l14['connects'].append({
        'meas_id': 'meas3069_omega_p66_cross_'
                   'layer_mlp_pool',
        'phase': 3069,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P66: cross-layer '
                        'MLP generalization L30-35 '
                        '- polarity stratification '
                        '(L30-33 MLP write positive '
                        '+48/+28/+188/+516, L34/L35 '
                        'write negative -96/-626; '
                        'write-sign flip at L33->34, '
                        'one layer before the c-sign '
                        'flip at 34->35) + amplitude '
                        'stratification (ALL-swap: '
                        'L35 0.732 = 206 pct of its '
                        'med_c, L33 0.135 = 30 pct, '
                        'upstream layers 0.6-3.4 pct '
                        '- direct effects live in '
                        'attention/propagation '
                        'upstream, in layer-own MLP '
                        'at the top) + concentration '
                        'tracks effect size (L35 '
                        'capture 0.965, L33 1.088, '
                        'L34 0.054 = suppressed '
                        'negative writer that loses '
                        'the competition). 3066 '
                        'distributed normalization '
                        'revised to amplitude-'
                        'concentrated L35>>L33. '
                        'Opens 3070: A L34 suppression '
                        'anatomy; B DS7B last-layer '
                        'control; C input-displacement '
                        'lineage dz_A/dz_B; D neuron '
                        'identity; E cross-prompt-'
                        'family generalization'})
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
if '## Phase 3069:' not in memo:
    sec = u'''## Phase 3069: Ω-P66 跨层 MLP 池推广——极性与幅度双重分层，L34 是被压制的负写者（cross_layer_distributed） [%(created)s]

**判决：`cross_layer_distributed`**（qwen3-4b 单模型 bf16 41.1s，619 次前向；五重锚全过：**a1：阶梯 rows 34/35 与 3066 npz diff=0.000e+00；a1ext：rows 30-35 全部 6 行与 3066 npz diff=0.000e+00——3066 的阶梯本就是全层同协议，跨 phase 6 行 bit 级互证**；a2：S_TOP(L35) 与 3067 S_A 集合相等；a3：PERM(L35,TOP) 与 3067 PERM_A35 diff=0.0；l35ctl：recov 复现 3067 参考 0.7062164995552644 diff=0.0；med_c 参考断言、a2b 9.4e-05、b0/b1/b4/b5/b6/b7l 全 bit 0.0）。SMOKE 零崩溃连续第五 phase。**如实入册：smoke（8 对）初判 concentrated 被权威（24 对）推翻——预注册统计纪律生效，避免了一次小样本误判**。

### 问题与设计
**问题**（3069 A，3068 菜单）：L35 的"top-128 集中承载 + 一池多工况"是否逐层普遍？L30-34 的 MLP 写入极性是什么（不预设正负）？设计：E1 注入层扩展为 L={30..35}×24 对（协议 3065/3066/3067 逐字一致），每层记 COS_LAD_l/PA/PF 探针+该层自身 act/m 差分；E1.5 逐层评分 score_l[j]=median_k(|dact_l[k][j]|·||W_down_l[:,j]||)→S_TOP_l=top-128、S_R_l=seed 3025+li 随机 128；E1.6 观察极性 cproj_l=(W32·dm_l)·TT/||TT|| 中位（声明近似）；E2 每层 {TOP,RAND,ALL}×24 对 swap-to-base（每层 down_proj forward-pre-hook 末位，b7l 每层恒等验证）；recov_l=median PERM_l - med_c_l（同层同注入参照）；capture_l=|recov_top|/|recov_all|（要求 |all|≥0.02 且同号）。

### 核心结果（重复三遍）
**① 极性分层（一）**：各层 MLP 自身差分的 TT 投影——L30-33 写**正**（+48/+28/+188/+516），L34/L35 写**负**（-96/-626）；**MLP 写入符号翻转在 L33→L34，比读出 c 符号翻转（med_c +0.149@34→-0.356@35）早一层**。幅度沿深度递增（正侧 30→33 增强、负侧 34→35 增强），末端两层 MLP 主导写入。**② 幅度分层（二）**：ALL 换回（该层 MLP 差分完全中和）的效应量 |recov_all|：**L35 0.732（其直接效应 med_c 的 206 percent）、L33 0.135（30 percent）、L34 0.040（27 percent）、L32 0.034（符号相反）、L31 0.012（3.4 percent）、L30 0.006（0.6 percent）**——上游注入的直接效应住在 attention/传播结构里、几乎不在自层 MLP；只有末端两层 MLP 主导自己层的直接效应。**③ 集中度跟随效应量（三遍）**：L35 capture **0.965**（top-128 拿走上界 96.5 percent，随机对照 -0.002）；L33 capture **1.088**（top 略超 ALL——轻度过冲/掩蔽；随机 -0.014，特异性 11 倍）；**L34 capture 0.054——它写负（obs -96）但因果足迹近零：一个输掉竞争的被压制负写者**；小效应层 L30-32 的 capture 数值不稳（0.02 floor 下 n/a）。附加：L32 符号悖论——obs 写正（+188）但换回 base 使 c 升（recov_all +0.034），直接写入与下游介导效应反向，未解。

### 机制综合（Ω-P66 拼图）
3066 的"分布式下游正化"修正为：**正贡献确实分布于 L30-33 MLP，但幅度集中于末端两层（L35≫L33）**；3067/3068 的"L35 一层独大"是深度分层曲线的尾部而非孤立异常。层级齿轮图：**L35=决定性负写者**（一层 carry 全部负效应+池集中 96.5 percent）、**L33=主要正写者**（自层承载 30 percent+层内集中）、**L34=压制态负写者**（想写负但输掉——其负写入被下游 L35 覆盖或被 attention 中和，是 3066 竞争图景的层级实例）、**L30-32=传导层**（MLP 足迹 0.6-3.4 percent）。"深部决定、浅部传导"：语言组合的因果负担不成比例地落在末端少数层。

### 硬伤与边界
- capture 在小效应层数值不稳（除以 |all|<0.02 无意义）——n_conc=2 的 distributed 判决忠实于预注册，但真实结构是"极性分层+幅度集中"，不是均匀分布式。
- L30-32 的 MLP 效应小可能部分因注入方式（V 替换前 4 位置）对上游层刺激弱——不能用本设计否定上游 MLP 在其他条件下的作用。
- obs cproj 是 fp32 unembed 无 final norm 的声明近似；recov 经 final_norm 非线性读出。
- 单 prompt 族、单模型；L32 符号悖论未解；L34 压制机制（L35 重写 vs attention 中和）未分解。

### 方法论入册
- **a1ext 全层 bit 锚**：跨 phase 6 行阶梯 bit 级互证（3066↔3069）——协议冻结的价值；后续 phase 直接扩展锚行数。
- **capture 类指标必须设效应量 floor**（0.02）：小分母产生无意义的大 capture。
- **SMOKE 小样本可给出相反判决**（8 对 concentrated → 24 对 distributed）——smoke 只验管线不预测结论，判决只认预注册全量。
- 注入层自捕获（ZACT_SELF@注入层）是测量"层自响应"的正确位置；下游层 act 差分是另一个量（3068 的 S_B 视角）。

### 智能理论洞察（第一性原理）
有限参数语言能力的**纵向组织**首次定量成像：极性翻转一层、幅度集中两层、池集中在效应大的层。结合 3067/3068 的横向组织（J 固定×输入方向×一池多工况），条件化齿轮组有了两个维度的图纸。语言能力的"计算负担分配"极其不均：**顶层 MLP 是杠杆点**（L35 一层承载 206 percent 的自身直接效应且 96.5 percent 集中于 128 个神经元）——这解释了为何单层解剖（3067）就能拿到主要负写入者。下一步的关键缝隙：**L34 压制机制**——它的负写入去哪了？（inj@34 时 L35 的 act 差分里应有 L34 负写入的下游投影，3068 的 S_B/S_A 分解正好是工具。）

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3069/omega_p66_cross_layer_mlp_pool/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3070 菜单**——A（主选）**L34 压制解剖**：inj@34 下分解 L34 负写入的去向（L35 act/m 差分响应 + 3068 的 S_B 视角 + attention 通路检查），回答"被压制的负写者输在哪"；B **DS7B 末层对照**：DS7B 末层 MLP 解剖复刻（跨模型检验）；C **输入方向谱系**：dz_A 是否=注入 V 签名的线性像、dz_B 传播路径分解；D **神经元身份**：S_TOP 各层的 up/gate 权重结构、W_U 语义方向关联；E **跨 prompt 族泛化**：新 prompt 族复测 L33/L35 集中度与极性分层。"好的，继续"即进 3070 A。
''' % {'created': created,
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
if '## 三十一、3069 增补' not in aud:
    add = u'''

---

## 三十一、3069 增补：跨层 MLP 池推广——极性与幅度双重分层，L34 是被压制的负写者（Omega-P66，判决 cross_layer_distributed）

1. **极性分层**：L30-33 MLP 自身差分写正（+48/+28/+188/+516），L34/L35 写负（-96/-626）；MLP 写入符号翻转在 L33→L34，比读出 c 符号翻转（34→35）早一层；幅度沿深度递增。
2. **幅度分层**：ALL 换回效应 L35 0.732（206 percent of med_c）、L33 0.135（30 percent）、上游 L30-31 仅 0.6-3.4 percent——上游直接效应住 attention/传播，末端层 MLP 才主导自己层；3066"分布式正化"修正为"多层正贡献、幅度集中 L35≫L33"。
3. **集中度跟随效应量**：L35 capture 0.965、L33 1.088（轻度过冲）、L34 0.054——L34 写负但因果足迹近零，是被压制的负写者（输掉竞争）；小效应层 capture 无意义（0.02 floor）。L32 符号悖论（obs 正/因果反）记录未解。
4. HDMCC 修正：n_conc=2 的预注册判决 faithful，但证据结构是"极性分层+幅度集中"而非均匀分布；capture 类指标必须设效应量 floor；SMOKE 小样本判决可被全量推翻（8 对 concentrated → 24 对 distributed），判决只认预注册全量。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-21.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3069' not in prev:
    line = ('- Phase 3069 Omega-P66 cross-layer '
            'MLP pool generalization L30-35 '
            '(qwen3-4b bf16 single, 41.1s, 619 '
            'forwards): verdict '
            'cross_layer_distributed (prereg '
            'n_conc=2). Five anchors bit-exact '
            'incl. a1ext rows 30-35 vs 3066 npz '
            '(full 6-row cross-phase ladder '
            'replication) and l35ctl diff 0.0. '
            'POLARITY stratified: L30-33 MLP '
            'write positive (+48/+28/+188/+516), '
            'L34/L35 negative (-96/-626); write-'
            'sign flip L33->34, one layer before '
            'the c-sign flip 34->35. AMPLITUDE '
            'stratified: ALL-swap L35 0.732 (206 '
            'pct of med_c), L33 0.135 (30 pct), '
            'upstream 0.6-3.4 pct - upstream '
            'direct effects live in attention/'
            'propagation. Concentration tracks '
            'effect size: L35 capture 0.965, L33 '
            '1.088, L34 0.054 = suppressed '
            'negative writer. L32 sign paradox '
            'recorded. Smoke 8-pair verdict '
            '(concentrated) OVERTURNED by the '
            '24-pair authoritative run. Audit 31; '
            'ledger 208/L14 176.\n')
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
if 'max=3069' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 被全量推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；阈值预注册；重叠类指标算随机期望；capture 类指标设效应量 floor（如 0.02）再除。

## 标准锚与精度
- 跨脚本 bit 级锚是"新脚本=旧机制"最强验证，已扩展到：标量行（a1）、全层多行（a1ext：3069 的 30-35 六行 vs 3066）、集合相等（a2）、因果置换结果（a3）、跨 phase 参考数（l35ctl recov diff 0.0）。
- **3066**：per-k 聚合显式累积后 median；cos 探针 fp32 权重副本；SMOKE 先行。
- **3067**：改模块输入用 forward-pre-hook；base 自替换=恒免 matched base（b7）；权重副本 .detach().float()；符号定位四件套。
- **3068**：m=W_down·act 线性→partition 恒等分解；ALL 置换=上界锚；重叠算随机期望（m^2/inter）。
- **3069**：注入层自捕获（ZACT_SELF@注入层）测"层自响应"；每层 hook 用 dict stateACT[li]；b7l 每层恒等；recov 参照=同注入条件 med_c_l；SMOKE 判决仅供管线验证不预测结论。

## 机制解释审计链（命名前依次检查）
…→3067 神经元级解析（定位 top-128、两状态共享、线性雅可比）→3068 竞争分解（一池两工况：重叠 76/128、P3 主导）→**3069 跨层推广（极性分层：L30-33 写正/L34-35 写负，翻转在 L33→34 早 c 翻转一层；幅度分层：L35 0.732=206 percent、L33 0.135=30 percent、上游 0.6-3.4 percent；集中度随效应量：L35 0.965/L33 1.088/L34 0.054 被压制）→层级齿轮图：L35 决定性负写者、L33 主要正写者、L34 压制态负写者、L30-32 传导层**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查（含 docstring 转义）。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3069）
Ω-P2（3011-3069）：…3067 mlp_reversal_localized_shared；3068 competition_splitpool_top_negative_both；**3069 cross_layer_distributed（极性/幅度双重分层；深部决定浅部传导；顶层 MLP=杠杆点）**。

## 下一步
- max=3069，下一个 3070（A 主选 **L34 压制解剖**：inj@34 下 L34 负写入去向=L35 act/m 响应+S_B 视角+attention 通路；B DS7B 末层对照；C 输入方向谱系 dz_A/dz_B；D 神经元身份；E 跨 prompt 族泛化）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3069')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
