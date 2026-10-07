# -*- coding: utf-8 -*-
"""Phase 3107 closeout (idempotent):
Ledger -> MEMO Phase 3107 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3107'
        r'\omega_p105_writehead_mapping')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
verb = res['verdict']
assert verb == 'sparse_but_material_specific', verb
g = res['gates']

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3107
           for m in led['measurements']):
    claim = (
        'Omega-P105 (3107, offline T3 on frozen 3105/3106 '
        'captures, no GPU) - mode-family to write-head '
        'mapping.  Verdict sparse_but_material_specific '
        '(G1 pass, G2 fail, G3 fail).  G1 SPARSITY '
        'PASSED 5/5 frozen T2 configs: Top-100 '
        'coordinates (3.9pct of 2560) refit at recorded '
        'lambda keeps AUC: 3105 last|L8 0.9999->0.9999, '
        'query_obj|L8 1.0000->0.9998, crit_obj|L3 '
        '0.7361->0.6742 (ratio 0.916, borderline); 3106 '
        'last|L8 1.0000->0.9978, query_obj|L8 '
        '1.0000->0.9993 -> truth signal lives in a '
        'compact ~100-dim subspace.  G2 CROSS-MATERIAL '
        'COORDINATE REUSE FAILED: Jaccard(Top200 of 3105 '
        'vs 3106 last|L8 W) = 0.0526, near the random '
        'expectation 200*200/2560 -> J 0.042; within-'
        'material overlaps equally low (query_obj vs '
        'last 0.047, 3106 chain vs scatter 0.061); '
        'adjacent-layer Top-200 Jaccard 0.04-0.21 (G4 '
        'curves).  Single-coordinate identity of the '
        'carrier set is NOT stable across materials, '
        'positions, or layers -> the write-head group, '
        'if real, is a SUBSPACE with degenerate '
        'coordinate bases (ridge L2 spreads load over '
        'near-degenerate directions), not a fixed '
        'coordinate set.  G3 READOUT DIRECTION '
        'ALIGNMENT FAILED but FUNCTIONAL ALIGNMENT '
        'HELD: |cos(w_dn, W_T2)| = 0.0050 (3105) / '
        '0.0174 (3106), all 9 layers <= 0.036 (chance '
        '~0.02) - the model unembed yes-no direction is '
        'near-orthogonal to every learned truth-probe '
        'direction; YET Spearman(m margin, probe score) '
        'on 3105 TEST = 0.6545 >= 0.40 gate component - '
        'both directions READ THE SAME latent truth '
        'variable through geometrically independent '
        'routes.  Multi-route readout of one latent: '
        'model unembed route and linear-probe route are '
        'functionally coupled, geometrically distinct.  '
        'KEY THEORETICAL UPDATE: truth variable in the '
        'residual stream is read out through a '
        'low-dimensional but coordinate-degenerate '
        'subspace; single-coordinate (or fixed '
        'coordinate-set) claims are unsupported; '
        'analysis must be rotation-aware (principal '
        'angles, spectrum), matching the AGENTS.md rule '
        'that a coordinate is not a concept.  NEXT '
        '3108: degeneracy separation - (a) same-material '
        'two-half-train Top-200 Jaccard control '
        '(ridge-noise floor), (b) subspace-level reuse '
        'via principal angles between W subspaces, (c) '
        'spectrum/participation-ratio of W per layer.')
    meas = {
        'meas_id': 'meas3107_omega_p105_'
                   'writehead_mapping',
        'phase': 3107,
        'claim': claim,
        'verdict': 'sparse_but_material_specific',
        'anchors': 'probe configs + K list + lambda '
                   'source frozen in design_seal.json '
                   'before computation; Top-K selected '
                   'on TRAIN |W| only, refit same '
                   'lambda, eval TEST; w_dn from '
                   'embed_tokens (tied) rows 9834/902',
        'artifacts': {
            'result': 'phase3107/omega_p105_'
                      'writehead_mapping/'
                      'result.json',
            'seal': 'phase3107/omega_p105_'
                    'writehead_mapping/'
                    'design_seal.json'},
        'hashes': {},
        'note': 'offline only (3105+3106 capture.npz); '
                'SMOKE caught bf16/np, family-meta, '
                '1-d spearman, json-ndarray bugs',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3107_omega_p105_writehead_mapping')
    led.pop('ledger_sha256_8', None)
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
    o.append('ledger already upserted')

# ---------- MEMO Phase 3107 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3107:' not in memo:
    sec = u'''## Phase 3107: Ω-P105 写入头组映射——信号集中于 ~100 维紧凑子空间（Top-100 AUC 保持 5/5），但坐标基跨材料简并旋转（Top-200 Jaccard 0.053≈随机），模型 unembed 方向与学习探针方向近正交而功能耦合（Spearman 0.654）→ sparse_but_material_specific（writehead_mapping，离线）[[NOW]]

**性质**：T3 首 Phase，纯离线（3105/3106 冻结 capture.npz + embed_tokens 行，免 GPU，1m15s）。预注册（design_seal.json 先于一切统计冻结）：探针配置（3105 T2 last|L8、query_obj|L8、crit_obj|L3；3106 T2 last|L8、query_obj|L8；3105 T1 crit_obj 全 9 层）、K 列表、λ 来源（各 Phase result.json 记录值，不重选）、Top-K 协议（train |W| 选择→同 λ 子空间重拟合→test 评估）。门：G1 Top-100 AUC ≥ 0.90×全维（每配置）；G2 跨材料 Top-200 Jaccard ≥ 0.30；G3 |cos(w_dn,W)| ≥ 0.15 且 Spearman(m,probe) ≥ 0.40；G4 层稳定性曲线（描述性）。

### 1. G1 通过（5/5）：真值信号集中在紧凑子空间
| 配置 | 全维 AUC | Top-100（3.9% 坐标）重拟合 |
| --- | --- | --- |
| 3105 last\\|L8 | 0.9999 | **0.9999** |
| 3105 query_obj\\|L8 | 1.0000 | **0.9998** |
| 3105 crit_obj\\|L3 | 0.7361 | 0.6742（比率 0.916，压线） |
| 3106 last\\|L8 | 1.0000 | **0.9978** |
| 3106 query_obj\\|L8 | 1.0000 | **0.9993** |

2560 维中 100 个坐标承载几乎全部真值判别信号——"写入头组"的**存在性**成立（信号非弥散全场）。

### 2. G2/G4 失败：坐标身份跨材料/跨层/跨位置均不稳定
Jaccard(3105, 3106 last\\|L8 Top-200) = **0.0526**，随机期望 200×200/2560 → J ≈ 0.042——恰在噪声水平。同材料内 query_obj vs last 0.047、3106 链 vs 散射 0.061、相邻层 0.04–0.21（T1 crit_obj 在 L5–L7 略高 0.19–0.21）。**承载信号的 ~100 维子空间真实存在（G1），但其坐标基随材料/位置/层旋转（G2/G4）**——ridge L2 在近简并方向间任意摊派载荷。结论：写入头组若存在，是**子空间级**对象，不是固定坐标集；单坐标（或固定坐标组）主张不成立——与 AGENTS.md"单坐标不等于概念"纪律一致，且给出了定量版本。

### 3. G3 分裂：方向近正交 + 功能强耦合（本 Phase 最重要的理论发现）
|cos(w_yes−w_no, W_T2)| = **0.0050**（3105）/ 0.0174（3106），全 9 层 ≤ 0.036（chance ~1/√2560 ≈ 0.02）——模型自有 unembed 方向与一切可学习真值探针方向**近正交**。但 Spearman(m margin, probe score) 在 3105 TEST = **0.6545** ≥ 0.40。两个近正交方向在测试记录上产生强秩相关分数 → **同一隐变量（in-context 真值）的多路读出**：模型 unembed 路与线性可分路几何独立、功能耦合。这否定"真值=残差流中单一方向/坐标"图景，支持"低维简并子空间 + 多路读出"图景——条件化齿轮组的读出端是多端口的。

### 4. 硬伤
① Jaccard 随机基线 0.042 是解析推算未做显式置换对照；② ridge 简并旋转与"材料特异写入头"未完全分离——若信号真由不同坐标组承载，也会呈低 Jaccard（3108 控制实验分离）；③ cos 在原始坐标空间计算，未在 per-dim 标准化空间复核；④ Top-100 门在 crit_obj|L3 压线（0.916 vs 0.90），该位置/λ 本为 T1 优化；⑤ 全部分析基于 ridge 线性读出，非线性承载体未检验。

### 5. 理论更新与 3108 预注册
三图谱增量：**内部响应图谱**新增"真值信号的子空间集中性（~100 维）与坐标简并性"；**关联机制**新增"多路读出律：unembed 路与探针路几何正交、功能耦合（Spearman 0.65）"。**3108 = 简并分离与子空间角度**：① 同材料两半 train 的 Top-200 Jaccard（ridge 噪声地板显式化）；② W 子空间间的 principal angles（旋转不变的复用度量）；③ 各层 W 谱/参与率（participation ratio）。门观测前冻结。若 ② 显示子空间级复用 ≥ 阈值 → 写入头组=跨材料稳定子空间；否则降级为"每材料独立低维读出"。之后 3109 T3 收口 → 3110+ T4 多步自回归。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3107/omega_p105_writehead_mapping/`（result.json、design_seal.json、run_log.txt）；脚本 `tests/glm5/phase3107_omega_p105_writehead_mapping.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3107)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3107 Omega-P105 (offline T3, no GPU): '
          'write-head mapping -> sparse_but_material_'
          'specific. G1 5/5: Top-100 coords (3.9pct) keep '
          'AUC (0.9999/0.9998/0.6742/0.9978/0.9993) -> '
          'signal in compact ~100-dim subspace; G2 FAIL: '
          'cross-material Top-200 Jaccard 0.053 ~ random '
          '0.042 (also within-material 0.04-0.06, layers '
          '0.04-0.21) -> coordinate basis degenerate/'
          'rotating; G3 SPLIT: |cos(w_dn,W)|=0.005 (all '
          'layers <=0.036) BUT Spearman(m,probe)=0.654 -> '
          'multi-route readout of one latent (unembed '
          'route vs probe route geometrically orthogonal, '
          'functionally coupled). KEY: write-head group '
          'is subspace-level, not coordinate-level; '
          'rotation-aware analysis required. NEXT 3108: '
          'degeneracy control + principal angles + '
          'spectrum.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3107 Omega-P105' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md rewrite ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3107' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3106）',
        '## 机制链状态（3107）\n'
        '- 3107：写入头组映射 sparse_but_material_'
        'specific。Top-100 坐标保持 AUC（信号集中 '
        '~100 维子空间）但坐标基跨材料简并旋转'
        '（Jaccard 0.053≈随机）；unembed 方向与探针'
        '方向近正交而功能耦合（Spearman 0.654）——'
        '多路读出。写入头组=子空间级对象，分析需'
        '旋转不变。\n'
        '- 3106：四门全过 composition_dose_tracked。')
    mem_new = mem_new.replace(
        'max=3106', 'max=3107').replace(
        '下一 3107：**T3 模式族↔写入头组映射**（探针权重结构/读出子空间）→',
        '下一 3108：**简并分离与子空间角度**（噪声地板+principal angles+谱）→')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
