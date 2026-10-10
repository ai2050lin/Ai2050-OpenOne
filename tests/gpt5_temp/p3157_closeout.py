# -*- coding: utf-8 -*-
# p3157_closeout.py: Phase 3157 五写 closeout（ledger / MEMO+3158 预注册 / daily / MEMORY / 自检）
# 纪律: 幂等、快照、BOM 二进制安全（decode->append->encode）
import os, io, json, time, hashlib, shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-08.md')
MEMORY = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
TMP = os.path.join(ROOT, 'tests', 'gpt5_temp')
out = []
NOW = time.strftime('%Y-%m-%d %H:%M')

# ============ 0. 快照 ============
for src in (LEDGER, MEMO, DAILY, MEMORY):
    if os.path.exists(src):
        dst = os.path.join(TMP, 'snapshot3157_' + os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
out.append('snapshots done')

# ============ 1. Ledger（幂等 by phase==3157） ============
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == 3157 for m in ms):
    out.append('ledger: 3157 already present (n=%d), skip' % len(ms))
else:
    entry = {
        'phase': 3157, 'name': 'g2p2_transform_algebra_commutator',
        'seal_sha8': 'cfa5c3ed', 'result_sha8': '0fe043bf',
        'evidence_level': 'statistical',
        'model_scope': 'qwen3-4b + qwen3-14b + glm4-9b',
        'n_rows': 128, 'prereg_id': 'p3157', 'superseded_by': None,
        'verdict': 'g2p2_commutative_partial|fpmin_0.986|exch_mean_1.273',
        'rev_note': ('transform algebra P1 (user 2026-10-08 attachment sec 13/14): 16 shared '
                     'ents x 2 rels(isa/hasa) x 2 pols(+/-) x 2 ctxs(0/k16 real prefix) = 128 '
                     'rows/model, last-token all-layer H; operators T_N negation, T_R relation, '
                     'T_C context (T_P identity established by 3156 B-arm); commutator exch_R = '
                     '||dR(+)-dR(-)||/mean||dR|| in D-space at KOUT: C=0 R 1.032/1.121/1.072 N '
                     '1.193/1.294/1.331 (4b/14b/glm4), C=1 R 0.876/0.900/1.008 N '
                     '1.090/1.051/1.107 -> context REDUCES commutator (operators commute better '
                     'under context); class partial (0.5-2 gate), significant vs 100-ent-shuffle '
                     'null95; value ~1.0-1.3 close to sqrt(2) -> relation and negation '
                     'differentials approximately ORTHOGONAL (act on separate subspaces with mild '
                     'warp), not anti-parallel; balanced ANOVA [E|R|N|C|2nd-inters|resid] KOUT '
                     'mean shares E .276 R .050 N .039 C .289 ExR .057 ExN .026 ExC .096 RxN '
                     '.0154 RxC .026 NxC .018 resid .108 -> RxN variance tiny (1.2-1.7%) while '
                     'geometric warp large (exch ~1.2) = low-energy high-distortion interaction, '
                     'echoes 3155 conditional-gate approximate separability; T_C relation-'
                     'independence: mean pairwise cos of dC across 64 (r,n) pairs = .561-.564 '
                     'near-identical across models (context operator = shared core direction + '
                     'content-specific component, matches 3156 rank-1 massive-axis + semantics); '
                     'fingerprint KOUT Pearson .9858/.987/.9921 KSTAR .9998/.9998/1.0 exchR-'
                     'curve .904/.878/.791 all pass 0.8; rxn argmax layer model-specific '
                     'L9/L13/L22; fixes: 2x2 interaction coding must be r*n product (diagonal '
                     '±1 absorbed by main effects -> RxN=0 artifact), min-pair assert >=4 '
                     'chars; determinism bitwise x3; per-model res/seal: 4b c0db3455/41916ed8, '
                     '14b 1270c921/98c06ab0, glm4 a1d9a98a/e8be1026, summary 0fe043bf/cfa5c3ed; '
                     'npz 95a25965/9552086d/23dd74eb'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: appended 3157 (n=%d) chain_sha8=%s' % (len(ms), led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')
sec_marker = '## Phase 3157: 变换代数 P1——算子对易子（G2-P2）'
if sec_marker in txt:
    out.append('memo: 3157 section already present, skip')
else:
    block = r'''## Phase 3157: 变换代数 P1——算子对易子（G2-P2）[__NOW__]

**主判决：`g2p2_commutative_partial|fpmin_0.986|exch_mean_1.273`（res `0fe043bf` / seal `cfa5c3ed`）——T_N×T_R×T_C 三算子对易子三模型一致落 partial 区间（1.0–1.3），指纹三链全过（KOUT 0.986–0.992、KSTAR ≥0.9998、exchR 层曲线 0.79–0.90）。**

### 设计与执行
- 装置合并（用户数学结构附件第 13/14 节落地）：16 共享实体 × 2 关系(isa/hasa) × 2 极性(±) × 2 上下文(∅/k16 真实前缀) = 128 行/模型 ×3；T_P≈恒等由 3156 B 臂确立（位置算子=单位元）。execution 观测前冻结；bitwise ×3；runtime 15.5/143.1/73.4s。

### 五发现（重复强调）
1. **对易子 ≈1.0–1.3，近似正交而非反平行**：exch_R(C=0) 1.032/1.121/1.072、exch_N 1.193/1.294/1.331——比值接近 √2（正交）远小于 2（反平行）：**否定与关系作用在近似分离的子空间上，带轻度扭曲**；"变换代数"存在但非严格交换，也非乘性灾难。
2. **上下文降低对易子**（C=1: R 0.88–1.01、N 1.05–1.11）：T_C 使算子更可交换——上下文压缩/钝化极性差分，代数结构在语境中被"软化"。
3. **RxN 交互 = 低能量高扭曲型**：方差份额仅 1.2–1.7%（vs ExR 4.6–6.3%），但几何扭曲 exch≈1.2——与 3155 条件门（16% 交互、近似可分离）同族：**算子代数在能量视角近乎对角、在方向视角带恒定小扭曲**。
4. **T_C 关系无关性 cos=0.561–0.564（三模型几乎相同）**：64 对 dC 的平均两两余弦——上下文算子 = 共享核方向 + 内容特异分量两段结构，与 3156 rank-1 massive 轴 + 语义分量完全衔接。
5. **对易子层曲线是跨模型不变量**（Pearson 0.79–0.90）；RxN 峰层模型特异（L9/L13/L22）。

### 诚实边界
- exch 为 KOUT 层 D 空间 last-token 度量；否定引入 1–2 字符（N 主效应吸收，但作用点差分未消）；partial 的扭曲来源（正交 vs 反平行）由比值近 √2 推断，未直接分解；SMOKE 期 RxN=0 伪影（对角编码被主效应吸收）已修为 r⊙n 乘积编码——ANOVA 交互编码教训入库。

### Phase 3158 预注册（G4-P1 输出等价类 P1；用户数学结构附件特性 8 + 第 12 节"找商结构"）
- **假设**：读出层存在等价类 h~h' ⟺ P(·|h)≈P(·|h')；3156 已给出首个案例（B 臂内部差 1e-2 级但 KL_B≤0.0023、top1 9/9）。
- **设计（轻量，CPU+单次权重加载）**：① 读出矩阵 U=lm_head：敏感度谱 s(u)=‖Uu‖/‖u‖ 的奇异值谱 → 输出敏感子空间维数与零空间维数；② 零空间 vs 随机方向扰动预算：在 3157 KOUT 状态上注 ε·u，扫描 ε 至 KL=0.1，比较 ε_zerospace/ε_random（零空间方向应容忍 ≥10× 扰动）；③ 等价类直径：3156 npz 复算 ‖h_B−h_A0‖ vs KL_B 全 k 曲线 → 商结构的实测直径-曲率关系；④ 跨模型：U 谱形状 + 等价类直径曲线指纹（Pearson ≥0.8）。
- **门**：零空间扰动 KL ≤ 随机方向 10%（商结构存在）；跨模型直径曲线 ≥0.8。zero-GPU 主分析（unembed 权重 CPU 加载）。
'''
    block = block.replace('__NOW__', NOW)
    txt2 = txt + block
    open(MEMO, 'wb').write(txt2.encode('utf-8'))
    out.append('memo: appended 3157 section + 3158 prereg')
out.append('memo check: %s' % (sec_marker in open(MEMO, 'rb').read().decode('utf-8')))

# ============ 3. daily（幂等 by marker） ============
mk = '3157 G2-P2 变换代数对易子闭环'
dtxt = open(DAILY, 'rb').read().decode('utf-8') if os.path.exists(DAILY) else ''
if mk in dtxt:
    out.append('daily: already present, skip')
else:
    dline = ('- 3157 G2-P2 变换代数对易子闭环：16x2x2x2=128 行x3 模型；对易子 exch 1.0-1.3=partial（近正交非反平行）；'
             'RxN 方差仅 1.2-1.7%% 但几何扭曲大=低能量高扭曲；T_C 关系无关 cos≈0.56 三模型一致；指纹 fpmin 0.986；'
             'ledger n=%d。3158 预注册=输出等价类 P1（G4-P1，零空间扰动预算）。\n' % len(ms))
    open(DAILY, 'ab').write(dline.encode('utf-8'))
    out.append('daily: appended')

# ============ 4. workspace MEMORY（幂等；不动他线节） ============
mtxt = open(MEMORY, 'rb').read().decode('utf-8')
mk2 = '3157 变换代数'
if mk2 in mtxt:
    out.append('memory: already updated, skip')
else:
    old_tag = '下一步 3157=**G2-P2 变换代数对易子**'
    if old_tag in mtxt:
        idx = mtxt.find(old_tag)
        seg_end = mtxt.find('。', idx) + 1
        new_seg = ('**✅ 3157 变换代数对易子闭环（2026-10-08）**：16×2×2×2=128 行×3；对易子 exch 1.0–1.3=partial'
                   '（近正交非反平行）；RxN 方差 1.2–1.7%% 但几何扭曲大=低能量高扭曲型；T_C 关系无关 cos≈0.56 '
                   '三模型一致；指纹 fpmin 0.986；res summary 0fe043bf/seal cfa5c3ed/ledger n=%d。'
                   '下一步 3158=**G4-P1 输出等价类 P1**（零空间扰动预算+直径-曲率曲线，zero-GPU 主分析）。' % len(ms))
        mtxt2 = mtxt[:idx] + new_seg + mtxt[seg_end:]
        open(MEMORY, 'wb').write(mtxt2.encode('utf-8'))
        out.append('memory: replaced 3157-next tag -> done state')
    else:
        mtxt2 = mtxt + ('\n- 3157 G2-P2 变换代数闭环(2026-10-08)：对易子 partial 近正交；下一步 3158=输出等价类 P1。\n')
        open(MEMORY, 'wb').write(mtxt2.encode('utf-8'))
        out.append('memory: appended fallback line')

# ============ 5. 自检 ============
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert any(m.get('phase') == 3157 for m in led2['measurements'])
mtxt2 = open(MEMO, 'rb').read().decode('utf-8')
assert sec_marker in mtxt2
assert 'G4-P1 输出等价类' in mtxt2
out.append('SELF-CHECK OK: ledger has 3157, memo has 3157 section + 3158 prereg')

with io.open(os.path.join(TMP, 'p3157_closeout_out.txt'), 'w', encoding='utf-8') as f:
    f.write(chr(10).join(out))
print('written')
