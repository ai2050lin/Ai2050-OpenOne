# -*- coding: utf-8 -*-
# p3158_closeout.py: Phase 3158 五写 closeout（ledger / MEMO+3159 预注册 / daily / MEMORY / 自检）
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
        dst = os.path.join(TMP, 'snapshot3158_' + os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
out.append('snapshots done')

# ============ 1. Ledger（幂等 by phase==3158） ============
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == 3158 for m in ms):
    out.append('ledger: 3158 already present (n=%d), skip' % len(ms))
else:
    entry = {
        'phase': 3158, 'name': 'g4p1_output_equivalence_class',
        'seal_sha8': 'e0c60629', 'result_sha8': 'fa5cca12',
        'evidence_level': 'statistical',
        'model_scope': 'qwen3-4b + qwen3-14b + glm4-9b',
        'n_rows': 16, 'prereg_id': 'p3158', 'superseded_by': None,
        'verdict': 'g4p1_fingerprint_consistent|quotient_0/3_mixed',
        'rev_note': ('output equivalence class P1 (user math-structure attachment property 8 / '
                     'sec 12 quotient): zero-model-forward protocol, logits = W_U @ h_slotNL '
                     '(HF output_hidden_states last slot ALREADY post-final-norm, verified vs '
                     '3156 stored LG cos=1.0000 top1 4/4 relerr .003 - protocol discovery); '
                     'anchors = 3157 H slot NL 16 rows; perturb p+a*||p||*u along bottom64/top64 '
                     'eigvecs of W_U^T W_U + rand QR, Jeffreys KL=0.1 budget, log-log interp; '
                     'FINDINGS: (1) unembed spectrum nearly FLAT all 3 models: sigma_max/min '
                     '58/57/66, e99 dim 2486/4967/3951 = 97% of D, soft-null only 2.9-3.5% -> '
                     'NO subspace-type quotient at readout; (2) budget ratio null/rand 1.03-1.33 '
                     'all absent (gate >=10), null/rand/top budgets within 1.0-1.7x -> readout '
                     'sensitivity nearly isotropic in Fisher metric; prereg >=10x expectation '
                     'falsified by flat spectrum; (3) part3 3156 recompute: position axis (B arm) '
                     'equivalence diameter d_rel .0169 (zh)/.0119 (en) at KL<=.0008 vs context '
                     'axis KL up to 4.81 - equivalence classes exist only as THIN neighborhoods '
                     '(continuous-elastic readout), position moves 3-4 orders cheaper in KL; '
                     '(4) cross-model fingerprint: spec128 Pearson .9855/.999/.9918, budget-curve '
                     'log10-KL Pearson .9995x3 -> elastic response function of readout is a '
                     'cross-model invariant; (5) PR 129.6/1239.4/286.3 (14b single 89.6 outlier '
                     'sigma head); determinism gram+base-logits bitwise; fixes: transpose '
                     '(V,ND)->(ND,V), einsum subscripts, spec128 fixed 128-pt fractional-rank '
                     'interp (np.unique gave len 98/102 per D), missing ||p|| factor in eps; '
                     'per-model res/seal: 4b bee3d02a/1b835795, 14b 0d671106/a510409e, glm4 '
                     '5e417cbc/5c3c0e06, summary fa5cca12/e0c60629; npz c4d9b9fb/7f53f3a0/2ea72cfe'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: appended 3158 (n=%d) chain_sha8=%s' % (len(ms), led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')
sec_marker = '## Phase 3158: 输出等价类 P1——读出映射的商结构（G4-P1）'
if sec_marker in txt:
    out.append('memo: 3158 section already present, skip')
else:
    block = r'''## Phase 3158: 输出等价类 P1——读出映射的商结构（G4-P1）[__NOW__]

**主判决：`g4p1_fingerprint_consistent|quotient_0/3_mixed`（res `fa5cca12` / seal `e0c60629`）——三模型一致**无子空间型商结构**（预算比 1.03–1.33 ≪ 门 10），但读出"弹性响应函数"是跨模型不变量（谱指纹 0.9855–0.999、预算曲线指纹 0.9995×3）。**

### 设计与执行
- 协议（零模型前向，GPU 仅线性代数）：logits = W_U @ h_slotNL；**协议发现：HF output_hidden_states 末槽已含 final RMSNorm**（vs 3156 存储真 logits LG：cos=1.0000、top1 4/4、relerr≈0.003）——pre-reg"KOUT 状态"操作化为槽 NL 精确读出输入。
- 锚 = 3157 H 槽 NL 16 行；扰动 p′=p+α·‖p‖·u，u ∈ bottom64/top64 右奇异向量 + rand(QR)，Jeffreys KL=0.1 预算（log-log 插值，float64 logsumexp 恒等式）；Part③ 3156 npz 复算直径-曲率。runtime 25.2/31.8/28.8s。

### 五发现（重复强调）
1. **unembed 谱近乎平坦（三模型一致）**：σ_max/σ_min = 58/57/66，e99 维 = 2486/4967/3951（占 D 的 97%），软零维仅 2.9–3.5%——**读出层不存在大零空间**；预注册"零空间容忍 ≥10×"预期被平坦谱证伪。
2. **预算比 null/rand = 1.03–1.33 全 absent**：bottom-σ/top-σ/随机方向的 KL 预算差 < 2×——读出敏感度在 Fisher 度量下**近各向同性**（KL 由 p-支撑上的方向方差决定，非全谱 RMS）。
3. **等价类实测 = 极薄邻域**：位置轴（B 臂）在 KL≤0.0008 内直径 d_rel≈1.7%（zh）/1.2%（en）；上下文轴 KL 达 4.8——两轴 KL 成本差 3–4 个量级；**商结构以"连续弹性"而非"子空间商"的形式存在**（特性 8 的答案）。
4. **弹性响应函数是跨模型不变量**：KL-vs-α 预算曲线三模型两两 Pearson **0.9995**；谱形状 0.9855–0.999——不同参数实现收敛到同一读出几何。
5. PR = 129.6/1239.4/286.3（14b 单一 89.6 巨头 σ）；p_norm cv 0.03–0.06（锚点范数均匀）。

### 诚实边界
- 预算比是对中位数比（删失无发生）；KL 支撑集中高概率 token，Fisher 加权方差与全谱 RMS 解耦（top 臂预算≈rand 的原因）；14b/glm4 无 3156 npz，Part③ 仅 4b；中间层动力学是否"主动消除"软零性未测（=3159）。

### Phase 3159 预注册（G4-P2 等价类动力学：软零方向的逐层流向）
- **假设**：3158 证读出谱平坦（无 inherited 零空间）；P2 问**动力学是否主动消除软零性**——在中间层注入读出谱意义下的 bottom-σ 方向，剩余层是否把它旋转回敏感子空间（quotient destroyed）还是保持（quotient stable）。
- **设计**：锚 = 3157 H 槽 L_mid=round(0.5·NL) 16 行；GPU 注入前向（hook 替换 last-token 态）；u 三臂同 3158（bottom64/top64/rand 各 8）；α 相对 ‖h‖ 扫描；测 ① mid-layer KL 预算比 ratio_mid（同 3158 门 10/3）② re-emergence：注入方向逐层投影 top-64 子空间能量份额曲线 ③ KOUT 读出谱预算 vs mid 注入预算传递。
- **门**：ratio_mid ≥ 3 → quotient_stable（动力学保持软零性）；< 3 → dynamics_destroyed（更强调制发现：各向同性是计算出来的）。GPU ~16 锚×24 方向×6α batch 化，预算 ~10min/模型。
'''
    block = block.replace('__NOW__', NOW)
    txt2 = txt + block
    open(MEMO, 'wb').write(txt2.encode('utf-8'))
    out.append('memo: appended 3158 section + 3159 prereg')
out.append('memo check: %s' % (sec_marker in open(MEMO, 'rb').read().decode('utf-8')))

# ============ 3. daily（幂等 by marker） ============
mk = '3158 G4-P1 输出等价类闭环'
dtxt = open(DAILY, 'rb').read().decode('utf-8') if os.path.exists(DAILY) else ''
if mk in dtxt:
    out.append('daily: already present, skip')
else:
    dline = ('- 3158 G4-P1 输出等价类闭环：unembed 谱近平坦（e99=97%% D，软零 2.9-3.5%%）；预算比 null/rand '
             '1.03-1.33 三模型全 absent（预注册 ≥10x 被证伪）；等价类=极薄邻域（B 臂 d_rel 1.7%% @ KL≤0.0008 vs '
             '上下文轴 KL 4.8）；弹性响应曲线跨模型指纹 0.9995；协议发现 HF 末槽已含 final norm；'
             'ledger n=%d。3159 预注册=等价类动力学 P2（G4-P2 软零方向逐层流向，GPU 注入）。\n' % len(ms))
    open(DAILY, 'ab').write(dline.encode('utf-8'))
    out.append('daily: appended')

# ============ 4. workspace MEMORY（幂等；不动他线节） ============
mtxt = open(MEMORY, 'rb').read().decode('utf-8')
mk2 = '3158 输出等价类'
if mk2 in mtxt:
    out.append('memory: already updated, skip')
else:
    old_tag = '下一步 3158=**G4-P1 输出等价类 P1**'
    if old_tag in mtxt:
        idx = mtxt.find(old_tag)
        seg_end = mtxt.find('。', idx) + 1
        new_seg = ('**✅ 3158 输出等价类闭环（2026-10-08）**：unembed 谱平坦（e99=97%% D、软零 2.9-3.5%%）；'
                   '预算比 null/rand 1.03–1.33 三模型全 absent（≥10× 预期证伪）；等价类=极薄邻域'
                   '（B 臂 d_rel 1.7%% @ KL≤0.0008）；弹性响应曲线跨模型指纹 0.9995；协议发现 HF 末槽已含 '
                   'final norm；res summary fa5cca12/seal e0c60629/ledger n=%d。'
                   '下一步 3159=**G4-P2 等价类动力学**（软零方向逐层流向+re-emergence，GPU 注入）。' % len(ms))
        mtxt2 = mtxt[:idx] + new_seg + mtxt[seg_end:]
        open(MEMORY, 'wb').write(mtxt2.encode('utf-8'))
        out.append('memory: replaced 3158-next tag -> done state')
    else:
        mtxt2 = mtxt + ('\n- 3158 G4-P1 输出等价类闭环(2026-10-08)：读出谱平坦无子空间商；下一步 3159=等价类动力学 P2。\n')
        open(MEMORY, 'wb').write(mtxt2.encode('utf-8'))
        out.append('memory: appended fallback line')

# ============ 5. 自检 ============
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert any(m.get('phase') == 3158 for m in led2['measurements'])
mtxt2 = open(MEMO, 'rb').read().decode('utf-8')
assert sec_marker in mtxt2
assert 'Phase 3159 预注册' in mtxt2
out.append('SELF-CHECK OK: ledger has 3158, memo has 3158 section + 3159 prereg')

with io.open(os.path.join(TMP, 'p3158_closeout_out.txt'), 'w', encoding='utf-8') as f:
    f.write(chr(10).join(out))
print('written')
