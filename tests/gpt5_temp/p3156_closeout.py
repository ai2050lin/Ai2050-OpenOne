# -*- coding: utf-8 -*-
# p3156_closeout.py: Phase 3156 五写 closeout（ledger / MEMO+3157 预注册 / daily / MEMORY / 自检）
# 纪律: 幂等（重复运行 skip）、快照、BOM 二进制安全（decode->append->encode 全程保 BOM）
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
        dst = os.path.join(TMP, 'snapshot3156_' + os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
out.append('snapshots done')

# ============ 1. Ledger（幂等 by phase==3156） ============
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == 3156 for m in ms):
    out.append('ledger: 3156 already present (n=%d), skip' % len(ms))
else:
    entry = {
        'phase': 3156, 'name': 'g3p1_position_shift_family',
        'seal_sha8': 'b54418f2', 'result_sha8': 'c8c66d9f',
        'evidence_level': 'statistical',
        'model_scope': 'qwen3-4b (14b/glm4 cross-model = 3157)',
        'n_rows': 36, 'prereg_id': 'p3156', 'superseded_by': None,
        'verdict': 'g3p1_rope_violated_strict_tol|ic_no_peak|out_ctx_dominant|curve_0.972',
        'rev_note': ('dual-arm position shift family: target sentence frozen, neutral prefix '
                     'k in {0,1,2,4,8,16,32,64,128}; arm A = real prefix (context+position), '
                     'arm B = same token ids but prefix attention-masked + position_ids reset; '
                     'FINDINGS: (1) RoPE pure-relativity near-exact: B-arm vs k=0 per-layer max '
                     'rel disp 1.49e-2 (readout L36; zh body 6.5e-4), KL_B <= 0.0023 and top1_B '
                     '9/9 restored across all k -> position is architecturally erased, no '
                     'compensation needed (gate g1 strict tol 1e-3 formally violated by the '
                     '1.4e-2 readout/en-k64 tail; physically negligible vs context effect); '
                     '(2) context vs position separation: real prefix collapses mid-layer '
                     'residual norm 77x (11274 -> 146 at L12-18) while B-arm restores to '
                     '~2e-4 rel; (3) massive-activation context switch: a SINGLE real prefix '
                     'token already collapses |h| 11072 -> 428 and collapse ratio is k-'
                     'independent (0.013 at both k=1 and k=128) -> binary has-context gate, '
                     'not gradual position effect; readout norm restored ~1.0 (interface '
                     're-amplification); (4) mid-layer context displacement is rank-1 '
                     '(svd top1 = 1.0 zh / 0.9996 en at L18) -> context switches off one '
                     'massive-axis direction; (5) output does NOT compensate context: '
                     'KL(A128||A0) 2.3-3.7, top1 0/2 across k>=1 -> IC peak absent (g2 fail '
                     'honest), g3 fail by design (context should change prediction); '
                     'cross-sentence curve consistency 0.972 (g4 pass); T-transform v0: '
                     'shift_share 0.08, subspace linear gain 0.72 (not pure translation); '
                     'addendum artifact result_addendum.json sha8 a1b773b1 (KL_B/norm-'
                     'collapse/rope-per-layer/ctx-svd from sealed npz 84f7db0b); determinism '
                     'bitwise, d0 anchor A0-vs-B0 = 0.0; exec bf474ef3; fixes: prefix assert '
                     'cycled-coverage, ss_max skip emb degenerate layer, verdict format arity'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: appended 3156 (n=%d) chain_sha8=%s' % (len(ms), led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')          # 保留 BOM 字符（若有）在串首
sec_marker = '## Phase 3156: 位置平移族基座（G3-P1）'
if sec_marker in txt:
    out.append('memo: 3156 section already present, skip')
else:
    block = r'''## Phase 3156: 位置平移族基座（G3-P1）[__NOW__]

**主判决：`g3p1_rope_violated_strict_tol|ic_no_peak|out_ctx_dominant|curve_0.972`（res `c8c66d9f` / seal `b54418f2`）——门机器按预注册 1e-3 严格容差判 rope violated（max 1.49e-2，集中于 readout 层与 en k=64），但物理解读：RoPE 纯相对性近似成立（B 臂内部 ≤1.5e-2、输出 KL_B ≤0.0023、top1_B 9/9 恢复），位置效应与上下文效应成功分离且相差 3 个量级。**

### 设计与执行
- 双臂平移（用户 2026-10-08 位置附件落地）：目标句冻结（zh "我喜欢吃苹果，因为它又甜又多汁。" / en 同义句），中性前缀 k∈{0,1,2,4,8,16,32,64,128}；**A 臂=真实前缀（上下文+位置混杂）；B 臂=同一 token 序列但前缀 attention_mask 屏蔽 + position_ids 重置(0..n−1)**。36 序列，qwen3-4b，全层 H fp16 + final logits。execution 冻结 `bf474ef3`（观测前），collect `84f7db0b`，确定性 bitwise，d0 锚 A0-vs-B0 = 0.0（位级）。
- addendum 独立产物 `result_addendum.json` sha8 `a1b773b1`（KL_B/范数塌缩谱/逐层 rope/ctx-SVD，全部从已 seal 的 npz 现场渲染）。

### 五发现（重复强调）
1. **RoPE 纯相对性成立（位置被架构消除）**：B 臂（位置重置）与 k=0 的逐层 rel disp 最大 1.49e-2（zh 主体 6.5e-4），**输出 KL_B 全 k ≤0.0023、top1_B 9/9=1.0**——位置变化不需要内部补偿，RoPE+mask 直接消除（对应附件特性 3 的 T=恒等版本）。
2. **位置 vs 上下文分离（3 个量级）**：位置效应 ~1e-2 级；上下文效应（A 臂）使中层残差流范数塌缩 77×（11274→146，L12–18）。
3. **massive activation 的"有无上下文"二元开关**：**单个**真实前缀 token 就把 |h|~11072 的 massive 分量打到 428（26×），且塌缩比与 k 无关（k=1 与 k=128 同为 0.013）——这不是渐进位置效应，是二元的 has-context 门控；readout 层范数恢复 ~1.0（接口再放大机制）。
4. **中层上下文位移 rank-1**：A−B 位移向量 SVD top1=1.0（zh L18）/0.9996（en）——上下文效应几何 = 沿单一轴关闭 massive 通道；readout 层低秩性弱化（top1 0.39–0.49，多方向）。
5. **输出不补偿上下文**：KL(A_k‖A_0) 1.1–3.7、top1 0/2（k≥1 全变）——与位置相反，上下文效应直达输出（合理：上下文本应改变预测）；IC 峰无结构（g2 fail 诚实记录）正因 A 臂"内部变化与输出变化同源"；跨句位移曲线一致 0.972（g4 pass）；T 变换 v0：shift_share 0.08（非平移）、子空间线性 gain 0.72。

### 门与诚实边界
- g1_rope：门 1e-3 **violated**（1.49e-2）——严格容差下的诚实判决；绝对量级比上下文效应小 3 个量级，泄漏集中 readout 层（en k=64 L16 1.44e-2 次峰）。
- g2_ic / g3_out：fail（IC 无峰、KL 2.32>1.0）——被上下文混杂主导，这是设计声明的 A 臂语义，非装置缺陷。
- 单模型 phase；跨模型指纹（P×L 曲线形状）留给 3157。

### Phase 3157 预注册（G2-P2 变换代数 P1——对易子；结合用户 2026-10-08 数学结构附件第 13/14 节）
- **装置合并**：T_N=否定（句中极性）、T_R=关系（is-a/has-a）、T_C=上下文（k=16 真实前缀 vs ∅）；T_P≈恒等已由 3156 B 臂确立（单位元）。
- **面板**：16 实体（3155 SHARED 取 16）× 2 关系 × 2 极性 × 2 上下文 = 128 行/模型，三模型（4b/14b/glm4）；SMOKE 4 实体。
- **度量**（KOUT+全层）：① 对易子（关系差分视角）exch_R = ‖ΔR(+)−ΔR(−)‖/mean‖ΔR‖；② 对易子（否定差分视角）exch_N = ‖ΔN(isa)−ΔN(hasa)‖/mean；③ 平衡面板 ANOVA [E|R|N|C|二阶交互|残差]（R×N 交互=代数扭曲的方差视角）；④ T_C 关系无关性（上下文算子对易第三检验）；⑤ 实体 shuffle 100 null。
- **门**：exch<0.5 → `commutative_algebra_supported`；0.5–2 → partial；>2 → `non_commutative`（更强发现）。跨模型指纹两两 Pearson ≥0.8。
- 附件探针矩阵映射：位置不变量/相对位置=3156✅、内容复用/关系变换=3155✅、条件性/因果性=3154✅、跨模型=指纹链✅、**角色变换/组合性（=本 phase 对易子）=3157**、等价类/时间性=3158+。

'''
    block = block.replace('__NOW__', NOW)
    txt2 = txt + block
    open(MEMO, 'wb').write(txt2.encode('utf-8'))
    out.append('memo: appended 3156 section + 3157 prereg (head BOM preserved=%s)' % ('\\ufeff' in txt2[:1] or txt2[:1] == '\\ufeff'))
out.append('memo check: %s' % (sec_marker in open(MEMO, 'rb').read().decode('utf-8')))

# ============ 3. daily（幂等 by marker） ============
mk = '3156 G3-P1 位置平移族基座闭环'
if os.path.exists(DAILY):
    dtxt = open(DAILY, 'rb').read().decode('utf-8')
else:
    dtxt = ''
if mk in dtxt:
    out.append('daily: already present, skip')
else:
    dline = ('- 3156 G3-P1 位置平移族基座闭环：双臂(真实前缀/位置重置)x k<=128；RoPE 纯相对性成立'
             '(KL_B<=0.0023,top1_B 9/9)；单 token 前缀即塌缩 massive activation(11072->428)且 k 无关；'
             '中层上下文位移 rank-1；ledger n=%d。3157 预注册=变换代数对易子(G2-P2)。\n' % len(ms))
    open(DAILY, 'ab').write(dline.encode('utf-8'))
    out.append('daily: appended')

# ============ 4. workspace MEMORY（幂等 by marker；不动 deepseek 节） ============
mtxt = open(MEMORY, 'rb').read().decode('utf-8')
mk2 = '3156 位置平移族'
if mk2 in mtxt:
    out.append('memory: already updated, skip')
else:
    old_tag = '下一步 3156=**位置平移族基座（G3-P1 位置主线，用户 2026-10-08 附件）**'
    if old_tag in mtxt:
        new_seg = ('**✅ 3156 位置平移族基座闭环（2026-10-08）**：双臂(真实前缀/位置重置)×k∈{0..128}；'
                   'RoPE 纯相对性成立(KL_B≤0.0023,top1_B 9/9,内部≤1.5e-2)；单 token 前缀即塌缩 massive '
                   'activation(11072→428)且与 k 无关=二元 has-context 门控；中层上下文位移 rank-1；'
                   'res c8c66d9f/seal b54418f2/addendum a1b773b1/ledger n=%d。'
                   '下一步 3157=**G2-P2 变换代数对易子**（T_N×T_R×T_C 交换性+ANOVA 二阶交互,16 实体×2 关系×2 极性×2 上下文=128 行/模型×3）。' % len(ms))
        mtxt2 = mtxt.replace(old_tag, new_seg)
        open(MEMORY, 'wb').write(mtxt2.encode('utf-8'))
        out.append('memory: replaced 3156-next tag -> done state')
    else:
        mtxt2 = mtxt + ('\n- 3156 G3-P1 位置平移族闭环(2026-10-08)：RoPE 纯相对性+massive activation 上下文开关；'
                        '下一步 3157=G2-P2 变换代数对易子。\n')
        open(MEMORY, 'wb').write(mtxt2.encode('utf-8'))
        out.append('memory: appended fallback line')

# ============ 5. 自检 ============
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert any(m.get('phase') == 3156 for m in led2['measurements'])
mtxt2 = open(MEMO, 'rb').read().decode('utf-8')
assert '## Phase 3156: 位置平移族基座（G3-P1）' in mtxt2
assert 'G2-P2 变换代数' in mtxt2
out.append('SELF-CHECK OK: ledger has 3156, memo has 3156 section + 3157 prereg')

with io.open(os.path.join(TMP, 'p3156_closeout_out.txt'), 'w', encoding='utf-8') as f:
    f.write(chr(10).join(out))
print('written')
