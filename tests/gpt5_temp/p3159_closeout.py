# -*- coding: utf-8 -*-
# 3159 closeout 五写: ledger append + MEMO(3159 节 + 3160 预注册) + daily + workspace MEMORY + self-check
# 幂等: ledger by phase; MEMO by 节标题; daily by marker; MEMORY by tag 替换
import io, json, os, shutil, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MDIR_MEM = os.path.join(ROOT, '.workbuddy', 'memory')
NOW = time.strftime('%Y-%m-%d %H:%M')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3159_closeout_out.txt')
out = []

# ---- 快照 ----
snap_dir = os.path.join(ROOT, 'tests', 'gpt5_temp', '_snap_3159')
os.makedirs(snap_dir, exist_ok=True)
for src in (LEDGER, MEMO):
    shutil.copy2(src, os.path.join(snap_dir, os.path.basename(src) + '.bak'))
out.append('snapshot: ledger+memo copied to _snap_3159')

# ============ 1. Ledger（幂等 by phase） ============
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms = led['measurements']
if any(m.get('phase') == 3159 for m in ms):
    out.append('ledger: 3159 already present, skip')
else:
    entry = {
        'phase': 3159, 'name': 'g4p2_equivalence_dynamics', 'line': 'G',
        'date': '2026-10-09', 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': 'g4p2_fingerprint_consistent|stable_0/3',
        'detail': ('G4-P2 equivalence-class dynamics: inject readout-spectrum directions at '
                   'slot L_mid=round(0.5*NL) (hook on block L_mid-1 output, last token), '
                   '16 anchors (=3158 anchor_idx) x arms{bottom64/top64/rand}x8 dirs x 6 '
                   'alphas rel ||h_mid||; Jeffreys KL from forward logits; budgets log-log '
                   'interpolation at KL=0.1. ratio_mid null/rand = 0.923/0.824/0.935 (4b/14b/'
                   'glm4) -> dynamics_destroyed 3/3 (gate >=3 stable): readout near-isotropy '
                   'is COMPUTED by remaining layers, not inherited from W_U spectrum; '
                   'top-sigma direction is ACTIVELY CONSUMED: share_top falls 0.999->0.27->'
                   '0.11 within 1-2 post-injection blocks (gain -0.913/-0.921/-0.942), '
                   'independent of anchor massive-norm (corr -0.20..0.07); bottom re-emergence '
                   'mild 0.0007->0.056/0.054/0.025; passthrough mid/kout budgets 2.5-2.9x '
                   '(null arm, top arm 2.9-5.4x); fingerprints KL-curve 0.9983-0.9994, '
                   're-emergence curve 0.9526-0.9755 (gate 0.8, all pass); addendum: '
                   'big-drop at first post-injection block (L19/L21/L21), per-anchor '
                   'consumption uniform; methodology: transformers output_capturing hook '
                   'semantics (slot i = block i-1 output, last slot post-norm), batch-size '
                   'changes bf16 kernel path (b16rel up to 2.2e-2) -> batch=1 anchor bitwise '
                   'vs 3157 + unified batch=6 for base/injection; determinism: base1 bitwise '
                   'x2, anchor3157 bitwise x3, pre-slot rel <1e-5, dir orth 4e-9; '
                   'per-model res/seal: 4b dccd4753/0fbb67c0, 14b 7c0c8cc6/a6320757, '
                   'glm4 6db4b360/d18055c4, summary f9c1fe35/db48b8a9; npz '
                   '8d282e5b/8394c72e/5267689b; addendum a3ba83c9'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: appended 3159 (n=%d) chain_sha8=%s' % (len(ms), led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')          # BOM 字符保留在串首
sec_marker = '## Phase 3159: 等价类动力学（G4-P2）'
if sec_marker in txt:
    out.append('memo: 3159 section already present, skip')
else:
    block = r'''## Phase 3159: 等价类动力学（G4-P2）[__NOW__]

**主判决：`g4p2_fingerprint_consistent|stable_0/3`——三模型一致 dynamics_destroyed（ratio_mid 0.923/0.824/0.935，门 ≥3=stable），读出层近各向同性不是 inherited 谱性质，而是被剩余层动力学计算出来的；指纹 KL 曲线 0.9983–0.9994、re-emergence 曲线 0.9526–0.9755 全过 0.8 门。**

### 设计与执行
- 预注册（3158 closeout 冻结，观测前）：锚 = 3157 H 槽 L_mid=round(0.5·NL) 16 行（=3158 anchor_idx 复用）；GPU 注入前向（hook 替换 last-token 态，块 L_mid−1 输出=槽 L_mid）；u 三臂同 3158（bottom64/top64/rand 各 8，复用 3158 npz 方向，断言单位范数+正交 4e-9）；α 相对 ‖h_mid‖ 扫 6 值 [0.003..1.0]；KL=Jeffreys float64（数值前向）；预算=log-log 插值 KL=0.1，删失=上限。
- 方法论两坑入库：① 本机 transformers 新版 hidden_states 捕获=output_capturing hook（槽 i=块 i−1 输出、末槽被 post-norm 覆盖——与 3158 实测一致）；② **batch 尺寸改变 bf16 kernel 数值路径**（batch=1 vs 6 相对差至 2.2e-2）→ 确定性协议改为：batch=1 锚前向 vs 3157 **bitwise**（×3 成立）+ 基线/注入统一 batch=6 公平对比 + batch 效应量化记录。
- 修 2 处：decoder layer 返回裸 Tensor（output[0] 取到 batch 切片 → IndexError）；summary 段 stables list 笔误。
- runtime 39.2/1729.3/1081.0s；base1 bitwise×2、pre-slot rel<1e-5、锚行重建 vs 3157 npz 逐元素一致。

### 五发现（重复强调）
1. **ratio_mid=0.923/0.824/0.935 → dynamics_destroyed 3/3**（门 ≥3=stable）：bottom-σ 方向在 mid 层注入**不比随机方向更被容忍**——3158 读出谱平坦（无 inherited 零空间）+ 本 Phase 剩余层也不保持软零性 → **各向同性是动力学主动计算的**。这是比 stable 更强的否定：商结构既不在谱里也不在传递里。
2. **top-σ 方向被剩余层强烈消耗**：注入后 share_top 0.999→0.27→0.11（前 1–2 块完成主要消耗，big-drop 层=L_mid+1），终层 gain **−0.913/−0.921/−0.942**——动力学把读出最敏感方向旋转走 90%+，且与锚 massive 程度无关（corr −0.20~0.07）→ 普适层动力学，非 massive 通道特异。
3. **bottom 方向轻度 re-emerge**：0.0007→0.056/0.054/0.025（+2.5–5.6%）——方向性回旋存在但量级太小，不构成商结构恢复；qwen 双胞胎几乎逐位一致（0.0558/0.0537），glm4 约减半。
4. **预算传递 pass-through**：mid/KOUT 预算 null 臂 2.51/2.84/2.48、top 臂 2.92/5.40/5.07——剩余层缓冲一切方向 2.5×+，top 方向被缓冲最多（与消耗一致）；消耗后三臂 KL 曲线同形（指纹 0.998+）→ 消耗不改变输出敏感性排序，读出谱形状由末端再放大（3156：readout 层范数恢复）+前段消耗共同决定。
5. **动力学整形方式本身是跨模型不变量**：KL 曲线对 0.9983–0.9994、re-emergence 曲线对 0.9526–0.9755——三个不同参数实现以相同形状主动重整化注入方向。

### 锚
4b res **dccd4753** seal 0fbb67c0（disk 后补）；14b res **7c0c8cc6** seal a6320757；glm4 res **6db4b360** seal d18055c4；summary res **f9c1fe35** seal db48b8a9。collect npz：`8d282e5b`/`8394c72e`/`5267689b`；addendum（4b）`a3ba83c9`。ledger n=**311**。产物 `phase3159\g4p2_equivalence_dynamics\{qwen3-4b,qwen3-14b,glm4,summary}\`。

### 预注册 Phase 3160：G4-P3 读出方向消耗的机制判别
- 假设：3159 证 top-σ 注入在剩余第 1–2 块被旋转消耗（0.999→0.27→0.11，与 massive 无关）；P3 问消耗载体：**MLP 压缩 vs attention 再分配**。
- 设计：(1) 零 GPU：SHARE 逐层分位曲线（消耗层位谱+每层降幅分布，三模型）；(2) GPU 层消融判别（4 锚×top 方向×α=0.1）：hook 置零块 L_mid+1 / L_mid+2 的 MLP 输出（残差流恒等替换）重跑注入——share_top 恢复 ≥0.5 → MLP 主因；<0.1 → attention 主因；之间 → 混合；(3) 位移去向：δh(L_mid+2) 在 3156 rank-1 massive 轴上的投影份额（吸收 vs 弥散）；(4) 跨模型消耗曲线指纹（截 min NH，Pearson ≥0.8）。
- 门：三分类判决 + 指纹门；GPU 预算 ~5min/模型。
'''
    block = block.replace('__NOW__', NOW)
    txt = txt.rstrip('\n') + '\n' + block
    open(MEMO, 'wb').write(txt.encode('utf-8'))
    out.append('memo: appended 3159 section + 3160 prereg')

# ============ 3. Daily（幂等 by marker） ============
daily = os.path.join(MDIR_MEM, '2026-10-09.md')
marker = '3159 等价类动力学闭环'
if os.path.exists(daily) and marker in io.open(daily, encoding='utf-8').read():
    out.append('daily: 3159 already present, skip')
else:
    n_ms = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    line = ('- 3159 等价类动力学闭环（G4-P2）：mid 层注入读出谱方向 16 锚×24 方向×6α×3 模型；'
            'ratio_mid 0.92/0.82/0.94 → dynamics_destroyed 3/3（各向同性是动力学算出来的）；'
            'top 方向被剩余层消耗 91–94%%（big-drop=注入后第 1 块）；bottom 轻度 re-emerge '
            '2.5–5.6%%；指纹 KL 0.998+/re-emerge 0.95+；方法论=新 transformers 捕获 hook 语义+'
            'batch 效应（b16rel 至 2.2e-2）→ batch=1 锚 bitwise vs 3157；ledger n=%d。'
            '3160 预注册=消耗机制判别（MLP vs attention 层消融）。\n' % n_ms)
    with io.open(daily, 'a', encoding='utf-8') as f:
        if not os.path.exists(daily) or os.path.getsize(daily) == 0:
            f.write('# 2026-10-09 工作日志\n\n')
        f.write(line)
    out.append('daily: appended 3159 line')

# ============ 4. Workspace MEMORY（tag 替换） ============
mp = os.path.join(MDIR_MEM, 'MEMORY.md')
mtxt = io.open(mp, encoding='utf-8').read()
tag_old = '下一步 3159=**G4-P2 等价类动力学**'
if '3159 等价类动力学闭环' in mtxt:
    out.append('workspace MEMORY: 3159 already present, skip')
else:
    seg_old = ('**✅ 3158 输出等价类闭环（2026-10-08）**')
    i = mtxt.rfind(seg_old)
    assert i >= 0, '3158 segment not found in workspace MEMORY'
    new_seg = ('**✅ 3159 等价类动力学闭环（2026-10-09）**：mid 层注入读出谱方向（16 锚×24 方向×6α×3 模型）；'
               'ratio_mid 0.92/0.82/0.94 → dynamics_destroyed 3/3（各向同性是剩余层算出来的，非 inherited）；'
               'top 方向被动力学消耗 91–94%（big-drop=注入后第 1 块，与 massive 无关）；bottom 轻度 re-emerge '
               '2.5–5.6%；pass-through mid/KOUT 2.5–2.9×；指纹 KL 0.998+/re-emerge 0.95+；'
               '方法论：新 transformers 捕获=hook（槽 i=块 i−1 输出+末槽 post-norm）、batch 效应 b16rel 至 2.2e-2 → '
               'batch=1 锚 bitwise vs 3157 + batch=6 统一注入；res 4b dccd4753/14b 7c0c8cc6/glm4 6db4b360/'
               'summary f9c1fe35/ledger n=311。下一步 3160=**G4-P3 消耗机制判别**（MLP vs attention 层消融+massive 轴去向）。')
    mtxt = mtxt[:i] + new_seg + mtxt[i + len(seg_old):]
    open(mp, 'wb').write(mtxt.encode('utf-8'))
    out.append('workspace MEMORY: 3159 segment replaced 3158 head')

# ============ 5. Self-check ============
ok = []
ok.append(('ledger_3159', any(m.get('phase') == 3159 for m in
                              json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])))
ok.append(('memo_3159', sec_marker in open(MEMO, 'rb').read().decode('utf-8')))
ok.append(('daily_3159', marker in io.open(daily, encoding='utf-8').read()))
ok.append(('memory_3159', '3159 等价类动力学闭环' in io.open(mp, encoding='utf-8').read()))
b = open(MEMO, 'rb').read()
ok.append(('memo_bom', b[:3] == b'\xef\xbb\xbf'))
bad = [k for k, v in ok if not v]
out.append('SELF-CHECK: %s' % ('OK' if not bad else 'FAIL %s' % bad))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out))
print('CLOSEOUT DONE')
