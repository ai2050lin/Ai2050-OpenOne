# -*- coding: utf-8 -*-
# 3160 closeout 五写: ledger append + MEMO(3160 节 + 3161 预注册) + daily + workspace MEMORY
#                     (含修复 3159 段尾残留错乱) + self-check
# 幂等: ledger by phase; MEMO by 节标题; daily by marker; MEMORY by 3160 段头
import io, json, os, shutil, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MDIR_MEM = os.path.join(ROOT, '.workbuddy', 'memory')
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3160', 'g4p3_consumption_mechanism')
NOW = time.strftime('%Y-%m-%d %H:%M')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_closeout_out.txt')
out = []

def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# ---- 产物 disk sha 现场渲染 ----
sha = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    sha['res_' + m] = sha8_file(os.path.join(PDIR, m, 'result.json'))
    sha['npz_' + m] = sha8_file(os.path.join(PDIR, m, 'collect.npz'))
sha['res_summary'] = sha8_file(os.path.join(PDIR, 'summary', 'result_summary.json'))
sha['res_zero'] = sha8_file(os.path.join(PDIR, 'zero', 'result_zero.json'))
sha['npz_zero'] = sha8_file(os.path.join(PDIR, 'zero', 'collect_zero.npz'))
out.append('disk sha: %s' % json.dumps(sha, indent=0))

# ---- 快照 ----
snap_dir = os.path.join(ROOT, 'tests', 'gpt5_temp', '_snap_3160')
os.makedirs(snap_dir, exist_ok=True)
for src in (LEDGER, MEMO):
    shutil.copy2(src, os.path.join(snap_dir, os.path.basename(src) + '.bak'))
out.append('snapshot: ledger+memo copied to _snap_3160')

# ============ 1. Ledger（幂等 by phase, 弹性并发断言） ============
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms = led['measurements']
n0 = len(ms)
if any(m.get('phase') == 3160 for m in ms):
    out.append('ledger: 3160 already present, skip')
else:
    entry = {
        'phase': 3160, 'name': 'g4p3_consumption_mechanism', 'line': 'G',
        'date': '2026-10-09', 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': 'g4p3_attention_reallocation_primary|fp_ok',
        'detail': ('G4-P3 consumption mechanism: zero mode (3159 SHARE top-arm quantile '
                   'curves) big-drop at first post-injection block for all 3 models '
                   '(q50 drop 0.331/0.319/0.229), gain -0.936/-0.945/-0.973, consumption-'
                   'window fingerprint fpmin 0.9945 (W=19; raw slot alignment 0.38 diluted '
                   'by L_mid 18 vs 20 misalignment); GPU layer ablation (4 anchors x 6 '
                   'top64 dirs x alpha=0.1, batch rows = dirs, batch=6 kernel path): '
                   'zeroing mlp output of blocks L_mid / L_mid+1 / L_mid+1+2 recovers '
                   'share_top(NL) by only -0.008..+0.006 (4b +0.0019/+0.0046/+0.0058, '
                   '14b -0.0031/-0.0064/-0.0076, glm4 -0.0017/-0.0001/-0.0011; gate 0.1) '
                   '-> attention_reallocation_primary 3/3: MLP contributes nothing to the '
                   'consumption, rotation is executed by attention reallocation; '
                   'destination of dh(L_mid+2) is DIFFUSE: share on anchor massive dim '
                   'e_d1 0.014/0.011/0.018, top-8 massive dims 0.027/0.018/0.028, '
                   'cos(dh,h_mid) -0.007..0.027 -> neither massive absorption nor anchor-'
                   'direction; summary fingerprint (none-config q50, L_mid-offset aligned '
                   'W=19) 0.9911/0.9912/0.9940 all pass 0.8 (raw slot align 0.381/0.377/'
                   '0.994 control); DESIGN DEVIATION (frozen before observation): prereg '
                   '3156 rank-1 axis not reproducible from 3156 npz (en A-arm last-token '
                   'L7 bitwise identical across k; best row-wise SVD share 0.685 vs '
                   'reported 0.9999 -> axis computed live, not persisted) -> replaced by '
                   '3157 anchor massive dim d1=0/731/2319 (4b d1=0 = 3156 collapse dim); '
                   'methodology: per-block mlp hooks with block-set gating (global-set '
                   'bug caught pre-run); determinism: anchor batch=1 bitwise vs 3157 '
                   '4/4 x3 models, pre-slot rel < 1e-5; per-model res/seal: 4b 6853c671/'
                   '3c4a601b, 14b 63845c74/39f3d025, glm4 421448b9/eebdff76, summary '
                   'a52e2ddd/3ce5cc3b, zero a9435ded/3e1501f0; npz disk 26da4d18/'
                   '4231760d/725de043/3ce5cc3b-summary'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    out.append('ledger: appended 3160 (n=%d, was %d) chain_sha8=%s' % (n1, n0, led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')          # BOM 字符保留在串首
sec_marker = '## Phase 3160: 消耗机制判别（G4-P3）'
if sec_marker in txt:
    out.append('memo: 3160 section already present, skip')
else:
    block = r'''## Phase 3160: 消耗机制判别（G4-P3）[__NOW__]

**主判决：`g4p3_attention_reallocation_primary|fp_ok`——三模型一致：top-σ 注入方向的消耗由 attention 再分配执行，MLP 无贡献（置零块 L_mid/+1/+1+2 的 MLP 输出 recover 仅 −0.008~+0.006，≪0.1 门）；δh(L_mid+2) 去向弥散（massive 维度份额 0.011–0.018、锚态方向 cos ≤0.027）——消耗是弥散式旋转，不是 massive 吸收。summary 指纹（消耗段对齐 W=19）0.9911/0.9912/0.9940 全过 0.8 门。**

### 设计与执行
- 预注册（3159 closeout 冻结，观测前）四项：① 零 GPU 分位曲线+层位谱；② GPU 层消融三分类（recover ≥0.5 MLP 主因 / <0.1 attention 主因）；③ δh 去向；④ 跨模型消耗曲线指纹。
- zero 模式：3159 SHARE top 臂逐层 q10/q50/q90；big-drop 三模型一致=注入后第 1 块（q50 drop 0.331/0.319/0.229）；gain −0.936/−0.945/−0.973；指纹**消耗段对齐（相对 L_mid 偏移，W=19）fpmin 0.9945**；raw 槽号对齐 0.38=错位稀释对照（L_mid 18 vs 20）。
- GPU 模型模式：4 锚（3159 anchor_idx 取 [0,42,85,127]）×6 top64 方向（=batch 行，batch=6 kernel 路径与 3159 一致）×α=0.1 相对 ‖h_mid‖；4 消融配置 none/mlp_mid/mlp_mid1/mlp_mid1_2（per-block mlp hook 按块号门控置零——**全局集合 bug 在运行前捕获修复**）；每配置 base+注入同路径前向，dh=pert−base。
- **设计偏差（观测前冻结）**：预注册的 3156 rank-1 轴不可从 3156 npz 复现——en A 臂 last-token L7 逐位恒等于 k=0、逐行位移 SVD 最高 share 0.685 ≪ 当时报告 0.9999，判定当时轴为现场计算未落盘 → 改用 3157 锚态 massive 维度 e_d1（d1=0/731/2319；4b d1=0 与 3156 塌缩维度一致），三模型统一可用。
- runtime 10.6/297.3/166.2s；锚 batch=1 bitwise vs 3157 4/4 ×3 模型；pre-slot rel<1e-5。

### 四发现（重复强调）
1. **MLP 置零 recover ≈ 0 → attention_reallocation_primary 3/3**：recover(mlp_mid1)=+0.0046/−0.0064/−0.0001、recover(mlp_mid1_2)=+0.0058/−0.0076/−0.0011（4b/14b/glm4），全部 ≪0.1 门——**旋转消耗的载体是 attention 再分配，MLP 压缩贡献为零**（置零甚至轻微更差）。
2. **δh 去向弥散**：dh(L_mid+2) 在 massive 维度 e_d1 份额 0.014/0.011/0.018、top-8 massive 维度集 0.027/0.018/0.028、与锚态 cos −0.007~0.027——**消耗不是把方向转进 massive 通道，也不是沿锚态主轴，而是弥散式再分配**。
3. **消耗动力学形状跨模型不变**：none 配置 q50 消耗段指纹 0.9911/0.9912/0.9940（对齐 W=19）；与 zero 模式 0.9945 互证——3159 的五发现之五（整形方式=跨模型不变量）在 head 前层级再次成立。
4. **口径教训入库**：跨模型逐层曲线指纹必须按 L_mid 偏移对齐（错位 2 层把 0.99 稀释到 0.38）；14b vs glm4（L_mid 同 20）raw 对齐 0.994 交叉验证了诊断。

### 锚
zero res **a9435ded** seal 3e1501f0；4b res **6853c671** seal 3c4a601b；14b res **63845c74** seal 39f3d025；glm4 res **421448b9** seal eebdff76；summary res **a52e2ddd** seal 3ce5cc3b。3160 disk: zero res/__ZERO_RESAK__ npz/__ZERO_NPZK__；4b res/__R4B__ npz/__N4B__；14b res/__R14B__ npz/__N14B__；glm4 res/__RGLM__ npz/__NGLM__；summary res/__RSUM__（无 npz）。ledger n=**312**。产物 `phase3160\g4p3_consumption_mechanism\{zero,qwen3-4b,qwen3-14b,glm4,summary}\`。

### 预注册 Phase 3161：G4-P4 消耗的 attention 头归因
- 假设：3160 证消耗载体=attention 再分配（MLP 无贡献）且 δh 弥散；P4 问执行旋转的头子集是否局部化。
- 设计：(1) GPU 逐头消融：对 big-drop 块（L_mid）全部注意力头逐头置零（hook 在 self_attn 输出，按 (b, T, head·dh:(head+1)·dh) 切片置零，batch 行=头配置压缩前向）；4 锚×top64 方向 6×α=0.1（同 3160 口径）；(2) head recover 曲线 → top-4 头集中度 = Σrecover(top4)/Σrecover(全部>0)；(3) 门：集中度 ≥0.5 → localized_heads / <0.2 → distributed_heads / 之间 → weakly_localized；(4) 跨模型头层位分布（相对深度）描述性对比 + 消耗曲线指纹（对齐口径）。
- 门：三分类判决 + 指纹门；GPU 预算 ~10min/模型（qwen3 32 头/块 ×GQA 注意 KV 头不切分 o_proj 输入维度）。
'''
    block = block.replace('__NOW__', NOW)
    for k, v in (('__ZERO_RESAK__', sha['res_zero']), ('__ZERO_NPZK__', sha['npz_zero']),
                 ('__R4B__', sha['res_qwen3-4b']), ('__N4B__', sha['npz_qwen3-4b']),
                 ('__R14B__', sha['res_qwen3-14b']), ('__N14B__', sha['npz_qwen3-14b']),
                 ('__RGLM__', sha['res_glm4']), ('__NGLM__', sha['npz_glm4']),
                 ('__RSUM__', sha['res_summary'])):
        assert block.count(k) == 1, ('placeholder count', k, block.count(k))
        block = block.replace(k, v)
    txt = txt.rstrip('\n') + '\n' + block
    open(MEMO, 'wb').write(txt.encode('utf-8'))
    out.append('memo: appended 3160 section + 3161 prereg')

# ============ 3. Daily（幂等 by marker） ============
daily = os.path.join(MDIR_MEM, '2026-10-09.md')
marker = '3160 消耗机制判别闭环'
if os.path.exists(daily) and marker in io.open(daily, encoding='utf-8').read():
    out.append('daily: 3160 already present, skip')
else:
    n_ms = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    line = ('- 3160 消耗机制判别闭环（G4-P3）：zero 分位曲线 big-drop=注入后第 1 块（三模型一致，'
            '消耗段对齐指纹 0.9945）；GPU 层消融置零块 L_mid/+1/+1+2 的 MLP 输出 recover −0.008~+0.006 '
            '≪0.1 门 → attention_reallocation_primary 3/3（MLP 无贡献，旋转由 attention 再分配执行）；'
            'δh(L_mid+2) 去向弥散（e1 份额 0.011–0.018、cos_h ≤0.027，非 massive 吸收）；summary 指纹 '
            '0.9911/0.9912/0.9940；口径教训=跨模型逐层曲线须按 L_mid 偏移对齐（错位把 0.99 稀释到 0.38）；'
            '设计偏差=3156 rank-1 轴不可从 npz 复现 → 改用 3157 锚态 massive 维度；ledger n=%d。'
            '3161 预注册=attention 头归因（逐头置零，top-4 集中度三分门）。\n' % n_ms)
    with io.open(daily, 'a', encoding='utf-8') as f:
        f.write(line)
    out.append('daily: appended 3160 line')

# ============ 4. Workspace MEMORY（3160 段插入 + 3159 段尾残留清理） ============
mp = os.path.join(MDIR_MEM, 'MEMORY.md')
mtxt = io.open(mp, encoding='utf-8').read()
if '3160 消耗机制判别闭环' in mtxt:
    out.append('workspace MEMORY: 3160 already present, skip')
else:
    m9 = '**✅ 3159 等价类动力学闭环（2026-10-09）**'
    i9 = mtxt.rfind(m9)
    assert i9 >= 0, '3159 segment head not found in workspace MEMORY'
    end9 = '（MLP vs attention 层消融+massive 轴去向）。'
    ie = mtxt.find(end9, i9)
    assert ie >= 0, '3159 segment clean end not found'
    ie += len(end9)
    seg9_clean = mtxt[i9:ie]                       # 3159 干净段(截掉残留)
    seg10 = ('**✅ 3160 消耗机制判别闭环（2026-10-09）**：zero 分位曲线 big-drop=注入后第 1 块（三模型一致，'
             '消耗段对齐指纹 0.9945，raw 槽号对齐 0.38=L_mid 错位稀释对照）；GPU 层消融（4 锚×6 top 方向×α=0.1）'
             '置零块 L_mid/+1/+1+2 的 MLP 输出 recover −0.008~+0.006 ≪0.1 门 → **attention_reallocation_primary '
             '3/3**（MLP 无贡献，旋转由 attention 再分配执行）；δh(L_mid+2) 去向弥散（e1 份额 0.014/0.011/0.018、'
             'top8 0.027/0.018/0.028、cos_h ≤0.027，非 massive 吸收非锚态方向）；summary 指纹（对齐 W=19）'
             '0.9911/0.9912/0.9940；设计偏差：3156 rank-1 轴不可从 npz 复现（en L7 逐位恒等、最高 0.685≪0.9999）'
             '→ 观测前改用 3157 锚态 massive 维度 d1=0/731/2319；res zero a9435ded/4b 6853c671/14b 63845c74/'
             'glm4 421448b9/summary a52e2ddd/ledger n=312。下一步 3161=**G4-P4 attention 头归因**'
             '（逐头置零，top-4 集中度三分门）。')
    mtxt = mtxt[:i9] + seg9_clean + seg10
    open(mp, 'wb').write(mtxt.encode('utf-8'))
    out.append('workspace MEMORY: 3159 segment cleaned + 3160 appended (trailing orphans removed)')

# ============ 5. Self-check ============
ok = []
ok.append(('ledger_3160', any(m.get('phase') == 3160 for m in
                              json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])))
b = open(MEMO, 'rb').read()
t = b.decode('utf-8')
ok.append(('memo_3160', sec_marker in t))
ok.append(('memo_3161', '预注册 Phase 3161' in t))
ok.append(('daily_3160', marker in io.open(daily, encoding='utf-8').read()))
mp_txt = io.open(mp, encoding='utf-8').read()
ok.append(('memory_3160', '3160 消耗机制判别闭环' in mp_txt))
ok.append(('memory_orphan_cleaned', '。：unembed 谱平坦' not in mp_txt and
           '。：k∈{0,1,2,4' not in mp_txt))
ok.append(('memo_bom', b[:3] == b'\xef\xbb\xbf'))
bad = [k for k, v in ok if not v]
out.append('SELF-CHECK: %s' % ('OK' if not bad else 'FAIL %s' % bad))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('CLOSEOUT DONE')
