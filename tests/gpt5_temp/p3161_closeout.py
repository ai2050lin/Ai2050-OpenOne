# -*- coding: utf-8 -*-
# 3161 closeout 五写: ledger append + MEMO(3161 节 + 3163 预注册) + daily + workspace MEMORY
#                     + self-check。数字一律 result 现场渲染; MEMO 块 CRLF; 幂等 by phase/节标题/marker。
import io, json, os, shutil, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MDIR_MEM = os.path.join(ROOT, '.workbuddy', 'memory')
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3161', 'g4p4_head_attribution')
NOW = time.strftime('%Y-%m-%d %H:%M')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3161_closeout_out.txt')
out = []

def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def jload(p):
    return json.load(io.open(p, encoding='utf-8'))

# ---- 结果现场渲染 ----
R = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    R[m] = jload(os.path.join(PDIR, m, 'result.json'))
RS = jload(os.path.join(PDIR, 'summary', 'result_summary.json'))
SM = jload(os.path.join(PDIR, 'qwen3-4b', 'smoke', 'result.json'))
MS = ('qwen3-4b', 'qwen3-14b', 'glm4')

def f(x, n=4):
    return ('%.' + str(n) + 'f') % x

cls_str = '/'.join(R[m]['cls'] for m in MS)
ctrls = [R[m]['ctrl_all3'] for m in MS]
Ts = [R[m]['T'] for m in MS]
C4s = [R[m]['C4'] for m in MS]
rands = [R[m]['rand_ratio'] for m in MS]
effs = [R[m]['det']['efficacy_maxabs_Lmid1'] for m in MS]
rts = [R[m]['runtime_s'] for m in MS]
Hs = [R[m]['model']['H'] for m in MS]
nl_none = [R[m]['share_nl_none'] for m in MS]
nl_all3 = [R[m]['ctrl_all3'] * 0 for m in MS]  # placeholder unused
fpmin = RS['fpmin_q50_none']
fpmin_spec = RS['fpmin_sortedR']
W = RS['consumption_window']
agree = RS['gates']['class_agreement']
ctrl_ok = RS['gates']['ctrl']
verdict_main = RS['verdict']
smoke_verdict = SM['verdict']

# all3 vs none 早段瞬态差(4b 现场算, 描述性)
import numpy as np
z4b = np.load(os.path.join(PDIR, 'qwen3-4b', 'collect.npz'))
L_MID = int(z4b['l_mid'])
q_n = np.median(z4b['NONE_CURVE'].astype(np.float64).reshape(-1, z4b['NONE_CURVE'].shape[2]), axis=0)
q_a = np.median(z4b['ALL3_CURVE'].astype(np.float64).reshape(-1, z4b['ALL3_CURVE'].shape[2]), axis=0)
gap123 = [float(q_a[L_MID + k] - q_n[L_MID + k]) for k in (1, 2, 3)]
nl4b = float(q_n[-1]); nl4b_a3 = float(q_a[-1])

# ---- 产物 disk sha 现场渲染 ----
sha = {}
for m in MS:
    sha['res_' + m] = sha8_file(os.path.join(PDIR, m, 'result.json'))
    sha['npz_' + m] = sha8_file(os.path.join(PDIR, m, 'collect.npz'))
sha['res_summary'] = sha8_file(os.path.join(PDIR, 'summary', 'result_summary.json'))
sha['res_smoke'] = sha8_file(os.path.join(PDIR, 'qwen3-4b', 'smoke', 'result.json'))
sha['npz_smoke'] = sha8_file(os.path.join(PDIR, 'qwen3-4b', 'smoke', 'collect.npz'))
out.append('disk sha: %s' % json.dumps(sha, indent=0))
# result 内嵌 sha 一致性（seal 哈希的是 pre-seal 文件态, 磁盘最终文件含 seal 字段,
# 字节级复验由独立磁盘复核脚本做; 此处验证 verdict 尾部内嵌 res_sha8 与字段一致）
def _verdict_sha_ok(r):
    return r['verdict'].endswith('|sha8_' + r['res_sha8'])

for m in MS:
    assert _verdict_sha_ok(R[m]), ('verdict sha mismatch', m)
assert _verdict_sha_ok(RS), ('verdict sha mismatch', 'summary')
out.append('seal consistency: verdict-embedded sha OK 4/4')

# ---- 快照 ----
snap_dir = os.path.join(ROOT, 'tests', 'gpt5_temp', '_snap_3161')
os.makedirs(snap_dir, exist_ok=True)
for src in (LEDGER, MEMO):
    shutil.copy2(src, os.path.join(snap_dir, os.path.basename(src) + '.bak'))
out.append('snapshot: ledger+memo copied to _snap_3161')

# ============ 1. Ledger ============
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3161 for m in ms_):
    out.append('ledger: 3161 already present, skip')
else:
    detail = (
        'G4-P4 head attribution: per-head zeroing at attention out-proj input (B,T,H*HD slices; '
        'KV heads untouched), blocks L_mid/+1/+2, 4 anchors x 6 top64 dirs x alpha=0.1 (3160 '
        'protocol verbatim), base+pert paired in same chunk (pre-slot dh bitwise 0), eps from '
        'batch=1 anchor state (bitwise vs 3157 4/4 x3). RESULT: ctrl (all-heads all-3-blocks '
        'recover) = ' + '/'.join(f(c) for c in ctrls) + ' (gate 0.2) -> '
        'consumption_not_in_attn_out 3/3: zeroing ALL attention of the consumption blocks does '
        'NOT restore the consumed direction either; combined with 3160 (MLP zeroing also fails) '
        'the mechanism chain 3159-3160-3161 concludes NO single-point consumer: consumption is '
        'redundant/distributed across the residual flow. Single-head total mass T = '
        '/'.join(f(t) for t in Ts) + ' (gate 0.1), C4 = ' + '/'.join(f(c) for c in C4s) +
        ' (noise ratios, meaningless at such T), rand_ratio = ' + '/'.join(f(r) for r in rands) +
        ' (random heads = average heads, no head specificity). Transient: all3 q50 curve runs '
        'ABOVE none by ' + '/'.join(f(g) for g in gap123) + ' at slots L_mid+1/+2/+3 (4b) but '
        'converges by NL (' + f(nl4b_a3) + ' vs ' + f(nl4b) + ') -> downstream redundancy closes '
        'the gap. This REFINES 3160 attention_reallocation_primary (elimination inference): '
        'neither MLP nor attention outputs of blocks L_mid..L_mid+2 execute the consumption. '
        'Summary: none-config q50 consumption curve fingerprint (L_mid-offset aligned W=' + str(W) +
        ') fpmin ' + f(fpmin) + ', sorted-R spectrum fpmin ' + f(fpmin_spec) + ', class agreement=' +
        str(agree) + ', verdict ' + verdict_main + '. Mechanical notes: (a) Qwen3-4B config '
        'head_dim=128 (explicit, != D/H=80) -> HD resolution fixed pre-SMOKE; (b) 14b bf16 '
        '(29.5GB) on the 16GB card runs via NVIDIA sysmem fallback -> weights streamed over '
        'PCIe every fwd; co-tenant memory pressure (GameViewer streaming / server.py / browser) '
        'trims the working set -> monotonic within-process slowdown; CHUNK 128->48->16 refrozen '
        'twice, then execution switched to per-anchor process isolation + collect merge '
        '(anchors independent & deterministic; 4b collect bitwise-equal physics to single-'
        'process); all refreezes before any formal 14b/glm4 observation. (c) PRECISION '
        'DEVIATION (frozen before any formal 14b/glm4 observation): 4b bf16 (8.0GB fits VRAM); '
        '14b (29.5GB) / glm4 (18.8GB) -> NF4 double-quant bitsandbytes (Q05 convention on this '
        'machine; bf16 sysmem-fallback measured infeasible: 1.3s/fwd warm -> pagefile 60s+/fwd '
        'cold under co-tenant memory pressure); NF4 anchor check = tolerance vs 3157 bf16 H '
        '(cos>=0.999, rel<=0.05); verdict gates relative; cross-model comparability at pattern '
        'level; 3160 parity for 14b/glm4 downgraded to pattern level. design_sha: ' +
        ', '.join(m + '=' + jload(os.path.join(PDIR, m, 'execution.json'))['design_sha'][:8] for m in MS) + '. '
        'Per-model res/seal: ' + ', '.join(m + ' ' + R[m]['res_sha8'] + '/' + R[m]['seal_sha8'] for m in MS) +
        '; summary ' + RS['res_sha8'] + '/' + RS['seal_sha8'] + '; smoke(4b) ' + SM['res_sha8'] + '/' + SM['seal_sha8'])
    entry = {
        'phase': 3161, 'name': 'g4p4_head_attribution', 'line': 'G',
        'date': '2026-10-09', 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': verdict_main, 'detail': detail,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms_.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    out.append('ledger: appended 3161 (n=%d, was %d) chain_sha8=%s' % (n1, n0, led['ledger_sha256_8']))

# ============ 2. MEMO ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')
sec_marker = '## Phase 3161: 消耗的头归因（G4-P4）'
if sec_marker in txt:
    out.append('memo: 3161 section already present, skip')
else:
    L = []
    L.append('## Phase 3161: 消耗的头归因（G4-P4）[' + NOW + ']')
    L.append('')
    L.append('**主判决：`' + verdict_main + '`（三模型一致）——attention 头归因的答案是否定的：置零消耗块全部注意力头也不恢复被消耗方向（ctrl recover ' +
             '/'.join(f(c) for c in ctrls) + '，≪0.2 门）；叠加 3160（MLP 置零同样不恢复）⇒ 机制链 3159→3160→3161 收官结论：**消耗无单点执行者——残差流的冗余分布式性质**。3160 的 attention_reallocation_primary 是排除法推理，被本 Phase 修正（MLP 与 attention 都不是执行者）。**')
    L.append('')
    L.append('### 设计与执行')
    L.append('- 预注册（3160 closeout + 3162 缺口①，观测前）：big-drop 块逐头置零，top-4 集中度三分门（≥0.5 localized / <0.2 distributed / 之间 weakly_localized）；GQA 只切 query 头（KV 不动）。')
    L.append('- 两点物理澄清（观测前冻结）：① 预注册字面「self_attn 输出 head·dh 切片」经 o_proj 后无头语义 → 按物理正确口径切 **o_proj 输入** (B,T,H×HD)（= 置零该头对 attn 输出的贡献）；② 块集合 {L_mid,+1,+2}=字面块超集，主门在字面块 L_mid，扩展预声明。')
    L.append('- 装置：4 锚×6 top64 方向×α=0.1（逐字同 3160）；base/pert 成对同 chunk（**dh 前 L_mid 槽逐位 0**，prm=0 实测）；CHUNK 显存自适应（4b=128 / glm4=48 / **14b=16 行**）；**正式跑=每 anchor 独立进程 + collect 合并**——14b bf16 29.5GB 于 16GB 卡经 NVIDIA sysmem fallback 每 fwd 经 PCIe 流式读权重，共租进程（GameViewer 串流/server.py/浏览器）挤压工作集导致进程内单调减速（121s→206s→1400s+，三次中止），fresh process 每 anchor 恢复全速（anchor 独立确定性：4b collect 与单进程结果同值）；eps 取 batch=1 锚态槽 L_mid（bitwise vs 3157）；随机对照每块 4 头（种子 3161）+ 每块全头 + 三块全头（ctrl）。')
    L.append('- 机械修正备注：① Qwen3-4B config 显式 **head_dim=128**（≠D/H=80）→ HD 解析改 config 优先（SMOKE 前修复）；② 14b CHUNK 128→48→16 两次重冻结（sysmem fallback PCIe 流式 + 工作集挤压减速，观测前）；③ 执行层改为每 anchor 独立进程 + collect（观测前重冻结）。design_sha（现场渲染）：' +
             '/'.join(m + '=' + jload(os.path.join(PDIR, m, 'execution.json'))['design_sha'][:8] for m in MS) + '。H/HD：' +
             '/'.join(m + '=' + str(R[m]['model']['H']) + 'x' + str(R[m]['model']['HD']) for m in MS) + '；oproj=' + R['qwen3-4b']['model']['oproj'] + '。')
    L.append('- 精度偏差（观测前冻结，本机 deepseek Q05 先例）：4b=bf16（8.0GB 入 VRAM）；**14b（29.5GB）/glm4（18.8GB）=NF4 double-quant**——bf16 sysmem fallback 实测不可行（温态 1.3s/fwd → 共租内存压力下 pagefile 磁盘流式 60s+/fwd）；NF4 锚检查降为容差（cos≥0.999 且 rel≤0.05，对照 3157 bf16 H）；判决门全为相对量（recover/C4/ctrl），跨模型对比保持 pattern 级；3160 的 14b/glm4 平价降为 pattern 级（记录在案）。')
    L.append('- runtime（s，=collect 进程时长；anchor 进程时长见各 run_log）：' + '/'.join(str(r) for r in rts) + '；效力门（全头置零槽 L_mid+1 maxabs 差）：' + '/'.join(str(e) for e in effs) + '；锚检查：' + '/'.join(str(R[m]['det'].get('anchor_check', '')) for m in MS) + '；none share(L_mid)=0.9998×3。')
    L.append('')
    L.append('### 三发现（重复强调）')
    L.append('1. **ctrl（三块全部注意力头置零）recover = ' + '/'.join(f(c) for c in ctrls) + '（门 0.2）**：消耗不由 attention 输出执行；rand_ratio ' +
             '/'.join(f(r) for r in rands) + '（随机头=平均头，**无特异性头**）；单头总质量 T=' + '/'.join(f(t) for t in Ts) + ' ≪0.1（top-4 集中度 ' + '/'.join(f(c) for c in C4s) + ' 为噪声比）——不存在局部化头结构。')
    L.append('2. **早段瞬态 + 下游冗余补完**：all3 q50 曲线在槽 L_mid+1/+2/+3 比 none 高 ' + '/'.join(f(g) for g in gap123) +
             '（4b），到 NL 收敛（' + f(nl4b_a3) + ' vs ' + f(nl4b) + '）——attention 只承担早段消耗一小部分，下游流把剩余旋转补完。')
    L.append('3. **机制链判闭：无单点消费者**。3159（注入方向被消耗）→3160（MLP 置零不恢复 recover ≤0.006）→3161（attention 置零不恢复 ctrl≈0）⇒ 消耗是残差流的冗余分布式性质；跨模型消耗曲线指纹（none q50，对齐 W=' + str(W) + '）fpmin=' + f(fpmin) +
             '、sorted-R 谱 fpmin=' + f(fpmin_spec) + '、类别一致=' + str(agree) + '——「冗余分布」本身是跨模型不变量。')
    L.append('')
    L.append('### 锚')
    L.append('4b res **' + R['qwen3-4b']['res_sha8'] + '** seal ' + R['qwen3-4b']['seal_sha8'] +
             '；14b res **' + R['qwen3-14b']['res_sha8'] + '** seal ' + R['qwen3-14b']['seal_sha8'] +
             '；glm4 res **' + R['glm4']['res_sha8'] + '** seal ' + R['glm4']['seal_sha8'] +
             '；summary res **' + RS['res_sha8'] + '** seal ' + RS['seal_sha8'] +
             '；smoke(4b) res ' + SM['res_sha8'] + ' seal ' + SM['seal_sha8'] + '。disk: ' +
             '；'.join(m + ' res/' + sha['res_' + m] + ' npz/' + sha['npz_' + m] for m in MS) +
             '；summary res/' + sha['res_summary'] + '；smoke res/' + sha['res_smoke'] + ' npz/' + sha['npz_smoke'] +
             '。ledger n=**__NLED__**。产物 `phase3161\\g4p4_head_attribution\\{qwen3-4b,qwen3-14b,glm4,summary}\\`。')
    L.append('')
    L.append('### 预注册 Phase 3163：G4-P5 消耗冗余性判别（机制链收官实验）')
    L.append('- 假设：3161 证单组件置零不恢复 → 冗余假说：任一组件被移除后，下游流补完旋转。P5 用「真恒等块」与「扩展窗」判别补完的位置与载体。')
    L.append('- 设计（4 锚×6 top64 方向×α=0.1 同口径）：配置 A=**联合置零**（全头+MLP）块 {L_mid,+1,+2}（块变恒等映射）→ 装置门：槽 L_mid+3 的 share ≥0.95；读 share(NL)；配置 B=**扩展窗全头置零**（块 L_mid..NL−1 全部注意力头）；配置 C=**扩展窗联合置零**（残差恒等装置门：NL share=1±0.05）。')
    L.append('- 门：|share_A(NL) − share_none(NL)| < 0.05 → `redundant_closing`（3 块内消耗完全冗余）/ ≥0.1 → `joint_localized`（3 块联合承担不可替代消耗）；B：share_B(NL) ≥0.5 → `attention_primary_extended` / ≤0.15 → `mlp_or_residual_primary`；C：share ≥0.95 装置恒等。GPU ~3min/模型。执行后回图谱主线 **G5-A2**（缺口②：C_steer / RoPE / massive 跨模型同口径复测）。')
    block = '\n'.join(L) + '\n'
    led_n = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    block = block.replace('__NLED__', str(led_n))
    block_crlf = block.replace('\n', '\r\n')
    txt2 = txt.rstrip('\n') + '\r\n' + block_crlf
    open(MEMO, 'wb').write(txt2.encode('utf-8'))
    out.append('memo: appended 3161 section + 3163 prereg (CRLF)')

# ============ 2b. MEMO 尾部 EOL 规范化(3162 节起, 若有裸 LF) ============
b = open(MEMO, 'rb').read()
i32 = b.find('## Phase 3162'.encode('utf-8'))
if i32 > 0:
    head, seg = b[:i32], b[i32:]
    bare = seg.count(b'\n') - seg.count(b'\r\n')
    if bare > 0:
        seg = seg.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        open(MEMO, 'wb').write(head + seg)
        out.append('memo: tail EOL normalized (bare_lf=%d)' % bare)
    else:
        out.append('memo: tail EOL clean')
b = open(MEMO, 'rb').read()
assert b[:3] == b'\xef\xbb\xbf', 'BOM lost'
assert b.count(b'\n') == b.count(b'\r\n'), 'EOL mixed after normalize'
out.append('memo: BOM ok, EOL uniform CRLF (lf=%d)' % b.count(b'\n'))

# ============ 3. Daily ============
daily = os.path.join(MDIR_MEM, '2026-10-09.md')
marker = '3161 消耗头归因闭环'
if os.path.exists(daily) and marker in io.open(daily, encoding='utf-8').read():
    out.append('daily: 3161 already present, skip')
else:
    n_ms = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    line = ('- 3161 消耗头归因闭环（G4-P4）：逐头置零 o_proj 输入（块 L_mid/+1/+2，4 锚×6 方向×α=0.1 同 3160 口径）——'
            'ctrl（三块全头）recover ' + '/'.join(f(c) for c in ctrls) + ' ≪0.2 门、单头总质量 T=' +
            '/'.join(f(t) for t in Ts) + ' ≪0.1、rand_ratio=' + '/'.join(f(r) for r in rands) +
            '（无特异性头）→ g4p4_consumption_not_in_attn_out 3/3；叠加 3160（MLP 同样不恢复）⇒ 机制链判闭：'
            '**消耗无单点执行者（残差流冗余分布式）**，3160 排除法结论被修正；all3 早段瞬态差 ' +
            '/'.join(f(g) for g in gap123) + '、NL 收敛；指纹 fpmin=' + f(fpmin) + '；'
            'HD 修正=Qwen3-4B head_dim=128 显式（design 未变，SMOKE 前修）；ledger n=' + str(n_ms) + '。'
            '3163 预注册=G4-P5 冗余性判别（联合置零恒等块 + 扩展窗）。\n')
    with io.open(daily, 'a', encoding='utf-8') as f2:
        f2.write(line)
    out.append('daily: appended 3161 line')

# ============ 4. Workspace MEMORY ============
mp = os.path.join(MDIR_MEM, 'MEMORY.md')
mtxt = io.open(mp, encoding='utf-8').read()
seg_anchor = '3161 决策=**推进**（机制链 3159→3160→3161 唯一未测环节，预注册不变）。'
if '3161 消耗头归因闭环' in mtxt:
    out.append('workspace MEMORY: 3161 already present, skip')
else:
    ia = mtxt.rfind(seg_anchor)
    assert ia >= 0, '3162 segment tail anchor not found'
    ip = ia + len(seg_anchor)
    seg11 = ('**✅ 3161 消耗头归因闭环（2026-10-09）**：逐头置零 o_proj 输入（块 L_mid/+1/+2，KV 不动；'
             '澄清=o_proj 输入才有头语义）ctrl（三块全头）recover ' + '/'.join(f(c) for c in ctrls) +
             ' ≪0.2、T=' + '/'.join(f(t) for t in Ts) + ' ≪0.1、rand_ratio=' + '/'.join(f(r) for r in rands) +
             ' → **g4p4_consumption_not_in_attn_out 3/3**；叠加 3160 ⇒ 机制链 3159→3160→3161 判闭：'
             '**消耗无单点执行者=残差流冗余分布式性质**（attention 早段瞬态差 ' + '/'.join(f(g) for g in gap123) +
             '、NL 收敛；3160 排除法结论被修正）；指纹 none-q50 fpmin ' + f(fpmin) + '/sortedR ' + f(fpmin_spec) +
             '；HD 修正=Qwen3-4B head_dim=128 显式；res 4b ' + R['qwen3-4b']['res_sha8'] + '/14b ' +
             R['qwen3-14b']['res_sha8'] + '/glm4 ' + R['glm4']['res_sha8'] + '/summary ' + RS['res_sha8'] +
             '/ledger n=' + str(len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])) +
             '。下一步 3163=**G4-P5 消耗冗余性判别**（联合置零恒等块+扩展窗，机制链收官）→ 之后 G5-A2 图谱缺口②。')
    mtxt = mtxt[:ip] + seg11
    open(mp, 'wb').write(mtxt.encode('utf-8'))
    out.append('workspace MEMORY: 3161 appended after 3162 segment')

# ============ 5. Self-check ============
ok = []
ok.append(('ledger_3161', any(m.get('phase') == 3161 for m in
                              json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])))
t = open(MEMO, 'rb').read().decode('utf-8')
ok.append(('memo_3161', sec_marker in t))
ok.append(('memo_3163', '预注册 Phase 3163' in t))
ok.append(('daily_3161', marker in io.open(daily, encoding='utf-8').read()))
ok.append(('memory_3161', '3161 消耗头归因闭环' in io.open(mp, encoding='utf-8').read()))
ok.append(('memo_bom', open(MEMO, 'rb').read()[:3] == b'\xef\xbb\xbf'))
bb = open(MEMO, 'rb').read()
ok.append(('memo_eol_uniform', bb.count(b'\n') == bb.count(b'\r\n')))
ok.append(('seal_consistency', all(_verdict_sha_ok(R[m]) for m in MS) and _verdict_sha_ok(RS)))
bad = [k for k, v in ok if not v]
out.append('SELF-CHECK: %s' % ('OK' if not bad else 'FAIL %s' % bad))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('CLOSEOUT DONE')
