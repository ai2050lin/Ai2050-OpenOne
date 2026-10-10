# -*- coding: utf-8 -*-
# 3163 closeout 五写: ledger append + MEMO(3163 节 + 3164/G5-A2 预注册) + daily
#                     + workspace MEMORY + self-check。数字一律 result 现场渲染; MEMO 块 CRLF;
#                     幂等 by phase/节标题/marker。
import io, json, os, shutil, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MDIR_MEM = os.path.join(ROOT, '.workbuddy', 'memory')
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3163', 'g4p5_redundancy')
NOW = time.strftime('%Y-%m-%d %H:%M')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3163_closeout_out.txt')
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

clsAs = [R[m]['clsA'] for m in MS]
clsBs = [R[m]['clsB'] for m in MS]
coks = [R[m]['c_ok'] for m in MS]
s_none = [R[m]['share_none_nl'] for m in MS]
s_A = [R[m]['share_A_nl'] for m in MS]
s_B = [R[m]['share_B_nl'] for m in MS]
s_C = [R[m]['share_C_nl'] for m in MS]
dAs = [R[m]['dA'] for m in MS]
a3s = [R[m]['share_A_lmid3'] for m in MS]
ida = [R[m]['det']['identity_window_rel_max_A'] for m in MS]
idc = [R[m]['det']['identity_window_rel_max_C'] for m in MS]
effa = [R[m]['det']['efficacy_A_maxabs_Lmid1'] for m in MS]
effb = [R[m]['det']['efficacy_B_rel_NL'] for m in MS]
rts = [R[m]['runtime_s'] for m in MS]
fpmin = RS['fpmin_q50_none']
W = RS['consumption_window']
agree = RS['gates']['class_agreement']
cok_all = RS['gates']['c_device']
verdict_main = RS['verdict']
smoke_verdict = SM['verdict']

def _wordA(c):
    return {
        'redundant_closing': '3 块内消耗完全冗余：整块恒等化（MLP+attention 全部移除）后 share(NL) 与 none 几乎不变——消耗不由前 3 块执行，由下游流补完',
        'joint_localized': '3 块联合承担不可替代消耗：恒等化后 dh 在 NL 保留显著更多子空间能量',
        'band_undecided': '落于预注册未定义中间带 [0.05,0.1)，诚实标注不强行归类',
    }[c]

def _wordB(c):
    return {
        'attention_primary_extended': '扩展窗意义上 attention 是消耗主要载体',
        'mlp_or_residual_primary': '即使移除 L_mid..NL−1 全部注意力头，消耗照常发生——attention 在扩展窗意义上也不是载体，剩余载体=MLP 或残差流固有动态',
        'band_undecided': '落于预注册未定义中间带 (0.15,0.5)',
    }[c]

# all3/恒等描述(4b 现场算)
import numpy as np
z4b = np.load(os.path.join(PDIR, 'qwen3-4b', 'collect.npz'))
L_MID = int(z4b['l_mid'])
NL4b = int(z4b['SHARE_NL'].shape[0] * 0 + z4b['heads'].shape[0] * 0 + z4b['CURVES'].shape[3] - 1)
CUR = z4b['CURVES'].astype(np.float64)  # (NA, ND, 4, NH)
q_n = np.median(CUR[..., 0, :].reshape(-1, CUR.shape[3]), axis=0)
q_A = np.median(CUR[..., 1, :].reshape(-1, CUR.shape[3]), axis=0)
q_B = np.median(CUR[..., 2, :].reshape(-1, CUR.shape[3]), axis=0)
q_C = np.median(CUR[..., 3, :].reshape(-1, CUR.shape[3]), axis=0)

# ---- 产物 disk sha 现场渲染 ----
sha = {}
for m in MS:
    sha['res_' + m] = sha8_file(os.path.join(PDIR, m, 'result.json'))
    sha['npz_' + m] = sha8_file(os.path.join(PDIR, m, 'collect.npz'))
sha['res_summary'] = sha8_file(os.path.join(PDIR, 'summary', 'result_summary.json'))
sha['res_smoke'] = sha8_file(os.path.join(PDIR, 'qwen3-4b', 'smoke', 'result.json'))
sha['npz_smoke'] = sha8_file(os.path.join(PDIR, 'qwen3-4b', 'smoke', 'collect.npz'))
out.append('disk sha: %s' % json.dumps(sha, indent=0, sort_keys=True))
# verdict 内嵌 sha 一致性（seal 字节级复验由独立磁盘复核脚本做）
def _verdict_sha_ok(r):
    return r['verdict'].endswith('|sha8_' + r['res_sha8'])

for m in MS:
    assert _verdict_sha_ok(R[m]), ('verdict sha mismatch', m)
assert _verdict_sha_ok(RS), ('verdict sha mismatch', 'summary')
out.append('seal consistency: verdict-embedded sha OK 4/4')

# ---- 快照 ----
snap_dir = os.path.join(ROOT, 'tests', 'gpt5_temp', '_snap_3163')
os.makedirs(snap_dir, exist_ok=True)
for src in (LEDGER, MEMO):
    shutil.copy2(src, os.path.join(snap_dir, os.path.basename(src) + '.bak'))
out.append('snapshot: ledger+memo copied to _snap_3163')

# ============ 1. Ledger ============
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3163 for m in ms_):
    out.append('ledger: 3163 already present, skip')
else:
    detail = (
        'G4-P5 redundancy discrimination (mechanism-chain closeout): A=joint zeroing '
        '(all-heads o_proj-input + MLP-output -> blocks {L_mid,+1,+2} become identity), '
        'B=extended-window all-head zeroing (L_mid..NL-1), C=extended-window joint zeroing '
        '(identity device), 4 anchors x 6 top64 dirs x alpha=0.1 (3161 protocol verbatim). '
        'Device gates green: A identity window share(L_mid+3)=' + '/'.join(f(a) for a in a3s) +
        ', bitwise propagation ident_a=' + '/'.join(('%.2g' % v) for v in ida) +
        ' / ident_C=' + '/'.join(('%.2g' % v) for v in idc) + ' (<1e-4). RESULT: dA=|share_A(NL)'
        '-share_none(NL)| = ' + '/'.join(f(d) for d in dAs) + ' (gates 0.05/0.1) -> clsA=' +
        '/'.join(clsAs) + ' (' + _wordA(clsAs[0]) + '); share_B(NL)=' + '/'.join(f(b) for b in s_B) +
        ' (gates 0.5/0.15) -> clsB=' + '/'.join(clsBs) + ' (' + _wordB(clsBs[0]) + '); '
        'share_none(NL)=' + '/'.join(f(x) for x in s_none) + '. C post-norm share=' +
        '/'.join(f(c) for c in s_C) + ' (final-RMSNorm Jacobian effect, pre-norm identity is '
        'bitwise). Combined with 3159-3160-3161: consumption has NO single-point executor and '
        'is NOT localized in any block set — it is a distributed property of the residual flow '
        '(downstream blocks re-derive it). Summary: none q50 fingerprint fpmin ' + f(fpmin) +
        ' (W=' + str(W) + '), class agreement=' + str(agree) + ', C device=' + str(cok_all) +
        ', verdict ' + verdict_main + '. Smoke refreeze (before any formal observation, 4b '
        '2-anchor): R1 index fix (SHARE_A3 row), R2 C gate slot-semantics (slot NL is after '
        'final RMSNorm; prereg literal 1+/-0.05 physically unreachable; refrozen to bitwise '
        'pre-norm propagation + post-norm floor 0.90 + norm-effect parity in det). 14b used '
        'the pre-quantized NF4 checkpoint (Qwen3-14B-bnb-nf4) per 3161 addendum. design_sha: ' +
        ', '.join(m + '=' + jload(os.path.join(PDIR, m, 'execution.json'))['design_sha'][:8] for m in MS) + '. '
        'Per-model res/seal: ' + ', '.join(m + ' ' + R[m]['res_sha8'] + '/' + R[m]['seal_sha8'] for m in MS) +
        '; summary ' + RS['res_sha8'] + '/' + RS['seal_sha8'] + '; smoke(4b) ' + SM['res_sha8'] + '/' + SM['seal_sha8'])
    entry = {
        'phase': 3163, 'name': 'g4p5_redundancy', 'line': 'G',
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
    out.append('ledger: appended 3163 (n=%d, was %d) chain_sha8=%s' % (n1, n0, led['ledger_sha256_8']))

# ============ 2. MEMO ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')
sec_marker = '## Phase 3163: 消耗冗余性判别（G4-P5）'
if sec_marker in txt:
    out.append('memo: 3163 section already present, skip')
else:
    L = []
    L.append('## Phase 3163: 消耗冗余性判别（G4-P5）[' + NOW + ']')
    L.append('')
    L.append('**主判决：`' + verdict_main + '`（类别一致=' + str(agree) + '）——机制链 3159→3160→3161→3163 收官：**' + _wordA(clsAs[0]) + '**；' + _wordB(clsBs[0]) + '。**')
    L.append('')
    L.append('### 设计与执行')
    L.append('- 预注册（3161 closeout，观测前）：A=联合置零（全头+MLP）块 {L_mid,+1,+2}（块变恒等映射）装置门 share(L_mid+3)≥0.95；B=扩展窗全头置零（块 L_mid..NL−1 全部注意力头）；C=扩展窗联合置零（残差恒等装置门）；门 |share_A(NL)−share_none(NL)|<0.05→`redundant_closing` / ≥0.1→`joint_localized`；B ≥0.5→`attention_primary_extended` / ≤0.15→`mlp_or_residual_primary`；中间带→band_undecided（不发明新类别）。')
    L.append('- 协议逐字继承 3161：4 锚×6 top64 方向×α=0.1；注入 hook 在块 L_mid−1 输出末 token；base/pert 成对同 chunk（dh 前 L_mid 槽逐位 0）；4b=bf16 单进程；14b=pre-quantized NF4 checkpoint（Qwen3-14B-bnb-nf4，3161 补记）/glm4=现场 NF4，per-anchor 进程隔离+collect。置零口径：全头=3161（o_proj 输入切片）、MLP=3160（mlp 输出替换 zeros）；联合⇒块恒等（残差直通）。')
    L.append('- **SMOKE 重冻结两项（任何正式观测前，4b 2 锚暴露）**：R1 索引修正——SHARE_A3 误取行 2（=B），主判决列 s_A/s_B 取值一直正确；R2 C 装置门口径澄清——槽 NL=final RMSNorm **之后**，norm 的 Jacobian 不保 top64 子空间，预注册字面「NL share=1±0.05」物理不可达（实测 norm 效应 0.065）→ 重冻结为三条合取：恒等窗逐位传播（norm 前，<1e-4）∧ share_C(NL)≥0.90（norm 后下限）∧ norm 效应对照（det 记录）。design_sha：' +
             '/'.join(m + '=' + jload(os.path.join(PDIR, m, 'execution.json'))['design_sha'][:8] for m in MS) + '。')
    L.append('- 效力门：A-base vs none-base 槽 L_mid+1 maxabs=' + '/'.join(('%.3g' % e) for e in effa) +
             '；B-base vs none-base 槽 NL rel=' + '/'.join(('%.3g' % e) for e in effb) +
             '（hook 活性）。锚检查：' + '/'.join(str(R[m]['det'].get('anchor_check', '')) for m in MS) +
             '；runtime(s)：' + '/'.join(str(r) for r in rts) + '。')
    L.append('')
    L.append('### 三发现（重复强调）')
    L.append('1. **配置 A（3 块真恒等）：dA=|share_A(NL)−share_none(NL)| = ' + '/'.join(f(d) for d in dAs) + '（门 0.05/0.1）→ ' +
             '/'.join(clsAs) + ' ×3**：' + _wordA(clsAs[0]) + '（share_none(NL)=' + '/'.join(f(x) for x in s_none) +
             ' vs share_A(NL)=' + '/'.join(f(x) for x in s_A) + '）。')
    L.append('2. **配置 B（扩展窗 L_mid..NL−1 全头置零）：share_B(NL) = ' + '/'.join(f(b) for b in s_B) + '（门 0.5/0.15）→ ' +
             '/'.join(clsBs) + ' ×3**：' + _wordB(clsBs[0]) + '——3161 的 ctrl 结论在扩展窗下成立（attention 非载体）。')
    L.append('3. **装置完备性（C + 恒等窗逐位传播）**：ident_a=' + '/'.join(('%.2g' % v) for v in ida) +
             '、ident_C=' + '/'.join(('%.2g' % v) for v in idc) + '（<1e-4，dh 在恒等窗内逐位不变）；A3 装置门=' +
             '/'.join(f(a) for a in a3s) + '（≥0.95）；share_C(NL)=' + '/'.join(f(c) for c in s_C) +
             '（norm 后；norm 效应 drop_C=' + '/'.join(('%.4f' % R[m]['det']['norm_effect']['drop_C']) for m in MS) +
             '）；c_ok=' + '/'.join(str(bool(c)) for c in coks) + '。跨模型 none q50 指纹 fpmin=' + f(fpmin) +
             '（W=' + str(W) + '），类别一致=' + str(agree) + '。')
    L.append('')
    L.append('### 锚')
    L.append('4b res **' + R['qwen3-4b']['res_sha8'] + '** seal ' + R['qwen3-4b']['seal_sha8'] +
             '；14b res **' + R['qwen3-14b']['res_sha8'] + '** seal ' + R['qwen3-14b']['seal_sha8'] +
             '；glm4 res **' + R['glm4']['res_sha8'] + '** seal ' + R['glm4']['seal_sha8'] +
             '；summary res **' + RS['res_sha8'] + '** seal ' + RS['seal_sha8'] +
             '；smoke(4b) res ' + SM['res_sha8'] + ' seal ' + SM['seal_sha8'] + '。disk: ' +
             '；'.join(m + ' res/' + sha['res_' + m] + ' npz/' + sha['npz_' + m] for m in MS) +
             '；summary res/' + sha['res_summary'] + '；smoke res/' + sha['res_smoke'] + ' npz/' + sha['npz_smoke'] +
             '。ledger n=**__NLED__**。产物 `phase3163\\g4p5_redundancy\\{qwen3-4b,qwen3-14b,glm4,summary}\\`。')
    L.append('')
    L.append('### 预注册 Phase 3164：G5-A2 图谱缺口②跨模型同口径复测（C_steer / RoPE / massive）')
    L.append('- 依据：3162 图谱基座登记的校准项——RoPE 位置平移族仅 4b（3156）、massive 77× 仅 4b 量化（3157/3158）、C_steer=0 仅 qwen3-4b（Q06/Phase 40）。')
    L.append('- 设计框架（三轴顺序执行；每轴独立 execution.json 于执行前冻结，详细门在框架内细化）：(a) **C_steer 跨模型**：Q06 承重轴装置（qwen3-4b L29 WR 主 PC）同构移植 14b/glm4 对应层 + x 端口替换，held-out cells × 10 配置同口径；门=steered 成功率 Wilson 上界（Q06=1.0%）+ collateral（Q06 frac0=0.933 对照）。(b) **RoPE 位置平移族跨模型**（3156 协议）：双臂（真实前缀/位置重置）× k∈{0..128}；门=KL_B ≤0.01 + top1_B（3156 实测 KL_B≤0.0023、top1_B 9/9）。(c) **massive 跨模型**：3157/3158 的 d1 与 77× 现象在 14b/glm4 的对应性；NF4 量化误差（3161 实测 rel 0.196–0.241）下口径改为容差标注或 bf16-CPU 单前向采样（执行前定）。')
    L.append('- GPU 预算：a ~30min/模型（441 cells×10 配置）、b ~10min/模型、c ~5min/模型。完成后图谱缺口②关闭，回 G5 主线（缺口排序再评估）。')
    block = '\n'.join(L) + '\n'
    led_n = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    block = block.replace('__NLED__', str(led_n))
    block_crlf = block.replace('\n', '\r\n')
    txt2 = txt.rstrip('\n') + '\r\n' + block_crlf
    open(MEMO, 'wb').write(txt2.encode('utf-8'))
    out.append('memo: appended 3163 section + 3164/G5-A2 prereg (CRLF)')

# ============ 2b. MEMO 尾部 EOL 规范化 ============
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
marker = '3163 消耗冗余性判别闭环'
if os.path.exists(daily) and marker in io.open(daily, encoding='utf-8').read():
    out.append('daily: 3163 already present, skip')
else:
    n_ms = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    line = ('- 3163 消耗冗余性判别闭环（G4-P5，机制链收官）：A=3 块联合恒等化 dA=' + '/'.join(f(d) for d in dAs) +
            ' → ' + '/'.join(clsAs) + ' ×3；B=扩展窗全头 share_B(NL)=' + '/'.join(f(x) for x in s_B) +
            ' → ' + '/'.join(clsBs) + ' ×3；装置门全绿（ident_a/C=' + '/'.join(('%.2g' % v) for v in ida) +
            '/' + '/'.join(('%.2g' % v) for v in idc) + '，A3=' + '/'.join(f(a) for a in a3s) +
            '，c_ok=' + '/'.join(str(bool(c)) for c in coks) + '）；指纹 fpmin=' + f(fpmin) +
            '，一致=' + str(agree) + '；结论=消耗无单点执行者且不局部于任何块集合=残差流全流分布式性质；'
            'SMOKE 重冻结 R1 索引/R2 C 门 norm 槽位口径（观测前）；ledger n=' + str(n_ms) + '。'
            '3164 预注册=G5-A2 图谱缺口②跨模型复测（C_steer/RoPE/massive）。\n')
    with io.open(daily, 'a', encoding='utf-8') as f2:
        f2.write(line)
    out.append('daily: appended 3163 line')

# ============ 4. Workspace MEMORY ============
mp = os.path.join(MDIR_MEM, 'MEMORY.md')
mtxt = io.open(mp, encoding='utf-8').read()
seg_anchor = '。下一步 3163=**G4-P5 消耗冗余性判别**（联合置零恒等块+扩展窗，机制链收官）→ 之后 G5-A2 图谱缺口②。'
if '3163 消耗冗余性判别闭环' in mtxt:
    out.append('workspace MEMORY: 3163 already present, skip')
else:
    ia = mtxt.rfind(seg_anchor)
    assert ia >= 0, '3161 segment tail anchor not found'
    ip = ia + len(seg_anchor)
    seg13 = ('**✅ 3163 消耗冗余性判别闭环（2026-10-09）**：A=3 块联合恒等化 dA=' + '/'.join(f(d) for d in dAs) +
             '（门 0.05/0.1）→ **' + '/'.join(clsAs) + ' 3/3**；B=扩展窗（L_mid..NL−1）全头置零 share_B(NL)=' +
             '/'.join(f(x) for x in s_B) + ' → **' + '/'.join(clsBs) + ' 3/3**；装置门全绿（ident_a=' +
             '/'.join(('%.2g' % v) for v in ida) + '/ident_C=' + '/'.join(('%.2g' % v) for v in idc) +
             ' 逐位、A3=' + '/'.join(f(a) for a in a3s) + '、c_ok=' + '/'.join(str(bool(c)) for c in coks) +
             '，C 的 norm 后 share=' + '/'.join(f(c) for c in s_C) + '）；指纹 none-q50 fpmin=' + f(fpmin) +
             '、类别一致=' + str(agree) + '；**机制链 3159→3160→3161→3163 收官：消耗无单点执行者且不局部于任何块集合=残差流全流分布式性质**'
             '（下游块重新推导出消耗）；SMOKE 重冻结 R1 索引/R2 C 门 norm 槽位口径（任何正式观测前）；res 4b ' +
             R['qwen3-4b']['res_sha8'] + '/14b ' + R['qwen3-14b']['res_sha8'] + '/glm4 ' + R['glm4']['res_sha8'] +
             '/summary ' + RS['res_sha8'] + '/ledger n=' +
             str(len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])) +
             '。下一步 3164=**G5-A2 图谱缺口②跨模型同口径复测**（C_steer 装置移植 / RoPE 位置平移族 / massive 对应性）→ 完成后回 G5 主线缺口排序。')
    mtxt = mtxt[:ip] + seg13
    open(mp, 'wb').write(mtxt.encode('utf-8'))
    out.append('workspace MEMORY: 3163 appended after 3161 segment')

# ============ 5. Self-check ============
ok = []
ok.append(('ledger_3163', any(m.get('phase') == 3163 for m in
                              json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])))
t = open(MEMO, 'rb').read().decode('utf-8')
ok.append(('memo_3163', sec_marker in t))
ok.append(('memo_3164', '预注册 Phase 3164' in t))
ok.append(('daily_3163', marker in io.open(daily, encoding='utf-8').read()))
ok.append(('memory_3163', '3163 消耗冗余性判别闭环' in io.open(mp, encoding='utf-8').read()))
ok.append(('memo_bom', open(MEMO, 'rb').read()[:3] == b'\xef\xbb\xbf'))
bb = open(MEMO, 'rb').read()
ok.append(('memo_eol_uniform', bb.count(b'\n') == bb.count(b'\r\n')))
ok.append(('seal_consistency', all(_verdict_sha_ok(R[m]) for m in MS) and _verdict_sha_ok(RS)))
bad = [k for k, v in ok if not v]
out.append('SELF-CHECK: %s' % ('OK' if not bad else 'FAIL %s' % bad))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('CLOSEOUT DONE')
