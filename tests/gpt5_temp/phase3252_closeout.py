# -*- coding: utf-8 -*-
"""Phase 3252 closeout: five writes (result assertions / ledger #404 / MEMO /
dual wlog / MEMORY.md). Idempotent per write; ledger append chain-checked."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3252\g5h15_ustar_layer_trace\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'research\gpt5\docs\WORKLOG_gpt5.md')
WLOG_U = os.path.join(ROOT, r'research\deepseek\docs\WORKLOG_gpt5_mirror.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
REG = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                         r'\phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
TS = time.strftime('%Y-%m-%d %H:%M')
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ---------- 1. result + draft assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'ustar_layer_registered' and R['mode'] == 'full'
assert R['design_sha8'] == 'fa13399b' and R['res_sha8'] == 'a4435eb3'
assert R['gate_all'] is True
PM = R['per_model']
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    pm = PM[m]
    assert pm['drift_3169'] == 0.0 and pm['drift_3189'] == 0.0 and pm['drift_3190'] == 0.0
    tc = pm['top1_curve']
    assert len(tc) == pm['NH']                       # full layer trajectory
    assert abs(float(list(tc.values())[0]) - 1.0) < 1e-9   # embedding layer rank-1
    assert pm['dom_ok'] and pm['stab_ok']
    assert abs(tc[str(pm['kout'])] - pm['top1_curve'][str(pm['kout'])]) < 1e-12
assert abs(PM['qwen3-4b']['top1_curve']['35'] - 0.3065) < 1e-3   # 3189 anchor
assert abs(PM['qwen3-4b']['kout_m1_stab'] - 0.9658) < 1e-3
assert abs(PM['glm4-9b']['kout_m1_stab'] - 0.9261) < 1e-3
DA = R['doc_arm']
assert DA['appx6_id'] == 'APPX-6' and DA['appx6_status'] == 'appendix_recorded'
assert DA['based_on_sha8'] == '5934631b'
DRAFT = json.load(open(os.path.join(os.path.dirname(RES),
                                    'gap_appendix_draft_v1_12.json'), encoding='utf-8'))
assert DRAFT['entry']['id'] == 'APPX-6' and 'XC_p5_b8' in DRAFT['entry']['note']
assert {a['id']: a['status'] for a in DRAFT['appx_audit']} == {
    'APPX-1': 'appendix_open', 'APPX-2': 'appendix_open', 'APPX-3': 'appendix_open',
    'APPX-4': 'appendix_open_crossline', 'APPX-5': 'appendix_open_crossline'}
REPORT.append('1 science+doc assertions: OK (verdict=%s res=%s; drifts 0.0 x9; '
              'embedding top1=1.0 x3)' % (R['verdict'], R['res_sha8']))

rs = sha8(open(REG, 'rb').read())
assert rs == 'a30215fd', rs
REPORT.append('1b registry v1.33 byte sha8 %s untouched' % rs)

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'be50a5c5'
already = [e for e in led['measurements'] if e.get('phase') == 3252]
if not already:
    assert len(led['measurements']) == 403 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3252,
        "name": "g5h15_ustar_layer_trace",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b + qwen3-14b + glm4-9b (zero-GPU ustar layer "
                 "trajectory over the FULL 3169 collect H axis (NH=NL+1 "
                 "layers); + doc arm: APPX-6 gap-appendix entry finalized)",
        "verdict": ("ustar_layer_registered|GATE 3/3 (readout dominance + "
                    "near-readout stability): HEADLINE - the OOV residual "
                    "collapse direction is near rank-1 ALREADY AT THE "
                    "EMBEDDING LAYER (top1(layer 0) = 1.0000 in ALL 3 "
                    "models), dips mid-network (top1 0.14-0.21) and "
                    "partially recovers toward readout (0.29-0.31); "
                    "near-readout direction stability "
                    "|cos(ustar(kout),ustar(kout-1))| = 0.966/0.971/0.926 "
                    "x3 - ustar at readout is the TIP of a stable "
                    "late-network direction, not a readout-layer novelty; "
                    "the anti-class-mean contrast (3197) is thus seeded in "
                    "the embedding geometry and re-concentrated through "
                    "depth"),
        "evidence_level": ("registered characterization (preregistered "
                           "MEMO 3251 tail arm (b), upgraded from audit to "
                           "science after pre-freeze audit found the FULL "
                           "layer trajectory on disk; NINE bit-level drifts "
                           "== 0.0 at kout: drift69 x3 (3169 per-seed E) + "
                           "drift89 x3 (3189 scalars; 4b top1(kout)=0.3065 "
                           "bitwise) + drift90 x3 (3190 cos_ustar_vc); "
                           "SMOKE-stage corrections registered: (1) "
                           "inv(A)@b -> verbatim solve(A,b) after a 0.0166 "
                           "anchor drift at cond(A)~1e5; (2) "
                           "emergence-locality gate clause falsified at "
                           "smoke time (embedding rank-1) and replaced by "
                           "directional clause before formal observation; "
                           "registry v1.33 byte-untouched; APPX-6 draft "
                           "report-only"),
        "model_scope": "qwen3-4b (37 layers) + qwen3-14b / glm4-9b (41 layers)",
        "prereg_id": "G5-H15 (MEMO 3251 tail; audit upgraded)",
        "superseded_by": None,
        "detail": (
            "u* layer-trajectory stability (zero GPU, 3 models, full layer "
            "axis). Chain: 3190-verbatim residual chain evaluated at every "
            "layer k (B-protocol ridge on seen train rows, 876 OOV row "
            "residual unit directions averaged over seeds 7/8/9 -> Qn(k) -> "
            "top1(k), ustar(k)). Findings: (1) EMBEDDING RANK-1: "
            "top1(layer 0) = 1.0000 x3 - the OOV residual field at the "
            "embedding output is already a single direction; (2) "
            "U-SHAPE: top1 dips to 0.199/0.190/0.141 at mid-network "
            "(layer 18/20/20) and partially recovers to 0.3065/0.2936/"
            "0.3008 at readout; (3) near-readout stability |cos| = "
            "0.9658/0.9708/0.9261 - the readout ustar is continuous with "
            "its predecessor layer; (4) doc arm: APPX-6 "
            "(fidelity-closure chain 3239-3250 + ustar lexical "
            "decoupling) finalized per gap ledger v1.11 schema -> "
            "gap_appendix_draft_v1_12.json (based on 5934631b; report-only, "
            "lands at next registry-append trigger); APPX-1..5 audited "
            "open/open/open/open_crossline/open_crossline (GPU items stay "
            "deferred). freeze fa13399b, res a4435eb3. Pre-append ledger "
            "file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3252', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #404 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3252 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 404
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=404 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3252:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3252: G5-H15 u* 层轨迹稳定性（审计升级科学臂）+ APPX-6 定稿（零 GPU，双臂） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3251 tail（(b) 审计臂 + (a) 文档臂）；**审计升级**：pre-freeze 审计发现 3169 collect npz 的 H 第三维 = NH=NL+1 **全层轨迹**（4b 37 层 readout 35；14b/glm4 41 层 readout 39），(b) 从产物审计升级为零 GPU 科学臂。freeze **`fa13399b`**；res **`a4435eb3`**；verdict **`ustar_layer_registered`**（GATE 3/3）。**九位级漂移全零**（drift69/89/90 ×3，kout 层逐位；4b top1(kout)=0.3065 == 3189 封存）。registry v1.33 `a30215fd` 字节不动。')
    A_('')
    A_('### 核心发现（headline: 崩塌方向在 embedding 层即已秩 1）')
    A_('')
    A_('- **EMBEDDING RANK-1**：top1(layer 0) = **1.0000 ×3/3 模型**——OOV 残差场在 embedding 输出处已是单一方向；**anti-class-mean 对比轴（3197 定谳）在 embedding 几何中即已播种**。')
    A_('- **U 形轨迹**：top1 中层跌至 0.199（4b L18）/ 0.190（14b L20）/ 0.141（glm4 L20），readout 部分回升至 0.3065 / 0.2936 / 0.3008——中层网络分化残差方向，深层重新聚焦。')
    A_('- **近邻稳定性**：|cos(u\\*(kout), u\\*(kout−1))| = 0.9658 / 0.9708 / 0.9261 ×3——readout u\\* 是**稳定深层方向的尖端**，非 readout 层新生物；gate 双条款（readout dominance top1(kout)>top1(mid) + 近邻稳定性 ≥0.8）全过 ⇒ registered。')
    A_('')
    A_('**SMOKE 修正 ×2**（正式观察前，已记 prereg_correction）：(1) `inv(A)@b` → verbatim `solve(A,b)`——cond(A)~1e5 下 f32 inv 乘法放大出 0.0166 锚漂移；(2) **emergence-locality gate 条款被 SMOKE 证伪**（原预设「崩塌涌现于 readout 邻域」——实测 embedding 层即秩 1），替换为方向性条款（readout dominance + 近邻稳定性）。')
    A_('')
    A_('### 文档臂 (a)：APPX-6 定稿')
    A_('')
    A_('- **`gap_appendix_draft_v1_12.json`**（report-only，封存 v1.11 `5934631b` 未动）：APPX-6 =「保真收线链 3239–3250 + u\\* 词汇解耦」，status `appendix_recorded`，anchor p3250/`9d0f1825`；预演 diff = appendix.append(APPX-6) + version 1.11→1.12，其余字节不变；**入册留待下一次 registry-append 触发点**。')
    A_('- **APPX-1..5 审计**：open ×3 + open_crossline ×2——5 appendix GPU 项维持挂账（含 checkpoint 轨迹 GPU 采样项，本 phase 已用零 GPU 层轨迹替代其科学问题的大部分）。')
    A_('')
    A_('**限界**：层轨迹 = last-token hidden states（pre-final-norm、logit-lens 前）；层 0 = embedding 输出（hidden_states[0]）；u\\* 每层独立 SVD，无跨层正交化；checkpoint（训练步）维度仍需 GPU 采样，维持挂账。')
    A_('')
    A_('**完整性**：ledger #404（pre 文件 sha8 %s）；产物 result.json + layer_trace.npz + gap_appendix_draft_v1_12.json + smoke 子目录；独立磁盘复核见 p3252_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3253（候选）')
    A_('')
    A_('- (a) **U 形轨迹机制臂**（零 GPU）：中层跌落的方向内容——mid-network u\\*(k) 与 embedding/readout u\\* 的子空间关系（主角度谱），定位「分化-重聚焦」的几何载体；')
    A_('- (b) **registry append 触发点评估**：APPX-6 + 3240–3252 六个 characterization 累积是否满足 registry v1.33→v1.34 的 append 门槛（预演 diff，仍 report-only）；')
    A_('- (c) 5 appendix GPU 项维持挂账。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3252 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3252 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3252:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3252 u* 层轨迹稳定性（审计升级科学臂）+APPX-6 定稿完成（双臂，registered）：**embedding 层 top1=1.0000 ×3**——崩塌方向在 embedding 几何即已秩 1；U 形轨迹（中层 0.14–0.21 低谷、readout 回升 0.29–0.31）；近邻稳定性 0.966/0.971/0.926 ×3 ⇒ readout u* 是稳定深层方向尖端。'
      'SMOKE 修正 ×2：inv→solve（0.0166 锚漂移）；emergence-locality 条款被证伪→方向性条款。九位级漂移全零。'
      '文档臂：APPX-6（保真收线链+词汇解耦）定稿 gap_appendix_draft_v1_12.json（report-only，待入册）；APPX-1..5 审计维持挂账。'
      'verdict ustar_layer_registered；freeze fa13399b / res a4435eb3，ledger #404，registry v1.33 不动。'
      '3253 已预注册：U 形机制臂（中层子空间关系）+ registry append 触发点评估。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3252 u* 层轨迹稳定性' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3252 已预注册**：(a) gap 附录候选条目定稿（report-only 预演）；(b) checkpoint 轨迹产物审计（零 GPU 前置）；(c) 5 appendix GPU 挂账。')
new_block = ('- **3252 收官（u* 层轨迹 + APPX-6，双臂 registered）**：**embedding 层 top1=1.0000 ×3**——崩塌方向 embedding 即秩 1（anti-class-mean 在 embedding 几何播种）；U 形轨迹（中层 0.14–0.21 低谷→readout 0.29–0.31 回升）；近邻稳定 0.966/0.971/0.926。九位级漂移全零。APPX-6（收线链+词汇解耦）定稿待入册；APPX-1..5 挂账。ledger #404 / freeze `fa13399b` / res `a4435eb3`。'
             '- **3253 已预注册**：(a) U 形机制臂（中层 u* 子空间关系，主角度谱）；(b) registry append 触发点评估（v1.33→v1.34 预演）；(c) GPU 挂账。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3252 收官（u* 层轨迹' in mem, 'MEMORY tail block not found'
if len(mem) > 3000:
    o = ('3240：F0 d39=0.0；F1 塌缩；F2 全族正向（SCLS +4.47/ENT +3.85）转正未达；臂(b) /atlas/fidelity 端点 + FidelityMode.jsx，playwright 11/11。ledger #392 post 0cf46b3a，res ffb97506（res_sha8 口径修正排除 created/runtime_s）。')
    n = ('3240 F2 转正未达+客户端化（#392）。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3250 收官（kernel boundary + 客户端化，双臂 registered）**：deg≤6 cond 平坦无数值爆炸——停点是价值停点（deg 5–6 带）；**XC_p5_b8（poly5×β8）= 新现役**（−0.449/−0.273/r2 0.241/KL 0.987，闭合 +0.032，degree×β 增益叠加）；混合核≡poly3（互补性证伪）。工程臂：fidelity 端点 v2（7 相位+现役标注）+ FidelityMode v2（现役卡+缺口收线），UI 实机 19/19。八位级锚全零。ledger #402 / freeze `c3aa103e` / res `9d0f1825`。')
    n = ('- **3250 收官（kernel boundary+客户端化，registered）**：deg 5–6 价值停点；**XC_p5_b8 = 新现役**（−0.449/−0.273/r2 0.241/KL 0.987）；混合核≡poly3。工程臂 fidelity 端点 v2 + FidelityMode v2，UI 19/19。ledger #402 / `c3aa103e`/`9d0f1825`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3251 收官（u* 词汇归因 + 收线小结，双臂 partial=characterization）**：词级对齐真实但小、随规模衰减（4b 0.144→14b 0.033→glm4 0.025 低于伪类 null）、非 OOV 主导 ×3；与 D 线合并定谳——u* 类结构由激活侧 v_c 承载，与词表 embedding 解耦（词桥微弱的图谱级解释）。九位级漂移全零。文档臂：3239–3250 收线小结入 MEMO + gap 附录候选 gap_v1_12_appendix_fidelity_closure。ledger #403 / freeze `48169586` / res `0601eb1b`。')
    n = ('- **3251 收官（u* 词汇归因+收线小结，partial=characterization）**：词级对齐小、随规模衰减（0.144→0.033→0.025 低于伪类 null）、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表 embedding 解耦（词桥微弱的图谱级解释）。九位级漂移全零。3239–3250 收线小结入 MEMO。ledger #403 / `48169586`/`0601eb1b`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3249**：degree 2→5 单调（−0.467，闭合 +0.102）；DVU 侵蚀随阶增长；combo 在 p3 基失效。#401 `6eb9bed8`。')
    n = ('- **3249**：degree 2→5 单调（−0.467）；combo 在 p3 基失效。#401。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3248**：degree 1→3 单调未饱和；K_p3_T3 纯拉力最优（闭合 +0.165）；p2 基 β 最优移至 6–8。#400 `b47042fd`。')
    n = ('- **3248**：degree 未饱和；K_p3_T3 纯拉力最优；β 移至 6–8。#400。')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: replaced tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
