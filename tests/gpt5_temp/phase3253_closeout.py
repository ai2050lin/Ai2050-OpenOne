# -*- coding: utf-8 -*-
"""Phase 3253 closeout: five writes (result assertions / ledger #405 / MEMO /
dual wlog / MEMORY.md). Idempotent per write; ledger append chain-checked."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3253\g5h16_ustar_subspace\result.json')
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


# ---------- 1. result + preview assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'ustar_subspace_registered' and R['mode'] == 'full'
assert R['design_sha8'] == '8b10eba4' and R['res_sha8'] == 'e19e2844'
assert R['gate_all'] is True
PM = R['per_model']
SIG = {'qwen3-4b': (23, 0.6283, 0.1390), 'qwen3-14b': (28, 0.7464, 0.1769),
       'glm4-9b': (15, 0.3352, 0.1011)}
for m in SIG:
    pm = PM[m]
    km, so, se = SIG[m]
    assert pm['drift_3169'] == 0.0 and pm['drift_3189'] == 0.0 \
        and pm['drift_3190'] == 0.0 and pm['drift_3252'] == 0.0
    assert pm['k_mid'] == km and pm['cont_ok'] and pm['aff_ok'] and pm['gate_model']
    assert abs(pm['sigma1']['mid_out'] - so) < 1e-3
    assert abs(pm['sigma1']['mid_emb'] - se) < 1e-3
    assert len(pm['top1_curve']) == pm['NH']
    assert len(pm['spectra']['mid_out']) == 8
AB = R['arm_b']
assert AB['trigger_status'] == 'append_trigger_MET'
assert all(v['pass'] for v in AB['criteria'].values())
assert AB['criteria']['c1_closed_phases_gte6']['measured'] == 14
PV = json.load(open(os.path.join(os.path.dirname(RES),
                                 'registry_append_preview_v1_34.json'),
                    encoding='utf-8'))
assert PV['report_only'] is True
assert PV['based_on']['registry'] == 'v1.33/a30215fd'
assert PV['based_on']['gap_ledger'] == 'v1.11/5934631b'
assert 'p3246' in PV['planned_diff']['registry']
REPORT.append('1 science+eval assertions: OK (verdict=%s res=%s; drifts 0.0 '
              'x12 incl drift_3252 bitwise; sigma1 mid-out 0.628/0.746/0.335; '
              'append_trigger_MET c1-c5 all pass)' % (R['verdict'], R['res_sha8']))

rs = sha8(open(REG, 'rb').read())
assert rs == 'a30215fd', rs
REPORT.append('1b registry v1.33 byte sha8 %s untouched (report-only held)' % rs)

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'fdca5c5c'
already = [e for e in led['measurements'] if e.get('phase') == 3253]
if not already:
    assert len(led['measurements']) == 404 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3253,
        "name": "g5h16_ustar_subspace",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b + qwen3-14b + glm4-9b (zero-GPU ustar SUBSPACE "
                 "mechanism arm: top-R=8 principal-angle spectra between "
                 "embedding/mid/readout layer bases over the 3169 collect H "
                 "axis; + arm (b): registry-append trigger evaluation, "
                 "report-only)",
        "verdict": ("ustar_subspace_registered|GATE 3/3 (continuity + "
                    "mid-readout affinity): HEADLINE - the mid-network dip is "
                    "ATTENUATION OF THE SAME AXIS, not replacement: "
                    "sigma1(top-8 mid basis, top-8 readout basis) = "
                    "0.628/0.746/0.335 x3 models (3-6x the mid-embedding "
                    "affinity 0.139/0.177/0.101; subspace overlap ovr "
                    "0.155/0.233/0.031 vs 0.005/0.007/0.003) - the readout "
                    "collapse direction remains the dominant single shared "
                    "direction through the dip; embedding ustar is a "
                    "SEPARATE phenomenon (mid-emb ~0.10-0.18, emb-out "
                    "0.06-0.20 all near-dissociated)"),
        "evidence_level": ("registered characterization (preregistered "
                           "MEMO 3252 tail arms (a)+(b); TWELVE bit-level "
                           "drifts == 0.0: drift69 x3 + drift89 x3 + drift90 "
                           "x3 at kout + drift_3252 x3 (top1 curve bitwise "
                           "equal to the sealed 3252 result - cross-phase "
                           "device anchor); gate pinned pre-observation "
                           "(continuity sigma1>=0.3 + affinity), glm4 "
                           "continuity is the tight pass at 0.335; registry "
                           "v1.33 byte-untouched; append preview "
                           "report-only)"),
        "model_scope": "qwen3-4b (37 layers) + qwen3-14b / glm4-9b (41 layers)",
        "prereg_id": "G5-H16 (MEMO 3252 tail)",
        "superseded_by": None,
        "detail": (
            "u* subspace mechanism arm (zero GPU, 3 models, full layer "
            "axis, R=8 basis). k_mid (argmin top1 over [1,kout]) = L23/L28/"
            "L15 with top1 0.174/0.173/0.131. Findings: (1) CONTINUITY - "
            "sigma1(mid,out) = 0.6283 (4b) / 0.7464 (14b) / 0.3352 (glm4, "
            "tightest pass): the readout ustar is partially present at the "
            "dip; the U-shape is amplitude modulation along one continuous "
            "carrier, so the anti-class-mean axis seeded at embedding "
            "(3252) re-concentrates through depth rather than being "
            "replaced; (2) AFFINITY - sigma1(mid,out) > sigma1(mid,emb) x3 "
            "(0.628 vs 0.139 / 0.746 vs 0.177 / 0.335 vs 0.101) and ovr "
            "ratios ~30-40x; (3) EMBEDDIING DISSOCIATION - mid-emb and "
            "emb-out sigma1 all <= 0.20: the embedding-layer rank-1 "
            "direction (3252) is NOT the direction carried to readout; the "
            "carrier axis is (re)formed downstream of embedding. Arm (b): "
            "append trigger MET (c1: 14 closed phases 3239-3252 >= 6; c2: "
            "APPX-6 draft based_on 5934631b; c3: no blocked/failed verdict; "
            "c4: incumbent XC_p5_b8 sealed in 3250; c5: registry v1.33 "
            "a30215fd) -> registry_append_preview_v1_34.json written "
            "(report-only; planned diff: gap v1.11->v1.12 appendix APPX-6 + "
            "registry v1.33->v1.34 provenance anchors p3246/p3250/p3252, "
            "features/failures untouched); decision: LAND at 3254 closeout. "
            "freeze 8b10eba4, res e19e2844. Pre-append ledger file sha8 "
            + pre_file_sha8 + "."),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3253', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #405 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3253 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 405
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=405 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3253:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3253: G5-H16 u* 子空间机制臂（U 形主角度谱）+ registry append 触发点评估（零 GPU，双臂） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3252 tail（(a) 科学臂 + (b) 评估臂）；freeze **`8b10eba4`**；res **`e19e2844`**；verdict **`ustar_subspace_registered`**（GATE 3/3）。**十二位级漂移全零**：drift69/89/90 ×3（kout 逐位）+ **drift_3252 ×3（top1 曲线与封存 3252 逐位相等——跨 phase 装置锚首次入链）**。registry v1.33 `a30215fd` 字节不动（append 预演 report-only）。')
    A_('')
    A_('### 核心发现（headline: 中层低谷 = 同一载轴的衰减，非替换）')
    A_('')
    A_('- **连续性**：σ₁(top-8 mid 基, top-8 readout 基) = **0.6283（4b）/ 0.7464（14b）/ 0.3352（glm4，最紧通过）**——readout 崩塌方向在低谷层仍部分在场；U 形轨迹是**沿一条连续载轴的幅度调制**，anti-class-mean 轴（3252 定谳 embedding 播种）经深度重新聚焦而非被替换。')
    A_('- **中-读出亲和**：σ₁(mid,out) > σ₁(mid,emb) ×3（0.628 vs 0.139 / 0.746 vs 0.177 / 0.335 vs 0.101）；子空间重叠 ovr 0.155/0.233/0.031 vs 0.005/0.007/0.003（**30–40×**）。')
    A_('- **embedding 解离定谳**：mid-emb 与 emb-out σ₁ 全部 ≤0.20——**embedding 层秩 1 方向（3252）并不是被输送到 readout 的方向**；载轴在 embedding 下游（重）形成。3251 词汇解耦 + 3252 embedding 秩 1 + 本 phase 载轴连续性合成完整 u* 几何图谱。')
    A_('')
    A_('**门条款（预注册，正式观察前钉定）**：continuity σ₁(mid,out)≥0.3 + affinity σ₁(mid,out)>σ₁(mid,emb)，3/3 全过（glm4 continuity 0.335 为最紧通过）；无 SMOKE 修正（SMOKE 一次通过）。')
    A_('')
    A_('### 评估臂 (b)：registry append 触发点 MET')
    A_('')
    A_('- **`registry_append_preview_v1_34.json`**（report-only，封存 v1.33 `a30215fd` / v1.11 `5934631b` 未动）：五判据全过——c1 收官 phase 14（3239–3252，门槛 ≥6）/ c2 APPX-6 draft `5934631b` / c3 无 blocked/failed verdict / c4 现役 XC_p5_b8 封存于 3250 / c5 registry sha 复核。')
    A_('- **预演 diff**：gap v1.11→v1.12（appendix.append(APPX-6)+版本号，GAP-1..5 与 APPX-1..5 字节不动）+ registry v1.33→v1.34（provenance.anchor_files 增 p3246/p3250/p3252(+p3253)；features/failures/证据分级**零语义改动**）。**决定：3254 收官时落册（append 落地点）**。')
    A_('')
    A_('**限界**：零 GPU；子空间基 = Qn(k) top-8 右奇异向量（R=8 预注册钉定）；k_mid = argmin top1 于 [1,kout]（L23/L28/L15）；σ₁ 为主角度余弦最大值（|cos| 口径）；5 appendix GPU 项维持挂账。')
    A_('')
    A_('**完整性**：ledger #405（pre 文件 sha8 %s）；产物 result.json + subspace.npz + registry_append_preview_v1_34.json + smoke 子目录；独立磁盘复核见 p3253_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3254（候选）')
    A_('')
    A_('- (a) **registry append 落册臂**：执行 3253 预演 diff——gap v1.11→v1.12（APPX-6 入册）+ registry v1.33→v1.34（anchor_files 增补），append-only 字节纪律（legacy 字节保全断言 + 落册后全锚复核）；')
    A_('- (b) **落册后审计**：registry/gap 新旧版本 diff 逐字段断言（仅计划字段变动）+ ledger/registry/gap 三账一致性；')
    A_('- (c) 5 appendix GPU 项维持挂账。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3253 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3253 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3253:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3253 u* 子空间机制臂 + registry append 触发点评估完成（双臂，registered）：**中层低谷 = 同一载轴的衰减，非替换**——σ₁(mid,out)=0.628/0.746/0.335 ×3（mid-emb 亲和的 3–6 倍，ovr 30–40×），anti-class-mean 轴经深度重新聚焦；embedding 秩 1 方向与被输送方向解离（全 σ₁≤0.20）。'
      '十二位级漂移全零（含 drift_3252 跨 phase 装置锚逐位）。评估臂：append 触发点 MET（c1–c5 全过，14 收官 phase），registry_append_preview_v1_34.json report-only，**决定 3254 落册**。'
      'verdict ustar_subspace_registered；freeze 8b10eba4 / res e19e2844，ledger #405，registry v1.33 不动。'
      '3254 已预注册：registry/gap append 落册（执行预演 diff，append-only 字节纪律）+ 落册后审计。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3253 u* 子空间机制臂' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3253 已预注册**：(a) U 形机制臂（中层 u* 子空间关系，主角度谱）；(b) registry append 触发点评估（v1.33→v1.34 预演）；(c) GPU 挂账。')
new_block = ('- **3253 收官（u* 子空间机制臂 + append 触发点评估，双臂 registered）**：**中层低谷 = 同一载轴衰减，非替换**——σ₁(mid,out)=0.628/0.746/0.335 ×3（mid-emb 的 3–6 倍、ovr 30–40×）；embedding 秩 1 方向与被输送方向解离（σ₁≤0.20）。十二位级漂移全零（含 drift_3252 跨 phase 锚）。append 触发点 MET（c1–c5 全过），预演 diff report-only，**决定 3254 落册**。ledger #405 / freeze `8b10eba4` / res `e19e2844`。'
             '- **3254 已预注册**：(a) registry/gap append 落册（gap v1.11→v1.12 APPX-6 + registry v1.33→v1.34，append-only 字节纪律）；(b) 落册后逐字段审计；(c) GPU 挂账。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3253 收官（u* 子空间机制臂' in mem, 'MEMORY tail block not found'
if len(mem) > 3000:
    o = ('- **3246**：FLIP=INFORMATION_GAP（LS oracle +0.329 转正；拉力上限证明→转正须偏移型分量）；XC_b4 现役（−0.772/r2 0.203）。#398 `df7ec079`。')
    n = ('- **3246**：FLIP=INFORMATION_GAP（LS oracle 转正）；XC_b4 −0.772。#398。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3247**：XC_poly_b4 新现役（−0.651/−0.435 零回退，闭合 +0.142）；增益载体=映射函数类；联合 donor/per-template 证伪。#399 `d0def478`。')
    n = ('- **3247**：XC_poly_b4 新现役（−0.651，闭合 +0.142）。#399。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3250 收官（kernel boundary+客户端化，registered）**：deg 5–6 价值停点；**XC_p5_b8 = 新现役**（−0.449/−0.273/r2 0.241/KL 0.987）；混合核≡poly3。工程臂 fidelity 端点 v2 + FidelityMode v2，UI 19/19。ledger #402 / `c3aa103e`/`9d0f1825`。')
    n = ('- **3250 收官（kernel boundary+客户端化，registered）**：deg 5–6 价值停点；**XC_p5_b8 = 新现役**（−0.449，r2 0.241/KL 0.987）；混合核≡poly3。工程臂 fidelity 端点 v2+FidelityMode v2，UI 19/19。ledger #402 / `c3aa103e`/`9d0f1825`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3251 收官（u* 词汇归因+收线小结，partial=characterization）**：词级对齐小、随规模衰减（0.144→0.033→0.025 低于伪类 null）、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表 embedding 解耦（词桥微弱的图谱级解释）。九位级漂移全零。3239–3250 收线小结入 MEMO。ledger #403 / `48169586`/`0601eb1b`。')
    n = ('- **3251 收官（u* 词汇归因，partial）**：词级对齐小、随规模衰减（0.144→0.033→0.025）、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表 embedding 解耦。九位级漂移全零。ledger #403 / `48169586`/`0601eb1b`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3252 收官（u* 层轨迹 + APPX-6，双臂 registered）**：**embedding 层 top1=1.0000 ×3**——崩塌方向 embedding 即秩 1（anti-class-mean 在 embedding 几何播种）；U 形轨迹（中层 0.14–0.21 低谷→readout 0.29–0.31 回升）；近邻稳定 0.966/0.971/0.926。九位级漂移全零。APPX-6（收线链+词汇解耦）定稿待入册；APPX-1..5 挂账。ledger #404 / freeze `fa13399b` / res `a4435eb3`。')
    n = ('- **3252 收官（u* 层轨迹 + APPX-6，双臂 registered）**：**embedding 层 top1=1.0000 ×3**——崩塌方向 embedding 即秩 1；U 形轨迹（中层 0.14–0.21 低谷→readout 0.29–0.31 回升）；近邻稳定 0.966/0.971/0.926。九位级漂移全零。APPX-6 定稿待入册；APPX-1..5 挂账。ledger #404 / `fa13399b`/`a4435eb3`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- 3237 收官（F21+GAP-5 三段式，registry v1.32 `3c3f0484` / gap v1.11 `5934631b`）；3225–3232 E3 干预线锚（registry v1.29–v1.31，细节回 MEMO）。')
    n = ('- 3237 收官（F21+GAP-5，registry v1.32/gap v1.11）；3225–3232 E3 锚（细节回 MEMO）。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- 3238–3245 收线链（census/KL 恢复/F2/G1e/T3_dual/信息极限/归因反转/combo，#390–#397）——细节回 MEMO。')
    n = ('- 3238–3245 收线链（#390–#397，细节回 MEMO）。')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: replaced tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
