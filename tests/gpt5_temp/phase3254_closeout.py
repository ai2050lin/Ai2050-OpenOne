# -*- coding: utf-8 -*-
"""Phase 3254 closeout: five writes (result assertions / ledger #406 / MEMO /
dual wlog / MEMORY.md). Idempotent per write; ledger append chain-checked."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3254\g5h17_registry_landing\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'research\gpt5\docs\WORKLOG_gpt5.md')
WLOG_U = os.path.join(ROOT, r'research\deepseek\docs\WORKLOG_gpt5_mirror.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = time.strftime('%Y-%m-%d %H:%M')
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ---------- 1. result + landed-file assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'registry_landed' and R['mode'] == 'full'
assert R['design_sha8'] == '8be6ccad' and R['res_sha8'] == '48ced87e'
assert R['revert_check'] == {'gap': True, 'registry': True}
G = R['gap_v112']
assert G['appendix_n'] == 6 and G['file_sha8'] == 'aa8cff21'
assert G['appx6']['id'] == 'APPX-6' and G['appx6']['anchor_sha8'] == '9d0f1825'
RG = R['registry_v134']
assert RG['anchor_files_n'] == 28 and RG['file_sha8'] == '3deabf25'
assert RG['anchors_added'] == ['p3246', 'p3250', 'p3252', 'p3253']
A = R['audit']
assert A['pointer_diff_on_disk'].startswith('planned set exactly')
assert all(A['three_ledger'].values())
# landed files on disk
RD = os.path.dirname(RES)
g12 = json.load(open(os.path.join(RD, 'gap_ledger_v1_12.json'), encoding='utf-8'))
r34 = json.load(open(os.path.join(RD, 'atlas_registry_v1_34.json'), encoding='utf-8'))
assert g12['version'] == '1.12' and len(g12['appendix']) == 6
assert g12['appendix'][-1]['id'] == 'APPX-6'
assert r34['version'] == '1.34' and len(r34['provenance']['anchor_files']) == 28
assert len(r34['upgrade_log']) == 13
# sealed predecessors byte-untouched
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
assert sha8(open(os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing'
                              r'\gap_ledger_v1_11.json'), 'rb').read()) == '5934631b'
assert sha8(open(os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank'
                              r'\atlas_registry_v1_33.json'), 'rb').read()) == 'a30215fd'
REPORT.append('1 landing assertions: OK (verdict=%s res=%s; gap v1.12 %s + '
              'registry v1.34 %s; revert check bitwise x2; audit 6/6; sealed '
              'v1.11/v1.33 untouched)' % (R['verdict'], R['res_sha8'],
                                          G['file_sha8'], RG['file_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'a13359da'
already = [e for e in led['measurements'] if e.get('phase') == 3254]
if not already:
    assert len(led['measurements']) == 405 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3254,
        "name": "g5h17_registry_landing",
        "line": "G",
        "date": "2026-10-20",
        "model": "doc-only landing (zero GPU): executes the 3253 previewed "
                 "append diff -- gap_ledger v1.11->v1.12 (APPX-6) + atlas "
                 "registry v1.33->v1.34 (4 anchors + upgrade_log)",
        "verdict": ("registry_landed|HEADLINE - the fidelity-closure chain "
                    "3239-3250 + ustar geometry trilogy (3251 lexical "
                    "decoupling / 3252 embedding rank-1 + layer trace / 3253 "
                    "subspace carrier continuity) is now IN the atlas: "
                    "gap v1.12 appendix APPX-6 (anchor p3250/9d0f1825); "
                    "registry v1.34 anchors p3246/p3250/p3252/p3253; "
                    "features/failures UNCHANGED (all characterization, no "
                    "E-level upgrade claimed)"),
        "evidence_level": ("doc landing (preregistered MEMO 3253 tail (a)+"
                           "(b); APPEND-ONLY BYTE DISCIPLINE: writer "
                           "json.dumps(ensure_ascii=False, indent=1) "
                           "round-trip byte-exact on both sealed "
                           "predecessors; REVERT CHECK bitwise x2; pointer "
                           "diff == planned set EXACTLY (gap 4 ptrs / "
                           "registry 7 ptrs); post-landing audit 6/6; "
                           "sealed v1.11 5934631b / v1.33 a30215fd "
                           "byte-unchanged after landing; SMOKE-stage "
                           "device corrections registered: pointer-diff "
                           "granularity normalization (list appends are "
                           "LEN-level events), DESIGN unchanged)"),
        "model_scope": "n/a (ledger/registry/gap files only)",
        "prereg_id": "G5-H17 (MEMO 3253 tail)",
        "superseded_by": None,
        "detail": (
            "Landing phase. gap_ledger_v1_12.json (file sha8 aa8cff21): "
            "appendix 5->6 (APPX-6 landed verbatim per 6-key schema from "
            "the frozen 3252 draft; anchor p3250/9d0f1825 = sealed 3250 "
            "result content sha), version 1.11->1.12, updated 2026-10-20, "
            "supersedes v1_11 pointer; GAP-1..5 + APPX-1..5 bytes "
            "untouched. atlas_registry_v1_34.json (file sha8 3deabf25): "
            "provenance.anchor_files 24->28 (p3246 41b42d9b / p3250 "
            "f989ff42 / p3252 d45c08e9 / p3253 dc2ccd4a, result.json FILE "
            "sha8 convention), upgrade_log 12->13 (append-landing entry), "
            "version 1.33->1.34, supersedes v1_33 pointer; features/"
            "failures/taxonomy/principles/census_3238 untouched. Audit "
            "arm (b): pointer diff on disk == planned set; three-ledger "
            "consistency 6/6 (APPX-6 anchor == #404 freeze; p3253 anchor "
            "sha == sealed result file sha; sealed predecessors "
            "byte-unchanged; versions bumped; APPX-1..5 intact before "
            "APPX-6; ledger phases 3246/3250/3252/3253 present). freeze "
            "8be6ccad, res 48ced87e. Pre-append ledger file sha8 "
            + pre_file_sha8 + "."),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3254', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #406 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3254 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 406
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=406 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3254:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3254: G5-H17 registry/gap append 落册（执行 3253 预演 diff）+ 落册后审计（零 GPU，文档落地点） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3253 tail（(a) 落册臂 + (b) 审计臂）；执行 3253 `append_trigger_MET` 预演。freeze **`8be6ccad`**；res **`48ced87e`**；verdict **`registry_landed`**。**append-only 字节纪律全过**：writer（indent=1）对两封存前驱 round-trip 逐字节成立；**REVERT CHECK 逐位 ×2**（新对象撤回计划改动 == 封存前驱原始字节）；指针 diff == 计划集**精确相等**（gap 4 ptrs / registry 7 ptrs）。')
    A_('')
    A_('### 落地内容')
    A_('')
    A_('- **`gap_ledger_v1_12.json`**（文件 sha8 **`aa8cff21`**）：appendix 5→6——**APPX-6（保真收线链 3239–3250 + u\\* 词汇解耦）入册**，逐字段取自 3252 冻结 draft 的 6 键 schema（anchor p3250/`9d0f1825` = 封存 3250 result 内容 sha）；version 1.11→1.12、updated→2026-10-20、supersedes 指针更新；GAP-1..5 与 APPX-1..5 字节不动。')
    A_('- **`atlas_registry_v1_34.json`**（文件 sha8 **`3deabf25`**）：provenance.anchor_files 24→28（p3246 `41b42d9b` / p3250 `f989ff42` / p3252 `d45c08e9` / p3253 `dc2ccd4a`，result.json **文件 sha** 口径——与 a1_behavioral_band `ac5d4d74` 同口径）；upgrade_log 12→13（append-landing 条目）；version 1.33→1.34；**features/failures/证据分级/principles/census_3238 零语义改动**（3239–3253 全为 characterization，无 E 级升级声明）。')
    A_('')
    A_('### 审计臂 (b)：三账一致性 6/6')
    A_('')
    A_('APPX-6 anchor == #404 冻结值 / registry p3253 anchor sha == 封存 3253 result 文件 sha / 封存 v1.11 `5934631b` + v1.33 `a30215fd` 落册后逐字节不变 / 版本号正确 / APPX-1..5 在 APPX-6 之前完好 / ledger 3246/3250/3252/3253 phase 在册。**SMOKE 装置修正 ×2**（DESIGN 不变，freeze 不动）：指针 diff 粒度归一化（list 追加为 LEN 级事件而非 [i] ADDED）+ 审计块同口径。')
    A_('')
    A_('**图谱状态**：G 线现役 registry **v1.34 `3deabf25`** + gap **v1.12 `aa8cff21`**（v1.33/v1.11 封存退役）。u\\* 几何三部曲（3251 词汇解耦 / 3252 embedding 秩 1+层轨迹 / 3253 载轴连续性）与保真收线链 3239–3250 全部在册。')
    A_('')
    A_('**完整性**：ledger #406（pre 文件 sha8 %s）；产物 gap_ledger_v1_12.json + atlas_registry_v1_34.json + result.json + smoke 子目录；独立磁盘复核见 p3254_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3255（候选）')
    A_('')
    A_('- (a) **registry v1.34 客户端化臂**：/atlas/gaps 端点 + 前端展示升级到 v1.34/v1.12（APPX-6 附录卡 + anchor_files 升级链），UI 实机校验；')
    A_('- (b) **u\\* 几何三部曲综合臂**（零 GPU 文档）：3189→3253 u\\* 链收束成 MEMO 专节（几何图谱：embedding 播种-中层衰减-readout 重聚焦-词汇解耦）；')
    A_('- (c) 5 appendix GPU 项 + N 线施加 + 水果类/K4 处置维持挂账。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3254 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3254 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3254:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3254 registry/gap append 落册完成（registry_landed）：gap v1.12 `aa8cff21`（APPX-6 入册，anchor p3250/9d0f1825）+ registry v1.34 `3deabf25`（anchor_files 24→28：p3246/p3250/p3252/p3253 文件 sha；upgrade_log +1）；features/failures 零语义改动。'
      'append-only 字节纪律：REVERT CHECK 逐位 ×2 + 指针 diff == 计划集精确（gap 4/registry 7 ptrs）；三账一致性审计 6/6；封存 v1.11/v1.33 字节不变。'
      'freeze 8be6ccad / res 48ced87e，ledger #406。G 线现役 registry v1.34 + gap v1.12。'
      '3255 已预注册：registry v1.34 客户端化（/atlas/gaps + UI）+ u* 几何三部曲综合（零 GPU 文档）。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3254 registry/gap append 落册' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md ----------
mem = open(MEM, encoding='utf-8').read()
o1 = ('## 2 G 线现状（registry v1.32 3c3f0484 + gap v1.11 5934631b）')
n1 = ('## 2 G 线现状（registry v1.34 3deabf25 + gap v1.12 aa8cff21；v1.33 a30215fd/v1.11 5934631b 封存）')
if o1 in mem:
    mem = mem.replace(o1, n1)
old_block = ('- **3254 已预注册**：(a) registry/gap append 落册（gap v1.11→v1.12 APPX-6 + registry v1.33→v1.34，append-only 字节纪律）；(b) 落册后逐字段审计；(c) GPU 挂账。')
new_block = ('- **3254 落册（registry_landed）**：gap v1.12 `aa8cff21`（APPX-6 入册）+ registry v1.34 `3deabf25`（anchors ×4+upgrade_log）；REVERT 逐位×2、指针 diff 精确、三账审计 6/6；features/failures 零改动。#406 `8be6ccad`/`48ced87e`。'
             '- **3255 已预注册**：(a) registry v1.34 客户端化（/atlas/gaps + UI 实机）；(b) u* 几何三部曲综合（零 GPU 文档）；(c) GPU 挂账。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3254 落册（registry_landed）' in mem, 'MEMORY tail block not found'
if len(mem) > 3000:
    o = ('- **3248**：degree 1→3 单调未饱和；K_p3_T3 纯拉力最优（闭合 +0.165）；p2 基 β 最优移至 6–8。#400 `b47042fd`。')
    n = ('- **3248**：degree 未饱和；K_p3_T3 最优；β 6–8。#400。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3250 收官（kernel boundary+客户端化，registered）**：deg 5–6 价值停点；**XC_p5_b8 = 新现役**（−0.449，r2 0.241/KL 0.987）；混合核≡poly3。工程臂 fidelity 端点 v2+FidelityMode v2，UI 19/19。ledger #402 / `c3aa103e`/`9d0f1825`。')
    n = ('- **3250**：deg 5–6 价值停点；**XC_p5_b8 = 新现役**（−0.449，r2 0.241/KL 0.987）；混合核≡poly3。工程臂 fidelity v2+UI 19/19。#402 `c3aa103e`/`9d0f1825`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3251 收官（u* 词汇归因，partial）**：词级对齐小、随规模衰减（0.144→0.033→0.025）、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表 embedding 解耦。九位级漂移全零。ledger #403 / `48169586`/`0601eb1b`。')
    n = ('- **3251**：词级对齐小、随规模衰减（0.144→0.033→0.025）、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表解耦。#403 `48169586`/`0601eb1b`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('append 触发点 MET（c1–c5 全过），预演 diff report-only，**决定 3254 落册**。ledger #405 / freeze `8b10eba4` / res `e19e2844`。')
    n = ('append 触发点 MET，**决定 3254 落册**。#405 `8b10eba4`/`e19e2844`。')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: updated header + tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
