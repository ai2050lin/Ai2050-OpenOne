# -*- coding: utf-8 -*-
"""Phase 3255 closeout: five writes (result assertions / ledger #407 / MEMO /
dual wlog / MEMORY.md). Idempotent per write; ledger append chain-checked."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3255\g5h18_client_synthesis\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'research\gpt5\docs\WORKLOG_gpt5.md')
WLOG_U = os.path.join(ROOT, r'research\deepseek\docs\WORKLOG_gpt5_mirror.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = time.strftime('%Y-%m-%d %H:%M')
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ---------- 1. result + engineering assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'client_synthesis_registered' and R['mode'] == 'full'
assert R['design_sha8'] == '1d19fc3d' and R['res_sha8'] == 'ac6bd1d9'
assert R['engineering']['endpoint_version'] == '1.12'
assert R['engineering']['appendix_n'] == 6
assert R['engineering']['ui_check'] == '17/17'
assert R['doc_arm_status'] == 'appended'
MEMO_TXT = open(MEMO, encoding='utf-8').read()
assert 'u* 几何图谱综论（3189→3253 链收束' in MEMO_TXT
assert MEMO_TXT.count('## u* 几何图谱综论') == 1
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
assert sha8(open(os.path.join(SRC_DIR, r'phase3254\g5h17_registry_landing'
                              r'\atlas_registry_v1_34.json'), 'rb').read()) == '3deabf25'
assert sha8(open(os.path.join(SRC_DIR, r'phase3254\g5h17_registry_landing'
                              r'\gap_ledger_v1_12.json'), 'rb').read()) == 'aa8cff21'
REPORT.append('1 assertions: OK (verdict=%s res=%s; endpoint v1.12 appendix 6; '
              'UI 17/17; synthesis section once; sealed v1.34/v1.12 untouched)'
              % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = '8fb0bce0'
already = [e for e in led['measurements'] if e.get('phase') == 3255]
if not already:
    assert len(led['measurements']) == 406 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3255,
        "name": "g5h18_client_synthesis",
        "line": "G",
        "date": "2026-10-20",
        "model": "engineering + doc (zero GPU): /atlas/gaps appendix field + "
                 "GapLedgerPanel mounted in AtlasLens (live-verified 17/17) "
                 "+ u* geometry trilogy synthesis into MEMO",
        "verdict": ("client_synthesis_registered|HEADLINE - u* geometry "
                    "trilogy synthesized as TWO-TRACK GEOMETRY: track I "
                    "(embedding rank-1 direction, dissipates before "
                    "mid-network, never reaches readout) and track II "
                    "(carrier axis, forms mid-network, refocuses deep, "
                    "dominates the readout collapse panel); lexical side "
                    "unreachable for BOTH tracks (3251) => the 3246 "
                    "INFORMATION_GAP cannot be closed from the vocabulary "
                    "side; XC_p5_b8 -0.449 (74% repair) is the "
                    "information-theoretic in-scope endpoint"),
        "evidence_level": ("engineering registered (preregistered MEMO 3254 "
                           "tail (a)+(b); live endpoint check: gaps v1.12 "
                           "auto-discovered with 12-generation chain, "
                           "appendix APPX-1..6, APPX-6 anchor p3250/9d0f1825 "
                           "verbatim; live playwright UI 17/17 PASS zero "
                           "console errors; sealed atlas v1.34 3deabf25 / "
                           "gap v1.12 aa8cff21 byte-untouched)"),
        "model_scope": "n/a (endpoint + frontend + MEMO doc)",
        "prereg_id": "G5-H18 (MEMO 3254 tail)",
        "superseded_by": None,
        "detail": (
            "Registry v1.34 client-side upgrade + u* trilogy synthesis. "
            "Engineering: (1) server /atlas/gaps response adds appendix "
            "field (v1.12 auto-discovered, chain 1.12->1.1 = 12 "
            "generations); (2) frontend GapLedgerPanel.jsx (protocol-key "
            "driven: gaps status row + APPX-1..6 appendix cards, APPX-6 "
            "outlined as newest) mounted in AtlasLens overview; stale "
            "v1.28 select title retired to the 3254 incumbent v1.34; "
            "(3) live verification: endpoint coroutine assertion + "
            "playwright 17/17 PASS zero console errors (GAP-1..5 statuses "
            "and APPX-1..6 cards rendered verbatim from the endpoint). "
            "Doc arm: MEMO special section 'u* 几何图谱综论' appended -- "
            "two-track geometry conclusion (embedding rank-1 track vs "
            "carrier axis track) + engineering implication (vocabulary "
            "side cannot close INFORMATION_GAP; XC_p5_b8 -0.449 is the "
            "in-scope endpoint; activation-side v_c injection is the only "
            "in-protocol lever, bounded by F20). SMOKE device fix: panel "
            "assertion key correction ('status' protocol key, not a "
            "literal status value). freeze 1d19fc3d, res ac6bd1d9. "
            "Pre-append ledger file sha8 " + pre_file_sha8 + "."),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3255', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #407 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3255 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 407
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=407 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3255:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3255: G5-H18 registry v1.34 客户端化（/atlas/gaps appendix + GAP 账本面板实机 17/17）+ u\\* 几何三部曲综合（零 GPU，工程+文档双臂） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3254 tail（(a) 工程臂 + (b) 文档臂）；freeze **`1d19fc3d`**；res **`ac6bd1d9`**；verdict **`client_synthesis_registered`**。封存 v1.34 `3deabf25` / v1.12 `aa8cff21` 字节不动。')
    A_('')
    A_('### 工程臂 (a)：registry v1.34 客户端化（实机验证）')
    A_('')
    A_('- **端点**：/atlas/gaps 响应增 `appendix` 字段——glob 自动发现已拾取 v1.12（**版本链 12 代** 1.12→1.1），GAP-1..5 状态原样；端点直调断言通过（APPX-6 anchor p3250/`9d0f1825` 逐字）。')
    A_('- **前端**：新组件 **`GapLedgerPanel.jsx`**（协议键驱动：id/title/status/note/anchor/anchor_sha8，零实验字面量；缺口状态行 + APPX-1..6 附录卡，APPX-6 描边为最新）挂入 AtlasLens 全景；registry 版本链 select 陈旧 v1.28 文案退役（改为 3254 现役 v1.34 标注）。')
    A_('- **实机校验**：playwright 实机 **17/17 PASS、零控制台错误**（面板标题/LIVE/v1.12/文件名/supersedes/GAP-1..5 状态/APPX-1..6 卡/APPX-6 锚/sha8/收线链标题/XC_p5_b8/74%/陈旧文案清除）。SMOKE 装置修正 ×1：面板断言键取协议键 `status` 而非字面值。')
    A_('')
    A_('### 文档臂 (b)：u\\* 几何图谱综论（MEMO 专节已附）')
    A_('')
    A_('**双轨道几何定谳**：轨道 I（embedding 秩 1 方向，中层前消散、不达 readout）与轨道 II（载轴：中层成形→深层重聚焦→readout 支配崩塌面板）。**词汇侧两轨皆不可及**（3251）⇒ 3246 INFORMATION_GAP 无法从词表侧补足；**XC_p5_b8 −0.449（修复 74%）即声明体制内信息论收线终点**；协议内唯一杠杆 = 激活侧 v_c 注入族，受 F20（读出几何≠行为控制）钉界。')
    A_('')
    A_('**限界**：零 GPU；封存文件不动；5 appendix GPU 项 + N 线施加维持挂账。')
    A_('')
    A_('**完整性**：ledger #407（pre 文件 sha8 %s）；产物 result.json + smoke 子目录；工程产物 server/ai_rnd_service.py + GapLedgerPanel.jsx + AtlasLens.jsx + rdc_fusion.css；独立磁盘复核见 p3255_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3256（候选）')
    A_('')
    A_('- (a) **收线终点封章臂**（零 GPU）：把「3239–3255 保真+u\\* 全链」写成一页式收线总结（oracle 上界/现役/双轨几何/F20 边界四件套），同步 registry v1.34 升级日志注记（append-only）；')
    A_('- (b) **N 线施加准备臂**：G 线收线包（registry v1.34 + gap v1.12 + 综论）作为跨线施加输入的清单化（文件/sha/依赖拓扑）；')
    A_('- (c) 5 appendix GPU 项维持挂账。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3255 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3255 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3255:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3255 registry v1.34 客户端化 + u* 三部曲综合完成（client_synthesis_registered）：/atlas/gaps 增 appendix（v1.12 自动发现，版本链 12 代）+ GapLedgerPanel 挂入 AtlasLens（playwright 实机 17/17、零控制台错误，APPX-6 卡描边）；陈旧 v1.28 文案退役。'
      '文档臂：MEMO 专节「u* 几何图谱综论」——**双轨道几何**（轨道 I embedding 秩 1 中层前消散不达 readout / 轨道 II 载轴中层成形→重聚焦→支配崩塌面板）；词汇侧两轨皆不可及 ⇒ 3246 INFORMATION_GAP 词表侧不可补，**XC_p5_b8 −0.449 即体制内收线终点**（激活侧 v_c 为唯一杠杆，受 F20 钉界）。'
      'verdict client_synthesis_registered；freeze 1d19fc3d / res ac6bd1d9，ledger #407，封存 v1.34/v1.12 字节不动。'
      '3256 已预注册：收线终点封章臂（一页式总结 + upgrade_log 注记）+ N 线施加准备臂。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3255 registry v1.34 客户端化' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3255 已预注册**：(a) registry v1.34 客户端化（/atlas/gaps + UI 实机）；(b) u* 几何三部曲综合（零 GPU 文档）；(c) GPU 挂账。')
new_block = ('- **3255**：/atlas/gaps 增 appendix + GapLedgerPanel 入 AtlasLens（实机 17/17）；MEMO 综论定谳**双轨道几何**（embedding 秩 1 轨不达 readout / 载轴轨支配崩塌）——**INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 即收线终点**。#407 `1d19fc3d`/`ac6bd1d9`。'
             '- **3256 已预注册**：(a) 收线终点封章臂（一页式总结 + upgrade_log 注记，append-only）；(b) N 线施加准备臂（收线包清单化）；(c) GPU 挂账。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3255 收官（客户端化' in mem, 'MEMORY tail block not found'
if len(mem) > 3000:
    o = ('- **3249**：degree 2→5 单调（−0.467）；combo 在 p3 基失效。#401。')
    n = ('- **3249**：deg 5 单调（−0.467）；combo p3 基失效。#401。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3252 收官（u* 层轨迹 + APPX-6，双臂 registered）**：**embedding 层 top1=1.0000 ×3**——崩塌方向 embedding 即秩 1；U 形轨迹（中层 0.14–0.21 低谷→readout 0.29–0.31 回升）；近邻稳定 0.966/0.971/0.926。九位级漂移全零。APPX-6 定稿待入册；APPX-1..5 挂账。ledger #404 / `fa13399b`/`a4435eb3`。')
    n = ('- **3252**：**embedding 层 top1=1.0000 ×3**（崩塌方向 embedding 即秩 1）；U 形（中层低谷→readout 回升 0.29–0.31）；近邻稳定 0.966/0.971/0.926。APPX-6 定稿待入册。#404 `fa13399b`/`a4435eb3`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3253 收官（u* 子空间机制臂 + append 触发点评估，双臂 registered）**：**中层低谷 = 同一载轴衰减，非替换**——σ₁(mid,out)=0.628/0.746/0.335 ×3（mid-emb 的 3–6 倍、ovr 30–40×）；embedding 秩 1 方向与被输送方向解离（σ₁≤0.20）。十二位级漂移全零（含 drift_3252 跨 phase 锚）。append 触发点 MET，**决定 3254 落册**。#405 `8b10eba4`/`e19e2844`。')
    n = ('- **3253**：**中层低谷 = 同一载轴衰减，非替换**——σ₁(mid,out)=0.628/0.746/0.335 ×3（mid-emb 的 3–6 倍）；embedding 秩 1 方向与被输送方向解离（σ₁≤0.20）。drift_3252 跨 phase 锚逐位。append 触发点 MET→3254 落册。#405 `8b10eba4`/`e19e2844`。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3250**：deg 5–6 价值停点；**XC_p5_b8 = 新现役**（−0.449，r2 0.241/KL 0.987）；混合核≡poly3。工程臂 fidelity v2+UI 19/19。#402 `c3aa103e`/`9d0f1825`。')
    n = ('- **3250**：deg 5–6 停点；**XC_p5_b8 现役**（−0.449，r2 0.241/KL 0.987）；混合核≡poly3；fidelity v2+UI 19/19。#402。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3251**：词级对齐小、随规模衰减（0.144→0.033→0.025）、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表解耦。#403 `48169586`/`0601eb1b`。')
    n = ('- **3251**：词级对齐小、随规模衰减、非 OOV 主导 ×3——u* 类结构由激活侧 v_c 承载，与词表解耦。#403。')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: replaced tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
