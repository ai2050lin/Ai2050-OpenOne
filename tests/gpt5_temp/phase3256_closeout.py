# -*- coding: utf-8 -*-
"""Phase 3256 closeout: five writes (result assertions / ledger #408 / MEMO /
dual wlog / MEMORY.md). Idempotent per write; ledger append chain-checked."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3256\g5h19_closure_seal\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'research\gpt5\docs\WORKLOG_gpt5.md')
WLOG_U = os.path.join(ROOT, r'research\deepseek\docs\WORKLOG_gpt5_mirror.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = time.strftime('%Y-%m-%d %H:%M')
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ---------- 1. result assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'closure_sealed_registered' and R['mode'] == 'formal'
assert R['design_sha8'] == '5061e1e8' and R['res_sha8'] == '01393c60'
assert all(R['gates'].values())
assert R['registry_landing']['to'] == '1.35'
assert R['registry_landing']['file_sha8'] == 'e8335331'
assert R['registry_landing']['revert_bitwise'] is True
assert R['facts']['repair_vs_zero'] == 0.7436
SRC = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
assert sha8(open(os.path.join(SRC, r'phase3254\g5h17_registry_landing'
                              r'\atlas_registry_v1_34.json'), 'rb').read()) == '3deabf25'
assert sha8(open(os.path.join(SRC, r'phase3254\g5h17_registry_landing'
                              r'\gap_ledger_v1_12.json'), 'rb').read()) == 'aa8cff21'
assert os.path.isfile(os.path.join(SRC, r'phase3256\g5h19_closure_seal'
                                       r'\closure_summary_3239_3255.md'))
assert os.path.isfile(os.path.join(SRC, r'phase3256\g5h19_closure_seal'
                                       r'\nline_handoff_manifest.json'))
REPORT.append('1 assertions: OK (verdict=%s res=%s; v1.35 e8335331 revert bitwise; '
              'repair 74.36%%; sealed v1.34/v1.12 untouched; summary+manifest on disk)'
              % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'b411f7ef'
already = [e for e in led['measurements'] if e.get('phase') == 3256]
if not already:
    assert len(led['measurements']) == 407 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3256,
        "name": "g5h19_closure_seal",
        "line": "G",
        "date": "2026-10-20",
        "model": "zero GPU: one-page closure summary (all numbers measured from "
                 "sealed disk) + registry v1.35 append (upgrade_log +1 seal "
                 "annotation) + N-line handoff manifest",
        "verdict": ("closure_sealed_registered|HEADLINE - the 3239-3255 "
                    "fidelity-closure chain is declared CLOSED in-regime: "
                    "incumbent XC_p5_b8 (3250) SCLS -0.449039 vs G0 -1.751489 "
                    "(repair-vs-zero 74.36%), residual located at "
                    "INFORMATION_GAP by ORC_ls_full oracle +0.329005 (3246); "
                    "u* dual-track geometry (3251/3252/3253) proves word-side "
                    "closure unreachable; F20 bounds the only remaining lever "
                    "(activation-side v_c injection family)"),
        "evidence_level": ("registered (preregistered MEMO 3255 tail (a)+(b); "
                           "pointer diff exact = planned 3 ptrs "
                           "($.version/$.supersedes/$.upgrade_log LEN) + "
                           "revert-check bitwise == v1.34 raw; facts consistency "
                           "gate: G0/ORC/incumbent equal sealed aggregates; "
                           "ledger closure chain 3239-3255 complete; sealed "
                           "v1.34 3deabf25 / v1.12 aa8cff21 byte-untouched)"),
        "model_scope": "n/a (bookkeeping + doc; zero GPU)",
        "prereg_id": "G5-H19 (MEMO 3255 tail)",
        "superseded_by": None,
        "detail": (
            "Closure seal. (a) closure_summary_3239_3255.md: one-page "
            "four-piece summary (G0 baseline / ORC_ls_full legal upper bound / "
            "incumbent XC_p5_b8 / F20 + dual-track boundary), every number "
            "measured from sealed phase3246/3250 result.json at run time. "
            "(b) registry v1.34 -> v1.35 (file sha8 e8335331): ONLY "
            "version/supersedes/upgrade_log+1 changed (closure annotation, "
            "no features/failures/anchors); SMOKE device fixes: ptr "
            "normalization (LEN-level list events, '$.' prefix), 'updated' "
            "key absent in v1.34 chain -> planned set 4->3 ptrs (DESIGN "
            "corrected in SMOKE, formal freeze re-done), 3251 dir name "
            "g5h14_ustar_lexical_attribution. (c) N-line handoff manifest "
            "nline_handoff_manifest.json: sealed artifacts with measured "
            "sha8 (registry v1.35/v1.34, gap v1.12, ledger #408), closure "
            "facts, u* trilogy res sha8, F20/INFORMATION_GAP boundary, GPU "
            "backlog, apply-protocol (N-line appends its own entries, never "
            "rewrites sealed files). GPU backlog: checkpoint training-step "
            "trajectory + APPX-1..5 GPU items. freeze 5061e1e8, res "
            "01393c60. Pre-append ledger file sha8 " + pre_file_sha8 + "."),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3256', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #408 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3256 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 408
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=408 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3256:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3256: G5-H19 收线终点封章（一页式收线总结 + registry v1.35 upgrade_log 封章注记）+ N 线施加准备（收线包清单化，零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3255 tail（(a) 封章臂 + (b) N 线施加准备臂）；freeze **`5061e1e8`**（SMOKE `52324280`）；res **`01393c60`**；verdict **`closure_sealed_registered`**。封存 v1.34 `3deabf25` / v1.12 `aa8cff21` 字节不动。')
    A_('')
    A_('### 封章臂 (a)：3239–3255 收线链 CLOSED in-regime')
    A_('')
    A_('- **四件套**（全部运行时实测自封存盘面，禁凭记忆）：起点 G0 SCLS **−1.751489**（3246 aggregate）；合法上界 ORC_ls_full SCLS **+0.329005**（3246，min(scls,ent) 同值）；现役 **XC_p5_b8**（3250）SCLS **−0.449039** / r2_oov 0.241094 / KL 0.986462，修复率（相对零目标）**74.36%**，剩余 +0.449039 = INFORMATION_GAP；边界 F20（读出几何≠行为控制）+ u\\* 双轨道几何（词汇侧不可达）。')
    A_('- **registry v1.35**（文件 sha8 `e8335331`）：仅 version/supersedes/upgrade_log +1 封章注记；指针 diff 精确（3 ptrs：version/supersedes/upgrade_log LEN）+ **revert 逐位 == v1.34 原始字节**；features/failures/anchors 零改动。SMOKE 装置修正：ptr 归一化（LEN 级 list 事件、`$.` 前缀）；v1.34 链无 `updated` 键 → 计划集 4→3 ptrs（DESIGN 于 SMOKE 修正，正式 freeze 重做）；3251 目录名 `g5h14_ustar_lexical_attribution`。')
    A_('- **一页式总结**：`closure_summary_3239_3255.md`（14 收官 phase 链条逐条）。')
    A_('')
    A_('### N 线施加准备臂 (b)：收线包清单')
    A_('')
    A_('- **`nline_handoff_manifest.json`**：封存产物全 sha8 实测（registry v1.35/v1.34、gap v1.12、ledger #408）、closure facts、u\\* 三部曲 res sha8、F20/INFORMATION_GAP 边界、GPU 挂账（checkpoint 训练步轨迹 + APPX-1..5 GPU 项）、apply-protocol（N 线自 append 条目，封存文件永不改写）。')
    A_('')
    A_('**限界**：零 GPU；封存文件不动；GPU 项维持挂账。')
    A_('')
    A_('**完整性**：ledger #408（pre 文件 sha8 %s）；产物 result.json + design.json + smoke 子目录；独立磁盘复核见 p3256_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3257（候选）')
    A_('')
    A_('- (a) **G 线收官核验臂**（零 GPU）：v1.35 客户端可见性复核（/atlas/registry 版本链自动发现 + UI）+ 账本/registry/gap 三账一致性终审；')
    A_('- (b) **N 线施加预演臂**：按 manifest 在 N 线侧做 read-only 施加预演（不写 G 线封存）；')
    A_('- (c) GPU 挂账维持。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3256 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3256 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3256:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3256 收线终点封章完成（closure_sealed_registered）：3239–3255 收线链 CLOSED in-regime——G0 SCLS −1.751489 / ORC_ls_full +0.329005 / 现役 XC_p5_b8 −0.449039（修复 74.36%，剩余=INFORMATION_GAP）/ F20+双轨几何边界，数字全部运行时实测自封存盘面。'
      'registry v1.35 `e8335331`（仅 upgrade_log +1 封章注记；指针 diff 精确 3 ptrs + revert 逐位 == v1.34 原始字节；features/failures/anchors 零改动）。'
      'N 线收线包 `nline_handoff_manifest.json`（封存产物全 sha8 + u* 三部曲 + apply-protocol）；一页式总结 closure_summary_3239_3255.md。'
      'verdict closure_sealed_registered；freeze 5061e1e8（SMOKE 52324280）/ res 01393c60，ledger #408，封存 v1.34/v1.12 字节不动。'
      '3257 已预注册：G 线收官核验臂（v1.35 客户端可见性 + 三账终审）+ N 线施加预演臂。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3256 收线终点封章' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3256 已预注册**：(a) 收线终点封章臂（一页式总结 + upgrade_log 注记，append-only）；(b) N 线施加准备臂（收线包清单化）；(c) GPU 挂账。')
new_block = ('- **3256 封章**：3239–3255 收线链 CLOSED（G0 −1.7515/ORC +0.3290/XC_p5_b8 −0.4490 修复 74.36%）；registry v1.35 `e8335331`（upgrade_log+1 注记，revert 逐位）；N 线收线包 manifest。#408 `5061e1e8`/`01393c60`。'
             '- **3257 已预注册**：(a) G 线收官核验臂（v1.35 客户端可见性 + 三账终审）；(b) N 线施加预演臂（read-only）；(c) GPU 挂账维持。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3256 封章' in mem, 'MEMORY tail block not found'
if len(mem) > 3000:
    o = ('- **3246**：FLIP=INFORMATION_GAP（LS oracle 转正）；XC_b4 −0.772。#398。')
    n = ('- **3246**：FLIP=INFORMATION_GAP；ORC +0.329；XC_b4 −0.772。#398。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3254 落册（registry_landed）**：gap v1.12 `aa8cff21`（APPX-6 入册）+ registry v1.34 `3deabf25`（anchors ×4+upgrade_log）；REVERT 逐位×2、指针 diff 精确、三账审计 6/6；features/failures 零改动。#406 `8be6ccad`/`48ced87e`。')
    n = ('- **3254 落册**：gap v1.12 `aa8cff21`（APPX-6）+ registry v1.34 `3deabf25`（anchors ×4+log）；REVERT 逐位、审计 6/6；features/failures 零改动。#406。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3253**：**中层低谷 = 同一载轴衰减，非替换**——σ₁(mid,out)=0.628/0.746/0.335 ×3（mid-emb 的 3–6 倍）；embedding 秩 1 方向与被输送方向解离（σ₁≤0.20）。drift_3252 跨 phase 锚逐位。append 触发点 MET→3254 落册。#405 `8b10eba4`/`e19e2844`。')
    n = ('- **3253**：**中层低谷 = 同一载轴衰减，非替换**（σ₁ mid-out 0.628/0.746/0.335，mid-emb 3–6 倍）；embedding 秩 1 方向与被输送方向解离（σ₁≤0.20）。drift_3252 锚逐位。MET→3254 落册。#405。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3252**：**embedding 层 top1=1.0000 ×3**（崩塌方向 embedding 即秩 1）；U 形（中层低谷→readout 回升 0.29–0.31）；近邻稳定 0.966/0.971/0.926。APPX-6 定稿待入册。#404 `fa13399b`/`a4435eb3`。')
    n = ('- **3252**：**embedding 层 top1=1.0000 ×3**（崩塌方向 embedding 即秩 1）；U 形（中层低谷→readout 回升 0.29–0.31）；近邻稳定 0.966/0.971/0.926。#404。')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: replaced tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
