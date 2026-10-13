# -*- coding: utf-8 -*-
"""Phase 3216 closeout: ledger #368 + MEMO + daily + MEMORY (report-only arm;
governance entry only)."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3216', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3216') else None
already = (bako0 is not None and len(disk0['measurements']) == 368
           and disk0['measurements'][367].get('phase') == 3216)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'bb1474e6', prev_post
if already:
    raw = open(LP + '.bak_3216', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 367

entry = {
    "phase": 3216,
    "name": "g5g16_standing_ledger_v2",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only final audit arm)",
    "verdict": "standing_ledger_v2_registered|M1_closure_sweep_3210_3215_x6_governance_entries|M2_appendix_x5_all_open_gpu_crossline|M3_nline_draft_5_pending_plus_lcorr_operations_4_on_disk_not_applied|M4_shuiguolei_K4_verbatim_3177_tail_user_confirm_pending|M5_ledger_untouched_n367_bb1474e6|standing_ledger_v2_sealed_207f2fa6",
    "evidence_level": "report_only_final_audit",
    "model_scope": "n/a",
    "prereg_id": "G5-G16",
    "superseded_by": None,
    "detail": (
        "G5-G16 standing-ledger v2 final audit arm (report-only, zero GPU; "
        "default prereg arm (a), per MEMO Phase 3215 tail). M1 closure "
        "sweep: 3210-3215 six governance ledger entries verdict prefixes "
        "match registry (g5g9/g5g10/g5g12/g5g13/g5g14/g5g15 all "
        "REGISTERED). M2: 5 appendix (gl15 4fb3f37d) item-by-item all "
        "open, GPU or cross-line dependent => remain standing (cannot be "
        "closed zero-GPU); table recorded. M3: N-line backfill draft "
        "f29da2de status draft_pending_cross_line with 5 entries, NOT "
        "applied + APPX-5 ledger_corrections_v1 beb775bd on disk (single "
        "correction record, 4 operations) not applied => remain "
        "standing. M4: 水果类、K4 still verbatim in MEMO 3177 tail "
        "(user-confirmed doc edit pending, 3210 disposition). M5: ledger "
        "untouched (n=367, bb1474e6 before==after). Output sealed: "
        "standing_ledger_v2.json sha8 207f2fa6 with closed=[FTR-04 "
        "glm4 closed@3202, FTR-15 held_out closed@3204, GAP-4 chain "
        "closed on the books, 水果类 recommend close, K4 recommend "
        "close] / open=[APPX-1..5, N-line apply, APPX-5 apply, UI "
        "iteration line] / pending_user_confirmation=[MEMO list edit "
        "水果类/K4]. Verdict standing_ledger_v2_registered, M1-M5 0 "
        "contradictions. Honest log: SMOKE run1 DIRTY(1) — M3 misread "
        "lcorr top-level structure (looked for 'corrections'/'entries' "
        "keys; actual schema is a single correction record with "
        "'operations' x4); fixed counter, run2 SMOKE CLEAN; FULL clean. "
        "design 4a1bda9a, res e46b2e51 / seal 09e6908a."),
    "chain": {
        "res_sha8": "e46b2e51",
        "seal_sha8": "09e6908a",
        "design_sha8": "4a1bda9a",
        "standing_ledger_v2": "207f2fa6",
        "gl15": "4fb3f37d", "gl16": "0f6423e0",
        "draft": "f29da2de", "lcorr": "beb775bd",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 368
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3216', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3216', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:367]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 368
print('ledger #368 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3216 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3216 \xe2\x80\x94 G5-G16' not in mraw:
    ADD = """

## Phase 3216 — G5-G16 挂账总账 v2 终审（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 两跑（首跑 DIRTY(1)、修后 CLEAN），FULL 一次收官**。零 GPU、report-only（M5 ledger 未动断言）。design `4a1bda9a`，res `e46b2e51` / seal `09e6908a`，verdict **standing_ledger_v2_registered**；封存产物 `standing_ledger_v2.json` `207f2fa6`。

### 终审结论（M1–M5 全过，0 矛盾）
- **M1 闭环 sweep**：3210–3215 六条账本治理条目 verdict 前缀与在册判定逐条一致（g5g9/g5g10/g5g12/g5g13/g5g14/g5g15 全 REGISTERED，ledger #362–#367）。
- **M2 appendix ×5**：gl15 `4fb3f37d` 逐项状态全 open（GPU/跨线依赖，零 GPU 不可关），表格在册。
- **M3 跨线项**：N 线草案 `f29da2de`（draft_pending_cross_line，5 条目）未施加 + APPX-5 `ledger_corrections_v1` `beb775bd` 在盘未施加（单条更正记录、operations ×4）。
- **M4 水果类/K4**：仍 verbatim 在 MEMO 3177 尾（用户确认编辑待办，3210 disposition）。
- **M5**：ledger 未动（n=367，bb1474e6 before==after）。

### 挂账清单 v2（standing_ledger_v2.json）
- **closed**：FTR-04 glm4（closed@3202）、FTR-15 held_out（closed@3204）、GAP-4 机制链（closed on the books）、水果类（recommend close）、K4（recommend close）。
- **open**：APPX-1..5（GPU/跨线）、N 线补录施加（跨线确认后）、APPX-5 跨线账本补丁施加、UI 迭代线（v1.6→v1.8 已交付，v1.9 可选）。
- **pending_user_confirmation**：MEMO 挂账列表删除「水果类、K4」（文档编辑需用户显式确认）。

### 诚实日志
SMOKE run1 DIRTY(1)：M3 误读 lcorr 顶层结构（找 `corrections`/`entries` 键；实际 schema 为单条更正记录、`operations` ×4）→ 修计数口径，run2 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3217（两支候选）
- (a) **UI v1.9**：版本链/账本尾块锚详情展开（对齐 v1.8 块视图，零 script 客户端）。
- (b) **G5-G17 跨线施加就绪盘点**（零 GPU report-only）：为 N 线施加 + APPX-5 补丁施加准备一次性施加清单（仅盘点与清单封存，不实际施加）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3216' not in draw:
    dadd = ("\n\n### Phase 3216 G5-G16 挂账总账 v2 终审收官（report-only，2026-10-11）\n"
            "- M1 闭环 sweep 3210-3215 ×6 治理条目一致 + M2 appendix ×5 全 open（GPU/跨线）+ M3 N 线草案/APPX-5 跨线未施加 + M4 水果类/K4 用户确认待办 + M5 ledger 未动；SMOKE run1 DIRTY(1)（lcorr 顶层 schema 误读，实为单条更正记录 operations ×4）→ 修后 CLEAN，FULL 收官 standing_ledger_v2_registered。\n"
            "- 封存 standing_ledger_v2.json 207f2fa6：closed 5 / open 4 / pending_user_confirmation 1（水果类/K4 列表编辑）。\n"
            "- ledger #368（治理条目，rollback 等价验证）；预注册 3217：默认 (a) UI v1.9；(b) 跨线施加就绪盘点。\n")
    open(DP, 'wb').write((draw.decode('utf-8') + dadd).encode('utf-8'))
print('daily', d_pre, '->', sha8_file(DP))

MP2 = r'.workbuddy\memory\MEMORY.md'
b = open(MP2, 'rb').read()
m2_pre = sha8_file(MP2)
t = b.decode('utf-8')
i = t.find('## 4 下一步')
j = t.find('## 5 ')
assert i >= 0 and j > i
new4 = """## 4 下一步
- **3217 已预注册 = (a) UI v1.9**：版本链/账本尾块锚详情展开（对齐 v1.8 块视图，零 script）；候选 (b) G5-G17 跨线施加就绪盘点（N 线 + APPX-5 一次性施加清单，仅封存不施加）。
- 3216 收官：G5-G16 挂账总账 v2 终审 REGISTERED（res e46b2e51/seal 09e6908a，design 4a1bda9a），ledger #368；standing_ledger_v2.json 207f2fa6（closed 5 / open 4 / pending 1）。closed=FTR-04@3202、FTR-15@3204、GAP-4 in-books、水果类/K4 recommend close。
- 3215/3214/3212：GAP-4 闭环确认（#367）；FTR-15 held_out closed@3204（#366）；FTR-04 glm4 closed@3202（#364）——归因/复读类挂账全部闭环。
- 3213/3211/3210：UI v1.8 ca76ec49 670/670（#365）；UI v1.7 62ba1bd6（#363）；水果类/K4 盘点（#362，**列表删除待用户确认**）。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5，beb775bd 单条更正 operations ×4）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
