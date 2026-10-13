# -*- coding: utf-8 -*-
"""Phase 3222 closeout: ledger #374 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3222', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3222') else None
already = (bako0 is not None and len(disk0['measurements']) == 374
           and disk0['measurements'][373].get('phase') == 3222)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'ff363d67', prev_post
if already:
    raw = open(LP + '.bak_3222', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 373

entry = {
    "phase": 3222,
    "name": "g5g24_gline_session_summary",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only session summary)",
    "verdict": "gline_session_summary_registered|G1_twelve_entries_362_373_sequence_verified|G2_census_closure3_ui5_audit4|G3_summary_sealed_52ba5d3a|G4_ledger_untouched_n373_ff363d67",
    "evidence_level": "report_only_session_summary",
    "model_scope": "n/a",
    "prereg_id": "G5-G24",
    "superseded_by": None,
    "detail": (
        "G5-G24 G-line session summary arm (report-only, zero GPU; default "
        "prereg arm (a), per MEMO Phase 3221 tail). G1: twelve ledger "
        "entries #362-#373 (phases 3210-3221) read from disk, "
        "phase/name/verdict-prefix sequence verified — all REGISTERED, "
        "zero contradictions against the on-record table. G2: category "
        "census closure_confirmation x3 (3212 FTR-04 / 3214 FTR-15 / "
        "3215 GAP-4), UI iteration x5 (3211 v1.7 / 3213 v1.8 / 3217 "
        "v1.9 / 3219 v1.10 / 3221 freeze), audit-inventory x4 (3210 "
        "standing items / 3216 standing-ledger v2 / 3218 apply "
        "readiness / 3220 appendix anchors). G3: "
        "gline_session_summary_v1.json sealed, content sha 52ba5d3a "
        "(file 17dfb237), incl closing_state (standing_ledger_v2 "
        "207f2fa6 closed5/open4/pending1; apply readiness N-line ready "
        "/ APPX-5 not-ready stale baseline; UI frozen at v1.10; "
        "appendix zero-GPU non-closable re-affirmed) and "
        "pending_user_confirmation items. G4: ledger untouched (n=373, "
        "ff363d67 before==after). Verdict gline_session_summary_"
        "registered, G1-G4 0 contradictions. Honest log: SMOKE run1 "
        "crashed (TABLE unpack arity) then DIRTY twice — G1 verdict-"
        "prefix table initially assumed name==verdict-prefix for "
        "closure/UI arms; disk is source of truth, prefixes corrected "
        "per on-disk values (#364/#366/#367 closure names, #369/#371 UI "
        "verdict forms), run4 CLEAN. design 7081779b, res a1de6a81 / "
        "seal 3b9cb4e6."),
    "chain": {
        "res_sha8": "a1de6a81",
        "seal_sha8": "3b9cb4e6",
        "design_sha8": "7081779b",
        "summary_doc": "52ba5d3a (file 17dfb237)",
        "window": "phases 3210-3221, ledger #362-#373",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 374
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3222', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3222', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:373]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 374
print('ledger #374 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3222 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3222 \xe2\x80\x94 G5-G24' not in mraw:
    ADD = """

## Phase 3222 — G5-G24 G 线轮次小结（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 三跑 DIRTY（G1 前缀表口径）+ FULL 收官 CLEAN**。零 GPU、report-only（G4 ledger 未动断言）。design `7081779b`，res `a1de6a81` / seal `3b9cb4e6`，verdict **gline_session_summary_registered**；总结档 `gline_session_summary_v1.json` 内容 sha **`52ba5d3a`**（文件 `17dfb237`）。

### 十二轮治理统计（G1–G2，窗口 3210–3221，ledger #362–#373，全部 REGISTERED）
- **闭环确认 ×3**：3212 FTR-04 glm4（closed@3202）/ 3214 FTR-15 held_out（closed@3204）/ 3215 GAP-4 机制链（closed on the books）。
- **UI 迭代 ×5**：3211 v1.7 / 3213 v1.8 / 3217 v1.9 / 3219 v1.10 / 3221 冻结+档案化（字段 137→611→670→698→727，五代零 script）。
- **审计盘点 ×4**：3210 水果类/K4 盘点 / 3216 挂账总账 v2 / 3218 跨线施加就绪 / 3220 appendix 锚复读。

### 收官状态（G3 封存）
- 挂账清单 v2 `207f2fa6`：closed 5 / open 4 / pending_user_confirmation 1。
- 施加就绪：N 线 ready（append 型）；APPX-5 not-ready（基线过期，需重基线）。
- UI 线**冻结于 v1.10**（`def433e0`，链档 `ca1b24ca`）。
- appendix：GPU 型 ×3 零 GPU 不可关再确证（3220）；跨线型 ×2 待确认施加。
- **待用户确认**：MEMO 挂账列表删除「水果类、K4」；跨线施加确认（N 线 / APPX-5 重基线后）。

### 诚实日志
SMOKE run1 崩溃（TABLE 解包元数）→ run2/run3 DIRTY（G1 verdict 前缀表假设 closure/UI 臂 name==前缀，与在盘不符）→ 逐条对盘修正 5 处（#364/#366/#367 closure name 型、#369/#371 UI verdict 型）→ run4 CLEAN、FULL CLEAN。在盘账本为唯一事实源。

### 预注册 Phase 3223（两支候选）
- (a) **静默待命**：G 线治理窗口收官，无预注册新臂；下一轮等待用户指令（水果类/K4 列表编辑确认、跨线施加确认、或新研究方向）。
- (b) **FTR E1→E2 批量资格预审**（零 GPU report-only）：对 E1 特征做资格盘点清单。
- 默认执行 (a)——如用户无新指令，本轮即 G 线治理窗口终点。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3222' not in draw:
    dadd = ("\n\n### Phase 3222 G5-G24 G 线轮次小结收官（report-only，2026-10-11）\n"
            "- G1 十二条 #362-#373 序列逐条验证（全 REGISTERED）+ G2 census（闭环 3 / UI 5 / 审计 4）+ G3 总结档 gline_session_summary_v1 52ba5d3a（17dfb237）封存 + G4 ledger 未动（n=373, ff363d67）。\n"
            "- 诚实日志：SMOKE 三跑 DIRTY（TABLE 解包 + verdict 前缀表 5 处口径）→ 逐条对盘修正后 CLEAN；在盘账本为唯一事实源。\n"
            "- ledger #374（治理条目，rollback 等价验证）；预注册 3223：默认 (a) 静默待命（治理窗口收官，等待用户指令）；(b) FTR E1→E2 资格预审。\n")
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
- **3223 已预注册 = (a) 静默待命**：G 线治理窗口（3210–3222）收官，无新臂；等待用户指令（水果类/K4 列表编辑确认 / 跨线施加确认 / 新研究方向）；候选 (b) FTR E1→E2 资格预审。
- 3222 收官：G5-G24 G 线轮次小结 REGISTERED（res a1de6a81/seal 3b9cb4e6，design 7081779b），ledger #374；总结档 52ba5d3a（12 轮：闭环 3 / UI 5 / 审计 4，全 REGISTERED）。
- 3221/3220/3219/3218：UI 冻结+链档 ca1b24ca（#373）；appendix 锚复读零 GPU 不可关再确证（#372）；UI v1.10 def433e0 727/727（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 3217/3216：UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2 207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项；UI 线冻结于 v1.10。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
