# -*- coding: utf-8 -*-
"""Phase 3210 closeout: ledger #362 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3210', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3210') else None
already = (bako0 is not None and len(disk0['measurements']) == 362
           and disk0['measurements'][361].get('phase') == 3210)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '46d53545', prev_post
if already:
    raw = open(LP + '.bak_3210', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 361

entry = {
    "phase": 3210,
    "name": "g5g9_standing_items_audit",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only audit arm)",
    "verdict": "g5g9_standing_items_audit|REGISTERED|D1_provenance_MEMO_anchored_2767_2800_2802_2810_2833_ledger_absent_finding|D2_memo_anchors_verbatim|D3_fruit_K4_inherited_first_at_3177_tail|D4_ledger_untouched_n361_46d53545|both_recommend_close",
    "evidence_level": "report_only_audit",
    "model_scope": "n/a",
    "prereg_id": "G5-G9",
    "superseded_by": None,
    "detail": (
        "G5-G9 fruit-class/K4 standing-item audit arm (report-only, zero "
        "GPU; default prereg arm (a)). D1 provenance for phases 2767 "
        "(类别知识图谱 v1, 5 类别 x 16 实体), 2800 (LPF-13 载流子 fruit "
        "画像), 2802 (LPF-15 apple 多义 带符号投票谱), 2810 (短语级档案调制, "
        "P-K4 虚词最小 FALSE), 2833 (Δh 通道分离, P-K2/P-K4 双否证) anchored "
        "verbatim to AGI_GPT5_MEMO headers; SMOKE probe found these LPF-line "
        "phases have NO shared-ledger entries (ledger n=361 covers G-line "
        "measurement records) => ledger absence recorded as explicit "
        "finding, not contradiction (3207 precedent). D2 memo verbatim "
        "anchors (5 类别 × 16 实体 / 带符号投票谱 / P-K4 虚词最小 / "
        "attention-sink/位置通用通道支配) all verbatim. D3 '水果类、K4' first "
        "listed at Phase 3177 tail => inherited placeholder items, no "
        "earlier G-line definition section, no G-line work product pending. "
        "D4 ledger untouched (n=361, 46d53545 before==after). Dispositions: "
        "水果类 => recommend close (LPF-line historical research theme, "
        "MEMO-anchored provenance); K4 => recommend close (P-K4 criterion "
        "FALSE at 2810, double-falsified at 2833, superseded by "
        "Δ_specific 通道分离 line); actual list edit is a user-confirmed "
        "doc edit, NOT a ledger mutation. Verdict "
        "standing_items_audit_registered, D1-D4 0 contradictions. Honest "
        "log: SMOKE run1 crashed (KeyError 'phase' — ledger entries without "
        "phase key, m['phase'] direct index); run2 DIRTY(2) — D1 assumed "
        "ledger entries for LPF phases (absent) + D2 segment truncated at "
        "'### Phase 2833:' subsection ('## Phase' substring match); fixed "
        "to MEMO-anchored D1 + finding + top-level '\\n## ' boundary; run3 "
        "SMOKE CLEAN; FULL clean. design e678dc42, res 3ae23744 / seal "
        "23008571."),
    "chain": {
        "res_sha8": "3ae23744",
        "seal_sha8": "23008571",
        "design_sha8": "e678dc42",
        "memo_src": "155cfebb",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 362
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3210', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3210', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:361]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 362
print('ledger #362 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3210 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3210 \xe2\x80\x94 G5-G9' not in mraw:
    ADD = """

## Phase 3210 — G5-G9 水果类/K4 挂账盘点臂（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 三跑（首跑 KeyError 崩溃、次跑 DIRTY(2)、修后 CLEAN），FULL 一次收官**。零 GPU、report-only（D4 ledger 未动断言）。design `e678dc42`，res `3ae23744` / seal `23008571`，verdict **standing_items_audit_registered**。

### 盘点结论（D1–D4 全过）
- **D1 出处锚定（修正口径）**：五 phase 均为 LPF 线 phase，逐字锚定 `AGI_GPT5_MEMO.md` 头行——2767 类别知识图谱 v1（5 类别 × 16 实体，水果为其一）/ 2800 LPF-13 载流子 fruit 画像 / 2802 LPF-15 apple 多义带符号投票谱 / 2810 短语级档案调制（P-K4 虚词最小 FALSE）/ 2833 Δh 通道分离（P-K2/P-K4 双否证）。**SMOKE 实测共享账本（n=361，G 线测量记录）无这些条目**——账本缺记作为显式 finding 入档，不构成 contradiction（沿 3207 先例）。
- **D2**：四条 verbatim 锚全过（含 `attention-sink/位置通用通道支配`；修复 D2 段边界 `'\n## '` 顶层标题匹配——原 `'## Phase'` 子串匹配被 `### Phase 2833:` 子标题截断）。
- **D3**：「水果类、K4」在 MEMO 首现于 **Phase 3177 尾挂账列表**——继承占位项，无更早 G 线定义段、无 G 线未结工作产物。
- **D4**：ledger 未动（n=361，46d53545 before==after）。

### disposition（两项均建议 close；列表编辑属用户确认的文档编辑，本 phase 零变更）
- **水果类**：LPF 线历史研究主题（2767/2800/2802 MEMO 锚定），建议自 G 线挂账列表关闭；若未来恢复水果类协议工作可转 appendix。
- **K4**：P-K4 判据 2810 FALSE（func 1.400 > content 1.269）、2833 与 P-K2 双否证，被 Δ_specific(cond) 通道分离线取代，建议关闭。

### 诚实日志
SMOKE run1 崩溃（`KeyError: 'phase'`——账本无 phase 键条目直接索引）；run2 DIRTY(2)（D1 预设账本条目而 LPF phase 实际缺记 + D2 子标题截断）；修为 MEMO 锚定 D1 + finding 记录 + 顶层边界后 run3 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3211（两支候选）
- (a) **G5-G10 UI v1.7 迭代**（非账本轮）：客户端加 per-feature 原始锚 values 视图（registry anchors asserts 全量展示）。
- (b) **G5-G11 FTR-04 glm4 弱特异性归因臂**：回读 FTR-04 glm4 特异性证据链，评估升格/挂账口径。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3210' not in draw:
    dadd = ("\n\n### Phase 3210 G5-G9 水果类/K4 挂账盘点收官（report-only，2026-10-11）\n"
            "- SMOKE run1 KeyError('phase') 崩溃、run2 DIRTY(2)（LPF phase 账本缺记 + D2 子标题截断）；修为 MEMO 锚定 D1 + finding + '\\n## ' 边界后 CLEAN；FULL 一次收官。\n"
            "- 盘点：水果类 = LPF 线 2767/2800/2802 历史主题；K4 = P-K4 判据 2810 FALSE + 2833 双否证被通道分离线取代；两者均为 3177 尾继承占位项，disposition 建议 close（用户确认文档编辑待办）。\n"
            "- ledger #362（治理条目，rollback 等价验证）；预注册 3211：默认 (a) UI v1.7 锚 values 视图；(b) FTR-04 glm4 归因臂。\n")
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
- **3211 已预注册 = G5-G10 UI v1.7 锚 values 视图**（非账本轮，默认）：客户端加 per-feature registry anchors asserts 全量展示；候选 (b) G5-G11 FTR-04 glm4 弱特异性归因臂。
- 3210 收官：G5-G9 水果类/K4 挂账盘点 REGISTERED（report-only，res 3ae23744/seal 23008571，design e678dc42），ledger #362；出处 MEMO 锚定（LPF 线 2767/2800/2802/2810/2833，账本缺记为 finding）；两项均建议 close，**用户确认文档编辑待办**（挂账列表删除两项）。
- 3209 收官：G5-G8 UI v1.6 零 script 客户端 REGISTERED，atlas_client_v1.html `bd230988`（73,507 字节，137/137 字段），ledger #361；3208：N 线补录草案 f29da2de（draft_pending_cross_line）；3207：5-appendix 复读 REPORT_ONLY_CLEAN；3206：FTR-22 链复读 + registry v1.28 5995bbe9 + atlas_v1_30 ebc1e122。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。（水果类/K4 待用户确认后自列表移除）
"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
