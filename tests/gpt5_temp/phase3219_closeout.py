# -*- coding: utf-8 -*-
"""Phase 3219 closeout: ledger #371 + MEMO + daily + MEMORY (UI arm;
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
bako0 = json.loads(open(LP + '.bak_3219', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3219') else None
already = (bako0 is not None and len(disk0['measurements']) == 371
           and disk0['measurements'][370].get('phase') == 3219)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'd85d6792', prev_post
if already:
    raw = open(LP + '.bak_3219', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 370

entry = {
    "phase": 3219,
    "name": "g5g21_ui_v110_standing_block",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, non-ledger UI arm)",
    "verdict": "ui_v110_standing_block_registered|C1_zero_script|C2_census_E2_20_E3_3_E1_3|C3_blocks_plus_standing_v2_5_4_1_readiness|C4_datak_727_727_0_0_0_0|C5_ledger_untouched_n370_d85d6792|client_def433e0_146547_bytes",
    "evidence_level": "ui_iteration_governance",
    "model_scope": "n/a",
    "prereg_id": "G5-G21",
    "superseded_by": None,
    "detail": (
        "G5-G21 UI v1.10 client arm (zero GPU, non-ledger; default prereg "
        "arm (a), per MEMO Phase 3218 tail). Base: v1.9 client (phase "
        "3217, 9c5b48da, 698 data-k fields) carried over in full. v1.10 "
        "adds the standing-ledger v2 + apply-readiness block (nav anchor "
        "#standing): closed x5 item/closed_at/evidence spans (FTR-04 "
        "closed@3202, FTR-15 closed@3204, GAP-4 in-books, 水果类/K4 "
        "recommend close), open x4 item/why spans, "
        "pending_user_confirmation x1 (MEMO list edit 水果类/K4), "
        "readiness spans N-line ready_append_type + APPX-5 "
        "not_ready_stale_baseline (from checklist d3f8c544), plus meta "
        "v2_sha8 cd105b50 + checklist_sha8 d3f8c544 — 29 new fields, "
        "727 total. C1 zero <script>; C2 census E2=20/E3=3/E1=3; C3 "
        "blocks incl standing markers; C4 data-k 727/727 (0/0/0/0); C5 "
        "ledger untouched (n=370, d85d6792 before==after). Output "
        "atlas_client_v1.html sha8 def433e0 (146547 bytes). Verdict "
        "ui_v110_standing_block_registered, C1-C5 0 contradictions. "
        "Honest log: SMOKE run1 DIRTY(1) — two new meta spans (v2_sha8/"
        "checklist_sha8) missing from the expected dict; added, run2 "
        "CLEAN; FULL clean. design 16d99a7e, res 6bdc1127 / seal "
        "5819a1b3."),
    "chain": {
        "res_sha8": "6bdc1127",
        "seal_sha8": "5819a1b3",
        "design_sha8": "16d99a7e",
        "client_sha8": "def433e0",
        "base_client": "9c5b48da (v1.9, phase 3217)",
        "registry": "5995bbe9", "gl16": "0f6423e0",
        "gl15": "4fb3f37d", "draft": "f29da2de",
        "v2": "cd105b50", "checklist": "d3f8c544",
        "ledger_pre": prev_post,
        "note": "non-ledger arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 371
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3219', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3219', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:370]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 371
print('ledger #371 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3219 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3219 \xe2\x80\x94 G5-G21' not in mraw:
    ADD = """

## Phase 3219 — G5-G21 UI v1.10 挂账总账 v2 + 施加就绪块（零 GPU，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 两跑（首跑 DIRTY(1)、修后 CLEAN），FULL 一次收官**。零 GPU、非账本轮（C5 ledger 未动断言）。design `16d99a7e`，res `6bdc1127` / seal `5819a1b3`，verdict **ui_v110_standing_block_registered**；客户端 `atlas_client_v1.html` **`def433e0`**（146,547 字节）。

### v1.10 相对 v1.9（9c5b48da，698 字段全保留）
- **新块「挂账总账 v2」**（nav 锚 #standing）：closed ×5（FTR-04 closed@3202 / FTR-15 closed@3204 / GAP-4 in-books / 水果类 / K4，item+closed_at+evidence 三锚）+ open ×4（item+why）+ pending_user_confirmation ×1（MEMO 列表编辑水果类/K4）。
- **施加就绪状态**（checklist `d3f8c544` 源）：N 线 = ready_append_type；APPX-5 = **not_ready_stale_baseline**。
- **门检 C1–C5**：零 script / census E2=20(17+3)/E3=3/E1=3 / 块齐全 / data-k **727/727**（0/0/0/0，新增 29 字段 = closed 15 + open 8 + pending 2 + ready 2 + meta 2）/ ledger 未动（n=370，`d85d6792` before==after）。

### 诚实日志
SMOKE run1 DIRTY(1)：两个新 meta 锚（`meta.v2_sha8`/`meta.checklist_sha8`）漏加进 expected 字典 → 补齐后 run2 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3220（两支候选）
- (a) **G5-G22 5-appendix 在盘锚复读**（零 GPU report-only）：对 APPX-1..3（GPU 型）做在盘锚哈希复读，确认「零 GPU 不可关」结论仍成立。
- (b) **UI 冻结 + G 线语义图档案化盘点**：v1.10 起冻结 UI 迭代线，产出全客户端版本链（v1.6→v1.10）与字段演进表（137→611→670→698→727）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3219' not in draw:
    dadd = ("\n\n### Phase 3219 G5-G21 UI v1.10 收官（非账本轮，2026-10-11）\n"
            "- v1.9 基座 698 字段全保留 + 新块挂账总账 v2（closed 5/open 4/pending 1，item/closed_at/evidence 锚）+ 施加就绪（N 线 ready_append_type / APPX-5 not_ready_stale_baseline）= 727/727 data-k 全过；零 script 保持；C5 ledger 未动（n=370, d85d6792）。\n"
            "- SMOKE run1 DIRTY(1)（两个新 meta 锚漏进 expected）→ 补齐后 CLEAN；FULL 收官 ui_v110_standing_block_registered，客户端 def433e0（146,547 字节）。\n"
            "- ledger #371（治理条目，rollback 等价验证）；预注册 3220：默认 (a) 5-appendix 在盘锚复读；(b) UI 冻结+版本链档案化盘点。\n")
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
- **3220 已预注册 = (a) G5-G22 5-appendix 在盘锚复读**（零 GPU report-only，确认零 GPU 不可关仍成立）；候选 (b) UI 冻结 + 客户端版本链档案化盘点（v1.6→v1.10，字段 137→611→670→698→727）。
- 3219 收官：G5-G21 UI v1.10 REGISTERED（res 6bdc1127/seal 5819a1b3，design 16d99a7e），ledger #371；客户端 def433e0 146,547 字节，727/727 data-k（v1.9 698 + standing 块 29），零 script 保持。
- 3218/3217/3216：跨线施加就绪盘点（#370，checklist e402da6e；N 线 ready / APPX-5 not-ready 基线过期）；UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2 207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 3213/3211/3210：UI v1.8 ca76ec49（#365）；UI v1.7 62ba1bd6（#363）；水果类/K4 盘点（#362）。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9（#358）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
