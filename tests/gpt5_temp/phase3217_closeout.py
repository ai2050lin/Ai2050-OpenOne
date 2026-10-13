# -*- coding: utf-8 -*-
"""Phase 3217 closeout: ledger #369 + MEMO + daily + MEMORY (UI arm;
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
bako0 = json.loads(open(LP + '.bak_3217', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3217') else None
already = (bako0 is not None and len(disk0['measurements']) == 369
           and disk0['measurements'][368].get('phase') == 3217)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '3cf5938b', prev_post
if already:
    raw = open(LP + '.bak_3217', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 368

entry = {
    "phase": 3217,
    "name": "g5g19_ui_v19_chain_ledger_details",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, non-ledger UI arm)",
    "verdict": "ui_v19_chain_ledger_details_registered|C1_zero_script|C2_census_E2_20_E3_3_E1_3|C3_blocks_gaps4_appendix5_draft5_chain8_ledgertail5|C4_datak_698_698_0_0_0_0|C5_ledger_untouched_n368_3cf5938b|client_9c5b48da_142904_bytes",
    "evidence_level": "ui_iteration_governance",
    "model_scope": "n/a",
    "prereg_id": "G5-G19",
    "superseded_by": None,
    "detail": (
        "G5-G19 UI v1.9 client arm (zero GPU, non-ledger; default prereg "
        "arm (a), per MEMO Phase 3216 tail). Base: v1.8 client (phase "
        "3213, ca76ec49, 670 data-k fields) carried over in full. v1.9 "
        "extends anchor details to the version-chain and ledger-tail "
        "blocks: version chain 8 entries now span chain.{v}.note (8 new "
        "fields); ledger tail 5 rows now span led.{idx}.phase/.name/"
        ".pre/.post in addition to verdict (20 new fields) — rows #364-"
        "#368 covering governance entries 3212..3216, pre->post chain "
        "consistent with ledger_sha256_8 3cf5938b on the final row. C1 "
        "zero <script> / no on* handlers / no javascript: URI; C2 "
        "census E2=20 (17 predictive + 3 held_out) / E3=3 / E1=3; C3 "
        "blocks gaps(4)/appendix(5)/draft(5)/chain(8)/ledger-tail(5) "
        "present; C4 data-k 698/698 (0 missing/extra/mismatch/dups); "
        "C5 ledger untouched (n=368, 3cf5938b before==after). Output "
        "atlas_client_v1.html sha8 9c5b48da (142904 bytes). Verdict "
        "ui_v19_chain_ledger_details_registered, C1-C5 0 "
        "contradictions. Honest log: SMOKE run1 crashed pre-gate — "
        "verify_html expects string values, led.{idx}.phase/.name were "
        "passed as int; str()-coerced, run2 SMOKE CLEAN, FULL clean. "
        "design 4313f47d, res 41efe346 / seal 5ae84bf8."),
    "chain": {
        "res_sha8": "41efe346",
        "seal_sha8": "5ae84bf8",
        "design_sha8": "4313f47d",
        "client_sha8": "9c5b48da",
        "base_client": "ca76ec49 (v1.8, phase 3213)",
        "registry": "5995bbe9", "gl16": "0f6423e0",
        "gl15": "4fb3f37d", "draft": "f29da2de",
        "ledger_pre": prev_post,
        "note": "non-ledger arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 369
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3217', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3217', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:368]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 369
print('ledger #369 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3217 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3217 \xe2\x80\x94 G5-G19' not in mraw:
    ADD = """

## Phase 3217 — G5-G19 UI v1.9 版本链/账本尾锚详情（零 GPU，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 两跑（首跑崩溃、修后 CLEAN），FULL 一次收官**。零 GPU、非账本轮（C5 ledger 未动断言）。design `4313f47d`，res `41efe346` / seal `5ae84bf8`，verdict **ui_v19_chain_ledger_details_registered**；客户端 `atlas_client_v1.html` **`9c5b48da`**（142,904 字节）。

### v1.9 相对 v1.8（ca76ec49，670 字段全保留）
- **版本链展开**：8 条版本（v1.28←v1.21）新增 `chain.{v}.note` 注记锚（+8 字段）。
- **账本尾展开**：末 5 行（#364–#368，覆盖 3212–3216 治理条目）新增 `led.{idx}.phase/.name/.pre/.post` 行级锚（+20 字段，v1.8 仅 verdict 有锚）；末行 post 与在盘 `ledger_sha256_8` `3cf5938b` 一致。
- **门检 C1–C5**：零 script / census E2=20(17+3)/E3=3/E1=3 / 块齐全 / data-k **698/698**（0/0/0/0）/ ledger 未动（n=368，`3cf5938b` before==after）。

### 诚实日志
SMOKE run1 门前崩溃：`verify_html` 要求 expected 值为字符串，`led.{idx}.phase/.name` 传了 int → `str()` 强转后 run2 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3218（两支候选）
- (a) **G5-G20 跨线施加就绪盘点**（零 GPU report-only）：为 N 线补录施加 + APPX-5 补丁施加准备一次性施加清单（仅盘点与清单封存，不实际施加）。
- (b) **UI v1.10**：挂账清单 v2（standing_ledger_v2 `207f2fa6`）作为客户端新块渲染（closed/open/pending 三组处置表）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3217' not in draw:
    dadd = ("\n\n### Phase 3217 G5-G19 UI v1.9 收官（非账本轮，2026-10-11）\n"
            "- v1.8 基座 670 字段全保留 + 版本链 8 条 note 锚 + 账本尾 5 行 phase/name/pre/post 锚 = 698/698 data-k 全过；零 script 保持；C5 ledger 未动（n=368, 3cf5938b）。\n"
            "- SMOKE run1 崩溃（verify_html expected 须字符串，phase/name 传 int）→ str() 修后 CLEAN；FULL 收官 ui_v19_chain_ledger_details_registered，客户端 9c5b48da（142,904 字节）。\n"
            "- ledger #369（治理条目，rollback 等价验证）；预注册 3218：默认 (a) 跨线施加就绪盘点；(b) UI v1.10 挂账清单 v2 块。\n")
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
- **3218 已预注册 = (a) G5-G20 跨线施加就绪盘点**（零 GPU report-only）：N 线补录 + APPX-5 补丁一次性施加清单（仅封存，不施加）；候选 (b) UI v1.10 挂账清单 v2 块（standing_ledger_v2 207f2fa6 渲染）。
- 3217 收官：G5-G19 UI v1.9 REGISTERED（res 41efe346/seal 5ae84bf8，design 4313f47d），ledger #369；客户端 9c5b48da 142,904 字节，698/698 data-k（v1.8 670 + 版本链 note ×8 + 账本尾行级 ×20），零 script 保持。
- 3216：挂账总账 v2 终审 REGISTERED（#368），standing_ledger_v2.json 207f2fa6（closed 5 / open 4 / pending 1）——closed=FTR-04@3202、FTR-15@3204、GAP-4 in-books、水果类/K4 recommend close（**列表删除待用户确认**）。
- 3213/3211：UI v1.8 ca76ec49 670/670（#365）；UI v1.7 62ba1bd6（#363）；3210：水果类/K4 盘点（#362）。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5，beb775bd 单条更正 operations ×4）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
