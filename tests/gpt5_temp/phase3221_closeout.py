# -*- coding: utf-8 -*-
"""Phase 3221 closeout: ledger #373 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3221', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3221') else None
already = (bako0 is not None and len(disk0['measurements']) == 373
           and disk0['measurements'][372].get('phase') == 3221)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '87b28799', prev_post
if already:
    raw = open(LP + '.bak_3221', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 372

entry = {
    "phase": 3221,
    "name": "g5g23_ui_freeze_archival",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only archival inventory)",
    "verdict": "ui_version_chain_archival_registered|F1_five_clients_disk_reread_bd230988_137_62ba1bd6_611_ca76ec49_670_9c5b48da_698_def433e0_727|F2_chain_sealed_ca1b24ca|F3_ui_line_frozen_at_v110|F4_ledger_untouched_n372_87b28799",
    "evidence_level": "report_only_archival_inventory",
    "model_scope": "n/a",
    "prereg_id": "G5-G23",
    "superseded_by": None,
    "detail": (
        "G5-G23 UI freeze + client version-chain archival arm (report-only, "
        "zero GPU; default prereg arm (a), per MEMO Phase 3220 tail). F1: "
        "five client generations re-read from disk with client file-sha + "
        "content res_sha8 + field_check.n_expected all matching — v1.6 "
        "(phase 3209, bd230988, 137), v1.7 (3211, 62ba1bd6, 611), v1.8 "
        "(3213, ca76ec49, 670), v1.9 (3217, 9c5b48da, 698), v1.10 (3219, "
        "def433e0, 727); zero-script constraint held across all five. F2: "
        "ui_client_version_chain_v1.json sealed, content sha ca1b24ca "
        "(file 660043ab), with per-generation milestones and field "
        "evolution 137->611->670->698->727. F3: UI iteration line FROZEN "
        "at v1.10 — further UI work requires a new prereg phase. F4: "
        "ledger untouched (n=372, 87b28799 before==after). Verdict "
        "ui_version_chain_archival_registered, F1-F4 0 contradictions. "
        "Honest log: first SMOKE/FULL DIRTY(1) — F1 compared result.json "
        "FILE bytes sha against the content res_sha8 lock for v1.6/v1.7 "
        "(content sha != file sha); corrected to field-value comparison, "
        "rerun CLEAN. design 09f68ccd, res 5941ed71 / seal 880a8bfa."),
    "chain": {
        "res_sha8": "5941ed71",
        "seal_sha8": "880a8bfa",
        "design_sha8": "09f68ccd",
        "chain_doc": "ca1b24ca (file 660043ab)",
        "clients": "bd230988/62ba1bd6/ca76ec49/9c5b48da/def433e0",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched; "
                "UI line frozen at v1.10",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 373
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3221', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3221', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:372]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 373
print('ledger #373 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3221 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3221 \xe2\x80\x94 G5-G23' not in mraw:
    ADD = """

## Phase 3221 — G5-G23 UI 冻结 + 客户端版本链档案化（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE/FULL 首轮 DIRTY(1)（口径误读）、修后 CLEAN 收官**。零 GPU、report-only、无状态变更（F4 ledger 未动断言）。design `09f68ccd`，res `5941ed71` / seal `880a8bfa`，verdict **ui_version_chain_archival_registered**；链档 `ui_client_version_chain_v1.json` 内容 sha **`ca1b24ca`**（文件 `660043ab`）。

### 版本链五代在盘重读（F1 全对上）
| 版本 | Phase | client sha8 | res（内容锁） | data-k 字段 |
|---|---|---|---|---|
| v1.6 | 3209 | `bd230988` | `eac7fe87` | 137 |
| v1.7 | 3211 | `62ba1bd6` | `ee976567` | 611 |
| v1.8 | 3213 | `ca76ec49` | `a398c290` | 670 |
| v1.9 | 3217 | `9c5b48da` | `41efe346` | 698 |
| v1.10 | 3219 | `def433e0` | `6bdc1127` | 727 |

字段演进 **137 → 611 → 670 → 698 → 727**；五代零 script 约束全程保持。

### 冻结声明（F3）
**UI 迭代线自 v1.10 起冻结**（`def433e0` 为冻结版）；后续 UI 工作须新预注册 Phase 方可重启。

### 诚实日志
首轮 F1 误将 result.json **文件字节 sha** 对内容 res_sha8 锁（v1.6/v1.7 两者不同：d081c834/eac7fe87、3a51f895/ee976567）→ 改为字段值比较后 CLEAN。封存数据未动。

### 预注册 Phase 3222（两支候选）
- (a) **G5-G24 G 线轮次小结**（零 GPU report-only）：3210–3221 十二轮治理收官统计（闭环 5 / 开放 4 / 待确认 1 + 施加就绪定级 + UI 冻结），产出会话级总结块封存。
- (b) **FTR-01..26 E1→E2 批量资格预审**：对剩余 E1 特征（FTR-14 留出臂 gate-fail 后的现状）做资格盘点，仅清单不改级。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3221' not in draw:
    dadd = ("\n\n### Phase 3221 G5-G23 UI 冻结+版本链档案化收官（report-only，2026-10-11）\n"
            "- 五代客户端在盘重读全对上（v1.6 bd230988/137 → v1.10 def433e0/727，字段 137→611→670→698→727，零 script 全程保持）；链档 ui_client_version_chain_v1 ca1b24ca（文件 660043ab）封存；UI 迭代线自 v1.10 冻结。\n"
            "- 诚实日志：首轮 F1 混淆文件字节 sha vs 内容 res_sha8（v1.6/v1.7 不同）→ 字段口径修后 CLEAN；FULL 收官 ui_version_chain_archival_registered。\n"
            "- ledger #373（治理条目，rollback 等价验证）；预注册 3222：默认 (a) G 线轮次小结；(b) E1→E2 资格预审。\n")
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
- **3222 已预注册 = (a) G5-G24 G 线轮次小结**（零 GPU report-only，3210–3221 治理统计封存）；候选 (b) FTR E1→E2 批量资格预审（仅清单不改级）。
- 3221 收官：G5-G23 UI 冻结+版本链档案化 REGISTERED（res 5941ed71/seal 880a8bfa，design 09f68ccd），ledger #373；链档 ca1b24ca（137→611→670→698→727，五代零 script）；**UI 迭代线自 v1.10 冻结**。
- 3220/3219/3218：5-appendix 锚复读（#372，零 GPU 不可关再确证）；UI v1.10 def433e0 727/727（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 3217/3216：UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2 207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项；UI 线已冻结。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
