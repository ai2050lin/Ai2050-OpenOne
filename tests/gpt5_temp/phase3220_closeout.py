# -*- coding: utf-8 -*-
"""Phase 3220 closeout: ledger #372 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3220', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3220') else None
already = (bako0 is not None and len(disk0['measurements']) == 372
           and disk0['measurements'][371].get('phase') == 3220)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '5cff05f6', prev_post
if already:
    raw = open(LP + '.bak_3220', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 371

entry = {
    "phase": 3220,
    "name": "g5g22_appendix_anchors_reread",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only anchor re-read)",
    "verdict": "appendix_anchors_reread_registered|A1_p3166_448af595_census_mixed_p3165_511d9b13_cross_model_K_p3155_46a5034d_k2_int_share_0.1519_0.1825_0.1641|A2_APPX1_3_appendix_open_unchanged_gpu_dependent|A3_agate_24c60160_draft_pending_lcorr_sealed_not_applied_unchanged|A4_ledger_untouched_n371_5cff05f6",
    "evidence_level": "report_only_anchor_reread",
    "model_scope": "n/a",
    "prereg_id": "G5-G22",
    "superseded_by": None,
    "detail": (
        "G5-G22 5-appendix on-disk anchor re-read arm (report-only, zero "
        "GPU, NO status change; default prereg arm (a), per MEMO Phase "
        "3219 tail). A1: three GPU-type anchors re-read with file-sha "
        "lock + key scalars — APPX-1 p3166 result.json 448af595 (census "
        "mixed, cls_agree False, not_confounded, device_ok_all x3), "
        "APPX-2 p3165 result.json 511d9b13 (cross_model_K + "
        "k_entity_vs_readout_4b tokenizer-side), APPX-3 p3155 summary/"
        "result_summary.json 46a5034d (k2_int_share_kout 0.15187579769212284/"
        "0.1824533831239199/0.16411745881744746). A2: APPX-1..3 statuses "
        "all appendix_open unchanged — each requires new GPU measurement "
        "(14b eff>=0.5 runs / per-model vocab rebuild / K2-K3 direct "
        "measurement) => zero-GPU non-closable conclusion RE-AFFIRMED. "
        "A3: crossline anchors unchanged — agate a_gate_closure_v1.json "
        "24c60160, N-line draft f29da2de draft_pending_cross_line (5), "
        "lcorr beb775bd SEALED_BUT_NOT_APPLIED (4 ops). A4: ledger "
        "untouched (n=371, 5cff05f6 before==after). Verdict "
        "appendix_anchors_reread_registered, A1-A4 0 contradictions. "
        "Honest log: SMOKE/FULL both clean first pass. design 4288f690, "
        "res 122adaf5 / seal 47f9213e."),
    "chain": {
        "res_sha8": "122adaf5",
        "seal_sha8": "47f9213e",
        "design_sha8": "4288f690",
        "p3166": "448af595", "p3165": "511d9b13", "p3155": "46a5034d",
        "agate": "24c60160", "gl15": "4fb3f37d",
        "draft": "f29da2de", "lcorr": "beb775bd",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 372
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3220', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3220', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:371]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 372
print('ledger #372 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3220 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3220 \xe2\x80\x94 G5-G22' not in mraw:
    ADD = """

## Phase 3220 — G5-G22 5-appendix 在盘锚复读（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE/FULL 各一次全 CLEAN**。零 GPU、report-only、无任何状态变更（A4 ledger 未动断言）。design `4288f690`，res `122adaf5` / seal `47f9213e`，verdict **appendix_anchors_reread_registered**。

### 复读结论（A1–A4 全过，0 矛盾）
- **A1 GPU 型锚三件在盘重读（文件 sha 锁 + 关键标量）**：APPX-1 = p3166 `448af595`（census mixed、cls_agree False、not_confounded、device_ok_all ×3）；APPX-2 = p3165 `511d9b13`（cross_model_K + k_entity_vs_readout_4b，tokenizer 面依赖）；APPX-3 = p3155 `summary/result_summary.json` `46a5034d`（k2_int_share_kout 0.15187579769212284 / 0.1824533831239199 / 0.16411745881744746）。
- **A2 零 GPU 不可关结论再确证**：APPX-1..3 各需新 GPU 测量（14b eff≥0.5 扫 / 逐模型词表重建 / K2-K3 直接测量）⇒ status 全部维持 appendix_open 不变。
- **A3 跨线锚不变**：agate `a_gate_closure_v1.json` `24c60160`；N 线草案 `f29da2de` 仍 draft_pending_cross_line（5 条）；lcorr `beb775bd` 仍 SEALED_BUT_NOT_APPLIED（4 操作）。
- **A4**：ledger 未动（n=371，`5cff05f6` before==after）。

### disposition
5 appendix 全部维持 open（GPU 型 ×3 零 GPU 不可关再确证；跨线型 ×2 待确认施加），无 registry/gap 动作。

### 预注册 Phase 3221（两支候选）
- (a) **UI 冻结 + 客户端版本链档案化盘点**（零 GPU report-only）：v1.6→v1.10 版本链与字段演进表（137→611→670→698→727）封存；UI 迭代线自 v1.10 起冻结。
- (b) **G 线轮次小结**：3210–3220 十一轮治理收官统计（闭环 5 / 开放 4 / 待确认 1 + 施加就绪定级），产出会话级总结块。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3220' not in draw:
    dadd = ("\n\n### Phase 3220 G5-G22 5-appendix 在盘锚复读收官（report-only，2026-10-11）\n"
            "- A1 三件 GPU 型锚文件 sha 锁重读（p3166 448af595 / p3165 511d9b13 / p3155 46a5034d，k2_int_share 0.1519/0.1825/0.1641 逐位）+ A2 零 GPU 不可关再确证 + A3 跨线锚不变 + A4 ledger 未动（n=371, 5cff05f6）。\n"
            "- SMOKE/FULL 一次全 CLEAN；ledger #372（治理条目，rollback 等价验证）；预注册 3221：默认 (a) UI 冻结+版本链档案化盘点；(b) G 线轮次小结。\n")
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
- **3221 已预注册 = (a) UI 冻结 + 客户端版本链档案化盘点**（v1.6→v1.10，字段 137→611→670→698→727，封存演进表；UI 迭代线自 v1.10 冻结）；候选 (b) G 线轮次小结（3210–3220 治理统计）。
- 3220 收官：G5-G22 5-appendix 在盘锚复读 REGISTERED（res 122adaf5/seal 47f9213e，design 4288f690），ledger #372；零 GPU 不可关再确证（GPU 型 ×3 维持 open），跨线锚不变（agate 24c60160 / draft pending / lcorr SEALED_BUT_NOT_APPLIED）。
- 3219/3218：UI v1.10 def433e0 727/727（#371）；跨线施加就绪盘点（#370，checklist e402da6e；N 线 ready / APPX-5 not-ready 基线过期）。
- 3217/3216：UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2 207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 3213/3211/3210：UI v1.8 ca76ec49（#365）；UI v1.7 62ba1bd6（#363）；水果类/K4 盘点（#362）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
