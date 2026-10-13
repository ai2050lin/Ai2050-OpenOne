# -*- coding: utf-8 -*-
"""Phase 3223 closeout: ledger #375 + MEMO + daily + MEMORY (design-only arm;
governance entry only)."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3223'
      r'\g5g25_presoftmax_band_design')
DESIGN_C = '87e51bcd'
DESIGN_F = '1a1c4978'
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3223', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3223') else None
already = (bako0 is not None and len(disk0['measurements']) == 375
           and disk0['measurements'][374].get('phase') == 3223)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'a1075729', prev_post
if already:
    raw = open(LP + '.bak_3223', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 374

entry = {
    "phase": 3223,
    "name": "g5g25_presoftmax_band_design",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, design-only arm, no inference)",
    "verdict": "presoftmax_band_design_registered|D1_residual_stream_npz_x3_sealed"
               "|D2_oov_panel_npz_x3_sealed|D3_wu_finding_snapshot_incomplete"
               "|D5_design_sealed_" + DESIGN_C + "|D6_ledger_untouched_n374_a1075729",
    "evidence_level": "sealed_design",
    "model_scope": "design targets qwen3-4b + qwen3-14b (2/3); glm4 deferred",
    "prereg_id": "G5-G25",
    "superseded_by": None,
    "detail": (
        "G5-G25 pre-softmax behavioral band upgrade gate — DESIGN-ONLY arm "
        "(zero GPU, no observation, no registry change). Honest prereg "
        "deviation: MEMO Phase 3222 tail preregistered 3223 = (a) standby; "
        "user selected the assistant-offered design arm via follow-up "
        "instruction (user-directed, not drift). D1: phase3154 "
        "g1p4_mfd_multifactor_disentangle collect.npz x3 (full residual "
        "stream H: 4b 208x37x2560 bb99438f / 14b 208x41x5120 6d142b9d / "
        "glm4 208x41x4096 cf2659f1). D2: phase3169 g5a6_oov_panel collect "
        "npz x3 (36eb4ff0 / 97f98575 / 9cc3f8b4). D3/D4 FINDING (not gate): "
        "W_U weights NOT locally usable — 4B-Base snapshot empty, 14B "
        "snapshot config-only, glm4 absent from HF hub; behavioral band "
        "scope = 2/3 models, 3224 execution precondition = one-time "
        "safetensors acquisition (download; zero GPU compute — ground truth "
        "logits = W_U @ ln_f(h_final) is exact CPU matrix math on cached "
        "tensors). D5: presoftmax_band_design_v1.json sealed, content sha "
        + DESIGN_C + " (file " + DESIGN_F + ") — objective (E2_held_out "
        "geometric band -> behavioral band, completion criterion #1), "
        "RMSNorm closed-form Jacobian linearized predictor vs exact "
        "displacement, pre-registered gates GA cos>=0.70 x>=3/4 folds x2 "
        "models / GB band<0.25 / GC rank-8 energy>=0.8, verdict taxonomy "
        "BEHAVIORAL_BAND_PASS/DIRECTIONAL/FAIL (FAIL to F ledger), upgrade "
        "path E2_held_out -> E2_behavioral (registry append, upgrade_log "
        "#10), chain 3224 A1 compute+seal / 3225 A2 gates+registry. D6: "
        "ledger untouched (n=374, a1075729 before==after). Honest log: "
        "SMOKE run1 crashed (os.path.relpath cross-drive C:/D:), run2 "
        "DIRTY(3) (D3 snapshot absence wrongly gated), restructured D3/D4 "
        "to finding-type, run3 CLEAN + FULL CLEAN. design = this entry "
        "chain.design_sha8; res " + RES_F + " / seal " + LOG_F + "."),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "design_sha8": DESIGN_C,
        "design_doc_file_sha8": DESIGN_F,
        "npz_anchors": {"3154": ["bb99438f", "6d142b9d", "cf2659f1"],
                        "3169": ["36eb4ff0", "97f98575", "9cc3f8b4"]},
        "w_u_finding": "not_locally_usable_3224_precondition_download",
        "ledger_pre": prev_post,
        "note": "design-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 375
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3223', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3223', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:374]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 375
print('ledger #375 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3223 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3223 \xe2\x80\x94 G5-G25' not in mraw:
    ADD = """

## Phase 3223 — G5-G25 pre-softmax 行为带升级门·预注册臂草案（design-only，2026-10-11）

**预注册偏离诚实记录**：3222 尾预注册 3223 = (a) 静默待命；用户对助手提出的 G5-G25 设计草案邀约回复「好的，继续」——**设计臂为用户指定**，非漂移。本臂仅冻结设计文档与门判据：零 GPU、无观测、无 registry 动作。

### 数据基础锚定（D1–D6，0 矛盾，verdict **presoftmax_band_design_registered**）
- **D1 全层残差流缓存 ×3 在盘封存**：3154 collect.npz——4b H(208,37,2560) `bb99438f` / 14b H(208,41,5120) `6d142b9d` / glm4 H(208,41,4096) `cf2659f1`；**D2 OOV 面板 ×3**：3169——`36eb4ff0`/`97f98575`/`9cc3f8b4`。
- **D3/D4 关键 finding（W_U 本地不可用）**：4B-Base 快照为空、14B 快照仅 config.json、glm4 不在 HF hub——行为带范围 = 2/3 模型（qwen 对），glm4 延后；**3224 执行前置 = 一次性权重获取（safetensors 下载；零 GPU 计算——真值 logit = W_U·ln_f(h_final) 是缓存张量上的精确 CPU 矩阵运算）**。
- **D5 设计档封存**：`presoftmax_band_design_v1.json` 内容 sha **`87e51bcd`**（文件 `1a1c4978`）——目标（E2_held_out 几何带→行为带，完成判据一）、数学（RMSNorm 闭式 Jacobian 线性化预测器 vs 精确位移）、预注册三门（GA cos≥0.70 ×≥3/4 折 ×2 模型 / GB 带<0.25 / GC rank-8 能量≥0.8）、verdict 分类（BEHAVIORAL_BAND_PASS/DIRECTIONAL/FAIL，FAIL 入 F 账本）、升级路径（E2_held_out→E2_behavioral，registry append，upgrade_log #10）。
- **D6**：ledger 未动（n=374，`a1075729` before==after）。

### 诚实日志
SMOKE run1 崩溃（os.path.relpath 跨盘 C:/D:）→ run2 DIRTY(3)（D3 快照缺失被误设为门）→ D3/D4 重构为 finding 型 → run3 CLEAN + FULL CLEAN。res `""" + RES_F + """` / seal `""" + LOG_F + """`，design `87e51bcd`；ledger **#375**（治理条目，rollback 等价验证）。

### 预注册 Phase 3224（两支候选）
- (a) **G5-G25 A1 计算+封存臂**（零 GPU 计算）：前置 = 权重获取（下载 qwen 对 safetensors，约 10 GB，**需用户确认下载授权**）→ 精确 Δℓ_true/Δℓ_lin/能量矩阵 per feature×fold×model 封存 npz；
- (b) **权重获取就绪盘点**（零动作 report-only）：下载来源/分片清单/磁盘余量/完整性校验方案封存。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3223' not in draw:
    dadd = ("\n\n### Phase 3223 G5-G25 pre-softmax 行为带升级门·预注册臂草案收官（design-only，2026-10-11）\n"
            "- 预注册偏离诚实记录：默认 (a) 静默待命，用户指定设计臂。D1 残差流 npz ×3 + D2 OOV 面板 ×3 封存锚定；D3/D4 finding：W_U 权重本地不可用（4B 快照空/14B 仅 config/glm4 缺）→ 3224 前置 = 权重获取（下载，零 GPU 计算）。\n"
            "- D5 设计档 presoftmax_band_design_v1.json 87e51bcd（1a1c4978）封存：三门 GA cos≥0.70×3/4 折×2 模型 / GB 带<0.25 / GC rank-8 能量≥0.8；升级路径 E2_held_out→E2_behavioral（upgrade_log #10）。\n"
            "- 诚实日志：SMOKE run1 跨盘 relpath 崩溃 → run2 DIRTY(3)（D3 误设门）→ finding 型重构后 CLEAN；ledger #375（治理条目，rollback 等价）。\n"
            "- 预注册 3224：(a) A1 计算+封存臂（前置权重下载，需用户确认）；(b) 权重获取就绪盘点。\n")
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
- **3224 已预注册**：(a) **G5-G25 A1 计算+封存臂**（零 GPU 计算；前置 = qwen 对 safetensors 权重下载约 10 GB，**待用户确认下载授权**）→ 精确 Δℓ_true/Δℓ_lin/能量封存；(b) 权重获取就绪盘点（零动作）。3225 = A2 三门评估 + registry 决策（E2_held_out→E2_behavioral，upgrade_log #10）。
- 3223 收官：G5-G25 pre-softmax 行为带设计档 87e51bcd（1a1c4978）封存（res %s/seal %s，ledger #375 post %s）；关键 finding：W_U 本地不可用（4B 快照空/14B 仅 config/glm4 缺 hub）；npz 锚 3154×3 + 3169×3 在案。
- 3222 收官：G5-G24 轮次小结（res a1de6a81/seal 3b9cb4e6，design 7081779b），ledger #374；总结档 52ba5d3a。
- 3221/3220/3219/3218：UI 冻结+链档 ca1b24ca（#373）；appendix 锚复读零 GPU 不可关再确证（#372）；UI v1.10 def433e0 727/727（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 3217/3216：UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2 207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项；UI 线冻结于 v1.10。

""" % (RES_F, LOG_F, new_post)
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
