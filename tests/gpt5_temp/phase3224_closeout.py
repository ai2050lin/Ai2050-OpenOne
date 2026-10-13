# -*- coding: utf-8 -*-
"""Phase 3224 closeout: ledger #376 + MEMO + daily + MEMORY (report-only arm;
governance entry only)."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3224'
      r'\g5g25b_weights_readiness_inventory')
INV_C = '0590e53b'
INV_F = 'c7a367a3'
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3224', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3224') else None
already = (bako0 is not None and len(disk0['measurements']) == 376
           and disk0['measurements'][375].get('phase') == 3224)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '5111392c', prev_post
if already:
    raw = open(LP + '.bak_3224', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 375

entry = {
    "phase": 3224,
    "name": "g5g25b_weights_readiness_inventory",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero-action report-only; metadata json fetches only)",
    "verdict": "weights_readiness_inventory_registered|R1_disk_headroom_ok"
               "|R2_weights_absent_reconfirmed|R3_wu_shard_map_pinned"
               "|R4_manifests_x3|R5_inventory_sealed_" + INV_C
               + "|R6_ledger_untouched_n375_5111392c",
    "evidence_level": "sealed_inventory",
    "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 optional plan C",
    "prereg_id": "G5-G25-B",
    "superseded_by": None,
    "detail": (
        "G5-G25-B weights-acquisition readiness inventory (zero-action "
        "report-only; metadata json fetches only, NO safetensors "
        "downloaded). R1: disk headroom C: 71.7GB free / D: 32.2GB free. "
        "R2: weights-level recheck — no .safetensors anywhere, consistent "
        "with 3223 D3/D4 findings; honest side effect: this arm's own R3 "
        "metadata fetch materialized the 4B snapshot dir (config/index "
        "json, ~KB) from SMOKE run1 onward, R2 criterion therefore set at "
        "weights level. R3: W_U shard map pinned via config+index fetch — "
        "4B tied=True (embed_tokens in model-00001, model.norm in "
        "model-00003) minimal 3.78GB; 14B tied=False (lm_head + model.norm "
        "both in model-00008) minimal 1.78GB; qwen-pair minimal total "
        "5.56GB. R4: API manifests x3 — 4B 7.50GB/3 shards, 14B "
        "27.52GB/8 shards, glm-4-9b-chat 17.51GB/10 shards. R5: "
        "weights_readiness_inventory_v1.json sealed, content sha " + INV_C
        + " (file " + INV_F + ") — plans A minimal 5.56GB (RECOMMENDED for "
        "3225 A1) / B full qwen pair 35.02GB / C +glm4 52.53GB, all disk "
        "verdicts OK; integrity plan: hub LFS sha256 at download + A1 "
        "freeze records per-shard sha8 + CPU matrix math only. R6: ledger "
        "untouched (n=375, 5111392c before==after). Honest log: SMOKE "
        "run1 crash-free but R2 DIRTY x2 (dir-status string criterion "
        "invalidated by own metadata fetch; restructured to weights-level "
        "criterion), run3 CLEAN + FULL CLEAN. res " + RES_F + " / seal "
        + LOG_F + "."),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "inventory_sha8": INV_C,
        "inventory_file_sha8": INV_F,
        "plans": {"A_minimal_gb": 5.56, "B_full_qwen_gb": 35.02,
                  "C_plus_glm4_gb": 52.53},
        "wu_shard_map": {"qwen3-4b": ["model-00001-of-00003.safetensors",
                                      "model-00003-of-00003.safetensors"],
                         "qwen3-14b": ["model-00008-of-00008.safetensors"]},
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 376
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3224', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3224', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:375]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 376
print('ledger #376 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3224 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3224 \xe2\x80\x94 G5-G25-B' not in mraw:
    ADD = """

## Phase 3224 — G5-G25-B 权重获取就绪盘点（report-only，2026-10-11）

**零动作 report-only 臂（仅抓取 config/index 元数据 json，未下载任何 safetensors）**，SMOKE run1/run2 DIRTY（R2 判据被自身元数据抓取失效）→ 重构为权重级判据 → run3 CLEAN + FULL CLEAN，verdict **weights_readiness_inventory_registered**。清单档 `weights_readiness_inventory_v1.json` 内容 sha **`0590e53b`**（文件 `c7a367a3`）。

### 盘点结论（R1–R6，0 矛盾）
- **R1 磁盘余量**：C: 71.7GB / D: 32.2GB——三方案全部 OK。
- **R2 权重级复检**：全盘无 safetensors，与 3223 D3/D4 finding 一致；诚实副作用注记：本臂自身 R3 元数据抓取（~KB json）自 run1 起落地了 4B 快照目录。
- **R3 W_U 分片图钉死**：4B tied=True（embed_tokens→shard1，model.norm→shard3）最小 3.78GB；14B tied=False（lm_head + model.norm **同在 shard8**）最小 1.78GB；**qwen 对最小合计 5.56GB**。
- **R4 API 清单 ×3**：4B 7.50GB/3 分片、14B 27.52GB/8 分片、glm-4-9b-chat 17.51GB/10 分片。
- **R5 下载方案封存**：**A 最小 5.56GB**（W_U+ln_f 分片，3225 A1 臂推荐）/ B 全仓 qwen 对 35.02GB / C +glm4 52.53GB；完整性方案 = hub LFS sha256 下载时校验 + A1 冻结逐分片 sha8 锚 + 仅 CPU 矩阵运算。
- **R6**：ledger 未动（n=375，`5111392c` before==after）。

### 诚实日志
SMOKE run1 无崩溃但 run1/run2 R2 DIRTY——目录状态字符串判据被本臂自身的元数据抓取失效（快照目录自 run1 起被落地）→ 判据重构为权重级（无任何 safetensors 即一致）→ run3 CLEAN。res `""" + RES_F + """` / seal `""" + LOG_F + """`；ledger **#376**（治理条目，rollback 等价验证）。

### 预注册 Phase 3225（两支候选）
- (a) **G5-G25 A1 计算+封存臂**（零 GPU 计算）：**前置 = 权重获取授权确认——方案 A 最小 5.56GB（推荐）/ B 35.02GB / C 52.53GB**，您回复方案字母即授权；随后精确 Δℓ_true/Δℓ_lin/能量矩阵 per feature×fold×model 封存 npz；
- (b) **静默待命**：等待您的方案字母或其他指令。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3224' not in draw:
    dadd = ("\n\n### Phase 3224 G5-G25-B 权重获取就绪盘点收官（report-only，2026-10-11）\n"
            "- 零动作（仅 config/index 元数据 json）；R3 W_U 分片图钉死：4B tied（shard1+shard3）3.78GB、14B lm_head+norm 同在 shard8 1.78GB、qwen 对最小 5.56GB；方案 A 5.56GB / B 35.02GB / C +glm4 52.53GB 全部磁盘 OK。\n"
            "- 诚实日志：SMOKE run1/run2 R2 DIRTY（目录状态判据被自身元数据抓取失效）→ 权重级判据重构 → run3 CLEAN；清单档 0590e53b（c7a367a3）；ledger #376（治理条目，rollback 等价）。\n"
            "- 预注册 3225：(a) A1 计算+封存臂（前置 = 用户回复方案字母授权下载）；(b) 静默待命。\n")
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
- **3225 待授权**：(a) **G5-G25 A1 计算+封存臂**——前置 = 权重获取授权：**方案 A 最小 5.56GB**（4B shard1+shard3 = 3.78GB + 14B shard8 = 1.78GB，W_U+ln_f，推荐）/ B 全仓 qwen 对 35.02GB / C +glm4 52.53GB（C 磁盘余 71.7GB 可容）；**用户回复方案字母即授权**。3226 = A2 三门评估（GA cos≥0.70×3/4 折×2 模型 / GB<0.25 / GC rank-8≥0.8）+ registry 决策（E2_held_out→E2_behavioral，upgrade_log #10）。
- 3224 收官：G5-G25-B 就绪盘点清单档 0590e53b（c7a367a3），res %s/seal %s，ledger #376 post %s；W_U 分片图钉死（4B tied shard1+3 / 14B shard8）。
- 3223 收官：G5-G25 行为带设计档 87e51bcd（1a1c4978），ledger #375 post 5111392c；W_U 本地不可用 finding。
- 3222 收官：G5-G24 轮次小结（res a1de6a81/seal 3b9cb4e6，design 7081779b），ledger #374；总结档 52ba5d3a。
- 3221/3220/3219/3218：UI 冻结+链档 ca1b24ca（#373）；appendix 锚复读零 GPU 不可关再确证（#372）；UI v1.10 def433e0 727/727（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 3217/3216：UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2 207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项；UI 线冻结于 v1.10。

""" % (RES_F, LOG_F, new_post)
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
