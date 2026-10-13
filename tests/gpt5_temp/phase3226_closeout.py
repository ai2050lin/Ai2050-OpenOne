# -*- coding: utf-8 -*-
"""Phase 3226 closeout: ledger #378 + MEMO + daily + MEMORY."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3226'
      r'\g5g25_a2_gate_decision')
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))
FZ_SHA = '1608c23d'
REG29_F, REG29_C = '85b8dc1f', 'f0b7444a'
GAP17_F, GAP17_C = '4f45fda0', 'c1acabcd'

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3226', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3226') else None
already = (bako0 is not None and len(disk0['measurements']) == 378
           and disk0['measurements'][377].get('phase') == 3226)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '969b6791', prev_post
if already:
    raw = open(LP + '.bak_3226', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 377

entry = {
    "phase": 3226,
    "name": "g5g25_a2_gate_decision",
    "line": "G",
    "date": TODAY,
    "model": "sealed-number arithmetic only (no model, no forward, no GPU)",
    "verdict": "a2_behavioral_upgrade_registered"
               "|GA_4of4_folds_x2_MODELS_min_cosS_1.0000"
               "|GB_BAND_MED_MAX_0.0052_gate_0.25"
               "|GC_SVD8_MIN_0.999904_gate_0.8"
               "|UPGRADE_FTR25_FTR26_E2_held_out->E2_behavioral_upgradelog10"
               "|REGISTRY_v1_29_%s_gap_v1_7_GAP5_%s|LEDGER_UNTOUCHED"
               % (REG29_F, GAP17_F),
    "evidence_level": "gate_decision_a2",
    "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 deferred (weights)",
    "prereg_id": "G5-G25 A2",
    "superseded_by": None,
    "detail": (
        "G5-G25 A2 gate evaluation + registry decision arm (zero GPU, pure "
        "arithmetic on sealed 3225 A1 scalars). Freeze " + FZ_SHA + " before "
        "any gate evaluation; gates verbatim from design 87e51bcd/1a1c4978. "
        "Gate results (alpha 0.5 gate scale; alpha 0.1 sensitivity, all "
        "PASS too): GA cos_S_med >= 0.70 on 4/4 folds x 2 models x 2 "
        "features (min 1.0000); GB band_med max 0.0052 << 0.25; GC rank-8 "
        "energy min 0.999904 >= 0.8. Feature verdicts: FTR-26 (u*) and "
        "FTR-25 (v_hat_held) both BEHAVIORAL_BAND_PASS (GA and GB per "
        "design taxonomy). Upgrade executed: E2_held_out -> E2_behavioral "
        "for FTR-25+FTR-26 via registry v1.29 (upgrade_log #10, anchors +1 "
        "x2, provenance a1 npz appended; supersedes v1.28 " + REG29_F + "/"
        + REG29_C + "); byte-preservation asserted on 24 legacy features "
        "+ legacy top-level fields. Gap ledger v1.7 (" + GAP17_F + "/"
        + GAP17_C + "): GAP-5 appended — nonlinearity localization "
        "(pre-softmax effectively linear; language-encoding behavioral "
        "nonlinearity lives POST-softmax; Big Task A re-confirmed on OOV "
        "panel rows); legacy 4 gaps byte-preserved. Ledger untouched in "
        "arm (n=377 " + prev_post + " before==after). Honest log: SMOKE "
        "run1 design phase int assert; run2 metric-key filter trailing "
        "underscore bug; run3 all_gate key missing 'a' prefix; run3/FULL "
        "CLEAN. res " + RES_F + " / seal " + LOG_F + "."),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "freeze_sha8": FZ_SHA,
        "registry_v129": {"file_sha8": REG29_F, "content_sha8": REG29_C},
        "gap_v17": {"file_sha8": GAP17_F, "content_sha8": GAP17_C},
        "a1_inputs": {"res": "ea1a0585", "npz": "ac5d4d74",
                      "freeze": "98d7fbd1", "design": "87e51bcd/1a1c4978"},
        "key_scalars": {"ga_min_cos_S_med": 1.0, "gb_band_med_max": 0.0052,
                        "gc_svd8_median_min": 0.999904},
        "ledger_pre": prev_post,
        "note": "registry v1.29 + gap v1.7 are NEW append-only files; "
                "v1.28/v1.6 sealed files untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 378
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3226', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3226', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:377]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post \
    and len(d2['measurements']) == 378
print('ledger #378 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3226 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3226 \xe2\x80\x94 G5-G25 A2' not in mraw:
    ADD = """

## Phase 3226 — G5-G25 A2 三门评估 + registry 决策臂（2026-10-11）

**执行模式**：零 GPU 纯封存数字算术——门判据 verbatim 取自设计档 `presoftmax_band_design_v1.json`（内容 `87e51bcd`/文件 `1a1c4978`），输入锁定 3225 封存（res `ea1a0585` / npz `ac5d4d74` / freeze `98d7fbd1`）；冻结 `1608c23d` 先于任何门评估。

### 门评估（verdict **a2_behavioral_upgrade_registered**，0 矛盾）
- **GA（行为留出方向）**：cos_S_med ≥ 0.70 于 **4/4 折 × 2 模型 × 2 特征**（门刻度 α=0.5；α=0.1 敏感性全 PASS），min cos_S = 1.0000；
- **GB（3158 带对齐）**：band_med 中位 max **0.0052** ≪ 0.25（~50× 余量，α=0.1 时 ≤0.0011）；
- **GC（秩充分）**：rank-8 SVD 能量 min **0.999904** ≥ 0.8；
- **特征级 verdict（设计分类学：PASS ⇔ GA∧GB）**：FTR-26（u*）与 FTR-25（v̂_held）双双 **BEHAVIORAL_BAND_PASS**。

### registry v1.29 + gap v1.7（append-only，字节保全断言）
- **升级执行**：FTR-25+FTR-26 `E2_held_out` → **`E2_behavioral`**（upgrade_log **#10**，双特征 anchors +1、provenance 增 a1 npz 锚、scope_limits 追加行为带注记）；registry v1.29 文件 `85b8dc1f`/内容 `f0b7444a`，supersedes v1.28（`5995bbe9`）——24 legacy 特征 + legacy 顶层字段**逐字节等**断言通过；v1.28 封存文件未动。
- **GAP-5 入账**（gap v1.7 文件 `4f45fda0`/内容 `c1acabcd`，legacy 4 gap 字节保全）：**非线性定位**——pre-softmax logit 空间 effectively 线性（cos≈1.0 / band ≤0.0052 / rank-8≈1.0），语言编码行为非线性定位于 **softmax 之后**（Big Task A 概率空间失效在 OOV 面板行再确证）；输出分布级行为带闭合仍 open，status `localized_linear_presoftmax`。
- 账本臂内未动（n=377，`969b6791` before==after）。

### 诚实日志
SMOKE 三跑：run1 设计档 phase 类型 int 断言；run2 metrics 键过滤尾随下划线 bug；run3 汇总键漏 `a` 前缀 → CLEAN；FULL CLEAN。res `""" + RES_F + """` / seal `""" + LOG_F + """`；冻结 `1608c23d`；ledger **#378**（治理条目，rollback 等价验证）。

### 预注册 Phase 3227（两支候选）
- (a) **G5-G26 判据二（充分性）E3 干预设计草案**（design-only，零 GPU）：最小干预集设计——patch Δv 方向 → 谱内判断翻转率 + u* 幅度预测谱外崩塌，将 E3 从 3/26 扩容的门判据冻结（补全 3223 完成判据映射的 criterion_2）；
- (b) **非线性压力扩程臂**（零 GPU 计算，需先修订 freeze）：α∈{1,2,4} 探测 RMSNorm/线性化 onset（band 何时偏离 0.25）。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3226' not in draw:
    dadd = ("\n\n### Phase 3226 G5-G25 A2 三门评估+registry 决策收官（2026-10-11）\n"
            "- 零 GPU 纯封存数字算术：门 verbatim 设计档 87e51bcd；输入 3225 封存（ea1a0585/ac5d4d74/98d7fbd1）；冻结 1608c23d 先于门评估。\n"
            "- 门结果：GA 4/4 折×2 模型×2 特征（min cos_S 1.0000）、GB band max 0.0052≪0.25、GC 0.999904≥0.8；FTR-26/FTR-25 双 BEHAVIORAL_BAND_PASS。\n"
            "- 升级执行：FTR-25+FTR-26 E2_held_out→E2_behavioral（upgrade_log #10）；registry v1.29 85b8dc1f/f0b7444a（24 legacy 特征字节保全）；GAP-5 非线性定位入 gap v1.7 4f45fda0/c1acabcd（pre-softmax 线性、非线性在 post-softmax）。\n"
            "- 诚实日志：SMOKE 三跑（phase int / 键过滤下划线 / 汇总键漏 a）→ CLEAN；ledger #378（rollback 等价）。预注册 3227：(a) G5-G26 判据二 E3 干预设计草案；(b) 非线性扩程臂（α∈{1,2,4}，需 freeze 修订）。\n")
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
- **3227 已预注册**：(a) **G5-G26 判据二（充分性）E3 干预设计草案**（design-only）：patch Δv → 谱内翻转率 + u* 幅度预测崩塌，E3 扩容门判据冻结（criterion_2）；(b) 非线性扩程臂（α∈{1,2,4} 探 onset，需 freeze 修订）。
- 3226 收官：A2 三门评估+registry 决策（res %s/seal %s，freeze 1608c23d），ledger #378 post %s；**GA 4/4×2×2 min cos 1.0000 / GB ≤0.0052 / GC 0.999904 ⇒ FTR-25+FTR-26 升 E2_behavioral（upgrade_log #10）**；registry v1.29 85b8dc1f（supersedes 5995bbe9，24 legacy 字节保全）；GAP-5 非线性定位（gap v1.7 4f45fda0：pre-softmax 线性、非线性在 post-softmax，Big Task A 再确证）。
- 3225 收官：A1 计算+封存（npz ac5d4d74，freeze 98d7fbd1，res ea1a0585/seal 4751db60），ledger #377 post 969b6791；cos_full=1.0000 / band 0.0008-0.0052 / 金标位移≈0；方案 A 权重入库（4B shard1+3、14B shard8）。
- 3224/3223：就绪盘点 0590e53b（#376 post 0543371d）；行为带设计档 87e51bcd（#375 post 5111392c）。
- 3222 收官：G5-G24 轮次小结（res a1de6a81/seal 3b9cb4e6），ledger #374；总结档 52ba5d3a。
- 3221/3220/3219/3218：UI 冻结+链档 ca1b24ca（#373）；appendix 锚复读（#372）；UI v1.10 def433e0（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 3217/3216：UI v1.9 9c5b48da（#369）；挂账总账 v2（#368，207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 挂账不变：APPX-5 施加（重基线后）；N 线施加（ready）；5 appendix GPU/跨线项；UI 线冻结于 v1.10。

""" % (RES_F, LOG_F, new_post)
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
