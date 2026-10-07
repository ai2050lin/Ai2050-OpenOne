# -*- coding: utf-8 -*-
"""
Phase 15 / N2h1-alpha-8 : seal amend1 —— 跨模型词表常量勘误（正式运行中被装置门抓到，运行后冻结）
=============================================================================
不改动原 seal 字节（保住审计链），另立 amend 文件记录纠正。
原 seal：tests/deepseek_temp/Phase15/N2h1a8_design_seal.json（sha8 6ea6bdc1）
输出：  tests/deepseek_temp/Phase15/N2h1a8_design_seal_amend1.json

背景（事故与处置）：
  seal 的 `panel.sup_id`（水果=104618 / 动物=101239 / 交通工具=118113 / 家具=104259 /
  金属=100843 / 颜色=102284）是 **qwen 词表的 token id**，被**全局**用于所有臂。
  A0(qwen3-4b) 与 A2(Qwen3-14B) 与之同族 ⇒ 正确；**glm4-9b-chat-hf 是另一套词表**
  ⇒ A1 的 `score_of()` 一直在读**错误的类别 token**。
  装置门 `F2_base_ok`（要求每个实例的受体类分数 sr0 > 0）**当场抓到**：
  `bad = 23/41`、受体类分数均值 +0.065 ≈ 0、`FULL_SWAP = +0.281`（4B 为 +10.75）、
  剂量–响应曲线平坦且低 α 段为负 ⇒ A1 的 Q0_device = FAIL，被自动排除出跨模型合取。

  这是**装置移植缺陷**，不是模型的机制差异 —— 正是「装置门必须独立于主结论」的价值所在。
  修正：`sup_id` 改为**每臂在加载后用该模型自己的 tokenizer 现场解析**，并新增硬断言
  「6/6 类别词各自 tokenize 为 **单 token** 且 `tokenizer.decode(id) == 词`」。
  该断言在任何实验观测之前对 A0/A2 给出与冻结值**逐位相同**的结果（自证修正不改变有效臂口径）。
"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
SEAL = os.path.join(P15T, 'N2h1a8_design_seal.json')
EXEC = os.path.join(P15T, 'execution_phase15.json')
OUT = os.path.join(P15T, 'N2h1a8_design_seal_amend1.json')
MDIR = os.path.join(ROOT, 'models', 'hf')

S = json.load(io.open(SEAL, encoding='utf-8'))
EX = json.load(io.open(EXEC, encoding='utf-8'))
SEAL_SHA = hashlib.sha256(io.open(SEAL, 'rb').read()).hexdigest()
EXEC_SHA = hashlib.sha256(io.open(EXEC, 'rb').read()).hexdigest()
FROZEN_SUP_ID = {k: int(v) for k, v in EX['sup_id'].items()}
SUPS = list(FROZEN_SUP_ID.keys())

# ---- 现场解析：每臂 tokenizer 的类别词 id（唯一依据） ----
from transformers import AutoTokenizer

PER_ARM = {}
for arm, cfg in EX['arms'].items():
    d = os.path.join(MDIR, cfg['dir'])
    tk = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    ids = {}
    ok = True
    for w in SUPS:
        t = tk.encode(w, add_special_tokens=False)
        rt = (len(t) == 1 and tk.decode([t[0]]) == w)
        ids[w] = dict(ids=list(t), single=bool(len(t) == 1), decode=[tk.decode([i]) for i in t],
                      roundtrip=bool(rt), frozen_id=FROZEN_SUP_ID[w],
                      matches_frozen=bool(list(t) == [FROZEN_SUP_ID[w]]))
        ok = ok and rt
    PER_ARM[arm] = dict(model=cfg['model'], dir=cfg['dir'], tokenizer=type(tk).__name__,
                        vocab_size=int(tk.vocab_size), sup_id={w: ids[w]['ids'][0] for w in SUPS},
                        roundtrip_ok=bool(ok), detail=ids)
    print('%-24s %-18s vocab=%-8d roundtrip_ok=%s  matches_frozen=%s'
          % (arm, cfg['model'], tk.vocab_size, ok,
             all(v['matches_frozen'] for v in ids.values())))

MISMATCH = {arm: [w for w in SUPS if not PER_ARM[arm]['detail'][w]['matches_frozen']]
            for arm in PER_ARM}

amend = {
    'phase': 15,
    'kind': 'schema_amend + apparatus fix (no hypothesis change)',
    'name': 'N2h1-alpha-8 amend1：跨模型词表常量勘误（sup_id 逐模型解析）',
    'amend_of_seal_sha256': SEAL_SHA,
    'amend_of_seal_sha8': SEAL_SHA[:8],
    'amend_of_execution_sha256': EXEC_SHA,
    'trigger': ('正式运行期（A0 已完成、A1 进行中）由装置门 F2_base_ok 触发；'
                'A0 的校准结论与本勘误无关且已完成，本轮 A1/A2 数据**作废重跑**。'),
    'defect_root_cause': (
        'seal/exec 的 `sup_id` 是【qwen 词表】的 token id（水果=104618 等），被**全局**用于三臂。'
        '`score_of(v, sup, sid)` 用 `v[SUP_ID[sup]]` 取「类别 token」的 logit；'
        'A0(qwen3-4b) 与 A2(Qwen3-14B) 与 qwen 同族 ⇒ 正确；'
        '**glm4-9b-chat-hf 词表不同**（vocab_size 151329 vs 151643），'
        '`SUP_ID[水果]=104618` 在 GLM4 词表里指向**另一个 token** ⇒ A1 全程读错类别。'
        '=> 新装置坑（#52）：**跨模型移植时，任何「词表相关常量」（类别 token id、实例 token id、'
        '特殊 token id）都必须由该模型自己的 tokenizer 现场解析，不得沿用源模型的硬编码值**；'
        '并须用装置门（此处 F2_base_ok）独立验证「读对了 token」。'
    ),
    'evidence_from_device_gate': {
        'note': 'A1 装置门实测（错 id 下）与 A0 对照，是本次判定的全部依据',
        'A1_F2_base_bad_n': 23,
        'A1_F2_base_bad_list': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁', '桌子', '椅子', '沙发',
                                '地毯', '窗帘', '铁', '铜', '铝', '金', '银', '锌', '铅', '红', '绿',
                                '黄', '黑', '白'],
        'A1_F2_receptor_class_score_mean': 0.065,
        'A1_F2_donor_class_score_mean': -0.127,
        'A1_FULL_SWAP': 0.280634,
        'A0_FULL_SWAP': 10.749740,
        'ratio': 0.0261,
        'interpretation': ('受体类分数均值 ≈ 0 且 23/41 实例为负 —— 若类别 token 读对，'
                           '这些实例的 is-a 关系不可能「不存在」；这是错 token 的必然表现。'),
    },
    'tokenizer_probe': PER_ARM,
    'mismatch_summary': MISMATCH,
    'fix': {
        'rule': ('`sup_id` 改为**每臂在该臂加载后**用该臂 tokenizer 解析：'
                 '`SUP_ID_ARM = {w: tok.encode(w, add_special_tokens=False) for w in SUPS}`，'
                 '并要求每个类别词恰为 **1 个 token**。'),
        'new_assertion_F1b': ('6/6 类别词均为单 token 且 `tok.decode([id]) == 词`；否则该臂 ABORT。'),
        'self_consistency_check': ('对 A0/A2，现场解析结果与 seal 冻结值**逐位相同**'
                                   '⇒ 修正不改变有效臂的任何数值口径，只修复 A1。'),
        'frozen_sup_id_kept_as': 'exec.sup_id 保留为「qwen 族参考值」，并新增 exec.sup_id_per_arm 承载逐臂真值。',
    },
    'what_is_NOT_changed': [
        '预注册 7 条预测 P1–P7 —— 不变',
        '判决表 Q0–Q5 与全部阈值（ARGS_GAP_MIN=3 / NULL_HIGH=0.70 / XH_FAITHFUL_TOL=0.05 等）—— 不变',
        '三臂定义、18 位点 × 14 α 网格、W=3、BP=2000、统计量定义、置换零假设方案 —— 全部不变',
        'inheritance anchors（Phase 12 已发表量）—— 全部不变',
        'honesty 6 条不变（新增第 7 条记录本次勘误）',
    ],
    'why_no_HARKing_concern': (
        '本次修正只把「读哪个 token」从**错误的硬编码 id** 改为**该模型 tokenizer 的直读结果**，'
        '不触及任何假设、判据、阈值或统计量选择；且修正由**装置门**（非主结论）触发，'
        '对 A0/A2 的数值口径零影响（逐位相同），只拯救被错 id 污染的 A1。'
    ),
    'added_honesty_7': (
        '7. A1(glm4-9b) 的首轮数据因 `sup_id` 词表误用（qwen id 用于 GLM4 词表）而**作废并重跑**；'
        '勘误冻结于 amend1。修正后 A0/A2 的类别 token id 与冻结值逐位相同，A1 由该臂 tokenizer 现场解析。'
        '审计链：原 seal 字节未动（sha8 %s），首轮 A1 日志保留在 `_formal_stdout.log` 供对照。'
        % SEAL_SHA[:8]
    ),
    'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(amend, f, ensure_ascii=False, indent=1)
    f.write('\n')

b = io.open(OUT, 'rb').read()
print('WROTE %s' % OUT)
print('  bytes = %d ; sha8 = %s' % (len(b), hashlib.sha256(b).hexdigest()[:8]))
print('  amend_of_seal_sha8 = %s ; amend_of_exec_sha8 = %s' % (SEAL_SHA[:8], EXEC_SHA[:8]))
print('  MISMATCH = %s' % json.dumps(MISMATCH, ensure_ascii=False))
