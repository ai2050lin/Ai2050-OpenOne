# -*- coding: utf-8 -*-
# Phase 3267 closeout: five writes (idempotent)
#  1. atlas_ledger.json append measurement #419 (chain: new_post = sha8(dumps(state with ledger_sha256_8=prev_post)))
#  2. AGI_GPT5_MEMO.md append Phase 3267 section (idempotent marker)
#  3. WORKLOG_gpt5.md append line
#  4. WORKLOG_gpt5_mirror.md (deepseek side) append line
#  5. .workbuddy/memory/MEMORY.md update (compressed line)
import json, hashlib, os, datetime, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
WLOG = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md')
WMIR = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

FREEZE = "05235373"
RES_SHA = "5519514c"
ART_SHA = "ce73984d"
GAP_V113 = "814ce808"
REG_V136 = "78d37211"

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def dump(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1)

def write_newline(p, text):
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(text)

# ---------------- write 1: ledger append ----------------
led = json.load(open(LED, encoding='utf-8'))
ms = led['measurements']
already = any(m.get('phase') == '3267' for m in ms)
if already:
    print('write1: ledger #419 already present (idempotent skip)')
else:
    prev_post = led['ledger_sha256_8']
    assert prev_post == '9b6ac3d1' and len(ms) == 423, (prev_post, len(ms))
    rec = {
        'phase': '3267', 'code': 'G5-H30',
        'name': 'appx23_protocol_v2_registered',
        'created': datetime.datetime.now().isoformat(),
        'evidence_level': 'governance',
        'freeze_sha8': FREEZE, 'res_sha8': RES_SHA,
        'artifact': 'tests/glm5/result/rdc_query_construction_20260913/phase3267/'
                    'g5h30_appx23_protocol/appx23_protocol_v2.json (%s)' % ART_SHA,
        'summary': ('APPX-2/3 protocol refinement (report-only, zero GPU, zero model '
                    'load): APPX-2 S_attr/S_syntax cross-model vocab port refined into '
                    'executable preregistered protocol v2 (S1 per-model vocab rebuild '
                    '[F7 axes persisted] / S2 profile gate / S3 p3165 alignment '
                    'verdicts verbatim / S4 synthesis; 5 candidate models tokenizer+'
                    'config+weights on-disk verified); APPX-3 K2/K3 direct measurement '
                    'refined with deathline criteria verbatim-frozen from disk (K2 '
                    'separation cos-drop>50%+behavior-intact; K3 top-50 coverage<30%; '
                    'K1 MEASURED_REJECTED 0.3733=7.5x for contrast); p3155 share band '
                    '15.2-18.3% documented as proxy-not-criterion; arm c: 3 pending '
                    'user confirmations unchanged (continue does not approve), APPX-1 '
                    'context facts recorded (Qwen3-14B + bnb-nf4 both on disk, status '
                    'untouched). GATE 6/6.'),
        'prev_post': prev_post,
    }
    cp = dict(led)
    cp['measurements'] = ms + [rec]
    cp['ledger_sha256_8'] = prev_post  # chain: field = prev post during hashing
    new_post = hashlib.sha256(dump(cp).encode('utf-8')).hexdigest()[:8]
    cp['ledger_sha256_8'] = new_post
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(dump(cp))
    print('write1: ledger #419 appended, post=%s (n=%d)' % (new_post, len(ms) + 1))

# ---------------- write 2: MEMO section ----------------
marker = '## Phase 3267 (G5-H30):'
if marker in open(MEMO, encoding='utf-8').read():
    print('write2: MEMO section already present (idempotent skip)')
else:
    sec = """
## Phase 3267 (G5-H30): APPX-2/3 协议细化（report-only）

- **verdict**: `appx23_protocol_v2_registered`（GATE 6/6，零 GPU、零模型加载、零账本写入）。
- **背景**：3266 预注册——三项 pending_user_confirmation（水果类/K4 删除、lcorr op4 施加、APPX-1 量化口径）不因「继续」批准 → 转入协议细化臂。
- **Arm a（APPX-2 → 协议 v2）**：S_attr/S_syntax 跨模型移植细化为可执行四步——S1 逐模型词表坐标重建（p2874/p2878 构造规则逐字沿用；**F7 教训内化：轴向量必须随产物持久化**）、S2 K_entity_eff_ge05 profile 门、S3 p3165 对齐判决机制逐字（weakly_separated 门、三判决对）、S4 跨模型综合；落册仅经独立 registry 相。5 候选模型（qwen2.5-3b / qwen2-7b / ds-7b / glm4-9b / gemma-3-4b）tokenizer+config+weights 在盘实测全过。
- **Arm b（APPX-3 → 协议 v2）**：K2/K3 直接测量协议细化，死线判据**从磁盘逐字冻结**（deepseek MEMO P35 线三死线表，内容锚非行号）：**K2** = φ_ℓ(c)⊥W_ℓ 分离后响应 cos 降 >50% 而行为不掉 → 弃条件门；**K3** = 3 模式族单坐标 top-50 覆盖率均 <30% → 弃单坐标机制；K1 对照（MEASURED_REJECTED，E_read=0.3733=门 7.5×）。关键口径澄清：p3155 交互份额带 15.2–18.3% 是 **proxy 非 criterion**（份额 <50% 暗示可分离但不检验「行为不掉」半边）。
- **Arm c（待裁决盘点）**：三项 pending 状态不变；APPX-1 语境事实记录（Qwen3-14B 与 Qwen3-14B-bnb-nf4 均在盘，状态不动）。
- 产物 `appx23_protocol_v2.json`（sha8 `ce73984d`，PREREGISTERED_DRAFT_V2，supersedes 3262 草案对应两节）；anchors 6 项实测在盘。freeze `05235373` / res `5519514c` / ledger #419 post `EC_POST_SLOT`（n=424）。
- **预注册 3268**：收线待命臂（report-only 现役快照复核）或 APPX-2/3 执行相（需用户点名启动；协议已备）。
"""
    # fill post value after read-back
    led2 = json.load(open(LED, encoding='utf-8'))
    post = led2['ledger_sha256_8']
    sec = sec.replace('EC_POST_SLOT', post)
    write_newline(MEMO, sec)
    print('write2: MEMO section appended (post=%s)' % post)

# ---------------- writes 3/4: wlogs ----------------
wline = ("- 3267 G5-H30 appx23_protocol_v2_registered（GATE 6/6，report-only 零 GPU 零账本写入）："
         "APPX-2 协议 v2（S1 逐模型词表重建 [F7 轴持久化] / S2 profile 门 / S3 p3165 判决逐字 / S4 综合；"
         "5 候选模型在盘实测）；APPX-3 协议 v2（K2/K3 死线判据磁盘逐字冻结；p3155 份额带 15.2-18.3pct "
         "澄清为 proxy 非 criterion；K1 对照 0.3733=门 7.5 倍）；arm c 三项 pending 不变（APPX-1 语境记录："
         "14B 与 14B-bnb-nf4 均在盘）。产物 ce73984d / freeze 05235373 / res 5519514c / ledger #419 "
         "post EC_POST_SLOT（n=424）。3268 预注册：收线待命或 APPX-2/3 执行相（需点名）。")
led3 = json.load(open(LED, encoding='utf-8'))
wline = wline.replace('EC_POST_SLOT', led3['ledger_sha256_8'])

def wlog_append(p, tag):
    txt = open(p, encoding='utf-8').read()
    if '## 3267 ' in txt or '3267 G5-H30' in txt:
        print('write(%s): already present (idempotent skip)' % tag)
        return
    write_newline(p, '\n## 3267 ' + wline + '\n')
    print('write(%s): appended' % tag)

wlog_append(WLOG, 'wlog')
wlog_append(WMIR, 'wmirror')

# ---------------- write 5: workspace MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '**3267' not in mem:
    o = '- **挂账**：水果类/K4 删除待确认。'
    n = ('- **挂账**：水果类/K4 删除待确认。\n'
         '- **3267**：APPX-2/3 协议 v2 细化（report-only）——K2/K3 死线磁盘逐字冻结；'
         'p3155 带系 proxy 非 criterion；三项 pending 不变。#419。')
    if o in mem:
        mem = mem.replace(o, n)
    else:
        mem = mem.rstrip('\n') + '\n' + n + '\n'
    # keep under 3000: drop the now-covered older tail line if needed
    while len(mem) > 3000:
        lines = mem.split('\n')
        # drop the shortest informational 325x/326x line that is not the new one
        cand = [(len(l), i) for i, l in enumerate(lines)
                if l.startswith('- **32') and '**3267' not in l and len(l) > 40]
        if not cand:
            break
        _ln, i = min(cand)
        lines.pop(i)
        mem = '\n'.join(lines)
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    print('write5: MEMORY updated (len=%d)' % len(mem))
else:
    print('write5: MEMORY already has 3267 (idempotent skip)')

print('CLOSEOUT DONE')
