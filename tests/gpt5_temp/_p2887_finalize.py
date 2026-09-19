# -*- coding: utf-8 -*-
"""Phase 2887 finalize: SHA + Ledger v2 (language axis, 4th family)
+ MEMO append + disk re-verify."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2887_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
PDIR = os.path.join(BASE, 'phase2887', 'language_axis_mlp')

g = io.open(OUT, 'w', encoding='utf-8')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


files = {
    'exec': os.path.join(PDIR, 'execution.json'),
    'result': os.path.join(PDIR, 'result.json'),
    'npz': os.path.join(PDIR, 'language_axis_mlp.npz'),
    'script': os.path.join(ROOT, r'tests\glm5'
                           r'\phase2887_language_axis_mlp.py'),
}
shas = {k: sha8(p) for k, p in files.items()}
for k in sorted(shas):
    g.write('SHA %s %s\n' % (k, shas[k]))

# ---------- ledger v2 update ----------
from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(verify_sha=True)
doc = led.doc
assert doc['version'] == 2

if not any(a.get('axis_id') == 'language' for a in doc['axes']):
    doc['axes'].append({
        'axis_id': 'language',
        'kind': 'linguistic',
        'words_n': 57,
        'labels_n': 2,
        'direction_source': {
            'path': 'phase2886/hourglass_cka/hourglass_cka.npz',
            'sha256_8': sha8(os.path.join(
                BASE, 'phase2886', 'hourglass_cka',
                'hourglass_cka.npz')),
            'phase': 2886,
            'note': 'lang_dir = unit(mean S_last[en,18] - mean '
                    'S_last[fr,18]); mid-stream sentence states, new '
                    'provenance (not unembed difference)'},
        'vocab_source': {
            'path': 'phase2887/language_axis_mlp/execution.json',
            'sha256_8': shas['exec'], 'phase': 2887,
            'note': '2878 translation pairs canonical (en,L), VG0+VG1 '
                    'verbatim, tid-level dedup (en 22 / L 35)'},
        'status': 'active'})

if not any(b.get('block_id') == 'B3_language' for b in doc['blocks']):
    doc['blocks'].append({
        'block_id': 'B3_language',
        'axis_id': 'language',
        'kind': 'mlp_response',
        'shape': [57, 10],
        'src': {'path': 'phase2887/language_axis_mlp/'
                        'language_axis_mlp.npz',
                'sha256_8': shas['npz'], 'phase': 2887,
                'key': 'B3_lang'},
        'notes': 'g_direct spectrum injected with lang_dir, window '
                 '[26,36); concept-pair control acc 0.0000 = pure '
                 'language-identity carrier'})

if not any(m['meas_id'] == 'M2887_language_axis'
           for m in doc['measurements']):
    doc['measurements'].append({
        'meas_id': 'M2887_language_axis', 'type': 'carrier_test',
        'model': 'qwen3-4b',
        'verdict': 'language_axis_in_mlp=True: E1 acc 0.7719 vs null '
                   'p95 0.6491 (mlp_carries_language_axis); E2 '
                   'same-language margin 0.1943 vs null p95 0.0482 '
                   '(4.0x); E3 concept-pair control acc 0.0000 (below '
                   'null p95 0.0877) = spectrum carries language '
                   'identity with ZERO concept content',
        'source': {'path': 'phase2887/language_axis_mlp/result.json',
                   'sha256_8': shas['result'], 'phase': 2887}})

if not any(r.get('point_id') == 'G_axis4_language'
           for r in doc['growth_curve']):
    doc['growth_curve'].append({
        'point_id': 'G_axis4_language',
        'axis_id': 'language',
        'block': 'B3_language',
        'components': 10,
        'acc': 0.7719,
        'phase': 2887,
        'notes': '4th axis family in the mlp channel; direction '
                 'extracted from mid-stream sentence states (2886 '
                 'li=18), not unembed; carrier curve now class 0.875 '
                 '/ attr 0.500 / syntax 0.708 / language 0.772'})

if not any(l.get('link_id') == 'L10_language_concept_split'
           for l in doc['linkage']):
    doc['linkage'].append({
        'link_id': 'L10_language_concept_split',
        'from': {'axis': 'language', 'model': 'qwen3-4b',
                 'block': 'B3_language'},
        'to': {'axis': 'class', 'model': 'qwen3-4b',
               'block': 'B3_mlp'},
        'evidence': 'same 10-dim mlp channel: language spectrum '
                    'retrieves language (E1 true) but concept pairs '
                    'at chance (E3 acc 0.0000) - language identity and '
                    'concept content are dissociable components of the '
                    'single channel; endpoint static axis absent '
                    '(N1/N4) while mid-stream-derived direction works',
        'phase': 2887,
        'status': 'confirmed'})
led.save()

led2 = AtlasLedger.load(verify_sha=True)
rep = []
stale2 = led2.verify(rep)
g.write('ledger v%s: %d axes / %d blocks / %d measurements / '
        '%d growth / %d linkage / %d negatives / stale=%d\n'
        % (led2.doc['version'], len(led2.doc['axes']),
           len(led2.doc['blocks']), len(led2.doc['measurements']),
           len(led2.doc['growth_curve']), len(led2.doc['linkage']),
           len(led2.doc['negatives']), len(stale2)))
g.write('ledger sha %s\n' % sha8(os.path.join(
    ROOT, r'research\gpt5\atlas\atlas_ledger.json')))

# ---------- MEMO append ----------
before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2887：语言身份轴进入 mlp 图谱——第四轴族成立

**日期**：2026-09-18。**模型**：qwen3-4b。运行 18.6s（57 词 × 3 条件）。

### 原理

2886 否定"中层剥离"后，语言身份方向仍在中层句状态完美可读（探针 1.0）。
本轮检验：从 2886 句状态提取的**中层语言方向**能否像 class/attr/syntax 轴
一样充当 mlp 载体方向（第四轴族）。方向来源为新登记的 provenance：
lang_dir = unit(mean S_last[en,18] − mean S_last[fr,18])（li=18 中层句质
心差，零前向）——**不是 unembed 端差分**（2878/2885 已证端点无轴）。

### 协议要点（预注册冻结于 execution.json）

- 词表：2878 翻译对转录为规范 (en, L) 形（fr20/de22/es30），VG0 同形排除
  （3 对）+ 单 token 过滤（verbatim 2878 逻辑）→ 35 对；**tid 级跨语言
  去重**（en 词仅登记一次）→ 57 词（en 22 / L 35）——消除同词副本在 LOO
  检索中的同语言泄漏
- 测量：2879 协议 verbatim（same/func/null 三条件，窗口 [26,36)，eps=1.0，
  cdir=lang_dir）；标签 en=0 / L=1（fr+de+es 合并）
- v1 重算确定性 = 精确 0.0；kernel-context noise 1.6e-3（描述性）

### 判决：language_axis_in_mlp = True

| 判据 | 观测 | 结果 |
|---|---|---|
| **E1** 语言检索（2 类） | acc **0.7719** vs null p95 0.6491 | **mlp_carries_language_axis** |
| **E2** 同语言边际 | **0.1943** vs null p95 0.0482（**4.0×**） | **language_margin_in_mlp** |
| **E3** 概念对照（描述） | acc **0.0000**（低于 null p95 0.0877） | 谱零概念含量 |
| **E4** 层方向结构（描述） | 窗口层句方向与 lang_dir 余弦 0.456→0.138 递减 | 语言方向随深度衰减 |

### 解读（三个实质发现，重复三遍）

1. **第四轴族成立，载体曲线补全四点**：class 0.875 / attr 0.500 /
   syntax 0.708 / **language 0.772**——四大轴族全部生活在同一个 mlp 响应
   通道内，且语言轴证明**通道方向可以来自残差流中层而非 unembed 端点**。
2. **语言/概念在通道内可分**：E3 概念对照精确 0——沿语言方向的 mlp 响应
   谱承载纯语言身份、零概念内容；与 2881 三族质心互负（互斥子空间）合看，
   单通道 = 多个可分离的身份/内容分量。
3. **端点阴性 + 中层阳性闭环**：2878/2885（端点无轴，N1/N4）+ 2886（中层
   探针 1.0）+ 本轮（中层方向驱动 mlp 载体）——语言身份的机制画像完成：
   **无静态端点轴，有中层方向，且该方向在深层 mlp 直写臂中有因果可读的
   载体响应**。层剖面递增至 li35（0.0653）与其他轴同型（深层主导）。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2887/language_axis_mlp/execution.json | %(exec)s |
| phase2887/language_axis_mlp/result.json | %(result)s |
| phase2887/language_axis_mlp/language_axis_mlp.npz | %(npz)s |
| tests/glm5/phase2887_language_axis_mlp.py | %(script)s |

硬伤：首跑 KeyError（func 词 the 未注册 tid_map）修复重跑（清产物纪律）。

### 接续（2888 候选）

- **A（主选）**：GLM4 冷启动（P3 第三模型，glm4-9b-chat-hf 32 层 → 窗口
  [23,32)；class 轴 mlp 载体 + 若 tokenizer 允许加翻译轴判别）
- B：语言轴跨模型（DS7B 语言方向载体，检验 L10 分离是否模型普适）
- C：content-token CKA 定位 li6→7 阶梯（2886 Gen2）

*（SHA 与判决以 result.json 为准；本节由 phase2887 收尾脚本追加。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO %d -> %d\n' % (before, after))
g.close()
print('ok')
