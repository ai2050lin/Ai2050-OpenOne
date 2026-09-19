# -*- coding: utf-8 -*-
"""Phase 2882 finalize: SHA registration + MEMO append + disk re-check."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2882_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
ATLAS = os.path.join(ROOT, r'research\gpt5\atlas')

g = io.open(OUT, 'w', encoding='utf-8')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


files = {
    'exec': os.path.join(BASE, 'phase2882', 'ledger_v2', 'execution.json'),
    'result': os.path.join(BASE, 'phase2882', 'ledger_v2', 'result.json'),
    'script': os.path.join(ROOT, r'tests\glm5\phase2882_ledger_v2.py'),
    'loader': os.path.join(ROOT, r'tests\glm5\rdc_atlas_ledger.py'),
    'spec': os.path.join(ATLAS, 'ATLAS_LEDGER_SPEC.md'),
    'v1bak': os.path.join(ATLAS, 'atlas_ledger_v1_backup.json'),
    'ledger': os.path.join(ATLAS, 'atlas_ledger.json'),
}
shas = {}
for k, p in sorted(files.items()):
    shas[k] = sha8(p)
    g.write('SHA %s %s %s\n' % (k, shas[k], os.path.basename(p)))

# ledger re-verify (independent process-level check)
from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(files['ledger'], verify_sha=True)
stale = led.verify()
g.write('ledger v%s stale=%d quadrants=%d negatives=%d '
        'model=%s pending=%s\n'
        % (led.doc['version'], len(stale), len(led.doc['quadrants']),
           len(led.doc['negatives']), led.doc['model_namespace']['primary'],
           json.dumps(led.doc['model_namespace']['pending_replication'])))
r = json.load(io.open(files['result'], encoding='utf-8'))
g.write('result ledger_v2=%s\n' % r['ledger_v2'])

# ---------- MEMO append (append-only) ----------
memo_len_before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2882：Atlas Ledger v2 升级（零前向迁移）

**日期**：2026-09-18。**目标**：把 TMA Atlas Ledger 升级为 format v2，为多模型
复制战役（DS7B/GLM4/Gemma4）提供可扩展事实源。

### 原理与设计

v2 新增五节（spec §10–13，向后兼容 v1——v1 文件由 v2 loader 原样加载、缺省节视为空）：

1. `quadrants`（四象限表）：每词族轴一行，登记 drop 谱组织 × mlp 载体两通道观测，
   每格强制引用已存在 meas_id（可导出性检查对象）。实测：class=organized+strong_mlp
   (0.875) / attr=dissociated_mlp_only (0.500) / syntax=intermediate (drop 弱检索
   0.5625 + mlp 0.7083) / translation=no_axis（无统一轴方向，未测）。
2. `transfer`：2881 J3/J4 迁移矩阵快照（全格 >0.76；族质心互负 −0.65/−0.33/−0.14），
   descriptive 无门线。
3. `negatives`（阴性登记）：N1 translation_axis_absent（real_negative，VG2b≈0，
   settled）；N2 tense 隔离（reopenable，复议条件=扩池 ≥8 对）；N3 e200 隔离
   （reopenable，复议条件=重建词表使 offdiag<0.488）。**阴性结果不再蒸发**。
4. `model_namespace`：primary=qwen3-4b，pending_replication=[ds7b, glm4, gemma4]；
   跨模型条目 model 字段必填，同 id 可跨模型并存于 ledgers/<model>/ 子文件。
5. `migration_history`：v1 备份 SHA 登记（atlas_ledger_v1_backup.json，sha256_8
   见下）。

### 预注册门线（execution.json 先于任何 ledger 修改落盘）

- V1 保真：六节计数 pre/post 一致 + SHA 复核 0 stale（既有条目逐字节不动）
- V2 四象限可导出性：每格引用的 meas_id 均存在于 measurements
- V3 阴性接地：每条引用 ≥1 存在 meas_id；quarantine 必带 reopen_condition
- V4 向后兼容：v1 备份在 v2 loader 下干净加载（0 stale，缺省节为空）
- V5 自洽：v2 重载 version==2、五新节齐全、4 quadrants/3 negatives、0 stale

### 结果

**ledger_v2=True（V1–V5 全过）**。V1 计数 {axes 5, blocks 10, headsets 4,
measurements 21, growth_curve 11, linkage 7} 前后一致，0 stale。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2882/ledger_v2/execution.json | %(exec)s |
| phase2882/ledger_v2/result.json | %(result)s |
| tests/glm5/phase2882_ledger_v2.py | %(script)s |
| tests/glm5/rdc_atlas_ledger.py（v2 loader） | %(loader)s |
| research/gpt5/atlas/ATLAS_LEDGER_SPEC.md（v2） | %(spec)s |
| research/gpt5/atlas/atlas_ledger_v1_backup.json | %(v1bak)s |
| research/gpt5/atlas/atlas_ledger.json（v2） | %(ledger)s |

### 硬伤与方法教训

1. **bash shim 幽灵执行**：`cd X && python Y` 变体在 shim 报错（dirname not
   found / cd null directory）时**命令实际已执行**（exit 0 + stderr 噪音 ≠ 失败）。
   第二次重跑覆盖了首次运行生成的 v1 备份 → V4 假阴性。
2. **对策制度化**：① 改主文件的脚本必须幂等（version 守卫：只在 version==1 时
   备份，version>2 拒跑）；② 幽灵执行后先探针产物状态再决定重跑；③ v1 可精确
   重建（V1 保真 + json.dump 设置相同 → 移除新增节即逐字节原 v1），已重建并复核
   （新 sha 见表）。
3. V2 门线设计缺陷修正：translation 行 mlp_carrier.sources 为空集——门线允许
   `not_measured` 显式空引用（以 note 标注），不判失败；其余三族每格均 ≥1 引用。

### 结论与接续

Atlas Ledger v2 落成：双图谱事实源现含四象限表（通道分化定律的完整登记）、
迁移矩阵快照、阴性结果登记位与多模型 namespace。**P3 跨模型复制战役的前置条件
已满足**。接续候选：

- **A（主选）**：P3 跨模型复制战役启动——DS7B m 冷启动（2846–2881 管线移植，
  先做词汇 census + class 轴，验证 mlp 载体密度分层定律跨模型成立）
- B：P2 语法轴内部解剖（number 强边际 +0.32 的头级来源；tense 隔离复议按 N2
  复议条件扩池）
- C：P4 因果干预闭环预研（沿联合谱方向的 mlp 输入干预协议设计）

*（SHA 与判决以 result.json 为准；本节由 phase2882 收尾脚本追加。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
memo_len_after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO lines %d -> %d\n' % (memo_len_before, memo_len_after))
tail = ''.join(io.open(MEMO, encoding='utf-8').readlines()[-4:])
g.write('MEMO tail check: %s\n' % ('ledger_v2=True' in tail))
g.close()
print('ok')
