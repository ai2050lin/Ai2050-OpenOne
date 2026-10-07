# -*- coding: utf-8 -*-
"""修复日志尾部（bash 反引号陷阱导致路径被吞），再以文件化方式追加干净小节。

纪律：先截断到追加前字节 offset（精确回滚），复核 sha256 前缀不变，再追加。
"""
import os, io, hashlib

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-01.md'
OFFSET = 22906          # 本轮追加前的字节数（已由两次 stat 精确得知）
B = chr(92)             # 反斜杠（避免任何 shell 层解析）

b = open(P, 'rb').read()
out = []
out.append('before bytes %d' % len(b))
keep = b[:OFFSET]
out.append('keep bytes %d ; prefix_sha8 %s' % (len(keep), hashlib.sha256(keep).hexdigest()[:8]))
out.append('tail_of_keep %r' % keep[-60:].decode('utf-8', errors='replace'))
with open(P, 'wb') as f:
    f.write(keep)

section = []
section.append('')
section.append('## 登记约定 v2：Phase 产物分目录（21:35）')
section.append('')
section.append(('- **新约定（用户指令）**：每个 Phase 的产物 → `tests{b}deepseek{b}Phase{{N}}{b}`（脚本）与 '
                '`tests{b}deepseek_temp{b}Phase{{N}}{b}`（报告 / seal / 校验 / memo·wlog 节源码）；'
                '非 Phase 轮次 → `_review{b}`、`_infra{b}`。**禁止把产物堆在 `tests{b}deepseek{b}`、'
                '`tests{b}deepseek_temp{b}` 根下**。本节取代 21:10 版「扁平存放」条。').format(b=B))
section.append(('- **执行**：186 项零损失归位（57 脚本 + 126 报告/台账 + 3 本轮证据）；186/186 sha256 '
                '前后一致、源根 0 残留、0 错误；临时草稿 4 个已删。台账 '
                '`tests{b}deepseek_temp{b}_infra{b}reorg_manifest.json`、`reorg_verify.txt`、'
                '`do_reorg_phasedirs.py`（幂等）。').format(b=B))
section.append(('- **归属判据**：证据法（文件名回查备忘录 → 最近前置 `## Phase {N}` 标题）；自动判定人工修正 4 处'
                '（迁移工具/复核目录 → 非 Phase；`check_tie` → Phase 4；`enc_check` + 设计轮 `probe_*` → Phase 1；'
                '`probe_e4` → `_infra`）。').replace('{N}', '{N}'))
section.append('- **记录**：deepseek 备忘录追加 `## 登记约定变更 v2` 节（136,771 → 142,272 B；前 136,771 B 的 '
               'sha8 = 895ca478 逐字节复算一致；BOM/CRLF 保持；hdr@L1567）。')
section.append('- **技能**：`rdc-main-axis-probe`（§0-3 路径、§2 seal 结构）与 `rdc-phase-closeout`（§收尾链前置说明）'
               '已更新为 v2 分目录约定。')
section.append('')
text = '\n'.join(section).replace('\r\n', '\n').replace('\n', '\r\n')
with open(P, 'ab') as f:
    f.write(text.encode('utf-8'))

b2 = open(P, 'rb').read()
out.append('after bytes %d (delta %d)' % (len(b2), len(b2) - len(keep)))
out.append('prefix_intact %s' % (b2[:len(keep)] == keep))
out.append('utf8_ok %s' % bool(b2.decode('utf-8')))
out.append('backtick_count_in_new %d' % text.count(chr(96)))
out.append('has_phase4_bucket %s' % ('Phase4' in text))
out.append('tail_note %s' % text.strip().splitlines()[-1][:80])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra\wlog_fix_verify.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
