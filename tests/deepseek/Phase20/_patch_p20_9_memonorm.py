# -*- coding: utf-8 -*-
"""补丁 9（E-baseline：MEMO 完整性事件登记 + 基线口径加固）。

事实（由 tests/deepseek_temp/_infra/audit_memo_drift_phase19.py 现场测量并冻结）：
  `post-append-phase19` 基线冻结于 2026-10-02 07:39:49（469161 B / sha8 2255b365）；
  随后 MEMO 于 **07:52:07** 被**就地改写** —— P10–P19 共 10 个标题由短形式 `[HH:MM]`
  规范化为完整形式 `[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 +110 B，行数不变、无文本丢失；
  该事件**未被任何 wlog / baseline / history 记录**。
后果：`post-append-phase19` 的 bytes/sha256 锚陈旧、`sections` 键（按「行前 44 字符」生成）
      有 3 项落成改写前的短形式。

处置（**不改写历史、不重跑任何臂**）：
  * `closeout_docs_phase20.py`：`sections` 键口径改为**完整标题行** + 碰撞断言；
    `base` 增 `sections_key_rule` / `drift_events`；`history` 中把 `post-append-phase19` 标 `stale`；
    wlog 增一条完整性事件条目（数字全部从审计件现场渲染）。
  * `gen_memo_phase20.py` §11 增 `E-baseline` 条目。
  * `gen_present_phase20.py` §G 的「同轮勘误」增 `E-baseline`。
  * `disk_verify_phase20.py` G10 增 G10h–G10n（审计自洽 + 新基线口径 + 陈旧标注）。
"""
import io

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20'
n = 0


def rep(path, pairs):
    global n
    s = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        c = s.count(old)
        assert c == 1, '%s :: count=%d :: %r' % (path, c, old[:80])
        s = s.replace(old, new); n += 1
    io.open(path, 'w', encoding='utf-8', newline='\n').write(s)


# ================================================================ 1. gen_memo
rep(BASE + r'\gen_memo_phase20.py', [
    # 1a. 载入审计件
    ("SEAL = json.load(io.open(os.path.join(P20T, 'N2h1a13_design_seal.json'), encoding='utf-8'))\n",
     "SEAL = json.load(io.open(os.path.join(P20T, 'N2h1a13_design_seal.json'), encoding='utf-8'))\n"
     "DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',\n"
     "                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))\n"),
    # 1b. §11 增 E-baseline（字符串拼接，避免字面量百分号）
    ("A('- **[E-probefull]** 探针初版把配对集也缩小（4 对）⇒ `U_ℓ` 秩退化为 2、`FULL_SWAP` 偏离锚。'\n"
     "  '修：PROBE 保留**全量配对与实例**，只缩网格与 BP —— 探针读数才有口径意义。')",
     "A('- **[E-probefull]** 探针初版把配对集也缩小（4 对）⇒ `U_ℓ` 秩退化为 2、`FULL_SWAP` 偏离锚。'\n"
     "  '修：PROBE 保留**全量配对与实例**，只缩网格与 BP —— 探针读数才有口径意义。')\n"
     "A('- **[E-baseline] 记录完整性事件（非本轮引入）**：`post-append-phase19` 基线（**'\n"
     "  + str(DRIFT['prev_baseline']['bytes']) + ' B** / `' + str(DRIFT['prev_baseline']['sha8'])\n"
     "  + '`，冻结于 ' + str(DRIFT['prev_baseline']['frozen_at']) + '）在快照后被**就地改写** —— '\n"
     "  + '改写在 **' + str(DRIFT['memo_mtime']) + '**：P10–P19 共 **'\n"
     "  + str(len(DRIFT['normalized_phases'])) + '** 个标题由短形式 `[HH:MM]` 规范化为完整形式 '\n"
     "  + '`[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+' + str(DRIFT['predicted_delta_bytes'])\n"
     "  + ' B**（与实盘差逐位一致，残差 ' + str(DRIFT['residual_bytes']) + ' B），行数不变（'\n"
     "  + str(DRIFT['memo_lines_at_audit']) + ' 行）、**无文本丢失**；该事件**未被任何 wlog / baseline / '\n"
     "  + 'history 记录**。**后果**：该基线的 bytes/sha256 锚，以及 3 项 `sections` 键（按「行前 44 字符」生成）'\n"
     "  + '均陈旧。**处置**：不改写历史 —— `history` 中把该条目标注 `stale`；本 Phase 的 '\n"
     "  + '`_infra/memo_baseline.json` 以新口径重新冻结（`sections` 键改为**完整标题行**并加碰撞断言，'\n"
     "  + '`drift_events` 登记本事件）。审计件：`tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json`。')"),
])

# ================================================================ 2. closeout_docs
rep(BASE + r'\closeout_docs_phase20.py', [
    # 2a. 载入审计件 + P19 磁盘复核 mtime
    ("PRE = json.load(io.open(os.path.join(P20T, 'memo_baseline_preappend_phase20.json'), encoding='utf-8'))\n",
     "PRE = json.load(io.open(os.path.join(P20T, 'memo_baseline_preappend_phase20.json'), encoding='utf-8'))\n"
     "DRIFT = json.load(io.open(os.path.join(INFRA, 'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))\n"
     "_DV19 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19', 'disk_verify_phase19.txt')\n"
     "DV19_T = (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(_DV19).st_mtime))\n"
     "          if os.path.exists(_DV19) else 'n/a')\n"),
    # 2b. sections 键口径：完整标题行 + 碰撞断言
    ("heads = {}\n"
     "for i, l in enumerate(mt):\n"
     "    if l.startswith('## '):\n"
     "        heads[l[:44]] = i + 1\n"
     "hist = []\n",
     "heads = {}\n"
     "for i, l in enumerate(mt):\n"
     "    if l.startswith('## '):\n"
     "        heads[l.rstrip()] = i + 1\n"
     "_n_hdr = sum(1 for l in mt if l.startswith('## '))\n"
     "assert len(heads) == _n_hdr, ('sections 键碰撞：%d 个标题行 -> %d 个键'\n"
     "                              % (_n_hdr, len(heads)))\n"
     "hist = []\n"),
    # 2c. history 陈旧标注
    ("hist = [h for h in hist if h.get('tag') != NEW_TAG]\n",
     "hist = [h for h in hist if h.get('tag') != NEW_TAG]\n"
     "# ---- 完整性事件登记：P19 基线快照之后 MEMO 被就地规范化 ----\n"
     "_STALE = {DRIFT['prev_baseline']['tag']:\n"
     "          ('快照后于 ' + str(DRIFT['memo_mtime']) + ' 被就地规范化：P10–P19 共 '\n"
     "           + str(len(DRIFT['normalized_phases'])) + ' 个标题由短形式 [HH:MM] 改为完整形式，+'\n"
     "           + str(DRIFT['observed_delta_bytes']) + ' B（行数不变、无文本丢失）'\n"
     "           + '⇒ bytes/sha256 锚陈旧；审计件 '\n"
     "           + 'tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json')}\n"
     "for _e in hist:\n"
     "    if _e.get('tag') in _STALE:\n"
     "        _e['stale'] = True\n"
     "        _e['note'] = _STALE[_e['tag']]\n"
     "w('history 陈旧标注：%s' % [h.get('tag') for h in hist if h.get('stale')])\n"),
    # 2d. base 增口径字段与漂移登记
    ("        'sections': heads, 'history': hist}\n",
     "        'sections_key_rule': 'full-heading-line',\n"
     "        'drift_events': [{'tag': DRIFT['prev_baseline']['tag'],\n"
     "                          'event': 'post-baseline in-place heading normalization',\n"
     "                          'memo_mtime': DRIFT['memo_mtime'],\n"
     "                          'delta_bytes': DRIFT['observed_delta_bytes'],\n"
     "                          'delta_lines': DRIFT['observed_delta_lines'],\n"
     "                          'phases': DRIFT['normalized_phases'],\n"
     "                          'residual_bytes': DRIFT['residual_bytes'],\n"
     "                          'artifact': ('tests/deepseek_temp/_infra/'\n"
     "                                       'memo_drift_phase19_postbaseline.json')}],\n"
     "        'sections': heads, 'history': hist}\n"),
    # 2e. wlog 增完整性事件条目（在「下一步（死线）」之前）
    ("p('- **下一步（死线）**：**Phase 21 最高优先 = 把跨精度检验推进到组件级向量预算与权重实现级** —— '",
     "p('- **⚠️ MEMO 完整性事件（登记在案；非本轮引入）**：`post-append-phase19` 基线（%d B / `%s`，冻结于 %s）'\n"
     "  '在快照后于 **%s** 被**就地改写** —— P10–P19 共 **%d** 个标题由短形式 `[HH:MM]` 规范化为完整形式 '\n"
     "  '`[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+%d B**（与实盘差逐位一致，残差 %+d B），行数不变（%d）、无文本丢失；'\n"
     "  '事件未被任何 wlog / baseline / history 记录（P19 独立磁盘复核 `disk_verify_phase19.txt` mtime %s 早于该改写，'\n"
     "  '且当时读到与基线一致的字节数并 PASS）。**处置**：不改写历史 —— `post-append-phase19` 在 `history` 中标注 `stale`；'\n"
     "  '本轮 `_infra/memo_baseline.json` 以新口径重新冻结（`sections` 键由「行前 44 字符」改为**完整标题行**并加碰撞断言，'\n"
     "  '`drift_events` 登记本事件）。现场审计件 `tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json` '\n"
     "  '+ `audit_memo_drift_phase19.txt`。'\n"
     "  % (int(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),\n"
     "     str(DRIFT['prev_baseline']['frozen_at']), str(DRIFT['memo_mtime']),\n"
     "     len(DRIFT['normalized_phases']), int(DRIFT['predicted_delta_bytes']),\n"
     "     int(DRIFT['observed_delta_bytes']), int(DRIFT['residual_bytes']),\n"
     "     int(DRIFT['memo_lines_at_audit']), DV19_T))\n"
     "p('- **下一步（死线）**：**Phase 21 最高优先 = 把跨精度检验推进到组件级向量预算与权重实现级** —— '"),
])

# ================================================================ 3. disk_verify
rep(BASE + r'\disk_verify_phase20.py', [
    ("chk('G10g', 'MEMO 长度 > 基线', len(mb) > int(BASE['bytes']), len(mb), int(BASE['bytes']))\n",
     "chk('G10g', 'MEMO 长度 > 基线', len(mb) > int(BASE['bytes']), len(mb), int(BASE['bytes']))\n"
     "# ---- 完整性事件：P19 基线快照之后 MEMO 被就地规范化 ----\n"
     "DR = load(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',\n"
     "                       'memo_drift_phase19_postbaseline.json'))\n"
     "chk('G10h', '漂移审计：预测字节增 == 实测字节增',\n"
     "    DR['predicted_delta_bytes'] == DR['observed_delta_bytes'],\n"
     "    DR['predicted_delta_bytes'], DR['observed_delta_bytes'])\n"
     "chk('G10i', '漂移审计：行数不变且残差为 0',\n"
     "    DR['observed_delta_lines'] == 0 and DR['residual_bytes'] == 0,\n"
     "    (DR['observed_delta_lines'], DR['residual_bytes']), (0, 0))\n"
     "chk('G10j', '漂移审计：规范化标题数 == 10', len(DR['normalized_phases']) == 10,\n"
     "    len(DR['normalized_phases']), 10)\n"
     "chk('G10k', '漂移审计：判别性证据全一致',\n"
     "    all(d['short_explains_key'] for d in DR['discriminating_evidence']),\n"
     "    sum(1 for d in DR['discriminating_evidence'] if d['short_explains_key']),\n"
     "    len(DR['discriminating_evidence']))\n"
     "_ib = load(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json'))\n"
     "chk('G10l', '新基线 sections 键口径 == 完整标题行',\n"
     "    _ib.get('sections_key_rule') == 'full-heading-line',\n"
     "    _ib.get('sections_key_rule'), 'full-heading-line')\n"
     "chk('G10m', '新基线 history 中 P19 条目标注 stale',\n"
     "    any(h.get('tag') == DR['prev_baseline']['tag'] and h.get('stale') is True\n"
     "        for h in _ib.get('history', [])))\n"
     "_n_hdr2 = sum(1 for l in lines if l.startswith('## '))\n"
     "chk('G10n', '新基线 sections 键数 == 标题行数（无碰撞）',\n"
     "    len(_ib.get('sections', {})) == _n_hdr2, len(_ib.get('sections', {})), _n_hdr2)\n"
     "chk('G10o', '新基线 drift_events 登记 1 条', len(_ib.get('drift_events', [])) == 1,\n"
     "    len(_ib.get('drift_events', [])), 1)\n"),
])

# ================================================================ 4. gen_present
rep(BASE + r'\gen_present_phase20.py', [
    ("PR = json.load(io.open(PROBER, encoding='utf-8')) if os.path.exists(PROBER) else None\n",
     "PR = json.load(io.open(PROBER, encoding='utf-8')) if os.path.exists(PROBER) else None\n"
     "DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',\n"
     "                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))\n"),
    ("         '<b>E-probefull</b> 探针必须<b>保留全量配对与实例</b>（否则 <code>U_ℓ</code> 秩退化为 2、FULL_SWAP 偏离锚）。</div>')",
     "         '<b>E-probefull</b> 探针必须<b>保留全量配对与实例</b>（否则 <code>U_ℓ</code> 秩退化为 2、FULL_SWAP 偏离锚）；'\n"
     "         '<b>E-baseline</b>（<b>记录完整性事件，非本轮引入</b>）'\n"
     "         '<code>post-append-phase19</code> 基线（%s B / <code>%s</code>）在快照后于 <b>%s</b> 被就地改写 —— '\n"
     "         'P10–P19 共 <b>%d</b> 个标题由 <code>[HH:MM]</code> 规范化为 <code>[YYYY-MM-DD HH:MM]</code>，'\n"
     "         '逐条 +11 B、合计 <b>+%d B</b>（行数不变、<b>无文本丢失</b>，事件未被任何 wlog / history 记录）'\n"
     "         '⇒ 该基线的 bytes/sha256 锚与 <code>sections</code> 键陈旧。<b>处置</b>：不改写历史，'\n"
     "         '在 <code>history</code> 标注 <code>stale</code>，并把 <code>sections</code> 键口径由「行前 44 字符」'\n"
     "         '改为<b>完整标题行</b>（加碰撞断言）。</div>'\n"
     "         % (str(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),\n"
     "            str(DRIFT['memo_mtime']), len(DRIFT['normalized_phases']),\n"
     "            DRIFT['observed_delta_bytes']))"),
])

print('PATCHED %d spots' % n)
