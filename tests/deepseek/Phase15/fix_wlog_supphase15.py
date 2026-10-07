# -*- coding: utf-8 -*-
"""修复 2026-10-02.md 的 Phase 15 收尾补记段：
上一版经 bash 内联 python 写入，反引号被 bash 做命令替换 ⇒ 所有 `code` 片段被吞。
本脚本：① 切掉被污染段并**断言前缀 sha256 == 上一版已复核值 8f69229b...**（38,233 B）；
② 重新追加正确的段（含反引号，本文件由编辑器写盘，不经 bash）；③ 落盘复核。
"""
import io, hashlib, time

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
GOOD_SHA256 = '8f69229b3e722d09400395ed95d3fd370602fb3545c8c52568828fbbedb842c3'
GOOD_BYTES = 38233
HDR = '## Phase 15 收尾补记'

b_cur = open(P, 'rb').read()
txt = b_cur.decode('utf-8')
i = txt.find(HDR)
assert i > 0, '找不到段头'

prefix = txt[:i].rstrip('\r\n')
prefix_bytes = prefix.encode('utf-8') + b'\r\n'
print('cut: current %d B -> prefix %d B' % (len(b_cur), len(prefix_bytes)))
print('prefix sha256 = %s' % hashlib.sha256(prefix_bytes).hexdigest())
assert len(prefix_bytes) == GOOD_BYTES, 'prefix bytes %d != %d' % (len(prefix_bytes), GOOD_BYTES)
assert hashlib.sha256(prefix_bytes).hexdigest() == GOOD_SHA256, 'prefix sha mismatch -> abort'
print('  ==> 回滚点与上一版已复核状态逐字节一致，安全')

BT = chr(96)


def c(s):
    """把 {x} 包裹成反引号片段，避免源码里直接出现反引号带来的任何转义歧义。"""
    return BT + s + BT


sec_lines = [
    '%s：MEMO 渲染器数据化修正 + 前缀锚 bug 修复 + 收尾链闭合（%s）' % (HDR, time.strftime('%H:%M')),
    '',
    '- **背景**：Phase 15 正式运行（三臂同一 nf4 口径）与判决已完成；本段为收尾链的「MEMO 渲染 → 追加 → wlog/基线 → 独立复核 → MEMORY/技能」环节。',
    '- **MEMO 渲染器 %s 的数据化修正（4 处写死散文与数据矛盾）**：首版 §5 item 3/4、§6 Q3 bullet、§9 一句话断言「%s 坐标在三臂上**都**超零假设」「%s 坐标才是有区分力的坐标」，'
    '而实测 **6 个「臂 × 坐标」格里只有 2 格**过零假设（A0·J 裕度 +0.1057、A1·xhalf +0.0357；其余 4 格为负：A0·xhalf −0.1528、A1·J −0.0256、A2·xhalf −0.0913、A2·J −0.1217）。'
    '修法：结论句一律走 %s 分支或现场 %s；新增「判决标签完整取值域」一段（让 %s / %s / %s 成为可检索锚点）；%s 键名写进正文。'
    '同时把中间量（%s）从 §5 **上移到 §0 之前**定义（否则 §0 引用报 %s）。渲染件 42,350 → **43,605 B / 255 行 / sha8 07a090c9**。'
    % (c('gen_memo_phase15.py'), c('J'), c('J'), c('q2j/q3j'), c('sum(...)'),
       c('CONC_JUDGE_INVALID_X_ALL'), c('ARGS_GAP_4B_SPECIFIC'), c('ARGS_GAP_MIXED'),
       c('L_star_own'), c('_n95/_jm/_jup/_xup/_ncell'), c('NameError')),
    '- **%s 的前缀锚 bug（假失败）**：旧写法 %s，原文件以 CRLF 结尾 ⇒ %s，%s ⇒ **追加成功但 %s 抛在写盘之后**（脚本报 FAIL、磁盘已改）。'
    '修法：前缀锚 = **写前读到的 %s 本身**（%s 以 %s 为前缀），并打印 %s 与 %s 两个布尔。'
    '**并加幂等探测**：%s ⇒ 命中即跳过写入、只做纯复核；重跑输出 %s。'
    % (c('do_append_phase15.py'), c('prefix = BOM + txt.rstrip(CRLF)'), c('len(prefix) = len(raw0) - 2'),
       c('sha256(rb[:len(prefix)]) != sha256(raw0)'), c('assert anchor_ok'),
       c('raw0'), c('rb'), c('raw0'), c('startswith'), c('sha_match'),
       c("ALREADY = ('## Phase 15:' in txt)"), c('ALL CHECKS PASSED')),
    '- **追加落盘（已复核）**：MEMO **346,025 → 389,885 B（+43,860）/ 3,427 → 3,682 行（+255）**，%s、%s、**%s**、Phase 标题 **14 → 15**、%s 行 = **L3429**、'
    '前缀锚 %s 逐字节未变。%s 刷新为 %s（history=7）；当日 wlog 29,906 → **38,233 B**。'
    % (c('bom=True'), c('crlf=3682'), c('bare_lf=0'), c('## Phase 15'), c('c9b4b3f5'),
       c('_infra/memo_baseline.json'), c('post-append-phase15')),
    '- **独立磁盘复核 %s：%s**（含新增分区 J：amend1 存在/sha 一致/首轮作废日志保留/三臂 tokenizer **独立重解析**与 %s 逐项比对/F1b/F2_base_bad 全空/Q0 全 PASS）。'
    % (c('disk_verify_phase15.py'), c('TOTAL checks = 122 ; FAIL = 0'), c('result.sup_id_per_arm')),
    '- **技能同步**：%s 新增**坑 55**（极值型 3-窗口占比在两个坐标、三个模型上同时失效：6 格只活 2 格；两种翻转；'
    '对策=同报 null95 与裕度、换对排序不敏感的量、不得把「argmax 距离」与「集中度显著」混谈；%s 量化换窗 ⇒ 门槛写最稳健量）'
    '→ 63,746 → **65,788 B**（14 臂 + **55 坑**）。%s 新增**教训 23**（前缀锚必须是写前 raw bytes；写盘与断言须可分辨）与**教训 24**'
    '（%s 幂等探测 + 渲染器结论句必须走 verdict 分支/现场取值；分支标签锚点缺失的正确修法是补「取值域」段而非删锚点）→ 31,963 → **34,893 B**（**24 教训**）。'
    % (c('rdc-main-axis-probe'), c('argmax_w_j'), c('rdc-phase-closeout'), c('do_append_*')),
    '- **MEMORY.md 重写**：Ledger n=**298**、基线 389,885 B / %s / 15 标题、收尾链「**八次 P8–P15**」、铁律 (z)(aa)(ab) 入册、'
    '§2 新增 P15 条目、§3 新增 P15 限界、§7 死线改为 **P16 =「写入窗 vs 集中窗」+ 集中度统计量重设计（并列最高优先）+ 家族/规模解耦**、§8 技能计数更新。'
    % c('0bd7bfd0'),
    '- **工具教训（本轮再次踩中）**：bash 内联 %s 的文本中**一律不得出现反引号** —— 会被 bash 做命令替换并静默吞掉（本段首版即因此重写）。'
    '对策：长文本一律**先 Write 成 .py 文件**再经解释器执行；必须内联时用 %s 拼接。'
    % (c('python -c'), c('chr(96)')),
    '- **下一步**：P16 判据冻结后开跑；第二优先为集中度统计量重设计（三臂 × 双坐标重算 P12–P14 集中度表）。',
]

sec = '\n'.join(sec_lines)
new = prefix + '\r\n\r\n' + sec.replace('\r\n', '\n').replace('\n', '\r\n') + '\r\n'
open(P, 'wb').write(new.encode('utf-8'))
b1 = open(P, 'rb').read()
print('wrote: %d -> %d B (+%d) ; lines %d -> %d'
      % (len(prefix_bytes), len(b1), len(b1) - len(prefix_bytes),
         len(prefix_bytes.split(b'\r\n')), len(b1.split(b'\r\n'))))
print('sha256 = %s' % hashlib.sha256(b1).hexdigest())
print('bare_lf = %d' % (b1.count(b'\n') - b1.count(b'\r\n')))
d = b1.decode('utf-8')
print('backticks = %d（污染修复后应 > 0）' % d.count(BT))
print('Phase15 主节数 = %d ; 补记数 = %d' % (d.count('## Phase 15 /'), d.count(HDR)))
print('无残留吞字标记 = %s' % ('生成了' not in d))
